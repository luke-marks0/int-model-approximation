"""Measure integerized-layer error against the Hopper QGMMA/Hawkeye teacher."""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from typing import Any

import torch

import int_model_approximation.__main__ as entry
from hopper_qgmma_teacher import qgmma_fp8_scaled_mm, require_hopper
from probe_hawkeye_l40s_fp8 import fp8_scaled_mm, per_token_fp8


def _parse_alpha(value: str) -> float:
    value = value.strip()
    if "/" in value:
        return float(Fraction(value))
    return float(value)


def _metrics(ref: torch.Tensor, cand: torch.Tensor) -> dict[str, float]:
    diff = ref.to(torch.float32) - cand.to(torch.float32)
    return {
        "bit_match": (
            ref.contiguous().view(torch.int16) == cand.contiguous().view(torch.int16)
        )
        .float()
        .mean()
        .item(),
        "mean_abs": diff.abs().mean().item(),
        "max_abs": diff.abs().max().item(),
        "l2": diff.norm().item(),
        "same_argmax": (ref.argmax(dim=1) == cand.argmax(dim=1)).float().mean().item(),
    }


def _random_case(
    m_size: int,
    n_size: int,
    k_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m_size, k_size), generator=generator, device="cuda", dtype=torch.bfloat16)
    w = torch.randn((n_size, k_size), generator=generator, device="cuda", dtype=torch.float32)
    w_scale = (w.abs().amax(dim=1, keepdim=True) / entry.FP8_E4M3_MAX).clamp_min(1e-12)
    w_fp8 = (w / w_scale).to(torch.float8_e4m3fn).contiguous()
    return x, w_fp8, w_scale, w_fp8.to(torch.float32) * w_scale, None, {"case": "random"}


def _real_weight_case(
    m_size: int,
    seed: int,
    model_id: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")
    entry._disable_compressed_tensor_hooks(model)
    for name, module in model.named_modules():
        if (
            hasattr(module, "weight")
            and hasattr(module, "weight_scale")
            and module.weight.dtype == torch.float8_e4m3fn
        ):
            w_fp8 = module.weight.detach().contiguous()
            w_scale = module.weight_scale.detach().to(torch.float32)
            weight = entry._dequantized_weight(module)
            bias = module.bias.detach() if module.bias is not None else None
            break
    else:
        raise RuntimeError("no FP8 weight with weight_scale found")

    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m_size, w_fp8.shape[1]), generator=generator, device="cuda", dtype=torch.bfloat16)
    meta = {"case": "real_weight", "model": model_id, "layer": name}
    return x, w_fp8, w_scale, weight, bias, meta


def _teacher_outputs(
    x: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_fp8, x_scale = per_token_fp8(x)
    cublas = fp8_scaled_mm(x_fp8, x_scale, w_fp8, w_scale)
    qgmma = qgmma_fp8_scaled_mm(x_fp8, x_scale, w_fp8, w_scale)
    if bias is not None:
        cublas = cublas + bias.to(cublas.dtype)
        qgmma = qgmma + bias.to(qgmma.dtype)
    return cublas, qgmma


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    capability = require_hopper()
    if args.real_weight:
        x, w_fp8, w_scale, weight, bias, meta = _real_weight_case(args.m, args.seed, args.model_id)
    else:
        x, w_fp8, w_scale, weight, bias, meta = _random_case(args.m, args.n, args.k, args.seed)

    cublas_teacher, qgmma_teacher = _teacher_outputs(x, w_fp8, w_scale, bias)
    int_layer = entry.Int32Linear(weight, bias, fp8_weight=w_fp8, fp8_weight_scale=w_scale)
    int_layer = int_layer.to("cuda").eval()

    alphas = [_parse_alpha(part) for part in args.alphas.split(",") if part.strip()]
    alpha_results: dict[str, Any] = {}
    with torch.inference_mode():
        for alpha in alphas:
            int_layer.codebook_alpha = alpha
            candidate = int_layer(x)
            alpha_results[f"{alpha:.10g}"] = {
                "vs_cublas_teacher": _metrics(cublas_teacher, candidate),
                "vs_hopper_qgmma_teacher": _metrics(qgmma_teacher, candidate),
            }

    best_qgmma = min(
        alpha_results.items(),
        key=lambda item: item[1]["vs_hopper_qgmma_teacher"]["mean_abs"],
    )
    default_alpha = entry.FP8_CODEBOOK_CORRECTION_ALPHA
    default_key = f"{default_alpha:.10g}"

    return {
        **meta,
        "shape": [int(x.shape[0]), int(w_fp8.shape[0]), int(w_fp8.shape[1])],
        "seed": args.seed,
        "device": torch.cuda.get_device_name(0),
        "compute_capability": [int(capability[0]), int(capability[1])],
        "current_default_alpha": default_alpha,
        "qgmma_vs_cublas_teacher": _metrics(cublas_teacher, qgmma_teacher),
        "alpha_results": alpha_results,
        "default_alpha_result": alpha_results.get(default_key),
        "best_alpha_by_qgmma_mean_abs": {
            "alpha": float(best_qgmma[0]),
            "result": best_qgmma[1],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--k", type=int, default=896)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--real-weight", action="store_true")
    parser.add_argument("--alphas", default="0,10/32,17/32,32/32")
    parser.add_argument("--model-id", default=entry.MODEL_ID)
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    if not hasattr(torch, "_scaled_mm"):
        raise SystemExit("torch._scaled_mm is required.")
    print(json.dumps(run_experiment(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
