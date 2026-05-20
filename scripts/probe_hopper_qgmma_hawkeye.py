"""Compare Hopper direct-QGMMA FP8 teacher against Hawkeye models.

This is the Hopper counterpart to the L40S/Triton probes.  It compares:

  * teacher A: torch._scaled_mm / cuBLAS
  * teacher B: a direct fixed-order Hopper QGMMA loop
  * model C: Hawkeye scalar model with public Hopper FP8 constants
  * model D: Freivalds-checkable moment buckets with the same constants

The direct teacher is intentionally narrow: real float8_e4m3fn operands, output
tiles of 64x128, K walked left-to-right in m64n128k32 WGMMA steps, and row/col
scale multiplication plus bf16 conversion after the QGMMA loop.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch

from hopper_qgmma_teacher import qgmma_fp8_scaled_mm, require_hopper
from probe_hawkeye_bucket_products import moment_bucket_sum
from probe_hawkeye_l40s_fp8 import (
    Candidate,
    exact_codebook_sum,
    fp8_scaled_mm,
    hawkeye_grouped_sum,
    metrics,
    per_token_fp8,
)


def _random_case(
    m_size: int,
    n_size: int,
    k_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m_size, k_size), generator=generator, device="cuda", dtype=torch.bfloat16)
    w = torch.randn((n_size, k_size), generator=generator, device="cuda", dtype=torch.float32)
    w_scale = (w.abs().amax(dim=1, keepdim=True) / 448.0).clamp_min(1e-12)
    w_fp8 = (w / w_scale).to(torch.float8_e4m3fn).contiguous()
    x_fp8, x_scale = per_token_fp8(x)
    return x_fp8, x_scale, w_fp8, w_scale, {"case": "random"}


def _real_weight_case(
    m_size: int,
    seed: int,
    model_id: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    for name, module in model.named_modules():
        if (
            hasattr(module, "weight")
            and hasattr(module, "weight_scale")
            and module.weight.dtype == torch.float8_e4m3fn
        ):
            w_fp8 = module.weight.detach().contiguous()
            w_scale = module.weight_scale.detach().to(torch.float32)
            layer_name = name
            break
    else:
        raise RuntimeError("no FP8 weight with weight_scale found")

    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(
        (m_size, w_fp8.shape[1]),
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    x_fp8, x_scale = per_token_fp8(x)
    return (
        x_fp8,
        x_scale,
        w_fp8,
        w_scale,
        {"case": "real_weight", "model": model_id, "layer": layer_name},
    )


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    capability = require_hopper()
    candidate = Candidate(
        name=f"hopper_g{args.group}_w{args.width}",
        products_per_group=args.group,
        internal_width=args.width,
    )
    if args.real_weight:
        x_fp8, x_scale, w_fp8, w_scale, meta = _real_weight_case(
            args.m,
            args.seed,
            args.model_id,
        )
    else:
        x_fp8, x_scale, w_fp8, w_scale, meta = _random_case(args.m, args.n, args.k, args.seed)

    cublas_teacher = fp8_scaled_mm(x_fp8, x_scale, w_fp8, w_scale)
    qgmma_teacher = qgmma_fp8_scaled_mm(
        x_fp8,
        x_scale,
        w_fp8,
        w_scale,
        verbose_build=args.verbose_build,
    )
    hawkeye = hawkeye_grouped_sum(x_fp8, x_scale, w_fp8, w_scale, candidate)
    moment, moment_stats = moment_bucket_sum(x_fp8, x_scale, w_fp8, w_scale, candidate)

    vs_qgmma_teacher: dict[str, object] = {
        "cublas_scaled_mm": metrics(qgmma_teacher, cublas_teacher),
        "hawkeye_scalar": metrics(qgmma_teacher, hawkeye),
        "moment_buckets": metrics(qgmma_teacher, moment),
    }
    vs_cublas_teacher: dict[str, object] = {
        "qgmma_teacher": metrics(cublas_teacher, qgmma_teacher),
        "hawkeye_scalar": metrics(cublas_teacher, hawkeye),
        "moment_buckets": metrics(cublas_teacher, moment),
    }
    if args.include_codebook:
        codebook = exact_codebook_sum(x_fp8, x_scale, w_fp8, w_scale)
        vs_qgmma_teacher["exact_codebook"] = metrics(qgmma_teacher, codebook)
        vs_cublas_teacher["exact_codebook"] = metrics(cublas_teacher, codebook)

    return {
        **meta,
        "shape": [int(x_fp8.shape[0]), int(w_fp8.shape[0]), int(x_fp8.shape[1])],
        "seed": args.seed,
        "device": torch.cuda.get_device_name(0),
        "compute_capability": [int(capability[0]), int(capability[1])],
        "teacher_b": {
            "instruction": "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3",
            "tile_m": 64,
            "tile_n": 128,
            "tile_k": 32,
            "k_order": "left_to_right",
            "output_dtype": "torch.bfloat16",
        },
        "candidate": asdict(candidate),
        "qgmma_vs_cublas_teacher": metrics(cublas_teacher, qgmma_teacher),
        "vs_qgmma_teacher": vs_qgmma_teacher,
        "vs_cublas_teacher": vs_cublas_teacher,
        "vs_hawkeye_scalar": {
            "moment_buckets": metrics(hawkeye, moment),
        },
        "bucket_products": {
            "moment_buckets": asdict(moment_stats),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--k", type=int, default=896)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group", type=int, default=32)
    parser.add_argument("--width", type=int, default=14)
    parser.add_argument("--real-weight", action="store_true")
    parser.add_argument("--include-codebook", action="store_true")
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--model-id", default="RedHatAI/Qwen2.5-0.5B-FP8-dynamic")
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    if not hasattr(torch, "_scaled_mm"):
        raise SystemExit("torch._scaled_mm is required.")
    result = run_experiment(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
