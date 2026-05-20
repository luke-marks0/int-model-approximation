"""Probe Hawkeye-style FP8 accumulation against torch._scaled_mm on L40S/Ada.

This is an empirical bridge from Hawkeye's Tensor Core model to this repo's
integer-kernel strategy.  It compares the FP8 teacher against:

  * exact integer FP8-codebook summation, i.e. the current alpha=32/32 endpoint
  * Hawkeye-style grouped fixed-point summation with configurable group size
    and internal significand width

The Hawkeye paper reports that Lovelace follows Ampere's accumulation structure
for the formats they characterize, while their public FP8 simulator exposes a
Hopper E4M3 model with a 14-bit internal significand and 32 products plus the
incoming accumulator in one group.  L40S is Lovelace (SM89), so this script
sweeps both Ada/Ampere-shaped grouping and Hopper-shaped FP8 constants.

The script is intentionally a probe, not a production path: Hawkeye-style
alignment depends on the max exponent of the product group for each output
element, so it is not the same clean A @ B integer product that the active
Freivalds path checks.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import torch


FP8_E4M3_MAX = 448.0
FP8_E4M3_CODE_SCALE = 512.0
FP32_SIGNIFICAND_WIDTH = 24
FP32_MIN_NONZERO_EXPONENT = -126
HAWKEYE_FP8_ZERO_EXPONENT = -139


@dataclass(frozen=True)
class Candidate:
    name: str
    products_per_group: int
    internal_width: int
    zero_exponent: int = HAWKEYE_FP8_ZERO_EXPONENT


def per_token_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows = x.reshape(-1, x.shape[-1])
    scale = rows.detach().abs().amax(dim=-1, keepdim=True).to(torch.float32)
    scale = (scale / FP8_E4M3_MAX).clamp_min(1e-12)
    q = (rows.to(torch.float32) / scale).to(torch.float8_e4m3fn)
    return q.contiguous(), scale


def fp8_scaled_mm(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    return torch._scaled_mm(
        x_fp8,
        w_fp8.t(),
        scale_a=x_scale,
        scale_b=w_scale.reshape(1, -1).to(torch.float32),
        out_dtype=torch.bfloat16,
    )


def exact_codebook_sum(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    x_int = (x_fp8.to(torch.float32) * FP8_E4M3_CODE_SCALE).round().to(torch.int64)
    w_int = (w_fp8.to(torch.float32) * FP8_E4M3_CODE_SCALE).round().to(torch.int64)
    accum = (x_int.unsqueeze(2) * w_int.t().unsqueeze(0)).sum(dim=1)
    y = accum.to(torch.float32) * (
        x_scale / FP8_E4M3_CODE_SCALE
    ) * (w_scale.reshape(1, -1).to(torch.float32) / FP8_E4M3_CODE_SCALE)
    return y.to(torch.bfloat16)


def _fp8_product_components(
    a_u8: torch.Tensor,
    b_u8: torch.Tensor,
    zero_exponent: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sign, exponent, significand for all pairwise FP8 products.

    a_u8: [M, G] raw e4m3fn bytes
    b_u8: [N, G] raw e4m3fn bytes

    Returns tensors shaped [M, N, G].  The representation mirrors Hawkeye's
    public `multiply_fp8e4m3_to_gfloat`: subnormal inputs keep their reduced
    significand, normal inputs get the implicit leading bit, and products are
    exact 4-bit x 4-bit significand products shifted into a 25-bit container.
    """
    a = a_u8.to(torch.int64)
    b = b_u8.to(torch.int64)

    sign_a = (a >> 7) & 1
    sign_b = (b >> 7) & 1
    exp_a = (a >> 3) & 0x0F
    exp_b = (b >> 3) & 0x0F
    mant_a = a & 0x07
    mant_b = b & 0x07

    sig_a = torch.where(exp_a != 0, mant_a | 0x08, mant_a)
    sig_b = torch.where(exp_b != 0, mant_b | 0x08, mant_b)
    exp_a = torch.where(exp_a != 0, exp_a, torch.ones_like(exp_a))
    exp_b = torch.where(exp_b != 0, exp_b, torch.ones_like(exp_b))

    sign = (sign_a[:, None, :] ^ sign_b[None, :, :]).bool()
    significand = (sig_a[:, None, :] * sig_b[None, :, :]) << 17
    exponent = exp_a[:, None, :] + exp_b[None, :, :] - 14
    exponent = torch.where(
        significand != 0,
        exponent,
        torch.full_like(exponent, zero_exponent),
    )
    return sign, exponent, significand


def _shift_right(x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    shift = shift.clamp_min(0)
    return torch.bitwise_right_shift(x, shift)


def _shift_left(x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    shift = shift.clamp_min(0)
    return torch.bitwise_left_shift(x, shift)


def _group_sum(
    acc_sign: torch.Tensor,
    acc_exp: torch.Tensor,
    acc_sig: torch.Tensor,
    prod_sign: torch.Tensor,
    prod_exp: torch.Tensor,
    prod_sig: torch.Tensor,
    internal_width: int,
    zero_exponent: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hawkeye-style grouped fixed-point summation.

    The returned significand is expanded/truncated back to FP32's 24-bit
    significand width, matching the public Hawkeye Gfloat representation used
    as the incoming accumulator for the next group.
    """
    prod_exp = torch.where(
        prod_sig != 0,
        prod_exp,
        torch.full_like(prod_exp, zero_exponent),
    )
    acc_exp_eff = torch.where(
        acc_sig != 0,
        acc_exp,
        torch.full_like(acc_exp, zero_exponent),
    )
    max_exp = torch.maximum(acc_exp_eff, prod_exp.max(dim=2).values)

    if internal_width < FP32_SIGNIFICAND_WIDTH:
        product_base = prod_sig >> (FP32_SIGNIFICAND_WIDTH - internal_width)
        acc_base = acc_sig >> (FP32_SIGNIFICAND_WIDTH - internal_width)
    else:
        product_base = prod_sig << (internal_width - FP32_SIGNIFICAND_WIDTH)
        acc_base = acc_sig << (internal_width - FP32_SIGNIFICAND_WIDTH)

    prod_shift = max_exp[:, :, None] - prod_exp
    aligned = _shift_right(product_base, prod_shift)
    signed = torch.where(prod_sign, -aligned, aligned)

    acc_shift = max_exp - acc_exp_eff
    aligned_acc = _shift_right(acc_base, acc_shift)
    signed_acc = torch.where(acc_sign, -aligned_acc, aligned_acc)
    total = signed.sum(dim=2) + signed_acc

    out_sign = total < 0
    magnitude = total.abs()
    nonzero = magnitude != 0
    width = torch.zeros_like(magnitude)
    if nonzero.any():
        width_nonzero = torch.floor(torch.log2(magnitude[nonzero].to(torch.float64))).to(
            torch.int64
        ) + 1
        width[nonzero] = width_nonzero

    out_exp = max_exp + width - internal_width
    right = (width - internal_width).clamp_min(0)
    left = (internal_width - width).clamp_min(0)
    out_sig = torch.where(
        width > internal_width,
        _shift_right(magnitude, right),
        _shift_left(magnitude, left),
    )

    subnormal_shift = (FP32_MIN_NONZERO_EXPONENT - out_exp).clamp_min(0)
    out_sig = torch.where(
        out_exp < FP32_MIN_NONZERO_EXPONENT,
        _shift_right(out_sig, subnormal_shift),
        out_sig,
    )
    out_exp = torch.maximum(
        out_exp,
        torch.full_like(out_exp, FP32_MIN_NONZERO_EXPONENT),
    )

    if internal_width < FP32_SIGNIFICAND_WIDTH:
        out_sig = out_sig << (FP32_SIGNIFICAND_WIDTH - internal_width)
    else:
        out_sig = out_sig >> (internal_width - FP32_SIGNIFICAND_WIDTH)

    out_sig = torch.where(nonzero & (out_sig != 0), out_sig, torch.zeros_like(out_sig))
    out_exp = torch.where(
        out_sig != 0,
        out_exp,
        torch.full_like(out_exp, zero_exponent),
    )
    out_sign = torch.where(out_sig != 0, out_sign, torch.zeros_like(out_sign))
    return out_sign, out_exp, out_sig


def _gfloat_to_float32(
    sign: torch.Tensor,
    exponent: torch.Tensor,
    significand: torch.Tensor,
) -> torch.Tensor:
    exponent_bits = exponent + 127
    leading_bit_missing = (significand & 0x800000) == 0
    exponent_bits = torch.where(
        leading_bit_missing & (significand != 0),
        exponent_bits - 1,
        exponent_bits,
    )
    bits = (
        (sign.to(torch.int64) << 31)
        | ((exponent_bits.to(torch.int64) & 0xFF) << 23)
        | (significand.to(torch.int64) & 0x7FFFFF)
    )
    return bits.to(torch.int32).view(torch.float32)


def hawkeye_grouped_sum(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
    candidate: Candidate,
) -> torch.Tensor:
    a_u8 = x_fp8.contiguous().view(torch.uint8)
    b_u8 = w_fp8.contiguous().view(torch.uint8)
    m_size, k_size = a_u8.shape
    n_size = b_u8.shape[0]

    acc_sign = torch.zeros((m_size, n_size), device=x_fp8.device, dtype=torch.bool)
    acc_exp = torch.full(
        (m_size, n_size),
        candidate.zero_exponent,
        device=x_fp8.device,
        dtype=torch.int64,
    )
    acc_sig = torch.zeros((m_size, n_size), device=x_fp8.device, dtype=torch.int64)

    for k0 in range(0, k_size, candidate.products_per_group):
        a = a_u8[:, k0 : k0 + candidate.products_per_group]
        b = b_u8[:, k0 : k0 + candidate.products_per_group]
        prod_sign, prod_exp, prod_sig = _fp8_product_components(
            a,
            b,
            candidate.zero_exponent,
        )
        acc_sign, acc_exp, acc_sig = _group_sum(
            acc_sign,
            acc_exp,
            acc_sig,
            prod_sign,
            prod_exp,
            prod_sig,
            candidate.internal_width,
            candidate.zero_exponent,
        )

    y = _gfloat_to_float32(acc_sign, acc_exp, acc_sig)
    y = y * x_scale * w_scale.reshape(1, -1).to(torch.float32)
    return y.to(torch.bfloat16)


def _bf16_bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int16)


def metrics(ref: torch.Tensor, cand: torch.Tensor) -> dict[str, float]:
    diff = ref.to(torch.float32) - cand.to(torch.float32)
    bit_match = (_bf16_bits(ref) == _bf16_bits(cand)).float().mean().item()
    return {
        "bit_match": bit_match,
        "mean_abs": diff.abs().mean().item(),
        "max_abs": diff.abs().max().item(),
        "same_argmax": (ref.argmax(dim=1) == cand.argmax(dim=1)).float().mean().item(),
    }


def run_once(
    m_size: int,
    n_size: int,
    k_size: int,
    seed: int,
    candidates: list[Candidate],
) -> dict[str, object]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m_size, k_size), generator=generator, device="cuda", dtype=torch.bfloat16)
    w = torch.randn((n_size, k_size), generator=generator, device="cuda", dtype=torch.float32)
    w_scale = (w.abs().amax(dim=1, keepdim=True) / FP8_E4M3_MAX).clamp_min(1e-12)
    w_fp8 = (w / w_scale).to(torch.float8_e4m3fn).contiguous()
    x_fp8, x_scale = per_token_fp8(x)
    ref = fp8_scaled_mm(x_fp8, x_scale, w_fp8, w_scale)

    result: dict[str, object] = {
        "shape": [m_size, n_size, k_size],
        "seed": seed,
        "device": torch.cuda.get_device_name(0),
        "exact_codebook": metrics(ref, exact_codebook_sum(x_fp8, x_scale, w_fp8, w_scale)),
        "candidates": {},
    }
    candidate_results: dict[str, object] = {}
    for candidate in candidates:
        y = hawkeye_grouped_sum(x_fp8, x_scale, w_fp8, w_scale, candidate)
        candidate_results[candidate.name] = {
            "products_per_group": candidate.products_per_group,
            "internal_width": candidate.internal_width,
            **metrics(ref, y),
        }
    result["candidates"] = candidate_results
    return result


def run_real_weight(
    m_size: int,
    seed: int,
    candidates: list[Candidate],
    model_id: str,
) -> dict[str, object]:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
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
    ref = fp8_scaled_mm(x_fp8, x_scale, w_fp8, w_scale)
    result: dict[str, object] = {
        "model": model_id,
        "layer": layer_name,
        "shape": [m_size, int(w_fp8.shape[0]), int(w_fp8.shape[1])],
        "seed": seed,
        "device": torch.cuda.get_device_name(0),
        "exact_codebook": metrics(ref, exact_codebook_sum(x_fp8, x_scale, w_fp8, w_scale)),
        "candidates": {},
    }
    candidate_results: dict[str, object] = {}
    for candidate in candidates:
        y = hawkeye_grouped_sum(x_fp8, x_scale, w_fp8, w_scale, candidate)
        candidate_results[candidate.name] = {
            "products_per_group": candidate.products_per_group,
            "internal_width": candidate.internal_width,
            **metrics(ref, y),
        }
    result["candidates"] = candidate_results
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--k", type=int, default=896)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--real-weight",
        action="store_true",
        help="Use the first real FP8 layer from the Hugging Face checkpoint instead of random weights.",
    )
    parser.add_argument(
        "--model-id",
        default="RedHatAI/Qwen2.5-0.5B-FP8-dynamic",
        help="Checkpoint used with --real-weight.",
    )
    parser.add_argument(
        "--widths",
        default="12,13,14,15,16,17,18,24",
        help="Comma-separated internal significand widths to sweep.",
    )
    parser.add_argument(
        "--groups",
        default="8,16,32",
        help="Comma-separated product counts per group. Ada/Ampere-like is 8; Hopper FP8 is 32.",
    )
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    if not hasattr(torch, "_scaled_mm"):
        raise SystemExit("torch._scaled_mm is required.")

    args = parse_args()
    widths = [int(x) for x in args.widths.split(",") if x]
    groups = [int(x) for x in args.groups.split(",") if x]
    candidates = [
        Candidate(name=f"hawkeye_g{group}_w{width}", products_per_group=group, internal_width=width)
        for group in groups
        for width in widths
    ]
    if args.real_weight:
        result = run_real_weight(args.m, args.seed, candidates, args.model_id)
    else:
        result = run_once(args.m, args.n, args.k, args.seed, candidates)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
