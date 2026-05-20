"""Bucket-product probes for Freivalds-checkable Hawkeye FP8 approximations.

This script follows `reports/hawkeye_freivalds_design.md`: it turns the
Hawkeye-style FP8 accumulator into products over FP8 exponent/significand
buckets.  The products here are computed with PyTorch for speed, but each one
has the shape of an ordinary integer matrix product that could be replaced by
the repo's Triton int kernel and checked with Freivalds.

Modes compared:

  * exact codebook: current FP8-codebook integer sum endpoint
  * hawkeye scalar: direct grouped accumulator from probe_hawkeye_l40s_fp8.py
  * moment buckets: 225 exponent-pair count products plus 225 signed
    significand moment products per K group
  * class-pair exact: optional exact decomposition over FP8 classes for small
    tensors; this is the expensive upper-bound construction
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import torch

from probe_hawkeye_l40s_fp8 import (
    FP32_MIN_NONZERO_EXPONENT,
    FP32_SIGNIFICAND_WIDTH,
    Candidate,
    _gfloat_to_float32,
    exact_codebook_sum,
    fp8_scaled_mm,
    hawkeye_grouped_sum,
    metrics,
    per_token_fp8,
)


EXP_BUCKETS = 15
SIGNED_SIG_OFFSET = 15
SIGNED_SIG_BUCKETS = 31


@dataclass(frozen=True)
class BucketStats:
    products_per_group: int
    groups: int
    count_products_per_group: int
    moment_products_per_group: int
    total_checkable_products: int
    static_upper_bound_products: int | None = None


def _decode_fp8_fields(x_fp8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = x_fp8.contiguous().view(torch.uint8).to(torch.int64)
    sign = ((raw >> 7) & 1).bool()
    exp_bits = (raw >> 3) & 0x0F
    mant = raw & 0x07
    sig_abs = torch.where(exp_bits != 0, mant | 0x08, mant)
    exp_eff = torch.where(exp_bits != 0, exp_bits, torch.ones_like(exp_bits))
    signed_sig = torch.where(sign, -sig_abs, sig_abs)
    nonzero = sig_abs != 0
    return exp_eff, signed_sig, nonzero


def _signed_shift_right_towards_zero(x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    shift = shift.clamp_min(0)
    mag = torch.bitwise_right_shift(x.abs(), shift)
    return torch.where(x < 0, -mag, mag)


def _scale_to_internal(x: torch.Tensor, internal_width: int) -> torch.Tensor:
    shift = internal_width - 7
    if shift >= 0:
        return torch.bitwise_left_shift(x, shift)
    return _signed_shift_right_towards_zero(x, torch.full_like(x, -shift))


def _accumulator_base(acc_sig: torch.Tensor, internal_width: int) -> torch.Tensor:
    if internal_width < FP32_SIGNIFICAND_WIDTH:
        return acc_sig >> (FP32_SIGNIFICAND_WIDTH - internal_width)
    return acc_sig << (internal_width - FP32_SIGNIFICAND_WIDTH)


def _normalize_total(
    total: torch.Tensor,
    max_exp: torch.Tensor,
    internal_width: int,
    zero_exponent: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out_sign = total < 0
    magnitude = total.abs()
    nonzero = magnitude != 0
    width = torch.zeros_like(magnitude)
    if nonzero.any():
        width[nonzero] = (
            torch.floor(torch.log2(magnitude[nonzero].to(torch.float64))).to(torch.int64) + 1
        )

    out_exp = max_exp + width - internal_width
    right = (width - internal_width).clamp_min(0)
    left = (internal_width - width).clamp_min(0)
    out_sig = torch.where(
        width > internal_width,
        torch.bitwise_right_shift(magnitude, right),
        torch.bitwise_left_shift(magnitude, left),
    )

    subnormal_shift = (FP32_MIN_NONZERO_EXPONENT - out_exp).clamp_min(0)
    out_sig = torch.where(
        out_exp < FP32_MIN_NONZERO_EXPONENT,
        torch.bitwise_right_shift(out_sig, subnormal_shift),
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
    out_exp = torch.where(out_sig != 0, out_exp, torch.full_like(out_exp, zero_exponent))
    out_sign = torch.where(out_sig != 0, out_sign, torch.zeros_like(out_sign))
    return out_sign, out_exp, out_sig


def _onehot_by_exp(
    exp_eff: torch.Tensor,
    signed_sig: torch.Tensor,
    nonzero: torch.Tensor,
    start: int,
    stop: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    exp_group = exp_eff[:, start:stop] - 1
    sig_group = signed_sig[:, start:stop]
    mask = nonzero[:, start:stop]
    onehot = torch.nn.functional.one_hot(exp_group.clamp(0, EXP_BUCKETS - 1), EXP_BUCKETS)
    onehot = onehot.permute(2, 0, 1).to(torch.float32)
    onehot = onehot * mask.unsqueeze(0).to(torch.float32)
    signed = onehot * sig_group.unsqueeze(0).to(torch.float32)
    return onehot, signed


def moment_bucket_sum(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
    candidate: Candidate,
) -> tuple[torch.Tensor, BucketStats]:
    """Approximate Hawkeye with exponent-pair counts and signed moments.

    This uses 225 count products and 225 signed-moment products per K group.
    It is exact for max-exponent recovery, but approximate for contribution
    values when individual product truncation happens before summation.
    """
    a_exp, a_sig, a_nonzero = _decode_fp8_fields(x_fp8)
    b_exp, b_sig, b_nonzero = _decode_fp8_fields(w_fp8)
    m_size, k_size = a_exp.shape
    n_size = b_exp.shape[0]
    device = x_fp8.device

    acc_sign = torch.zeros((m_size, n_size), device=device, dtype=torch.bool)
    acc_exp = torch.full((m_size, n_size), candidate.zero_exponent, device=device, dtype=torch.int64)
    acc_sig = torch.zeros((m_size, n_size), device=device, dtype=torch.int64)
    exp_values = torch.arange(1, EXP_BUCKETS + 1, device=device, dtype=torch.int64)
    prod_exp = exp_values[:, None] + exp_values[None, :] - 14
    groups = 0

    for k0 in range(0, k_size, candidate.products_per_group):
        groups += 1
        k1 = min(k0 + candidate.products_per_group, k_size)
        a_mask, a_signed = _onehot_by_exp(a_exp, a_sig, a_nonzero, k0, k1)
        b_mask, b_signed = _onehot_by_exp(b_exp, b_sig, b_nonzero, k0, k1)

        counts = torch.einsum("emk,fnk->efmn", a_mask, b_mask).round().to(torch.int64)
        moments = torch.einsum("emk,fnk->efmn", a_signed, b_signed).round().to(torch.int64)

        present_exp = torch.where(
            counts != 0,
            prod_exp[:, :, None, None],
            torch.full_like(counts, candidate.zero_exponent),
        ).amax(dim=(0, 1))
        acc_exp_eff = torch.where(
            acc_sig != 0,
            acc_exp,
            torch.full_like(acc_exp, candidate.zero_exponent),
        )
        max_exp = torch.maximum(acc_exp_eff, present_exp)

        scaled = _scale_to_internal(moments, candidate.internal_width)
        shifts = max_exp[None, None, :, :] - prod_exp[:, :, None, None]
        contribution = _signed_shift_right_towards_zero(scaled, shifts).sum(dim=(0, 1))

        acc_base = _accumulator_base(acc_sig, candidate.internal_width)
        aligned_acc = _signed_shift_right_towards_zero(acc_base, max_exp - acc_exp_eff)
        aligned_acc = torch.where(acc_sign, -aligned_acc, aligned_acc)
        total = aligned_acc + contribution
        acc_sign, acc_exp, acc_sig = _normalize_total(
            total,
            max_exp,
            candidate.internal_width,
            candidate.zero_exponent,
        )

    y = _gfloat_to_float32(acc_sign, acc_exp, acc_sig)
    y = y * x_scale * w_scale.reshape(1, -1).to(torch.float32)
    stats = BucketStats(
        products_per_group=candidate.products_per_group,
        groups=groups,
        count_products_per_group=EXP_BUCKETS * EXP_BUCKETS,
        moment_products_per_group=EXP_BUCKETS * EXP_BUCKETS,
        total_checkable_products=groups * EXP_BUCKETS * EXP_BUCKETS * 2,
    )
    return y.to(torch.bfloat16), stats


def _class_ids(
    exp_eff: torch.Tensor,
    signed_sig: torch.Tensor,
    nonzero: torch.Tensor,
) -> torch.Tensor:
    ids = (exp_eff - 1) * SIGNED_SIG_BUCKETS + (signed_sig + SIGNED_SIG_OFFSET)
    return torch.where(nonzero, ids, torch.full_like(ids, -1))


def _onehot_by_class(
    class_ids: torch.Tensor,
    classes: torch.Tensor,
    start: int,
    stop: int,
) -> torch.Tensor:
    group = class_ids[:, start:stop]
    return (group.unsqueeze(0) == classes[:, None, None]).to(torch.float32)


def class_pair_exact_sum(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
    candidate: Candidate,
) -> tuple[torch.Tensor, BucketStats]:
    """Exact Hawkeye replay via FP8 class-pair count products.

    This is only intended for small tensors.  It demonstrates that the scalar
    Hawkeye accumulator can be represented with ordinary bucket-count matmuls.
    """
    a_exp, a_sig, a_nonzero = _decode_fp8_fields(x_fp8)
    b_exp, b_sig, b_nonzero = _decode_fp8_fields(w_fp8)
    a_class = _class_ids(a_exp, a_sig, a_nonzero)
    b_class = _class_ids(b_exp, b_sig, b_nonzero)
    all_classes = torch.unique(torch.cat([a_class[a_class >= 0], b_class[b_class >= 0]]))

    m_size, k_size = a_class.shape
    n_size = b_class.shape[0]
    device = x_fp8.device
    acc_sign = torch.zeros((m_size, n_size), device=device, dtype=torch.bool)
    acc_exp = torch.full((m_size, n_size), candidate.zero_exponent, device=device, dtype=torch.int64)
    acc_sig = torch.zeros((m_size, n_size), device=device, dtype=torch.int64)
    groups = 0
    dynamic_products = 0

    class_exp = all_classes // SIGNED_SIG_BUCKETS + 1
    class_sig = all_classes % SIGNED_SIG_BUCKETS - SIGNED_SIG_OFFSET
    prod_exp = class_exp[:, None] + class_exp[None, :] - 14
    signed_product = class_sig[:, None] * class_sig[None, :]

    for k0 in range(0, k_size, candidate.products_per_group):
        groups += 1
        k1 = min(k0 + candidate.products_per_group, k_size)
        a_mask = _onehot_by_class(a_class, all_classes, k0, k1)
        b_mask = _onehot_by_class(b_class, all_classes, k0, k1)
        counts = torch.einsum("cmk,dnk->cdmn", a_mask, b_mask).round().to(torch.int64)
        dynamic_products += int(all_classes.numel() ** 2)

        present_exp = torch.where(
            counts != 0,
            prod_exp[:, :, None, None],
            torch.full_like(counts, candidate.zero_exponent),
        ).amax(dim=(0, 1))
        acc_exp_eff = torch.where(
            acc_sig != 0,
            acc_exp,
            torch.full_like(acc_exp, candidate.zero_exponent),
        )
        max_exp = torch.maximum(acc_exp_eff, present_exp)

        scaled = _scale_to_internal(signed_product[:, :, None, None], candidate.internal_width)
        shifts = max_exp[None, None, :, :] - prod_exp[:, :, None, None]
        aligned_value = _signed_shift_right_towards_zero(scaled, shifts)
        contribution = (counts * aligned_value).sum(dim=(0, 1))

        acc_base = _accumulator_base(acc_sig, candidate.internal_width)
        aligned_acc = _signed_shift_right_towards_zero(acc_base, max_exp - acc_exp_eff)
        aligned_acc = torch.where(acc_sign, -aligned_acc, aligned_acc)
        total = aligned_acc + contribution
        acc_sign, acc_exp, acc_sig = _normalize_total(
            total,
            max_exp,
            candidate.internal_width,
            candidate.zero_exponent,
        )

    y = _gfloat_to_float32(acc_sign, acc_exp, acc_sig)
    y = y * x_scale * w_scale.reshape(1, -1).to(torch.float32)
    static_class_products = EXP_BUCKETS * (SIGNED_SIG_BUCKETS - 1)
    stats = BucketStats(
        products_per_group=candidate.products_per_group,
        groups=groups,
        count_products_per_group=int(all_classes.numel() ** 2),
        moment_products_per_group=0,
        total_checkable_products=dynamic_products,
        static_upper_bound_products=groups * static_class_products**2,
    )
    return y.to(torch.bfloat16), stats


def _random_case(
    m_size: int,
    n_size: int,
    k_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m_size, k_size), generator=generator, device="cuda", dtype=torch.bfloat16)
    w = torch.randn((n_size, k_size), generator=generator, device="cuda", dtype=torch.float32)
    w_scale = (w.abs().amax(dim=1, keepdim=True) / 448.0).clamp_min(1e-12)
    w_fp8 = (w / w_scale).to(torch.float8_e4m3fn).contiguous()
    x_fp8, x_scale = per_token_fp8(x)
    return x_fp8, x_scale, w_fp8, w_scale


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
    x = torch.randn((m_size, w_fp8.shape[1]), generator=generator, device="cuda", dtype=torch.bfloat16)
    x_fp8, x_scale = per_token_fp8(x)
    meta = {"model": model_id, "layer": layer_name}
    return x_fp8, x_scale, w_fp8, w_scale, meta


def _metrics_against(ref: torch.Tensor, cand: torch.Tensor) -> dict[str, float]:
    return metrics(ref, cand)


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    candidate = Candidate(
        name=f"g{args.group}_w{args.width}",
        products_per_group=args.group,
        internal_width=args.width,
    )
    if args.real_weight:
        x_fp8, x_scale, w_fp8, w_scale, meta = _real_weight_case(args.m, args.seed, args.model_id)
    else:
        x_fp8, x_scale, w_fp8, w_scale = _random_case(args.m, args.n, args.k, args.seed)
        meta = {}

    ref = fp8_scaled_mm(x_fp8, x_scale, w_fp8, w_scale)
    codebook = exact_codebook_sum(x_fp8, x_scale, w_fp8, w_scale)
    hawkeye = hawkeye_grouped_sum(x_fp8, x_scale, w_fp8, w_scale, candidate)
    moment, moment_stats = moment_bucket_sum(x_fp8, x_scale, w_fp8, w_scale, candidate)

    result: dict[str, object] = {
        **meta,
        "shape": [int(x_fp8.shape[0]), int(w_fp8.shape[0]), int(x_fp8.shape[1])],
        "seed": args.seed,
        "device": torch.cuda.get_device_name(0),
        "candidate": asdict(candidate),
        "vs_fp8_teacher": {
            "exact_codebook": _metrics_against(ref, codebook),
            "hawkeye_scalar": _metrics_against(ref, hawkeye),
            "moment_buckets": _metrics_against(ref, moment),
        },
        "vs_hawkeye_scalar": {
            "moment_buckets": _metrics_against(hawkeye, moment),
        },
        "bucket_products": {
            "moment_buckets": asdict(moment_stats),
        },
    }
    if args.exact_class:
        exact, exact_stats = class_pair_exact_sum(x_fp8, x_scale, w_fp8, w_scale, candidate)
        result["vs_fp8_teacher"]["class_pair_exact"] = _metrics_against(ref, exact)
        result["vs_hawkeye_scalar"]["class_pair_exact"] = _metrics_against(hawkeye, exact)
        result["bucket_products"]["class_pair_exact"] = asdict(exact_stats)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--k", type=int, default=896)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group", type=int, default=32)
    parser.add_argument("--width", type=int, default=17)
    parser.add_argument("--real-weight", action="store_true")
    parser.add_argument("--exact-class", action="store_true")
    parser.add_argument("--model-id", default="RedHatAI/Qwen2.5-0.5B-FP8-dynamic")
    return parser.parse_args()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")
    result = run_experiment(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
