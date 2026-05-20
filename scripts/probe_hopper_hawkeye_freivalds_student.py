"""Exact Freivalds-checkable Hawkeye student for Hopper FP8 QGMMA.

This implements the expensive-but-clean construction from
reports/hawkeye_freivalds_design.md.  For each K=32 group, FP8 operands are
bucketed by exact `(effective_exponent, signed_significand)` class.  Bucket
counts are computed by ordinary integer matrix products:

    A_chunk[M * class_a, K_group] @ B_chunk[K_group, N * class_b]

Each such product is Freivalds-checkable with the same verifier shape as any
normal integer GEMM.  Everything after the checked products is deterministic
Hawkeye replay: max exponent, truncating shifts, accumulator normalization,
scale multiplication, and bf16 cast.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

import torch

import int_model_approximation.__main__ as entry
from int_model_approximation.hawkeye_freivalds import exact_hawkeye_fp8_sum
from hopper_qgmma_teacher import qgmma_fp8_scaled_mm, require_hopper
from probe_hawkeye_bucket_products import (
    SIGNED_SIG_BUCKETS,
    _accumulator_base,
    _class_ids,
    _decode_fp8_fields,
    _normalize_total,
    _signed_shift_right_towards_zero,
    _scale_to_internal,
)
from probe_hawkeye_l40s_fp8 import (
    Candidate,
    _gfloat_to_float32,
    fp8_scaled_mm,
    hawkeye_grouped_sum,
    metrics,
    per_token_fp8,
)

PACKED_COUNT_BASE_LOG2 = 6
PACKED_COUNT_BASE = 1 << PACKED_COUNT_BASE_LOG2
MAX_PACKED_COUNT_LANES = 6


@dataclass(frozen=True)
class FreivaldsStats:
    products_per_group: int
    groups: int
    class_products_first_pass: int
    class_products_replay_pass: int
    total_checkable_products: int
    max_activation_classes_per_group: int
    max_weight_classes_per_group: int
    class_chunk: int
    packed_count_lanes: int


def _class_chunk_product(
    a_class_group: torch.Tensor,
    b_class_group: torch.Tensor,
    a_classes: torch.Tensor,
    b_classes: torch.Tensor,
) -> torch.Tensor:
    """Return class-pair counts from one ordinary int32 matmul."""
    m_size, group_k = a_class_group.shape
    n_size = b_class_group.shape[0]
    a_masks = (a_class_group.unsqueeze(0) == a_classes[:, None, None]).to(torch.int32)
    b_masks = (b_class_group.unsqueeze(0) == b_classes[:, None, None]).to(torch.int32)
    a_flat = a_masks.reshape(a_classes.numel() * m_size, group_k).contiguous()
    b_flat = b_masks.permute(2, 0, 1).reshape(group_k, b_classes.numel() * n_size)
    counts = entry._int32_raw_matmul(a_flat, b_flat.contiguous())
    return counts.reshape(a_classes.numel(), m_size, b_classes.numel(), n_size).permute(
        0,
        2,
        1,
        3,
    )


def _packed_class_chunk_product(
    a_class_group: torch.Tensor,
    b_class_group: torch.Tensor,
    a_classes: torch.Tensor,
    b_classes: torch.Tensor,
    packed_count_lanes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return packed class-pair counts from one ordinary int32 matmul."""
    m_size, group_k = a_class_group.shape
    n_size = b_class_group.shape[0]
    packed_groups = (b_classes.numel() + packed_count_lanes - 1) // packed_count_lanes
    packed_b_classes = torch.full(
        (packed_groups, packed_count_lanes),
        -2,
        device=b_classes.device,
        dtype=b_classes.dtype,
    )
    packed_b_classes.reshape(-1)[: b_classes.numel()] = b_classes
    lane_powers = torch.tensor(
        [PACKED_COUNT_BASE**lane for lane in range(packed_count_lanes)],
        device=b_classes.device,
        dtype=torch.int32,
    )

    a_masks = (a_class_group.unsqueeze(0) == a_classes[:, None, None]).to(torch.int32)
    b_matches = b_class_group[None, None, :, :] == packed_b_classes[:, :, None, None]
    b_packed = (
        b_matches.to(torch.int32) * lane_powers[None, :, None, None]
    ).sum(dim=1).to(torch.int32)
    a_flat = a_masks.reshape(a_classes.numel() * m_size, group_k).contiguous()
    b_flat = b_packed.permute(2, 0, 1).reshape(group_k, packed_groups * n_size)
    counts = entry._int32_raw_matmul(a_flat, b_flat.contiguous())
    counts = counts.reshape(a_classes.numel(), m_size, packed_groups, n_size).permute(
        0,
        2,
        1,
        3,
    )
    return counts, packed_b_classes


def _iter_class_chunks(classes: torch.Tensor, chunk_size: int) -> list[torch.Tensor]:
    return [classes[start : start + chunk_size] for start in range(0, classes.numel(), chunk_size)]


def _group_present_exponent(
    a_exp_group: torch.Tensor,
    b_exp_group: torch.Tensor,
    a_nonzero_group: torch.Tensor,
    b_nonzero_group: torch.Tensor,
    zero_exponent: int,
) -> torch.Tensor:
    present = a_nonzero_group[:, None, :] & b_nonzero_group[None, :, :]
    prod_exp = a_exp_group[:, None, :] + b_exp_group[None, :, :] - 14
    return torch.where(
        present,
        prod_exp,
        torch.full_like(prod_exp, zero_exponent),
    ).amax(dim=2)


def freivalds_exact_hawkeye_sum(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
    candidate: Candidate,
    class_chunk: int,
    packed_count_lanes: int,
) -> tuple[torch.Tensor, FreivaldsStats]:
    if packed_count_lanes < 1 or packed_count_lanes > MAX_PACKED_COUNT_LANES:
        raise ValueError(
            f"packed_count_lanes must be in [1, {MAX_PACKED_COUNT_LANES}], "
            f"got {packed_count_lanes}"
        )
    a_exp, a_sig, a_nonzero = _decode_fp8_fields(x_fp8)
    b_exp, b_sig, b_nonzero = _decode_fp8_fields(w_fp8)
    a_class = _class_ids(a_exp, a_sig, a_nonzero)
    b_class = _class_ids(b_exp, b_sig, b_nonzero)

    m_size, k_size = a_class.shape
    n_size = b_class.shape[0]
    device = x_fp8.device
    acc_sign = torch.zeros((m_size, n_size), device=device, dtype=torch.bool)
    acc_exp = torch.full((m_size, n_size), candidate.zero_exponent, device=device, dtype=torch.int64)
    acc_sig = torch.zeros((m_size, n_size), device=device, dtype=torch.int64)

    groups = 0
    first_pass_products = 0
    replay_pass_products = 0
    max_a_classes = 0
    max_b_classes = 0

    for k0 in range(0, k_size, candidate.products_per_group):
        groups += 1
        k1 = min(k0 + candidate.products_per_group, k_size)
        a_group = a_class[:, k0:k1]
        b_group = b_class[:, k0:k1]
        a_exp_group = a_exp[:, k0:k1]
        b_exp_group = b_exp[:, k0:k1]
        a_nonzero_group = a_nonzero[:, k0:k1]
        b_nonzero_group = b_nonzero[:, k0:k1]
        a_classes = torch.unique(a_group[a_group >= 0])
        b_classes = torch.unique(b_group[b_group >= 0])
        max_a_classes = max(max_a_classes, int(a_classes.numel()))
        max_b_classes = max(max_b_classes, int(b_classes.numel()))

        acc_exp_eff = torch.where(
            acc_sig != 0,
            acc_exp,
            torch.full_like(acc_exp, candidate.zero_exponent),
        )
        present_exp = _group_present_exponent(
            a_exp_group,
            b_exp_group,
            a_nonzero_group,
            b_nonzero_group,
            candidate.zero_exponent,
        )
        a_chunks = _iter_class_chunks(a_classes, class_chunk)
        b_chunk_size = class_chunk if packed_count_lanes == 1 else class_chunk * packed_count_lanes
        b_chunks = _iter_class_chunks(b_classes, b_chunk_size)

        max_exp = torch.maximum(acc_exp_eff, present_exp)
        contribution = torch.zeros((m_size, n_size), device=device, dtype=torch.int64)

        for a_chunk in a_chunks:
            a_chunk_exp = a_chunk // SIGNED_SIG_BUCKETS + 1
            a_chunk_sig = a_chunk % SIGNED_SIG_BUCKETS - 15
            for b_chunk in b_chunks:
                if packed_count_lanes == 1:
                    b_chunk_exp = b_chunk // SIGNED_SIG_BUCKETS + 1
                    b_chunk_sig = b_chunk % SIGNED_SIG_BUCKETS - 15
                    counts = _class_chunk_product(a_group, b_group, a_chunk, b_chunk)
                    replay_pass_products += 1
                    prod_exp = a_chunk_exp[:, None] + b_chunk_exp[None, :] - 14
                    signed_product = a_chunk_sig[:, None] * b_chunk_sig[None, :]
                    scaled = _scale_to_internal(
                        signed_product[:, :, None, None],
                        candidate.internal_width,
                    )
                    shifts = max_exp[None, None, :, :] - prod_exp[:, :, None, None]
                    aligned_value = _signed_shift_right_towards_zero(scaled, shifts)
                    contribution += (counts * aligned_value).sum(dim=(0, 1))
                    continue

                packed_counts, packed_b_classes = _packed_class_chunk_product(
                    a_group,
                    b_group,
                    a_chunk,
                    b_chunk,
                    packed_count_lanes,
                )
                replay_pass_products += 1
                lane_shifts = torch.arange(
                    packed_count_lanes,
                    device=packed_counts.device,
                    dtype=torch.int64,
                ) * PACKED_COUNT_BASE_LOG2
                counts = torch.bitwise_and(
                    torch.bitwise_right_shift(
                        packed_counts[:, :, None, :, :],
                        lane_shifts[None, None, :, None, None],
                    ),
                    PACKED_COUNT_BASE - 1,
                )
                valid = packed_b_classes >= 0
                counts = torch.where(
                    valid[None, :, :, None, None],
                    counts,
                    torch.zeros_like(counts),
                )
                b_chunk_exp = packed_b_classes // SIGNED_SIG_BUCKETS + 1
                b_chunk_sig = packed_b_classes % SIGNED_SIG_BUCKETS - 15
                prod_exp = a_chunk_exp[:, None, None] + b_chunk_exp[None, :, :] - 14
                signed_product = a_chunk_sig[:, None, None] * b_chunk_sig[None, :, :]
                scaled = _scale_to_internal(
                    signed_product[:, :, :, None, None],
                    candidate.internal_width,
                )
                shifts = max_exp[None, None, None, :, :] - prod_exp[:, :, :, None, None]
                aligned_value = _signed_shift_right_towards_zero(scaled, shifts)
                contribution += (counts * aligned_value).sum(dim=(0, 1, 2))

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
    stats = FreivaldsStats(
        products_per_group=candidate.products_per_group,
        groups=groups,
        class_products_first_pass=first_pass_products,
        class_products_replay_pass=replay_pass_products,
        total_checkable_products=first_pass_products + replay_pass_products,
        max_activation_classes_per_group=max_a_classes,
        max_weight_classes_per_group=max_b_classes,
        class_chunk=class_chunk,
        packed_count_lanes=packed_count_lanes,
    )
    return y.to(torch.bfloat16), stats


def _random_case(
    m_size: int,
    n_size: int,
    k_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m_size, k_size), generator=generator, device="cuda", dtype=torch.bfloat16)
    w = torch.randn((n_size, k_size), generator=generator, device="cuda", dtype=torch.float32)
    w_scale = (w.abs().amax(dim=1, keepdim=True) / entry.FP8_E4M3_MAX).clamp_min(1e-12)
    w_fp8 = (w / w_scale).to(torch.float8_e4m3fn).contiguous()
    x_fp8, x_scale = per_token_fp8(x)
    return x_fp8, x_scale, w_fp8, w_scale, {"case": "random"}


def _real_weight_case(
    m_size: int,
    seed: int,
    model_id: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
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
            break
    else:
        raise RuntimeError("no FP8 weight with weight_scale found")

    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m_size, w_fp8.shape[1]), generator=generator, device="cuda", dtype=torch.bfloat16)
    x_fp8, x_scale = per_token_fp8(x)
    return (
        x_fp8,
        x_scale,
        w_fp8,
        w_scale,
        {"case": "real_weight", "model": model_id, "layer": name},
    )


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    capability = require_hopper()
    candidate = Candidate(
        name=f"freivalds_exact_hopper_g{args.group}_w{args.width}",
        products_per_group=args.group,
        internal_width=args.width,
    )
    if args.real_weight:
        x_fp8, x_scale, w_fp8, w_scale, meta = _real_weight_case(args.m, args.seed, args.model_id)
    else:
        x_fp8, x_scale, w_fp8, w_scale, meta = _random_case(args.m, args.n, args.k, args.seed)

    qgmma_teacher = qgmma_fp8_scaled_mm(x_fp8, x_scale, w_fp8, w_scale)
    cublas_teacher = fp8_scaled_mm(x_fp8, x_scale, w_fp8, w_scale)
    hawkeye_scalar = hawkeye_grouped_sum(x_fp8, x_scale, w_fp8, w_scale, candidate)
    if args.packed_count_lanes == 1 and args.count_matmul == "int8":
        raw_matmul = entry._int8_count_matmul
        count_dtype = torch.int8
    elif args.packed_count_lanes == 1 and args.count_matmul == "int32":
        raw_matmul = entry._int32_count_matmul
        count_dtype = torch.int32
    else:
        raw_matmul = entry._int32_raw_matmul
        count_dtype = torch.int32
    freivalds_student, stats = exact_hawkeye_fp8_sum(
        x_fp8,
        x_scale,
        w_fp8,
        w_scale,
        raw_matmul,
        products_per_group=candidate.products_per_group,
        internal_width=candidate.internal_width,
        zero_exponent=candidate.zero_exponent,
        class_chunk=args.class_chunk,
        packed_count_lanes=args.packed_count_lanes,
        fused_replay=args.fused_replay,
        cache_weight_chunks=args.cache_weight_chunks,
        count_dtype=count_dtype,
    )

    return {
        **meta,
        "shape": [int(x_fp8.shape[0]), int(w_fp8.shape[0]), int(x_fp8.shape[1])],
        "seed": args.seed,
        "device": torch.cuda.get_device_name(0),
        "compute_capability": [int(capability[0]), int(capability[1])],
        "candidate": asdict(candidate),
        "student": {
            "name": "exact_class_pair_hawkeye",
            "matmul_shape": (
                "(class_chunk * M) x K_group @ "
                "K_group x (class_chunk * N), with packed_count_lanes logical classes per column"
            ),
            "all_expensive_work": "ordinary int32 x int32 -> int64 matmuls",
            "freivalds_checkable": True,
            "packed_count_base": PACKED_COUNT_BASE,
            "fused_replay": args.fused_replay,
            "cache_weight_chunks": args.cache_weight_chunks,
            "count_matmul": args.count_matmul,
        },
        "checkable_products": asdict(stats),
        "qgmma_vs_cublas_teacher": metrics(cublas_teacher, qgmma_teacher),
        "vs_qgmma_teacher": {
            "hawkeye_scalar": metrics(qgmma_teacher, hawkeye_scalar),
            "freivalds_student": metrics(qgmma_teacher, freivalds_student),
        },
        "vs_hawkeye_scalar": {
            "freivalds_student": metrics(hawkeye_scalar, freivalds_student),
        },
        "vs_cublas_teacher": {
            "hawkeye_scalar": metrics(cublas_teacher, hawkeye_scalar),
            "freivalds_student": metrics(cublas_teacher, freivalds_student),
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
    parser.add_argument("--class-chunk", type=int, default=32)
    parser.add_argument("--packed-count-lanes", type=int, default=MAX_PACKED_COUNT_LANES)
    parser.add_argument("--fused-replay", action="store_true")
    parser.add_argument("--cache-weight-chunks", action="store_true")
    parser.add_argument("--count-matmul", choices=["int64", "int32", "int8"], default="int64")
    parser.add_argument("--real-weight", action="store_true")
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
