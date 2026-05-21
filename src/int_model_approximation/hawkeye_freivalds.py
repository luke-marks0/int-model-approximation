"""Freivalds-checkable exact Hawkeye FP8 reconstruction utilities."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import triton
import triton.language as tl


FP32_SIGNIFICAND_WIDTH = 24
FP32_MIN_NONZERO_EXPONENT = -126
SIGNED_SIG_BUCKETS = 31
SIGNED_SIG_OFFSET = 15
PACKED_COUNT_BASE_LOG2 = 6
PACKED_COUNT_BASE = 1 << PACKED_COUNT_BASE_LOG2
MAX_PACKED_COUNT_LANES = 6


@dataclass(frozen=True)
class HawkeyeFreivaldsStats:
    products_per_group: int
    groups: int
    packed_group_lanes: int
    class_products_first_pass: int
    class_products_replay_pass: int
    total_checkable_products: int
    max_activation_classes_per_group: int
    max_weight_classes_per_group: int
    class_chunk: int
    packed_count_lanes: int


RawIntMatmul = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


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


def _class_ids(
    exp_eff: torch.Tensor,
    signed_sig: torch.Tensor,
    nonzero: torch.Tensor,
) -> torch.Tensor:
    ids = (exp_eff - 1) * SIGNED_SIG_BUCKETS + (signed_sig + SIGNED_SIG_OFFSET)
    return torch.where(nonzero, ids, torch.full_like(ids, -1))


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
    for bit in range(0, 63):
        width = torch.where(magnitude >= (1 << bit), bit + 1, width)

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
    out_exp = torch.maximum(out_exp, torch.full_like(out_exp, FP32_MIN_NONZERO_EXPONENT))

    if internal_width < FP32_SIGNIFICAND_WIDTH:
        out_sig = out_sig << (FP32_SIGNIFICAND_WIDTH - internal_width)
    else:
        out_sig = out_sig >> (internal_width - FP32_SIGNIFICAND_WIDTH)

    out_sig = torch.where(nonzero & (out_sig != 0), out_sig, torch.zeros_like(out_sig))
    out_exp = torch.where(out_sig != 0, out_exp, torch.full_like(out_exp, zero_exponent))
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
    exponent_bits = exponent_bits.clamp(0, 254)
    mantissa = significand.to(torch.int64) & 0x7FFFFF
    bits = (
        (sign.to(torch.int64) << 31)
        | (exponent_bits.to(torch.int64) << 23)
        | mantissa
    )
    bits = torch.where(
        significand != 0,
        bits,
        torch.zeros_like(bits),
    )
    return bits.to(torch.int32).view(torch.float32)


def _class_chunk_product(
    a_class_group: torch.Tensor,
    b_class_group: torch.Tensor,
    a_classes: torch.Tensor,
    b_classes: torch.Tensor,
    raw_matmul: RawIntMatmul,
    *,
    count_dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    counts, a_size, b_size = _class_chunk_product_raw(
        a_class_group,
        b_class_group,
        a_classes,
        b_classes,
        raw_matmul,
        count_dtype=count_dtype,
    )
    m_size = a_class_group.shape[0]
    n_size = b_class_group.shape[0]
    return counts.reshape(a_size, m_size, b_size, n_size).permute(0, 2, 1, 3)


def _class_chunk_product_raw(
    a_class_group: torch.Tensor,
    b_class_group: torch.Tensor,
    a_classes: torch.Tensor,
    b_classes: torch.Tensor,
    raw_matmul: RawIntMatmul,
    *,
    count_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, int, int]:
    m_size, group_k = a_class_group.shape
    n_size = b_class_group.shape[0]
    a_masks = (a_class_group.unsqueeze(0) == a_classes[:, None, None]).to(count_dtype)
    b_masks = (b_class_group.unsqueeze(0) == b_classes[:, None, None]).to(count_dtype)
    a_flat = a_masks.reshape(a_classes.numel() * m_size, group_k).contiguous()
    b_flat = b_masks.permute(2, 0, 1).reshape(group_k, b_classes.numel() * n_size)
    counts = raw_matmul(a_flat, b_flat.contiguous())
    return counts, int(a_classes.numel()), int(b_classes.numel())


def _weight_chunk_flat(
    b_class_group: torch.Tensor,
    b_classes: torch.Tensor,
    *,
    count_dtype: torch.dtype = torch.int32,
) -> torch.Tensor:
    n_size, _group_k = b_class_group.shape
    b_masks = (b_class_group.unsqueeze(0) == b_classes[:, None, None]).to(count_dtype)
    return b_masks.permute(2, 0, 1).reshape(
        b_class_group.shape[1],
        b_classes.numel() * n_size,
    ).contiguous()


def _class_chunk_product_raw_with_b_flat(
    a_class_group: torch.Tensor,
    a_classes: torch.Tensor,
    b_flat: torch.Tensor,
    b_size: int,
    raw_matmul: RawIntMatmul,
) -> tuple[torch.Tensor, int, int]:
    m_size, group_k = a_class_group.shape
    a_masks = (a_class_group.unsqueeze(0) == a_classes[:, None, None]).to(b_flat.dtype)
    a_flat = a_masks.reshape(a_classes.numel() * m_size, group_k).contiguous()
    counts = raw_matmul(a_flat, b_flat)
    return counts, int(a_classes.numel()), b_size


@triton.jit
def _unpacked_contribution_kernel(
    counts_ptr,
    a_exp_ptr,
    a_sig_ptr,
    b_exp_ptr,
    b_sig_ptr,
    max_exp_ptr,
    contribution_ptr,
    m_size: tl.constexpr,
    n_size: tl.constexpr,
    a_size,
    b_size,
    internal_width: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    class_chunk: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    mask = (offs_m[:, None] < m_size) & (offs_n[None, :] < n_size)
    max_exp = tl.load(
        max_exp_ptr + offs_m[:, None] * n_size + offs_n[None, :],
        mask=mask,
        other=0,
    ).to(tl.int64)
    accum = tl.zeros((block_m, block_n), dtype=tl.int64)

    for ai in range(0, class_chunk):
        a_valid = ai < a_size
        a_exp = tl.load(a_exp_ptr + ai, mask=a_valid, other=0).to(tl.int64)
        a_sig = tl.load(a_sig_ptr + ai, mask=a_valid, other=0).to(tl.int64)
        for bi in range(0, class_chunk):
            valid = a_valid & (bi < b_size)
            b_exp = tl.load(b_exp_ptr + bi, mask=valid, other=0).to(tl.int64)
            b_sig = tl.load(b_sig_ptr + bi, mask=valid, other=0).to(tl.int64)
            prod_exp = a_exp + b_exp - 14
            signed_product = a_sig * b_sig
            if internal_width >= 7:
                scaled = signed_product << (internal_width - 7)
            else:
                scaled_abs = tl.abs(signed_product)
                scaled_mag = scaled_abs >> (7 - internal_width)
                scaled = tl.where(signed_product < 0, -scaled_mag, scaled_mag)
            shift = max_exp - prod_exp
            shift = tl.minimum(tl.maximum(shift, 0), 62)
            aligned_mag = tl.abs(scaled) >> shift
            aligned = tl.where(scaled < 0, -aligned_mag, aligned_mag)
            count = tl.load(
                counts_ptr
                + (ai * m_size + offs_m[:, None]) * (b_size * n_size)
                + bi * n_size
                + offs_n[None, :],
                mask=mask & valid,
                other=0,
            ).to(tl.int64)
            accum += count * aligned

    old = tl.load(
        contribution_ptr + offs_m[:, None] * n_size + offs_n[None, :],
        mask=mask,
        other=0,
    ).to(tl.int64)
    tl.store(
        contribution_ptr + offs_m[:, None] * n_size + offs_n[None, :],
        old + accum,
        mask=mask,
    )


def _add_unpacked_contribution_fused(
    contribution: torch.Tensor,
    counts: torch.Tensor,
    a_chunk_exp: torch.Tensor,
    a_chunk_sig: torch.Tensor,
    b_chunk_exp: torch.Tensor,
    b_chunk_sig: torch.Tensor,
    max_exp: torch.Tensor,
    *,
    a_size: int,
    b_size: int,
    internal_width: int,
    class_chunk: int,
) -> None:
    if a_size > class_chunk or b_size > class_chunk:
        raise RuntimeError("fused contribution received chunk larger than class_chunk")
    if contribution.device.type != "cuda":
        raise RuntimeError("fused contribution is CUDA-only")
    m_size, n_size = contribution.shape
    block_m = 4
    block_n = 32
    _unpacked_contribution_kernel[(triton.cdiv(m_size, block_m), triton.cdiv(n_size, block_n))](
        counts,
        a_chunk_exp.contiguous(),
        a_chunk_sig.contiguous(),
        b_chunk_exp.contiguous(),
        b_chunk_sig.contiguous(),
        max_exp,
        contribution,
        m_size,
        n_size,
        a_size,
        b_size,
        internal_width,
        block_m=block_m,
        block_n=block_n,
        class_chunk=class_chunk,
        num_warps=4,
    )


@triton.jit
def _count_product_contribution_kernel(
    a_ptr,
    b_ptr,
    counts_ptr,
    a_exp_ptr,
    a_sig_ptr,
    b_exp_ptr,
    b_sig_ptr,
    max_exp_ptr,
    contribution_ptr,
    m_size: tl.constexpr,
    n_size: tl.constexpr,
    k_size: tl.constexpr,
    a_size: tl.constexpr,
    b_size: tl.constexpr,
    internal_width: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    total_n: tl.constexpr = b_size * n_size
    total_m: tl.constexpr = a_size * m_size
    offs_rows = pid_m * block_m + tl.arange(0, block_m)
    offs_cols = pid_n * block_n + tl.arange(0, block_n)
    m_offsets = offs_rows % m_size
    a_offsets = offs_rows // m_size
    n_offsets = offs_cols % n_size
    b_offsets = offs_cols // n_size
    mask = (offs_rows[:, None] < total_m) & (offs_cols[None, :] < total_n)
    accum = tl.zeros((block_m, block_n), dtype=tl.int64)

    for k0 in range(0, k_size, block_k):
        for kk in range(0, block_k):
            k = k0 + kk
            a = tl.load(
                a_ptr + offs_rows * k_size + k,
                mask=(offs_rows < total_m) & (k < k_size),
                other=0,
            ).to(tl.int64)
            b = tl.load(
                b_ptr + k * total_n + offs_cols,
                mask=(k < k_size) & (offs_cols < total_n),
                other=0,
            ).to(tl.int64)
            accum += a[:, None] * b[None, :]

    tl.store(
        counts_ptr + offs_rows[:, None] * total_n + offs_cols[None, :],
        accum,
        mask=mask,
    )

    a_sig = tl.load(a_sig_ptr + a_offsets, mask=offs_rows < total_m, other=0).to(tl.int64)
    a_exp = tl.load(a_exp_ptr + a_offsets, mask=offs_rows < total_m, other=0).to(tl.int64)
    b_exp = tl.load(b_exp_ptr + b_offsets, mask=offs_cols < total_n, other=0).to(tl.int64)
    b_sig = tl.load(b_sig_ptr + b_offsets, mask=offs_cols < total_n, other=0).to(tl.int64)
    max_exp = tl.load(
        max_exp_ptr + m_offsets[:, None] * n_size + n_offsets[None, :],
        mask=mask,
        other=0,
    ).to(tl.int64)

    prod_exp = a_exp[:, None] + b_exp[None, :] - 14
    signed_product = a_sig[:, None] * b_sig[None, :]
    if internal_width >= 7:
        scaled = signed_product << (internal_width - 7)
    else:
        scaled_abs = tl.abs(signed_product)
        scaled_mag = scaled_abs >> (7 - internal_width)
        scaled = tl.where(signed_product < 0, -scaled_mag, scaled_mag)
    shift = max_exp - prod_exp
    shift = tl.minimum(tl.maximum(shift, 0), 62)
    aligned_mag = tl.abs(scaled) >> shift
    aligned = tl.where(scaled < 0, -aligned_mag, aligned_mag)
    delta = accum * aligned
    tl.atomic_add(
        contribution_ptr + m_offsets[:, None] * n_size + n_offsets[None, :],
        delta,
        sem="relaxed",
        mask=mask,
    )


def _class_chunk_product_add_contribution_fused(
    contribution: torch.Tensor,
    a_class_group: torch.Tensor,
    b_class_group: torch.Tensor,
    a_classes: torch.Tensor,
    b_classes: torch.Tensor,
    a_chunk_exp: torch.Tensor,
    a_chunk_sig: torch.Tensor,
    b_chunk_exp: torch.Tensor,
    b_chunk_sig: torch.Tensor,
    max_exp: torch.Tensor,
    *,
    internal_width: int,
) -> torch.Tensor:
    m_size, group_k = a_class_group.shape
    n_size = b_class_group.shape[0]
    a_masks = (a_class_group.unsqueeze(0) == a_classes[:, None, None]).to(torch.int32)
    b_masks = (b_class_group.unsqueeze(0) == b_classes[:, None, None]).to(torch.int32)
    a_flat = a_masks.reshape(a_classes.numel() * m_size, group_k).contiguous()
    b_flat = b_masks.permute(2, 0, 1).reshape(group_k, b_classes.numel() * n_size).contiguous()
    counts = torch.empty(
        (a_flat.shape[0], b_flat.shape[1]),
        device=a_class_group.device,
        dtype=torch.int64,
    )
    _count_product_contribution_kernel[(triton.cdiv(a_flat.shape[0], 16), triton.cdiv(b_flat.shape[1], 16))](
        a_flat,
        b_flat,
        counts,
        a_chunk_exp.contiguous(),
        a_chunk_sig.contiguous(),
        b_chunk_exp.contiguous(),
        b_chunk_sig.contiguous(),
        max_exp,
        contribution,
        m_size,
        n_size,
        group_k,
        int(a_classes.numel()),
        int(b_classes.numel()),
        internal_width,
        block_m=16,
        block_n=16,
        block_k=32,
        num_warps=4,
    )
    return counts


@triton.jit
def _packed_contribution_kernel(
    packed_counts_ptr,
    packed_b_classes_ptr,
    a_exp_ptr,
    a_sig_ptr,
    max_exp_ptr,
    contribution_ptr,
    m_size: tl.constexpr,
    n_size: tl.constexpr,
    a_size: tl.constexpr,
    packed_groups: tl.constexpr,
    packed_count_lanes: tl.constexpr,
    internal_width: tl.constexpr,
    signed_sig_buckets: tl.constexpr,
    signed_sig_offset: tl.constexpr,
    packed_count_base_log2: tl.constexpr,
    packed_count_base: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    class_chunk: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    mask = (offs_m[:, None] < m_size) & (offs_n[None, :] < n_size)
    max_exp = tl.load(
        max_exp_ptr + offs_m[:, None] * n_size + offs_n[None, :],
        mask=mask,
        other=0,
    ).to(tl.int64)
    accum = tl.zeros((block_m, block_n), dtype=tl.int64)

    for ai in range(0, class_chunk):
        a_valid = ai < a_size
        a_exp = tl.load(a_exp_ptr + ai, mask=a_valid, other=0).to(tl.int64)
        a_sig = tl.load(a_sig_ptr + ai, mask=a_valid, other=0).to(tl.int64)
        for group in range(0, packed_groups):
            packed_count = tl.load(
                packed_counts_ptr
                + ((ai * packed_groups + group) * m_size + offs_m[:, None]) * n_size
                + offs_n[None, :],
                mask=mask & a_valid,
                other=0,
            ).to(tl.int64)
            for lane in range(0, packed_count_lanes):
                b_class = tl.load(
                    packed_b_classes_ptr + group * packed_count_lanes + lane,
                    mask=a_valid,
                    other=-2,
                ).to(tl.int64)
                valid = a_valid & (b_class >= 0)
                b_exp = b_class // signed_sig_buckets + 1
                b_sig = b_class % signed_sig_buckets - signed_sig_offset
                prod_exp = a_exp + b_exp - 14
                signed_product = a_sig * b_sig
                if internal_width >= 7:
                    scaled = signed_product << (internal_width - 7)
                else:
                    scaled_abs = tl.abs(signed_product)
                    scaled_mag = scaled_abs >> (7 - internal_width)
                    scaled = tl.where(signed_product < 0, -scaled_mag, scaled_mag)
                shift = max_exp - prod_exp
                shift = tl.minimum(tl.maximum(shift, 0), 62)
                aligned_mag = tl.abs(scaled) >> shift
                aligned = tl.where(scaled < 0, -aligned_mag, aligned_mag)
                count = (packed_count >> (lane * packed_count_base_log2)) & (
                    packed_count_base - 1
                )
                accum += tl.where(valid, count * aligned, 0)

    old = tl.load(
        contribution_ptr + offs_m[:, None] * n_size + offs_n[None, :],
        mask=mask,
        other=0,
    ).to(tl.int64)
    tl.store(
        contribution_ptr + offs_m[:, None] * n_size + offs_n[None, :],
        old + accum,
        mask=mask,
    )


def _add_packed_contribution_fused(
    contribution: torch.Tensor,
    packed_counts: torch.Tensor,
    packed_b_classes: torch.Tensor,
    a_chunk_exp: torch.Tensor,
    a_chunk_sig: torch.Tensor,
    max_exp: torch.Tensor,
    *,
    a_size: int,
    internal_width: int,
    class_chunk: int,
) -> None:
    if a_size > class_chunk:
        raise RuntimeError("packed contribution received chunk larger than class_chunk")
    if contribution.device.type != "cuda":
        raise RuntimeError("packed contribution fusion is CUDA-only")
    m_size, n_size = contribution.shape
    block_m = 4
    block_n = 32
    packed_groups = int(packed_b_classes.shape[0])
    packed_count_lanes = int(packed_b_classes.shape[1])
    _packed_contribution_kernel[(triton.cdiv(m_size, block_m), triton.cdiv(n_size, block_n))](
        packed_counts.contiguous(),
        packed_b_classes.contiguous(),
        a_chunk_exp.contiguous(),
        a_chunk_sig.contiguous(),
        max_exp,
        contribution,
        m_size,
        n_size,
        a_size,
        packed_groups,
        packed_count_lanes,
        internal_width,
        SIGNED_SIG_BUCKETS,
        SIGNED_SIG_OFFSET,
        PACKED_COUNT_BASE_LOG2,
        PACKED_COUNT_BASE,
        block_m=block_m,
        block_n=block_n,
        class_chunk=class_chunk,
        num_warps=4,
    )


def _packed_class_chunk_product(
    a_class_group: torch.Tensor,
    b_class_group: torch.Tensor,
    a_classes: torch.Tensor,
    b_classes: torch.Tensor,
    raw_matmul: RawIntMatmul,
    packed_count_lanes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    counts = raw_matmul(a_flat, b_flat.contiguous())
    counts = counts.reshape(a_classes.numel(), m_size, packed_groups, n_size).permute(
        0,
        2,
        1,
        3,
    )
    return counts, packed_b_classes


def _packed_group_class_chunk_product(
    a_class_pack: torch.Tensor,
    b_class_pack: torch.Tensor,
    a_classes: torch.Tensor,
    b_classes: torch.Tensor,
    raw_matmul: RawIntMatmul,
    *,
    products_per_group: int,
    packed_group_lanes: int,
    packed_count_lanes: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    m_size, pack_k = a_class_pack.shape
    n_size = b_class_pack.shape[0]
    group_count = triton.cdiv(pack_k, products_per_group)
    if group_count > packed_group_lanes:
        raise RuntimeError("group pack contains more groups than packed_group_lanes")
    if packed_group_lanes < 1 or packed_group_lanes > packed_count_lanes:
        raise ValueError("packed_group_lanes must fit inside packed_count_lanes")
    b_class_lanes = packed_count_lanes // packed_group_lanes
    if b_class_lanes < 1:
        raise ValueError("packed_group_lanes leaves no lanes for weight classes")
    total_lanes = packed_group_lanes * b_class_lanes
    if total_lanes > MAX_PACKED_COUNT_LANES:
        raise ValueError("too many packed count lanes")

    packed_b_groups = (b_classes.numel() + b_class_lanes - 1) // b_class_lanes
    packed_b_classes = torch.full(
        (packed_b_groups, b_class_lanes),
        -2,
        device=b_classes.device,
        dtype=b_classes.dtype,
    )
    packed_b_classes.reshape(-1)[: b_classes.numel()] = b_classes
    lane_powers = torch.tensor(
        [PACKED_COUNT_BASE**lane for lane in range(total_lanes)],
        device=b_classes.device,
        dtype=torch.int32,
    )
    group_ids = torch.div(
        torch.arange(pack_k, device=b_classes.device, dtype=torch.int64),
        products_per_group,
        rounding_mode="floor",
    )

    a_masks = (a_class_pack.unsqueeze(0) == a_classes[:, None, None]).to(torch.int32)
    b_matches = b_class_pack[None, None, :, :] == packed_b_classes[:, :, None, None]
    b_packed = torch.zeros(
        (packed_b_groups, n_size, pack_k),
        device=b_classes.device,
        dtype=torch.int32,
    )
    for group_lane in range(group_count):
        k_mask = group_ids == group_lane
        group_lane_powers = lane_powers[
            group_lane * b_class_lanes : (group_lane + 1) * b_class_lanes
        ]
        group_values = (
            b_matches.to(torch.int32) * group_lane_powers[None, :, None, None]
        ).sum(dim=1)
        b_packed += torch.where(k_mask[None, None, :], group_values, 0)

    a_flat = a_masks.reshape(a_classes.numel() * m_size, pack_k).contiguous()
    b_flat = b_packed.permute(2, 0, 1).reshape(pack_k, packed_b_groups * n_size)
    counts = raw_matmul(a_flat, b_flat.contiguous())
    counts = counts.reshape(a_classes.numel(), m_size, packed_b_groups, n_size).permute(
        0,
        2,
        1,
        3,
    )
    return counts, packed_b_classes, b_class_lanes


@triton.jit
def _packed_group_contribution_kernel(
    packed_counts_ptr,
    packed_b_classes_ptr,
    a_exp_ptr,
    a_sig_ptr,
    max_exp_ptr,
    contribution_ptr,
    m_size: tl.constexpr,
    n_size: tl.constexpr,
    a_size: tl.constexpr,
    packed_groups: tl.constexpr,
    group_lane: tl.constexpr,
    b_class_lanes: tl.constexpr,
    internal_width: tl.constexpr,
    signed_sig_buckets: tl.constexpr,
    signed_sig_offset: tl.constexpr,
    packed_count_base_log2: tl.constexpr,
    packed_count_base: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    class_chunk: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    mask = (offs_m[:, None] < m_size) & (offs_n[None, :] < n_size)
    max_exp = tl.load(
        max_exp_ptr + offs_m[:, None] * n_size + offs_n[None, :],
        mask=mask,
        other=0,
    ).to(tl.int64)
    accum = tl.zeros((block_m, block_n), dtype=tl.int64)

    for ai in range(0, class_chunk):
        a_valid = ai < a_size
        a_exp = tl.load(a_exp_ptr + ai, mask=a_valid, other=0).to(tl.int64)
        a_sig = tl.load(a_sig_ptr + ai, mask=a_valid, other=0).to(tl.int64)
        for group in range(0, packed_groups):
            packed_count = tl.load(
                packed_counts_ptr
                + ((ai * packed_groups + group) * m_size + offs_m[:, None]) * n_size
                + offs_n[None, :],
                mask=mask & a_valid,
                other=0,
            ).to(tl.int64)
            for b_lane in range(0, b_class_lanes):
                b_class = tl.load(
                    packed_b_classes_ptr + group * b_class_lanes + b_lane,
                    mask=a_valid,
                    other=-2,
                ).to(tl.int64)
                valid = a_valid & (b_class >= 0)
                b_exp = b_class // signed_sig_buckets + 1
                b_sig = b_class % signed_sig_buckets - signed_sig_offset
                prod_exp = a_exp + b_exp - 14
                signed_product = a_sig * b_sig
                if internal_width >= 7:
                    scaled = signed_product << (internal_width - 7)
                else:
                    scaled_abs = tl.abs(signed_product)
                    scaled_mag = scaled_abs >> (7 - internal_width)
                    scaled = tl.where(signed_product < 0, -scaled_mag, scaled_mag)
                shift = max_exp - prod_exp
                shift = tl.minimum(tl.maximum(shift, 0), 62)
                aligned_mag = tl.abs(scaled) >> shift
                aligned = tl.where(scaled < 0, -aligned_mag, aligned_mag)
                lane = group_lane * b_class_lanes + b_lane
                count = (packed_count >> (lane * packed_count_base_log2)) & (
                    packed_count_base - 1
                )
                accum += tl.where(valid, count * aligned, 0)

    old = tl.load(
        contribution_ptr + offs_m[:, None] * n_size + offs_n[None, :],
        mask=mask,
        other=0,
    ).to(tl.int64)
    tl.store(
        contribution_ptr + offs_m[:, None] * n_size + offs_n[None, :],
        old + accum,
        mask=mask,
    )


def _add_packed_group_contribution_fused(
    contribution: torch.Tensor,
    packed_counts: torch.Tensor,
    packed_b_classes: torch.Tensor,
    a_chunk_exp: torch.Tensor,
    a_chunk_sig: torch.Tensor,
    max_exp: torch.Tensor,
    *,
    group_lane: int,
    b_class_lanes: int,
    a_size: int,
    internal_width: int,
    class_chunk: int,
) -> None:
    if a_size > class_chunk:
        raise RuntimeError("packed group contribution received chunk larger than class_chunk")
    if contribution.device.type != "cuda":
        raise RuntimeError("packed group contribution fusion is CUDA-only")
    m_size, n_size = contribution.shape
    block_m = 4
    block_n = 32
    packed_groups = int(packed_b_classes.shape[0])
    _packed_group_contribution_kernel[(triton.cdiv(m_size, block_m), triton.cdiv(n_size, block_n))](
        packed_counts.contiguous(),
        packed_b_classes.contiguous(),
        a_chunk_exp.contiguous(),
        a_chunk_sig.contiguous(),
        max_exp,
        contribution,
        m_size,
        n_size,
        a_size,
        packed_groups,
        group_lane,
        b_class_lanes,
        internal_width,
        SIGNED_SIG_BUCKETS,
        SIGNED_SIG_OFFSET,
        PACKED_COUNT_BASE_LOG2,
        PACKED_COUNT_BASE,
        block_m=block_m,
        block_n=block_n,
        class_chunk=class_chunk,
        num_warps=4,
    )


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


def _exact_hawkeye_fp8_sum_packed_groups(
    a_exp: torch.Tensor,
    a_nonzero: torch.Tensor,
    b_exp: torch.Tensor,
    b_nonzero: torch.Tensor,
    a_class: torch.Tensor,
    b_class: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    raw_matmul: RawIntMatmul,
    *,
    products_per_group: int,
    internal_width: int,
    zero_exponent: int,
    class_chunk: int,
    packed_count_lanes: int,
    packed_group_lanes: int,
) -> tuple[torch.Tensor, HawkeyeFreivaldsStats]:
    if packed_group_lanes < 1 or packed_group_lanes > packed_count_lanes:
        raise ValueError("packed_group_lanes must fit inside packed_count_lanes")
    if packed_count_lanes // packed_group_lanes < 1:
        raise ValueError("packed_group_lanes leaves no lanes for weight classes")
    if products_per_group > PACKED_COUNT_BASE:
        raise ValueError(
            "packed group counts require products_per_group <= PACKED_COUNT_BASE "
            "to avoid cross-lane carries"
        )

    m_size, k_size = a_class.shape
    n_size = b_class.shape[0]
    device = a_class.device
    acc_sign = torch.zeros((m_size, n_size), device=device, dtype=torch.bool)
    acc_exp = torch.full((m_size, n_size), zero_exponent, device=device, dtype=torch.int64)
    acc_sig = torch.zeros((m_size, n_size), device=device, dtype=torch.int64)

    groups = 0
    replay_pass_products = 0
    max_a_classes = 0
    max_b_classes = 0
    pack_width = products_per_group * packed_group_lanes

    for pack_k0 in range(0, k_size, pack_width):
        pack_k1 = min(pack_k0 + pack_width, k_size)
        pack_group_count = triton.cdiv(pack_k1 - pack_k0, products_per_group)
        a_pack = a_class[:, pack_k0:pack_k1]
        b_pack = b_class[:, pack_k0:pack_k1]
        a_classes = torch.unique(a_pack[a_pack >= 0])
        b_classes = torch.unique(b_pack[b_pack >= 0])
        max_a_classes = max(max_a_classes, int(a_classes.numel()))
        max_b_classes = max(max_b_classes, int(b_classes.numel()))
        a_chunks = _iter_class_chunks(a_classes, class_chunk)
        b_class_lanes = packed_count_lanes // packed_group_lanes
        b_chunk_size = class_chunk * b_class_lanes
        b_chunks = _iter_class_chunks(b_classes, b_chunk_size)

        # Later groups need the normalized accumulator produced by earlier groups,
        # so the packed count products are computed once and replayed sequentially.
        packed_chunk_results: list[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]
        ] = []
        for a_chunk in a_chunks:
            a_chunk_exp = a_chunk // SIGNED_SIG_BUCKETS + 1
            a_chunk_sig = a_chunk % SIGNED_SIG_BUCKETS - SIGNED_SIG_OFFSET
            for b_chunk in b_chunks:
                packed_counts, packed_b_classes, chunk_b_class_lanes = (
                    _packed_group_class_chunk_product(
                        a_pack,
                        b_pack,
                        a_chunk,
                        b_chunk,
                        raw_matmul,
                        products_per_group=products_per_group,
                        packed_group_lanes=packed_group_lanes,
                        packed_count_lanes=packed_count_lanes,
                    )
                )
                replay_pass_products += 1
                packed_chunk_results.append(
                    (
                        packed_counts,
                        packed_b_classes,
                        a_chunk_exp,
                        a_chunk_sig,
                        chunk_b_class_lanes,
                    )
                )

        for group_lane in range(pack_group_count):
            groups += 1
            k0 = pack_k0 + group_lane * products_per_group
            k1 = min(k0 + products_per_group, k_size)
            a_exp_group = a_exp[:, k0:k1]
            b_exp_group = b_exp[:, k0:k1]
            a_nonzero_group = a_nonzero[:, k0:k1]
            b_nonzero_group = b_nonzero[:, k0:k1]
            acc_exp_eff = torch.where(
                acc_sig != 0,
                acc_exp,
                torch.full_like(acc_exp, zero_exponent),
            )
            present_exp = _group_present_exponent(
                a_exp_group,
                b_exp_group,
                a_nonzero_group,
                b_nonzero_group,
                zero_exponent,
            )
            max_exp = torch.maximum(acc_exp_eff, present_exp)
            contribution = torch.zeros((m_size, n_size), device=device, dtype=torch.int64)
            for (
                packed_counts,
                packed_b_classes,
                a_chunk_exp,
                a_chunk_sig,
                chunk_b_class_lanes,
            ) in packed_chunk_results:
                _add_packed_group_contribution_fused(
                    contribution,
                    packed_counts,
                    packed_b_classes,
                    a_chunk_exp,
                    a_chunk_sig,
                    max_exp,
                    group_lane=group_lane,
                    b_class_lanes=chunk_b_class_lanes,
                    a_size=int(a_chunk_exp.numel()),
                    internal_width=internal_width,
                    class_chunk=class_chunk,
                )

            acc_base = _accumulator_base(acc_sig, internal_width)
            aligned_acc = _signed_shift_right_towards_zero(acc_base, max_exp - acc_exp_eff)
            aligned_acc = torch.where(acc_sign, -aligned_acc, aligned_acc)
            acc_sign, acc_exp, acc_sig = _normalize_total(
                aligned_acc + contribution,
                max_exp,
                internal_width,
                zero_exponent,
            )

    y = _gfloat_to_float32(acc_sign, acc_exp, acc_sig)
    y = y * x_scale * w_scale.reshape(1, -1).to(torch.float32)
    stats = HawkeyeFreivaldsStats(
        products_per_group=products_per_group,
        groups=groups,
        packed_group_lanes=packed_group_lanes,
        class_products_first_pass=0,
        class_products_replay_pass=replay_pass_products,
        total_checkable_products=replay_pass_products,
        max_activation_classes_per_group=max_a_classes,
        max_weight_classes_per_group=max_b_classes,
        class_chunk=class_chunk,
        packed_count_lanes=packed_count_lanes,
    )
    return y.to(torch.bfloat16), stats


def exact_hawkeye_fp8_sum(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
    raw_matmul: RawIntMatmul,
    *,
    products_per_group: int = 32,
    internal_width: int = 14,
    zero_exponent: int = -139,
    class_chunk: int = 32,
    packed_count_lanes: int = 1,
    packed_group_lanes: int = 1,
    fused_replay: bool = False,
    cache_weight_chunks: bool = False,
    count_dtype: torch.dtype = torch.int32,
) -> tuple[torch.Tensor, HawkeyeFreivaldsStats]:
    """Replay Hawkeye exactly from Freivalds-checkable class-count products."""
    if count_dtype not in {torch.int8, torch.int32}:
        raise ValueError(f"count_dtype must be torch.int8 or torch.int32, got {count_dtype}")
    if count_dtype == torch.int8 and (packed_count_lanes != 1 or fused_replay):
        raise ValueError("int8 count products are only supported for unpacked non-fused replay")
    if packed_count_lanes < 1 or packed_count_lanes > MAX_PACKED_COUNT_LANES:
        raise ValueError(
            f"packed_count_lanes must be in [1, {MAX_PACKED_COUNT_LANES}], "
            f"got {packed_count_lanes}"
        )
    if packed_group_lanes < 1 or packed_group_lanes > packed_count_lanes:
        raise ValueError("packed_group_lanes must fit inside packed_count_lanes")
    if packed_group_lanes > 1 and (count_dtype != torch.int32 or fused_replay or cache_weight_chunks):
        raise ValueError(
            "packed_group_lanes > 1 currently requires int32 count products without "
            "fused_replay or cache_weight_chunks"
        )
    a_exp, a_sig, a_nonzero = _decode_fp8_fields(x_fp8)
    b_exp, b_sig, b_nonzero = _decode_fp8_fields(w_fp8)
    a_class = _class_ids(a_exp, a_sig, a_nonzero)
    b_class = _class_ids(b_exp, b_sig, b_nonzero)

    if packed_group_lanes > 1:
        return _exact_hawkeye_fp8_sum_packed_groups(
            a_exp,
            a_nonzero,
            b_exp,
            b_nonzero,
            a_class,
            b_class,
            x_scale,
            w_scale,
            raw_matmul,
            products_per_group=products_per_group,
            internal_width=internal_width,
            zero_exponent=zero_exponent,
            class_chunk=class_chunk,
            packed_count_lanes=packed_count_lanes,
            packed_group_lanes=packed_group_lanes,
        )

    m_size, k_size = a_class.shape
    n_size = b_class.shape[0]
    device = x_fp8.device
    acc_sign = torch.zeros((m_size, n_size), device=device, dtype=torch.bool)
    acc_exp = torch.full((m_size, n_size), zero_exponent, device=device, dtype=torch.int64)
    acc_sig = torch.zeros((m_size, n_size), device=device, dtype=torch.int64)

    groups = 0
    first_pass_products = 0
    replay_pass_products = 0
    max_a_classes = 0
    max_b_classes = 0

    for k0 in range(0, k_size, products_per_group):
        groups += 1
        k1 = min(k0 + products_per_group, k_size)
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
            torch.full_like(acc_exp, zero_exponent),
        )
        present_exp = _group_present_exponent(
            a_exp_group,
            b_exp_group,
            a_nonzero_group,
            b_nonzero_group,
            zero_exponent,
        )
        a_chunks = _iter_class_chunks(a_classes, class_chunk)
        b_chunk_size = class_chunk if packed_count_lanes == 1 else class_chunk * packed_count_lanes
        b_chunks = _iter_class_chunks(b_classes, b_chunk_size)
        cached_b_chunks = None
        if cache_weight_chunks and packed_count_lanes == 1 and not fused_replay:
            cached_b_chunks = [
                (
                    b_chunk,
                    b_chunk // SIGNED_SIG_BUCKETS + 1,
                    b_chunk % SIGNED_SIG_BUCKETS - SIGNED_SIG_OFFSET,
                    _weight_chunk_flat(b_group, b_chunk, count_dtype=count_dtype),
                )
                for b_chunk in b_chunks
            ]

        max_exp = torch.maximum(acc_exp_eff, present_exp)
        contribution = torch.zeros((m_size, n_size), device=device, dtype=torch.int64)

        for a_chunk in a_chunks:
            a_chunk_exp = a_chunk // SIGNED_SIG_BUCKETS + 1
            a_chunk_sig = a_chunk % SIGNED_SIG_BUCKETS - SIGNED_SIG_OFFSET
            if cached_b_chunks is not None:
                for _b_chunk, b_chunk_exp, b_chunk_sig, b_flat in cached_b_chunks:
                    replay_pass_products += 1
                    counts_raw, a_size, b_size = _class_chunk_product_raw_with_b_flat(
                        a_group,
                        a_chunk,
                        b_flat,
                        int(b_chunk_exp.numel()),
                        raw_matmul,
                    )
                    counts = counts_raw.reshape(a_size, m_size, b_size, n_size).permute(
                        0,
                        2,
                        1,
                        3,
                    )
                    prod_exp = a_chunk_exp[:, None] + b_chunk_exp[None, :] - 14
                    signed_product = a_chunk_sig[:, None] * b_chunk_sig[None, :]
                    scaled = _scale_to_internal(
                        signed_product[:, :, None, None],
                        internal_width,
                    )
                    shifts = max_exp[None, None, :, :] - prod_exp[:, :, None, None]
                    aligned_value = _signed_shift_right_towards_zero(scaled, shifts)
                    contribution += (counts * aligned_value).sum(dim=(0, 1))
                continue

            for b_chunk in b_chunks:
                if packed_count_lanes == 1:
                    b_chunk_exp = b_chunk // SIGNED_SIG_BUCKETS + 1
                    b_chunk_sig = b_chunk % SIGNED_SIG_BUCKETS - SIGNED_SIG_OFFSET
                    replay_pass_products += 1
                    if fused_replay:
                        _class_chunk_product_add_contribution_fused(
                            contribution,
                            a_group,
                            b_group,
                            a_chunk,
                            b_chunk,
                            a_chunk_exp,
                            a_chunk_sig,
                            b_chunk_exp,
                            b_chunk_sig,
                            max_exp,
                            internal_width=internal_width,
                        )
                    else:
                        counts = _class_chunk_product(
                            a_group,
                            b_group,
                            a_chunk,
                            b_chunk,
                            raw_matmul,
                            count_dtype=count_dtype,
                        )
                        prod_exp = a_chunk_exp[:, None] + b_chunk_exp[None, :] - 14
                        signed_product = a_chunk_sig[:, None] * b_chunk_sig[None, :]
                        scaled = _scale_to_internal(
                            signed_product[:, :, None, None],
                            internal_width,
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
                    raw_matmul,
                    packed_count_lanes,
                )
                replay_pass_products += 1
                _add_packed_contribution_fused(
                    contribution,
                    packed_counts,
                    packed_b_classes,
                    a_chunk_exp,
                    a_chunk_sig,
                    max_exp,
                    a_size=int(a_chunk.numel()),
                    internal_width=internal_width,
                    class_chunk=class_chunk,
                )

        acc_base = _accumulator_base(acc_sig, internal_width)
        aligned_acc = _signed_shift_right_towards_zero(acc_base, max_exp - acc_exp_eff)
        aligned_acc = torch.where(acc_sign, -aligned_acc, aligned_acc)
        acc_sign, acc_exp, acc_sig = _normalize_total(
            aligned_acc + contribution,
            max_exp,
            internal_width,
            zero_exponent,
        )

    y = _gfloat_to_float32(acc_sign, acc_exp, acc_sig)
    y = y * x_scale * w_scale.reshape(1, -1).to(torch.float32)
    stats = HawkeyeFreivaldsStats(
        products_per_group=products_per_group,
        groups=groups,
        packed_group_lanes=packed_group_lanes,
        class_products_first_pass=first_pass_products,
        class_products_replay_pass=replay_pass_products,
        total_checkable_products=first_pass_products + replay_pass_products,
        max_activation_classes_per_group=max_a_classes,
        max_weight_classes_per_group=max_b_classes,
        class_chunk=class_chunk,
        packed_count_lanes=packed_count_lanes,
    )
    return y.to(torch.bfloat16), stats
