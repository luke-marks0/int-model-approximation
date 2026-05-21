"""Direct Hawkeye FP8 accumulator replay.

This module models Hopper FP8 QGMMA accumulation using integer transition logic.
It is hardware-faithful for the direct QGMMA teacher, but it is not a single
matrix product and therefore is not cheaply Freivalds-checkable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True)
class HawkeyeStats:
    products_per_group: int
    groups: int
    direct_product_terms: int
    total_checkable_products: int
    block_m: int
    block_n: int


@triton.jit
def _decode_fp8_e4m3(raw):
    raw = raw.to(tl.int32)
    sign = (raw >> 7) & 1
    exp_bits = (raw >> 3) & 15
    mant = raw & 7
    sig_abs = tl.where(exp_bits != 0, mant | 8, mant)
    exp_eff = tl.where(exp_bits != 0, exp_bits, 1)
    signed_sig = tl.where(sign != 0, -sig_abs, sig_abs)
    nonzero = sig_abs != 0
    return exp_eff, signed_sig, nonzero


@triton.jit
def _scale_to_internal(signed_product, internal_width: tl.constexpr):
    if internal_width >= 7:
        return signed_product << (internal_width - 7)
    scaled_mag = tl.abs(signed_product) >> (7 - internal_width)
    return tl.where(signed_product < 0, -scaled_mag, scaled_mag)


@triton.jit
def _signed_shift_right_towards_zero(x, shift):
    shift = tl.minimum(tl.maximum(shift, 0), 62)
    mag = tl.abs(x) >> shift
    return tl.where(x < 0, -mag, mag)


@triton.jit
def _accumulator_base(acc_sig, internal_width: tl.constexpr):
    if internal_width < 24:
        return acc_sig >> (24 - internal_width)
    return acc_sig << (internal_width - 24)


@triton.jit
def _bit_width(magnitude):
    width = tl.zeros_like(magnitude)
    for bit in range(0, 31):
        width = tl.where(magnitude >= (1 << bit), bit + 1, width)
    return width


@triton.jit
def _normalize_total(total, max_exp, internal_width: tl.constexpr, zero_exponent: tl.constexpr):
    out_sign = total < 0
    magnitude = tl.abs(total)
    nonzero = magnitude != 0
    width = _bit_width(magnitude)

    out_exp = max_exp + width - internal_width
    right = tl.maximum(width - internal_width, 0)
    left = tl.maximum(internal_width - width, 0)
    out_sig = tl.where(
        width > internal_width,
        magnitude >> right,
        magnitude << left,
    )

    subnormal_shift = tl.maximum(-126 - out_exp, 0)
    out_sig = tl.where(
        out_exp < -126,
        out_sig >> subnormal_shift,
        out_sig,
    )
    out_exp = tl.maximum(out_exp, -126)

    if internal_width < 24:
        out_sig = out_sig << (24 - internal_width)
    else:
        out_sig = out_sig >> (internal_width - 24)

    keep = nonzero & (out_sig != 0)
    out_sig = tl.where(keep, out_sig, 0)
    out_exp = tl.where(keep, out_exp, zero_exponent)
    out_sign_i32 = tl.where(keep & out_sign, 1, 0)
    return out_sign_i32, out_exp, out_sig


@triton.jit
def _gfloat_to_float32(sign_i32, exponent, significand):
    exponent_bits = exponent + 127
    leading_bit_missing = (significand & 0x800000) == 0
    exponent_bits = tl.where(
        leading_bit_missing & (significand != 0),
        exponent_bits - 1,
        exponent_bits,
    )
    exponent_bits = tl.minimum(tl.maximum(exponent_bits, 0), 254)
    mantissa = significand & 0x7FFFFF
    bits = (
        (sign_i32.to(tl.uint32) << 31)
        | (exponent_bits.to(tl.uint32) << 23)
        | mantissa.to(tl.uint32)
    )
    bits = tl.where(significand != 0, bits, tl.zeros_like(bits))
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _hawkeye_fp8_sum_kernel(
    x_raw_ptr,
    x_scale_ptr,
    w_raw_ptr,
    w_scale_ptr,
    out_ptr,
    m_size: tl.constexpr,
    n_size: tl.constexpr,
    k_size: tl.constexpr,
    products_per_group: tl.constexpr,
    internal_width: tl.constexpr,
    zero_exponent: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * block_m + tl.arange(0, block_m)
    offs_n = pid_n * block_n + tl.arange(0, block_n)
    mask = (offs_m[:, None] < m_size) & (offs_n[None, :] < n_size)

    acc_sign = tl.zeros((block_m, block_n), dtype=tl.int32)
    acc_exp = tl.full((block_m, block_n), zero_exponent, dtype=tl.int32)
    acc_sig = tl.zeros((block_m, block_n), dtype=tl.int32)

    for k0 in range(0, k_size, products_per_group):
        acc_exp_eff = tl.where(acc_sig != 0, acc_exp, zero_exponent)
        max_exp = acc_exp_eff

        for kk in range(0, products_per_group):
            k = k0 + kk
            k_valid = k < k_size
            a_raw = tl.load(
                x_raw_ptr + offs_m * k_size + k,
                mask=(offs_m < m_size) & k_valid,
                other=0,
            )
            b_raw = tl.load(
                w_raw_ptr + offs_n * k_size + k,
                mask=(offs_n < n_size) & k_valid,
                other=0,
            )
            a_exp, _a_sig, a_nonzero = _decode_fp8_e4m3(a_raw)
            b_exp, _b_sig, b_nonzero = _decode_fp8_e4m3(b_raw)
            present = a_nonzero[:, None] & b_nonzero[None, :] & k_valid
            prod_exp = a_exp[:, None] + b_exp[None, :] - 14
            max_exp = tl.maximum(
                max_exp,
                tl.where(present, prod_exp, zero_exponent),
            )

        contribution = tl.zeros((block_m, block_n), dtype=tl.int32)
        for kk in range(0, products_per_group):
            k = k0 + kk
            k_valid = k < k_size
            a_raw = tl.load(
                x_raw_ptr + offs_m * k_size + k,
                mask=(offs_m < m_size) & k_valid,
                other=0,
            )
            b_raw = tl.load(
                w_raw_ptr + offs_n * k_size + k,
                mask=(offs_n < n_size) & k_valid,
                other=0,
            )
            a_exp, a_sig, a_nonzero = _decode_fp8_e4m3(a_raw)
            b_exp, b_sig, b_nonzero = _decode_fp8_e4m3(b_raw)
            present = a_nonzero[:, None] & b_nonzero[None, :] & k_valid
            prod_exp = a_exp[:, None] + b_exp[None, :] - 14
            signed_product = a_sig[:, None] * b_sig[None, :]
            scaled = _scale_to_internal(signed_product, internal_width)
            aligned = _signed_shift_right_towards_zero(scaled, max_exp - prod_exp)
            contribution += tl.where(present, aligned, 0)

        acc_base = _accumulator_base(acc_sig, internal_width)
        aligned_acc = acc_base >> tl.minimum(tl.maximum(max_exp - acc_exp_eff, 0), 62)
        aligned_acc = tl.where(acc_sign != 0, -aligned_acc, aligned_acc)
        acc_sign, acc_exp, acc_sig = _normalize_total(
            aligned_acc + contribution,
            max_exp,
            internal_width,
            zero_exponent,
        )

    y = _gfloat_to_float32(acc_sign, acc_exp, acc_sig)
    x_scale = tl.load(x_scale_ptr + offs_m, mask=offs_m < m_size, other=0.0).to(tl.float32)
    w_scale = tl.load(w_scale_ptr + offs_n, mask=offs_n < n_size, other=0.0).to(tl.float32)
    y = y * x_scale[:, None] * w_scale[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * n_size + offs_n[None, :],
        y,
        mask=mask,
    )


def hawkeye_fp8_sum(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    products_per_group: int = 32,
    internal_width: int = 14,
    zero_exponent: int = -139,
    block_m: int = 4,
    block_n: int = 32,
) -> tuple[torch.Tensor, HawkeyeStats]:
    """Replay the Hawkeye FP8 accumulator with direct integer transition logic."""
    if x_fp8.device.type != "cuda" or w_fp8.device.type != "cuda":
        raise RuntimeError("Hawkeye replay is CUDA-only")
    if x_fp8.dtype != torch.float8_e4m3fn or w_fp8.dtype != torch.float8_e4m3fn:
        raise RuntimeError("Hawkeye replay requires float8_e4m3fn operands")
    if x_fp8.ndim != 2 or w_fp8.ndim != 2:
        raise RuntimeError(f"Hawkeye replay needs 2D operands, got {x_fp8.shape} and {w_fp8.shape}")
    m_size, k_size = x_fp8.shape
    n_size, k2 = w_fp8.shape
    if k_size != k2:
        raise RuntimeError(f"Hawkeye shape mismatch: {x_fp8.shape} vs {w_fp8.shape}")
    if products_per_group <= 0:
        raise ValueError("products_per_group must be positive")

    x_raw = x_fp8.contiguous().view(torch.uint8)
    w_raw = w_fp8.contiguous().view(torch.uint8)
    out = torch.empty((m_size, n_size), device=x_fp8.device, dtype=torch.bfloat16)
    _hawkeye_fp8_sum_kernel[(triton.cdiv(m_size, block_m), triton.cdiv(n_size, block_n))](
        x_raw,
        x_scale.reshape(-1).contiguous(),
        w_raw,
        w_scale.reshape(-1).contiguous(),
        out,
        m_size,
        n_size,
        k_size,
        products_per_group,
        internal_width,
        zero_exponent,
        block_m=block_m,
        block_n=block_n,
        num_warps=4,
    )
    stats = HawkeyeStats(
        products_per_group=products_per_group,
        groups=triton.cdiv(k_size, products_per_group),
        direct_product_terms=m_size * n_size * k_size,
        total_checkable_products=0,
        block_m=block_m,
        block_n=block_n,
    )
    return out, stats
