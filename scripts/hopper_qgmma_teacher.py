"""Hopper FP8 QGMMA teacher used by the Hawkeye probe.

The extension wraps a direct
`wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3` tile.  It is meant as
an experiment-only teacher: inputs are real `torch.float8_e4m3fn`, Python
advances K in fixed 32-wide tiles, the FP32 accumulator is passed back into the
next QGMMA call, and scaling/bf16 conversion happens after the QGMMA loop.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load


_EXTENSION: ModuleType | None = None


def _source_paths() -> list[str]:
    source_dir = Path(__file__).resolve().parent / "hopper_qgmma"
    return [
        str(source_dir / "fp8_e4m3_wgmma_ext.cpp"),
        str(source_dir / "fp8_e4m3_wgmma.cu"),
    ]


def require_hopper() -> tuple[int, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Hopper QGMMA teacher")
    capability = torch.cuda.get_device_capability(0)
    if capability[0] < 9:
        raise RuntimeError(
            "Hopper QGMMA FP8 requires SM90+; "
            f"found SM{capability[0]}{capability[1]} on {torch.cuda.get_device_name(0)}"
        )
    return capability


def load_hopper_qgmma_extension(verbose: bool = False) -> ModuleType:
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    require_hopper()
    old_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    try:
        _EXTENSION = load(
            name="int_model_approximation_hopper_qgmma_fp8",
            sources=_source_paths(),
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "-std=c++17", "--use_fast_math"],
            with_cuda=True,
            verbose=verbose,
        )
    finally:
        if old_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old_arch_list
    return _EXTENSION


def _flatten_scale(scale: torch.Tensor, expected: int, name: str) -> torch.Tensor:
    flat = scale.reshape(-1).to(dtype=torch.float32)
    if flat.numel() != expected:
        raise RuntimeError(f"{name} has {flat.numel()} values; expected {expected}")
    return flat.contiguous()


def qgmma_fp8_scaled_mm(
    x_fp8: torch.Tensor,
    x_scale: torch.Tensor,
    w_fp8: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    verbose_build: bool = False,
) -> torch.Tensor:
    """Return bf16 `(x_fp8 @ w_fp8.T) * x_scale * w_scale` from direct QGMMA."""
    if x_fp8.dtype != torch.float8_e4m3fn or w_fp8.dtype != torch.float8_e4m3fn:
        raise RuntimeError("qgmma_fp8_scaled_mm expects float8_e4m3fn operands")
    if x_fp8.device.type != "cuda" or w_fp8.device.type != "cuda":
        raise RuntimeError("qgmma_fp8_scaled_mm expects CUDA operands")
    if x_fp8.device != w_fp8.device:
        raise RuntimeError("x_fp8 and w_fp8 must be on the same CUDA device")
    if x_fp8.dim() != 2 or w_fp8.dim() != 2:
        raise RuntimeError("qgmma_fp8_scaled_mm expects x_fp8=[M,K], w_fp8=[N,K]")
    if x_fp8.shape[1] != w_fp8.shape[1]:
        raise RuntimeError(f"shape mismatch: {tuple(x_fp8.shape)} vs {tuple(w_fp8.shape)}")

    x_scale_flat = _flatten_scale(x_scale.to(x_fp8.device), x_fp8.shape[0], "x_scale")
    w_scale_flat = _flatten_scale(w_scale.to(w_fp8.device), w_fp8.shape[0], "w_scale")
    extension = load_hopper_qgmma_extension(verbose=verbose_build)
    x_fp8 = x_fp8.contiguous()
    w_fp8 = w_fp8.contiguous()

    m_size, k_size = x_fp8.shape
    n_size = w_fp8.shape[0]
    raw = torch.empty((m_size, n_size), device=x_fp8.device, dtype=torch.float32)

    for m0 in range(0, m_size, 64):
        m1 = min(m0 + 64, m_size)
        tile_m = m1 - m0
        for n0 in range(0, n_size, 128):
            n1 = min(n0 + 128, n_size)
            tile_n = n1 - n0
            acc = torch.zeros((64, 128), device=x_fp8.device, dtype=torch.float32)
            for k0 in range(0, k_size, 32):
                k1 = min(k0 + 32, k_size)
                tile_k = k1 - k0
                a_pad = torch.zeros((64, 32), device=x_fp8.device, dtype=torch.float8_e4m3fn)
                b_pad = torch.zeros((128, 32), device=x_fp8.device, dtype=torch.float8_e4m3fn)
                a_pad[:tile_m, :tile_k] = x_fp8[m0:m1, k0:k1]
                b_pad[:tile_n, :tile_k] = w_fp8[n0:n1, k0:k1]
                acc = extension.fp8_e4m3_wgmma_tile(a_pad, b_pad, acc)
            raw[m0:m1, n0:n1] = acc[:tile_m, :tile_n]

    scaled = raw * x_scale_flat[:, None] * w_scale_flat[None, :]
    return scaled.to(torch.bfloat16)
