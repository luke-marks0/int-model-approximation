from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

from int_model_approximation.hawkeye import hawkeye_fp8_sum


def _hopper_fp8_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability(0)
    return major >= 9


def test_hawkeye_rejects_cpu_operands() -> None:
    x = torch.zeros((1, 32), dtype=torch.float8_e4m3fn)
    w = torch.zeros((1, 32), dtype=torch.float8_e4m3fn)

    with pytest.raises(RuntimeError, match="CUDA-only"):
        hawkeye_fp8_sum(x, torch.ones((1, 1)), w, torch.ones(1))


@pytest.mark.skipif(
    not _hopper_fp8_available(),
    reason="requires SM90+ for direct Hopper FP8 QGMMA",
)
def test_hawkeye_matches_qgmma_tile() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.hopper_qgmma_teacher import load_hopper_qgmma_extension

    generator = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn((64, 32), generator=generator, device="cuda", dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    w = torch.randn((128, 32), generator=generator, device="cuda", dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    extension = load_hopper_qgmma_extension()
    qgmma = extension.fp8_e4m3_wgmma_tile(
        x.contiguous(),
        w.contiguous(),
        torch.zeros((64, 128), device="cuda", dtype=torch.float32),
    ).to(torch.bfloat16)
    hawkeye, stats = hawkeye_fp8_sum(
        x,
        torch.ones((64, 1), device="cuda", dtype=torch.float32),
        w,
        torch.ones((128,), device="cuda", dtype=torch.float32),
        products_per_group=32,
        internal_width=14,
        block_m=4,
        block_n=32,
    )

    assert stats.total_checkable_products == 0
    assert torch.equal(hawkeye, qgmma)
