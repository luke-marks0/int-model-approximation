from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

import int_model_approximation.__main__ as entry
from int_model_approximation.hawkeye_freivalds import (
    PACKED_COUNT_BASE,
    PACKED_COUNT_BASE_LOG2,
    _class_chunk_product,
    exact_hawkeye_fp8_sum,
    _gfloat_to_float32,
    _packed_class_chunk_product,
    _packed_group_class_chunk_product,
)


def _hopper_fp8_available() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability(0)
    return major >= 9


def test_gfloat_zero_exponent_with_zero_significand_is_zero() -> None:
    y = _gfloat_to_float32(
        torch.tensor([False]),
        torch.tensor([-139], dtype=torch.int64),
        torch.tensor([0], dtype=torch.int64),
    )

    assert y.item() == 0.0


def test_gfloat_underflow_does_not_wrap_to_large_finite_value() -> None:
    y = _gfloat_to_float32(
        torch.tensor([False]),
        torch.tensor([-139], dtype=torch.int64),
        torch.tensor([1], dtype=torch.int64),
    )

    assert torch.isfinite(y).all()
    assert y.abs().item() < 1e-35


def test_packed_class_chunk_product_recovers_unpacked_counts() -> None:
    a_class_group = torch.tensor(
        [
            [1, 2, 1, 3],
            [2, 1, 3, 1],
        ],
        dtype=torch.int64,
    )
    b_class_group = torch.tensor(
        [
            [4, 5, 4, 6],
            [5, 4, 6, 4],
            [7, 4, 5, 6],
        ],
        dtype=torch.int64,
    )
    a_classes = torch.tensor([1, 2, 3], dtype=torch.int64)
    b_classes = torch.tensor([4, 5, 6, 7], dtype=torch.int64)

    def raw_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a.to(torch.int64) @ b.to(torch.int64)

    unpacked = _class_chunk_product(
        a_class_group,
        b_class_group,
        a_classes,
        b_classes,
        raw_matmul,
    )
    packed, packed_b_classes = _packed_class_chunk_product(
        a_class_group,
        b_class_group,
        a_classes,
        b_classes,
        raw_matmul,
        packed_count_lanes=3,
    )
    decoded = torch.zeros_like(unpacked)
    for b_idx in range(b_classes.numel()):
        packed_group = b_idx // 3
        lane = b_idx % 3
        assert packed_b_classes[packed_group, lane] == b_classes[b_idx]
        decoded[:, b_idx, :, :] = torch.bitwise_and(
            torch.bitwise_right_shift(packed[:, packed_group, :, :], lane * PACKED_COUNT_BASE_LOG2),
            PACKED_COUNT_BASE - 1,
        )

    assert torch.equal(decoded, unpacked)


def test_packed_group_class_chunk_product_recovers_per_group_counts() -> None:
    a_class_pack = torch.tensor(
        [
            [1, 2, 1, 3, 2, 1, 3, 1],
            [2, 1, 3, 1, 1, 3, 2, 1],
        ],
        dtype=torch.int64,
    )
    b_class_pack = torch.tensor(
        [
            [4, 5, 4, 6, 4, 5, 7, 4],
            [5, 4, 6, 4, 6, 4, 5, 7],
            [7, 4, 5, 6, 5, 7, 4, 6],
        ],
        dtype=torch.int64,
    )
    a_classes = torch.tensor([1, 2, 3], dtype=torch.int64)
    b_classes = torch.tensor([4, 5, 6, 7], dtype=torch.int64)

    def raw_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a.to(torch.int64) @ b.to(torch.int64)

    packed, packed_b_classes, b_class_lanes = _packed_group_class_chunk_product(
        a_class_pack,
        b_class_pack,
        a_classes,
        b_classes,
        raw_matmul,
        products_per_group=4,
        packed_group_lanes=2,
        packed_count_lanes=6,
    )

    for group_lane in range(2):
        start = group_lane * 4
        stop = start + 4
        unpacked = _class_chunk_product(
            a_class_pack[:, start:stop],
            b_class_pack[:, start:stop],
            a_classes,
            b_classes,
            raw_matmul,
        )
        decoded = torch.zeros_like(unpacked)
        for b_idx in range(b_classes.numel()):
            packed_group = b_idx // b_class_lanes
            b_lane = b_idx % b_class_lanes
            assert packed_b_classes[packed_group, b_lane] == b_classes[b_idx]
            lane = group_lane * b_class_lanes + b_lane
            decoded[:, b_idx, :, :] = torch.bitwise_and(
                torch.bitwise_right_shift(packed[:, packed_group, :, :], lane * PACKED_COUNT_BASE_LOG2),
                PACKED_COUNT_BASE - 1,
            )

        assert torch.equal(decoded, unpacked)


@pytest.mark.skipif(
    not _hopper_fp8_available(),
    reason="requires SM90+ for direct Hopper FP8 QGMMA",
)
def test_exact_hawkeye_class_counts_match_qgmma_tile() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.hopper_qgmma_teacher import load_hopper_qgmma_extension

    generator = torch.Generator(device="cuda").manual_seed(3)
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
    exact, stats = exact_hawkeye_fp8_sum(
        x,
        torch.ones((64, 1), device="cuda", dtype=torch.float32),
        w,
        torch.ones((128,), device="cuda", dtype=torch.float32),
        entry._int32_raw_matmul,
        products_per_group=32,
        internal_width=14,
        class_chunk=32,
        packed_count_lanes=6,
    )

    assert stats.total_checkable_products > 0
    assert torch.equal(exact, qgmma)


@pytest.mark.skipif(
    not _hopper_fp8_available(),
    reason="requires SM90+ for direct Hopper FP8 QGMMA",
)
def test_exact_hawkeye_packed_groups_match_two_qgmma_tiles() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.hopper_qgmma_teacher import qgmma_fp8_scaled_mm

    generator = torch.Generator(device="cuda").manual_seed(4)
    x = torch.randn((64, 64), generator=generator, device="cuda", dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    w = torch.randn((128, 64), generator=generator, device="cuda", dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    qgmma = qgmma_fp8_scaled_mm(
        x,
        torch.ones((64, 1), device="cuda", dtype=torch.float32),
        w,
        torch.ones((128,), device="cuda", dtype=torch.float32),
    )
    exact, stats = exact_hawkeye_fp8_sum(
        x,
        torch.ones((64, 1), device="cuda", dtype=torch.float32),
        w,
        torch.ones((128,), device="cuda", dtype=torch.float32),
        entry._int32_raw_matmul,
        products_per_group=32,
        internal_width=14,
        class_chunk=512,
        packed_count_lanes=6,
        packed_group_lanes=2,
    )

    assert stats.groups == 2
    assert stats.total_checkable_products == 1
    assert torch.equal(exact, qgmma)
