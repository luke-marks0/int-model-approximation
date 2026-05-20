from __future__ import annotations

import torch

from int_model_approximation.hawkeye_freivalds import (
    PACKED_COUNT_BASE,
    PACKED_COUNT_BASE_LOG2,
    _class_chunk_product,
    _gfloat_to_float32,
    _packed_class_chunk_product,
)


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
