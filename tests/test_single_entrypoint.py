from __future__ import annotations

from pathlib import Path

import pytest
import torch

from int_model_approximation.__main__ import Int32Linear, _l2_stats


def test_int32_linear_rejects_cpu_weights():
    with pytest.raises(RuntimeError, match="CUDA"):
        Int32Linear(torch.randn(4, 4), None)


def test_l2_stats_reports_total_l2():
    ref = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    cand = torch.tensor([[1.0, 1.0], [1.0, 4.0]])
    stats = _l2_stats(ref, cand)
    assert stats["l2"] == pytest.approx(5.0 ** 0.5)
    assert stats["per_position_max"] == pytest.approx(2.0)


def test_only_one_package_entrypoint_file_remains():
    package_files = {
        p.name
        for p in (Path(__file__).parents[1] / "src" / "int_model_approximation").glob("*.py")
    }
    assert package_files == {"__init__.py", "__main__.py", "hawkeye.py", "metrics.py"}
