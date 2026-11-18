"""
Tests for experiments/aggregate.py's pure aggregation functions (no Excel,
no Gurobi -- just plain dicts, so these run fast and test the statistics
directly).
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from aggregate import (  # noqa: E402
    shifted_geometric_mean, compute_per_instance, compute_per_method,
)


def _row(instance, method, seed, reason="optimal", feasible=True, proved_optimal=True,
         best_obj=10.0, bound=10.0, gap=0.0, time_sec=1.0, iterations=5,
         first_feasible_time=0.5, cut_mode="structural", probe_k=0):
    return dict(
        instance=instance, method=method, seed=seed, cut_mode=cut_mode, probe_k=probe_k,
        reason=reason, feasible=feasible, proved_optimal=proved_optimal,
        best_obj=best_obj, bound=bound, gap=gap, time_sec=time_sec, iterations=iterations,
        first_feasible_time=first_feasible_time,
    )


class TestShiftedGeometricMean:
    def test_empty_is_none(self):
        assert shifted_geometric_mean([]) is None

    def test_all_none_is_none(self):
        assert shifted_geometric_mean([None, None]) is None

    def test_single_value(self):
        assert shifted_geometric_mean([5.0]) == pytest.approx(5.0, abs=1e-9)

    def test_near_zero_values_dont_blow_up(self):
        # plain geometric mean of [1e-9, 1e-9, 100] would be dominated by
        # the tiny values in a way that misrepresents "typical" performance;
        # the shift keeps it sane.
        gm = shifted_geometric_mean([1e-9, 1e-9, 100.0])
        assert gm is not None
        assert 0 < gm < 100


class TestComputePerInstance:
    def test_feasible_and_optimal_rates(self):
        runs = [
            _row("i1", "Struct", 0, proved_optimal=True),
            _row("i1", "Struct", 1, proved_optimal=False, feasible=True, reason="cycle"),
        ]
        out = compute_per_instance(runs)
        assert len(out) == 1
        row = out[0]
        assert row["instance"] == "i1" and row["method"] == "Struct"
        assert row["n_valid_seeds"] == 2
        assert row["feasible_rate"] == 1.0
        assert row["optimal_rate"] == 0.5

    def test_invalid_rows_excluded(self):
        runs = [
            _row("i1", "Struct", 0),
            dict(_row("i1", "Struct", 1), reason="skipped_genint", feasible=False, best_obj=None),
        ]
        out = compute_per_instance(runs)
        assert out[0]["n_valid_seeds"] == 1

    def test_seed_varied_detects_divergence(self):
        same = [_row("i1", "Struct", s, best_obj=10.0, iterations=5) for s in (0, 1)]
        diff = [_row("i2", "Struct", 0, best_obj=10.0), _row("i2", "Struct", 1, best_obj=12.0)]
        out = compute_per_instance(same + diff)
        by_inst = {r["instance"]: r for r in out}
        assert by_inst["i1"]["seed_varied"] is False
        assert by_inst["i2"]["seed_varied"] is True

    def test_single_seed_never_varied(self):
        out = compute_per_instance([_row("i1", "Struct", 0)])
        assert out[0]["seed_varied"] is False


class TestComputePerMethod:
    def test_restricts_to_comparable_instances(self):
        runs = [
            _row("i1", "A", 0), _row("i1", "B", 0),
            _row("i2", "A", 0),  # B has no run on i2 -> i2 not comparable
        ]
        out = compute_per_method(runs)
        by_method = {r["method"]: r for r in out}
        assert by_method["A"]["n_comparable_instances"] == 1
        assert by_method["B"]["n_comparable_instances"] == 1

    def test_majority_vs_average_feasible_count(self):
        runs = [
            _row("i1", "A", 0, feasible=True), _row("i1", "A", 1, feasible=False, reason="cycle",
                                                      proved_optimal=False, best_obj=None, bound=None, gap=None),
            _row("i1", "B", 0, feasible=True), _row("i1", "B", 1, feasible=True),
        ]
        out = compute_per_method(runs)
        by_method = {r["method"]: r for r in out}
        # A: 1/2 seeds feasible -> majority threshold (>=0.5) counts it; avg = 0.5
        assert by_method["A"]["feasible_count_majority"] == 1
        assert by_method["A"]["feasible_count_avg"] == pytest.approx(0.5)
        assert by_method["B"]["feasible_count_majority"] == 1
        assert by_method["B"]["feasible_count_avg"] == pytest.approx(1.0)

    def test_seed_varied_fraction(self):
        runs = [
            _row("i1", "A", 0, best_obj=10.0), _row("i1", "A", 1, best_obj=10.0),  # not varied
            _row("i2", "A", 0, best_obj=10.0), _row("i2", "A", 1, best_obj=12.0),  # varied
        ]
        out = compute_per_method(runs)
        row = next(r for r in out if r["method"] == "A")
        assert row["seed_varied_fraction"] == pytest.approx(0.5)
