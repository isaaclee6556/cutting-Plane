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
    compute_paired_stats, _effective_gap, read_jsonl, append_jsonl, jsonl_path_for,
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

    def test_par1_uses_time_limit_for_unproved_runs(self):
        runs = [
            _row("i1", "A", 0, proved_optimal=True, time_sec=2.0),
            _row("i1", "A", 1, proved_optimal=False, reason="cycle", time_sec=50.0,
                 best_obj=None, bound=None, gap=None),
        ]
        out_no_limit = compute_per_method(runs)
        assert out_no_limit[0]["time_sec_par1"] is None

        out = compute_per_method(runs, time_limit=100.0)
        row = out[0]
        # PAR1 replaces the unproved run's time (50.0) with the time_limit
        # (100.0) before taking the SGM -- so it must differ from the plain
        # SGM (which uses the raw 2.0/50.0).
        assert row["time_sec_par1"] != pytest.approx(row["time_sec_sgm"])
        assert row["time_sec_par1"] == pytest.approx(shifted_geometric_mean([2.0, 100.0]))

    def test_sgm_proved_restricts_to_proved_runs(self):
        runs = [
            _row("i1", "A", 0, proved_optimal=True, time_sec=2.0),
            _row("i1", "A", 1, proved_optimal=False, reason="cycle", time_sec=50.0,
                 best_obj=None, bound=None, gap=None),
        ]
        out = compute_per_method(runs)
        row = out[0]
        assert row["time_sec_sgm_proved"] == pytest.approx(shifted_geometric_mean([2.0]))

    def test_gap_mean_unproved_only_averages_failed_runs(self):
        runs = [
            _row("i1", "A", 0, proved_optimal=True, gap=0.0),
            _row("i1", "A", 1, proved_optimal=False, reason="cycle", gap=0.3),
        ]
        out = compute_per_method(runs)
        row = out[0]
        assert row["gap_mean_unproved"] == pytest.approx(0.3)
        assert row["gap_mean"] == pytest.approx(0.15)


class TestEffectiveGap:
    def test_uses_real_gap_when_present(self):
        assert _effective_gap(_row("i1", "A", 0, gap=0.42)) == pytest.approx(0.42)

    def test_defaults_to_one_for_valid_run_without_incumbent(self):
        row = _row("i1", "A", 0, reason="cycle", proved_optimal=False, feasible=False,
                    best_obj=None, bound=None, gap=None)
        assert _effective_gap(row) == 1.0

    def test_none_for_proven_infeasible(self):
        row = _row("i1", "A", 0, reason="infeasible", feasible=False,
                    best_obj=None, bound=None, gap=None)
        assert _effective_gap(row) is None

    def test_none_for_unbounded_relaxation(self):
        row = _row("i1", "A", 0, reason="unbounded_relaxation", feasible=False,
                    proved_optimal=False, best_obj=None, bound=None, gap=None)
        assert _effective_gap(row) is None

    def test_none_for_invalid_row(self):
        row = _row("i1", "A", 0, reason="skipped_genint", feasible=False,
                    proved_optimal=False, best_obj=None, bound=None, gap=None)
        assert _effective_gap(row) is None


class TestSeedVariedExcludesTimeLimit:
    def test_time_limit_rows_excluded_from_comparison(self):
        # Both seeds timed out with slightly different iteration counts
        # (expected Gurobi-seed noise) but the same reason/best_obj/
        # proved_optimal -- must NOT count as seed_varied once time_limit
        # rows are excluded from the comparison per M13.1.
        runs = [
            _row("i1", "A", 0, reason="time_limit", proved_optimal=False,
                 best_obj=None, bound=5.0, gap=None, iterations=174),
            _row("i1", "A", 1, reason="time_limit", proved_optimal=False,
                 best_obj=None, bound=5.0, gap=None, iterations=181),
        ]
        out = compute_per_instance(runs)
        assert out[0]["seed_varied"] is False

    def test_best_obj_rounded_before_comparison(self):
        runs = [
            _row("i1", "A", 0, best_obj=10.0000001),
            _row("i1", "A", 1, best_obj=10.0000002),
        ]
        out = compute_per_instance(runs)
        assert out[0]["seed_varied"] is False

    def test_real_divergence_still_detected(self):
        runs = [
            _row("i1", "A", 0, reason="optimal", proved_optimal=True, best_obj=10.0),
            _row("i1", "A", 1, reason="cycle", proved_optimal=False, best_obj=None,
                 bound=None, gap=None),
        ]
        out = compute_per_instance(runs)
        assert out[0]["seed_varied"] is True


class TestComputePairedStats:
    def test_identical_methods_are_all_ties_with_p_one(self):
        runs = [
            _row("i1", "Struct", 0, time_sec=3.0), _row("i1", "Struct", 1, time_sec=3.0),
            _row("i1", "NoGood", 0, time_sec=3.0), _row("i1", "NoGood", 1, time_sec=3.0),
        ]
        out = compute_paired_stats(runs)
        row = next(r for r in out if (r["method_a"], r["method_b"]) == ("Struct", "NoGood"))
        assert row["wins"] == 0 and row["losses"] == 0 and row["ties"] == 1
        assert row["wilcoxon_p"] == pytest.approx(1.0)

    def test_higher_optimal_rate_wins(self):
        runs = [
            _row("i1", "Struct", 0, proved_optimal=True),
            _row("i1", "NoGood", 0, proved_optimal=False, reason="cycle",
                 best_obj=None, bound=None, gap=None),
        ]
        out = compute_paired_stats(runs)
        row = next(r for r in out if (r["method_a"], r["method_b"]) == ("Struct", "NoGood"))
        assert row["wins"] == 1 and row["losses"] == 0 and row["ties"] == 0

    def test_tie_on_optimal_rate_breaks_on_median_time(self):
        runs = [
            _row("i1", "Struct", 0, proved_optimal=True, time_sec=1.0),
            _row("i1", "NoGood", 0, proved_optimal=True, time_sec=5.0),
        ]
        out = compute_paired_stats(runs)
        row = next(r for r in out if (r["method_a"], r["method_b"]) == ("Struct", "NoGood"))
        assert row["wins"] == 1 and row["losses"] == 0  # Struct faster -> wins

    def test_unpaired_methods_report_none(self):
        # Struct+P2 vs NoGood+P2 never run in this batch -> no instances to
        # compare, no crash.
        runs = [_row("i1", "Struct", 0), _row("i1", "NoGood", 0)]
        out = compute_paired_stats(runs)
        row = next(r for r in out if (r["method_a"], r["method_b"]) == ("Struct+P2", "NoGood+P2"))
        assert row["n_instances_compared"] == 0
        assert row["wilcoxon_p"] is None


class TestJsonlStorage:
    def test_jsonl_path_for_appends_suffix(self):
        assert jsonl_path_for(Path("/tmp/foo.xlsx")) == Path("/tmp/foo.xlsx.jsonl")

    def test_append_and_read_roundtrip(self, tmp_path):
        path = tmp_path / "out.xlsx.jsonl"
        append_jsonl(path, {"instance": "i1", "method": "A", "seed": 0})
        append_jsonl(path, {"instance": "i1", "method": "A", "seed": 1})
        rows = read_jsonl(path)
        assert len(rows) == 2
        assert rows[0]["seed"] == 0 and rows[1]["seed"] == 1

    def test_missing_file_reads_as_empty(self, tmp_path):
        assert read_jsonl(tmp_path / "nope.jsonl") == []

    def test_corrupt_last_line_is_skipped_not_fatal(self, tmp_path):
        path = tmp_path / "out.xlsx.jsonl"
        append_jsonl(path, {"instance": "i1", "method": "A", "seed": 0})
        with open(path, "a") as f:
            f.write('{"instance": "i1", "method": "A", "seed": 1,\n')  # truncated/corrupt
        rows = read_jsonl(path)
        assert len(rows) == 1
