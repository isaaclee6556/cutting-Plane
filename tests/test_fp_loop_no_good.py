"""
Tests for feasibility pump with the no-good-cut-only ablation baseline
(use_no_good_cuts=True).

Unlike use_structural_cuts (which detects all 7 structures via
StructureIndex), use_no_good_cuts adds exactly one VertexCut for the
visited point z each non-feasible, non-cycling iteration.
"""
import pytest
import gurobipy as gp

from structural_fp.fp_loop import feasibility_pump
from structural_fp.io import (
    make_simple_binary_milp,
    make_covering_milp,
    make_infeasible_lp_milp,
    make_fractional_cycle_milp,
)


# ---------------------------------------------------------------------------
# Correctness: no-good cuts must not break finding feasible solutions
# ---------------------------------------------------------------------------

class TestCorrectnessWithNoGoodCuts:
    def test_simple_binary_feasible(self):
        res = feasibility_pump(make_simple_binary_milp(), use_no_good_cuts=True)
        assert res.feasible

    def test_simple_binary_reason_feasible(self):
        res = feasibility_pump(make_simple_binary_milp(), use_no_good_cuts=True)
        assert res.reason == "feasible"

    def test_simple_binary_solution_binary(self):
        res = feasibility_pump(make_simple_binary_milp(), use_no_good_cuts=True)
        assert all(v in (0, 1) for v in res.solution.values())

    def test_covering_6var_feasible(self):
        res = feasibility_pump(make_covering_milp(n=6), use_no_good_cuts=True)
        assert res.feasible

    def test_lp_infeasible_unaffected_by_no_good_cuts(self):
        res = feasibility_pump(make_infeasible_lp_milp(), use_no_good_cuts=True)
        assert not res.feasible
        assert res.reason == "lp_infeasible"


# ---------------------------------------------------------------------------
# cuts_added tracking: exactly one VertexCut per non-feasible iteration
# ---------------------------------------------------------------------------

class TestCutsAddedNoGood:
    def test_no_cuts_without_flag(self):
        res = feasibility_pump(make_simple_binary_milp(), use_no_good_cuts=False)
        assert res.cuts_added == 0

    def test_cuts_added_is_int(self):
        res = feasibility_pump(make_simple_binary_milp(), use_no_good_cuts=True)
        assert isinstance(res.cuts_added, int)

    def test_cycle_instance_adds_one_cut_per_visited_point(self):
        # Each unique visited (non-feasible, non-cycling) z adds exactly one
        # no-good cut, so cuts_added must equal the number of history entries.
        res = feasibility_pump(
            make_fractional_cycle_milp(),
            max_iter=20,
            use_no_good_cuts=True,
        )
        assert res.cuts_added == len(res.history)

    def test_cycle_instance_avoids_cycle(self):
        # The no-good cut for the repeated rounded point makes the LP
        # infeasible rather than cycling indefinitely.
        res = feasibility_pump(
            make_fractional_cycle_milp(),
            max_iter=20,
            use_no_good_cuts=True,
        )
        assert res.reason != "cycle"


# ---------------------------------------------------------------------------
# Mutual exclusivity with use_structural_cuts
# ---------------------------------------------------------------------------

class TestMutualExclusivity:
    def test_both_flags_raises(self):
        with pytest.raises(ValueError):
            feasibility_pump(
                make_simple_binary_milp(),
                use_structural_cuts=True,
                use_no_good_cuts=True,
            )


# ---------------------------------------------------------------------------
# cuts_by_type breakdown
# ---------------------------------------------------------------------------

class TestCutsByType:
    def test_no_good_only_vertex_cuts(self):
        res = feasibility_pump(
            make_fractional_cycle_milp(), max_iter=20, use_no_good_cuts=True
        )
        assert set(res.cuts_by_type) <= {"VertexCut"}
        assert sum(res.cuts_by_type.values()) == res.cuts_added

    def test_structural_breakdown_sums_to_cuts_added(self):
        res = feasibility_pump(
            make_covering_milp(n=10), max_iter=50, use_structural_cuts=True
        )
        assert sum(res.cuts_by_type.values()) == res.cuts_added

    def test_plain_has_empty_breakdown(self):
        res = feasibility_pump(make_covering_milp(n=6))
        assert res.cuts_by_type == {}
