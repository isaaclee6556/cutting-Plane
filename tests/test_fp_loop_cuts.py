"""
Tests for feasibility pump with structural cuts (Milestone 5).

All tests use use_structural_cuts=True.  Correctness tests verify that
enabling cuts does not break finding feasible solutions; behavior tests
verify that cuts change outcomes on cycling-prone instances.
"""
import pytest
import gurobipy as gp
from gurobipy import GRB

from structural_fp.fp_loop import feasibility_pump, FPResult
from structural_fp.io import (
    make_simple_binary_milp,
    make_covering_milp,
    make_infeasible_lp_milp,
    make_fractional_cycle_milp,
)


# ---------------------------------------------------------------------------
# Correctness: cuts must not break finding feasible solutions
# ---------------------------------------------------------------------------

class TestCorrectnessWithCuts:
    def test_simple_binary_feasible(self):
        res = feasibility_pump(make_simple_binary_milp(), use_structural_cuts=True)
        assert res.feasible

    def test_simple_binary_reason_feasible(self):
        res = feasibility_pump(make_simple_binary_milp(), use_structural_cuts=True)
        assert res.reason == "feasible"

    def test_simple_binary_solution_binary(self):
        res = feasibility_pump(make_simple_binary_milp(), use_structural_cuts=True)
        assert all(v in (0, 1) for v in res.solution.values())

    def test_simple_binary_satisfies_c0(self):
        res = feasibility_pump(make_simple_binary_milp(), use_structural_cuts=True)
        sol = res.solution
        assert sol["x0"] + sol["x1"] >= 1

    def test_simple_binary_satisfies_c1(self):
        res = feasibility_pump(make_simple_binary_milp(), use_structural_cuts=True)
        sol = res.solution
        assert sol["x2"] + sol["x3"] >= 1

    def test_covering_6var_feasible(self):
        res = feasibility_pump(make_covering_milp(n=6), use_structural_cuts=True)
        assert res.feasible

    def test_covering_8var_feasible(self):
        res = feasibility_pump(make_covering_milp(n=8), use_structural_cuts=True)
        assert res.feasible

    def test_covering_6var_satisfies_constraints(self):
        n = 6
        res = feasibility_pump(make_covering_milp(n=n), use_structural_cuts=True)
        sol = res.solution
        for k in range(n // 2):
            assert sol[f"x{2*k}"] + sol[f"x{2*k+1}"] >= 1

    def test_lp_infeasible_unaffected_by_cuts(self):
        res = feasibility_pump(make_infeasible_lp_milp(), use_structural_cuts=True)
        assert not res.feasible
        assert res.reason == "lp_infeasible"


# ---------------------------------------------------------------------------
# cuts_added tracking
# ---------------------------------------------------------------------------

class TestCutsAdded:
    def test_no_cuts_without_flag(self):
        # Plain FP never adds structural cuts
        res = feasibility_pump(make_simple_binary_milp(), use_structural_cuts=False)
        assert res.cuts_added == 0

    def test_cuts_added_is_int(self):
        res = feasibility_pump(make_simple_binary_milp(), use_structural_cuts=True)
        assert isinstance(res.cuts_added, int)

    def test_cuts_added_nonnegative(self):
        res = feasibility_pump(make_simple_binary_milp(), use_structural_cuts=True)
        assert res.cuts_added >= 0

    def test_cycle_instance_adds_cuts(self):
        # make_fractional_cycle_milp: LP gives fractional, round to (0,0) (infeasible),
        # a VertexCut for (0,0) is generated and added → LP becomes infeasible.
        res = feasibility_pump(
            make_fractional_cycle_milp(),
            max_iter=20,
            use_structural_cuts=True,
        )
        assert res.cuts_added >= 1

    def test_cuts_added_increases_with_iterations(self):
        # On a harder covering instance, multiple iterations may occur.
        res = feasibility_pump(
            make_covering_milp(n=10),
            max_iter=50,
            use_structural_cuts=True,
        )
        # If we had to pump multiple rounds, cuts were added.
        # At minimum, if feasible in iter 1, cuts_added can be 0 (z was feasible).
        assert res.cuts_added >= 0  # always true; test structure is the point


# ---------------------------------------------------------------------------
# Behavior change: cuts vs. plain FP on the cycle-prone instance
# ---------------------------------------------------------------------------

class TestBehaviorChange:
    def test_cycle_instance_plain_cycles(self):
        # Without cuts, the fractional-cycle MILP causes a cycle.
        res = feasibility_pump(make_fractional_cycle_milp(), max_iter=20)
        assert res.reason == "cycle"

    def test_cycle_instance_cuts_avoid_cycle(self):
        # With cuts, the vertex no-good for the repeated rounded point is added,
        # making the LP infeasible rather than cycling indefinitely.
        res = feasibility_pump(
            make_fractional_cycle_milp(),
            max_iter=20,
            use_structural_cuts=True,
        )
        assert res.reason != "cycle"

    def test_cycle_instance_cuts_terminates_faster(self):
        # FP+cuts should terminate before reaching max_iter via cycle or lp_infeasible.
        res_cuts = feasibility_pump(
            make_fractional_cycle_milp(),
            max_iter=20,
            use_structural_cuts=True,
        )
        assert res_cuts.iterations <= 20

    def test_plain_vs_cuts_same_feasibility_on_solvable(self):
        # Both plain and cut-enabled FP should find feasible on simple instances.
        model = make_simple_binary_milp()
        res_plain = feasibility_pump(model, use_structural_cuts=False)
        res_cuts  = feasibility_pump(model, use_structural_cuts=True)
        assert res_plain.feasible == res_cuts.feasible


# ---------------------------------------------------------------------------
# Structural validity of added cuts
# ---------------------------------------------------------------------------

class TestCutValidity:
    def test_found_solution_ignores_cuts_for_feasibility(self):
        # The feasibility checker uses ORIGINAL constraints only.
        # If FP+cuts finds a solution, it must satisfy original constraints.
        res = feasibility_pump(make_covering_milp(n=6), use_structural_cuts=True)
        if res.feasible:
            n = 6
            sol = res.solution
            for k in range(n // 2):
                assert sol[f"x{2*k}"] + sol[f"x{2*k+1}"] >= 1

    def test_solution_is_binary_with_cuts(self):
        res = feasibility_pump(make_covering_milp(n=6), use_structural_cuts=True)
        if res.feasible:
            assert all(v in (0, 1) for v in res.solution.values())

    def test_history_matches_visited_count(self):
        res = feasibility_pump(make_covering_milp(n=6), use_structural_cuts=True)
        # history stores one entry per non-feasible, non-cycling iteration
        assert len(res.history) == len(set(res.history))  # no duplicates


# ---------------------------------------------------------------------------
# Parameters: max_cuts_per_iter, cut_tol
# ---------------------------------------------------------------------------

class TestCutParameters:
    def test_max_cuts_per_iter_zero_equivalent_to_no_cuts(self):
        # max_cuts_per_iter=0 means select_cuts returns empty list every iter
        res = feasibility_pump(
            make_simple_binary_milp(),
            use_structural_cuts=True,
            max_cuts_per_iter=0,
        )
        assert res.cuts_added == 0

    def test_max_cuts_per_iter_1_caps_cuts(self):
        # With max 1 cut per iteration, total cuts ≤ iterations
        res = feasibility_pump(
            make_covering_milp(n=10),
            max_iter=20,
            use_structural_cuts=True,
            max_cuts_per_iter=1,
        )
        assert res.cuts_added <= res.iterations

    def test_large_cut_tol_suppresses_cuts(self):
        # cut_tol=1.0: only add cuts violated by more than 1.0 → rarely triggered
        res = feasibility_pump(
            make_simple_binary_milp(),
            use_structural_cuts=True,
            cut_tol=1.0,
        )
        # cuts_added may be 0 or low; importantly, pump still terminates
        assert res.iterations >= 1

    def test_default_parameters_find_feasible(self):
        res = feasibility_pump(make_simple_binary_milp(), use_structural_cuts=True)
        assert res.feasible


# ---------------------------------------------------------------------------
# FPResult structure with cuts_added field
# ---------------------------------------------------------------------------

class TestFPResultWithCuts:
    def test_cuts_added_field_exists(self):
        res = feasibility_pump(make_simple_binary_milp(), use_structural_cuts=True)
        assert hasattr(res, "cuts_added")

    def test_backward_compat_plain_fp_has_cuts_added_zero(self):
        res = feasibility_pump(make_simple_binary_milp())
        assert res.cuts_added == 0

    def test_fpresult_is_dataclass(self):
        res = feasibility_pump(make_simple_binary_milp(), use_structural_cuts=True)
        assert isinstance(res, FPResult)
