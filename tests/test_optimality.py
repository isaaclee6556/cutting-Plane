"""
Tests for mode="optimality" (Milestone 11).

Correctness is checked against a direct Gurobi solve of the same model
(the ground truth), not against hand-picked expected numbers, since the
whole point of optimality mode is that it must agree with the MILP's real
optimum whenever it claims proved_optimal.
"""
from __future__ import annotations
import time

import gurobipy as gp
from gurobipy import GRB
import pytest

from structural_fp.fp_loop import feasibility_pump, FPResult
from structural_fp.io import (
    make_covering_milp,
    make_random_covering_milp,
    make_infeasible_lp_milp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ground_truth(model: gp.Model) -> float:
    gt = model.copy()
    gt.setParam("OutputFlag", 0)
    gt.optimize()
    assert gt.Status == GRB.OPTIMAL
    return gt.ObjVal


def _weighted_covering_milp(n: int, m: int, seed: int, weights) -> gp.Model:
    """Same constraints as make_random_covering_milp, but with a non-uniform
    objective (weights[i] on x_i) instead of plain sum(x)."""
    model = make_random_covering_milp(n=n, m=m, seed=seed)
    xs = {v.VarName: v for v in model.getVars()}
    model.setObjective(gp.quicksum(weights[i] * xs[f"x{i}"] for i in range(n)), GRB.MINIMIZE)
    model.update()
    return model


# ---------------------------------------------------------------------------
# Correctness against ground truth
# ---------------------------------------------------------------------------

class TestOptimalAgainstGroundTruth:
    @pytest.mark.parametrize("make,kwargs", [
        (lambda: make_covering_milp(n=8), dict()),
        (lambda: make_covering_milp(n=8), dict(use_no_good_cuts=True)),
        (lambda: make_covering_milp(n=8), dict(use_structural_cuts=True)),
        (lambda: make_random_covering_milp(n=16, m=10, seed=1), dict()),
        (lambda: make_random_covering_milp(n=16, m=10, seed=1),
         dict(use_structural_cuts=True, use_face_probe=True, face_k=3)),
        (lambda: make_random_covering_milp(n=16, m=10, seed=1),
         dict(use_structural_cuts=True, use_face_probe=True, face_k=4)),
        (lambda: _weighted_covering_milp(14, 9, 2, [i + 1 for i in range(14)]), dict()),
        (lambda: _weighted_covering_milp(14, 9, 2, [0.37 * (i + 1) for i in range(14)]),
         dict(use_structural_cuts=True)),
    ])
    def test_matches_ground_truth_when_proved_optimal(self, make, kwargs):
        # NOTE: a cut only excludes an integer point from being an *LP*
        # solution -- it does not stop the rounding step from landing back
        # on that same bit pattern from a different fractional x_hat, so the
        # heuristic can still cycle (give up) on a hard enough instance even
        # with cuts and perturbation. The guarantee we actually check (per
        # spec) is conditional: *when* it claims "optimal", that claim must
        # be correct -- not that it always reaches that claim in bounded time.
        model = make()
        gt = _ground_truth(model)
        res = feasibility_pump(
            model, mode="optimality", max_iter=2000,
            max_perturbations=20, perturb_scale=0.4, random_seed=0,
            **kwargs,
        )
        assert isinstance(res, FPResult)
        if res.reason == "optimal":
            assert res.proved_optimal
            assert res.feasible
            assert res.best_obj == pytest.approx(gt, abs=1e-6)
            assert res.bound == pytest.approx(gt, abs=1e-6)
            assert res.gap == pytest.approx(0.0, abs=1e-6)
        else:
            # Didn't converge in time -- still must never claim something
            # better than the true optimum, nor a bound looser than it looks.
            if res.best_obj is not None:
                assert res.best_obj >= gt - 1e-6
            if res.bound is not None:
                assert res.bound <= gt + 1e-6


# ---------------------------------------------------------------------------
# Infeasible MILP
# ---------------------------------------------------------------------------

class TestInfeasible:
    def test_proves_infeasible_no_incumbent(self):
        res = feasibility_pump(make_infeasible_lp_milp(), mode="optimality", max_iter=50)
        assert res.reason in ("infeasible", "lp_infeasible")
        assert not res.feasible
        assert res.best_obj is None
        assert res.best_solution is None
        assert res.proved_optimal


# ---------------------------------------------------------------------------
# Maximize sign handling
# ---------------------------------------------------------------------------

class TestMaximize:
    def test_knapsack_like_maximize(self):
        model = gp.Model()
        model.setParam("OutputFlag", 0)
        xs = [model.addVar(vtype=GRB.BINARY, name=f"x{i}") for i in range(10)]
        model.addConstr(gp.quicksum(xs) <= 4)
        model.setObjective(gp.quicksum((i + 1) * xs[i] for i in range(10)), GRB.MAXIMIZE)
        model.update()

        gt = _ground_truth(model)
        res = feasibility_pump(model, mode="optimality", max_iter=500)
        assert res.reason == "optimal"
        assert res.proved_optimal
        assert res.best_obj == pytest.approx(gt, abs=1e-6)
        assert res.bound == pytest.approx(gt, abs=1e-6)


# ---------------------------------------------------------------------------
# Incumbent trace
# ---------------------------------------------------------------------------

class TestIncumbentTrace:
    def test_trace_is_monotonically_improving(self):
        model = make_random_covering_milp(n=20, m=14, seed=9)
        res = feasibility_pump(
            model, mode="optimality", max_iter=1000,
            use_structural_cuts=True, use_face_probe=True, face_k=3,
            max_perturbations=15, perturb_scale=0.4, random_seed=4,
        )
        us = [u for (_, _, u) in res.incumbent_trace]
        assert us, "expected at least one incumbent update"
        assert all(us[i + 1] <= us[i] for i in range(len(us) - 1))
        if res.proved_optimal:
            assert res.best_obj == pytest.approx(_ground_truth(model), abs=1e-6)

    def test_n_incumbent_updates_matches_trace_length(self):
        model = make_random_covering_milp(n=20, m=14, seed=9)
        res = feasibility_pump(
            model, mode="optimality", max_iter=1000,
            use_structural_cuts=True, max_perturbations=10, perturb_scale=0.4, random_seed=4,
        )
        assert res.n_incumbent_updates == len(res.incumbent_trace)
        assert res.n_feasible_found >= res.n_incumbent_updates


# ---------------------------------------------------------------------------
# Cover invariant holds across the full cut-mode x probe matrix
# ---------------------------------------------------------------------------

class TestCoverInvariantOptimality:
    @pytest.mark.parametrize("kwargs", [
        dict(),
        dict(use_no_good_cuts=True),
        dict(use_structural_cuts=True),
        dict(use_structural_cuts=True, use_face_probe=True, face_k=3),
        dict(use_structural_cuts=True, use_face_probe=True, face_k=4),
    ])
    def test_runs_without_crashing(self, kwargs):
        model = make_random_covering_milp(n=16, m=10, seed=3)
        debug_kwargs = dict(debug_check_cover=True) if "use_structural_cuts" in kwargs else dict()
        res = feasibility_pump(
            model, mode="optimality", max_iter=300,
            max_perturbations=10, perturb_scale=0.4, random_seed=1,
            **kwargs, **debug_kwargs,
        )
        assert isinstance(res, FPResult)
        assert res.mode == "optimality"


# ---------------------------------------------------------------------------
# time_limit preserves the incumbent found so far
# ---------------------------------------------------------------------------

class TestTimeLimitPreservesIncumbent:
    def test_short_time_limit_still_returns_quickly_and_keeps_incumbent(self):
        model = make_random_covering_milp(n=40, m=30, seed=5)
        t0 = time.perf_counter()
        res = feasibility_pump(
            model, mode="optimality", max_iter=10_000_000, time_limit=1e-9,
            use_structural_cuts=True, use_face_probe=True, face_k=4,
            max_perturbations=1000, perturb_scale=0.5,
        )
        wall = time.perf_counter() - t0
        assert res.reason == "time_limit"
        assert not res.proved_optimal
        assert wall < 10  # never blows far past the (near-zero) budget

    def test_incumbent_found_before_timeout_is_kept(self):
        # A generous budget that's still short enough to likely not finish
        # proving optimality on a moderately sized instance, but long enough
        # for at least one feasible point to be found first.
        model = make_random_covering_milp(n=30, m=20, seed=11)
        res = feasibility_pump(
            model, mode="optimality", max_iter=10_000_000, time_limit=0.2,
            use_structural_cuts=True, use_face_probe=True, face_k=4,
            max_perturbations=1000, perturb_scale=0.5,
        )
        assert res.reason in ("time_limit", "optimal")
        if res.n_feasible_found > 0:
            assert res.best_obj is not None
            assert res.best_solution is not None


# ---------------------------------------------------------------------------
# M13.1: unbounded relaxation is not infeasibility; n_feasible_found doesn't
# double-count a z re-evaluated before a perturbation's visited.clear()
# ---------------------------------------------------------------------------

class TestUnboundedRelaxation:
    def test_unbounded_initial_relaxation_is_not_proved_infeasible(self):
        # min -y, y continuous >= 0, no upper bound, no constraint linking
        # it to x: the relaxation itself is unbounded before any cuts even
        # exist, which must not be reported/treated as a proof of
        # infeasibility (a real, distinct outcome).
        model = gp.Model()
        model.setParam("OutputFlag", 0)
        x = model.addVar(vtype=GRB.BINARY, name="x")
        y = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="y")
        model.setObjective(-y, GRB.MINIMIZE)
        model.update()

        res = feasibility_pump(model, mode="optimality", max_iter=10)
        assert res.reason == "unbounded_relaxation"
        assert not res.proved_optimal
        assert not res.feasible


class TestFeasibleFoundNotDoubleCounted:
    def test_repeated_same_point_across_perturbations_not_double_counted(self):
        # Known repro (see M13.1 discussion): plain mode (no cuts) on this
        # weighted covering instance settles into oscillating between 3
        # distinct points across perturbation epochs, one of them feasible
        # and revisited on consecutive iterations before each perturbation
        # resets `visited`. Confirmed via git-stash comparison that the
        # pre-fix code counted it twice per epoch (196) vs. once (98) after
        # guarding on `z_key not in visited` -- regression-pin that count,
        # not just "some" count, so a future change that reintroduces the
        # double-count would actually fail this test.
        model = make_random_covering_milp(14, 9, seed=2)
        xs = {v.VarName: v for v in model.getVars()}
        model.setObjective(sum((i + 1) * xs[f"x{i}"] for i in range(14)), GRB.MINIMIZE)
        model.update()

        res = feasibility_pump(
            model, mode="optimality", max_iter=5000,
            max_perturbations=100, perturb_scale=0.5, random_seed=0,
        )
        assert res.reason == "cycle"
        assert res.n_feasible_found == 98
        assert res.n_feasible_found < res.iterations
