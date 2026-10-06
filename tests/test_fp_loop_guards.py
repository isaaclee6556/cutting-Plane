"""
Guards and cover invariant for feasibility_pump with cuts:
  - structural/no-good modes reject integer variables outside {0,1};
  - structural mode guarantees every visited point is excluded by some added row.
"""
import pytest
import gurobipy as gp
from gurobipy import GRB

import structural_fp.fp_loop as fp_loop
from structural_fp.cuts import VertexCut
from structural_fp.fp_loop import feasibility_pump, FPResult
from structural_fp.io import (
    make_covering_milp,
    make_random_covering_milp,
    make_fractional_cycle_milp,
)


def _model_with_general_int(ub: float = 5.0, vtype=GRB.INTEGER) -> gp.Model:
    m = gp.Model()
    m.setParam("OutputFlag", 0)
    x = m.addVar(vtype=vtype, lb=0, ub=ub, name="x")
    y = m.addVar(vtype=GRB.BINARY, name="y")
    m.addConstr(x + y >= 1)
    m.setObjective(x + y, GRB.MINIMIZE)
    m.update()
    return m


# ---------------------------------------------------------------------------
# A. binary guard
# ---------------------------------------------------------------------------

class TestBinaryGuard:
    @pytest.mark.parametrize("kwargs", [
        {"use_structural_cuts": True},
        {"use_no_good_cuts": True},
    ])
    def test_rejects_general_integer_in_cut_modes(self, kwargs):
        with pytest.raises(ValueError,
                           match=r"1 integer variable\(s\) have other bounds: x"):
            feasibility_pump(_model_with_general_int(), **kwargs)

    def test_error_message_points_to_plain_fp(self):
        with pytest.raises(ValueError, match="plain FP"):
            feasibility_pump(_model_with_general_int(), use_structural_cuts=True)

    def test_plain_mode_accepts_general_integer(self):
        res = feasibility_pump(_model_with_general_int())
        assert isinstance(res, FPResult)

    def test_integer_with_01_bounds_passes_guard(self):
        m = _model_with_general_int(ub=1.0)
        res = feasibility_pump(m, use_structural_cuts=True)
        assert isinstance(res, FPResult)


# ---------------------------------------------------------------------------
# B. cover invariant
# ---------------------------------------------------------------------------

class TestCoverGuarantee:
    def test_fallback_fires_when_selected_cut_misses_z(self, monkeypatch):
        # Replace select_cuts with one that returns only a VertexCut of a
        # different point, which does not exclude the rejected z = 0.
        def misses_z(cuts, x_hat, k, tol=1e-6, prune=True):
            n = cuts[0].n
            return [VertexCut(z=0b11, n=n)]

        monkeypatch.setattr(fp_loop, "select_cuts", misses_z)
        res = feasibility_pump(
            make_fractional_cycle_milp(), max_iter=5,
            use_structural_cuts=True, debug_check_cover=True,
        )
        assert res.fallback_cuts >= 1

    def test_no_fallback_when_selected_cut_already_covers_z(self, monkeypatch):
        def covers_z(cuts, x_hat, k, tol=1e-6, prune=True):
            return [c for c in cuts if isinstance(c, VertexCut) and c.z == 0]

        monkeypatch.setattr(fp_loop, "select_cuts", covers_z)
        res = feasibility_pump(
            make_fractional_cycle_milp(), max_iter=5,
            use_structural_cuts=True, debug_check_cover=True,
        )
        assert res.fallback_cuts == 0

    @pytest.mark.parametrize("make", [
        lambda: make_covering_milp(n=6),
        lambda: make_covering_milp(n=8),
        lambda: make_covering_milp(n=10),
        lambda: make_random_covering_milp(n=12, m=6, seed=0),
        lambda: make_random_covering_milp(n=16, m=8, seed=2),
        lambda: make_random_covering_milp(n=20, m=10, seed=3),
    ])
    def test_invariant_holds_on_synthetic_instances(self, make):
        # debug_check_cover raises AssertionError if any visited point is uncovered
        res = feasibility_pump(
            make(), max_iter=50, use_structural_cuts=True, debug_check_cover=True,
        )
        assert isinstance(res, FPResult)


# ---------------------------------------------------------------------------
# C. no-good regression (values from experiments/results/before_fix.csv)
# ---------------------------------------------------------------------------

class TestNoGoodRegression:
    @pytest.mark.parametrize("make, max_perturb, expected_iter, expected_reason", [
        (lambda: make_random_covering_milp(n=16, m=8, seed=2), 0, 2, "cycle"),
        (lambda: make_random_covering_milp(n=16, m=8, seed=2), 10, 5, "feasible"),
        (lambda: make_random_covering_milp(n=20, m=10, seed=3), 0, 2, "cycle"),
        (lambda: make_random_covering_milp(n=20, m=10, seed=3), 10, 11, "feasible"),
        (make_fractional_cycle_milp, 0, 1, "lp_infeasible"),
        (make_fractional_cycle_milp, 10, 1, "lp_infeasible"),
    ])
    def test_matches_before_fix_baseline(self, make, max_perturb, expected_iter, expected_reason):
        res = feasibility_pump(
            make(), max_iter=200, use_no_good_cuts=True,
            max_perturbations=max_perturb, perturb_scale=0.5, random_seed=0,
        )
        assert (res.iterations, res.reason) == (expected_iter, expected_reason)
