"""Tests for the plain feasibility pump (Milestone 4)."""
import pytest
import gurobipy as gp
from gurobipy import GRB

from structural_fp.fp_loop import feasibility_pump, FPResult
from structural_fp.io import (
    make_simple_binary_milp,
    make_covering_milp,
    make_infeasible_lp_milp,
    make_fractional_cycle_milp,
    make_random_covering_milp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_covering(sol: dict[str, int], pairs: list[tuple[str, str]]) -> bool:
    return all(sol[a] + sol[b] >= 1 for a, b in pairs)


# ---------------------------------------------------------------------------
# Simple binary MILP (4 vars, 2 covering constraints)
# ---------------------------------------------------------------------------

class TestSimpleBinary:
    def setup_method(self):
        self.model = make_simple_binary_milp()

    def test_finds_feasible(self):
        res = feasibility_pump(self.model)
        assert res.feasible

    def test_reason_is_feasible(self):
        res = feasibility_pump(self.model)
        assert res.reason == "feasible"

    def test_solution_is_dict(self):
        res = feasibility_pump(self.model)
        assert isinstance(res.solution, dict)

    def test_solution_keys(self):
        res = feasibility_pump(self.model)
        assert set(res.solution.keys()) == {"x0", "x1", "x2", "x3"}

    def test_solution_is_binary(self):
        res = feasibility_pump(self.model)
        assert all(v in (0, 1) for v in res.solution.values())

    def test_solution_satisfies_c0(self):
        res = feasibility_pump(self.model)
        sol = res.solution
        assert sol["x0"] + sol["x1"] >= 1

    def test_solution_satisfies_c1(self):
        res = feasibility_pump(self.model)
        sol = res.solution
        assert sol["x2"] + sol["x3"] >= 1

    def test_iterations_at_least_1(self):
        res = feasibility_pump(self.model)
        assert res.iterations >= 1

    def test_history_is_list(self):
        res = feasibility_pump(self.model)
        assert isinstance(res.history, list)


# ---------------------------------------------------------------------------
# Set covering MILP (variable sizes)
# ---------------------------------------------------------------------------

class TestCoveringMilp:
    def test_6var_feasible(self):
        res = feasibility_pump(make_covering_milp(n=6))
        assert res.feasible

    def test_8var_feasible(self):
        res = feasibility_pump(make_covering_milp(n=8))
        assert res.feasible

    def test_6var_solution_binary(self):
        res = feasibility_pump(make_covering_milp(n=6))
        assert all(v in (0, 1) for v in res.solution.values())

    def test_6var_satisfies_constraints(self):
        n = 6
        res = feasibility_pump(make_covering_milp(n=n))
        sol = res.solution
        for k in range(n // 2):
            assert sol[f"x{2*k}"] + sol[f"x{2*k+1}"] >= 1

    def test_8var_satisfies_constraints(self):
        n = 8
        res = feasibility_pump(make_covering_milp(n=n))
        sol = res.solution
        for k in range(n // 2):
            assert sol[f"x{2*k}"] + sol[f"x{2*k+1}"] >= 1


# ---------------------------------------------------------------------------
# LP-infeasible instance
# ---------------------------------------------------------------------------

class TestInfeasibleLP:
    def test_not_feasible(self):
        res = feasibility_pump(make_infeasible_lp_milp())
        assert not res.feasible

    def test_reason_lp_infeasible(self):
        res = feasibility_pump(make_infeasible_lp_milp())
        assert res.reason == "lp_infeasible"

    def test_solution_is_none(self):
        res = feasibility_pump(make_infeasible_lp_milp())
        assert res.solution is None

    def test_iterations_zero(self):
        res = feasibility_pump(make_infeasible_lp_milp())
        assert res.iterations == 0


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------

class TestCycleDetection:
    def test_cycle_terminates(self):
        # x0+x1=0.5 has no binary solution; FP must detect a cycle.
        res = feasibility_pump(make_fractional_cycle_milp(), max_iter=20)
        assert not res.feasible
        assert res.reason == "cycle"

    def test_cycle_within_max_iter(self):
        res = feasibility_pump(make_fractional_cycle_milp(), max_iter=20)
        assert res.iterations <= 20

    def test_history_nonempty_on_cycle(self):
        res = feasibility_pump(make_fractional_cycle_milp(), max_iter=20)
        assert len(res.history) >= 1


# ---------------------------------------------------------------------------
# max_iter respected
# ---------------------------------------------------------------------------

class TestMaxIter:
    def test_max_iter_1_terminates(self):
        res = feasibility_pump(make_covering_milp(n=10), max_iter=1)
        assert res.iterations <= 1

    def test_max_iter_returns_max_iter_reason_if_no_early_exit(self):
        # With max_iter=1, either feasible (if LP gives integral solution) or max_iter
        res = feasibility_pump(make_covering_milp(n=10), max_iter=1)
        assert res.reason in ("feasible", "max_iter", "cycle")

    def test_max_iter_2_terminates(self):
        res = feasibility_pump(make_covering_milp(n=10), max_iter=2)
        assert res.iterations <= 2


# ---------------------------------------------------------------------------
# Integer variable auto-detection
# ---------------------------------------------------------------------------

class TestIntVarDetection:
    def test_autodetect_finds_solution(self):
        res = feasibility_pump(make_simple_binary_milp(), int_vars=None)
        assert res.feasible

    def test_explicit_int_vars_matches_autodetect(self):
        model = make_simple_binary_milp()
        res1 = feasibility_pump(model, int_vars=None)
        res2 = feasibility_pump(model, int_vars=["x0", "x1", "x2", "x3"])
        assert res1.feasible == res2.feasible

    def test_subset_int_vars(self):
        # Treat only x0, x1 as integer; x2, x3 relaxed.
        # Covering LP with x2,x3 continuous -> always feasible.
        model = make_simple_binary_milp()
        res = feasibility_pump(model, int_vars=["x0", "x1"])
        assert res.feasible
        # x0+x1 rounded to satisfy c0
        sol = res.solution
        assert set(sol.keys()) == {"x0", "x1"}


# ---------------------------------------------------------------------------
# FPResult type and structure
# ---------------------------------------------------------------------------

class TestFPResult:
    def test_result_is_fpresult(self):
        res = feasibility_pump(make_simple_binary_milp())
        assert isinstance(res, FPResult)

    def test_feasible_is_bool(self):
        res = feasibility_pump(make_simple_binary_milp())
        assert isinstance(res.feasible, bool)

    def test_iterations_is_int(self):
        res = feasibility_pump(make_simple_binary_milp())
        assert isinstance(res.iterations, int)

    def test_reason_is_str(self):
        res = feasibility_pump(make_simple_binary_milp())
        assert isinstance(res.reason, str)

    def test_history_is_list_of_tuples(self):
        res = feasibility_pump(make_simple_binary_milp())
        assert all(isinstance(h, tuple) for h in res.history)

    def test_infeasible_solution_is_none(self):
        res = feasibility_pump(make_infeasible_lp_milp())
        assert res.solution is None


# ---------------------------------------------------------------------------
# _FeasChecker must judge feasibility on the original constraints alone
# ---------------------------------------------------------------------------

class TestFeasCheckerUnboundedObjective:
    """
    model.relax() keeps the model's original objective. If _FeasChecker
    doesn't zero it out, fixing the integer variables can still leave a
    continuous variable unbounded in the objective's favorable direction,
    so optimize() reports Status=UNBOUNDED (not OPTIMAL) even though the
    original *constraints* (ignoring the objective) are clearly satisfied
    -- is_feasible() would wrongly return False for a genuinely feasible z.
    """

    def test_feasible_point_not_reported_infeasible_when_objective_unbounded(self):
        model = gp.Model()
        model.setParam("OutputFlag", 0)
        x = model.addVar(vtype=GRB.BINARY, name="x")
        y = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="y")  # no upper bound
        model.addConstr(x + y >= 1)
        # Unbounded as y -> inf, regardless of x's fixed value.
        model.setObjective(-y, GRB.MINIMIZE)
        model.update()

        from structural_fp.fp_loop import _FeasChecker
        checker = _FeasChecker(model, ["x"])
        assert checker.is_feasible({"x": 1})  # x=1, y=0 satisfies x+y>=1
        assert checker.is_feasible({"x": 0})  # x=0, y>=1 (e.g. y=1) also satisfies it


# ---------------------------------------------------------------------------
# gurobi_seed (M13): reproducibility, no effect on the perturbation RNG
# ---------------------------------------------------------------------------

class TestGurobiSeed:
    def test_same_seed_twice_is_fully_reproducible(self):
        kwargs = dict(
            mode="optimality", use_structural_cuts=True, use_face_probe=True, face_k=3,
            max_iter=300, max_perturbations=20, perturb_scale=0.5,
            random_seed=42, gurobi_seed=7,
        )
        a = feasibility_pump(make_random_covering_milp(n=18, m=12, seed=5), **kwargs)
        b = feasibility_pump(make_random_covering_milp(n=18, m=12, seed=5), **kwargs)
        assert (a.reason, a.iterations, a.best_obj, a.bound, a.perturbations, a.cuts_added) == \
               (b.reason, b.iterations, b.best_obj, b.bound, b.perturbations, b.cuts_added)

    def test_none_leaves_gurobi_seed_param_untouched(self):
        # Just needs to not raise / not change behavior vs. not passing it.
        kwargs = dict(max_iter=50, random_seed=0)
        a = feasibility_pump(make_covering_milp(n=8), **kwargs)
        b = feasibility_pump(make_covering_milp(n=8), gurobi_seed=None, **kwargs)
        assert (a.feasible, a.iterations, a.reason) == (b.feasible, b.iterations, b.reason)
