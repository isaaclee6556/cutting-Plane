"""
Tests for cycle-breaking perturbation (Milestone 6).

Perturbation fires when a cycle is detected AND perturbations < max_perturbations.
It clears the visited set and uses a perturbed projection objective, then continues.
"""
import pytest

from structural_fp.fp_loop import feasibility_pump, FPResult
from structural_fp.io import (
    make_simple_binary_milp,
    make_covering_milp,
    make_infeasible_lp_milp,
    make_fractional_cycle_milp,
    make_random_covering_milp,
)


# ---------------------------------------------------------------------------
# FPResult.perturbations field
# ---------------------------------------------------------------------------

class TestPerturbationsField:
    def test_field_exists(self):
        res = feasibility_pump(make_simple_binary_milp())
        assert hasattr(res, "perturbations")

    def test_zero_without_perturbation_setting(self):
        res = feasibility_pump(make_simple_binary_milp(), max_perturbations=0)
        assert res.perturbations == 0

    def test_zero_when_no_cycle_occurs(self):
        # Simple binary: finds feasible in iter 1, no cycle, no perturbation
        res = feasibility_pump(make_simple_binary_milp(), max_perturbations=5)
        assert res.perturbations == 0

    def test_type_is_int(self):
        res = feasibility_pump(make_simple_binary_milp(), max_perturbations=3)
        assert isinstance(res.perturbations, int)


# ---------------------------------------------------------------------------
# max_perturbations=0 (default): stop immediately on first cycle
# ---------------------------------------------------------------------------

class TestNoPerturbation:
    def test_cycle_milp_returns_cycle(self):
        res = feasibility_pump(make_fractional_cycle_milp(), max_perturbations=0)
        assert res.reason == "cycle"

    def test_cycle_milp_zero_perturbations(self):
        res = feasibility_pump(make_fractional_cycle_milp(), max_perturbations=0)
        assert res.perturbations == 0

    def test_feasible_milp_unaffected(self):
        res = feasibility_pump(make_simple_binary_milp(), max_perturbations=0)
        assert res.feasible

    def test_default_max_perturbations_is_zero(self):
        # Default should be equivalent to max_perturbations=0
        res1 = feasibility_pump(make_fractional_cycle_milp())
        res2 = feasibility_pump(make_fractional_cycle_milp(), max_perturbations=0)
        assert res1.reason == res2.reason
        assert res1.perturbations == res2.perturbations


# ---------------------------------------------------------------------------
# Perturbation extends iterations on the cycle MILP
# ---------------------------------------------------------------------------

class TestPerturbationExtendsIterations:
    def test_more_perturbations_more_iterations(self):
        # Each perturbation + re-detection costs ~2 iterations.
        # max_perturbations=0: ~2 iters; max_perturbations=3: ~8 iters.
        res0 = feasibility_pump(make_fractional_cycle_milp(), max_perturbations=0)
        res3 = feasibility_pump(make_fractional_cycle_milp(), max_perturbations=3)
        assert res0.iterations < res3.iterations

    def test_all_perturbations_used_on_cycle_milp(self):
        # The cycle MILP always rounds to (0,0) regardless of perturbation,
        # so all max_perturbations perturbations are exhausted before giving up.
        res = feasibility_pump(make_fractional_cycle_milp(), max_perturbations=3)
        assert res.perturbations == 3

    def test_reason_still_cycle_after_exhausting_perturbations(self):
        res = feasibility_pump(make_fractional_cycle_milp(), max_perturbations=2)
        assert res.reason == "cycle"

    def test_perturbation_1_uses_one_perturbation(self):
        res = feasibility_pump(make_fractional_cycle_milp(), max_perturbations=1)
        assert res.perturbations == 1


# ---------------------------------------------------------------------------
# Perturbation does not break feasible instances
# ---------------------------------------------------------------------------

class TestPerturbationOnFeasible:
    def test_simple_binary_still_feasible(self):
        res = feasibility_pump(make_simple_binary_milp(), max_perturbations=5)
        assert res.feasible

    def test_covering_6var_still_feasible(self):
        res = feasibility_pump(make_covering_milp(n=6), max_perturbations=5)
        assert res.feasible

    def test_covering_8var_still_feasible(self):
        res = feasibility_pump(make_covering_milp(n=8), max_perturbations=5)
        assert res.feasible

    def test_solution_satisfies_constraints(self):
        n = 6
        res = feasibility_pump(make_covering_milp(n=n), max_perturbations=5)
        assert res.feasible
        sol = res.solution
        for k in range(n // 2):
            assert sol[f"x{2*k}"] + sol[f"x{2*k+1}"] >= 1

    def test_lp_infeasible_unaffected(self):
        res = feasibility_pump(make_infeasible_lp_milp(), max_perturbations=5)
        assert res.reason == "lp_infeasible"
        assert res.perturbations == 0


# ---------------------------------------------------------------------------
# Perturbation combined with structural cuts
# ---------------------------------------------------------------------------

class TestPerturbationWithCuts:
    def test_cuts_and_perturbation_find_feasible(self):
        res = feasibility_pump(
            make_simple_binary_milp(),
            use_structural_cuts=True,
            max_perturbations=5,
        )
        assert res.feasible

    def test_covering_with_cuts_and_perturbation(self):
        res = feasibility_pump(
            make_covering_milp(n=8),
            use_structural_cuts=True,
            max_perturbations=5,
        )
        assert res.feasible

    def test_cycle_milp_cuts_and_perturbation_terminates(self):
        res = feasibility_pump(
            make_fractional_cycle_milp(),
            use_structural_cuts=True,
            max_perturbations=3,
            max_iter=20,
        )
        assert res.iterations <= 20

    def test_cuts_and_perturbation_solution_binary(self):
        res = feasibility_pump(
            make_covering_milp(n=8),
            use_structural_cuts=True,
            max_perturbations=3,
        )
        if res.feasible:
            assert all(v in (0, 1) for v in res.solution.values())


# ---------------------------------------------------------------------------
# random_seed for reproducibility
# ---------------------------------------------------------------------------

class TestRandomSeed:
    def test_same_seed_same_result(self):
        model = make_fractional_cycle_milp()
        res1 = feasibility_pump(model, max_perturbations=3, random_seed=42)
        res2 = feasibility_pump(model, max_perturbations=3, random_seed=42)
        assert res1.iterations == res2.iterations
        assert res1.perturbations == res2.perturbations
        assert res1.reason == res2.reason

    def test_seed_does_not_affect_no_perturbation_run(self):
        model = make_simple_binary_milp()
        res1 = feasibility_pump(model, max_perturbations=0, random_seed=1)
        res2 = feasibility_pump(model, max_perturbations=0, random_seed=99)
        assert res1.feasible == res2.feasible
        assert res1.reason == res2.reason


# ---------------------------------------------------------------------------
# perturb_scale parameter
# ---------------------------------------------------------------------------

class TestPerturbScale:
    def test_zero_scale_acts_like_standard_projection(self):
        # scale=0.0 means no noise added — still breaks cycle via visited.clear()
        res = feasibility_pump(
            make_fractional_cycle_milp(),
            max_perturbations=2,
            perturb_scale=0.0,
        )
        assert res.perturbations == 2

    def test_large_scale_still_terminates(self):
        res = feasibility_pump(
            make_simple_binary_milp(),
            max_perturbations=5,
            perturb_scale=10.0,
        )
        assert res.iterations >= 1  # terminates normally
