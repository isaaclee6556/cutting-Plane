"""
Face-probe step (use_face_probe / face_k): an additional per-iteration step on
top of use_structural_cuts. Instead of only inserting the single rounded
point z, it checks all 2^face_k corners of the face spanned by the face_k
most-fractional directions (fixing every other int var at z): any feasible
corner ends the pump immediately; otherwise a single CubeCut covering the
whole face is added unconditionally (cuts.CubeCut generalized to any face
size, not just k=3 — see its docstring).
"""
import pytest

from structural_fp.fp_loop import feasibility_pump, FPResult
from structural_fp.io import (
    make_covering_milp,
    make_random_covering_milp,
    make_fractional_cycle_milp,
)


class TestValidation:
    def test_requires_structural_cuts(self):
        with pytest.raises(ValueError, match="use_face_probe requires use_structural_cuts"):
            feasibility_pump(make_fractional_cycle_milp(), use_face_probe=True)

    def test_default_off_matches_plain_structural(self):
        # use_face_probe defaults to False: identical to calling without it.
        kwargs = dict(use_structural_cuts=True, max_iter=50, random_seed=0)
        a = feasibility_pump(make_random_covering_milp(n=16, m=8, seed=2), **kwargs)
        b = feasibility_pump(make_random_covering_milp(n=16, m=8, seed=2), use_face_probe=False, **kwargs)
        assert (a.iterations, a.reason, a.cuts_added, a.cuts_by_type) == \
               (b.iterations, b.reason, b.cuts_added, b.cuts_by_type)
        assert b.face_probe_cuts == 0


class TestFaceProbe:
    def test_finds_feasible_corner_directly(self):
        res = feasibility_pump(
            make_random_covering_milp(n=16, m=8, seed=2),
            use_structural_cuts=True, use_face_probe=True, face_k=4,
            max_iter=50, random_seed=42,
        )
        assert isinstance(res, FPResult)
        assert res.feasible
        # Known to need 9 iterations without face probe (see test_fp_loop_guards.py
        # baseline); probing a 4-fractional-var face should find a feasible point
        # no later than the plain-structural run would.
        assert res.iterations <= 9

    def test_all_infeasible_adds_cube_cut_and_keeps_cover(self):
        res = feasibility_pump(
            make_fractional_cycle_milp(),
            use_structural_cuts=True, use_face_probe=True, face_k=4,
            max_iter=20, debug_check_cover=True,
        )
        assert res.face_probe_cuts >= 1
        assert res.cuts_by_type.get("CubeCut", 0) >= 1

    def test_face_k_clamped_to_n_ivars(self):
        # face_k=4 on a 2-variable instance: clamp to k_eff=2 (4 corners), no crash.
        res = feasibility_pump(
            make_fractional_cycle_milp(),
            use_structural_cuts=True, use_face_probe=True, face_k=4,
            max_iter=10,
        )
        assert isinstance(res, FPResult)

    @pytest.mark.parametrize("make", [
        lambda: make_covering_milp(n=6),
        lambda: make_covering_milp(n=8),
        lambda: make_random_covering_milp(n=12, m=6, seed=0),
        lambda: make_random_covering_milp(n=20, m=10, seed=3),
    ])
    def test_cover_invariant_holds_with_face_probe(self, make):
        res = feasibility_pump(
            make(), use_structural_cuts=True, use_face_probe=True, face_k=4,
            max_iter=50, debug_check_cover=True,
        )
        assert isinstance(res, FPResult)
