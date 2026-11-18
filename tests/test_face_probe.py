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
    def test_face_probe_no_longer_requires_structural_cuts(self):
        # M12: probing is independent of cut_mode -- this used to raise
        # ValueError, now runs standalone under cut_mode="none" (no cuts at
        # all, probing only decides which point is returned).
        res = feasibility_pump(make_fractional_cycle_milp(), use_face_probe=True, face_k=4, max_iter=20)
        assert isinstance(res, FPResult)
        assert res.cuts_added == 0
        assert res.face_probe_cuts == 0
        assert res.probe_vertex_cuts == 0

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


class TestFaceProbeNoEcho:
    """
    face_k == 3 (the default) matches StructureIndex's native cube size
    exactly, so inserting all 8 corners of a probed face also re-detects
    that identical cube through the normal cuts_from_index/select_cuts
    path. Regression test for fp_loop's exact-duplicate-row guard, which
    keeps the face-probe CubeCut from being added a second time via that
    native re-detection.

    Note: since select_cuts now adds every dominance-surviving cut
    unconditionally (no violation filtering/cap), a Star/Tulip/Propeller
    anchored at one of these corners that is NOT literally dominated by
    this cube (e.g. its leaves reach outside this cube's free directions
    because the corner also touches another already-visited point from a
    different face probe) is expected to show up as its own row now --
    that used to be silently dropped by the old violation filter even
    though dominance_prune never actually proved it redundant. Only the
    face cut's own row is guaranteed unique.
    """

    def test_default_face_k_adds_no_duplicate_cube_row(self):
        res = feasibility_pump(
            make_random_covering_milp(n=40, m=35, seed=1),
            use_structural_cuts=True, use_face_probe=True,  # face_k defaults to 3
            debug_check_cover=True,
            max_perturbations=5, perturb_scale=0.4, random_seed=1, max_iter=30,
        )
        assert res.face_probe_cuts >= 1
        # Exactly one CubeCut row per face probe -- never duplicated via
        # the native re-detection of the same face.
        assert res.cuts_by_type.get("CubeCut", 0) == res.face_probe_cuts

    def test_face_k4_no_echo_across_later_iterations(self):
        """
        Regression test for a bug where a face_k>=4 probe's own CubeCut
        only suppressed its face's native sub-structures (sub-cubes/
        squares/stars/tulips/propellers) in the SAME iteration it was
        created. fp_loop only passed `[face_cut]` (this iteration's cut)
        into the dominance pass, not the full history of face cuts -- so on
        any LATER iteration, cuts_from_index(idx) re-derives those same
        sub-structures from idx (which never forgets), dominance_prune has
        no face cut in the pool to prune them against, and they leak back
        into the LP as new (weaker, redundant) rows. With face_k=3 this was
        masked because the echo is an exact-duplicate row of the face cut
        itself (caught by the row-key guard); at k=4 the echoes are
        genuinely different (weaker) rows that only cube-dominance, fed the
        right history, can catch.
        """
        res = feasibility_pump(
            make_random_covering_milp(n=40, m=35, seed=1),
            use_structural_cuts=True, use_face_probe=True, face_k=4,
            debug_check_cover=True,
            max_perturbations=5, perturb_scale=0.4, random_seed=1, max_iter=30,
        )
        assert res.face_probe_cuts >= 1
        assert res.cuts_by_type.get("CubeCut", 0) == res.face_probe_cuts


class TestCutModeProbeMatrix:
    """
    M12: probing is independent of cut_mode. Covers the 3 (cut_mode) x
    (probe_k in {0,2,3,4}) combinations that the method registry (M13)
    actually dispatches to.
    """

    @pytest.mark.parametrize("probe_k", [2, 3, 4])
    @pytest.mark.parametrize("cut_kwargs", [
        dict(),                                   # cut_mode="none"
        dict(use_no_good_cuts=True),              # cut_mode="nogood"
        dict(use_structural_cuts=True),           # cut_mode="structural"
    ])
    def test_runs_and_keeps_cover_invariant(self, probe_k, cut_kwargs):
        debug_kwargs = dict(debug_check_cover=True) if "use_structural_cuts" in cut_kwargs else dict()
        res = feasibility_pump(
            make_random_covering_milp(n=16, m=10, seed=3),
            use_face_probe=True, face_k=probe_k,
            max_perturbations=10, perturb_scale=0.4, random_seed=1, max_iter=200,
            **cut_kwargs, **debug_kwargs,
        )
        assert isinstance(res, FPResult)

    def test_nogood_probe_only_adds_vertex_cuts(self):
        res = feasibility_pump(
            make_random_covering_milp(n=20, m=14, seed=2),
            use_no_good_cuts=True, use_face_probe=True, face_k=4,
            max_perturbations=10, perturb_scale=0.4, random_seed=1, max_iter=200,
        )
        assert set(res.cuts_by_type.keys()) <= {"VertexCut"}
        assert res.face_probe_cuts == 0

    def test_nogood_probe_counts_probe_vertex_cuts_separately(self):
        res = feasibility_pump(
            make_fractional_cycle_milp(),
            use_no_good_cuts=True, use_face_probe=True, face_k=4,
            max_perturbations=5, max_iter=20,
        )
        assert res.probe_vertex_cuts >= 1
        assert res.probe_vertex_cuts <= res.cuts_by_type.get("VertexCut", 0)

    def test_probe_k_zero_matches_no_probe_at_all(self):
        for cut_kwargs in (dict(), dict(use_no_good_cuts=True), dict(use_structural_cuts=True)):
            kwargs = dict(max_iter=50, random_seed=0, **cut_kwargs)
            a = feasibility_pump(make_random_covering_milp(n=16, m=8, seed=2), use_face_probe=False, **kwargs)
            b = feasibility_pump(make_random_covering_milp(n=16, m=8, seed=2), use_face_probe=True, face_k=0, **kwargs)
            assert (a.iterations, a.reason, a.cuts_added, a.cuts_by_type) == \
                   (b.iterations, b.reason, b.cuts_added, b.cuts_by_type)
