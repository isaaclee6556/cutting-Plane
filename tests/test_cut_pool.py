"""
Unit tests for cut_pool.py (Milestone 3).

Tests are organized around:
  - efficacy()
  - filter_by_violation()
  - top_k_by_efficacy()
  - dominance_prune()
  - select_cuts()  (end-to-end pipeline)

All fractional points are synthetic; no Gurobi required.
"""

import math
import pytest
from structural_fp import StructureIndex
from structural_fp.cuts import (
    VertexCut, EdgeCut, SquareCut, StarCut, CubeCut, TulipCut,
    cuts_from_index,
)
from structural_fp.cut_pool import (
    efficacy, filter_by_violation, top_k_by_efficacy,
    dominance_prune, select_cuts,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def square_index(n=3) -> StructureIndex:
    """StructureIndex with square {4,5,6,7}, n=3."""
    idx = StructureIndex(n=n)
    for p in [4, 5, 6, 7]:
        idx.insert(p)
    return idx


def star_index(n=3) -> StructureIndex:
    """StructureIndex with star at 0, leaves {1,2,4}, n=3."""
    idx = StructureIndex(n=n)
    for p in [0, 1, 2, 4]:
        idx.insert(p)
    return idx


def full_cube_index(n=3) -> StructureIndex:
    """StructureIndex with all 8 points, n=3."""
    idx = StructureIndex(n=n)
    for p in range(8):
        idx.insert(p)
    return idx


# ---------------------------------------------------------------------------
# efficacy()
# ---------------------------------------------------------------------------

class TestEfficacy:
    def test_violated_cut(self):
        # VertexCut(z=0, n=3): x0+x1+x2 >= 1
        # x_hat=(0.2,0.3,0.1): LHS=0.6, violation=0.4, ||a||=sqrt(3)
        cut = VertexCut(z=0, n=3)
        x_hat = {0: 0.2, 1: 0.3, 2: 0.1}
        expected = 0.4 / math.sqrt(3)
        assert efficacy(cut, x_hat) == pytest.approx(expected, rel=1e-6)

    def test_not_violated_returns_zero(self):
        # VertexCut(z=0, n=3): x0+x1+x2 >= 1; x_hat with LHS=1.5 → not violated
        cut = VertexCut(z=0, n=3)
        x_hat = {0: 0.6, 1: 0.5, 2: 0.4}
        assert efficacy(cut, x_hat) == pytest.approx(0.0)

    def test_exactly_on_boundary_returns_zero(self):
        # x0+x1+x2 = 1.0 exactly → not violated → efficacy 0
        cut = VertexCut(z=0, n=3)
        x_hat = {0: 0.5, 1: 0.3, 2: 0.2}
        assert efficacy(cut, x_hat) == pytest.approx(0.0)

    def test_deeper_violation_higher_efficacy(self):
        # x_hat_deep is farther from the cut than x_hat_shallow
        cut = VertexCut(z=0, n=3)
        x_shallow = {0: 0.4, 1: 0.1, 2: 0.1}   # LHS=0.6 violation=0.4
        x_deep    = {0: 0.1, 1: 0.1, 2: 0.1}   # LHS=0.3 violation=0.7
        assert efficacy(cut, x_deep) > efficacy(cut, x_shallow)

    def test_star_efficacy(self):
        # StarCut(z=0, leaves={0,1,2}): x0+x1+x2 >= 2
        # x_hat=(0.5,0.5,0.5): LHS=1.5, violation=0.5, ||a||=sqrt(3)
        cut = StarCut(z=0, leaves=frozenset({0, 1, 2}))
        x_hat = {0: 0.5, 1: 0.5, 2: 0.5}
        expected = 0.5 / math.sqrt(3)
        assert efficacy(cut, x_hat) == pytest.approx(expected, rel=1e-6)

    def test_higher_norm_lowers_efficacy(self):
        # PropellerCut has larger coefficients → lower efficacy for same violation
        # Compare VertexCut and a cut with same violation but smaller norm
        edge_cut = EdgeCut(z=4, axis=0, n=3)
        # EdgeCut(4, 0, 3): coeffs={1:1, 2:-1}, rhs=0
        # x_hat={0:0, 1:0.1, 2:0.5}: LHS=0.1-0.5=-0.4, violation=0.4, norm=sqrt(2)
        x_hat = {0: 0.0, 1: 0.1, 2: 0.5}
        expected = 0.4 / math.sqrt(2)
        assert efficacy(edge_cut, x_hat) == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# filter_by_violation()
# ---------------------------------------------------------------------------

class TestFilterByViolation:
    def test_all_violated(self):
        idx = star_index()
        cuts = cuts_from_index(idx)
        # x_hat at the centroid of visited points: easily violated by most cuts
        x_hat = {0: 0.1, 1: 0.1, 2: 0.1}
        result = filter_by_violation(cuts, x_hat, tol=1e-6)
        assert len(result) > 0

    def test_none_violated(self):
        # VertexCut(z=0, n=3): x0+x1+x2 >= 1
        # x_hat=(0.5,0.4,0.3): LHS=1.2 ≥ 1 → NOT violated
        cuts = [VertexCut(z=0, n=3)]
        x_hat = {0: 0.5, 1: 0.4, 2: 0.3}
        result = filter_by_violation(cuts, x_hat, tol=1e-6)
        assert result == []

    def test_tolerance_boundary(self):
        # VertexCut(z=0, n=3): x0+x1+x2 >= 1; x_hat: LHS = 1 - 1e-7
        cuts = [VertexCut(z=0, n=3)]
        x_hat = {0: (1 - 1e-7) / 3, 1: (1 - 1e-7) / 3, 2: (1 - 1e-7) / 3}

        # tol=1e-6: LHS ≈ 1-1e-7, so (rhs - LHS) ≈ 1e-7 < tol → NOT flagged as violated
        result_strict = filter_by_violation(cuts, x_hat, tol=1e-6)
        assert result_strict == []

        # tol=1e-9: 1e-7 > tol → IS flagged
        result_loose = filter_by_violation(cuts, x_hat, tol=1e-9)
        assert len(result_loose) == 1

    def test_mixed_violated_and_not(self):
        # Build two cuts with different rhs
        # StarCut(z=0, {0,1,2}): rhs=2 — violated by x_hat=(0.5,0.5,0.5) (LHS=1.5)
        # VertexCut(z=7, n=3): -x0-x1-x2 >= -2 — NOT violated (LHS=-1.5 > -2)
        cuts = [
            StarCut(z=0, leaves=frozenset({0, 1, 2})),
            VertexCut(z=7, n=3),
        ]
        x_hat = {0: 0.5, 1: 0.5, 2: 0.5}
        result = filter_by_violation(cuts, x_hat, tol=1e-6)
        assert len(result) == 1
        assert isinstance(result[0], StarCut)


# ---------------------------------------------------------------------------
# top_k_by_efficacy()
# ---------------------------------------------------------------------------

class TestTopKByEfficacy:
    def setup_method(self):
        # VertexCut(z=0): x0+x1+x2 >= 1; StarCut(z=0): x0+x1+x2 >= 2
        # For x_hat=(0.2,0.2,0.2):
        #   Vertex: violation=1-0.6=0.4, norm=sqrt(3) → efficacy=0.4/sqrt(3)
        #   Star:   violation=2-0.6=1.4, norm=sqrt(3) → efficacy=1.4/sqrt(3)
        self.vertex = VertexCut(z=0, n=3)
        self.star   = StarCut(z=0, leaves=frozenset({0, 1, 2}))
        self.x_hat  = {0: 0.2, 1: 0.2, 2: 0.2}

    def test_ranking_order(self):
        cuts = [self.vertex, self.star]
        ranked = top_k_by_efficacy(cuts, self.x_hat, k=2)
        # Star has larger violation → higher efficacy → comes first
        assert isinstance(ranked[0], StarCut)
        assert isinstance(ranked[1], VertexCut)

    def test_k_limits_results(self):
        cuts = [self.vertex, self.star]
        ranked = top_k_by_efficacy(cuts, self.x_hat, k=1)
        assert len(ranked) == 1
        assert isinstance(ranked[0], StarCut)

    def test_k_larger_than_pool(self):
        cuts = [self.vertex, self.star]
        ranked = top_k_by_efficacy(cuts, self.x_hat, k=100)
        assert len(ranked) == 2

    def test_empty_input(self):
        assert top_k_by_efficacy([], {}, k=10) == []


# ---------------------------------------------------------------------------
# dominance_prune()
# ---------------------------------------------------------------------------

class TestDominancePruneEdgeVertex:
    """Edge cuts dominate vertex cuts at both endpoints."""

    def test_edge_removes_both_vertex_cuts(self):
        # EdgeCut(z=4, axis=0): endpoints {4, 5}
        cuts = [
            VertexCut(z=4, n=3),
            VertexCut(z=5, n=3),
            EdgeCut(z=4, axis=0, n=3),
        ]
        result = dominance_prune(cuts)
        types = [type(c) for c in result]
        assert VertexCut not in types
        assert EdgeCut in types

    def test_vertex_survives_without_edge(self):
        cuts = [VertexCut(z=4, n=3), VertexCut(z=5, n=3)]
        result = dominance_prune(cuts)
        assert len(result) == 2  # nothing to dominate them


class TestDominancePruneSquareEdge:
    """Square cuts dominate edge cuts on their boundary."""

    def test_square_removes_all_four_edges(self):
        idx = square_index()
        cuts = cuts_from_index(idx)
        result = dominance_prune(cuts)
        # After pruning: only the SquareCut should remain (no VertexCuts, no EdgeCuts)
        assert all(isinstance(c, SquareCut) for c in result)
        assert len(result) == 1

    def test_square_cut_is_kept(self):
        idx = square_index()
        cuts = cuts_from_index(idx)
        result = dominance_prune(cuts)
        sq = result[0]
        assert sq.z == 4
        assert sq.free == frozenset({0, 1})


class TestDominancePruneCubeSquare:
    """Cube cuts dominate all 6 face squares."""

    def test_full_cube_prune(self):
        idx = full_cube_index()
        cuts = cuts_from_index(idx)
        result = dominance_prune(cuts)
        # Should have: 1 CubeCut + 8 StarCuts (one per vertex)
        cube_cuts = [c for c in result if isinstance(c, CubeCut)]
        star_cuts = [c for c in result if isinstance(c, StarCut)]
        assert len(cube_cuts) == 1
        assert len(star_cuts) == 8
        # No vertex, edge, or square cuts should survive
        assert not any(isinstance(c, VertexCut) for c in result)
        assert not any(isinstance(c, EdgeCut) for c in result)
        assert not any(isinstance(c, SquareCut) for c in result)


class TestDominancePruneStar:
    """Only the maximal leaf-set star is kept per center."""

    def test_max_star_kept(self):
        # Two stars at center=0: one with 3 leaves, one with 2 (subset) → keep 3
        cuts = [
            StarCut(z=0, leaves=frozenset({0, 1, 2})),
            StarCut(z=0, leaves=frozenset({0, 1})),      # subset → dominated
        ]
        result = dominance_prune(cuts)
        assert len(result) == 1
        assert result[0].leaves == frozenset({0, 1, 2})

    def test_different_centers_both_kept(self):
        cuts = [
            StarCut(z=0, leaves=frozenset({0, 1, 2})),
            StarCut(z=7, leaves=frozenset({0, 1, 2})),
        ]
        result = dominance_prune(cuts)
        assert len(result) == 2


class TestDominancePruneTulip:
    """Only the max extra-leaf-set tulip is kept per (center, K)."""

    def test_max_tulip_kept(self):
        K = frozenset({0, 1, 2})
        cuts = [
            TulipCut(z=0, K=K, L=frozenset({3}), n=5),   # more leaves
            TulipCut(z=0, K=K, L=frozenset(), n=5),        # fewer → dominated
        ]
        result = dominance_prune(cuts)
        assert len(result) == 1
        assert result[0].L == frozenset({3})

    def test_different_K_both_kept(self):
        cuts = [
            TulipCut(z=0, K=frozenset({0, 1, 2}), L=frozenset(), n=4),
            TulipCut(z=0, K=frozenset({1, 2, 3}), L=frozenset(), n=4),
        ]
        result = dominance_prune(cuts)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# select_cuts() — end-to-end pipeline
# ---------------------------------------------------------------------------

class TestSelectCuts:
    def test_basic_pipeline(self):
        # Square index: after prune → 1 SquareCut(4, {0,1})
        # SquareCut formula: -x2 >= 0  (x2 ≤ 0)
        # x_hat with x2=0.7 → violated
        idx = square_index()
        cuts = cuts_from_index(idx)
        x_hat = {0: 0.5, 1: 0.5, 2: 0.7}
        result = select_cuts(cuts, x_hat, k=10, tol=1e-6)
        assert len(result) == 1
        assert isinstance(result[0], SquareCut)

    def test_returns_empty_when_nothing_violated(self):
        # SquareCut: -x2 >= 0 → not violated when x2=0
        idx = square_index()
        cuts = cuts_from_index(idx)
        x_hat = {0: 0.5, 1: 0.5, 2: 0.0}
        result = select_cuts(cuts, x_hat, k=10, tol=1e-6)
        assert result == []

    def test_k_cap_respected(self):
        idx = star_index()
        cuts = cuts_from_index(idx)
        # x_hat that violates many cuts
        x_hat = {0: 0.1, 1: 0.1, 2: 0.1}
        result = select_cuts(cuts, x_hat, k=2, tol=1e-6)
        assert len(result) <= 2

    def test_result_ordered_by_efficacy(self):
        # Use star index: StarCut(rhs=2) should have higher efficacy than VertexCut(rhs=1)
        # at x_hat near zero (since both have same ||a|| but star has bigger violation)
        idx = star_index()
        cuts = cuts_from_index(idx)
        x_hat = {0: 0.1, 1: 0.1, 2: 0.1}
        result = select_cuts(cuts, x_hat, k=10, tol=1e-6, prune=False)
        # Compute efficacies and verify they're non-increasing
        effs = [efficacy(c, x_hat) for c in result]
        assert effs == sorted(effs, reverse=True)

    def test_prune_false_includes_dominated(self):
        # Without pruning, vertex cuts should also be returned (if violated)
        idx = square_index()
        cuts = cuts_from_index(idx)
        x_hat = {0: 0.5, 1: 0.5, 2: 0.7}
        result_pruned   = select_cuts(cuts, x_hat, k=100, tol=1e-6, prune=True)
        result_unpruned = select_cuts(cuts, x_hat, k=100, tol=1e-6, prune=False)
        # Unpruned should have at least as many cuts
        assert len(result_unpruned) >= len(result_pruned)

    def test_star_cuts_not_dominated_by_cube(self):
        # In the full cube, star cuts should survive dominance pruning
        idx = full_cube_index()
        cuts = cuts_from_index(idx)
        x_hat = {0: 0.1, 1: 0.1, 2: 0.1}
        result = select_cuts(cuts, x_hat, k=100, tol=1e-6)
        assert any(isinstance(c, StarCut) for c in result)
