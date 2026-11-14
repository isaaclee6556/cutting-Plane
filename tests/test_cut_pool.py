"""
Unit tests for cut_pool.py (Milestone 3).

Tests are organized around:
  - dominance_prune()
  - select_cuts()  (end-to-end pipeline: dominance_prune, keep every survivor)

All fractional points are synthetic; no Gurobi required.
"""

from structural_fp import StructureIndex
from structural_fp.cuts import (
    VertexCut, EdgeCut, SquareCut, StarCut, CubeCut, TulipCut,
    cuts_from_index,
)
from structural_fp.cut_pool import dominance_prune, select_cuts


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


def embedded_cube_index(n=5, free=(0, 1, 2)) -> StructureIndex:
    """StructureIndex with one 3-cube spanning `free`, embedded in a larger
    space (n > len(free)) so the cube's own inequality is a real,
    non-vacuous constraint (at least one direction stays outside `free`)."""
    idx = StructureIndex(n=n)
    for combo in range(8):
        p = 0
        for j, pos in enumerate(free):
            if (combo >> j) & 1:
                p |= 1 << pos
        idx.insert(p)
    return idx


def full_cube_index(n=3) -> StructureIndex:
    """StructureIndex with all 8 points, n=3."""
    idx = StructureIndex(n=n)
    for p in range(8):
        idx.insert(p)
    return idx


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


class TestDominancePruneCubeStarTulip:
    """
    Cube cuts dominate Star/Tulip cuts anchored at their own corners, when
    the star's/tulip's directions are confined to the cube's free dirs.

    This guards against fp_loop's face-probe step: inserting all 2^k
    corners of a face at once (face_k=3) simultaneously re-detects the
    identical cube AND forms a Star + Tulip at every one of those corners
    (each corner has exactly 3 freshly-created edges), so without this
    rule a single face-probe cut would be shadowed by up to 16 redundant
    weaker cuts.
    """

    def test_embedded_cube_prunes_its_own_star_and_tulip(self):
        idx = embedded_cube_index(n=5, free=(0, 1, 2))
        cuts = cuts_from_index(idx)
        # Sanity: the raw (unpruned) pool really does contain the echo
        # this rule is meant to remove.
        assert any(isinstance(c, StarCut) for c in cuts)
        assert any(isinstance(c, TulipCut) for c in cuts)

        result = dominance_prune(cuts)
        cube_cuts = [c for c in result if isinstance(c, CubeCut)]
        assert len(cube_cuts) == 1
        assert not any(isinstance(c, StarCut) for c in result)
        assert not any(isinstance(c, TulipCut) for c in result)

    def test_star_with_leaf_outside_free_survives(self):
        # Star's leaves are not a subset of the cube's free dirs -> the
        # dominance proof doesn't apply, so it must survive.
        cube = CubeCut(z=0, free=frozenset({0, 1, 2}), n=5)
        star = StarCut(z=0, leaves=frozenset({0, 1, 3}), n=5)
        result = dominance_prune([cube, star])
        assert any(isinstance(c, StarCut) for c in result)

    def test_tulip_anchored_off_cube_survives(self):
        # Tulip's anchor disagrees with the cube's base outside free (bit 3
        # differs) -> not one of the cube's own corners, so it must survive.
        cube = CubeCut(z=0, free=frozenset({0, 1, 2}), n=5)
        tulip = TulipCut(z=0b01000, K=frozenset({0, 1, 2}), L=frozenset(), n=5)
        result = dominance_prune([cube, tulip])
        assert any(isinstance(c, TulipCut) for c in result)

    def test_degenerate_full_free_cube_keeps_stars(self):
        # free spans every direction (n == len(free)): cc's inequality is
        # the unsatisfiable "0 >= 1", so it must not be used to prune
        # anything (also covered by TestDominancePruneCubeSquare above).
        idx = full_cube_index()  # n=3, free={0,1,2}
        cuts = cuts_from_index(idx)
        result = dominance_prune(cuts)
        assert any(isinstance(c, StarCut) for c in result)


class TestDominancePruneStar:
    """Only the maximal leaf-set star is kept per center."""

    def test_max_star_kept(self):
        # Two stars at center=0: one with 3 leaves, one with 2 (subset) → keep 3
        cuts = [
            StarCut(z=0, leaves=frozenset({0, 1, 2}), n=3),
            StarCut(z=0, leaves=frozenset({0, 1}), n=3),      # subset → dominated
        ]
        result = dominance_prune(cuts)
        assert len(result) == 1
        assert result[0].leaves == frozenset({0, 1, 2})

    def test_different_centers_both_kept(self):
        cuts = [
            StarCut(z=0, leaves=frozenset({0, 1, 2}), n=3),
            StarCut(z=7, leaves=frozenset({0, 1, 2}), n=3),
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
# select_cuts() — end-to-end pipeline (dominance_prune, keep every survivor)
# ---------------------------------------------------------------------------

class TestSelectCuts:
    def test_basic_pipeline(self):
        # Square index: after prune → 1 SquareCut(4, {0,1}); no vertex/edge
        # cuts survive since the square dominates all of them.
        idx = square_index()
        cuts = cuts_from_index(idx)
        result = select_cuts(cuts)
        assert len(result) == 1
        assert isinstance(result[0], SquareCut)

    def test_matches_dominance_prune(self):
        idx = star_index()
        cuts = cuts_from_index(idx)
        assert select_cuts(cuts) == dominance_prune(cuts)

    def test_prune_false_includes_dominated(self):
        # Without pruning, vertex/edge cuts are also returned.
        idx = square_index()
        cuts = cuts_from_index(idx)
        result_pruned   = select_cuts(cuts, prune=True)
        result_unpruned = select_cuts(cuts, prune=False)
        assert result_unpruned == cuts
        assert len(result_unpruned) > len(result_pruned)

    def test_star_cuts_not_dominated_by_cube(self):
        # In the full cube, star cuts should survive dominance pruning
        idx = full_cube_index()
        cuts = cuts_from_index(idx)
        result = select_cuts(cuts)
        assert any(isinstance(c, StarCut) for c in result)
