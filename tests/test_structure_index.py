"""
Hand-computed tests for StructureIndex and the INSERT algorithm.

All examples use n=3 (3 binary variables) so every point fits in 3 bits.
Points are written as 0bXXX where bit 2 = x2, bit 1 = x1, bit 0 = x0.

  0 = 000, 1 = 001, 2 = 010, 3 = 011
  4 = 100, 5 = 101, 6 = 110, 7 = 111
"""

import pytest
from structural_fp import StructureIndex, Square
from structural_fp.bitops import bit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_index(*points: int, n: int = 3) -> StructureIndex:
    idx = StructureIndex(n)
    for p in points:
        idx.insert(p)
    return idx


def edge_set(idx: StructureIndex) -> set[tuple[int, int, int]]:
    """Return edges as a set of (min_z, max_z, direction)."""
    return {
        (min(z, nb), max(z, nb), d)
        for z, nb, d in idx.get_edges()
    }


# ---------------------------------------------------------------------------
# Single point
# ---------------------------------------------------------------------------

class TestSinglePoint:
    def test_H(self):
        idx = make_index(5)
        assert idx.H == {5}

    def test_no_edges(self):
        idx = make_index(5)
        assert idx.get_edges() == []

    def test_no_squares(self):
        idx = make_index(5)
        assert idx.get_squares() == []

    def test_no_stars(self):
        idx = make_index(5)
        assert idx.stars == {}

    def test_idempotent_insert(self):
        idx = make_index(5)
        idx.insert(5)  # inserting again should be a no-op
        assert idx.H == {5}


# ---------------------------------------------------------------------------
# Two adjacent points → one edge
# ---------------------------------------------------------------------------

class TestOneEdge:
    def test_edge_detected(self):
        # 4 (100) and 5 (101) differ in bit 0
        idx = make_index(4, 5)
        assert edge_set(idx) == {(4, 5, 0)}

    def test_D_masks(self):
        idx = make_index(4, 5)
        assert idx.D[4] == bit(0)
        assert idx.D[5] == bit(0)

    def test_no_square(self):
        idx = make_index(4, 5)
        assert idx.get_squares() == []

    def test_two_non_adjacent(self):
        # 0 (000) and 3 (011) differ in bits 0 and 1 — NOT adjacent
        idx = make_index(0, 3)
        assert idx.get_edges() == []
        assert idx.D[0] == 0
        assert idx.D[3] == 0


# ---------------------------------------------------------------------------
# Four points forming a square in directions {0, 1}
#   4 = 100, 5 = 101, 6 = 110, 7 = 111
#   Edges: 4-5 (dir 0), 4-6 (dir 1), 5-7 (dir 1), 6-7 (dir 0)
# ---------------------------------------------------------------------------

class TestSquare:
    def setup_method(self):
        self.idx = make_index(4, 5, 6, 7)

    def test_all_four_edges(self):
        expected = {
            (4, 5, 0),
            (4, 6, 1),
            (5, 7, 1),
            (6, 7, 0),
        }
        assert edge_set(self.idx) == expected

    def test_D_masks(self):
        idx = self.idx
        # each corner has exactly 2 neighbors (directions 0 and 1)
        for z in [4, 5, 6, 7]:
            assert idx.D[z] == bit(0) | bit(1)

    def test_one_square_detected(self):
        squares = self.idx.get_squares()
        assert len(squares) == 1
        sq = squares[0]
        assert sq.base == 4
        assert sq.dirs == frozenset({0, 1})

    def test_square_corners(self):
        sq = self.idx.get_squares()[0]
        assert set(sq.corners()) == {4, 5, 6, 7}

    def test_no_stars(self):
        # Only 2 neighbors per vertex → popcount(D) = 2 < 3
        assert self.idx.stars == {}

    def test_Lz_updated(self):
        lz = self.idx.Lz
        # For each corner, direction 0 links to direction 1 and vice-versa
        for z in [4, 5, 6, 7]:
            assert 1 in lz[z].get(0, set()), f"Lz[{z}][0] missing 1"
            assert 0 in lz[z].get(1, set()), f"Lz[{z}][1] missing 0"


# ---------------------------------------------------------------------------
# Star: center 0 with three leaf-neighbors in directions 0, 1, 2
#   0 = 000, 1 = 001 (dir 0), 2 = 010 (dir 1), 4 = 100 (dir 2)
# ---------------------------------------------------------------------------

class TestStar:
    def setup_method(self):
        self.idx = make_index(0, 1, 2, 4)

    def test_three_edges(self):
        expected = {(0, 1, 0), (0, 2, 1), (0, 4, 2)}
        assert edge_set(self.idx) == expected

    def test_star_at_center(self):
        assert 0 in self.idx.stars
        assert self.idx.stars[0] == bit(0) | bit(1) | bit(2)

    def test_no_star_at_leaves(self):
        for leaf in [1, 2, 4]:
            assert leaf not in self.idx.stars

    def test_no_squares(self):
        # 1,2 differ in 2 bits; 1,4 differ in 2 bits; 2,4 differ in 2 bits
        # — but none of the diagonal points (3,5,6) are in H
        assert self.idx.get_squares() == []


# ---------------------------------------------------------------------------
# Star grows: inserting an extra leaf updates the star's leaf bitmask
# ---------------------------------------------------------------------------

class TestStarGrows:
    def test_star_leaf_added(self):
        idx = make_index(0, 1, 2, 4)  # star with leaves {0,1,2}
        # 0 already has D=7; inserting more points won't add new leaf dirs
        # because n=3 is exhausted. Test the bitmask directly.
        assert idx.stars[0] == 0b111  # all three directions

    def test_star_appears_on_third_neighbor(self):
        # After 0,1,2 the center (0) has only 2 neighbors → no star yet
        idx = make_index(0, 1, 2)
        assert 0 not in idx.stars
        # After adding the third neighbor a star emerges
        idx.insert(4)
        assert 0 in idx.stars


# ---------------------------------------------------------------------------
# Full 3-dimensional hypercube (all 8 points)
# ---------------------------------------------------------------------------

class TestFullCube:
    def setup_method(self):
        self.idx = make_index(*range(8))

    def test_all_12_edges(self):
        # 3-cube has 12 edges
        assert len(self.idx.get_edges()) == 12

    def test_all_6_squares(self):
        # 3-cube has 6 faces (2-faces)
        assert len(self.idx.get_squares()) == 6

    def test_D_masks(self):
        # Every vertex in a 3-cube has degree 3 (all 3 directions)
        for z in range(8):
            assert self.idx.D[z] == 0b111

    def test_star_at_every_vertex(self):
        for z in range(8):
            assert z in self.idx.stars
            assert self.idx.stars[z] == 0b111

    def test_one_cube_detected(self):
        assert len(self.idx.cubes) == 1
        canon_key = list(self.idx.cubes.keys())[0]
        min_corner, dirs = canon_key
        assert min_corner == 0
        assert dirs == frozenset({0, 1, 2})

    def test_no_propellers(self):
        # A propeller needs |B| >= 3 on an axis; max degree in 3-cube is 3,
        # but D[z] & D[z^a] shares only 2 directions after removing a → |B|=2
        assert self.idx.propellers == {}


# ---------------------------------------------------------------------------
# Tulip detection (n=4 example)
#
# To get a tulip we need:
#   center z, three core directions {i,j,k}, so that squares (i,j),(j,k),(i,k)
#   all exist at z. That requires 7 specific points.
#
# Using n=4, center = 0 (0000), core dirs = {0,1,2}:
#   Required points: 0,1,2,4,3(=1^2),5(=1^4),6(=2^4)
#   — but also z^ij=3, z^ik=5, z^jk=6 must be in H (they're same as above)
#   To get Lz[0][0] = {1,2}: squares (0,1) and (0,2) at center 0:
#     square(0,1): points 0,1,2,3 — but 2 (010) and 3 (011) needed
#     square(0,2): points 0,1,4,5 — but 4 (0100) and 5 (0101) needed
#     square(1,2): points 0,2,4,6 — but 2,4,6 needed
#   So minimum point set: {0,1,2,3,4,5,6}  (7 points)
# ---------------------------------------------------------------------------

class TestTulip:
    def setup_method(self):
        # n=4, insert the 7 points needed for a tulip at center=0, K={0,1,2}
        # 0=0000,1=0001,2=0010,3=0011,4=0100,5=0101,6=0110
        self.idx = make_index(0, 1, 2, 3, 4, 5, 6, n=4)

    def test_tulip_detected(self):
        K = frozenset({0, 1, 2})
        assert (0, K) in self.idx.tulips

    def test_no_cube_without_opposite_corner(self):
        # Point 7 (0111 = 0^1^2^4... wait for n=4, 0^bit(0)^bit(1)^bit(2) = 7)
        # 7 is NOT in H → no cube
        K = frozenset({0, 1, 2})
        assert (0, K) not in self.idx.cubes

    def test_cube_after_adding_opposite_corner(self):
        self.idx.insert(7)  # 0^bit(0)^bit(1)^bit(2) = 7
        K = frozenset({0, 1, 2})
        # Now all 8 corners of the sub-cube {0,1,2,3,4,5,6,7} are in H
        assert any(dirs == K for dirs in self.idx.cubes.values())
