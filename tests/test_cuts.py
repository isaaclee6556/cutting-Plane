"""
Unit tests for cut formula generation (Milestone 2).

All coefficients and rhs values are verified by hand against the spec Section 3 table.
Convention: x is represented as a dict {variable_index: value}.
"""

import pytest
from structural_fp.cuts import (
    VertexCut, EdgeCut, SquareCut, StarCut, CubeCut, TulipCut, PropellerCut,
    evaluate, is_violated, cuts_from_index,
)
from structural_fp import StructureIndex


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def lhs(coeffs, x: dict) -> float:
    return sum(c * x.get(i, 0) for i, c in coeffs.items())


def assert_satisfies(coeffs, rhs, x: dict):
    assert lhs(coeffs, x) >= rhs - 1e-9, (
        f"Expected satisfied but violated: lhs={lhs(coeffs,x):.4f} rhs={rhs} x={x}"
    )


def assert_violates(coeffs, rhs, x: dict):
    assert lhs(coeffs, x) < rhs - 1e-9, (
        f"Expected violated but satisfied: lhs={lhs(coeffs,x):.4f} rhs={rhs} x={x}"
    )


# ---------------------------------------------------------------------------
# _delta_to_x conversion (tested implicitly through each cut)
# ---------------------------------------------------------------------------

# z=0 (000): all z_i=0  →  delta_i = x_i  (positive coefficients)
# z=7 (111): all z_i=1  →  delta_i = 1-x_i (negative coefficients, offset in rhs)
# z=5 (101): mixed


# ---------------------------------------------------------------------------
# VertexCut
# ---------------------------------------------------------------------------

class TestVertexCut:
    def test_z_all_zeros(self):
        # z=0 (000): delta_i=x_i → x0+x1+x2 >= 1
        coeffs, rhs = VertexCut(z=0, n=3).to_inequality()
        assert coeffs == {0: 1.0, 1: 1.0, 2: 1.0}
        assert rhs == pytest.approx(1.0)

    def test_z_all_ones(self):
        # z=7 (111): delta_i=1-x_i → -x0-x1-x2+3 >= 1 → -x0-x1-x2 >= -2
        coeffs, rhs = VertexCut(z=7, n=3).to_inequality()
        assert coeffs == {0: -1.0, 1: -1.0, 2: -1.0}
        assert rhs == pytest.approx(-2.0)

    def test_z_mixed(self):
        # z=5 (101): z0=1,z1=0,z2=1 → -x0+x1-x2+2 >= 1 → -x0+x1-x2 >= -1
        coeffs, rhs = VertexCut(z=5, n=3).to_inequality()
        assert coeffs == {0: -1.0, 1: 1.0, 2: -1.0}
        assert rhs == pytest.approx(-1.0)

    def test_z_violates_own_cut(self):
        # z must violate its own vertex cut
        coeffs, rhs = VertexCut(z=5, n=3).to_inequality()
        assert_violates(coeffs, rhs, {0: 1, 1: 0, 2: 1})

    def test_other_points_satisfy(self):
        coeffs, rhs = VertexCut(z=0, n=3).to_inequality()
        for x_int in range(1, 8):  # all binary points except z=0
            x = {i: (x_int >> i) & 1 for i in range(3)}
            assert_satisfies(coeffs, rhs, x)


# ---------------------------------------------------------------------------
# EdgeCut
# ---------------------------------------------------------------------------

class TestEdgeCut:
    def test_coefficients(self):
        # z=4 (100), axis=0, n=3 → directions {1,2}
        # z1=0→+1, z2=1→-1; rhs: 1.0 - 1.0 = 0.0
        coeffs, rhs = EdgeCut(z=4, axis=0, n=3).to_inequality()
        assert coeffs == {1: 1.0, 2: -1.0}
        assert rhs == pytest.approx(0.0)

    def test_axis_excluded(self):
        # direction 0 must NOT appear in coefficients
        coeffs, _ = EdgeCut(z=4, axis=0, n=3).to_inequality()
        assert 0 not in coeffs

    def test_both_endpoints_violate(self):
        # Edge (4,5) via direction 0 → both endpoints in X → both violate
        coeffs, rhs = EdgeCut(z=4, axis=0, n=3).to_inequality()
        assert_violates(coeffs, rhs, {0: 0, 1: 0, 2: 1})  # z=4 (100): x1=0,x2=1
        assert_violates(coeffs, rhs, {0: 1, 1: 0, 2: 1})  # z^0=5 (101): same x1,x2

    def test_non_edge_points_satisfy(self):
        coeffs, rhs = EdgeCut(z=4, axis=0, n=3).to_inequality()
        assert_satisfies(coeffs, rhs, {0: 0, 1: 1, 2: 0})  # (010): 1-0=1 >= 0


# ---------------------------------------------------------------------------
# SquareCut
# ---------------------------------------------------------------------------

class TestSquareCut:
    def test_coefficients(self):
        # z=4 (100), free={0,1}, n=3 → only direction 2 remains
        # z2=1 → coeff=-1; rhs: 1.0 - 1.0 = 0.0
        coeffs, rhs = SquareCut(z=4, free=frozenset({0, 1}), n=3).to_inequality()
        assert coeffs == {2: -1.0}
        assert rhs == pytest.approx(0.0)

    def test_free_dirs_excluded(self):
        coeffs, _ = SquareCut(z=4, free=frozenset({0, 1}), n=3).to_inequality()
        assert 0 not in coeffs and 1 not in coeffs

    def test_all_four_corners_violate(self):
        # Square corners {4,5,6,7} all have x2=1 → -1 < 0 → violate
        coeffs, rhs = SquareCut(z=4, free=frozenset({0, 1}), n=3).to_inequality()
        for z_int in [4, 5, 6, 7]:
            x = {i: (z_int >> i) & 1 for i in range(3)}
            assert_violates(coeffs, rhs, x)

    def test_complement_points_satisfy(self):
        # Points with x2=0 satisfy the cut
        coeffs, rhs = SquareCut(z=4, free=frozenset({0, 1}), n=3).to_inequality()
        for z_int in [0, 1, 2, 3]:
            x = {i: (z_int >> i) & 1 for i in range(3)}
            assert_satisfies(coeffs, rhs, x)


# ---------------------------------------------------------------------------
# StarCut
# ---------------------------------------------------------------------------

class TestStarCut:
    def test_coefficients_z0_L012(self):
        # z=0 (000), L={0,1,2}: delta_i=x_i → x0+x1+x2 >= 2
        coeffs, rhs = StarCut(z=0, leaves=frozenset({0, 1, 2})).to_inequality()
        assert coeffs == {0: 1.0, 1: 1.0, 2: 1.0}
        assert rhs == pytest.approx(2.0)

    def test_rhs_scales_with_L(self):
        # |L|=4 → rhs = 3.0
        coeffs, rhs = StarCut(z=0, leaves=frozenset({0, 1, 2, 3})).to_inequality()
        assert rhs == pytest.approx(3.0)

    def test_center_and_all_leaves_violate(self):
        # Center z=0 and each single-leaf neighbor must violate
        coeffs, rhs = StarCut(z=0, leaves=frozenset({0, 1, 2})).to_inequality()
        for z_int in [0, 1, 2, 4]:  # 0, 0^bit0, 0^bit1, 0^bit2
            x = {i: (z_int >> i) & 1 for i in range(3)}
            assert_violates(coeffs, rhs, x)

    def test_two_leaf_points_satisfy(self):
        # Any binary point with >=2 bits set satisfies
        coeffs, rhs = StarCut(z=0, leaves=frozenset({0, 1, 2})).to_inequality()
        for z_int in [3, 5, 6, 7]:  # two or three bits set
            x = {i: (z_int >> i) & 1 for i in range(3)}
            assert_satisfies(coeffs, rhs, x)

    def test_z_mixed_star(self):
        # z=7 (111), L={0,1,2}: delta_i=1-x_i → -(x0+x1+x2)+3 >= 2 → -(x0+x1+x2) >= -1
        coeffs, rhs = StarCut(z=7, leaves=frozenset({0, 1, 2})).to_inequality()
        assert coeffs == {0: -1.0, 1: -1.0, 2: -1.0}
        assert rhs == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# CubeCut
# ---------------------------------------------------------------------------

class TestCubeCut:
    def test_coefficients_n4(self):
        # z=0 (0000), free={0,1,2}, n=4 → only direction 3 remains
        # z3=0 → coeff=+1; rhs=1.0
        coeffs, rhs = CubeCut(z=0, free=frozenset({0, 1, 2}), n=4).to_inequality()
        assert coeffs == {3: 1.0}
        assert rhs == pytest.approx(1.0)

    def test_free_dirs_excluded(self):
        coeffs, _ = CubeCut(z=0, free=frozenset({0, 1, 2}), n=4).to_inequality()
        for d in [0, 1, 2]:
            assert d not in coeffs

    def test_all_8_cube_corners_violate(self):
        # n=4, cube embedded in directions {0,1,2}, z=0
        # All 8 corners have x3=0 → coefficient 1 * 0 = 0 < 1 → violate
        coeffs, rhs = CubeCut(z=0, free=frozenset({0, 1, 2}), n=4).to_inequality()
        for z_int in range(8):          # all 3-bit combos → x3=0
            x = {i: (z_int >> i) & 1 for i in range(4)}
            assert_violates(coeffs, rhs, x)

    def test_opposite_half_satisfies(self):
        # Points with x3=1 satisfy the cut
        coeffs, rhs = CubeCut(z=0, free=frozenset({0, 1, 2}), n=4).to_inequality()
        for z_int in range(8, 16):      # all 4-bit combos with bit3=1
            x = {i: (z_int >> i) & 1 for i in range(4)}
            assert_satisfies(coeffs, rhs, x)


# ---------------------------------------------------------------------------
# TulipCut
# ---------------------------------------------------------------------------

class TestTulipCut:
    def setup_method(self):
        # z=0, K={0,1,2}, L={}, n=4
        # weights: i∈K→1, i∉K∪L={3}→3
        # x0+x1+x2+3*x3 >= 3
        self.coeffs, self.rhs = TulipCut(
            z=0, K=frozenset({0, 1, 2}), L=frozenset(), n=4
        ).to_inequality()

    def test_coefficients(self):
        assert self.coeffs == {0: 1.0, 1: 1.0, 2: 1.0, 3: 3.0}
        assert self.rhs == pytest.approx(3.0)

    def test_all_7_tulip_points_violate(self):
        # Required 7 points: 0,1,2,4,3,5,6 (all with x3=0)
        for z_int in [0, 1, 2, 4, 3, 5, 6]:
            x = {i: (z_int >> i) & 1 for i in range(4)}
            assert_violates(self.coeffs, self.rhs, x)

    def test_points_with_x3_satisfy(self):
        # Any point with x3=1 → contribution 3 >= 3 → satisfied
        for z_int in range(8, 16):
            x = {i: (z_int >> i) & 1 for i in range(4)}
            assert_satisfies(self.coeffs, self.rhs, x)

    def test_point_7_satisfies(self):
        # x=7 (0111, x3=0): x0+x1+x2=3 >= 3 → satisfied (boundary)
        x = {0: 1, 1: 1, 2: 1, 3: 0}
        assert_satisfies(self.coeffs, self.rhs, x)

    def test_with_extra_leaves(self):
        # z=0, K={0,1,2}, L={3}, n=5
        # weights: 0,1,2→1; 3→2; 4→3
        # x0+x1+x2+2*x3+3*x4 >= 3
        coeffs, rhs = TulipCut(
            z=0, K=frozenset({0, 1, 2}), L=frozenset({3}), n=5
        ).to_inequality()
        assert coeffs == {0: 1.0, 1: 1.0, 2: 1.0, 3: 2.0, 4: 3.0}
        assert rhs == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# PropellerCut
# ---------------------------------------------------------------------------

class TestPropellerCut:
    def setup_method(self):
        # z=0, axis=0, blades={1,2,3}, n=4
        # i=0 excluded; i=1,2,3 in B→coeff 1; no remaining dirs
        # x1+x2+x3 >= 2
        self.coeffs, self.rhs = PropellerCut(
            z=0, axis=0, blades=frozenset({1, 2, 3}), n=4
        ).to_inequality()

    def test_coefficients(self):
        assert self.coeffs == {1: 1.0, 2: 1.0, 3: 1.0}
        assert self.rhs == pytest.approx(2.0)

    def test_axis_excluded(self):
        assert 0 not in self.coeffs

    def test_axis_and_single_blade_violate(self):
        # Axis endpoints z=0 and z^axis=1 both have x1=x2=x3=0 → 0 < 2
        assert_violates(self.coeffs, self.rhs, {0: 0, 1: 0, 2: 0, 3: 0})  # z=0
        assert_violates(self.coeffs, self.rhs, {0: 1, 1: 0, 2: 0, 3: 0})  # z^0=1

    def test_two_blades_active_satisfies(self):
        # x with x1=x2=1, x3=0 → 2 >= 2 → satisfied
        assert_satisfies(self.coeffs, self.rhs, {0: 0, 1: 1, 2: 1, 3: 0})

    def test_with_remaining_directions(self):
        # z=0, axis=0, blades={1,2,3}, n=5
        # i=4: not in B∪{0} → coeff 2
        # x1+x2+x3+2*x4 >= 2
        coeffs, rhs = PropellerCut(
            z=0, axis=0, blades=frozenset({1, 2, 3}), n=5
        ).to_inequality()
        assert coeffs == {1: 1.0, 2: 1.0, 3: 1.0, 4: 2.0}
        assert rhs == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# evaluate / is_violated helpers
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_positive_slack(self):
        assert evaluate({0: 1.0}, 1.0, {0: 2.0}) == pytest.approx(1.0)

    def test_exactly_satisfied(self):
        assert evaluate({0: 1.0}, 1.0, {0: 1.0}) == pytest.approx(0.0)

    def test_violated(self):
        assert evaluate({0: 1.0}, 1.0, {0: 0.0}) == pytest.approx(-1.0)

    def test_is_violated_true(self):
        assert is_violated({0: 1.0}, 1.0, {0: 0.5})

    def test_is_violated_false_on_boundary(self):
        assert not is_violated({0: 1.0}, 1.0, {0: 1.0})


# ---------------------------------------------------------------------------
# cuts_from_index integration
# ---------------------------------------------------------------------------

class TestCutsFromIndex:
    def test_single_point_gives_one_vertex_cut(self):
        idx = StructureIndex(n=3)
        idx.insert(5)
        cuts = cuts_from_index(idx)
        assert len([c for c in cuts if isinstance(c, VertexCut)]) == 1

    def test_square_index_cut_counts(self):
        idx = StructureIndex(n=3)
        for p in [4, 5, 6, 7]:
            idx.insert(p)
        cuts = cuts_from_index(idx)
        assert len([c for c in cuts if isinstance(c, VertexCut)]) == 4
        assert len([c for c in cuts if isinstance(c, EdgeCut)]) == 4
        assert len([c for c in cuts if isinstance(c, SquareCut)]) == 1

    def test_star_index_cut_counts(self):
        idx = StructureIndex(n=3)
        for p in [0, 1, 2, 4]:
            idx.insert(p)
        cuts = cuts_from_index(idx)
        assert len([c for c in cuts if isinstance(c, StarCut)]) == 1
        star = [c for c in cuts if isinstance(c, StarCut)][0]
        assert star.z == 0
        assert star.leaves == frozenset({0, 1, 2})

    def test_full_cube_cut_counts(self):
        idx = StructureIndex(n=3)
        for p in range(8):
            idx.insert(p)
        cuts = cuts_from_index(idx)
        assert len([c for c in cuts if isinstance(c, CubeCut)]) == 1
        assert len([c for c in cuts if isinstance(c, SquareCut)]) == 6
        assert len([c for c in cuts if isinstance(c, StarCut)]) == 8

    def test_tulip_index_cut_counts(self):
        idx = StructureIndex(n=4)
        for p in [0, 1, 2, 3, 4, 5, 6]:
            idx.insert(p)
        cuts = cuts_from_index(idx)
        assert len([c for c in cuts if isinstance(c, TulipCut)]) >= 1
