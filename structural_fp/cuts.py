r"""
Symbolic cut classes for conv({0,1}^n \ X).

Each class represents a structural cut symbolically (center point + metadata).
Call .to_inequality() to get (coeffs, rhs) in x-space:
    sum_i coeffs[i] * x[i] >= rhs

Reference: spec Section 3 — Structure catalog and cut formulas.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Union

from .bitops import Point, bits

# A cut row in x-space: coefficients dict + rhs scalar.
CutRow = tuple[dict[int, float], float]


# ---------------------------------------------------------------------------
# Core conversion helper
# ---------------------------------------------------------------------------

def _delta_to_x(z: Point, weighted_dirs: dict[int, float], rhs_delta: float) -> CutRow:
    """
    Convert  sum_i w[i] * delta_i(x; z) >= rhs_delta  (delta-space)
    to       sum_i c[i] * x[i]           >= rhs_x      (x-space).

    delta_i(x; z) = z_i*(1-x_i) + (1-z_i)*x_i
      z_i = 0  →  delta_i = x_i          (coefficient +w, constant 0)
      z_i = 1  →  delta_i = 1 - x_i      (coefficient -w, constant +w)
    """
    coeffs: dict[int, float] = {}
    rhs_x = rhs_delta
    for i, w in weighted_dirs.items():
        if (z >> i) & 1:   # z_i = 1
            coeffs[i] = -w
            rhs_x -= w     # move constant w to rhs: ...>= rhs_delta - w
        else:              # z_i = 0
            coeffs[i] = w
    return coeffs, rhs_x


# ---------------------------------------------------------------------------
# Symbolic cut classes  (spec Section 3 table)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VertexCut:
    """
    No-good for a single visited point z.
    sum_{i=0}^{n-1} delta_i(x; z) >= 1
    """
    z: Point
    n: int

    def to_inequality(self) -> CutRow:
        weighted = {i: 1.0 for i in range(self.n)}
        return _delta_to_x(self.z, weighted, 1.0)


@dataclass(frozen=True)
class EdgeCut:
    """
    Cut for an edge (z, z^a).
    sum_{i != a} delta_i(x; z) >= 1
    """
    z: Point
    axis: int   # direction a of the edge
    n: int

    def to_inequality(self) -> CutRow:
        weighted = {i: 1.0 for i in range(self.n) if i != self.axis}
        return _delta_to_x(self.z, weighted, 1.0)


@dataclass(frozen=True)
class SquareCut:
    """
    Cut for a 2-face (square).
    sum_{i not in free} delta_i(x; z) >= 1
    """
    z: Point
    free: frozenset   # {a, b}: the two free directions of the square
    n: int

    def to_inequality(self) -> CutRow:
        weighted = {i: 1.0 for i in range(self.n) if i not in self.free}
        return _delta_to_x(self.z, weighted, 1.0)


@dataclass(frozen=True)
class StarCut:
    """
    Cut for a star (center z with |L| >= 3 leaf directions).
    sum_{i in L} delta_i(x; z) >= |L| - 1
    """
    z: Point
    leaves: frozenset   # L: set of direction indices

    def to_inequality(self) -> CutRow:
        L = self.leaves
        weighted = {i: 1.0 for i in L}
        return _delta_to_x(self.z, weighted, float(len(L) - 1))


@dataclass(frozen=True)
class CubeCut:
    """
    Cut for a 3-cube (8 corners).
    sum_{i not in free} delta_i(x; z) >= 1
    """
    z: Point
    free: frozenset   # {a, b, c}: the three free directions
    n: int

    def to_inequality(self) -> CutRow:
        weighted = {i: 1.0 for i in range(self.n) if i not in self.free}
        return _delta_to_x(self.z, weighted, 1.0)


@dataclass(frozen=True)
class TulipCut:
    """
    Cut for a tulip (center z, 3 core directions K, extra leaves L).
    sum_{i in K}     1 * delta_i
    + sum_{i in L}   2 * delta_i
    + sum_{i not in K|L}  3 * delta_i  >=  3
    """
    z: Point
    K: frozenset   # 3 core directions {a, b, c}
    L: frozenset   # extra leaf directions
    n: int

    def to_inequality(self) -> CutRow:
        KuL = self.K | self.L
        weighted: dict[int, float] = {}
        for i in range(self.n):
            if i in self.K:
                weighted[i] = 1.0
            elif i in self.L:
                weighted[i] = 2.0
            else:
                weighted[i] = 3.0
        return _delta_to_x(self.z, weighted, 3.0)


@dataclass(frozen=True)
class PropellerCut:
    """
    Cut for a propeller (axis (z, z^a), blade set B, |B| >= 3).
    sum_{i in B}         1 * delta_i
    + sum_{i not in B|{a}}  2 * delta_i  >=  2
    (axis direction a is excluded from the sum entirely)
    """
    z: Point      # canonical axis base (min of z, z^a)
    axis: int     # direction a
    blades: frozenset   # B
    n: int

    def to_inequality(self) -> CutRow:
        weighted: dict[int, float] = {}
        for i in range(self.n):
            if i == self.axis:
                continue   # axis direction: coefficient 0
            elif i in self.blades:
                weighted[i] = 1.0
            else:
                weighted[i] = 2.0
        return _delta_to_x(self.z, weighted, 2.0)


SymbolicCut = Union[
    VertexCut, EdgeCut, SquareCut, StarCut, CubeCut, TulipCut, PropellerCut
]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(coeffs: dict[int, float], rhs: float, x: dict[int, float]) -> float:
    """Return LHS - RHS.  Negative → cut is violated by x."""
    return sum(c * x.get(i, 0.0) for i, c in coeffs.items()) - rhs


def is_violated(
    coeffs: dict[int, float], rhs: float, x: dict[int, float], tol: float = 1e-9
) -> bool:
    return evaluate(coeffs, rhs, x) < -tol


# ---------------------------------------------------------------------------
# Extract symbolic cuts from a StructureIndex
# ---------------------------------------------------------------------------

def cuts_from_index(idx) -> list[SymbolicCut]:
    """
    Generate one symbolic cut per detected structure.
    Dominance pruning (edge > vertex, square > edge, etc.) is deferred
    to the cut pool layer (Milestone 3).
    """
    n = idx.n
    result: list[SymbolicCut] = []

    # Vertices
    for z in idx.H:
        result.append(VertexCut(z=z, n=n))

    # Edges — use the lower endpoint as the reference z
    for z, nb, a in idx.get_edges():
        result.append(EdgeCut(z=min(z, nb), axis=a, n=n))

    # Squares
    for sq in idx.get_squares():
        result.append(SquareCut(z=sq.base, free=sq.dirs, n=n))

    # Stars
    for center, d_mask in idx.stars.items():
        result.append(StarCut(z=center, leaves=frozenset(bits(d_mask))))

    # Cubes
    for (min_corner, dirs), _ in idx.cubes.items():
        result.append(CubeCut(z=min_corner, free=dirs, n=n))

    # Tulips
    for (center, K), extra_mask in idx.tulips.items():
        L = frozenset(bits(extra_mask))
        result.append(TulipCut(z=center, K=K, L=L, n=n))

    # Propellers
    for (canon_z, a), b_mask in idx.propellers.items():
        result.append(PropellerCut(z=canon_z, axis=a, blades=frozenset(bits(b_mask)), n=n))

    return result
