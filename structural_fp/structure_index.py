"""
StructureIndex: incremental detection of substructures in the skeleton graph G(X).

Given a set X of visited binary points, the skeleton graph G(X) has a vertex
for every point in X and an edge between z and z^bit(i) whenever both are in X.

Structures detected: vertex, edge, square, star, cube, tulip, propeller.
Reference: Cornuéjols & Lee (2018), Section 3-4 of the spec.
"""

from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Iterator

from .bitops import Point, bit, bits, popcount


# ---------------------------------------------------------------------------
# Square dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Square:
    """
    A 2-face of the unit hypercube.

    base : minimum-index corner (canonical representative).
    dirs : frozenset of exactly 2 direction indices {i, j}.

    The four corners are: base, base^i, base^j, base^ij.
    """
    base: Point
    dirs: frozenset  # frozenset[int], len == 2

    def corners(self) -> list[Point]:
        i, j = sorted(self.dirs)
        b = self.base
        return [b, b ^ bit(i), b ^ bit(j), b ^ bit(i) ^ bit(j)]

    def free_coords(self) -> tuple[int, int]:
        """Return the two free direction indices in sorted order."""
        return tuple(sorted(self.dirs))  # type: ignore[return-value]

    def axes(self) -> list[tuple[Point, int]]:
        """
        Return the four directed edges (z, a) of this square,
        where z is a corner and a is a direction to an adjacent corner.
        """
        i, j = sorted(self.dirs)
        b = self.base
        return [
            (b,           i),
            (b ^ bit(i),  i),
            (b,           j),
            (b ^ bit(j),  j),
        ]


def _canonical_square(p: Point, i: int, j: int) -> Square:
    corners = [p, p ^ bit(i), p ^ bit(j), p ^ bit(i) ^ bit(j)]
    return Square(base=min(corners), dirs=frozenset({i, j}))


# ---------------------------------------------------------------------------
# StructureIndex
# ---------------------------------------------------------------------------

class StructureIndex:
    """
    Maintains the set H of visited binary points and incrementally detects
    substructures as new points are inserted.

    Parameters
    ----------
    n : int
        Number of binary variables (dimension of the hypercube).
    """

    def __init__(self, n: int) -> None:
        self.n = n

        # Core data structures (spec Section 2)
        self.H: set[Point] = set()
        self.D: dict[Point, int] = {}                        # neighbor-direction bitmask
        self.Lz: dict[Point, dict[int, set[int]]] = {}      # link-graph adjacency

        # Detected structures (symbolic)
        self._seen_squares: set[Square] = set()

        # star:  center -> direction bitmask (union of all neighbor directions)
        self.stars: dict[Point, int] = {}

        # cube:  (min_corner, frozenset of 3 dirs) -> frozenset of 3 dirs
        self.cubes: dict[tuple[Point, frozenset], frozenset] = {}

        # tulip: (center, frozenset K of 3 core dirs) -> extra-leaf direction bitmask
        self.tulips: dict[tuple[Point, frozenset], int] = {}

        # propeller: (canonical_z, a) -> blade direction bitmask B  (|B| >= 3)
        self.propellers: dict[tuple[Point, int], int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(self, p: Point) -> None:
        """
        Insert a new visited binary point p.

        Runs the incremental INSERT algorithm from spec Section 4 and
        updates all detected structures in-place.
        """
        if p in self.H:
            return

        self.H.add(p)
        self.D.setdefault(p, 0)
        self.Lz.setdefault(p, {})

        new_squares: list[Square] = []
        affected_centers: set[Point] = {p}

        # 1. New edges -------------------------------------------------------
        for i in range(self.n):
            q = p ^ bit(i)
            if q in self.H:
                self.D[p] |= bit(i)
                self.D[q] |= bit(i)
                affected_centers.add(q)

        # 2. New squares (pairs of directions in D[p]) -----------------------
        dir_list = list(bits(self.D[p]))
        for i, j in combinations(dir_list, 2):
            r = p ^ bit(i) ^ bit(j)
            if r in self.H:
                sq = _canonical_square(p, i, j)
                if sq not in self._seen_squares:
                    self._seen_squares.add(sq)
                    new_squares.append(sq)
                    for corner in sq.corners():
                        lz = self.Lz.setdefault(corner, {})
                        lz.setdefault(i, set()).add(j)
                        lz.setdefault(j, set()).add(i)

        # 3. Stars: recompute for p and its immediate neighbors --------------
        for z in affected_centers:
            if popcount(self.D[z]) >= 3:
                self._upsert_star(z, self.D[z])

        # 4. Tulips & cubes: triggered by new link-graph edges ---------------
        for sq in new_squares:
            for corner in sq.corners():
                i, j = sq.free_coords()
                lz = self.Lz.get(corner, {})
                common = lz.get(i, set()) & lz.get(j, set())
                for k in common:
                    K = frozenset({i, j, k})
                    extra_leaves = set(bits(self.D[corner])) - K
                    self._upsert_tulip(corner, K, extra_leaves)
                    if (corner ^ bit(i) ^ bit(j) ^ bit(k)) in self.H:
                        self._upsert_cube(corner, K)

        # 5. Propellers: triggered by new squares' axes ----------------------
        for sq in new_squares:
            for z, a in sq.axes():
                neighbor = z ^ bit(a)
                if neighbor not in self.H:
                    continue
                B = set(bits(self.D[z] & self.D[neighbor])) - {a}
                if len(B) >= 3:
                    self._upsert_propeller(z, a, B)

    def get_squares(self) -> list[Square]:
        return list(self._seen_squares)

    def get_edges(self) -> list[tuple[Point, Point, int]]:
        """Return undirected edges as (z, neighbor, direction)."""
        seen: set[tuple[Point, Point, int]] = set()
        result = []
        for z, d_mask in self.D.items():
            for i in bits(d_mask):
                nb = z ^ bit(i)
                key = (min(z, nb), max(z, nb), i)
                if key not in seen:
                    seen.add(key)
                    result.append((z, nb, i))
        return result

    # ------------------------------------------------------------------
    # Upsert helpers
    # ------------------------------------------------------------------

    def _upsert_star(self, center: Point, d_mask: int) -> None:
        """Record/update a star: keep maximal leaf-direction bitmask."""
        self.stars[center] = self.stars.get(center, 0) | d_mask

    def _upsert_tulip(
        self, center: Point, K: frozenset, extra_leaves: set[int]
    ) -> None:
        """Record/update a tulip (center, 3-clique K, extra leaves)."""
        key = (center, K)
        extra_mask = sum(bit(i) for i in extra_leaves)
        self.tulips[key] = self.tulips.get(key, 0) | extra_mask

    def _upsert_cube(self, corner: Point, dirs: frozenset) -> None:
        """Record a cube defined by corner and 3 free directions."""
        i, j, k = sorted(dirs)
        b = corner
        all_corners = [
            b, b ^ bit(i), b ^ bit(j), b ^ bit(k),
            b ^ bit(i) ^ bit(j),
            b ^ bit(i) ^ bit(k),
            b ^ bit(j) ^ bit(k),
            b ^ bit(i) ^ bit(j) ^ bit(k),
        ]
        canon = (min(all_corners), dirs)
        self.cubes[canon] = dirs

    def _upsert_propeller(self, z: Point, a: int, B: set[int]) -> None:
        """Record/update a propeller with axis (z, z^a) and blade set B."""
        neighbor = z ^ bit(a)
        canon_z = min(z, neighbor)
        key = (canon_z, a)
        b_mask = sum(bit(i) for i in B)
        self.propellers[key] = self.propellers.get(key, 0) | b_mask
