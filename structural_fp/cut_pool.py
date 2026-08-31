r"""
Cut pool: violation filtering, efficacy ranking, and dominance pruning.

Given symbolic cuts from cuts_from_index() and a fractional LP solution x_hat,
this module selects the most useful cuts to add to the LP.

Pipeline (select_cuts):
  1. dominance_prune  — remove cuts made redundant by stronger ones
  2. filter_by_violation — keep only cuts violated at x_hat
  3. top_k_by_efficacy — rank by normalized violation, return top k

Reference: spec Section 5.
"""

from __future__ import annotations
import math

from .bitops import bit, bits
from .cuts import (
    CutRow, SymbolicCut,
    VertexCut, EdgeCut, SquareCut, StarCut,
    CubeCut, TulipCut, PropellerCut,
    is_violated,
)


# ---------------------------------------------------------------------------
# Efficacy
# ---------------------------------------------------------------------------

def efficacy(cut: SymbolicCut, x_hat: dict[int, float]) -> float:
    """
    efficacy = (rhs - a^T x_hat) / ||a||_2

    Measures how deeply x_hat violates the cut, normalized by cut length.
    Higher = more violated = higher priority to add.
    Returns 0.0 when the cut is not violated or the coefficient vector is zero.
    """
    coeffs, rhs = cut.to_inequality()
    lhs = sum(c * x_hat.get(i, 0.0) for i, c in coeffs.items())
    violation = rhs - lhs
    if violation <= 0.0:
        return 0.0
    norm = math.sqrt(sum(c * c for c in coeffs.values()))
    return violation / norm if norm > 1e-12 else 0.0


# ---------------------------------------------------------------------------
# Filtering and ranking
# ---------------------------------------------------------------------------

def filter_by_violation(
    cuts: list[SymbolicCut],
    x_hat: dict[int, float],
    tol: float = 1e-6,
) -> list[SymbolicCut]:
    """Return only cuts for which a^T x_hat < rhs - tol."""
    return [c for c in cuts if is_violated(*c.to_inequality(), x_hat, tol=tol)]


def top_k_by_efficacy(
    cuts: list[SymbolicCut],
    x_hat: dict[int, float],
    k: int,
) -> list[SymbolicCut]:
    """Return up to k cuts ordered by efficacy (highest first)."""
    ranked = sorted(cuts, key=lambda c: efficacy(c, x_hat), reverse=True)
    return ranked[:k]


# ---------------------------------------------------------------------------
# Dominance pruning helpers
# ---------------------------------------------------------------------------

def _square_corners(sc: SquareCut) -> frozenset:
    i, j = sorted(sc.free)
    b = sc.z
    return frozenset([b, b ^ bit(i), b ^ bit(j), b ^ bit(i) ^ bit(j)])


def _cube_corners(cc: CubeCut) -> frozenset:
    a, b_dir, c_dir = sorted(cc.free)
    base = cc.z
    return frozenset([
        base,
        base ^ bit(a), base ^ bit(b_dir), base ^ bit(c_dir),
        base ^ bit(a) ^ bit(b_dir),
        base ^ bit(a) ^ bit(c_dir),
        base ^ bit(b_dir) ^ bit(c_dir),
        base ^ bit(a) ^ bit(b_dir) ^ bit(c_dir),
    ])


# ---------------------------------------------------------------------------
# Dominance pruning
# ---------------------------------------------------------------------------

def dominance_prune(cuts: list[SymbolicCut]) -> list[SymbolicCut]:
    """
    Remove cuts that are dominated by stronger ones in the pool.

    Dominance hierarchy (stronger cuts make weaker ones redundant):
      Cube  >  Square  >  Edge  >  Vertex
      Star, Tulip, Propeller: keep only the maximal structure per anchor.

    A cut A dominates B when: every x satisfying A also satisfies B
    (i.e., A's feasible region ⊆ B's feasible region).

    Proof sketch for hierarchy:
      EdgeCut   removes axis direction a → LHS_edge ≤ LHS_vertex, but
                if LHS_edge >= 1 then LHS_vertex = LHS_edge + delta_a >= 1.
      SquareCut removes two directions → dominates both edge cuts on those axes.
      CubeCut   removes three directions → dominates all 6 face squares.
    """
    dominated: set[int] = set()  # id() of dominated cut objects

    vertex_cuts    = [c for c in cuts if isinstance(c, VertexCut)]
    edge_cuts      = [c for c in cuts if isinstance(c, EdgeCut)]
    square_cuts    = [c for c in cuts if isinstance(c, SquareCut)]
    cube_cuts      = [c for c in cuts if isinstance(c, CubeCut)]
    star_cuts      = [c for c in cuts if isinstance(c, StarCut)]
    tulip_cuts     = [c for c in cuts if isinstance(c, TulipCut)]
    propeller_cuts = [c for c in cuts if isinstance(c, PropellerCut)]

    # --- Edge > Vertex ---
    # An EdgeCut(z, a) dominates VertexCut at both endpoints (z and z^a),
    # because delta_a >= 0 means the edge's LHS ≤ vertex's LHS.
    edge_endpoints: set[int] = set()
    for ec in edge_cuts:
        edge_endpoints.add(ec.z)
        edge_endpoints.add(ec.z ^ bit(ec.axis))

    for vc in vertex_cuts:
        if vc.z in edge_endpoints:
            dominated.add(id(vc))

    # --- Square > Edge ---
    # SquareCut(base, {i,j}) dominates EdgeCut(z, a) when:
    #   z is a corner of the square  AND  a ∈ {i, j}
    # (removing both i and j is stronger than removing just one)
    sq_corner_map: dict[int, list[SquareCut]] = {}
    for sc in square_cuts:
        for corner in _square_corners(sc):
            sq_corner_map.setdefault(corner, []).append(sc)

    for ec in edge_cuts:
        for ep in (ec.z, ec.z ^ bit(ec.axis)):
            for sc in sq_corner_map.get(ep, []):
                if ec.axis in sc.free:
                    dominated.add(id(ec))
                    break
            if id(ec) in dominated:
                break

    # --- Cube > Square ---
    # CubeCut(base, {a,b,c}) dominates SquareCut(z, {i,j}) when:
    #   z is a corner of the cube  AND  {i,j} ⊆ {a,b,c}
    cube_corner_map: dict[int, list[CubeCut]] = {}
    for cc in cube_cuts:
        for corner in _cube_corners(cc):
            cube_corner_map.setdefault(corner, []).append(cc)

    for sc in square_cuts:
        for cc in cube_corner_map.get(sc.z, []):
            if sc.free <= cc.free:
                dominated.add(id(sc))
                break

    # --- Star: keep only the max-leaf-set per center ---
    star_best: dict[int, StarCut] = {}
    for sc in star_cuts:
        prev = star_best.get(sc.z)
        if prev is None or len(sc.leaves) > len(prev.leaves):
            star_best[sc.z] = sc
    for sc in star_cuts:
        if sc is not star_best[sc.z]:
            dominated.add(id(sc))

    # --- Tulip: keep max extra-leaf set per (center, K) ---
    tulip_best: dict[tuple, TulipCut] = {}
    for tc in tulip_cuts:
        key = (tc.z, tc.K)
        prev = tulip_best.get(key)
        if prev is None or len(tc.L) > len(prev.L):
            tulip_best[key] = tc
    for tc in tulip_cuts:
        if tc is not tulip_best[(tc.z, tc.K)]:
            dominated.add(id(tc))

    # --- Propeller: keep max blade set per (axis_base, axis_dir) ---
    prop_best: dict[tuple, PropellerCut] = {}
    for pc in propeller_cuts:
        key = (pc.z, pc.axis)
        prev = prop_best.get(key)
        if prev is None or len(pc.blades) > len(prev.blades):
            prop_best[key] = pc
    for pc in propeller_cuts:
        if pc is not prop_best[(pc.z, pc.axis)]:
            dominated.add(id(pc))

    return [c for c in cuts if id(c) not in dominated]


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def select_cuts(
    cuts: list[SymbolicCut],
    x_hat: dict[int, float],
    k: int = 30,
    tol: float = 1e-6,
    prune: bool = True,
) -> list[SymbolicCut]:
    """
    Full cut-selection pipeline.

    Parameters
    ----------
    cuts  : all symbolic cuts from cuts_from_index()
    x_hat : current fractional LP solution  {var_index: float}
    k     : max cuts to return (spec recommends 10–50)
    tol   : violation tolerance (spec recommends 1e-6)
    prune : apply dominance pruning before filtering (default True)

    Returns
    -------
    Up to k cuts that are violated at x_hat, ranked by efficacy.
    """
    pool = dominance_prune(cuts) if prune else cuts
    violated = filter_by_violation(pool, x_hat, tol=tol)
    return top_k_by_efficacy(violated, x_hat, k=k)
