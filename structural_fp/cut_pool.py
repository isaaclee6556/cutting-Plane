r"""
Cut pool: dominance pruning.

Given symbolic cuts from cuts_from_index(), this module removes cuts made
redundant by stronger ones; every surviving cut is added to the LP.

Reference: spec Section 5.
"""

from __future__ import annotations

from .bitops import bit, bits
from .cuts import (
    CutRow, SymbolicCut,
    VertexCut, EdgeCut, SquareCut, StarCut,
    CubeCut, TulipCut, PropellerCut,
)


# ---------------------------------------------------------------------------
# Dominance pruning helpers
# ---------------------------------------------------------------------------

def _square_corners(sc: SquareCut) -> frozenset:
    i, j = sorted(sc.free)
    b = sc.z
    return frozenset([b, b ^ bit(i), b ^ bit(j), b ^ bit(i) ^ bit(j)])


def _cube_corners(cc: CubeCut) -> frozenset:
    """
    All 2**|free| corners of the face spanned by `free`, anchored at `z`.

    Generalized over the number of free directions (not just 3) so a
    face-probe CubeCut with |free| = k (any k, e.g. 4) can be dominance-
    checked the same way as a native 3-dim cube from StructureIndex.
    """
    dirs = sorted(cc.free)
    corners = [cc.z]
    for d in dirs:
        corners = corners + [c ^ bit(d) for c in corners]
    return frozenset(corners)


def _agrees_outside(p: int, q: int, dirs: frozenset, n: int) -> bool:
    """True if p and q have identical bits at every position not in `dirs`."""
    mask = 0
    for i in dirs:
        mask |= bit(i)
    outside = ((1 << n) - 1) & ~mask
    return (p & outside) == (q & outside)


# ---------------------------------------------------------------------------
# Dominance pruning
# ---------------------------------------------------------------------------

def dominance_prune(cuts: list[SymbolicCut]) -> list[SymbolicCut]:
    """
    Remove cuts that are dominated by stronger ones in the pool.

    Dominance hierarchy (stronger cuts make weaker ones redundant):
      Cube  >  Square  >  Edge  >  Vertex
      Cube  >  Star, Tulip anchored at one of the cube's own corners
               (only when the star's leaves / the tulip's K|L lie entirely
               within the cube's free directions — see proof below).
      Star, Tulip, Propeller: keep only the maximal structure per anchor.

    A cut A dominates B when: every x satisfying A also satisfies B
    (i.e., A's feasible region ⊆ B's feasible region).

    Proof sketch for hierarchy:
      EdgeCut   removes axis direction a → LHS_edge ≤ LHS_vertex, but
                if LHS_edge >= 1 then LHS_vertex = LHS_edge + delta_a >= 1.
      SquareCut removes two directions → dominates both edge cuts on those axes.
      CubeCut   removes three directions → dominates all 6 face squares.
      Cube > Star/Tulip: let cc = CubeCut(base, free). A StarCut/TulipCut
      anchored at corner c with c agreeing with base outside `free` (i.e. c
      is one of the cube's own 2^|free| corners), and whose leaves (Star: L;
      Tulip: K ∪ L) are a subset of `free`, has every direction outside
      `free` in its *unweighted* (highest-coefficient) group. Since
      delta_i(x;c) == delta_i(x;base) for i outside free (c and base share
      those bits), that group's sum is >= cc's own LHS, which is >= 1 by
      cc's inequality. Star weights that group by 2 (rhs 2) and Tulip by 3
      (rhs 3), so 2*1 >= 2 and 3*1 >= 3 — the weaker cut's inequality holds
      automatically whenever cc's does. This matters because fp_loop's
      face-probe step inserts all 2^k corners of a face at once: besides
      re-detecting the same cube, that simultaneously satisfies the Star
      trigger (popcount(D) >= 3) and the Tulip trigger at every one of the
      corners it just added, so without this rule each face-probe cut would
      be shadowed by up to 2*2^k redundant Star/Tulip cuts.
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

    # --- Cube > Star/Tulip anchored at one of the cube's own corners ---
    # See the proof in the docstring above. Only applies when the anchor
    # agrees with the cube's base outside `free` (it's one of the cube's
    # own corners) and the star's/tulip's directions are confined to `free`.
    for cc in cube_cuts:
        if len(cc.free) >= cc.n:
            # Degenerate case: free spans every direction, so cc's own
            # inequality is "0 >= 1" over an empty sum — never satisfiable
            # by any x. The dominance argument is vacuously true (cc's
            # feasible set is empty) but operationally meaningless here:
            # adding cc already makes the relaxation infeasible outright,
            # so there is nothing to gain by also pruning Star/Tulip based
            # on it (and tests pin the current behavior of keeping them).
            continue
        for sc in star_cuts:
            if id(sc) in dominated:
                continue
            if sc.leaves <= cc.free and _agrees_outside(sc.z, cc.z, cc.free, cc.n):
                dominated.add(id(sc))
        for tc in tulip_cuts:
            if id(tc) in dominated:
                continue
            if (tc.K | tc.L) <= cc.free and _agrees_outside(tc.z, cc.z, cc.free, cc.n):
                dominated.add(id(tc))

    # --- Cube > Cube (a bigger face dominates one of its own sub-faces) ---
    # Same argument as Cube > Square, generalized: if free_B is a strict
    # subset of free_A and B's anchor is one of A's own corners, then every
    # corner of B is also a corner of A (B's face is a sub-face of A's), so
    # A's inequality already forces exclusion of all of B's corners. This is
    # what lets face_k=4 collapse the native 3-dim sub-cubes/squares/stars
    # that StructureIndex re-detects inside the same probed 4-face once all
    # 16 corners are confirmed infeasible, instead of adding them alongside
    # the one face-spanning cut as separate (weaker, redundant) rows.
    for cc_big in cube_cuts:
        if len(cc_big.free) >= cc_big.n:
            continue
        for cc_small in cube_cuts:
            if cc_small is cc_big or id(cc_small) in dominated:
                continue
            if cc_small.free < cc_big.free and _agrees_outside(cc_small.z, cc_big.z, cc_big.free, cc_big.n):
                dominated.add(id(cc_small))

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
    prune: bool = True,
) -> list[SymbolicCut]:
    """
    Full cut-selection pipeline: dominance-prune, then keep every survivor.

    Parameters
    ----------
    cuts  : all symbolic cuts from cuts_from_index()
    prune : apply dominance pruning (default True)

    Returns
    -------
    Every cut not made redundant by a stronger one in the same pool.
    """
    return dominance_prune(cuts) if prune else list(cuts)
