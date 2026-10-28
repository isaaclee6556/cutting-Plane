"""
Feasibility pump with optional structural cuts and cycle-breaking perturbation
(Milestones 4–6).

Algorithm (per iteration):
  1. Solve LP relaxation (+ any accumulated structural cuts) → x_hat.
  2. Round each integer var to nearest integer → z_dict.
  3. Feasibility check: fix integer vars to z_dict, solve LP.
     If feasible → return z_dict.
  4. Cycle detection: if z_dict already visited →
       - if perturbations < max_perturbations: perturb, clear visited, continue.
       - else: stop (reason="cycle").
  5. (Optional) Insert z into StructureIndex; generate and add violated cuts.
  6. Change objective to minimize L1 distance to z_dict (projection).
  7. Solve projection LP → new x_hat.  Repeat.

Pass use_structural_cuts=True to enable Milestone 5 cut generation.
Pass max_perturbations > 0 to enable Milestone 6 cycle-breaking.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import random
import gurobipy as gp
from gurobipy import GRB

from .bitops import Point
from .structure_index import StructureIndex
from .cuts import cuts_from_index, CutRow, SymbolicCut, VertexCut, CubeCut
from .cut_pool import select_cuts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _integer_var_names(model: gp.Model) -> list[str]:
    """Return names of all binary/integer variables in the model."""
    return [v.VarName for v in model.getVars()
            if v.VType in (GRB.BINARY, GRB.INTEGER)]


def _to_bitmask(z_dict: dict[str, int], sorted_ivars: list[str]) -> Point:
    """Convert {var_name: 0/1} to bitmask: bit i = value of sorted_ivars[i]."""
    mask = 0
    for i, name in enumerate(sorted_ivars):
        if z_dict.get(name, 0):
            mask |= 1 << i
    return mask


def _require_binary(model: gp.Model, int_vars: list[str]) -> None:
    """
    Reject integer variables outside {0,1}: the cut formulas assume x in [0,1]
    and _to_bitmask folds any nonzero value to 1, so such models would silently
    produce invalid cuts.
    """
    by_name = {v.VarName: v for v in model.getVars()}
    bad = []
    for name in int_vars:
        v = by_name[name]
        if v.VType == GRB.BINARY:
            continue
        if v.VType == GRB.INTEGER and v.LB >= 0 and v.UB <= 1:
            continue
        bad.append(name)
    if bad:
        shown = ", ".join(bad[:5])
        more = f" (+{len(bad) - 5} more)" if len(bad) > 5 else ""
        raise ValueError(
            f"structural/no-good cuts require 0/1 variables; {len(bad)} integer "
            f"variable(s) have other bounds: {shown}{more}. Run without cut flags "
            "(plain FP) to handle general integer variables."
        )


def _add_row_to_relax(
    relax: gp.Model,
    coeffs: dict[int, float],
    rhs: float,
    sorted_ivars: list[str],
) -> CutRow:
    """Add one (coeffs, rhs) row as a linear constraint to the relaxation LP."""
    expr = gp.LinExpr()
    for i, c in coeffs.items():
        if i < len(sorted_ivars):
            v = relax.getVarByName(sorted_ivars[i])
            if v is not None:
                expr.addTerms(c, v)
    relax.addConstr(expr >= rhs)
    return coeffs, rhs


def _add_cut_to_relax(
    relax: gp.Model,
    cut: SymbolicCut,
    sorted_ivars: list[str],
) -> CutRow:
    """Add one symbolic cut as a linear constraint to the relaxation LP."""
    coeffs, rhs = cut.to_inequality()
    return _add_row_to_relax(relax, coeffs, rhs, sorted_ivars)


def _row_key(coeffs: dict[int, float], rhs: float) -> tuple:
    """Hashable identity for a row, used to skip exact-duplicate cuts."""
    return (tuple(sorted(coeffs.items())), rhs)


def _row_excludes(row: CutRow, z_mask: Point, tol: float = 1e-9) -> bool:
    coeffs, rhs = row
    lhs = sum(c * ((z_mask >> i) & 1) for i, c in coeffs.items())
    return lhs < rhs - tol


def _set_perturbed_objective(
    relax: gp.Model,
    int_vars: list[str],
    z_dict: dict[str, int],
    rng: random.Random,
    scale: float,
) -> None:
    """
    Set a randomly perturbed projection objective to escape a cycle.

    Each variable's coefficient is the standard projection sign ±1 plus
    uniform noise in [-scale, scale].  This breaks symmetric LP optima
    without moving too far from the intended projection direction.
    """
    obj = gp.LinExpr()
    for name in int_vars:
        v = relax.getVarByName(name)
        if v is None:
            continue
        base = 1.0 if z_dict[name] == 0 else -1.0
        noise = rng.uniform(-scale, scale)
        obj.addTerms(base + noise, v)
    relax.setObjective(obj, GRB.MINIMIZE)


def _set_projection_objective(
    relax: gp.Model,
    int_vars: list[str],
    z_dict: dict[str, int],
) -> None:
    """
    Set LP objective to minimize L1 distance to z_dict over integer variables.

    For z_i = 0: penalize +x_i  (want x_i near 0)
    For z_i = 1: penalize -x_i  (want x_i near 1, equiv. min 1-x_i up to constant)
    """
    obj = gp.LinExpr()
    for name in int_vars:
        v = relax.getVarByName(name)
        if v is None:
            continue
        if z_dict[name] == 0:
            obj += v
        else:
            obj -= v
    relax.setObjective(obj, GRB.MINIMIZE)


# ---------------------------------------------------------------------------
# Feasibility checker (reusable across iterations)
# ---------------------------------------------------------------------------

class _FeasChecker:
    """
    Reusable LP feasibility checker.

    Holds a separate relaxed copy of the model.  Each call to is_feasible()
    temporarily fixes integer variable bounds to the given z values, solves,
    then resets bounds.  Uses a single Gurobi model throughout to amortize
    model-build overhead.
    """

    def __init__(self, model: gp.Model, int_vars: list[str]) -> None:
        lp = model.relax()
        lp.setParam("OutputFlag", 0)
        self._lp = lp
        self._vmap: dict[str, gp.Var] = {v.VarName: v for v in lp.getVars()}
        self._lb0: dict[str, float] = {n: self._vmap[n].LB for n in int_vars}
        self._ub0: dict[str, float] = {n: self._vmap[n].UB for n in int_vars}
        self._int_vars = int_vars

    def is_feasible(self, z_dict: dict[str, int]) -> bool:
        lp = self._lp
        for n, val in z_dict.items():
            v = self._vmap[n]
            v.LB = v.UB = float(val)
        lp.update()
        lp.optimize()
        ok = lp.Status == GRB.OPTIMAL
        # Reset to original relaxation bounds
        for n in self._int_vars:
            v = self._vmap[n]
            v.LB = self._lb0[n]
            v.UB = self._ub0[n]
        lp.update()
        return ok


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class FPResult:
    """
    Return value from feasibility_pump().

    feasible      : True if a feasible integer solution was found.
    solution      : {var_name: int_value} for integer variables when feasible,
                    None otherwise.
    iterations    : number of pump iterations executed (1-based).
    reason        : "feasible" | "cycle" | "lp_infeasible" | "max_iter"
    history       : sequence of rounded points (as sorted tuples) visited.
    cuts_added    : total structural cut constraints added to the LP.
    perturbations : number of cycle-breaking perturbations applied.
    cuts_by_type  : {cut class name: count} of cuts added, e.g.
                    {"VertexCut": 5, "EdgeCut": 2}. Sums to cuts_added.
    fallback_cuts : VertexCuts added only to guarantee the visited point is
                    excluded (structural mode); already included in cuts_added.
    face_probe_cuts : CubeCuts added by the face-probe step (use_face_probe);
                    already included in cuts_added and in cuts_by_type["CubeCut"].
    """
    feasible: bool
    solution: Optional[dict[str, int]]
    iterations: int
    reason: str
    history: list[tuple] = field(default_factory=list)
    cuts_added: int = 0
    perturbations: int = 0
    cuts_by_type: dict[str, int] = field(default_factory=dict)
    fallback_cuts: int = 0
    face_probe_cuts: int = 0


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def feasibility_pump(
    model: gp.Model,
    int_vars: Optional[list[str]] = None,
    max_iter: int = 200,
    verbose: bool = False,
    use_structural_cuts: bool = False,
    use_no_good_cuts: bool = False,
    max_cuts_per_iter: int = 30,
    cut_tol: float = 1e-6,
    max_perturbations: int = 0,
    perturb_scale: float = 0.3,
    random_seed: Optional[int] = None,
    debug_check_cover: bool = False,
    use_face_probe: bool = False,
    face_k: int = 3,
) -> FPResult:
    """
    Feasibility pump with optional structural cuts and cycle-breaking perturbation.

    Parameters
    ----------
    model                : Gurobi MILP (not yet solved; will not be modified).
    int_vars             : names of integer/binary variables; auto-detected if None.
    max_iter             : iteration cap.
    verbose              : print per-iteration info to stdout.
    use_structural_cuts  : enable Milestone-5 structural cut generation
                           (all 7 structures, dominance-pruned). Invariant:
                           after each insertion, the newly visited z is
                           excluded by at least one row added so far; if
                           none is, an unconditional VertexCut(z) is added.
                           Requires 0/1 variables (see _require_binary).
    use_no_good_cuts     : ablation baseline — add exactly one VertexCut
                           (classic no-good cut) for the visited point z each
                           iteration, instead of full structural cuts.
                           Mutually exclusive with use_structural_cuts.
    max_cuts_per_iter    : max cuts added per iteration (efficacy-ranked;
                           only applies to use_structural_cuts).
    cut_tol              : minimum violation to consider a cut active.
    max_perturbations    : max cycle-breaking perturbations before giving up.
                           0 (default) means stop immediately on first cycle.
    perturb_scale        : noise magnitude added to projection coefficients.
    random_seed          : seed for the perturbation RNG (None = non-deterministic).
    debug_check_cover    : (tests/debugging) after each structural insertion, assert
                           every visited point is excluded by some added row.
    use_face_probe       : additional per-iteration step (requires use_structural_cuts).
                           Instead of relying on structure to emerge opportunistically
                           over many iterations, pick the face_k most-fractional
                           variables at x_hat, fix every other int var at its rounded
                           value z, and check feasibility of all 2^face_k corners of
                           that face directly. Any feasible corner is returned as the
                           solution immediately; if all 2^face_k are infeasible, every
                           corner is inserted into the StructureIndex (so the existing
                           edge/square/star/... detection can still pick up emergent
                           structure across them) and a single CubeCut spanning the
                           whole face is added unconditionally (see cuts.CubeCut — the
                           formula is valid for any face size, not just k=3).
    face_k                : number of fractional directions probed by use_face_probe
                           (2**face_k feasibility checks per iteration). Clamped to
                           the number of integer variables AND to the number of
                           directions actually fractional at x_hat this iteration
                           (never padded with already-integral directions).

    Returns
    -------
    FPResult — check .feasible, .solution, .cuts_added, .perturbations.
    """
    if use_structural_cuts and use_no_good_cuts:
        raise ValueError(
            "use_structural_cuts and use_no_good_cuts are mutually exclusive"
        )
    if use_face_probe and not use_structural_cuts:
        raise ValueError("use_face_probe requires use_structural_cuts=True")

    if int_vars is None:
        int_vars = _integer_var_names(model)
    if use_structural_cuts or use_no_good_cuts:
        _require_binary(model, int_vars)
    sorted_ivars = sorted(int_vars)
    n_ivars = len(sorted_ivars)

    rng = random.Random(random_seed)

    # Projection LP (shared; objective + cuts accumulate here)
    relax = model.relax()
    relax.setParam("OutputFlag", 0)

    # Feasibility checker uses original constraints only (no cuts)
    checker = _FeasChecker(model, int_vars)

    # Structure index (only used when use_structural_cuts=True)
    idx = StructureIndex(n_ivars) if use_structural_cuts else None

    # Initial LP solve
    relax.optimize()
    if relax.Status != GRB.OPTIMAL:
        return FPResult(False, None, 0, "lp_infeasible")

    visited: set[tuple] = set()
    history: list[tuple] = []
    total_cuts = 0
    cuts_by_type: dict[str, int] = {}
    fallback_cuts = 0
    face_probe_cuts = 0
    added_rows: list[CutRow] = []
    n_perturbations = 0

    for k in range(max_iter):
        x_hat = {v.VarName: v.X for v in relax.getVars()}

        # Step 2 — round each integer variable
        z_dict = {n: int(round(x_hat[n])) for n in int_vars}
        z_key  = tuple(z_dict[n] for n in sorted_ivars)

        if verbose:
            frac = sum(abs(x_hat[n] - round(x_hat[n])) > 1e-4 for n in int_vars)
            print(
                f"[FP] iter {k:3d}: frac={frac}/{len(int_vars)}"
                f"  z={z_key}  cuts={total_cuts}  perturb={n_perturbations}"
            )

        # Step 3 — feasibility check (original constraints only)
        if checker.is_feasible(z_dict):
            if verbose:
                print(f"[FP] feasible at iter {k}")
            return FPResult(True, z_dict, k + 1, "feasible", history,
                            total_cuts, n_perturbations, dict(cuts_by_type), fallback_cuts, face_probe_cuts)

        # Step 4 — cycle detection with optional perturbation
        if z_key in visited:
            if n_perturbations >= max_perturbations:
                if verbose:
                    print(f"[FP] cycle at iter {k}, no perturbations left")
                return FPResult(False, None, k + 1, "cycle", history,
                                total_cuts, n_perturbations, dict(cuts_by_type), fallback_cuts, face_probe_cuts)
            # Perturb: add noise to projection objective and reset visit set
            n_perturbations += 1
            visited.clear()
            if verbose:
                print(f"[FP] cycle at iter {k}, applying perturbation {n_perturbations}")
            _set_perturbed_objective(relax, int_vars, z_dict, rng, perturb_scale)
            relax.optimize()
            if relax.Status != GRB.OPTIMAL:
                return FPResult(False, None, k + 1, "lp_infeasible", history,
                                total_cuts, n_perturbations, dict(cuts_by_type), fallback_cuts, face_probe_cuts)
            continue  # skip normal projection; go to next iteration

        visited.add(z_key)
        history.append(z_key)

        # Step 5 — structural cuts (Milestone 5)
        if use_structural_cuts and idx is not None:
            z_mask = _to_bitmask(z_dict, sorted_ivars)

            # Face probe (optional): check all 2^face_k corners of the face
            # spanned by the face_k most-fractional directions, fixing every
            # other int var at z. Any feasible corner ends the pump outright;
            # if all are infeasible, insert them all (so the usual edge/
            # square/star/... detection below can pick up emergent structure)
            # and add a single CubeCut for the whole face unconditionally.
            if use_face_probe:
                # Clamp to the number of directions actually fractional at x_hat.
                # Without this, an instance with fewer fractional vars than
                # face_k gets padded with already-integral directions (tied at
                # |x-0.5|=0.5, broken by index) — probing the *same* irrelevant
                # pair of directions every iteration, which both wastes
                # feasibility checks and floods idx with near-duplicate points
                # (observed: 775 cuts in 25 iterations on a 100-var instance
                # with only 2 fractional vars, vs. the intended handful).
                n_fractional = sum(
                    1 for name in sorted_ivars if abs(x_hat[name] - round(x_hat[name])) > 1e-6
                )
                k_eff = min(face_k, n_ivars, max(n_fractional, 1))
                face_positions = sorted(
                    range(n_ivars), key=lambda i: abs(x_hat[sorted_ivars[i]] - 0.5)
                )[:k_eff]
                corner_masks = []
                feasible_corner = None
                for combo in range(1 << k_eff):
                    corner_mask = z_mask
                    for j, pos in enumerate(face_positions):
                        if (combo >> j) & 1:
                            corner_mask |= (1 << pos)
                        else:
                            corner_mask &= ~(1 << pos)
                    corner_masks.append(corner_mask)
                    if corner_mask == z_mask:
                        continue  # already confirmed infeasible this iteration
                    corner_dict = {
                        name: (corner_mask >> i) & 1 for i, name in enumerate(sorted_ivars)
                    }
                    if checker.is_feasible(corner_dict):
                        feasible_corner = corner_dict
                        break

                if feasible_corner is not None:
                    if verbose:
                        print(f"[FP] face probe found feasible corner at iter {k}")
                    return FPResult(True, feasible_corner, k + 1, "feasible", history,
                                    total_cuts, n_perturbations, dict(cuts_by_type),
                                    fallback_cuts, face_probe_cuts)

                for cm in corner_masks:
                    idx.insert(cm)
                face_free = frozenset(face_positions)
                face_base = z_mask
                for pos in face_positions:
                    face_base &= ~(1 << pos)
                face_cut = CubeCut(z=face_base, free=face_free, n=n_ivars)
                added_rows.append(_add_cut_to_relax(relax, face_cut, sorted_ivars))
                total_cuts += 1
                face_probe_cuts += 1
                cuts_by_type["CubeCut"] = cuts_by_type.get("CubeCut", 0) + 1
                if verbose:
                    print(f"[FP]   face probe: all {1 << k_eff} corners infeasible, "
                          f"added CubeCut over {sorted(face_positions)} (total {total_cuts})")

            idx.insert(z_mask)

            # x_hat in index space: bit i → sorted_ivars[i]
            x_hat_idx = {i: x_hat[sorted_ivars[i]] for i in range(n_ivars)}

            all_cuts = cuts_from_index(idx)
            selected = select_cuts(all_cuts, x_hat_idx, k=max_cuts_per_iter, tol=cut_tol)

            # Exact-duplicate guard: dominance_prune only sees candidates
            # from idx, so it can't know about rows added outside this call
            # (e.g. the face-probe CubeCut just added above). When
            # StructureIndex re-detects that same face as a native cube
            # (always happens when face_k == 3, its own cube size), this
            # row-level check is what actually stops it being added twice.
            added_row_keys = {_row_key(c, r) for c, r in added_rows}
            n_added = 0
            for cut in selected:
                coeffs, rhs = cut.to_inequality()
                key = _row_key(coeffs, rhs)
                if key in added_row_keys:
                    continue
                added_row_keys.add(key)
                added_rows.append(_add_row_to_relax(relax, coeffs, rhs, sorted_ivars))
                name = type(cut).__name__
                cuts_by_type[name] = cuts_by_type.get(name, 0) + 1
                n_added += 1
            total_cuts += n_added

            if verbose and n_added:
                print(f"[FP]   added {n_added} cuts (total {total_cuts})")

            # Cover guarantee: z must be excluded by some row added so far.
            # Dominated or unviolated cuts may leave z uncovered, so add its
            # VertexCut unconditionally in that case.
            if not any(_row_excludes(r, z_mask) for r in added_rows):
                added_rows.append(_add_cut_to_relax(relax, VertexCut(z=z_mask, n=n_ivars), sorted_ivars))
                total_cuts += 1
                fallback_cuts += 1
                cuts_by_type["VertexCut"] = cuts_by_type.get("VertexCut", 0) + 1

                if verbose:
                    print(f"[FP]   z not excluded by any row; added VertexCut (total {total_cuts})")

            if debug_check_cover:
                uncovered = [p for p in idx.H if not any(_row_excludes(r, p) for r in added_rows)]
                if uncovered:
                    raise AssertionError(
                        f"cover invariant violated at iter {k}: {len(uncovered)} visited "
                        f"point(s) not excluded by any added row, e.g. masks {uncovered[:5]}"
                    )

        elif use_no_good_cuts:
            z_mask = _to_bitmask(z_dict, sorted_ivars)
            cut = VertexCut(z=z_mask, n=n_ivars)
            _add_cut_to_relax(relax, cut, sorted_ivars)
            total_cuts += 1
            cuts_by_type["VertexCut"] = cuts_by_type.get("VertexCut", 0) + 1

            if verbose:
                print(f"[FP]   added no-good cut (total {total_cuts})")

        # Step 6 — projection LP
        _set_projection_objective(relax, int_vars, z_dict)
        relax.optimize()
        if relax.Status != GRB.OPTIMAL:
            return FPResult(False, None, k + 1, "lp_infeasible", history,
                            total_cuts, n_perturbations, dict(cuts_by_type), fallback_cuts, face_probe_cuts)

    if verbose:
        print(f"[FP] max_iter={max_iter} reached without feasible solution")
    return FPResult(False, None, max_iter, "max_iter", history,
                    total_cuts, n_perturbations, dict(cuts_by_type), fallback_cuts, face_probe_cuts)
