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
from .cuts import cuts_from_index, SymbolicCut, VertexCut
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


def _add_cut_to_relax(
    relax: gp.Model,
    cut: SymbolicCut,
    sorted_ivars: list[str],
) -> None:
    """Add one symbolic cut as a linear constraint to the relaxation LP."""
    coeffs, rhs = cut.to_inequality()
    expr = gp.LinExpr()
    for i, c in coeffs.items():
        if i < len(sorted_ivars):
            v = relax.getVarByName(sorted_ivars[i])
            if v is not None:
                expr.addTerms(c, v)
    relax.addConstr(expr >= rhs)


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
    """
    feasible: bool
    solution: Optional[dict[str, int]]
    iterations: int
    reason: str
    history: list[tuple] = field(default_factory=list)
    cuts_added: int = 0
    perturbations: int = 0
    cuts_by_type: dict[str, int] = field(default_factory=dict)


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
                           (all 7 structures, dominance-pruned). If no
                           structural cut is violated at the current x_hat,
                           falls back to an unconditional VertexCut for z
                           so every rejected point is still guaranteed to
                           be excluded going forward (matches the guarantee
                           use_no_good_cuts always provides).
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

    Returns
    -------
    FPResult — check .feasible, .solution, .cuts_added, .perturbations.
    """
    if use_structural_cuts and use_no_good_cuts:
        raise ValueError(
            "use_structural_cuts and use_no_good_cuts are mutually exclusive"
        )

    if int_vars is None:
        int_vars = _integer_var_names(model)
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
                            total_cuts, n_perturbations, dict(cuts_by_type))

        # Step 4 — cycle detection with optional perturbation
        if z_key in visited:
            if n_perturbations >= max_perturbations:
                if verbose:
                    print(f"[FP] cycle at iter {k}, no perturbations left")
                return FPResult(False, None, k + 1, "cycle", history,
                                total_cuts, n_perturbations, dict(cuts_by_type))
            # Perturb: add noise to projection objective and reset visit set
            n_perturbations += 1
            visited.clear()
            if verbose:
                print(f"[FP] cycle at iter {k}, applying perturbation {n_perturbations}")
            _set_perturbed_objective(relax, int_vars, z_dict, rng, perturb_scale)
            relax.optimize()
            if relax.Status != GRB.OPTIMAL:
                return FPResult(False, None, k + 1, "lp_infeasible", history,
                                total_cuts, n_perturbations, dict(cuts_by_type))
            continue  # skip normal projection; go to next iteration

        visited.add(z_key)
        history.append(z_key)

        # Step 5 — structural cuts (Milestone 5)
        if use_structural_cuts and idx is not None:
            z_mask = _to_bitmask(z_dict, sorted_ivars)
            idx.insert(z_mask)

            # x_hat in index space: bit i → sorted_ivars[i]
            x_hat_idx = {i: x_hat[sorted_ivars[i]] for i in range(n_ivars)}

            all_cuts = cuts_from_index(idx)
            selected = select_cuts(all_cuts, x_hat_idx, k=max_cuts_per_iter, tol=cut_tol)

            for cut in selected:
                _add_cut_to_relax(relax, cut, sorted_ivars)
                name = type(cut).__name__
                cuts_by_type[name] = cuts_by_type.get(name, 0) + 1
            total_cuts += len(selected)

            if verbose and selected:
                print(f"[FP]   added {len(selected)} cuts (total {total_cuts})")

            # Fallback: select_cuts requires a cut to be violated at the
            # current x_hat before it's added. On instances with many
            # simultaneously-fractional variables, even the plain VertexCut
            # for z can fail this check (sum of per-coordinate distances
            # exceeds 1), so nothing gets added at all. A no-good cut's job
            # is to guarantee z itself is never produced by the LP again —
            # that guarantee must not depend on current LP geometry, so add
            # it unconditionally whenever nothing else was selected.
            if not selected:
                fallback = VertexCut(z=z_mask, n=n_ivars)
                _add_cut_to_relax(relax, fallback, sorted_ivars)
                total_cuts += 1
                cuts_by_type["VertexCut"] = cuts_by_type.get("VertexCut", 0) + 1

                if verbose:
                    print(f"[FP]   no violated structural cut; added fallback no-good cut (total {total_cuts})")

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
                            total_cuts, n_perturbations, dict(cuts_by_type))

    if verbose:
        print(f"[FP] max_iter={max_iter} reached without feasible solution")
    return FPResult(False, None, max_iter, "max_iter", history,
                    total_cuts, n_perturbations, dict(cuts_by_type))
