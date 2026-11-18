"""
Feasibility pump with optional structural cuts and cycle-breaking perturbation
(Milestones 4-6), plus an optional optimality-search overlay (M11).

Algorithm (per iteration), mode="feasibility" (default, unchanged from before M11):
  1. Solve LP relaxation (+ any accumulated structural cuts) -> x_hat.
  2. Round each integer var to nearest integer -> z_dict.
  3. Feasibility check: fix integer vars to z_dict, solve LP.
     If feasible -> return z_dict.
  4. Cycle detection: if z_dict already visited ->
       - if perturbations < max_perturbations: perturb, clear visited, continue.
       - else: stop (reason="cycle").
  5. (Optional) Insert z into StructureIndex; generate and add every
     dominance-surviving cut (no violation filtering, no cap).
  6. Change objective to minimize L1 distance to z_dict (projection).
  7. Solve projection LP -> new x_hat.  Repeat.

Algorithm, mode="optimality" (M11): same loop, but step 3 evaluates the
*original* objective's optimal continuous completion instead of just
feasibility, tracks an incumbent (best feasible solution found so far) and
a valid lower/upper bound on the true optimum (via the LP relaxation plus
whatever cuts + a cutoff constraint have been added so far), and terminates
with a proof of optimality (or infeasibility) once the bound can no longer
beat the incumbent. Every rounded/probed point -- whether infeasible or
feasible-but-not-improving -- is still inserted into the StructureIndex and
still receives cuts exactly as in feasibility mode, since the goal is to
exhaustively rule out the region "at least as good as the incumbent", not
to stop at the first feasible point.

Pass use_structural_cuts=True to enable Milestone 5 cut generation.
Pass max_perturbations > 0 to enable Milestone 6 cycle-breaking.
Pass mode="optimality" to enable Milestone 11 optimality search.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import random
import time
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
# Optimality-mode helpers (M11)
# ---------------------------------------------------------------------------

def _solve_status_kind(m: gp.Model) -> str:
    """
    Classify a just-completed optimize() call as one of:
    "optimal" | "infeasible" | "unbounded" | "time_limit" | "other".

    INF_OR_UNBD is disambiguated by re-solving with DualReductions=0 (left at
    0 afterward -- slightly less presolve, but this model is only ever used
    for cheap LP/relaxation solves, and disambiguating is more important than
    that marginal speed).
    """
    s = m.Status
    if s == GRB.OPTIMAL:
        return "optimal"
    if s == GRB.INFEASIBLE:
        return "infeasible"
    if s == GRB.UNBOUNDED:
        return "unbounded"
    if s == GRB.TIME_LIMIT:
        return "time_limit"
    if s == GRB.INF_OR_UNBD:
        m.setParam("DualReductions", 0)
        m.optimize()
        return _solve_status_kind(m)
    return "other"


def _apply_time_limit(m: gp.Model, time_limit: Optional[float], elapsed: float) -> None:
    """Give `m` whatever's left of the overall wall-clock budget, if any."""
    if time_limit is not None:
        m.setParam("TimeLimit", max(0.0, time_limit - elapsed))


def _obj_is_pure_integer(model: gp.Model, int_var_set: set[str]) -> bool:
    """True if every continuous variable has a zero objective coefficient and
    every integer/binary variable has an integer objective coefficient --
    i.e. the objective can only ever take integer values, so any improvement
    is by at least 1."""
    obj = model.getObjective()
    for i in range(obj.size()):
        v = obj.getVar(i)
        c = obj.getCoeff(i)
        if v.VarName in int_var_set:
            if abs(c - round(c)) > 1e-9:
                return False
        else:
            if abs(c) > 1e-9:
                return False
    return True


def _effective_delta(user_obj_tol: Optional[float], obj_is_integer: bool, U: Optional[float]) -> float:
    """Minimum improvement required to count as "better than the incumbent"."""
    if user_obj_tol is not None:
        return user_obj_tol
    if obj_is_integer:
        return 1.0 - 1e-6
    return max(1e-6, 1e-6 * abs(U if U is not None else 0.0))


def _improves(sense: int, obj: float, U: Optional[float], delta: float) -> bool:
    """True if `obj` is a strict improvement over incumbent `U` (or there is
    no incumbent yet)."""
    if U is None:
        return True
    return (obj <= U - delta) if sense == GRB.MINIMIZE else (obj >= U + delta)


def _bound_proves_optimal(sense: int, L: float, U: Optional[float], delta: float) -> bool:
    """True if bound `L` rules out any solution better than incumbent `U` by
    more than `delta` -- i.e. U is proven (near-)optimal. False (not yet
    proven) whenever there is no incumbent to compare against."""
    if U is None:
        return False
    return (L >= U - delta) if sense == GRB.MINIMIZE else (L <= U + delta)


# ---------------------------------------------------------------------------
# Feasibility checker (reusable across iterations)
# ---------------------------------------------------------------------------

class _FeasChecker:
    """
    Reusable LP feasibility/evaluation checker.

    Holds a separate relaxed copy of the model.  Each call to is_feasible()
    or evaluate() temporarily fixes integer variable bounds to the given z
    values, solves, then resets bounds.  Uses a single Gurobi model
    throughout to amortize model-build overhead.
    """

    def __init__(self, model: gp.Model, int_vars: list[str], gurobi_seed: Optional[int] = None) -> None:
        lp = model.relax()
        lp.setParam("OutputFlag", 0)
        if gurobi_seed is not None:
            lp.setParam("Seed", gurobi_seed)
        # model.relax() carries over the original objective/sense verbatim;
        # capture both before overwriting the objective below, so evaluate()
        # can still solve the *original* objective on demand (M11).
        self._orig_obj_expr = lp.getObjective()
        self._orig_sense = lp.ModelSense
        # Feasibility-only default: a zero objective means optimize() reports
        # OPTIMAL as soon as any feasible point exists, and can never come
        # back UNBOUNDED. Without this, a continuous direction left unbounded
        # by the fixed integer bounds would make is_feasible() return False
        # for a point that is actually feasible (Status would be UNBOUNDED,
        # not OPTIMAL).
        lp.setObjective(gp.LinExpr(), GRB.MINIMIZE)
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

    def evaluate(self, z_dict: dict[str, int]) -> tuple[str, Optional[float], Optional[dict[str, float]]]:
        """
        Fix integer vars to z_dict, solve the *original* objective (native
        sense) over the continuous completion.

        Returns (status, native_obj_value, full_solution):
          status in {"feasible", "infeasible", "unbounded"}.
          native_obj_value, full_solution are None unless status=="feasible".
          full_solution is {var_name: value} for *every* variable (int and
          continuous), captured right after the solve so it reflects this
          exact z (not whatever z the checker is next used for).
        """
        lp = self._lp
        for n, val in z_dict.items():
            v = self._vmap[n]
            v.LB = v.UB = float(val)
        lp.setObjective(self._orig_obj_expr, self._orig_sense)
        lp.update()
        lp.optimize()
        kind = _solve_status_kind(lp)
        if kind == "optimal":
            result = ("feasible", lp.ObjVal, {v.VarName: v.X for v in lp.getVars()})
        elif kind == "unbounded":
            result = ("unbounded", None, None)
        else:  # "infeasible" or unexpected ("time_limit"/"other" -> treat as infeasible)
            result = ("infeasible", None, None)
        # Reset to original relaxation bounds and the feasibility-only
        # zero objective, so a later is_feasible() call behaves as documented.
        for n in self._int_vars:
            v = self._vmap[n]
            v.LB = self._lb0[n]
            v.UB = self._ub0[n]
        lp.setObjective(gp.LinExpr(), GRB.MINIMIZE)
        lp.update()
        return result


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class FPResult:
    """
    Return value from feasibility_pump().

    feasible      : mode="feasibility": True if a feasible integer solution
                    was found. mode="optimality": True if at least one
                    incumbent was found (equivalent to n_feasible_found > 0).
    solution      : {var_name: int_value} for integer variables when
                    feasible, None otherwise (the best incumbent's integer
                    part, in optimality mode).
    iterations    : number of pump iterations executed (1-based).
    reason        : "feasible" | "cycle" | "lp_infeasible" | "max_iter"
                    (mode="feasibility"), or "optimal" | "infeasible" |
                    "unbounded" | "cycle" | "max_iter" | "time_limit"
                    (mode="optimality").
    history       : sequence of rounded points (as sorted tuples) visited.
    cuts_added    : total structural cut constraints added to the LP.
    perturbations : number of cycle-breaking perturbations applied.
    cuts_by_type  : {cut class name: count} of cuts added, e.g.
                    {"VertexCut": 5, "EdgeCut": 2}. Sums to cuts_added.
    fallback_cuts : VertexCuts added only to guarantee the visited point is
                    excluded (structural mode); already included in cuts_added.
    face_probe_cuts : CubeCuts added by the face-probe step under cut_mode=
                    "structural" (use_structural_cuts); already included in
                    cuts_added and in cuts_by_type["CubeCut"].
    probe_vertex_cuts : VertexCuts added for probed *corners* (not the
                    rounded point z itself) under cut_mode="nogood"
                    (use_no_good_cuts); already included in cuts_added and
                    in cuts_by_type["VertexCut"]. 0 under any other cut_mode.

    Optimality-mode fields (M11; all None/0/empty in mode="feasibility"):
    mode              : "feasibility" | "optimality".
    best_obj          : incumbent objective value, original sense/sign.
    best_solution     : {var_name: value} for *every* variable (int and
                        continuous) of the best incumbent, or None.
    bound             : latest valid bound on the true optimum (original
                        sense/sign), or None if never computed.
    gap               : |best_obj - bound| / max(|best_obj|, 1e-10), or None
                        if best_obj or bound is unavailable.
    proved_optimal    : True if reason is "optimal" or "infeasible" (a real
                        proof, not just giving up).
    n_feasible_found  : number of feasible points evaluated (incumbent or not).
    n_incumbent_updates : number of times the incumbent actually improved.
    first_feasible_iter : iteration (0-based) of the first feasible point found.
    first_feasible_time : wall-clock seconds at that point.
    incumbent_trace   : list of (iteration, wall_clock_seconds, best_obj) at
                        each incumbent update.
    time_sec          : total wall-clock seconds for this call (optimality
                        mode only; 0.0 in feasibility mode).
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
    probe_vertex_cuts: int = 0
    mode: str = "feasibility"
    best_obj: Optional[float] = None
    best_solution: Optional[dict[str, float]] = None
    bound: Optional[float] = None
    gap: Optional[float] = None
    proved_optimal: bool = False
    n_feasible_found: int = 0
    n_incumbent_updates: int = 0
    first_feasible_iter: Optional[int] = None
    first_feasible_time: Optional[float] = None
    incumbent_trace: list = field(default_factory=list)
    time_sec: float = 0.0


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
    max_perturbations: int = 0,
    perturb_scale: float = 0.3,
    random_seed: Optional[int] = None,
    debug_check_cover: bool = False,
    use_face_probe: bool = False,
    face_k: int = 3,
    mode: str = "feasibility",
    time_limit: Optional[float] = None,
    obj_tol: Optional[float] = None,
    gurobi_seed: Optional[int] = None,
) -> FPResult:
    """
    Feasibility pump with optional structural cuts, cycle-breaking
    perturbation, and an optional optimality-search overlay.

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
    max_perturbations    : max cycle-breaking perturbations before giving up.
                           0 (default) means stop immediately on first cycle.
    perturb_scale        : noise magnitude added to projection coefficients.
    random_seed          : seed for the perturbation RNG (None = non-deterministic).
    debug_check_cover    : (tests/debugging) after each structural insertion, assert
                           every visited point is excluded by some added row.
    use_face_probe       : additional per-iteration step, independent of which (if
                           any) cut mechanism is active (M12: no longer requires
                           use_structural_cuts). Instead of relying on structure to
                           emerge opportunistically over many iterations, pick the
                           face_k most-fractional variables at x_hat, fix every
                           other int var at its rounded value z, and check
                           feasibility of all 2^face_k corners of that face
                           directly. mode="feasibility": any feasible corner is
                           returned as the solution immediately. mode="optimality":
                           every corner is evaluated (not just checked for
                           feasibility) and may update the incumbent. What happens
                           to the probed corners afterward depends on which cut
                           mechanism is active: use_structural_cuts -- all corners
                           are inserted into the StructureIndex and the whole face
                           is always cut (see cuts.CubeCut — the formula is valid
                           for any face size, not just k=3); use_no_good_cuts --
                           each probed corner (other than z itself) gets its own
                           VertexCut; neither -- probing only affects which point
                           is returned/which incumbent is tracked, with no cuts at
                           all (isolates probing's own effect).
    face_k                : number of fractional directions probed by use_face_probe
                           (2**face_k feasibility checks per iteration). Clamped to
                           the number of integer variables AND to the number of
                           directions actually fractional at x_hat this iteration
                           (never padded with already-integral directions).
    mode                  : "feasibility" (default, finds and returns the first
                           feasible point) or "optimality" (M11: tracks an
                           incumbent and a valid bound, and only stops once the
                           bound proves the incumbent -- or infeasibility -- or
                           it gives up via cycle/max_iter/time_limit). See
                           FPResult's optimality-mode fields.
    time_limit            : overall wall-clock budget in seconds. Checked once
                           per iteration and also handed to Gurobi as each
                           solve's own TimeLimit, so a single slow solve can't
                           blow far past the budget. None (default) = no limit.
    obj_tol               : minimum improvement (in the original objective's
                           own units) required for a new incumbent to count as
                           "better", and slack in the optimality bound check.
                           None (default) picks it automatically: 1 - 1e-6 if
                           the objective can only take integer values (every
                           continuous var has coefficient 0, every integer
                           var's coefficient is an integer), else
                           max(1e-6, 1e-6 * |incumbent|). Only meaningful in
                           mode="optimality".
    gurobi_seed           : if not None, sets Gurobi's own "Seed" parameter
                           (M13) on both the projection/bound LP and the
                           feasibility/evaluation checker's LP -- the only
                           thing a run's seed controls; the perturbation RNG
                           is driven solely by random_seed. None (default)
                           leaves Gurobi's Seed parameter untouched.

    Returns
    -------
    FPResult — check .feasible, .solution, .cuts_added, .perturbations, and
    (mode="optimality") .best_obj, .bound, .gap, .proved_optimal.
    """
    if mode not in ("feasibility", "optimality"):
        raise ValueError(f"mode must be 'feasibility' or 'optimality', got {mode!r}")
    if use_structural_cuts and use_no_good_cuts:
        raise ValueError(
            "use_structural_cuts and use_no_good_cuts are mutually exclusive"
        )
    # Internal normalization (M12): cut_mode and probe_k drive the rest of
    # this function; use_face_probe no longer requires use_structural_cuts
    # -- the face-probe step is independent of which (if any) cut mechanism
    # is active, and reacts to the probed corners according to cut_mode.
    cut_mode = "structural" if use_structural_cuts else ("nogood" if use_no_good_cuts else "none")
    probe_k = face_k if use_face_probe else 0

    if int_vars is None:
        int_vars = _integer_var_names(model)
    if use_structural_cuts or use_no_good_cuts or use_face_probe:
        # Face probing builds each corner via bit-flips on int_vars, which
        # only makes sense for 0/1 variables -- same requirement as the cut
        # mechanisms, now also needed when probing runs under cut_mode="none"
        # (no longer gated behind use_structural_cuts).
        _require_binary(model, int_vars)
    sorted_ivars = sorted(int_vars)
    n_ivars = len(sorted_ivars)
    int_var_set = set(int_vars)

    rng = random.Random(random_seed)
    t_start = time.perf_counter()

    def _elapsed() -> float:
        return time.perf_counter() - t_start

    # Projection LP (shared; objective + cuts accumulate here)
    relax = model.relax()
    relax.setParam("OutputFlag", 0)
    if gurobi_seed is not None:
        relax.setParam("Seed", gurobi_seed)
    # Captured before anything overwrites relax's objective (M11 bound step
    # and the optimality cutoff constraint both need the *original* one).
    relax_orig_obj_expr = relax.getObjective()
    orig_sense = relax.ModelSense

    # Feasibility checker uses original constraints only (no cuts)
    checker = _FeasChecker(model, int_vars, gurobi_seed=gurobi_seed)

    # Structure index (only used when use_structural_cuts=True)
    idx = StructureIndex(n_ivars) if cut_mode == "structural" else None

    # --- Optimality-mode bookkeeping (M11). Harmless no-ops in feasibility
    # mode: initialized unconditionally (cheap) so _finish() below can
    # always read them, but never populated or returned by any
    # mode="feasibility" code path. ---
    U: Optional[float] = None               # incumbent objective (native sense)
    bound: Optional[float] = None           # latest valid bound (native sense)
    best_solution: Optional[dict[str, float]] = None
    n_feasible_found = 0
    n_incumbent_updates = 0
    first_feasible_iter: Optional[int] = None
    first_feasible_time: Optional[float] = None
    incumbent_trace: list[tuple] = []
    cutoff_constr = None                     # gp.Constr | None
    obj_is_integer = _obj_is_pure_integer(model, int_var_set) if mode == "optimality" else False

    def _delta() -> float:
        return _effective_delta(obj_tol, obj_is_integer, U)

    def _update_cutoff(new_U: float) -> None:
        nonlocal cutoff_constr
        rhs = new_U - _delta() if orig_sense == GRB.MINIMIZE else new_U + _delta()
        if cutoff_constr is None:
            if orig_sense == GRB.MINIMIZE:
                cutoff_constr = relax.addConstr(relax_orig_obj_expr <= rhs)
            else:
                cutoff_constr = relax.addConstr(relax_orig_obj_expr >= rhs)
        else:
            cutoff_constr.RHS = rhs
        relax.update()

    def _record_feasible(k: int, obj: float, sol: dict[str, float]) -> None:
        nonlocal U, best_solution, n_feasible_found, n_incumbent_updates
        nonlocal first_feasible_iter, first_feasible_time
        n_feasible_found += 1
        if first_feasible_iter is None:
            first_feasible_iter = k
            first_feasible_time = _elapsed()
        if _improves(orig_sense, obj, U, _delta()):
            U = obj
            best_solution = sol
            n_incumbent_updates += 1
            incumbent_trace.append((k, _elapsed(), U))
            _update_cutoff(U)

    def _finish(reason: str, iterations: int) -> FPResult:
        gap = None
        if U is not None and bound is not None:
            gap = abs(U - bound) / max(abs(U), 1e-10)
        sol_int = None
        if best_solution is not None:
            sol_int = {n: int(round(best_solution[n])) for n in int_vars}
        return FPResult(
            U is not None, sol_int, iterations, reason, history,
            total_cuts, n_perturbations, dict(cuts_by_type), fallback_cuts, face_probe_cuts,
            probe_vertex_cuts=probe_vertex_cuts,
            mode=mode, best_obj=U, best_solution=best_solution, bound=bound, gap=gap,
            proved_optimal=reason in ("optimal", "infeasible"),
            n_feasible_found=n_feasible_found, n_incumbent_updates=n_incumbent_updates,
            first_feasible_iter=first_feasible_iter, first_feasible_time=first_feasible_time,
            incumbent_trace=list(incumbent_trace), time_sec=_elapsed(),
        )

    # Initial LP solve
    history: list[tuple] = []
    total_cuts = 0
    cuts_by_type: dict[str, int] = {}
    fallback_cuts = 0
    face_probe_cuts = 0
    probe_vertex_cuts = 0
    n_perturbations = 0

    _apply_time_limit(relax, time_limit, _elapsed())
    relax.optimize()
    kind0 = _solve_status_kind(relax)
    if kind0 == "time_limit":
        if mode == "optimality":
            return _finish("time_limit", 0)
        return FPResult(False, None, 0, "time_limit", time_sec=_elapsed())
    if kind0 != "optimal":
        if mode == "optimality":
            # Only a genuine INFEASIBLE status proves infeasibility; an
            # unbounded *relaxation* (e.g. an unconstrained continuous
            # variable with a favorable objective coefficient, before any
            # cuts exist) is a real, distinct outcome, not a proof of
            # infeasibility -- conflating the two would falsely claim
            # proved_optimal=True.
            if kind0 == "unbounded":
                return _finish("unbounded_relaxation", 0)
            return _finish("infeasible", 0)
        return FPResult(False, None, 0, "lp_infeasible")

    visited: set[tuple] = set()
    added_rows: list[CutRow] = []
    face_cuts: list[CubeCut] = []
    # Persistent dedup state (both grow monotonically across iterations, so
    # rebuilding them from `added_rows` every iteration would cost
    # O(iterations x history) overall; symbolic cuts are frozen dataclasses,
    # hence hashable, so membership is O(1) amortized either way).
    added_symbolic: set[SymbolicCut] = set()
    added_row_keys: set[tuple] = set()

    for k in range(max_iter):
        elapsed = _elapsed()
        if time_limit is not None and elapsed >= time_limit:
            if mode == "optimality":
                return _finish("time_limit", k)
            return FPResult(False, None, k, "time_limit", history,
                            total_cuts, n_perturbations, dict(cuts_by_type), fallback_cuts, face_probe_cuts,
                            probe_vertex_cuts=probe_vertex_cuts, time_sec=elapsed)

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

        # Step 3 — feasibility/evaluation check (original constraints only)
        if mode == "feasibility":
            if checker.is_feasible(z_dict):
                if verbose:
                    print(f"[FP] feasible at iter {k}")
                return FPResult(True, z_dict, k + 1, "feasible", history,
                                total_cuts, n_perturbations, dict(cuts_by_type), fallback_cuts, face_probe_cuts,
                                probe_vertex_cuts=probe_vertex_cuts)
        else:
            status, obj, sol = checker.evaluate(z_dict)
            if status == "unbounded":
                if verbose:
                    print(f"[FP] unbounded at iter {k}")
                return _finish("unbounded", k + 1)
            if status == "feasible" and z_key not in visited:
                # z_key in visited means this exact point was already
                # evaluated on an earlier iteration (before this cycle came
                # back around) -- the result is identical (deterministic),
                # so counting it again would double (or endlessly re-)count
                # the same feasible point in n_feasible_found.
                _record_feasible(k, obj, sol)
                if verbose:
                    print(f"[FP]   z feasible, obj={obj}, incumbent U={U}")

        # Step 4 — cycle detection with optional perturbation
        if z_key in visited:
            if n_perturbations >= max_perturbations:
                if verbose:
                    print(f"[FP] cycle at iter {k}, no perturbations left")
                if mode == "optimality":
                    return _finish("cycle", k + 1)
                return FPResult(False, None, k + 1, "cycle", history,
                                total_cuts, n_perturbations, dict(cuts_by_type), fallback_cuts, face_probe_cuts,
                                probe_vertex_cuts=probe_vertex_cuts)
            # Perturb: add noise to projection objective and reset visit set
            n_perturbations += 1
            visited.clear()
            if verbose:
                print(f"[FP] cycle at iter {k}, applying perturbation {n_perturbations}")
            _set_perturbed_objective(relax, int_vars, z_dict, rng, perturb_scale)
            _apply_time_limit(relax, time_limit, _elapsed())
            relax.optimize()
            kind = _solve_status_kind(relax)
            if kind == "time_limit":
                if mode == "optimality":
                    return _finish("time_limit", k + 1)
                return FPResult(False, None, k + 1, "time_limit", history,
                                total_cuts, n_perturbations, dict(cuts_by_type), fallback_cuts, face_probe_cuts,
                                probe_vertex_cuts=probe_vertex_cuts, time_sec=_elapsed())
            if kind != "optimal":
                if mode == "optimality":
                    # Only a genuine INFEASIBLE status is a proof (the region
                    # under cuts+cutoff is provably empty); "unbounded"/
                    # "other" here means something numerically went wrong,
                    # not that optimality/infeasibility was established.
                    if kind == "infeasible":
                        if U is not None:
                            bound = U  # region now empty -> U is exactly optimal, gap=0
                        return _finish("optimal" if U is not None else "infeasible", k + 1)
                    return _finish("numerical", k + 1)
                return FPResult(False, None, k + 1, "lp_infeasible", history,
                                total_cuts, n_perturbations, dict(cuts_by_type), fallback_cuts, face_probe_cuts,
                                probe_vertex_cuts=probe_vertex_cuts)
            continue  # skip normal projection; go to next iteration

        visited.add(z_key)
        history.append(z_key)

        # Step 5 — face probe (cut_mode-independent, M12) + cut generation
        # (cut_mode-dependent: "structural" | "nogood" | "none").
        z_mask = _to_bitmask(z_dict, sorted_ivars)

        if probe_k > 0:
            # Clamp to the number of directions actually fractional at x_hat.
            # Without this, an instance with fewer fractional vars than
            # probe_k gets padded with already-integral directions (tied at
            # |x-0.5|=0.5, broken by index) — probing the *same* irrelevant
            # pair of directions every iteration, which both wastes
            # feasibility checks and floods idx with near-duplicate points
            # (observed: 775 cuts in 25 iterations on a 100-var instance
            # with only 2 fractional vars, vs. the intended handful).
            n_fractional = sum(
                1 for name in sorted_ivars if abs(x_hat[name] - round(x_hat[name])) > 1e-6
            )
            k_eff = min(probe_k, n_ivars, max(n_fractional, 1))
            face_positions = sorted(
                range(n_ivars), key=lambda i: abs(x_hat[sorted_ivars[i]] - 0.5)
            )[:k_eff]
            corner_masks = []

            if mode == "feasibility":
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
                                    fallback_cuts, face_probe_cuts,
                                    probe_vertex_cuts=probe_vertex_cuts)
            else:
                # Optimality mode: never stop early -- evaluate every
                # corner (skipping z_mask itself, already evaluated in
                # Step 3 above), update the incumbent as needed, and
                # unconditionally cut the whole face below (every corner
                # ends up excluded either way: infeasible, or feasible
                # but not an improvement).
                for combo in range(1 << k_eff):
                    corner_mask = z_mask
                    for j, pos in enumerate(face_positions):
                        if (combo >> j) & 1:
                            corner_mask |= (1 << pos)
                        else:
                            corner_mask &= ~(1 << pos)
                    corner_masks.append(corner_mask)
                    if corner_mask == z_mask:
                        continue
                    corner_dict = {
                        name: (corner_mask >> i) & 1 for i, name in enumerate(sorted_ivars)
                    }
                    status, obj, sol = checker.evaluate(corner_dict)
                    if status == "unbounded":
                        if verbose:
                            print(f"[FP] unbounded (face-probe corner) at iter {k}")
                        return _finish("unbounded", k + 1)
                    if status == "feasible":
                        _record_feasible(k, obj, sol)

            # React to the probed corners according to cut_mode. Corner
            # generation/evaluation above is identical regardless of
            # cut_mode; only what we do with the result differs.
            if cut_mode == "structural":
                for cm in corner_masks:
                    idx.insert(cm)
                face_free = frozenset(face_positions)
                face_base = z_mask
                for pos in face_positions:
                    face_base &= ~(1 << pos)
                face_cut = CubeCut(z=face_base, free=face_free, n=n_ivars)
                face_cuts.append(face_cut)
                face_row = _add_cut_to_relax(relax, face_cut, sorted_ivars)
                added_rows.append(face_row)
                added_symbolic.add(face_cut)
                added_row_keys.add(_row_key(*face_row))
                total_cuts += 1
                face_probe_cuts += 1
                cuts_by_type["CubeCut"] = cuts_by_type.get("CubeCut", 0) + 1
                if verbose:
                    print(f"[FP]   face probe: added CubeCut over {sorted(face_positions)} "
                          f"(total {total_cuts})")
            elif cut_mode == "nogood":
                # One VertexCut per probed corner (z's own VertexCut is
                # added below, as always -- skip it here to avoid a
                # duplicate). Row-deduped per spec, unlike z's own cut
                # below (which the pre-M12 no-good branch never deduped).
                n_probe_added = 0
                for cm in corner_masks:
                    if cm == z_mask:
                        continue
                    coeffs, rhs = VertexCut(z=cm, n=n_ivars).to_inequality()
                    key = _row_key(coeffs, rhs)
                    if key in added_row_keys:
                        continue
                    added_row_keys.add(key)
                    added_rows.append(_add_row_to_relax(relax, coeffs, rhs, sorted_ivars))
                    total_cuts += 1
                    probe_vertex_cuts += 1
                    n_probe_added += 1
                    cuts_by_type["VertexCut"] = cuts_by_type.get("VertexCut", 0) + 1
                if verbose and n_probe_added:
                    print(f"[FP]   face probe (nogood): added {n_probe_added} corner "
                          f"VertexCuts (total {total_cuts})")
            # cut_mode == "none": probing only moves the incumbent (already
            # done above); no cuts of any kind.

        if cut_mode == "structural":
            idx.insert(z_mask)

            all_cuts = cuts_from_index(idx)
            if face_cuts:
                # Feed every face-probe cut added so far (not just this
                # iteration's) into the same dominance pass as the
                # natively-detected shapes. StructureIndex's own edge/
                # square/cube/star/tulip/propeller detection keeps
                # re-discovering the sub-faces of an already-probed face on
                # every later iteration (idx never forgets anything), and
                # dominance_prune only ever sees what's in `all_cuts` — so a
                # face_cut from iteration k that isn't still passed in at
                # iteration k+1 lets its own sub-structures look "not yet
                # dominated" again and leak back into the LP as new
                # (redundant, weaker) rows. Passing the full history here
                # lets the (now size-agnostic) Cube>Cube / Cube>Square /
                # Cube>Star / Cube>Tulip / Cube>Propeller rules keep pruning
                # them on every iteration, not just the one they were born in.
                all_cuts = all_cuts + face_cuts
            selected = select_cuts(all_cuts)

            # Dedup against everything added so far, two layers deep:
            # (1) added_symbolic -- skip to_inequality() entirely for a cut
            #     that is structurally identical (same type/anchor/dirs) to
            #     one already added, which is the common case and the one
            #     worth avoiding the O(n) dict-building cost for, since
            #     cuts_from_index regenerates the *entire* accumulated
            #     structure every iteration.
            # (2) added_row_keys -- exact-duplicate-row guard for the rare
            #     case of two different symbolic cuts producing the same
            #     (coeffs, rhs) row (e.g. a face-probe CubeCut built
            #     directly vs. the same face re-detected natively, when
            #     those don't already compare equal as dataclasses).
            n_added = 0
            for cut in selected:
                if cut in added_symbolic:
                    continue
                coeffs, rhs = cut.to_inequality()
                key = _row_key(coeffs, rhs)
                added_symbolic.add(cut)
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
                fallback_vc = VertexCut(z=z_mask, n=n_ivars)
                fallback_row = _add_cut_to_relax(relax, fallback_vc, sorted_ivars)
                added_rows.append(fallback_row)
                added_symbolic.add(fallback_vc)
                added_row_keys.add(_row_key(*fallback_row))
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

        elif cut_mode == "nogood":
            cut = VertexCut(z=z_mask, n=n_ivars)
            _add_cut_to_relax(relax, cut, sorted_ivars)
            total_cuts += 1
            cuts_by_type["VertexCut"] = cuts_by_type.get("VertexCut", 0) + 1

            if verbose:
                print(f"[FP]   added no-good cut (total {total_cuts})")

        # Step 6 — bound (optimality mode only) + projection LP
        if mode == "optimality":
            relax.setObjective(relax_orig_obj_expr, orig_sense)
            _apply_time_limit(relax, time_limit, _elapsed())
            relax.optimize()
            kind = _solve_status_kind(relax)
            if kind == "time_limit":
                return _finish("time_limit", k + 1)
            if kind == "optimal":
                bound = relax.ObjVal
                if _bound_proves_optimal(orig_sense, bound, U, _delta()):
                    bound = U  # tightened: proof means the true optimum is exactly U, gap=0
                    return _finish("optimal" if U is not None else "infeasible", k + 1)
            elif kind == "infeasible":
                if U is not None:
                    bound = U  # region now empty -> U is exactly optimal, gap=0
                return _finish("optimal" if U is not None else "infeasible", k + 1)
            # else ("unbounded"/"other"): can't conclude from this solve;
            # keep the previous `bound` and fall through to projection.

            _set_projection_objective(relax, int_vars, z_dict)
            _apply_time_limit(relax, time_limit, _elapsed())
            relax.optimize()
            kind2 = _solve_status_kind(relax)
            if kind2 == "time_limit":
                return _finish("time_limit", k + 1)
            if kind2 != "optimal":
                # Same feasible region as the bound solve just above (only
                # the objective differs), so an INFEASIBLE status here is
                # just as valid a proof as it would have been there; but
                # "unbounded"/"other" is a numerical anomaly, not a proof.
                if kind2 == "infeasible":
                    if U is not None:
                        bound = U
                    return _finish("optimal" if U is not None else "infeasible", k + 1)
                return _finish("numerical", k + 1)
        else:
            _set_projection_objective(relax, int_vars, z_dict)
            relax.optimize()
            if relax.Status != GRB.OPTIMAL:
                return FPResult(False, None, k + 1, "lp_infeasible", history,
                                total_cuts, n_perturbations, dict(cuts_by_type), fallback_cuts, face_probe_cuts,
                                probe_vertex_cuts=probe_vertex_cuts)

    if verbose:
        print(f"[FP] max_iter={max_iter} reached without feasible solution")
    if mode == "optimality":
        return _finish("max_iter", max_iter)
    return FPResult(False, None, max_iter, "max_iter", history,
                    total_cuts, n_perturbations, dict(cuts_by_type), fallback_cuts, face_probe_cuts,
                    probe_vertex_cuts=probe_vertex_cuts)
