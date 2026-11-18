"""
Runs exactly one (instance, method, seed) job and prints one line of JSON
to stdout. Launched as its own subprocess by run_matrix.py so a hung or
crashed run can be killed/caught without taking down the batch (same
pattern as run_all_benchmarks.py's per-instance subprocess).

CLI: python _run_matrix_worker.py <instance_path> <method_name> <seed>
                                   <time_limit> <use_gurobi_seed:0|1>
"""
from __future__ import annotations
import datetime
import json
import sys
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from methods import METHODS, fp_kwargs  # noqa: E402
from structural_fp.fp_loop import feasibility_pump  # noqa: E402

EMPTY_ROW = dict(
    reason=None, feasible=False, proved_optimal=False,
    best_obj=None, bound=None, gap=None,
    first_feasible_iter=None, first_feasible_time=None,
    n_feasible_found=0, n_incumbent_updates=0,
    iterations=0, perturbations=0, cuts_added=0, cuts_by_type={},
    face_probe_cuts=0, probe_vertex_cuts=0, fallback_cuts=0,
    time_sec=0.0,
)


def _n_general_int(model: gp.Model) -> int:
    return sum(1 for v in model.getVars()
               if v.VType == GRB.INTEGER and not (v.LB >= 0 and v.UB <= 1))


def run_job(instance_path: str, method_name: str, seed: int,
            time_limit: float, use_gurobi_seed: bool) -> dict:
    model = gp.read(instance_path)
    model.setParam("OutputFlag", 0)
    model.setParam("Threads", 1)  # fair, reproducible per-run timing

    n_bin = model.NumBinVars
    # Gurobi's NumIntVars counts *all* integer-typed vars, binaries included
    # (confirmed empirically: a pure-binary instance has NumIntVars ==
    # NumBinVars, not 0) -- so continuous count is NumVars - NumIntVars, not
    # NumVars - NumBinVars - NumIntVars (that double-subtracts binaries and
    # can go negative).
    n_cont = model.NumVars - model.NumIntVars
    n_cons = model.NumConstrs
    inst_type = "IP" if n_cont == 0 else "MIP"
    n_genint = _n_general_int(model)

    spec = METHODS[method_name]
    cut_mode, probe_k = spec["cut_mode"], spec["probe_k"]

    meta = dict(
        type=inst_type, n_bin=n_bin, n_cont=n_cont, n_cons=n_cons,
        cut_mode=cut_mode, probe_k=probe_k,
        timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
    )

    # Cut-dependent methods can't run on general-integer instances
    # (feasibility_pump's own binary guard would raise); only cut_mode=
    # "none" with no probing (i.e. "Plain") is safe there, and that
    # restriction stays regardless of method (matches run_all_benchmarks.py's
    # existing skipped_genint policy).
    if n_genint > 0 and not (cut_mode == "none" and probe_k == 0):
        row = dict(EMPTY_ROW, reason="skipped_genint")
        return dict(meta, **row)

    kwargs = fp_kwargs(method_name, time_limit=time_limit, random_seed=42)
    if use_gurobi_seed:
        kwargs["gurobi_seed"] = seed
    res = feasibility_pump(model, **kwargs)
    row = dict(
        reason=res.reason, feasible=res.feasible, proved_optimal=res.proved_optimal,
        best_obj=res.best_obj, bound=res.bound, gap=res.gap,
        first_feasible_iter=res.first_feasible_iter, first_feasible_time=res.first_feasible_time,
        n_feasible_found=res.n_feasible_found, n_incumbent_updates=res.n_incumbent_updates,
        iterations=res.iterations, perturbations=res.perturbations,
        cuts_added=res.cuts_added, cuts_by_type=dict(res.cuts_by_type),
        face_probe_cuts=res.face_probe_cuts, probe_vertex_cuts=res.probe_vertex_cuts,
        fallback_cuts=res.fallback_cuts, time_sec=res.time_sec,
    )
    return dict(meta, **row)


def main() -> None:
    instance_path, method_name, seed_str, time_limit_str, use_gseed_str = sys.argv[1:6]
    out = run_job(
        instance_path, method_name, int(seed_str),
        float(time_limit_str), use_gseed_str == "1",
    )
    print(json.dumps(out))


if __name__ == "__main__":
    main()
