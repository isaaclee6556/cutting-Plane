"""
Benchmark: plain FP vs FP+structural cuts on synthetic binary MILPs.

Usage:
    python experiments/benchmark.py
    python experiments/benchmark.py --csv results/bench.csv
    python experiments/benchmark.py --verbose

Columns:
    instance   : instance name
    n_vars     : number of binary variables
    n_cons     : number of constraints
    plain_iter : iterations to feasibility (plain FP)
    cuts_iter  : iterations to feasibility (FP + structural cuts)
    cuts_iter_p: iterations (FP + cuts + perturbation, max_perturbations=10)
    plain_ok   : 1 if plain FP found a feasible solution
    cuts_ok    : 1 if FP+cuts found a feasible solution
    cuts_added : structural cuts added by cuts run
    plain_t    : wall-clock time (plain FP), seconds
    cuts_t     : wall-clock time (FP + cuts), seconds
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import gurobipy as gp

# Add project root to path when running as a script from any CWD
_HERE = Path(__file__).parent.parent
sys.path.insert(0, str(_HERE))

from structural_fp.fp_loop import feasibility_pump
from structural_fp.io import (
    make_simple_binary_milp,
    make_covering_milp,
    make_random_covering_milp,
    make_fractional_cycle_milp,
)


# ---------------------------------------------------------------------------
# Instance registry
# ---------------------------------------------------------------------------

INSTANCES: list[tuple[str, Callable[[], gp.Model]]] = [
    ("simple_bin_4",      lambda: make_simple_binary_milp()),
    ("covering_6",        lambda: make_covering_milp(n=6)),
    ("covering_8",        lambda: make_covering_milp(n=8)),
    ("covering_10",       lambda: make_covering_milp(n=10)),
    ("covering_12",       lambda: make_covering_milp(n=12)),
    ("rnd_cover_n8_m4",   lambda: make_random_covering_milp(n=8,  m=4,  seed=0)),
    ("rnd_cover_n12_m6",  lambda: make_random_covering_milp(n=12, m=6,  seed=1)),
    ("rnd_cover_n16_m8",  lambda: make_random_covering_milp(n=16, m=8,  seed=2)),
    ("rnd_cover_n20_m10", lambda: make_random_covering_milp(n=20, m=10, seed=3)),
    ("cycle_trap",        lambda: make_fractional_cycle_milp()),
]


# ---------------------------------------------------------------------------
# Single-run wrapper
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    ok: bool
    iters: int
    reason: str
    cuts: int
    perturbs: int
    elapsed: float   # seconds


def run_one(
    make_fn: Callable[[], gp.Model],
    use_cuts: bool,
    max_perturb: int,
    max_iter: int = 200,
) -> RunResult:
    model = make_fn()
    t0 = time.perf_counter()
    res = feasibility_pump(
        model,
        max_iter=max_iter,
        use_structural_cuts=use_cuts,
        max_perturbations=max_perturb,
        random_seed=0,
    )
    elapsed = time.perf_counter() - t0
    return RunResult(
        ok=res.feasible,
        iters=res.iterations,
        reason=res.reason,
        cuts=res.cuts_added,
        perturbs=res.perturbations,
        elapsed=elapsed,
    )


def _model_stats(make_fn: Callable[[], gp.Model]) -> tuple[int, int]:
    m = make_fn()
    n_vars = m.NumVars
    n_cons = m.NumConstrs
    return n_vars, n_cons


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

_HDR = (
    f"{'Instance':<22} {'n':>4} {'m':>4} "
    f"{'Plain':>6} {'Cuts':>6} {'Cut+P':>6} "
    f"{'CutsAdd':>8} "
    f"{'PlainT(ms)':>11} {'CutsT(ms)':>10} {'Speedup':>8}"
)
_SEP = "-" * len(_HDR)


def _fmt_iters(r: RunResult) -> str:
    ok = "✓" if r.ok else r.reason[0].upper()
    return f"{r.iters}{ok}"


def print_row(name: str, n: int, m: int,
              rp: RunResult, rc: RunResult, rcp: RunResult) -> None:
    speedup = rp.elapsed / rc.elapsed if rc.elapsed > 1e-9 else float("inf")
    print(
        f"{name:<22} {n:>4} {m:>4} "
        f"{_fmt_iters(rp):>6} {_fmt_iters(rc):>6} {_fmt_iters(rcp):>6} "
        f"{rc.cuts:>8} "
        f"{rp.elapsed*1000:>11.2f} {rc.elapsed*1000:>10.2f} {speedup:>8.2f}x"
    )


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

CSV_FIELDS = [
    "instance", "n_vars", "n_cons",
    "plain_ok", "plain_iter", "plain_reason", "plain_t",
    "cuts_ok",  "cuts_iter",  "cuts_reason",  "cuts_t", "cuts_added",
    "cutsP_ok", "cutsP_iter", "cutsP_reason", "cutsP_perturbs",
]


def write_csv(path: str, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults written to {p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="FP benchmark: plain vs cuts")
    ap.add_argument("--csv",     metavar="FILE", help="save results to CSV")
    ap.add_argument("--verbose", action="store_true", help="show FP trace")
    ap.add_argument("--max-iter",    type=int, default=200)
    ap.add_argument("--max-perturb", type=int, default=10,
                    help="max perturbations for the cuts+perturb column")
    args = ap.parse_args()

    print(_HDR)
    print(_SEP)

    csv_rows: list[dict] = []

    for name, make_fn in INSTANCES:
        n, m = _model_stats(make_fn)
        rp  = run_one(make_fn, use_cuts=False, max_perturb=0,              max_iter=args.max_iter)
        rc  = run_one(make_fn, use_cuts=True,  max_perturb=0,              max_iter=args.max_iter)
        rcp = run_one(make_fn, use_cuts=True,  max_perturb=args.max_perturb, max_iter=args.max_iter)

        print_row(name, n, m, rp, rc, rcp)

        csv_rows.append({
            "instance":     name,
            "n_vars":       n,
            "n_cons":       m,
            "plain_ok":     int(rp.ok),
            "plain_iter":   rp.iters,
            "plain_reason": rp.reason,
            "plain_t":      f"{rp.elapsed:.6f}",
            "cuts_ok":      int(rc.ok),
            "cuts_iter":    rc.iters,
            "cuts_reason":  rc.reason,
            "cuts_t":       f"{rc.elapsed:.6f}",
            "cuts_added":   rc.cuts,
            "cutsP_ok":     int(rcp.ok),
            "cutsP_iter":   rcp.iters,
            "cutsP_reason": rcp.reason,
            "cutsP_perturbs": rcp.perturbs,
        })

    print(_SEP)
    print("Legend: iters + ✓(feasible) | C(cycle) | I(lp_infeasible) | M(max_iter)")

    if args.csv:
        write_csv(args.csv, csv_rows)


if __name__ == "__main__":
    main()
