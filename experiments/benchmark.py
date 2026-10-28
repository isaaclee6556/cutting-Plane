"""
Benchmark: plain FP vs FP+no-good cuts vs FP+structural cuts (with and
without the face-probe step) on synthetic binary MILPs.

Every mode is run twice: without perturbation, and with the same perturbation
settings (--max-perturb, --perturb-scale, --seed) so modes stay comparable.

Usage:
    python experiments/benchmark.py
    python experiments/benchmark.py --csv results/bench.csv
    python experiments/benchmark.py --max-perturb 10 --perturb-scale 0.5
    python experiments/benchmark.py --list-instances
    python experiments/benchmark.py --instances rnd_cover_n16_m8 rnd_cover_n20_m10
    python experiments/benchmark.py --face-k 5

Table columns (iterations + outcome letter):
    Plain / NoGood / Struct / StructFP         : no perturbation
    Plain+P / NoGood+P / Struct+P / StructFP+P : with perturbation
    StructFP(+P) uses the face-probe step (use_face_probe, face_k) on top of
    structural cuts — see fp_loop.feasibility_pump docstring.
    Last column: cuts added by Struct+P and StructFP+P, by type
    ("[fp=N]" = how many of StructFP+P's cuts came from the face-probe step).

Outcome letters: ✓ feasible | C cycle | L lp_infeasible | M max_iter
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
from dataclasses import dataclass, field
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
INSTANCE_NAMES = [name for name, _ in INSTANCES]

# (key, mode, with_perturbation) — one run per entry per instance.
# mode: "plain" | "nogood" | "struct" | "structfp" (struct + face probe)
CONFIGS: list[tuple[str, str, bool]] = [
    ("plain",      "plain",    False),
    ("nogood",     "nogood",   False),
    ("struct",     "struct",   False),
    ("structFP",   "structfp", False),
    ("plainP",     "plain",    True),
    ("nogoodP",    "nogood",   True),
    ("structP",    "struct",   True),
    ("structFPP",  "structfp", True),
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
    by_type: dict[str, int] = field(default_factory=dict)
    face_probe_cuts: int = 0


def run_one(
    make_fn: Callable[[], gp.Model],
    mode: str,
    max_perturb: int,
    perturb_scale: float,
    seed: int,
    max_iter: int,
    face_k: int,
    verbose: bool = False,
) -> RunResult:
    model = make_fn()
    use_structural = mode in ("struct", "structfp")
    t0 = time.perf_counter()
    res = feasibility_pump(
        model,
        max_iter=max_iter,
        use_structural_cuts=use_structural,
        use_no_good_cuts=(mode == "nogood"),
        use_face_probe=(mode == "structfp"),
        face_k=face_k,
        max_perturbations=max_perturb,
        perturb_scale=perturb_scale,
        random_seed=seed,
        verbose=verbose,
    )
    elapsed = time.perf_counter() - t0
    return RunResult(
        ok=res.feasible,
        iters=res.iterations,
        reason=res.reason,
        cuts=res.cuts_added,
        perturbs=res.perturbations,
        elapsed=elapsed,
        by_type=dict(res.cuts_by_type),
        face_probe_cuts=res.face_probe_cuts,
    )


def _model_stats(make_fn: Callable[[], gp.Model]) -> tuple[int, int]:
    m = make_fn()
    return m.NumVars, m.NumConstrs


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

_HDR = (
    f"{'Instance':<20} {'n':>4} {'m':>4} "
    f"{'Plain':>6} {'NoGood':>7} {'Struct':>7} {'StructFP':>9} "
    f"{'Plain+P':>8} {'NoGood+P':>9} {'Struct+P':>9} {'StructFP+P':>11}  "
    f"cuts (Struct+P | StructFP+P)"
)
_SEP = "-" * (len(_HDR) + 20)


def _fmt_iters(r: RunResult) -> str:
    ok = "✓" if r.ok else {"cycle": "C", "lp_infeasible": "L", "max_iter": "M"}.get(r.reason, "?")
    return f"{r.iters}{ok}"


def fmt_by_type(r: RunResult) -> str:
    if not r.by_type:
        return "-"
    s = " ".join(f"{k.replace('Cut', '')}={v}" for k, v in sorted(r.by_type.items()))
    if r.face_probe_cuts:
        s += f" [fp={r.face_probe_cuts}]"
    return s


def print_row(name: str, n: int, m: int, r: dict[str, RunResult]) -> None:
    print(
        f"{name:<20} {n:>4} {m:>4} "
        f"{_fmt_iters(r['plain']):>6} {_fmt_iters(r['nogood']):>7} "
        f"{_fmt_iters(r['struct']):>7} {_fmt_iters(r['structFP']):>9} "
        f"{_fmt_iters(r['plainP']):>8} {_fmt_iters(r['nogoodP']):>9} "
        f"{_fmt_iters(r['structP']):>9} {_fmt_iters(r['structFPP']):>11}  "
        f"{fmt_by_type(r['structP'])} | {fmt_by_type(r['structFPP'])}"
    )


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

CSV_FIELDS = ["instance", "n_vars", "n_cons"]
for _key, _, _ in CONFIGS:
    CSV_FIELDS += [f"{_key}_ok", f"{_key}_iter", f"{_key}_reason", f"{_key}_t",
                   f"{_key}_cuts", f"{_key}_perturbs", f"{_key}_by_type", f"{_key}_face_probe_cuts"]


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
    ap = argparse.ArgumentParser(description="FP benchmark: plain vs no-good vs structural (+ face probe) cuts")
    ap.add_argument("--csv",     metavar="FILE", help="save results to CSV")
    ap.add_argument("--verbose", action="store_true", help="show FP trace")
    ap.add_argument("--max-iter",      type=int,   default=200)
    ap.add_argument("--max-perturb",   type=int,   default=10,
                    help="max perturbations for the +P columns (all modes)")
    ap.add_argument("--perturb-scale", type=float, default=0.5)
    ap.add_argument("--seed",          type=int,   default=0)
    ap.add_argument("--face-k",        type=int,   default=4,
                    help="face_k for the StructFP/StructFP+P columns (use_face_probe)")
    ap.add_argument("--instances", nargs="+", metavar="NAME",
                     help="run only these instances (see --list-instances for valid names); "
                          "default is all registered instances")
    ap.add_argument("--list-instances", action="store_true",
                     help="print the registered instance names and exit")
    args = ap.parse_args()

    if args.list_instances:
        for name in INSTANCE_NAMES:
            print(name)
        return

    if args.instances:
        unknown = [n for n in args.instances if n not in INSTANCE_NAMES]
        if unknown:
            print(f"Unknown instance name(s): {', '.join(unknown)}", file=sys.stderr)
            print(f"Valid names: {', '.join(INSTANCE_NAMES)}", file=sys.stderr)
            sys.exit(1)
        selected = [(n, fn) for n, fn in INSTANCES if n in args.instances]
    else:
        selected = INSTANCES

    print(_HDR)
    print(_SEP)

    csv_rows: list[dict] = []

    for name, make_fn in selected:
        n, m = _model_stats(make_fn)
        results: dict[str, RunResult] = {}
        for key, mode, with_perturb in CONFIGS:
            results[key] = run_one(
                make_fn, mode,
                max_perturb=args.max_perturb if with_perturb else 0,
                perturb_scale=args.perturb_scale,
                seed=args.seed,
                max_iter=args.max_iter,
                face_k=args.face_k,
                verbose=args.verbose,
            )

        print_row(name, n, m, results)

        row: dict = {"instance": name, "n_vars": n, "n_cons": m}
        for key, r in results.items():
            row[f"{key}_ok"] = int(r.ok)
            row[f"{key}_iter"] = r.iters
            row[f"{key}_reason"] = r.reason
            row[f"{key}_t"] = f"{r.elapsed:.6f}"
            row[f"{key}_cuts"] = r.cuts
            row[f"{key}_perturbs"] = r.perturbs
            row[f"{key}_by_type"] = ";".join(f"{k}={v}" for k, v in sorted(r.by_type.items()))
            row[f"{key}_face_probe_cuts"] = r.face_probe_cuts
        csv_rows.append(row)

    print(_SEP)
    print("Legend: iters + ✓(feasible) | C(cycle) | L(lp_infeasible) | M(max_iter)")
    print(f"+P columns: max_perturb={args.max_perturb}, scale={args.perturb_scale}, seed={args.seed}; face_k={args.face_k}")

    if args.csv:
        write_csv(args.csv, csv_rows)


if __name__ == "__main__":
    main()
