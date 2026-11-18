"""
Batch runner for the method matrix (M13/M13.1): every (instance, method,
seed) combination is one run, launched as its own subprocess via
_run_matrix_worker.py (real OS-level timeout + crash isolation -- a hung or
crashed run can't take the whole batch down), optionally in parallel.

Raw results are appended one JSON line at a time to a companion .jsonl file
next to --out (flushed+fsynced after every run -- see aggregate.append_jsonl)
rather than loaded/saved as xlsx per row: re-saving an xlsx workbook costs
more the bigger it already is, so appending one row at a time that way is
O(n) per row / O(n^2) overall (measured: ~3.2s per append at 10k existing
rows) and, since openpyxl's save isn't atomic, a run killed mid-save can
leave the whole workbook corrupted. The JSONL is the actual source of
truth; the xlsx (Runs + summary sheets) is rebuilt from it once at the end
of the batch, or any time via `python experiments/aggregate.py <out.xlsx>`.

Resumable: a (instance, method, seed) already present in the JSONL is
skipped, so an interrupted run can just be re-invoked with the same --out.

Usage:
    python experiments/run_matrix.py --limit 30 --seeds 0 1 --time-limit 120
    python experiments/run_matrix.py --seeds 0 1 2 3 4 --time-limit 300 --workers 4
    python experiments/run_matrix.py --out results/foo.xlsx --methods Plain NoGood
"""
from __future__ import annotations
import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from methods import METHODS, METHOD_ORDER, COMMON  # noqa: E402
from aggregate import (  # noqa: E402
    jsonl_path_for, read_jsonl, append_jsonl, rebuild_workbook,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKER = Path(__file__).parent / "_run_matrix_worker.py"
DEFAULT_DIR = Path(__file__).parent / "binary_instances"

EXT_PRIORITY = [".mps.gz", ".mps", ".lp.gz", ".lp"]


def git_short_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def base_name(p: Path) -> str:
    n = p.name
    for ext in EXT_PRIORITY:
        if n.endswith(ext):
            return n[: -len(ext)]
    return p.stem


def discover_instances(d: Path) -> list[Path]:
    groups: dict[str, dict[str, Path]] = {}
    for p in d.iterdir():
        if not p.is_file():
            continue
        for ext in EXT_PRIORITY:
            if p.name.endswith(ext):
                groups.setdefault(p.name[: -len(ext)], {})[ext] = p
                break
    chosen = []
    for base in sorted(groups):
        for ext in EXT_PRIORITY:
            if ext in groups[base]:
                chosen.append(groups[base][ext])
                break
    return chosen


def run_one(instance_path: Path, method: str, seed: int, time_limit: float, use_gseed: bool) -> dict:
    try:
        proc = subprocess.run(
            [sys.executable, str(WORKER), str(instance_path), method, str(seed),
             str(time_limit), "1" if use_gseed else "0"],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
            timeout=time_limit + 60,
        )
    except subprocess.TimeoutExpired:
        return dict(reason="timeout", feasible=False, proved_optimal=False,
                    best_obj=None, bound=None, gap=None,
                    first_feasible_iter=None, first_feasible_time=None,
                    n_feasible_found=0, n_incumbent_updates=0,
                    iterations=0, perturbations=0, cuts_added=0, cuts_by_type={},
                    face_probe_cuts=0, probe_vertex_cuts=0, fallback_cuts=0, time_sec=0.0,
                    type=None, n_bin=None, n_cont=None, n_cons=None,
                    cut_mode=METHODS[method]["cut_mode"], probe_k=METHODS[method]["probe_k"],
                    timestamp=datetime.datetime.now().isoformat(timespec="seconds"))
    if proc.returncode != 0 or not proc.stdout.strip():
        note = (proc.stderr or "").strip().splitlines()
        return dict(reason="error", feasible=False, proved_optimal=False,
                    best_obj=None, bound=None, gap=None,
                    first_feasible_iter=None, first_feasible_time=None,
                    n_feasible_found=0, n_incumbent_updates=0,
                    iterations=0, perturbations=0, cuts_added=0, cuts_by_type={},
                    face_probe_cuts=0, probe_vertex_cuts=0, fallback_cuts=0, time_sec=0.0,
                    type=None, n_bin=None, n_cont=None, n_cons=None,
                    cut_mode=METHODS[method]["cut_mode"], probe_k=METHODS[method]["probe_k"],
                    timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
                    error_note=(note[-1] if note else f"exit {proc.returncode}")[:300])
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _load_existing_keys(jsonl_path: Path) -> set[tuple]:
    rows = read_jsonl(jsonl_path)
    return {(r["instance"], r["method"], int(r["seed"])) for r in rows}


def _write_config_sheet(path: Path, args: argparse.Namespace) -> None:
    import gurobipy as gp
    import openpyxl
    if path.exists():
        wb = openpyxl.load_workbook(path)
    else:
        wb = openpyxl.Workbook()
        wb.remove(wb.active)
    if "Config" in wb.sheetnames:
        del wb["Config"]
    ws = wb.create_sheet("Config")
    ws.append(["key", "value"])
    rows = [
        ("date", datetime.date.today().isoformat()),
        ("git_commit", git_short_hash()),
        ("gurobi_version", ".".join(map(str, gp.gurobi.version()))),
        ("dir", str(args.dir)),
        ("limit", args.limit),
        ("instances", ",".join(args.instances) if args.instances else None),
        ("seeds", ",".join(map(str, args.seeds))),
        ("methods", ",".join(args.methods)),
        ("time_limit", args.time_limit),
        ("workers", args.workers),
        ("use_gurobi_seed", args.use_gseed),
        ("COMMON", json.dumps(COMMON)),
        ("METHODS", json.dumps(METHODS)),
    ]
    for k, v in rows:
        ws.append([k, v])
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR,
                     help=f"instance directory to scan (default: {DEFAULT_DIR.name}/)")
    ap.add_argument("--limit", type=int, default=None,
                     help="only use the first N discovered instances")
    ap.add_argument("--instances", nargs="*", default=None,
                     help="explicit instance basenames (overrides --limit's selection, not --dir)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--methods", nargs="+", default=METHOD_ORDER, choices=METHOD_ORDER)
    ap.add_argument("--time-limit", type=float, default=300.0,
                     help="per-run wall-clock budget in seconds, handed to feasibility_pump "
                          "and to the subprocess timeout (+60s)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tag", type=str, default="matrix")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    ap.add_argument("--gurobi-seed", dest="use_gseed", action="store_true", default=True)
    ap.add_argument("--no-gurobi-seed", dest="use_gseed", action="store_false")
    args = ap.parse_args()

    instances = discover_instances(args.dir)
    if args.instances:
        want = set(args.instances)
        instances = [p for p in instances if base_name(p) in want]
    elif args.limit:
        instances = instances[: args.limit]

    if not instances:
        print(f"No instances found under {args.dir} (after filtering).")
        return

    out_path = args.out or (
        Path(__file__).parent / "results" /
        f"{datetime.date.today():%Y%m%d}_{git_short_hash()}_{args.tag}.xlsx"
    )
    jsonl_path = jsonl_path_for(out_path)
    done = _load_existing_keys(jsonl_path)

    jobs = [
        (p, m, s)
        for p in instances for m in args.methods for s in args.seeds
        if (base_name(p), m, s) not in done
    ]
    total_jobs = len(instances) * len(args.methods) * len(args.seeds)
    worst_case_hours = len(jobs) * args.time_limit / max(1, args.workers) / 3600

    print(f"{len(instances)} instances x {len(args.methods)} methods x {len(args.seeds)} seeds "
          f"= {total_jobs} runs ({len(done)} already in {jsonl_path.name}, {len(jobs)} to do).")
    print(f"Worst case: ~{worst_case_hours:.2f}h with {args.workers} workers "
          f"(time_limit={args.time_limit}s/run).")
    if not jobs:
        print("Nothing to do.")
        if done:
            _write_config_sheet(out_path, args)
            rebuild_workbook(out_path)
        return
    if not args.yes:
        resp = input("Proceed? [y/N] ")
        if resp.strip().lower() != "y":
            print("Aborted.")
            return

    t_start = time.perf_counter()
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(run_one, p, m, s, args.time_limit, args.use_gseed): (p, m, s)
            for p, m, s in jobs
        }
        for fut in as_completed(futs):
            p, m, s = futs[fut]
            result = fut.result()
            row = dict(result)
            row["instance"] = base_name(p)
            row["method"] = m
            row.setdefault("cut_mode", METHODS[m]["cut_mode"])
            row.setdefault("probe_k", METHODS[m]["probe_k"])
            row["seed"] = s
            row.setdefault("git_commit", git_short_hash())
            append_jsonl(jsonl_path, row)
            n_done += 1
            print(f"[{n_done}/{len(jobs)}] {row['instance']} / {m} / seed={s} "
                  f"-> {row.get('reason')} ({row.get('time_sec', 0):.2f}s)")

    elapsed = time.perf_counter() - t_start
    print(f"Done: {n_done} runs in {elapsed:.1f}s ({elapsed / max(1, n_done):.2f}s/run avg).")

    _write_config_sheet(out_path, args)
    rebuild_workbook(out_path)
    print(f"Wrote {out_path} (raw rows: {jsonl_path})")


if __name__ == "__main__":
    main()
