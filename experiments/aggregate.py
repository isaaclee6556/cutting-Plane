"""
Aggregation (M13): turns the raw "Runs" sheet (one row per (instance,
method, seed)) into "Per Instance" and "Per Method" summary sheets.

The pure functions here (compute_per_instance / compute_per_method) take
and return plain lists of dicts, so they're testable without touching
Excel or Gurobi at all; CLI usage re-derives both summary sheets in-place
from an existing workbook's "Runs" sheet (the only one treated as source
of truth -- Per Instance/Per Method can always be rebuilt from it):

    python experiments/aggregate.py results/20261118_abc1234_matrix.xlsx
"""
from __future__ import annotations
import math
import statistics
import sys
from pathlib import Path

RUNS_SHEET = "Runs"
PER_INSTANCE_SHEET = "Per Instance"
PER_METHOD_SHEET = "Per Method"

# Rows with these `reason` values never happened as a real optimality-search
# attempt (either skipped on purpose, or the subprocess itself died/hung) --
# excluded from every statistic, and from "did this instance get a usable
# run for every method" comparability checks.
INVALID_REASONS = {"skipped_genint", "error", "timeout", None}


def _is_valid(row: dict) -> bool:
    return row.get("reason") not in INVALID_REASONS


def _mean(xs: list[float]):
    return statistics.fmean(xs) if xs else None


def _std(xs: list[float]):
    return statistics.pstdev(xs) if len(xs) > 1 else (0.0 if xs else None)


def shifted_geometric_mean(values: list[float], shift: float = 1.0):
    """Standard MIP-benchmarking stat: geometric mean of (v + shift), shift
    removed at the end, so a few near-zero times don't dominate/explode the
    mean the way a plain geometric mean would."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    log_sum = sum(math.log(v + shift) for v in vals)
    return math.exp(log_sum / len(vals)) - shift


def compute_per_instance(runs: list[dict]) -> list[dict]:
    """One row per (instance, method), aggregated over that pair's seeds."""
    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        groups.setdefault((r["instance"], r["method"]), []).append(r)

    out = []
    for (instance, method), group in sorted(groups.items()):
        valid = [r for r in group if _is_valid(r)]
        n_valid = len(valid)
        feasible = [r for r in valid if r.get("feasible")]
        optimal = [r for r in valid if r.get("proved_optimal")]
        times = [r["time_sec"] for r in valid if r.get("time_sec") is not None]
        iters = [r["iterations"] for r in valid if r.get("iterations") is not None]
        fft = [r["first_feasible_time"] for r in valid if r.get("first_feasible_time") is not None]
        gaps = [r["gap"] for r in valid if r.get("gap") is not None]

        # seed_varied: did reason/iterations/best_obj differ across this
        # (instance, method)'s valid seeds? (a quick, cheap divergence
        # signal -- not every field, just the ones most likely to expose a
        # seed-dependent bug or a genuinely unstable search).
        seed_varied = False
        if len(valid) > 1:
            keys = {(r.get("reason"), r.get("iterations"), r.get("best_obj")) for r in valid}
            seed_varied = len(keys) > 1

        out.append(dict(
            instance=instance, method=method,
            cut_mode=group[0].get("cut_mode"), probe_k=group[0].get("probe_k"),
            n_valid_seeds=n_valid,
            feasible_rate=(len(feasible) / n_valid) if n_valid else None,
            optimal_rate=(len(optimal) / n_valid) if n_valid else None,
            time_sec_mean=_mean(times), time_sec_std=_std(times),
            time_sec_min=min(times) if times else None, time_sec_max=max(times) if times else None,
            iterations_mean=_mean(iters), iterations_std=_std(iters),
            iterations_min=min(iters) if iters else None, iterations_max=max(iters) if iters else None,
            first_feasible_time_mean=_mean(fft), first_feasible_time_std=_std(fft),
            first_feasible_time_min=min(fft) if fft else None,
            first_feasible_time_max=max(fft) if fft else None,
            gap_mean=_mean(gaps),
            seed_varied=seed_varied,
        ))
    return out


def compute_per_method(runs: list[dict], per_instance: list[dict] | None = None) -> list[dict]:
    """
    One row per method, restricted to instances that have at least one
    valid run for *every* method in `runs` (so every method is judged on
    the same instance set) -- n_comparable_instances reports how many.
    """
    if per_instance is None:
        per_instance = compute_per_instance(runs)

    methods = sorted({r["method"] for r in runs})
    instances = sorted({r["instance"] for r in runs})

    by_pair = {(pi["instance"], pi["method"]): pi for pi in per_instance}
    comparable = [
        inst for inst in instances
        if all((by_pair.get((inst, m)) or {}).get("n_valid_seeds", 0) > 0 for m in methods)
    ]
    n_comparable = len(comparable)

    valid_runs_by_method: dict[str, list[dict]] = {m: [] for m in methods}
    for r in runs:
        if r["instance"] in comparable and _is_valid(r):
            valid_runs_by_method[r["method"]].append(r)

    out = []
    for m in methods:
        pi_rows = [by_pair[(inst, m)] for inst in comparable if (inst, m) in by_pair]
        vruns = valid_runs_by_method[m]
        all_runs_for_method = [r for r in runs if r["method"] == m]

        feasible_majority = sum(1 for pi in pi_rows if (pi["feasible_rate"] or 0) >= 0.5)
        optimal_majority = sum(1 for pi in pi_rows if (pi["optimal_rate"] or 0) >= 0.5)
        feasible_avg = sum(pi["feasible_rate"] or 0 for pi in pi_rows)
        times = [r["time_sec"] for r in vruns if r.get("time_sec") is not None]
        fft = [r["first_feasible_time"] for r in vruns if r.get("first_feasible_time") is not None]
        gaps = [r["gap"] for r in vruns if r.get("gap") is not None]
        iters = [r["iterations"] for r in vruns if r.get("iterations") is not None]

        n_varied = sum(1 for pi in per_instance if pi["method"] == m and pi.get("seed_varied"))
        n_with_seeds = sum(1 for pi in per_instance if pi["method"] == m and pi["n_valid_seeds"] > 1)

        out.append(dict(
            method=m,
            cut_mode=(all_runs_for_method[0].get("cut_mode") if all_runs_for_method else None),
            probe_k=(all_runs_for_method[0].get("probe_k") if all_runs_for_method else None),
            n_comparable_instances=n_comparable,
            feasible_count_majority=feasible_majority,
            feasible_count_avg=feasible_avg,
            optimal_count_majority=optimal_majority,
            time_sec_sgm=shifted_geometric_mean(times),
            first_feasible_time_sgm=shifted_geometric_mean(fft),
            gap_mean=_mean(gaps),
            iterations_mean=_mean(iters),
            seed_varied_fraction=(n_varied / n_with_seeds) if n_with_seeds else None,
        ))
    return out


# ---------------------------------------------------------------------------
# Excel I/O
# ---------------------------------------------------------------------------

def read_sheet_as_dicts(path: Path, sheet_name: str) -> list[dict]:
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True)
    if sheet_name not in wb.sheetnames:
        return []
    ws = wb[sheet_name]
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []
    headers = rows[0]
    out = []
    for row in rows[1:]:
        if not row or row[0] is None:
            continue
        d = dict(zip(headers, row))
        # cuts_by_type round-trips as a JSON string (dicts aren't a native
        # Excel cell type); leave it as a string here since aggregation
        # never needs to look inside it.
        out.append(d)
    return out


def write_dicts_as_sheet(wb, sheet_name: str, rows: list[dict], columns: list[str]) -> None:
    if sheet_name in wb.sheetnames:
        del wb[sheet_name]
    ws = wb.create_sheet(sheet_name)
    ws.append(columns)
    for r in rows:
        ws.append([r.get(c) for c in columns])


PER_INSTANCE_COLUMNS = [
    "instance", "method", "cut_mode", "probe_k", "n_valid_seeds",
    "feasible_rate", "optimal_rate",
    "time_sec_mean", "time_sec_std", "time_sec_min", "time_sec_max",
    "iterations_mean", "iterations_std", "iterations_min", "iterations_max",
    "first_feasible_time_mean", "first_feasible_time_std",
    "first_feasible_time_min", "first_feasible_time_max",
    "gap_mean", "seed_varied",
]
PER_METHOD_COLUMNS = [
    "method", "cut_mode", "probe_k", "n_comparable_instances",
    "feasible_count_majority", "feasible_count_avg", "optimal_count_majority",
    "time_sec_sgm", "first_feasible_time_sgm", "gap_mean", "iterations_mean",
    "seed_varied_fraction",
]


def rebuild_summary_sheets(path: Path) -> None:
    import openpyxl
    runs = read_sheet_as_dicts(path, RUNS_SHEET)
    if not runs:
        raise ValueError(f"{path} has no '{RUNS_SHEET}' sheet (or it's empty) -- nothing to aggregate")
    per_instance = compute_per_instance(runs)
    per_method = compute_per_method(runs, per_instance)

    wb = openpyxl.load_workbook(path)
    write_dicts_as_sheet(wb, PER_INSTANCE_SHEET, per_instance, PER_INSTANCE_COLUMNS)
    write_dicts_as_sheet(wb, PER_METHOD_SHEET, per_method, PER_METHOD_COLUMNS)
    wb.save(path)


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: python {sys.argv[0]} <results.xlsx>")
        raise SystemExit(1)
    path = Path(sys.argv[1])
    rebuild_summary_sheets(path)
    print(f"Rebuilt '{PER_INSTANCE_SHEET}' and '{PER_METHOD_SHEET}' in {path}")


if __name__ == "__main__":
    main()
