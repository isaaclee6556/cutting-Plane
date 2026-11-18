"""
Aggregation (M13 / M13.1): turns raw per-run rows (one per (instance,
method, seed)) into "Per Instance", "Per Method", and "Paired Stats"
summary sheets.

The raw rows live in a JSONL file (run_matrix.py appends one line per
completed run, flushed+fsynced immediately -- see its module docstring for
why), not in the xlsx directly; the "Runs" sheet in the output workbook is
just a materialized view of that JSONL, rebuilt whenever this module's CLI
or run_matrix.py itself calls rebuild_workbook(). The actual statistics
(compute_per_instance / compute_per_method / compute_paired_stats) are pure
functions over plain lists of dicts, so they're testable without touching
Excel, Gurobi, or even a JSONL file at all.

CLI usage -- rebuilds every sheet in <results.xlsx> from its companion
<results.xlsx>.jsonl (which must exist; that's the source of truth):

    python experiments/aggregate.py results/20261118_abc1234_matrix.xlsx
"""
from __future__ import annotations
import json
import math
import statistics
import sys
from pathlib import Path

RUNS_SHEET = "Runs"
PER_INSTANCE_SHEET = "Per Instance"
PER_METHOD_SHEET = "Per Method"
PAIRED_STATS_SHEET = "Paired Stats"
CONFIG_SHEET = "Config"

# Method pairs compared head-to-head in "Paired Stats": structural cuts vs.
# no-good cuts at the same probe_k, since that's the comparison the whole
# structural-cuts hypothesis rests on (does the extra structure help beyond
# what probing alone already buys you).
PAIRED_METHODS = [
    ("Struct", "NoGood"),
    ("Struct+P2", "NoGood+P2"),
    ("Struct+P3", "NoGood+P3"),
    ("Struct+P4", "NoGood+P4"),
]

# Rows with these `reason` values never happened as a real optimality-search
# attempt (either skipped on purpose, or the subprocess itself died/hung) --
# excluded from every statistic, and from "did this instance get a usable
# run for every method" comparability checks.
INVALID_REASONS = {"skipped_genint", "error", "timeout", None}

# Reasons for which "gap" is not a meaningful concept at all: a proven-
# infeasible instance has no optimum to measure a gap against, and an
# unbounded relaxation is a degenerate/ill-posed case, not a search that
# failed to close a gap.
NO_GAP_REASONS = {"infeasible", "unbounded_relaxation"}


def _is_valid(row: dict) -> bool:
    return row.get("reason") not in INVALID_REASONS


def _mean(xs: list[float]):
    return statistics.fmean(xs) if xs else None


def _std(xs: list[float]):
    return statistics.pstdev(xs) if len(xs) > 1 else (0.0 if xs else None)


def _median(xs: list[float]):
    return statistics.median(xs) if xs else None


def shifted_geometric_mean(values: list[float], shift: float = 1.0):
    """Standard MIP-benchmarking stat: geometric mean of (v + shift), shift
    removed at the end, so a few near-zero times don't dominate/explode the
    mean the way a plain geometric mean would."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    log_sum = sum(math.log(v + shift) for v in vals)
    return math.exp(log_sum / len(vals)) - shift


def _effective_gap(row: dict):
    """
    gap, with the "only successful runs count" bias removed: a valid run
    that never found any incumbent is the worst possible outcome for a
    feasible instance, so it counts as gap=1.0 rather than being silently
    dropped from the average (which would otherwise make a method's
    gap_mean look better the more often it simply fails to find anything).
    Proven-infeasible/unbounded-relaxation rows have no such outcome to
    penalize -- gap isn't a meaningful concept for them at all, so they're
    excluded (return None) rather than forced to either 0 or 1.
    """
    if row.get("gap") is not None:
        return row["gap"]
    if row.get("reason") in NO_GAP_REASONS:
        return None
    return 1.0 if _is_valid(row) else None


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
        gaps_eff = [g for g in (_effective_gap(r) for r in valid) if g is not None]

        # seed_varied (M13.1): did (reason, best_obj, proved_optimal) differ
        # across this (instance, method)'s valid seeds? time_limit rows are
        # excluded from the comparison -- two runs that both simply ran out
        # of time report different iteration counts almost by definition
        # (Gurobi's own seed-dependent tie-breaking alone guarantees it),
        # which isn't the kind of "did something actually differ" signal
        # this flag is meant to catch. best_obj is rounded to 6dp since it's
        # a float and seed-dependent LP tie-breaking can perturb the last
        # few bits without the result being meaningfully different.
        comparable = [r for r in valid if r.get("reason") != "time_limit"]
        seed_varied = False
        if len(comparable) > 1:
            keys = {
                (r.get("reason"),
                 round(r["best_obj"], 6) if r.get("best_obj") is not None else None,
                 r.get("proved_optimal"))
                for r in comparable
            }
            seed_varied = len(keys) > 1

        out.append(dict(
            instance=instance, method=method,
            cut_mode=group[0].get("cut_mode"), probe_k=group[0].get("probe_k"),
            n_valid_seeds=n_valid,
            feasible_rate=(len(feasible) / n_valid) if n_valid else None,
            optimal_rate=(len(optimal) / n_valid) if n_valid else None,
            time_sec_mean=_mean(times), time_sec_median=_median(times), time_sec_std=_std(times),
            time_sec_min=min(times) if times else None, time_sec_max=max(times) if times else None,
            iterations_mean=_mean(iters), iterations_std=_std(iters),
            iterations_min=min(iters) if iters else None, iterations_max=max(iters) if iters else None,
            first_feasible_time_mean=_mean(fft), first_feasible_time_std=_std(fft),
            first_feasible_time_min=min(fft) if fft else None,
            first_feasible_time_max=max(fft) if fft else None,
            gap_mean=_mean(gaps_eff),
            seed_varied=seed_varied,
        ))
    return out


def compute_per_method(runs: list[dict], per_instance: list[dict] | None = None,
                        time_limit: float | None = None) -> list[dict]:
    """
    One row per method, restricted to instances that have at least one
    valid run for *every* method in `runs` (so every method is judged on
    the same instance set) -- n_comparable_instances reports how many.

    time_limit (read from the "Config" sheet by rebuild_workbook) is only
    used for time_sec_par1 (PAR1 -- unsolved runs counted at exactly the
    time budget, the standard way of not letting "gave up early" look
    better than "timed out slowly" in a mean); every other stat ignores it.
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
        times_proved = [r["time_sec"] for r in vruns if r.get("proved_optimal") and r.get("time_sec") is not None]
        fft = [r["first_feasible_time"] for r in vruns if r.get("first_feasible_time") is not None]
        gaps_eff = [g for g in (_effective_gap(r) for r in vruns) if g is not None]
        gaps_unproved = [g for g in (_effective_gap(r) for r in vruns if not r.get("proved_optimal")) if g is not None]
        iters = [r["iterations"] for r in vruns if r.get("iterations") is not None]

        par1_vals = None
        if time_limit is not None:
            par1_vals = [
                r["time_sec"] if r.get("proved_optimal") else time_limit
                for r in vruns if r.get("time_sec") is not None or not r.get("proved_optimal")
            ]

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
            time_sec_sgm_proved=shifted_geometric_mean(times_proved),
            time_sec_par1=shifted_geometric_mean(par1_vals) if par1_vals is not None else None,
            first_feasible_time_sgm=shifted_geometric_mean(fft),
            gap_mean=_mean(gaps_eff),
            gap_mean_unproved=_mean(gaps_unproved),
            iterations_mean=_mean(iters),
            seed_varied_fraction=(n_varied / n_with_seeds) if n_with_seeds else None,
        ))
    return out


def compute_paired_stats(runs: list[dict], per_instance: list[dict] | None = None) -> list[dict]:
    """
    Head-to-head comparison for each (Struct[+Pk], NoGood[+Pk]) pair (see
    PAIRED_METHODS): per-instance win/tie/loss (1st: optimal_rate, 2nd:
    median time_sec) plus a Wilcoxon signed-rank test on each instance's
    own mean log(time_sec + 1), paired across the two methods.

    scipy is imported lazily (same reason as openpyxl elsewhere in this
    module): keeps every *other* function usable without it installed.
    """
    if per_instance is None:
        per_instance = compute_per_instance(runs)
    by_pair = {(pi["instance"], pi["method"]): pi for pi in per_instance}

    log_time_by_pair: dict[tuple, float] = {}
    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        groups.setdefault((r["instance"], r["method"]), []).append(r)
    for key, group in groups.items():
        vals = [math.log(r["time_sec"] + 1) for r in group if _is_valid(r) and r.get("time_sec") is not None]
        if vals:
            log_time_by_pair[key] = _mean(vals)

    out = []
    for method_a, method_b in PAIRED_METHODS:
        all_instances = sorted({inst for (inst, m) in by_pair if m in (method_a, method_b)})
        wins = ties = losses = 0
        paired_log_a: list[float] = []
        paired_log_b: list[float] = []

        for inst in all_instances:
            a = by_pair.get((inst, method_a))
            b = by_pair.get((inst, method_b))
            if not a or not b or not a["n_valid_seeds"] or not b["n_valid_seeds"]:
                continue
            oa, ob = a["optimal_rate"] or 0.0, b["optimal_rate"] or 0.0
            if abs(oa - ob) > 1e-12:
                wins, losses = (wins + 1, losses) if oa > ob else (wins, losses + 1)
            else:
                ta, tb = a.get("time_sec_median"), b.get("time_sec_median")
                if ta is not None and tb is not None and abs(ta - tb) > 1e-12:
                    wins, losses = (wins + 1, losses) if ta < tb else (wins, losses + 1)
                else:
                    ties += 1

            la, lb = log_time_by_pair.get((inst, method_a)), log_time_by_pair.get((inst, method_b))
            if la is not None and lb is not None:
                paired_log_a.append(la)
                paired_log_b.append(lb)

        p_value = None
        if len(paired_log_a) >= 1:
            if any(abs(a - b) > 1e-12 for a, b in zip(paired_log_a, paired_log_b)):
                try:
                    from scipy.stats import wilcoxon
                    _, p_value = wilcoxon(paired_log_a, paired_log_b)
                except (ImportError, ValueError):
                    p_value = None
            else:
                p_value = 1.0  # every pair identical -> no evidence of a difference

        out.append(dict(
            method_a=method_a, method_b=method_b,
            n_instances_compared=wins + ties + losses,
            wins=wins, ties=ties, losses=losses,
            n_paired_for_wilcoxon=len(paired_log_a),
            wilcoxon_p=p_value,
        ))
    return out


# ---------------------------------------------------------------------------
# JSONL storage (source of truth for raw run rows)
# ---------------------------------------------------------------------------

def jsonl_path_for(out_path: Path) -> Path:
    return Path(str(out_path) + ".jsonl")


def read_jsonl(path: Path) -> list[dict]:
    """Reads every row, skipping (and warning about) any line that fails to
    parse -- e.g. a half-written final line from a process killed mid-append
    (run_matrix.py flushes+fsyncs after every row specifically so this is
    the only way a line can end up corrupt)."""
    if not path.exists():
        return []
    rows = []
    with open(path, "r") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"warning: {path}:{i} is not valid JSON, skipping", file=sys.stderr)
    return rows


def append_jsonl(path: Path, row: dict) -> None:
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


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
        out.append(dict(zip(headers, row)))
    return out


def write_dicts_as_sheet(wb, sheet_name: str, rows: list[dict], columns: list[str]) -> None:
    if sheet_name in wb.sheetnames:
        del wb[sheet_name]
    ws = wb.create_sheet(sheet_name)
    ws.append(columns)
    for r in rows:
        ws.append([r.get(c) for c in columns])


RUNS_COLUMNS = [
    "instance", "type", "n_bin", "n_cont", "n_cons",
    "method", "cut_mode", "probe_k", "seed",
    "reason", "feasible", "proved_optimal", "best_obj", "bound", "gap",
    "first_feasible_iter", "first_feasible_time", "n_feasible_found", "n_incumbent_updates",
    "iterations", "perturbations", "cuts_added", "cuts_by_type",
    "face_probe_cuts", "probe_vertex_cuts", "fallback_cuts",
    "time_sec", "git_commit", "timestamp",
]
PER_INSTANCE_COLUMNS = [
    "instance", "method", "cut_mode", "probe_k", "n_valid_seeds",
    "feasible_rate", "optimal_rate",
    "time_sec_mean", "time_sec_median", "time_sec_std", "time_sec_min", "time_sec_max",
    "iterations_mean", "iterations_std", "iterations_min", "iterations_max",
    "first_feasible_time_mean", "first_feasible_time_std",
    "first_feasible_time_min", "first_feasible_time_max",
    "gap_mean", "seed_varied",
]
PER_METHOD_COLUMNS = [
    "method", "cut_mode", "probe_k", "n_comparable_instances",
    "feasible_count_majority", "feasible_count_avg", "optimal_count_majority",
    "time_sec_sgm", "time_sec_sgm_proved", "time_sec_par1", "first_feasible_time_sgm",
    "gap_mean", "gap_mean_unproved", "iterations_mean",
    "seed_varied_fraction",
]
PAIRED_STATS_COLUMNS = [
    "method_a", "method_b", "n_instances_compared", "wins", "ties", "losses",
    "n_paired_for_wilcoxon", "wilcoxon_p",
]


def _read_config_time_limit(wb) -> float | None:
    if CONFIG_SHEET not in wb.sheetnames:
        return None
    for row in wb[CONFIG_SHEET].iter_rows(min_row=2, values_only=True):
        if row and row[0] == "time_limit" and row[1] is not None:
            try:
                return float(row[1])
            except (TypeError, ValueError):
                return None
    return None


def rebuild_workbook(out_path: Path) -> None:
    """
    Rebuilds every sheet (Runs, Per Instance, Per Method, Paired Stats) in
    `out_path` from its companion JSONL file (jsonl_path_for(out_path)) --
    the JSONL is the actual source of truth; the "Runs" sheet is just its
    latest materialized view, not something edited directly. Leaves the
    "Config" sheet alone if one already exists (run_matrix.py writes it
    separately, since it needs the CLI args, not just the raw rows).
    """
    import openpyxl
    jsonl_path = jsonl_path_for(out_path)
    runs = read_jsonl(jsonl_path)
    if not runs:
        raise ValueError(f"{jsonl_path} has no rows -- nothing to aggregate")

    if out_path.exists():
        wb = openpyxl.load_workbook(out_path)
    else:
        wb = openpyxl.Workbook()
        wb.remove(wb.active)
    time_limit = _read_config_time_limit(wb)

    # cuts_by_type is a dict in the JSONL; Excel cells can't hold one, so it
    # round-trips as a JSON string in the Runs sheet specifically -- on a
    # copy of each row, since `runs` itself (with the real dict) is what
    # the stats functions below need.
    runs_for_sheet = [
        dict(r, cuts_by_type=json.dumps(r["cuts_by_type"])) if isinstance(r.get("cuts_by_type"), dict) else r
        for r in runs
    ]
    write_dicts_as_sheet(wb, RUNS_SHEET, runs_for_sheet, RUNS_COLUMNS)

    per_instance = compute_per_instance(runs)
    per_method = compute_per_method(runs, per_instance, time_limit=time_limit)
    paired = compute_paired_stats(runs, per_instance)
    write_dicts_as_sheet(wb, PER_INSTANCE_SHEET, per_instance, PER_INSTANCE_COLUMNS)
    write_dicts_as_sheet(wb, PER_METHOD_SHEET, per_method, PER_METHOD_COLUMNS)
    write_dicts_as_sheet(wb, PAIRED_STATS_SHEET, paired, PAIRED_STATS_COLUMNS)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out_path)


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: python {sys.argv[0]} <results.xlsx>  (needs a companion .jsonl next to it)")
        raise SystemExit(1)
    path = Path(sys.argv[1])
    rebuild_workbook(path)
    print(f"Rebuilt '{RUNS_SHEET}', '{PER_INSTANCE_SHEET}', '{PER_METHOD_SHEET}', "
          f"'{PAIRED_STATS_SHEET}' in {path}")


if __name__ == "__main__":
    main()
