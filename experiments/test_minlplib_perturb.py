import os
import sys
import time
from structural_fp import feasibility_pump
import gurobipy as gp

# 1. Load the instance file directly using Gurobi
# (gurobipy.read handles both .mps and .lp formats seamlessly)
# CLI usage: python experiments/test_minlplib_perturb.py [instance_path] [sheet_name] [instance_type]
# When an instance_path is given (e.g. by run_all_benchmarks.py --perturb), results are
# also appended to the Excel log; with no args it just prints, as before.
instance_path = sys.argv[1] if len(sys.argv) > 1 else \
    "/Users/isaac/Desktop/research_cutting_plane/experiments/miplib_benchmark/dano3_5.mps.gz" # Change this path to test other instances
model = gp.read(instance_path)

# 2. All three variants share the exact same perturbation settings, so the
#    only difference between runs is which cuts are added each iteration.
COMMON = dict(
    max_iter=500,
    max_perturbations=10,
    perturb_scale=0.5,
    random_seed=42,
    verbose=False,   # set True to print per-iteration z vectors (very long on big instances)
)

print("--- Plain FP + Perturbation ---")
t0 = time.perf_counter()
res_plain = feasibility_pump(model, **COMMON)
time_plain = time.perf_counter() - t0

print("\n--- No-Good Cuts + Perturbation ---")
t0 = time.perf_counter()
res_no_good = feasibility_pump(model, use_no_good_cuts=True, **COMMON)
time_no_good = time.perf_counter() - t0

print("\n--- Structural Cuts + Perturbation ---")
t0 = time.perf_counter()
res_cuts = feasibility_pump(
    model,
    use_structural_cuts=True,
    max_cuts_per_iter=30,
    cut_tol=1e-6,
    **COMMON,
)
time_cuts = time.perf_counter() - t0

def fmt_cuts(res):
    if not res.cuts_by_type:
        return "none"
    return ", ".join(f"{name.replace('Cut', '')}={n}" for name, n in sorted(res.cuts_by_type.items()))


# 3. Print out the final comparison summary
print("\n=== Benchmark Summary (all with perturbation) ===")
print(f"Plain+Perturb      | Iterations: {res_plain.iterations}, Reason: {res_plain.reason}, Perturbations: {res_plain.perturbations}")
print(f"No-Good+Perturb    | Iterations: {res_no_good.iterations}, Reason: {res_no_good.reason}, Cuts Added: {res_no_good.cuts_added}, Perturbations: {res_no_good.perturbations}")
print(f"Structural+Perturb | Iterations: {res_cuts.iterations}, Reason: {res_cuts.reason}, Cuts Added: {res_cuts.cuts_added}, Perturbations: {res_cuts.perturbations}")

print("\n=== Cuts added by type ===")
print(f"No-Good    | {fmt_cuts(res_no_good)}")
print(f"Structural | {fmt_cuts(res_cuts)}")

# 4. Append to the Excel log when driven with an explicit instance path
if len(sys.argv) > 1:
    from results_logger import log_results, PERTURB_SHEET
    from instance_info import classify_type

    instance_name = os.path.basename(instance_path)
    for ext in (".mps.gz", ".mps", ".lp.gz", ".lp"):
        if instance_name.endswith(ext):
            instance_name = instance_name[: -len(ext)]
            break
    sheet_name = sys.argv[2] if len(sys.argv) > 2 else PERTURB_SHEET
    instance_type = sys.argv[3] if len(sys.argv) > 3 else classify_type(model)

    def row(method, res, secs, with_cuts):
        r = {"method": method, "iterations": res.iterations, "cuts_added": res.cuts_added,
             "perturbations": res.perturbations, "reason": res.reason, "time_sec": secs}
        if with_cuts and res.cuts_by_type:
            r["note"] = "컷 종류: " + fmt_cuts(res)
        return r

    log_results(instance_name, instance_type, [
        row("Plain+Perturb", res_plain, time_plain, False),
        row("No-Good+Perturb", res_no_good, time_no_good, True),
        row("Structural+Perturb", res_cuts, time_cuts, True),
    ], sheet_name=sheet_name)
