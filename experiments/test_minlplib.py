import os
import sys
import time
from structural_fp import feasibility_pump
from results_logger import log_results, format_duration, MAIN_SHEET
from instance_info import classify_type
import gurobipy as gp

# 1. Load the instance file directly using Gurobi
# (gurobipy.read handles both .mps and .lp formats seamlessly)
# CLI usage: python experiments/test_minlplib.py [instance_path] [sheet_name] [instance_type]
#   instance_path : defaults to rmatr100-p10 if omitted
#   sheet_name    : Excel sheet to log into (default: "Benchmark Results")
#   instance_type : "IP"/"MIP" if already known (e.g. from the batch runner);
#                   otherwise computed here from the loaded model
instance_path = sys.argv[1] if len(sys.argv) > 1 else \
    r"C:\Users\isaac\OneDrive\Desktop\cutting-Plane-main\experiments\miplib_benchmark\rmatr100-p10.mps.gz"
sheet_name = sys.argv[2] if len(sys.argv) > 2 else MAIN_SHEET
instance_name = os.path.basename(instance_path)
for ext in (".mps.gz", ".mps", ".lp.gz", ".lp"):
    if instance_name.endswith(ext):
        instance_name = instance_name[: -len(ext)]
        break
model = gp.read(instance_path)
instance_type = sys.argv[3] if len(sys.argv) > 3 else classify_type(model)

# 2. Run Feasibility Pump and compare performance across settings
print("--- Plain Feasibility Pump ---")
t0 = time.perf_counter()
res_plain = feasibility_pump(model, max_iter=500)
time_plain = time.perf_counter() - t0

print("\n--- FP with Structural Cuts ---")
t0 = time.perf_counter()
res_cuts = feasibility_pump(model, use_structural_cuts=True, max_iter=500)
time_cuts = time.perf_counter() - t0

print("\n--- FP with No-Good Cuts Only ---")
t0 = time.perf_counter()
res_no_good = feasibility_pump(model, use_no_good_cuts=True, max_iter=500)
time_no_good = time.perf_counter() - t0

print("\n--- FP with Cuts & Perturbation ---")
t0 = time.perf_counter()
res_perturb = feasibility_pump(
    model,
    use_structural_cuts=True,
    max_perturbations=20,
    perturb_scale=0.3,
    max_cuts_per_iter=30,
    cut_tol=1e-6,
    random_seed=42,
    verbose=(len(sys.argv) <= 1),  # quiet when driven by the batch runner
)
time_perturb = time.perf_counter() - t0

# 3. Print out the final comparison summary
print("\n=== Benchmark Summary ===")
print(f"Plain        | Iterations: {res_plain.iterations}, Reason: {res_plain.reason}, Time: {format_duration(time_plain)}")
print(f"No-Good      | Iterations: {res_no_good.iterations}, Reason: {res_no_good.reason}, Cuts Added: {res_no_good.cuts_added}, Time: {format_duration(time_no_good)}")
print(f"Cuts         | Iterations: {res_cuts.iterations}, Reason: {res_cuts.reason}, Cuts Added: {res_cuts.cuts_added}, Time: {format_duration(time_cuts)}")
print(f"Cuts+Perturb | Iterations: {res_perturb.iterations}, Reason: {res_perturb.reason}, Cuts Added: {res_perturb.cuts_added}, Perturbations: {res_perturb.perturbations}, Time: {format_duration(time_perturb)}")

# 4. Append this instance's results to the running Excel log
log_results(instance_name, instance_type, [
    {"method": "Plain", "iterations": res_plain.iterations, "cuts_added": res_plain.cuts_added,
     "perturbations": res_plain.perturbations, "reason": res_plain.reason, "time_sec": time_plain},
    {"method": "No-Good", "iterations": res_no_good.iterations, "cuts_added": res_no_good.cuts_added,
     "perturbations": res_no_good.perturbations, "reason": res_no_good.reason, "time_sec": time_no_good},
    {"method": "Cuts", "iterations": res_cuts.iterations, "cuts_added": res_cuts.cuts_added,
     "perturbations": res_cuts.perturbations, "reason": res_cuts.reason, "time_sec": time_cuts},
    {"method": "Cuts+Perturb", "iterations": res_perturb.iterations, "cuts_added": res_perturb.cuts_added,
     "perturbations": res_perturb.perturbations, "reason": res_perturb.reason, "time_sec": time_perturb},
], sheet_name=sheet_name)
