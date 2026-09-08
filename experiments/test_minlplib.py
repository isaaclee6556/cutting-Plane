from structural_fp import feasibility_pump
import gurobipy as gp

# 1. Load the instance file directly using Gurobi
# (gurobipy.read handles both .mps and .lp formats seamlessly)
instance_path = "/Users/isaac/Desktop/research_cutting_plane/experiments/miplib_benchmark/beasleyC3.mps.gz" # Change this path to test other instances
model = gp.read(instance_path)

# 2. Run Feasibility Pump and compare performance across settings
print("--- Plain Feasibility Pump ---")
res_plain = feasibility_pump(model, max_iter=500)

print("\n--- FP with Structural Cuts ---")
res_cuts = feasibility_pump(model, use_structural_cuts=True, max_iter=500)

print("\n--- FP with Cuts & Perturbation ---")
res_perturb = feasibility_pump(
    model,
    use_structural_cuts=True,
    max_perturbations=20,
    perturb_scale=0.3,
    max_cuts_per_iter=30,
    cut_tol=1e-6,
    random_seed=42,
    verbose=True
)

# 3. Print out the final comparison summary
print("\n=== Benchmark Summary ===")
print(f"Plain        | Iterations: {res_plain.iterations}, Reason: {res_plain.reason}")
print(f"Cuts         | Iterations: {res_cuts.iterations}, Reason: {res_cuts.reason}, Cuts Added: {res_cuts.cuts_added}")
print(f"Cuts+Perturb | Iterations: {res_perturb.iterations}, Reason: {res_perturb.reason}, Cuts Added: {res_perturb.cuts_added}, Perturbations: {res_perturb.perturbations}")