from structural_fp import feasibility_pump
import gurobipy as gp

# 1. Load the instance file directly using Gurobi
# (gurobipy.read handles both .mps and .lp formats seamlessly)
instance_path = "/Users/isaac/Desktop/research_cutting_plane/experiments/miplib_benchmark/dano3_5.mps.gz" # Change this path to test other instances
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
res_plain = feasibility_pump(model, **COMMON)

print("\n--- No-Good Cuts + Perturbation ---")
res_no_good = feasibility_pump(model, use_no_good_cuts=True, **COMMON)

print("\n--- Structural Cuts + Perturbation ---")
res_cuts = feasibility_pump(
    model,
    use_structural_cuts=True,
    max_cuts_per_iter=30,
    cut_tol=1e-6,
    **COMMON,
)

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
