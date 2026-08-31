# Structural Feasibility Pump — Implementation Spec (Python + Gurobi)

Source: Cornuéjols & Lee (2018) structural cuts for `conv({0,1}^n \ X)`, applied
incrementally inside a feasibility-pump-style heuristic.

This document is meant to be dropped into a Claude Code project as context.
It specifies data structures, algorithms, and a recommended file layout so
implementation can proceed milestone by milestone with tests at each step.

---

## 1. Goal

Given a MILP `min{c^T x : Ax <= b, x_i in Z for i in I}`, run a feasibility-pump-like
loop that:

1. Solves LP relaxations with Gurobi.
2. Rounds fractional points to binary/integer candidates.
3. Maintains the set `X` of all binary points visited during rounding.
4. Incrementally detects substructures in the skeleton graph `G(X)`
   (vertex, edge, square, star, cube, tulip, propeller) using bitset/hash
   operations — **not** generic subgraph isomorphism.
5. Generates valid cuts from maximal detected structures, filters by
   violation/efficacy, and adds only the useful ones to the projection LP.
6. Repeats until an integral point in `P` is found (or a cycle-breaking
   perturbation is triggered).

---

## 2. Core data structures

Represent each binary point as a Python `int` bitmask (n <= ~62 fits in a
native int; use `numpy` uint64 arrays or `gmpy2` if n is larger).

```python
Point = int  # bitmask, bit i = value of x_i

class StructureIndex:
    H: set[Point]                     # all visited points
    D: dict[Point, int]               # neighbor-direction bitmask per point
    Lz: dict[Point, dict[int, set[int]]]  # link graph adjacency per point:
                                           # Lz[z][i] = set of j with z^(1<<i)^(1<<j) in H
```

Key operations (all O(1) or O(popcount)):

- `flip(z, i) -> Point`: `z ^ (1 << i)`
- `flip2(z, i, j) -> Point`: `z ^ (1 << i) ^ (1 << j)`
- membership test: `q in H`
- `D(z)` as bitmask, iterate set bits with `while mask: i = (mask & -mask).bit_length()-1; mask &= mask-1`

## 3. Structure catalog and cut formulas

For a visited point `z` and coordinate set, define per-coordinate distance term:

```
delta_i(x; z) = z_i*(1-x_i) + (1-z_i)*x_i
```

(equals 1 exactly when `x_i != z_i` at a binary `x`).

| Structure  | Required points                                  | Cut (>= rhs)                                                                 |
|------------|---------------------------------------------------|-------------------------------------------------------------------------------|
| Vertex     | z                                                   | sum_i delta_i(x;z) >= 1                                                      |
| Edge       | z, z^a                                              | sum_{i != a} delta_i(x;z) >= 1                                               |
| Square     | z, z^a, z^b, z^{ab}                                 | sum_{i not in {a,b}} delta_i(x;z) >= 1                                       |
| Star       | z, {z^i : i in L}, \|L\| >= 3                         | sum_{i in L} delta_i(x;z) >= \|L\| - 1                                          |
| Cube       | z^J for J subset {a,b,c}                            | sum_{i not in {a,b,c}} delta_i(x;z) >= 1                                     |
| Tulip      | z, z^a, z^b, z^c, z^{ab}, z^{bc}, z^{ca} (+leaves L) | sum_{i in K} delta_i + 2*sum_{i in L} delta_i + 3*sum_{i not in K∪L} delta_i >= 3 |
| Propeller  | axis z,z^a, blade B (\|B\|>=3)                        | sum_{i in B} delta_i + 2*sum_{i not in B∪{a}} delta_i >= 2                    |

Maximal-set dominance: always use the **largest available** leaf/blade set
for a given center/axis — it dominates (is at least as strong as) any subset.
Only store one cut per center (star), per axis (propeller), per core (tulip).

## 4. Incremental insertion algorithm

```
INSERT(p, H, D, Lz):
    H.add(p)
    D.setdefault(p, 0)
    new_squares = []
    affected_centers = {p}

    # 1. new edges
    for i in range(n):
        q = p ^ (1 << i)
        if q in H:
            D[p] |= (1 << i)
            D[q] |= (1 << i)
            affected_centers.add(q)

    # 2. new squares (only pairs within D[p])
    dirs = bits(D[p])
    for i, j in combinations(dirs, 2):
        r = p ^ (1 << i) ^ (1 << j)
        if r in H:
            square = canonical_square(p, i, j)
            if square not in seen_squares:
                seen_squares.add(square)
                new_squares.append(square)
                for corner in square.corners():
                    Lz[corner].setdefault(i, set()).add(j)
                    Lz[corner].setdefault(j, set()).add(i)

    # 3. stars: recompute for p and its immediate neighbors
    for z in affected_centers:
        if popcount(D[z]) >= 3:
            upsert_star(z, D[z])

    # 4. tulips & cubes: triggered by new link-graph edges
    for square in new_squares:
        for corner in square.corners():
            i, j = square.free_coords()
            common = Lz[corner].get(i, set()) & Lz[corner].get(j, set())
            for k in common:
                K = frozenset({i, j, k})
                upsert_tulip(corner, K, extra_leaves=bits(D[corner]) - K)
                if (corner ^ bit(i) ^ bit(j) ^ bit(k)) in H:
                    upsert_cube(corner, K)

    # 5. propellers: triggered by new/affected axes
    for square in new_squares:
        for (z, a) in square.axes():
            B = bits(D[z] & D[z ^ bit(a)]) - {a}
            if len(B) >= 3:
                upsert_propeller(z, a, B)
```

Canonicalization: a square/cube can be discovered from multiple corners —
key by `(free_coordinate_set, fixed_bit_pattern_outside_those_coords)`.

## 5. Cut pool management (separate from structure index)

- Structures are stored **symbolically** (e.g. `Star(center=z, leaves=L)`),
  not as materialized Gurobi rows.
- At each fractional point `x_hat`, evaluate violation of each symbolic
  cut's inequality.
- Add to the Gurobi model only cuts with violation > tolerance (e.g. 1e-6),
  capped at K best by efficacy = `(rhs - a^T x_hat) / ||a||_2`.
- Recommended starting parameters: K in [10, 50] cuts/iteration, tolerance 1e-6.
- Dominance pruning for the *active* pool (not the structure index):
  edge dominates its 2 vertex no-goods; square dominates its 4 edges;
  cube dominates its faces; only maximal star/propeller/tulip-leaf-set kept.

## 6. Feasibility pump main loop (sketch)

```python
import gurobipy as gp
from gurobipy import GRB

def feasibility_pump(model: gp.Model, integer_vars: list[str], max_iter=500):
    relax = model.relax()
    structure_index = StructureIndex()
    cut_pool = []

    for k in range(max_iter):
        relax.optimize()
        x_frac = {v: relax.getVarByName(v).X for v in integer_vars}

        if all(is_close_to_int(x_frac[v]) for v in integer_vars):
            return extract_solution(relax)  # feasible

        x_round = round_point(x_frac)  # bitmask for binary case
        structure_index.insert(x_round)

        new_cuts = structure_index.generate_symbolic_cuts()
        active_cuts = filter_by_violation(new_cuts, x_frac, tol=1e-6)
        active_cuts = top_k_by_efficacy(active_cuts, k=30)

        add_or_refresh_cuts(relax, cut_pool, active_cuts)

        # projection LP: minimize distance to x_round subject to relax's constraints
        set_projection_objective(relax, x_round, integer_vars)

        if stagnating(x_round):  # cycle detection
            perturb(relax, x_round)

    return None  # no feasible solution found
```

## 7. Recommended file layout

```
structural_fp/
├── pyproject.toml
├── README.md
├── structural_fp/
│   ├── __init__.py
│   ├── bitops.py           # flip, flip2, popcount, bit iteration helpers
│   ├── structure_index.py  # StructureIndex, INSERT, D(z), Lz
│   ├── cuts.py             # symbolic cut classes + inequality generation
│   ├── cut_pool.py         # violation filtering, efficacy, dominance pruning
│   ├── fp_loop.py          # feasibility_pump() main loop, Gurobi integration
│   └── io.py                # MPS/LP instance loading (MIPLIB)
├── tests/
│   ├── test_bitops.py
│   ├── test_structure_index.py   # unit tests using the paper's small examples
│   ├── test_cuts.py               # check cut formulas against hand-derived cases
│   └── test_fp_loop.py            # small MILP end-to-end test
└── experiments/
    ├── run_benchmark.py    # run over MIPLIB subset, log iterations/time/gap
    └── results/
```

## 8. Milestones (recommended build order)

1. **bitops + StructureIndex, no cuts yet** — implement `INSERT`, verify with
   unit tests that D(z), Lz, squares/stars/cubes/tulips/propellers match
   hand-computed examples (use the worked example from the source
   conversation: `x0=(0.8,0.3,0.6)` rounding trajectory).
2. **Cut formula generation** — given a symbolic structure, generate the
   `(coefficients, rhs)` row; unit test against the table in Section 3.
3. **Cut pool + violation filtering** — no Gurobi yet, just test filtering
   logic on synthetic fractional points.
4. **Gurobi wiring** — plain feasibility pump (rounding + projection LP,
   no structural cuts) on a couple of small MIPLIB instances; confirm it
   finds feasible solutions before adding structural cuts.
5. **Integrate structural cuts into the loop** — turn on cut generation,
   compare iteration count / cycling behavior vs. plain FP baseline.
6. **Cycle-breaking perturbation + benchmarking** — run over a MIPLIB subset,
   log iterations-to-feasibility, solution quality, wall-clock time,
   compare against plain no-good-cut-only baseline.

## 9. Suggested first Claude Code prompt

> "Read feasibility_pump_structural_cuts_spec.md in this repo. Implement
> Milestone 1 only: bitops.py and structure_index.py with the INSERT
> algorithm from Section 4. Write tests in tests/test_structure_index.py
> that reproduce the worked example from Section 8 by hand. Don't touch
> Gurobi yet."

Then proceed milestone by milestone, asking Claude Code to run the tests
after each one before moving on.
