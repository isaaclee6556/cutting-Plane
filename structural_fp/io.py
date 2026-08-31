"""
Instance loading and small MILP factory functions for testing.
"""
from __future__ import annotations
import gurobipy as gp
from gurobipy import GRB


def load_mps(path: str) -> gp.Model:
    """Load a MILP from an MPS file (Gurobi auto-detects format)."""
    m = gp.read(path)
    m.setParam("OutputFlag", 0)
    return m


def make_simple_binary_milp() -> gp.Model:
    """
    4-variable feasible binary MILP:
        min  x0 + x1 + x2 + x3
        s.t. x0 + x1 >= 1
             x2 + x3 >= 1
             x_i in {0, 1}

    LP relaxation optima are vertices, e.g. (1,0,1,0).
    """
    m = gp.Model("simple_binary")
    m.setParam("OutputFlag", 0)
    x = [m.addVar(vtype=GRB.BINARY, name=f"x{i}") for i in range(4)]
    m.addConstr(x[0] + x[1] >= 1, "c0")
    m.addConstr(x[2] + x[3] >= 1, "c1")
    m.setObjective(gp.quicksum(x), GRB.MINIMIZE)
    m.update()
    return m


def make_covering_milp(n: int = 6) -> gp.Model:
    """
    Set covering: n binary vars, n/2 pair constraints.
        min sum x_i
        s.t. x_{2k} + x_{2k+1} >= 1  for k=0..n/2-1
             x_i in {0, 1}

    n must be even. Integral optimal: one variable per pair = 1.
    """
    if n % 2 != 0:
        raise ValueError("n must be even")
    m = gp.Model(f"covering_{n}")
    m.setParam("OutputFlag", 0)
    x = [m.addVar(vtype=GRB.BINARY, name=f"x{i}") for i in range(n)]
    for k in range(n // 2):
        m.addConstr(x[2 * k] + x[2 * k + 1] >= 1, f"c{k}")
    m.setObjective(gp.quicksum(x), GRB.MINIMIZE)
    m.update()
    return m


def make_infeasible_lp_milp() -> gp.Model:
    """Binary MILP whose LP relaxation is infeasible (x >= 2, x in {0,1})."""
    m = gp.Model("infeasible_lp")
    m.setParam("OutputFlag", 0)
    x = m.addVar(vtype=GRB.BINARY, name="x0")
    m.addConstr(x >= 2, "lb_force")
    m.setObjective(x, GRB.MINIMIZE)
    m.update()
    return m


def make_random_covering_milp(n: int, m: int, seed: int = 0) -> gp.Model:
    """
    Random set covering: n binary variables, m covering constraints.
    Each constraint picks a random subset of ~n/3 variables and requires
    their sum >= 1.  Always feasible (setting all vars=1 is a solution).

    Parameters
    ----------
    n    : number of binary variables
    m    : number of covering constraints
    seed : random seed for reproducibility
    """
    import random as _rnd
    rng = _rnd.Random(seed)
    model = gp.Model(f"rnd_covering_n{n}_m{m}_s{seed}")
    model.setParam("OutputFlag", 0)
    x = [model.addVar(vtype=GRB.BINARY, name=f"x{i}") for i in range(n)]
    for j in range(m):
        k = max(2, n // 3)
        subset = rng.sample(range(n), k)
        model.addConstr(gp.quicksum(x[i] for i in subset) >= 1, f"c{j}")
    model.setObjective(gp.quicksum(x), GRB.MINIMIZE)
    model.update()
    return model


def make_fractional_cycle_milp() -> gp.Model:
    """
    Binary MILP with no integral feasible solution.
    LP is feasible (x0+x1=0.5 has solutions in [0,1]^2) but no {0,1}^2
    point satisfies the equality, so feasibility_pump will detect a cycle.

        min x0
        s.t. x0 + x1 = 0.5
             x0, x1 in {0, 1}
    """
    m = gp.Model("fractional_cycle")
    m.setParam("OutputFlag", 0)
    x0 = m.addVar(vtype=GRB.BINARY, name="x0")
    x1 = m.addVar(vtype=GRB.BINARY, name="x1")
    m.addConstr(x0 + x1 == 0.5, "eq_half")
    m.setObjective(x0, GRB.MINIMIZE)
    m.update()
    return m
