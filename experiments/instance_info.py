"""
Shared helper for classifying an MPS instance as pure IP (no continuous
variables) or MIP (mixed integer + continuous).
"""
from __future__ import annotations
import gurobipy as gp

_ENV: gp.Env | None = None


def _get_env() -> gp.Env:
    global _ENV
    if _ENV is None:
        _ENV = gp.Env(empty=True)
        _ENV.setParam("OutputFlag", 0)
        _ENV.start()
    return _ENV


def classify_type(model_or_path) -> str:
    """
    Return "IP" if every variable is binary/integer (no continuous
    variables), otherwise "MIP". Accepts either a loaded gp.Model or a
    path to an MPS/LP file (in which case it is read quietly, without
    affecting the caller's default environment/output).
    """
    if isinstance(model_or_path, gp.Model):
        m = model_or_path
    else:
        m = gp.read(str(model_or_path), env=_get_env())
    n_int = m.NumIntVars + m.NumBinVars
    return "IP" if n_int == m.NumVars else "MIP"
