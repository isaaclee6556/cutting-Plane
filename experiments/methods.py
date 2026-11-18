"""
Method registry (M12/M13): a single source of truth for which (cut_mode,
probe_k) combination each named method in the batch experiments corresponds
to, plus the mode/perturbation settings shared by all of them.

feasibility_pump() itself still takes the legacy flags (use_structural_cuts,
use_no_good_cuts, use_face_probe, face_k) as its public signature -- cut_mode
and probe_k are normalized internally (see structural_fp/fp_loop.py). This
module is the one place that translates a method name into those flags, so
experiments/run_matrix.py and anything else that runs the matrix never has
to duplicate the mapping.
"""
from __future__ import annotations

# "Plain+P2" (cut_mode="none", probe_k=2) was intentionally dropped from the
# matrix (2026-11-18): probing with no cut mechanism at all only moves the
# incumbent found by brute corner evaluation, with nothing excluding a
# rejected point from being re-proposed -- not a useful comparison point for
# this batch.
METHODS: dict[str, dict] = {
    "Plain":     dict(cut_mode="none",       probe_k=0),
    "NoGood":    dict(cut_mode="nogood",     probe_k=0),
    "NoGood+P2": dict(cut_mode="nogood",     probe_k=2),
    "NoGood+P3": dict(cut_mode="nogood",     probe_k=3),
    "NoGood+P4": dict(cut_mode="nogood",     probe_k=4),
    "Struct":    dict(cut_mode="structural", probe_k=0),
    "Struct+P2": dict(cut_mode="structural", probe_k=2),
    "Struct+P3": dict(cut_mode="structural", probe_k=3),
    "Struct+P4": dict(cut_mode="structural", probe_k=4),
}

METHOD_ORDER = list(METHODS.keys())

# Shared across every method in the matrix, so the only difference between
# rows is the cut_mode/probe_k combination itself.
COMMON: dict = dict(
    mode="optimality",
    max_iter=5000,
    max_perturbations=100,
    perturb_scale=1.5,
)


def fp_kwargs(method_name: str, **overrides) -> dict:
    """
    Build the kwargs feasibility_pump() actually accepts for a named method
    (COMMON plus that method's cut_mode/probe_k translated to legacy flags).
    `overrides` wins over both COMMON and the derived flags (e.g. a specific
    time_limit, random_seed, or a smaller max_iter for a quick test).
    """
    if method_name not in METHODS:
        raise KeyError(f"unknown method {method_name!r}; known methods: {METHOD_ORDER}")
    spec = METHODS[method_name]
    cut_mode = spec["cut_mode"]
    probe_k = spec["probe_k"]

    kwargs = dict(COMMON)
    kwargs["use_structural_cuts"] = (cut_mode == "structural")
    kwargs["use_no_good_cuts"] = (cut_mode == "nogood")
    if probe_k > 0:
        kwargs["use_face_probe"] = True
        kwargs["face_k"] = probe_k
    kwargs.update(overrides)
    return kwargs
