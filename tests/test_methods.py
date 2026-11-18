"""
Tests for experiments/methods.py (M13 method registry).

Keep in tests/ (not experiments/) so `python -m pytest` from the project
root exercises it like everything else -- experiments/methods.py has no
Gurobi-batch/Excel side effects, just a dict and a translation function.
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from methods import METHODS, METHOD_ORDER, COMMON, fp_kwargs  # noqa: E402

from structural_fp.fp_loop import feasibility_pump, FPResult
from structural_fp.io import make_random_covering_milp


class TestRegistryContents:
    def test_plain_p2_was_removed(self):
        assert "Plain+P2" not in METHODS

    def test_nine_methods(self):
        assert len(METHODS) == 9
        assert METHOD_ORDER == list(METHODS.keys())

    @pytest.mark.parametrize("name", list(METHODS.keys()))
    def test_every_method_has_cut_mode_and_probe_k(self, name):
        spec = METHODS[name]
        assert spec["cut_mode"] in ("none", "nogood", "structural")
        assert spec["probe_k"] >= 0

    def test_common_runs_in_optimality_mode(self):
        assert COMMON["mode"] == "optimality"


class TestFpKwargsTranslation:
    def test_unknown_method_raises(self):
        with pytest.raises(KeyError):
            fp_kwargs("NotAMethod")

    @pytest.mark.parametrize("name,expect_structural,expect_nogood,expect_probe,expect_k", [
        ("Plain", False, False, False, None),
        ("NoGood", False, True, False, None),
        ("NoGood+P3", False, True, True, 3),
        ("Struct", True, False, False, None),
        ("Struct+P4", True, False, True, 4),
    ])
    def test_flags_match_spec(self, name, expect_structural, expect_nogood, expect_probe, expect_k):
        kwargs = fp_kwargs(name)
        assert kwargs["use_structural_cuts"] == expect_structural
        assert kwargs["use_no_good_cuts"] == expect_nogood
        assert kwargs.get("use_face_probe", False) == expect_probe
        if expect_probe:
            assert kwargs["face_k"] == expect_k

    def test_overrides_win(self):
        kwargs = fp_kwargs("Struct+P3", max_iter=10, random_seed=7)
        assert kwargs["max_iter"] == 10
        assert kwargs["random_seed"] == 7
        assert kwargs["mode"] == "optimality"  # still inherited from COMMON


class TestEveryMethodActuallyRuns:
    @pytest.mark.parametrize("name", list(METHODS.keys()))
    def test_runs_without_crashing(self, name):
        model = make_random_covering_milp(n=14, m=9, seed=4)
        kwargs = fp_kwargs(name, max_iter=100, max_perturbations=5, random_seed=0)
        res = feasibility_pump(model, **kwargs)
        assert isinstance(res, FPResult)
        assert res.mode == "optimality"
