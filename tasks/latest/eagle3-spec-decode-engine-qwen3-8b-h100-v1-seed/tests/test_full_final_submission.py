"""Trusted submission routing: complete suite without relaxing replay authority."""
import ast
from pathlib import Path
import pytest
from conftest import ROOT


def route(context):
    # Execute the real selector without importing CUDA/model libraries.
    tree = ast.parse((ROOT / "images/judge/verify_eagle3.py").read_text())
    selector = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_evaluation_request")
    namespace = {"Path": Path, "EVALUATION_CONTEXT": Path("/trusted/context"), "FINAL_SUITE_ID": "final-holdout-v1", "strict_json": lambda *a, **kw: context, "_strict_json": lambda *a, **kw: context}
    exec(compile(ast.Module(body=[selector], type_ignores=[]), "trusted-selector", "exec"), namespace)
    return namespace["load_evaluation_request"]()


@pytest.mark.parametrize("round_id", ["agent-1", "agent-100", "auto-1", "manual-1", "candidate-17"])
def test_every_scored_phase_selects_full_final(round_id):
    assert route({"round_id": round_id})[0] == "final"


@pytest.mark.parametrize("round_id", ["agent-1", "auto-1"])
def test_full_suite_does_not_authorize_interactive_version_override(round_id):
    with pytest.raises(RuntimeError):
        route({"round_id": round_id, "candidate_version": 17})


def test_trusted_host_replay_can_still_select_saved_version():
    assert route({"round_id": "candidate-17", "candidate_version": 17})[2] == 17


@pytest.mark.parametrize("version", [True, 0, 1000, "17"])
def test_malformed_override_remains_rejected(version):
    with pytest.raises(RuntimeError):
        route({"round_id": "manual-1", "candidate_version": version})


def test_unknown_round_remains_rejected():
    with pytest.raises(RuntimeError):
        route({"round_id": "worker-supplied"})
