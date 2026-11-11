from __future__ import annotations

import json
import os
import shutil
import subprocess

from conftest import LAB_ROOT, TASK_ROOT


RUNNER = TASK_ROOT / "images/work/public_tests/run_public_tests.sh"


def test_public_runner_documents_the_instruction_modes():
    process = subprocess.run(
        ["bash", str(RUNNER), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    for flag in ("--cpu-only", "--quick", "--candidate-version"):
        assert flag in process.stdout


def test_public_runner_rejects_unknown_arguments():
    process = subprocess.run(
        ["bash", str(RUNNER), "--not-a-real-mode"],
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 2
    assert "unknown argument" in process.stderr


def test_public_cpu_rejects_imperative_optimize_model_export(tmp_path):
    lab = tmp_path / "orpo_lab"
    shutil.copytree(LAB_ROOT, lab)
    candidate = lab / "solutions/objective_v001/candidate.py"
    candidate.write_text(
        """\
import torch

def build_objective(contract):
    return torch.nn.Identity()

def optimize_model(model, contract):
    return model
""",
        encoding="utf-8",
    )
    environment = dict(os.environ, ORPO_LAB_ROOT=str(lab))
    process = subprocess.run(
        ["bash", str(RUNNER), "--cpu-only"],
        text=True,
        capture_output=True,
        check=False,
        env=environment,
    )
    assert process.returncode == 1
    report = json.loads(process.stdout)
    assert report["status"] == "FAIL"
    assert report["formal"] is False
    assert "optimize_model" in report["error"]


def test_public_train_probe_uses_scored_completion_throughput():
    source = (TASK_ROOT / "images/work/public_tests/public_train_probe.py").read_text()
    assert 'reference["throughput_scored_tokens_per_second"]' in source
    assert 'candidate["throughput_scored_tokens_per_second"]' in source
    assert '"--candidate-version"' in source


def test_public_objective_probe_names_the_official_reference_precisely():
    source = (TASK_ROOT / "images/work/public_tests/public_eval.py").read_text()
    assert '"official_hf_trl_reference": official_reference_metrics' in source
    assert '"baseline": baseline_metrics' not in source
