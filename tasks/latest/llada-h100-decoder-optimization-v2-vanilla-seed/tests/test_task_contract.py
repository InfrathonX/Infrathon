from __future__ import annotations

import json


def test_task_contract(task_root):
    task = json.loads(
        (task_root / "tasks/llada_h100_decoder_optimization_v2.json").read_text()
    )
    assert task["task_id"] == "llada_h100_decoder_optimization_v2"
    assert task["name"] == (
        "LLaDA H100 Decoder Optimization v2 — Stratified H100, Vanilla Seed"
    )
    assert task["internet"] is True
    assert task["submit_paths"] == ["solutions", "FINAL_VERSION", "notes.md"]
    assert task["judge"]["score_direction"] == "maximize"
    assert task["judge"]["selection"] == "valid_then_score"
    assert task["judge"]["eval_timeout"] == 7200
    assert task["judge"]["private_artifact_path"] == "/logs/verifier/decoder_private.json"


def test_agent_contract_exposes_reproducible_runtime_without_workload_matrix(task_root):
    text = (task_root / "images/work/AGENT_TASK.md").read_text()
    for literal in (
        "LLaDA-8B-Instruct",
        "6059b30c9531767ce47e0ca6a8161a6bb0c67693",
        "NVIDIA H100",
        "GenerationSpec",
    ):
        assert literal in text
    for forbidden in (
        "GSM8K",
        "MATH",
        "HumanEval",
        "MBPP",
        "40 generations",
        "19,240",
        "9,076,480",
        "7,200",
    ):
        assert forbidden not in text
