from __future__ import annotations

from collections import Counter
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import types

from conftest import private_data_dir


def load_public_eval(task_root: Path):
    return _load_torch_module(
        task_root / "images/work/public_tests/public_eval.py",
        "llada_h100_v2_public_eval",
    )


def load_runtime_api(task_root: Path):
    return _load_torch_module(
        task_root / "images/work/workdir/decoder_lab/runtime/api.py",
        "llada_h100_v2_runtime_api",
    )


def _load_torch_module(path: Path, name: str):
    inserted_torch_stub = False
    if "torch" not in sys.modules:
        try:
            __import__("torch")
        except ModuleNotFoundError:
            sys.modules["torch"] = types.ModuleType("torch")
            inserted_torch_stub = True
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered while it executes: the runtime API's dataclasses look their
    # module up by name.
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted_torch_stub:
            del sys.modules["torch"]
    return module


def load_public_quality(task_root: Path):
    path = task_root / "images/work/public_tests/public_quality.py"
    spec = importlib.util.spec_from_file_location("llada_h100_v2_public_quality", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_profile_selection_is_fail_closed_and_balanced(task_root):
    module = load_public_eval(task_root)
    cases = json.loads(
        (task_root / "images/work/public_tests/public_cases.json").read_text()
    )

    quick = module.select_public_cases(cases, True)
    full = module.select_public_cases(cases, False)

    assert len(quick) == 4
    assert len(full) == 40
    assert {case["benchmark"] for case in quick} == set(module.BENCHMARKS)
    assert all(
        sum(case["benchmark"] == benchmark for case in full) == 10
        for benchmark in module.BENCHMARKS
    )

    damaged = [case for case in cases if case["case_id"] != full[-1]["case_id"]]
    try:
        module.select_public_cases(damaged, False)
    except RuntimeError as exc:
        assert "expected 40" in str(exc)
    else:
        raise AssertionError("a truncated public profile must fail closed")


def test_public_suite_runs_its_own_valid_generation_lengths(task_root):
    module = load_public_eval(task_root)
    api = load_runtime_api(task_root)
    cases = json.loads(
        (task_root / "images/work/public_tests/public_cases.json").read_text()
    )

    assert module.PUBLIC_GEN_LENGTH == {128: 128, 256: 192, 512: 384, 1024: 768}
    ran = Counter(module.public_gen_length(case) for case in cases)
    assert ran == {128: 4, 192: 19, 384: 19, 768: 2}
    for case in cases:
        # Block length divides it and prompt plus generation fits the model.
        spec = api.GenerationSpec(module.public_gen_length(case), case["block_length"])
        spec.validate(case["prompt_token_count"])

    try:
        module.public_gen_length({"case_id": "unlisted", "gen_length": 320})
    except RuntimeError as exc:
        assert "no public length" in str(exc)
    else:
        raise AssertionError("an unlisted generation length must fail closed")


def test_no_public_run_uses_a_scored_configuration(task_root):
    module = load_public_eval(task_root)
    public = json.loads(
        (task_root / "images/work/public_tests/public_cases.json").read_text()
    )
    hidden = json.loads(
        gzip.decompress((private_data_dir() / "hidden_cases.json.gz").read_bytes())
    )
    scored = {
        (case["benchmark"], case["fewshot"], case["gen_length"], case["block_length"])
        for case in hidden
    }
    assert len(scored) == 12
    ran = {
        (case["benchmark"], case["fewshot"], module.public_gen_length(case), case["block_length"])
        for case in public
    }
    assert ran.isdisjoint(scored)


def test_public_dataset_summary_reports_each_category(task_root):
    module = load_public_eval(task_root)
    rows = []
    for index, benchmark in enumerate(module.BENCHMARKS, start=1):
        rows.append(
            {
                "benchmark": benchmark,
                "quality": {"evaluated": True, "correct": index % 2 == 0, "parsed": True},
                "seconds": float(index),
                "useful_tokens": index * 10,
                "nfe": index * 2,
            }
        )

    summary = module.summarize_datasets(rows)

    assert list(summary) == list(module.BENCHMARKS)
    assert all(item["cases"] == 1 and item["evaluated"] == 1 for item in summary.values())
    assert summary["gsm8k"]["accuracy"] == 0.0
    assert summary["math"]["accuracy"] == 1.0
    assert summary["mbpp"]["parsed"] == 1
    assert summary["humaneval"]["throughput_tok_s"] == 10.0


def test_public_mbpp_completion_is_executed_for_correctness(task_root, monkeypatch):
    module = load_public_quality(task_root)
    monkeypatch.setattr(module.os, "geteuid", lambda: 1000)
    case = {
        "benchmark": "mbpp",
        "tests": ["assert add_one(1) == 2", "assert add_one(-1) == 0"],
        "test_setup_code": "",
    }

    passing = module.score_completion(
        case, "def add_one(value):\n    return value + 1"
    )
    failing = module.score_completion(
        case, "def add_one(value):\n    return value - 1"
    )

    assert passing == {"correct": True, "parsed": True, "code_reason": "passed"}
    assert failing["correct"] is False
    assert failing["parsed"] is True
