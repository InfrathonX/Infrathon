from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def _marked(value: dict) -> str:
    return (
        ">>>>> Start Structured Result\n"
        + json.dumps(value)
        + "\n>>>>> End Structured Result\n"
    )


def test_task_runtime_requires_one_authoritative_marker_pair(runtime_module) -> None:
    parser = runtime_module.strict_parse_structured_json
    payload = {"valid": False, "score": 0.0}
    assert parser(_marked(payload)) == payload

    with pytest.raises(ValueError):
        parser(json.dumps({"valid": True, "score": 999.0}))
    with pytest.raises(ValueError):
        parser(_marked(payload) + _marked(payload))
    with pytest.raises(ValueError):
        parser(
            ">>>>> Start Structured Result\n"
            '{"valid":true,"valid":false,"score":0}'
            "\n>>>>> End Structured Result\n"
        )
    with pytest.raises(ValueError):
        parser(
            ">>>>> Start Structured Result\n"
            '{"valid":true,"score":NaN}'
            "\n>>>>> End Structured Result\n"
        )


def test_missing_valid_and_timeout_are_never_implicitly_valid(runtime_module) -> None:
    from sforge.harness.grading import EvalReport, _grade_structured

    assert EvalReport(task_id="t", submission_id="s").valid is False
    task = SimpleNamespace(task_id="t")

    missing = _grade_structured(
        task,
        _marked({"score": 10.0, "details": []}),
        "s",
        False,
        1.0,
        "raw",
    )
    assert missing.valid is False

    timed_out = _grade_structured(
        task,
        _marked({"valid": True, "score": 10.0, "details": []}),
        "s",
        True,
        1.0,
        "raw",
    )
    assert timed_out.valid is False
