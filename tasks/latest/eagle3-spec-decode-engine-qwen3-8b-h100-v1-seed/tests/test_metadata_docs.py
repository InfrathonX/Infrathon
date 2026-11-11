"""Metadata and documentation completeness tests."""

from __future__ import annotations

import json

from conftest import ROOT, maintainer_doc


def test_task_metadata_declares_manual_versioned_scoring() -> None:
    spec = json.loads(
        (ROOT / "tasks/eagle3_spec_decode_engine_qwen3_8b_h100_v1.json").read_text()
    )
    assert spec["disable_auto_eval"] is True
    assert spec["version_enumeration"]["directory_pattern"] == "engine_v([0-9]{3})"
    assert spec["scoring"]["score_key"] == "score_tokens_per_s"


def test_a2_guarantee_and_limit_are_verbatim() -> None:
    scoring = maintainer_doc("docs/SCORING_CONTRACT.md")
    assert "The oracle guarantees that, across the evaluated run" in scoring
    assert "It does not prove bit-exact equality with one vanilla stream" in scoring
    assert "INT8-MLP round trip and FP16 LM head pass" in scoring


def test_qualified_docs_do_not_claim_an_unmeasured_agent_result() -> None:
    for name in ("CALIBRATION.md", "RUN_RESULTS.md"):
        text = maintainer_doc(f"docs/{name}")
        assert "QUALIFIED" in text
        assert "No agent score is claimed" in text
