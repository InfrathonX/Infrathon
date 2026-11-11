"""Cross-file task contract and immutable ABI tests."""

from __future__ import annotations

import ast
import hashlib
import json

from conftest import ROOT, private_data

REPOSITORY = ROOT.parents[2]


# Must match ABI_SHA256 in scripts/validate_layout.py.
ABI_SHA256 = "477fd276807239a04180003cd318d8c9289321fe96ae4043a6206274869b38ed"


def test_api_matches_the_recorded_abi() -> None:
    shipping = ROOT / "images/work/workdir/spec_lab/runtime/api.py"
    assert hashlib.sha256(shipping.read_bytes()).hexdigest() == ABI_SHA256


def test_source_admission_is_shared() -> None:
    judge = ROOT / "images/judge/verifier/source_admission.py"
    public = ROOT / "images/work/public_tests/source_admission.py"
    assert judge.read_bytes() == public.read_bytes()


def test_policy_lifecycle_with_frozen_tiers_and_identities() -> None:
    policy = json.loads((private_data() / "policy.json").read_text())
    assert policy["status"] in {"QUALIFIED", "PENDING_QUALIFICATION"}
    assert len(policy["tiers"]) == 8
    for name in (
        "work_image_id",
        "judge_image_id",
        "source_digest",
        "runtime_provenance_sha256",
    ):
        assert len(policy[name]) >= 64


def test_metadata_contract() -> None:
    task = json.loads(
        (ROOT / "tasks/eagle3_spec_decode_engine_qwen3_8b_h100_v1.json").read_text()
    )
    assert task["base_image"] == "pytorch280_cu129"
    assert task["cwd"] == "/app/workdir/spec_lab"
    assert task["disable_auto_eval"] is True
    assert task["scoring"]["score_key"] == "score_tokens_per_s"
    assert task["judge"]["eval_timeout"] == 5400
    assert task["submit_paths"] == ["solutions", "FINAL_VERSION", "notes.md"]


def test_all_python_parses() -> None:
    for path in ROOT.rglob("*.py"):
        if "__pycache__" not in path.parts:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
