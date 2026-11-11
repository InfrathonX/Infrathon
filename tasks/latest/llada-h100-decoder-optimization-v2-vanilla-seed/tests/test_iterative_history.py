from __future__ import annotations

import hashlib

import pytest

from conftest import load_module


def test_interactive_history_accepts_contiguous_manual_rounds(task_root):
    module = load_module(
        "iterative_formal_history",
        task_root / "scripts/validate_formal_artifacts.py",
    )
    history = {
        "run_id": "run-1",
        "agent_submissions": 2,
        "auto_submissions": 0,
        "public_submissions": 0,
        "best_score": 12.5,
        "best_round": "agent-2",
        "best_pass_rate": 1.0,
        "entries": [
            {"type": "submission", "status": "completed", "round": "agent-1"},
            {"type": "submission", "status": "completed", "round": "agent-2"},
        ],
    }
    result = {
        "total_rounds": 2,
        "agent_submissions": 2,
        "auto_submissions": 0,
        "best_score": 12.5,
        "best_round": "agent-2",
        "best_pass_rate": 1.0,
    }
    assert module.validate_interactive_history(result, history, "run-1") == {
        "agent": 2,
        "auto": 0,
        "public": 0,
    }


def test_interactive_history_rejects_missing_round(task_root):
    module = load_module(
        "iterative_formal_history_bad",
        task_root / "scripts/validate_formal_artifacts.py",
    )
    history = {
        "run_id": "run-1",
        "agent_submissions": 1,
        "auto_submissions": 0,
        "public_submissions": 0,
        "best_score": None,
        "best_round": "",
        "best_pass_rate": 0.0,
        "entries": [
            {"type": "submission", "status": "completed", "round": "agent-2"}
        ],
    }
    result = {
        "total_rounds": 1,
        "agent_submissions": 1,
        "auto_submissions": 0,
        "best_score": None,
        "best_round": "",
        "best_pass_rate": 0.0,
    }
    with pytest.raises(ValueError, match="contiguous"):
        module.validate_interactive_history(result, history, "run-1")


def _full_private(module):
    records = []
    rows = {}
    index = 0
    for row_id, count in module.FINAL_ROW_COUNTS.items():
        rows[row_id] = {
            "cases": count,
            "correct": 0,
            "parsed": count,
            "useful_tokens": count,
            "seconds": float(count),
            "throughput_tok_s": 1.0,
        }
        for _ in range(count):
            records.append(
                {
                    "row_id": row_id,
                    "case_sha256": hashlib.sha256(
                        f"case-{index}".encode("utf-8")
                    ).hexdigest(),
                    "correct": False,
                    "parsed": True,
                    "useful_tokens": 1,
                    "seconds": 1.0,
                }
            )
            index += 1
    case_set_sha256 = hashlib.sha256(
        "\n".join(sorted(record["case_sha256"] for record in records)).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "evaluation_mode": "final",
        "workload_profile": module.WORKLOAD_PROFILE,
        "generation_count": module.GENERATION_COUNT,
        "requested_slots": module.REQUESTED_SLOTS,
        "cohort_case_count": module.GENERATION_COUNT,
        "cohort_row_counts": dict(module.FINAL_ROW_COUNTS),
        "cohort_category_definition": module.CATEGORY_DEFINITION,
        "cohort_category_counts": module.CATEGORY_COUNTS,
        "cohort_bonus_categories": [],
        "submission_number": None,
        "cohort_cycle": None,
        "cohort_slot": None,
        "cohort_case_set_sha256": case_set_sha256,
        "rows": rows,
        "records": records,
    }


def test_final_workload_provenance_recomputes_records(task_root):
    module = load_module(
        "iterative_formal_full_workload",
        task_root / "scripts/validate_formal_artifacts.py",
    )
    private = _full_private(module)
    module.validate_final_workload(private)

    private["records"][1]["case_sha256"] = private["records"][0]["case_sha256"]
    with pytest.raises(ValueError, match="repeated"):
        module.validate_final_workload(private)


def test_final_workload_provenance_rejects_metadata_only_claim(task_root):
    module = load_module(
        "iterative_formal_incomplete_workload",
        task_root / "scripts/validate_formal_artifacts.py",
    )
    private = _full_private(module)
    private["rows"][next(iter(module.FINAL_ROW_COUNTS))]["cases"] -= 1
    with pytest.raises(ValueError, match="row count mismatch"):
        module.validate_final_workload(private)


def test_private_cohort_secret_is_bound_to_manifest_commitment(task_root, tmp_path):
    module = load_module(
        "iterative_formal_cohort_secret",
        task_root / "scripts/validate_formal_artifacts.py",
    )
    secret = "ab" * 32
    secret_path = tmp_path / ".cohort-secret"
    secret_path.write_text(secret + "\n", encoding="ascii")
    secret_path.chmod(0o600)
    commitment = hashlib.sha256(secret.encode("ascii")).hexdigest()
    module.validate_cohort_secret(tmp_path, commitment)

    with pytest.raises(ValueError, match="commitment mismatch"):
        module.validate_cohort_secret(tmp_path, "0" * 64)
    secret_path.chmod(0o644)
    with pytest.raises(ValueError, match="permissions"):
        module.validate_cohort_secret(tmp_path, commitment)
