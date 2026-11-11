"""The launcher's experiment manifest carries the private cohort the formal precheck expects."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

TASK_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = TASK_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import validate_formal_artifacts  # noqa: E402


def launcher_private_cohort() -> dict:
    """private_cohort as scripts/run_fixed_budget.sh writes it (same jq expression)."""

    contract = subprocess.run(
        [sys.executable, str(SCRIPTS / "private_cohort_contract.py")],
        check=True, capture_output=True, text=True,
    ).stdout
    program = (
        "$private_cohort_contract + {interactive_case_timeout_seconds: 60, "
        "interactive_judge_timeout_seconds: 600, secret_sha256: $secret}"
    )
    result = subprocess.run(
        ["jq", "-n", "--argjson", "private_cohort_contract", contract,
         "--arg", "secret", "0" * 64, program],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def test_launcher_private_cohort_passes_the_formal_precheck() -> None:
    validate_formal_artifacts.check_private_cohort(launcher_private_cohort())


def test_launcher_builds_the_private_cohort_from_the_contract() -> None:
    runner = (SCRIPTS / "run_fixed_budget.sh").read_text(encoding="utf-8")
    assert "private_cohort_contract.py" in runner
    assert "private_cohort: ($private_cohort_contract + {" in runner
    assert "hmac_balanced_random_without_replacement_v1" not in runner


@pytest.mark.parametrize("field,value", [
    ("selection", "hmac_balanced_random_without_replacement_v1"),
    ("total_cases", 50),
    ("category_base_quota", 12),
    ("bonus_categories", 2),
])
def test_a_stale_cohort_is_refused(field, value) -> None:
    cohort = launcher_private_cohort()
    cohort[field] = value
    with pytest.raises(ValueError, match="private-cohort contract mismatch"):
        validate_formal_artifacts.check_private_cohort(cohort)
