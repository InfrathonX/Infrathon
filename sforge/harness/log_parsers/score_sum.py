"""Parser for score-based evaluation output (e.g., competitive programming judges).

Expected format:
  CASE 0000 OK score=12461
  CASE 0001 OK score=13335.5
  CASE 0002 TLE score=0
  CASE 0003 RE score=0
  CASE 0004 WA score=0
  CASE 0005 CE score=0
  ...
  TOTAL_SCORE 826577
  CASES_OK 48
  CASES_TOTAL 50

Status codes:
  OK  — ran successfully, score > 0
  TLE — time limit exceeded
  RE  — runtime error
  WA  — wrong answer (invalid output)
  CE  — compile error
"""

from __future__ import annotations

import re

from sforge.harness.constants import PASSED, FAILED


def parse_score_sum(test_output: str) -> list[dict]:
    """Parse score-based evaluation output.

    Returns list of {"name": str, "status": "PASSED"|"FAILED"}.
    """
    results: list[dict] = []

    # Match lines: CASE <id> <status> score=<n>
    case_pattern = re.compile(
        r"^CASE\s+(\S+)\s+(OK|TLE|RE|WA|CE)\s+score=(inf|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
        re.MULTILINE,
    )

    for match in case_pattern.finditer(test_output):
        case_id = match.group(1)
        status = match.group(2)
        score = float(match.group(3))

        if status == "OK":
            results.append({"name": f"case_{case_id}", "status": PASSED})
        else:
            results.append({"name": f"case_{case_id}_{status}", "status": FAILED})

    # If no cases found, check for compile error
    if not results:
        if "CE" in test_output or "compile" in test_output.lower():
            results.append({"name": "compile", "status": FAILED})
        else:
            results.append({"name": "unknown", "status": FAILED})

    return results
