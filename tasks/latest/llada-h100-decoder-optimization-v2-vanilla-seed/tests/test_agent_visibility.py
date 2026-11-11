from __future__ import annotations

import json
import re
import sys
from pathlib import Path


OLD_IDENTITY = re.compile(r"fast[-_ ]?dllm|\bdllm\b", re.IGNORECASE)
ACADEMIC_PROVENANCE = re.compile(
    r"paper[- ](?:derived|era|aligned)|paper (?:repository|benchmark|experiment)|"
    r"section\s*4|tables?\s*[145]\b|table[145]_",
    re.IGNORECASE,
)
PRIVATE_DESIGN = re.compile(
    r"\b(?:hidden|private)\s+(?:case|cases|data|dataset|evaluation|evaluator|"
    r"suite|row|rows|workload|reference|threshold|feedback|test|tests)\b|"
    r"\b(?:reference|integer)\s+tolerance\b|40 generations|12 rows|"
    r"\bcanonical\s+(?:pool|workload)\b|\bcohort\s+(?:secret|membership|order)\b|"
    r"\b400\s+(?:cases|records|targets)\b|\b40[- ](?:case|record|target)",
    re.IGNORECASE,
)


def _assert_neutral(label: str, text: str) -> None:
    for pattern in (OLD_IDENTITY, ACADEMIC_PROVENANCE, PRIVATE_DESIGN):
        match = pattern.search(text)
        assert match is None, f"{label} exposes {match.group(0)!r}"


def test_agent_identity_query_and_generated_prompt_are_neutral(task_root: Path) -> None:
    task = json.loads(
        (task_root / "tasks/llada_h100_decoder_optimization_v2.json").read_text(
            encoding="utf-8"
        )
    )
    visible_identity = "\n".join(
        (
            task["task_id"],
            task["name"],
            task["cwd"],
            task["work"]["agent_query"],
        )
    )
    _assert_neutral("task identity/query", visible_identity)

    repository_root = task_root.parents[2]
    sys.path.insert(0, str(repository_root))
    try:
        from sforge.harness.evolve_scripts import generate_evolve_prompt

        prompt = generate_evolve_prompt(
            task["work"]["agent_query"],
            submit_paths=task["submit_paths"],
            internet=task["internet"],
            max_submissions=None,
            submission_cooldown=300,
        )
    finally:
        sys.path.pop(0)

    _assert_neutral("generated agent prompt", prompt)
    lowered = prompt.lower()
    for term in ("private", "hidden", "timeout boundary"):
        assert term not in lowered
    assert "sforge-submit" in prompt
    assert "minimum interval of 300s" in lowered


def test_work_build_context_contains_no_origin_or_private_design_leak(task_root: Path) -> None:
    work_root = task_root / "images/work"
    for path in sorted(work_root.rglob("*")):
        relative = path.relative_to(work_root)
        _assert_neutral("Work path", relative.as_posix())
        if not path.is_file() or path.name == "public_cases.json":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        _assert_neutral(f"Work file {relative}", text)

    lab = work_root / "workdir/decoder_lab"
    assert not (lab / "runtime/quality.py").exists()
    assert not any(path.name == "hidden_cases.json" for path in work_root.rglob("*"))
    assert not any(path.name == "reference_quality.json" for path in work_root.rglob("*"))


def test_public_and_final_source_policy_are_identical(task_root: Path) -> None:
    public_policy = (
        task_root
        / "images/work/workdir/decoder_lab/runtime/source_policy.py"
    ).read_bytes()
    final_policy = (
        task_root / "images/judge/verifier/source_admission.py"
    ).read_bytes()
    assert public_policy == final_policy
