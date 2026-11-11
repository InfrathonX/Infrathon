"""Select the complete frozen stratified workload for every scored round.

Interactive and terminal requests both execute all 520 committed generations,
with 130 distinct targets per dataset across the same twelve score rows.
Trusted agent-N identity and submission number are retained for audit; neither
round number nor a private secret changes membership or ordering.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


TASK_ID = "llada_h100_decoder_optimization_v2"
WORKLOAD_PROFILE = "fast_dllm_v1_llada_12row_stratified520_batch1"
CATEGORY_DEFINITION = {
    "kind": "dataset",
    "field": "benchmark",
    "categories": ["gsm8k", "math", "humaneval", "mbpp"],
}
CATEGORY_COUNTS = {category: 130 for category in CATEGORY_DEFINITION["categories"]}
INTERACTIVE_SELECTION = "all_frozen_final_v1"
INTERACTIVE_TOTAL_CASES = 520
INTERACTIVE_CATEGORY_BASE_QUOTA = 130
INTERACTIVE_BONUS_CATEGORIES = 0
INTERACTIVE_MINIMUM_PER_ROW = 1
FINAL_ROW_COUNTS = {
    "table1_gsm8k_5shot_g256": 39,
    "table1_gsm8k_5shot_g512": 39,
    "table1_math_4shot_g256": 65,
    "table1_math_4shot_g512": 65,
    "table1_humaneval_0shot_g256": 65,
    "table1_humaneval_0shot_g512": 65,
    "table1_mbpp_3shot_g256": 65,
    "table1_mbpp_3shot_g512": 65,
    "table4_gsm8k_5shot_g1024": 13,
    "table5_gsm8k_8shot_g256": 13,
    "table5_gsm8k_8shot_g512": 13,
    "table5_gsm8k_8shot_g1024": 13,
}
ROUND_ID_KEY = "round_id"
RUN_ID_KEY = "run_id"
FINAL_MODE = "final"
INTERACTIVE_MODE = "interactive"
_AGENT_ROUND = re.compile(r"agent-([1-9][0-9]*)\Z")
_POLICY_PATH = Path(__file__).resolve().parents[1] / "private_data/workload_policy.json"


class CohortError(ValueError):
    """The trusted context, workload policy, or canonical cases are invalid."""


def _load_policy(path: Path = _POLICY_PATH) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CohortError("trusted workload policy is missing or invalid") from exc
    if not isinstance(value, dict):
        raise CohortError("trusted workload policy must be a JSON object")
    if (
        value.get("schema_version") != 1
        or value.get("task_id") != TASK_ID
        or value.get("profile") != WORKLOAD_PROFILE
    ):
        raise CohortError("trusted workload policy identity mismatch")
    return value


@dataclass(frozen=True, slots=True)
class RowPlan:
    benchmark: str
    canonical_count: int


def _row_plan(policy: Mapping[str, Any]) -> dict[str, RowPlan]:
    manifest = policy.get("manifest")
    interactive = policy.get("interactive")
    if not isinstance(manifest, Mapping) or not isinstance(interactive, Mapping):
        raise CohortError("workload policy sections are invalid")
    row_counts = manifest.get("row_counts")
    row_order = manifest.get("row_order")
    row_benchmarks = manifest.get("row_benchmarks")
    if not all(isinstance(item, Mapping) for item in (row_counts, row_benchmarks)):
        raise CohortError("workload policy row mappings are invalid")
    if (
        interactive.get("category_definition") != CATEGORY_DEFINITION
        or interactive.get("selection") != INTERACTIVE_SELECTION
        or interactive.get("total_cases") != INTERACTIVE_TOTAL_CASES
        or interactive.get("row_atomic") is not True
        or interactive.get("timeout_policy") != "same_as_final"
        or interactive.get("category_base_quota")
        != INTERACTIVE_CATEGORY_BASE_QUOTA
        or interactive.get("bonus_categories") != INTERACTIVE_BONUS_CATEGORIES
        or interactive.get("minimum_per_row") != INTERACTIVE_MINIMUM_PER_ROW
        or "per_row_quotas" in interactive
        or "per_category_quotas" in interactive
    ):
        raise CohortError("interactive complete 520-case policy is invalid")
    if (
        not isinstance(row_order, list)
        or len(row_order) != len(set(row_order))
        or set(row_order) != set(row_counts)
        or set(row_counts) != set(row_benchmarks)
    ):
        raise CohortError("workload policy row mappings disagree")
    if row_order != list(FINAL_ROW_COUNTS):
        raise CohortError("workload policy row order is not canonical")
    if dict(row_counts) != FINAL_ROW_COUNTS:
        raise CohortError("workload policy row counts differ from the frozen profile")
    plan: dict[str, RowPlan] = {}
    for row_id in row_order:
        count = row_counts[row_id]
        benchmark = row_benchmarks[row_id]
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or not isinstance(benchmark, str)
            or not benchmark
        ):
            raise CohortError(f"invalid policy for row {row_id!r}")
        plan[str(row_id)] = RowPlan(benchmark, count)
    if sum(item.canonical_count for item in plan.values()) != 520:
        raise CohortError("workload policy is not the exact 520-generation profile")
    canonical_categories = Counter()
    for item in plan.values():
        canonical_categories[item.benchmark] += item.canonical_count
    if dict(canonical_categories) != CATEGORY_COUNTS:
        raise CohortError("final workload must contain 130 targets per dataset")
    if (
        INTERACTIVE_CATEGORY_BASE_QUOTA * len(CATEGORY_COUNTS)
        + INTERACTIVE_BONUS_CATEGORIES
        != INTERACTIVE_TOTAL_CASES
    ):
        raise AssertionError("interactive category arithmetic is inconsistent")
    return plan


WORKLOAD_POLICY = _load_policy()
ROW_PLAN = _row_plan(WORKLOAD_POLICY)
CANONICAL_CASES = sum(plan.canonical_count for plan in ROW_PLAN.values())
INTERACTIVE_CASES = INTERACTIVE_TOTAL_CASES


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    mode: str
    submission_number: int | None = None
    run_id: str = ""
    cohort_secret: str = field(default="", repr=False)


@dataclass(frozen=True, slots=True)
class CohortSelection:
    mode: str
    submission_number: int | None
    cycle: int | None
    slot: int | None
    cases: tuple[Mapping[str, Any], ...]
    row_counts: tuple[tuple[str, int], ...]
    category_counts: tuple[tuple[str, int], ...]
    bonus_categories: tuple[str, ...]
    row_windows: tuple[tuple[str, int, int, int], ...]


def context_from_metadata(
    metadata: Mapping[str, Any],
    cohort_secret: str = "",
    *,
    calibration: bool = False,
) -> EvaluationContext:
    """Select the full workload without rewriting the trusted request identity."""
    raw_round_id = metadata.get(ROUND_ID_KEY)
    round_id = raw_round_id.strip() if isinstance(raw_round_id, str) else ""
    match = _AGENT_ROUND.fullmatch(round_id)
    if round_id.startswith(("agent-", "auto-")) and metadata.get("candidate_version") is not None:
        raise CohortError("interactive candidate_version override is forbidden")
    if round_id.startswith("agent-"):
        if match is None:
            raise CohortError("interactive round must have the form agent-N for N >= 1")
        run_id = metadata.get(RUN_ID_KEY)
        if not isinstance(run_id, str) or not run_id.strip():
            raise CohortError("interactive evaluation requires a trusted SForge run id")
        return EvaluationContext(FINAL_MODE, int(match.group(1)), run_id.strip())
    return EvaluationContext(FINAL_MODE)


def _required_text(case: Mapping[str, Any], field: str) -> str:
    value = case.get(field)
    if not isinstance(value, str) or not value:
        raise CohortError(f"canonical case has invalid {field!r}")
    return value


def _validated_buckets(
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    if len(cases) != CANONICAL_CASES:
        raise CohortError(
            f"canonical workload must contain exactly {CANONICAL_CASES} cases"
        )
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    case_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, Mapping):
            raise CohortError("canonical workload entries must be mappings")
        row_id = _required_text(case, "row_id")
        benchmark = _required_text(case, "benchmark")
        case_id = _required_text(case, "case_id")
        _required_text(case, "record_id")
        plan = ROW_PLAN.get(row_id)
        if plan is None:
            raise CohortError(f"canonical workload has unsupported row {row_id!r}")
        if benchmark != plan.benchmark:
            raise CohortError(
                f"row {row_id!r} belongs to {plan.benchmark!r}, not {benchmark!r}"
            )
        if case_id in case_ids:
            raise CohortError(f"canonical workload repeats case_id {case_id!r}")
        case_ids.add(case_id)
        buckets[row_id].append(case)

    for row_id, plan in ROW_PLAN.items():
        bucket = buckets.get(row_id, [])
        if len(bucket) != plan.canonical_count:
            raise CohortError(
                f"row {row_id!r} has {len(bucket)} cases; expected {plan.canonical_count}"
            )
        record_ids = [_required_text(case, "record_id") for case in bucket]
        if len(record_ids) != len(set(record_ids)):
            raise CohortError(f"row {row_id!r} repeats a target record")

    # A target belongs to exactly one score row.  This avoids measuring the
    # same model output shape/configuration repeatedly while retaining all 12
    # Fast-dLLM score rows.
    by_benchmark: dict[str, set[str]] = defaultdict(set)
    for row_id, plan in ROW_PLAN.items():
        row_records = {_required_text(case, "record_id") for case in buckets[row_id]}
        overlap = by_benchmark[plan.benchmark] & row_records
        if overlap:
            raise CohortError(
                f"{plan.benchmark} target records repeat across score rows"
            )
        by_benchmark[plan.benchmark].update(row_records)
    if {key: len(value) for key, value in by_benchmark.items()} != CATEGORY_COUNTS:
        raise CohortError("canonical workload must have 130 unique targets per dataset")
    return dict(buckets)


def select_cohort(
    canonical_cases: Sequence[Mapping[str, Any]],
    context: EvaluationContext,
) -> CohortSelection:
    """Execute exactly the frozen final cohort, including interactive requests."""
    _validated_buckets(canonical_cases)
    if context.mode not in {FINAL_MODE, INTERACTIVE_MODE}:
        raise CohortError(f"unsupported evaluation mode: {context.mode!r}")
    if context.mode == INTERACTIVE_MODE and (
        isinstance(context.submission_number, bool)
        or not isinstance(context.submission_number, int)
        or context.submission_number < 1
        or not context.run_id
    ):
        raise CohortError("interactive context has invalid trusted request identity")
    selected = tuple(canonical_cases)
    counts = Counter(_required_text(case, "row_id") for case in selected)
    categories = Counter(_required_text(case, "benchmark") for case in selected)
    if dict(categories) != CATEGORY_COUNTS:
        raise CohortError("complete cohort category totals differ from the frozen policy")
    return CohortSelection(
        mode=FINAL_MODE,
        submission_number=context.submission_number,
        cycle=None,
        slot=None,
        cases=selected,
        row_counts=tuple((row_id, counts[row_id]) for row_id in ROW_PLAN),
        category_counts=tuple((category, categories[category]) for category in CATEGORY_DEFINITION["categories"]),
        bonus_categories=(),
        row_windows=(),
    )
