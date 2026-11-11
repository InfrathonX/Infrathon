#!/usr/bin/env python3
"""Freeze public fixtures and the Judge-only stratified evaluation manifest.

This maintainer-only script queries pinned Hugging Face dataset revisions. The
standard test splits are public and reproducible, but their materialized
prompts, ordering, and outputs are copied only into the Judge image.  Public
fixtures deliberately use train data or synthetic examples so they consume no
otherwise eligible official test record.
"""

from __future__ import annotations

import argparse
from collections import deque
import gzip
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


DATASETS = {
    "gsm8k": {
        "dataset": "openai/gsm8k",
        "config": "main",
        "revision": "e53f048856ff4f594e959d75785d2c2d37b678ee",
    },
    "math": {
        "dataset": "EleutherAI/hendrycks_math",
        "revision": "21a5633873b6a120296cce3e2df9d5550074f4a3",
    },
    "humaneval": {
        "dataset": "openai/openai_humaneval",
        "config": "openai_humaneval",
        "revision": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
    },
    "mbpp": {
        "dataset": "google-research-datasets/mbpp",
        "config": "full",
        "revision": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
    },
}

TOKENIZER_REPO_ID = "GSAI-ML/LLaDA-8B-Instruct"
TOKENIZER_REVISION = "6059b30c9531767ce47e0ca6a8161a6bb0c67693"
TRANSFORMERS_VERSION = "4.49.0"
TOKENIZERS_VERSION = "0.21.0"
MODEL_CONTEXT_LIMIT = 4096
TOKENIZER_FILES = {
    "config.json": (
        1385,
        "5f99fefe855fdb5100bb6cadb57bdb09fae723ad54811f95c00ecacf29d58a6a",
    ),
    "special_tokens_map.json": (
        747,
        "e9f1f0f6fa133da3169ae69b1969c51cced047a900050893d7a0711c6bd063aa",
    ),
    "tokenizer.json": (
        6098988,
        "071eb20fe7bd601550b1b7838ff696d6c93b88f51257f753303d3df23163c381",
    ),
    "tokenizer_config.json": (
        51692,
        "b9a6102c39a4f42f47f0d63fc6cee58a2381199d90f2a4f39b3f6871f32b4119",
    ),
}

TASK_ID = "llada_h100_decoder_optimization_v2"
WORKLOAD_PROFILE = "fast_dllm_v1_llada_12row_stratified520_batch1"
FAST_DLLM_REVISION = "a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b"
LM_EVAL_REVISION = "6d2abda4fd171e68a8789330c4149e37c1ca0bda"
EXPECTED_GENERATIONS = 520
EXPECTED_REQUESTED_SLOTS = 216_320
MBPP_CONTEXT_EXCLUSION = "test:482/task:493"

# Frozen ordering nonce.  These exact bytes are part of the scientific identity
# of the committed 520-case suite: every stratified sort, every tie-break and
# every row assignment is keyed on them, so editing the literal silently
# re-draws the whole hidden workload instead of reproducing the frozen one.
# The literal is an opaque nonce, not a label -- do not rename it, reformat it
# or "modernise" it.  ORDERING_SEED_SHA256 is the value the frozen
# docs/dataset_manifest.json records, and check_ordering_seed() compares the
# two before anything is generated, so an edit fails loudly at start-up rather
# than producing a different benchmark that still passes every shape check.
ORDERING_SEED = b"edgebench-h100-v2-stratified520-order-2026-08-15"
ORDERING_SEED_SHA256 = "d6bf180decc6e5e642eda324be566e52129aed012409076b662f92297dca2d50"
MATH_CONFIGS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)

# Public fixtures are intentionally outside all official test splits used by
# the Judge.  HumanEval has no train/dev split in the pinned dataset, so its
# public fixture is task-owned synthetic code.
GSM_PUBLIC = [("train", index) for index in range(8, 18)]
MATH_PUBLIC = [
    ("algebra", "train", 4),
    ("algebra", "train", 5),
    ("counting_and_probability", "train", 4),
    ("counting_and_probability", "train", 5),
    ("geometry", "train", 4),
    ("geometry", "train", 5),
    ("intermediate_algebra", "train", 4),
    ("number_theory", "train", 4),
    ("prealgebra", "train", 4),
    ("precalculus", "train", 4),
]
MBPP_PUBLIC = [("train", index) for index in range(10, 20)]

PUBLIC_GSM_ROW_SPECS = (
    ("table1_gsm8k_5shot", 256, 5, 3),
    ("table1_gsm8k_5shot", 512, 5, 3),
    ("table4_gsm8k_5shot", 1024, 5, 1),
    ("table5_gsm8k_8shot", 256, 8, 1),
    ("table5_gsm8k_8shot", 512, 8, 1),
    ("table5_gsm8k_8shot", 1024, 8, 1),
)
PUBLIC_MATH_ROW_SPECS = (
    ("table1_math_4shot", 256, 4, 5),
    ("table1_math_4shot", 512, 4, 5),
)
PUBLIC_HUMANEVAL_ROW_SPECS = (
    ("table1_humaneval_0shot", 256, 0, 5),
    ("table1_humaneval_0shot", 512, 0, 5),
)
PUBLIC_MBPP_ROW_SPECS = (
    ("table1_mbpp_3shot", 256, 3, 5),
    ("table1_mbpp_3shot", 512, 3, 5),
)

GSM_ROW_SPECS = (
    ("table1_gsm8k_5shot", 256, 5, 39),
    ("table1_gsm8k_5shot", 512, 5, 39),
    ("table4_gsm8k_5shot", 1024, 5, 13),
    ("table5_gsm8k_8shot", 256, 8, 13),
    ("table5_gsm8k_8shot", 512, 8, 13),
    ("table5_gsm8k_8shot", 1024, 8, 13),
)
MATH_ROW_SPECS = (
    ("table1_math_4shot", 256, 4, 65),
    ("table1_math_4shot", 512, 4, 65),
)
HUMANEVAL_ROW_SPECS = (
    ("table1_humaneval_0shot", 256, 0, 65),
    ("table1_humaneval_0shot", 512, 0, 65),
)
MBPP_ROW_SPECS = (
    ("table1_mbpp_3shot", 256, 3, 65),
    ("table1_mbpp_3shot", 512, 3, 65),
)

HIDDEN_ROW_ORDER = (
    "table1_gsm8k_5shot_g256",
    "table1_gsm8k_5shot_g512",
    "table1_math_4shot_g256",
    "table1_math_4shot_g512",
    "table1_humaneval_0shot_g256",
    "table1_humaneval_0shot_g512",
    "table1_mbpp_3shot_g256",
    "table1_mbpp_3shot_g512",
    "table4_gsm8k_5shot_g1024",
    "table5_gsm8k_8shot_g256",
    "table5_gsm8k_8shot_g512",
    "table5_gsm8k_8shot_g1024",
)

_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

MBPP_FEWSHOT = [
    {
        "text": "Write a function to find the similar elements from the given two tuple lists.",
        "code": "def similar_elements(test_tup1, test_tup2):\n  res = tuple(set(test_tup1) & set(test_tup2))\n  return (res) ",
        "test_list": [
            "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
            "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
            "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
        ],
    },
    {
        "text": "Write a python function to identify non-prime numbers.",
        "code": "import math\ndef is_not_prime(n):\n    result = False\n    for i in range(2,int(math.sqrt(n)) + 1):\n        if n % i == 0:\n            result = True\n    return result",
        "test_list": [
            "assert is_not_prime(2) == False",
            "assert is_not_prime(10) == True",
            "assert is_not_prime(35) == True",
        ],
    },
    {
        "text": "Write a function to find the largest integers from a given list of numbers using heap queue algorithm.",
        "code": "import heapq as hq\ndef heap_queue_largest(nums,n):\n  largest_nums = hq.nlargest(n, nums)\n  return largest_nums",
        "test_list": [
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ",
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ",
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
        ],
    },
]

_ROW_CACHE: dict[tuple[str, str, str, int], dict[str, Any]] = {}
_LAST_REQUEST_AT = 0.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_ordering_seed() -> None:
    """Fail before generating anything if the frozen ordering nonce moved."""
    observed = hashlib.sha256(ORDERING_SEED).hexdigest()
    if observed != ORDERING_SEED_SHA256:
        raise RuntimeError(
            "ORDERING_SEED no longer matches the frozen stratified-520 ordering "
            f"nonce (expected {ORDERING_SEED_SHA256}, got {observed}); every "
            "stratified sort and row assignment is keyed on those exact bytes, "
            "so generation would draw a different hidden workload"
        )


def _tree_data_files(dataset: str, revision: str) -> dict[str, str]:
    """Content ids of the data files a dataset repository carries at a revision."""
    url = (
        "https://huggingface.co/api/datasets/"
        + urllib.parse.quote(dataset, safe="/")
        + "/tree/"
        + urllib.parse.quote(revision, safe="")
        + "?recursive=1"
    )
    request = urllib.request.Request(
        url, headers={"User-Agent": "mlsysforge-data-freezer/1"}
    )
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                entries = json.load(response)
            break
        except Exception as error:  # transient proxy / rate-limit failures
            last_error = error
            if attempt == 4:
                raise RuntimeError(
                    f"cannot resolve pinned revision {revision} of {dataset}: {last_error}"
                ) from error
            time.sleep(min(30, 2 ** (attempt + 1)))
    files: dict[str, str] = {}
    for entry in entries:
        if entry.get("type") != "file":
            continue
        path = str(entry.get("path", ""))
        if not path.endswith((".parquet", ".json", ".jsonl", ".csv", ".zip")):
            continue
        lfs = entry.get("lfs") or {}
        content_id = lfs.get("oid") or entry.get("oid")
        if not isinstance(content_id, str):
            raise RuntimeError(f"no content id for {dataset}@{revision}:{path}")
        files[path] = content_id
    if not files:
        raise RuntimeError(f"no data files found for {dataset}@{revision}")
    return files


def check_pinned_dataset_revisions() -> None:
    """Prove the pinned revisions are the data the row endpoint will serve.

    The ``/rows`` endpoint takes no revision: it answers from the repository's
    current default branch and ignores the ``revision`` query parameter this
    script sends, so a pin alone does not make a fetch reproducible.  Compare
    the data files carried at the pinned revision with the ones carried at the
    branch the endpoint reads, and refuse to generate when they differ instead
    of silently freezing whatever upstream published last.
    """
    for name, spec in sorted(DATASETS.items()):
        dataset = spec["dataset"]
        pinned = _tree_data_files(dataset, spec["revision"])
        served = _tree_data_files(dataset, "main")
        if pinned != served:
            drifted = sorted(
                path
                for path in set(pinned) | set(served)
                if pinned.get(path) != served.get(path)
            )
            raise RuntimeError(
                f"{name}: the data files of {dataset} moved since revision "
                f"{spec['revision']}; the row endpoint serves the default "
                "branch and cannot be pinned, so this workload is no longer "
                f"reproducible from the pins (differing paths: {drifted})"
            )


def resolve_tokenizer_snapshot(model_repo: Path) -> Path:
    """Resolve and authenticate the exact tokenizer used by the Judge image."""
    direct = model_repo.resolve()
    snapshot = (
        direct
        if (direct / "tokenizer.json").is_file()
        else direct / "snapshots" / TOKENIZER_REVISION
    )
    if not snapshot.is_dir():
        raise RuntimeError(
            f"missing pinned tokenizer snapshot {TOKENIZER_REVISION} under {model_repo}"
        )
    for name, (expected_size, expected_sha256) in TOKENIZER_FILES.items():
        path = snapshot / name
        if not path.is_file():
            raise RuntimeError(f"pinned tokenizer file is missing: {path}")
        actual_size = path.stat().st_size
        actual_sha256 = file_sha256(path)
        if (actual_size, actual_sha256) != (expected_size, expected_sha256):
            raise RuntimeError(
                f"pinned tokenizer file drift for {name}: "
                f"{actual_size}/{actual_sha256}"
            )
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    if config.get("max_sequence_length") != MODEL_CONTEXT_LIMIT:
        raise RuntimeError("pinned model context limit differs from 4096")
    return snapshot


def load_pinned_tokenizer(model_repo: Path) -> Any:
    snapshot = resolve_tokenizer_snapshot(model_repo)
    try:
        import tokenizers
        import transformers
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "exact context validation requires transformers==4.49.0 and "
            "tokenizers==0.21.0; run this generator in the pinned Work image"
        ) from exc
    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"transformers version drift: {transformers.__version__} "
            f"!= {TRANSFORMERS_VERSION}"
        )
    if tokenizers.__version__ != TOKENIZERS_VERSION:
        raise RuntimeError(
            f"tokenizers version drift: {tokenizers.__version__} "
            f"!= {TOKENIZERS_VERSION}"
        )
    return AutoTokenizer.from_pretrained(
        str(snapshot), local_files_only=True, trust_remote_code=False
    )


def exact_prompt_token_count(tokenizer: Any, request_text: str) -> int:
    """Mirror verifier.encode_prompt byte-for-byte at the API level."""
    chat = [{"role": "user", "content": request_text}]
    rendered = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, tokenize=False
    )
    encoded = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if (
        not isinstance(encoded, list)
        or not encoded
        or any(isinstance(token, bool) or not isinstance(token, int) for token in encoded)
    ):
        raise RuntimeError("pinned tokenizer returned invalid prompt ids")
    return len(encoded)


def context_eligible(
    *,
    benchmark: str,
    identity: str,
    request_text_by_fewshot: dict[int, str],
    specs: tuple[tuple[str, int, int, int], ...],
    tokenizer: Any,
    exclusions: list[dict[str, Any]],
) -> dict[int, int] | None:
    """Reject a target if any retained request would exceed model context."""
    prompt_counts = {
        fewshot: exact_prompt_token_count(tokenizer, request_text)
        for fewshot, request_text in sorted(request_text_by_fewshot.items())
    }
    violations = []
    for spec in specs:
        current_row = row_id(spec)
        _group, gen_length, fewshot, _count = spec
        prompt_count = prompt_counts[fewshot]
        total = prompt_count + gen_length
        if total > MODEL_CONTEXT_LIMIT:
            violations.append(
                {
                    "row_id": current_row,
                    "prompt_token_count": prompt_count,
                    "gen_length": gen_length,
                    "total_tokens": total,
                }
            )
    if violations:
        exclusions.append(
            {
                "benchmark": benchmark,
                "record_id": identity,
                "violations": violations,
            }
        )
        return None
    return prompt_counts


def _fetch_payload(params: dict[str, str]) -> dict[str, Any]:
    global _LAST_REQUEST_AT
    url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(params)
    payload = None
    for attempt in range(8):
        try:
            delay = 1.1 - (time.monotonic() - _LAST_REQUEST_AT)
            if delay > 0:
                time.sleep(delay)
            request = urllib.request.Request(url, headers={"User-Agent": "mlsysforge-data-freezer/1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.load(response)
            _LAST_REQUEST_AT = time.monotonic()
            return payload
        except Exception:
            if attempt == 7:
                raise
            time.sleep(min(60, 2 ** (attempt + 1)))
    raise AssertionError("unreachable")


def prefetch(name: str, split: str, indices: list[int], *, config: str | None = None) -> None:
    spec = DATASETS[name]
    resolved_config = config or spec["config"]
    wanted = sorted(set(indices))
    runs: list[list[int]] = []
    for index in wanted:
        if not runs or index != runs[-1][-1] + 1:
            runs.append([index])
        else:
            runs[-1].append(index)
    for run in runs:
        params = {
            "dataset": spec["dataset"],
            "config": resolved_config,
            "split": split,
            "revision": spec["revision"],
            "offset": str(run[0]),
            "length": str(len(run)),
        }
        payload = _fetch_payload(params)
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != len(run) or payload.get("partial"):
            raise RuntimeError(f"unexpected dataset response for {name}/{split}/{run}")
        for expected, wrapped in zip(run, rows):
            if wrapped.get("row_idx") != expected:
                raise RuntimeError(f"row index mismatch for {name}/{split}/{expected}")
            _ROW_CACHE[(name, resolved_config, split, expected)] = wrapped["row"]


def prefetch_all(name: str, split: str, *, config: str | None = None) -> int:
    """Fetch a complete pinned split in bounded datasets-server pages."""
    spec = DATASETS[name]
    resolved_config = config or spec["config"]
    offset = 0
    total: int | None = None
    page_size = 100
    while total is None or offset < total:
        params = {
            "dataset": spec["dataset"],
            "config": resolved_config,
            "split": split,
            "revision": spec["revision"],
            "offset": str(offset),
            "length": str(page_size),
        }
        payload = _fetch_payload(params)
        rows = payload.get("rows")
        reported_total = payload.get("num_rows_total")
        if not isinstance(rows, list) or not isinstance(reported_total, int):
            raise RuntimeError(f"unexpected dataset response for {name}/{split}/{offset}")
        if payload.get("partial") or reported_total <= 0:
            raise RuntimeError(f"partial dataset response for {name}/{split}/{offset}")
        if total is None:
            total = reported_total
        elif total != reported_total:
            raise RuntimeError(f"dataset size changed during fetch for {name}/{split}")
        expected = min(page_size, total - offset)
        if len(rows) != expected:
            raise RuntimeError(f"wrong page size for {name}/{split}/{offset}")
        for row_offset, wrapped in enumerate(rows):
            index = offset + row_offset
            if wrapped.get("row_idx") != index:
                raise RuntimeError(f"row index mismatch for {name}/{split}/{index}")
            _ROW_CACHE[(name, resolved_config, split, index)] = wrapped["row"]
        offset += len(rows)
    assert total is not None
    return total


def fetch_row(name: str, split: str, index: int, *, config: str | None = None) -> dict[str, Any]:
    spec = DATASETS[name]
    resolved_config = config or spec["config"]
    cache_key = (name, resolved_config, split, index)
    if cache_key in _ROW_CACHE:
        return _ROW_CACHE[cache_key]
    params = {
        "dataset": spec["dataset"],
        "config": resolved_config,
        "split": split,
        "revision": spec["revision"],
        "offset": str(index),
        "length": "1",
    }
    payload = _fetch_payload(params)
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 1 or rows[0].get("row_idx") != index:
        raise RuntimeError(f"unexpected dataset response for {name}/{split}/{index}")
    if payload.get("partial"):
        raise RuntimeError(f"partial dataset response for {name}/{split}/{index}")
    _ROW_CACHE[cache_key] = rows[0]["row"]
    return _ROW_CACHE[cache_key]


def lexical_token_count(text: str) -> int:
    """Return a dependency-free, revision-stable prompt-length proxy."""
    return len(_TOKEN_PATTERN.findall(text))


def stable_key(namespace: str, identity: str) -> bytes:
    return hashlib.sha256(
        ORDERING_SEED + b"\0" + namespace.encode() + b"\0" + identity.encode()
    ).digest()


def quantile_bins(
    candidates: list[dict[str, Any]], metric: str, namespace: str, *, bins: int = 4
) -> dict[str, int]:
    """Assign exact rank quantiles, with seeded tie-breaking."""
    ordered = sorted(
        candidates,
        key=lambda item: (
            item[metric],
            stable_key(f"{namespace}:{metric}:tie", item["identity"]),
        ),
    )
    return {
        item["identity"]: min(bins - 1, rank * bins // len(ordered))
        for rank, item in enumerate(ordered)
    }


def stratified_order(
    candidates: list[dict[str, Any]],
    primary: str,
    secondary: str | None,
    namespace: str,
) -> list[dict[str, Any]]:
    """Round-robin primary and secondary strata in deterministic seed order."""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for candidate in candidates:
        primary_value = str(candidate[primary])
        secondary_value = str(candidate[secondary]) if secondary else "all"
        grouped.setdefault(primary_value, {}).setdefault(secondary_value, []).append(candidate)

    primary_order = sorted(
        grouped,
        key=lambda value: stable_key(f"{namespace}:primary", value),
    )
    secondary_order = sorted(
        {
            secondary_value
            for primary_value in primary_order
            for secondary_value in grouped[primary_value]
        },
        key=lambda value: stable_key(f"{namespace}:secondary", value),
    )
    queues = {
        (primary_value, secondary_value): deque(
            sorted(
                grouped[primary_value][secondary_value],
                key=lambda item: stable_key(
                    f"{namespace}:member:{primary_value}:{secondary_value}",
                    item["identity"],
                ),
            )
        )
        for primary_value in primary_order
        for secondary_value in grouped[primary_value]
    }
    primary_turn = {primary_value: 0 for primary_value in primary_order}
    ordered: list[dict[str, Any]] = []
    while True:
        progressed = False
        for primary_position, primary_value in enumerate(primary_order):
            start = (primary_position + primary_turn[primary_value]) % len(
                secondary_order
            )
            for delta in range(len(secondary_order)):
                position = (start + delta) % len(secondary_order)
                secondary_value = secondary_order[position]
                queue = queues.get((primary_value, secondary_value))
                if queue:
                    ordered.append(queue.popleft())
                    primary_turn[primary_value] += delta + 1
                    progressed = True
                    break
        if not progressed:
            return ordered


def row_id(spec: tuple[str, int, int, int]) -> str:
    group, gen_length, _fewshot, _count = spec
    return f"{group}_g{gen_length}"


def assign_rows(
    candidates: list[dict[str, Any]],
    specs: tuple[tuple[str, int, int, int], ...],
    primary: str,
    secondary: str | None,
    namespace: str,
) -> dict[str, list[dict[str, Any]]]:
    """Take one stratified sample and partition it across disjoint rows."""
    ordered = stratified_order(candidates, primary, secondary, namespace)
    expected = sum(spec[3] for spec in specs)
    if len(ordered) < expected:
        raise RuntimeError(
            f"insufficient eligible {namespace} targets: {len(ordered)} < {expected}"
        )
    assignments: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    for spec in specs:
        count = spec[3]
        assignments[row_id(spec)] = ordered[cursor : cursor + count]
        cursor += count
    assigned = [
        candidate["identity"]
        for current_row in assignments.values()
        for candidate in current_row
    ]
    if len(assigned) != expected or len(set(assigned)) != expected:
        raise RuntimeError(f"{namespace} row assignment repeated a target")
    return assignments


def last_boxed(solution: str) -> str:
    marker = max(solution.rfind("\\boxed"), solution.rfind("\\fbox"))
    if marker < 0:
        raise ValueError("MATH solution has no boxed answer")
    tail = solution[marker:]
    if tail.startswith("\\boxed "):
        return tail[len("\\boxed ") :].split("$")[0]
    start = tail.find("{")
    if start < 0:
        raise ValueError("malformed boxed answer")
    depth = 0
    for index in range(start, len(tail)):
        if tail[index] == "{":
            depth += 1
        elif tail[index] == "}":
            depth -= 1
            if depth == 0:
                return tail[start + 1 : index]
    raise ValueError("unterminated boxed answer")


def gsm_context(doc: dict[str, Any], shots: list[dict[str, Any]]) -> str:
    chunks = [f"Question: {row['question']}\nAnswer:{row['answer']}" for row in shots]
    chunks.append(f"Question: {doc['question']}\nAnswer:")
    return "\n\n".join(chunks)


def math_context(doc: dict[str, Any], shots: list[dict[str, Any]]) -> str:
    chunks = [
        f"Problem: {row['problem']}\nAnswer:{last_boxed(row['solution'])}"
        for row in shots
    ]
    chunks.append(f"Problem: {doc['problem']}\nAnswer:")
    return "\n\n".join(chunks)


def mbpp_text(doc: dict[str, Any]) -> str:
    tests = doc["test_list"]
    return (
        "You are an expert Python programmer, and here is your task: "
        f"{doc['text']} Your code should pass these tests:\n\n"
        f"{tests[0]}\n{tests[1]}\n{tests[2]}\n[BEGIN]\n"
    )


def mbpp_context(doc: dict[str, Any]) -> str:
    chunks = [mbpp_text(row) + row["code"] + "\n[DONE]" for row in MBPP_FEWSHOT]
    chunks.append(mbpp_text(doc))
    return "\n\n".join(chunks)


def gsm_case(index: int, fewshot: int, *, split: str = "test") -> dict[str, Any]:
    shots = [fetch_row("gsm8k", "train", i) for i in range(fewshot)]
    doc = fetch_row("gsm8k", split, index)
    return {
        "benchmark": "gsm8k",
        "record_id": f"{split}:{index}",
        "source_split": split,
        "fewshot": fewshot,
        "request_text": gsm_context(doc, shots),
        "stop": ["Question:", "</s>", "<|im_end|>"],
        "gold": doc["answer"].split("####")[-1].strip(),
    }


def math_case(config: str, index: int, *, split: str = "test") -> dict[str, Any]:
    shots = [fetch_row("math", "train", i, config=config) for i in range(4)]
    doc = fetch_row("math", split, index, config=config)
    return {
        "benchmark": "math",
        "record_id": f"{config}/{split}:{index}",
        "source_split": split,
        "fewshot": 4,
        "request_text": math_context(doc, shots),
        "stop": ["Problem:"],
        "gold": last_boxed(doc["solution"]),
        "math_solution": doc["solution"],
    }


def humaneval_case(index: int) -> dict[str, Any]:
    doc = fetch_row("humaneval", "test", index)
    return {
        "benchmark": "humaneval",
        "record_id": doc["task_id"],
        "source_split": "test",
        "fewshot": 0,
        "request_text": doc["prompt"],
        "stop": ["\nclass", "\ndef", "\n#", "\nif", "\nprint"],
        "prompt": doc["prompt"],
        "entry_point": doc["entry_point"],
        "tests": doc["test"] + f"\ncheck({doc['entry_point']})\n",
    }


HUMANEVAL_PUBLIC = (
    (
        "mlsysforge_sum_squares",
        "values",
        "Return the sum of the squares of the integers in values.",
        "assert candidate([]) == 0\n"
        "    assert candidate([1, -2, 3]) == 14\n"
        "    assert candidate([5]) == 25",
    ),
    (
        "mlsysforge_clamp",
        "value, lower, upper",
        "Clamp value to the inclusive interval from lower to upper.",
        "assert candidate(5, 0, 10) == 5\n"
        "    assert candidate(-3, 0, 10) == 0\n"
        "    assert candidate(13, 0, 10) == 10",
    ),
    (
        "mlsysforge_count_vowels",
        "text",
        "Count ASCII vowels in text without regard to case.",
        "assert candidate('MLSys Forge') == 3\n"
        "    assert candidate('rhythms') == 0\n"
        "    assert candidate('AEIOU') == 5",
    ),
    (
        "mlsysforge_pair_differences",
        "values",
        "Return each value minus the value immediately before it.",
        "assert candidate([]) == []\n"
        "    assert candidate([4]) == []\n"
        "    assert candidate([4, 9, 3, 3]) == [5, -6, 0]",
    ),
    (
        "mlsysforge_is_palindrome",
        "text",
        "Return whether text is the same forward and backward.",
        "assert candidate('level') is True\n"
        "    assert candidate('edge') is False\n"
        "    assert candidate('') is True",
    ),
    (
        "mlsysforge_flatten_pairs",
        "pairs",
        "Flatten a sequence of two-item pairs into one list.",
        "assert candidate([]) == []\n"
        "    assert candidate([(1, 2), (3, 4)]) == [1, 2, 3, 4]\n"
        "    assert candidate([('a', 'b')]) == ['a', 'b']",
    ),
    (
        "mlsysforge_gcd",
        "a, b",
        "Return the nonnegative greatest common divisor of two integers.",
        "assert candidate(54, 24) == 6\n"
        "    assert candidate(-12, 8) == 4\n"
        "    assert candidate(0, 7) == 7",
    ),
    (
        "mlsysforge_rotate_left",
        "values, steps",
        "Return a list rotated left by steps positions.",
        "assert candidate([], 3) == []\n"
        "    assert candidate([1, 2, 3, 4], 1) == [2, 3, 4, 1]\n"
        "    assert candidate([1, 2, 3], 5) == [3, 1, 2]",
    ),
    (
        "mlsysforge_word_lengths",
        "words",
        "Map each string in words to its length.",
        "assert candidate([]) == []\n"
        "    assert candidate(['a', 'edge', '']) == [1, 4, 0]\n"
        "    assert candidate(['GPU']) == [3]",
    ),
    (
        "mlsysforge_unique_stable",
        "values",
        "Remove duplicate values while preserving first-occurrence order.",
        "assert candidate([]) == []\n"
        "    assert candidate([3, 1, 3, 2, 1]) == [3, 1, 2]\n"
        "    assert candidate(['a', 'a', 'b']) == ['a', 'b']",
    ),
)


def synthetic_humaneval_public_case(index: int) -> dict[str, Any]:
    """Return one task-owned public code fixture outside HumanEval."""
    entry_point, arguments, docstring, assertions = HUMANEVAL_PUBLIC[index]
    prompt = f'def {entry_point}({arguments}):\n    """{docstring}"""\n'
    return {
        "benchmark": "humaneval",
        "record_id": f"synthetic:{entry_point}_v1",
        "source_split": "synthetic",
        "fewshot": 0,
        "request_text": prompt,
        "stop": ["\nclass", "\ndef", "\n#", "\nif", "\nprint"],
        "prompt": prompt,
        "entry_point": entry_point,
        "tests": (
            "def check(candidate):\n"
            f"    {assertions}\n"
            f"check({entry_point})\n"
        ),
    }


def mbpp_case(index: int, *, split: str = "test") -> dict[str, Any]:
    doc = fetch_row("mbpp", split, index)
    return {
        "benchmark": "mbpp",
        "record_id": f"{split}:{index}/task:{doc['task_id']}",
        "source_split": split,
        "fewshot": 3,
        "request_text": mbpp_context(doc),
        "stop": ["[DONE]"],
        "tests": [*doc["test_list"], *doc.get("challenge_test_list", [])],
        "test_setup_code": doc.get("test_setup_code", ""),
    }


def gsm_candidates(
    total: int, tokenizer: Any, exclusions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    candidates = []
    shots_by_count = {
        fewshot: [fetch_row("gsm8k", "train", i) for i in range(fewshot)]
        for fewshot in {spec[2] for spec in GSM_ROW_SPECS}
    }
    for index in range(total):
        doc = fetch_row("gsm8k", "test", index)
        prompt_counts = context_eligible(
            benchmark="gsm8k",
            identity=f"test:{index}",
            request_text_by_fewshot={
                fewshot: gsm_context(doc, shots)
                for fewshot, shots in shots_by_count.items()
            },
            specs=GSM_ROW_SPECS,
            tokenizer=tokenizer,
            exclusions=exclusions,
        )
        if prompt_counts is None:
            continue
        candidates.append(
            {
                "identity": f"test:{index}",
                "index": index,
                "prompt_tokens": lexical_token_count(doc["question"]),
                "answer_tokens": lexical_token_count(doc["answer"]),
                "exact_prompt_tokens": prompt_counts,
            }
        )
    prompt_bins = quantile_bins(candidates, "prompt_tokens", "gsm8k")
    answer_bins = quantile_bins(candidates, "answer_tokens", "gsm8k")
    for candidate in candidates:
        identity = candidate["identity"]
        candidate["prompt_quartile"] = prompt_bins[identity]
        candidate["answer_quartile"] = answer_bins[identity]
        candidate["stratum"] = {
            "answer_length_quartile": answer_bins[identity],
            "prompt_length_quartile": prompt_bins[identity],
        }
    return candidates


def math_candidates(
    totals: dict[str, int], tokenizer: Any, exclusions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    candidates = []
    for config in MATH_CONFIGS:
        shots = [fetch_row("math", "train", i, config=config) for i in range(4)]
        for index in range(totals[config]):
            doc = fetch_row("math", "test", index, config=config)
            try:
                last_boxed(doc["solution"])
            except ValueError:
                continue
            level = doc.get("level") or "unknown"
            identity = f"{config}/test:{index}"
            prompt_counts = context_eligible(
                benchmark="math",
                identity=identity,
                request_text_by_fewshot={4: math_context(doc, shots)},
                specs=MATH_ROW_SPECS,
                tokenizer=tokenizer,
                exclusions=exclusions,
            )
            if prompt_counts is None:
                continue
            candidates.append(
                {
                    "identity": identity,
                    "index": index,
                    "config": config,
                    "subject": config,
                    "difficulty": level,
                    "stratum": {"difficulty": level, "subject": config},
                    "exact_prompt_tokens": prompt_counts,
                }
            )
    return candidates


def humaneval_candidates(
    total: int, tokenizer: Any, exclusions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    candidates = []
    for index in range(total):
        doc = fetch_row("humaneval", "test", index)
        prompt_counts = context_eligible(
            benchmark="humaneval",
            identity=doc["task_id"],
            request_text_by_fewshot={0: doc["prompt"]},
            specs=HUMANEVAL_ROW_SPECS,
            tokenizer=tokenizer,
            exclusions=exclusions,
        )
        if prompt_counts is None:
            continue
        candidates.append(
            {
                "identity": doc["task_id"],
                "index": index,
                "prompt_tokens": lexical_token_count(doc["prompt"]),
                "exact_prompt_tokens": prompt_counts,
            }
        )
    prompt_bins = quantile_bins(candidates, "prompt_tokens", "humaneval")
    for candidate in candidates:
        quartile = prompt_bins[candidate["identity"]]
        candidate["prompt_quartile"] = quartile
        candidate["stratum"] = {"prompt_length_quartile": quartile}
    return candidates


def mbpp_candidates(
    total: int, tokenizer: Any, exclusions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    candidates = []
    for index in range(total):
        doc = fetch_row("mbpp", "test", index)
        if len(doc.get("test_list", [])) < 3:
            continue
        identity = f"test:{index}/task:{doc['task_id']}"
        prompt_counts = context_eligible(
            benchmark="mbpp",
            identity=identity,
            request_text_by_fewshot={3: mbpp_context(doc)},
            specs=MBPP_ROW_SPECS,
            tokenizer=tokenizer,
            exclusions=exclusions,
        )
        if prompt_counts is None:
            continue
        candidates.append(
            {
                "identity": identity,
                "index": index,
                "prompt_tokens": lexical_token_count(mbpp_text(doc)),
                "assertions": len(doc["test_list"])
                + len(doc.get("challenge_test_list", [])),
                "exact_prompt_tokens": prompt_counts,
            }
        )
    prompt_bins = quantile_bins(candidates, "prompt_tokens", "mbpp")
    assertion_bins = quantile_bins(candidates, "assertions", "mbpp")
    for candidate in candidates:
        identity = candidate["identity"]
        candidate["prompt_quartile"] = prompt_bins[identity]
        candidate["assertion_quartile"] = assertion_bins[identity]
        candidate["stratum"] = {
            "assertion_count_quartile": assertion_bins[identity],
            "prompt_length_quartile": prompt_bins[identity],
        }
    return candidates


def add_workloads(cases: list[dict[str, Any]], lengths: list[int], group: str) -> list[dict[str, Any]]:
    out = []
    for length in lengths:
        for case in cases:
            item = dict(case)
            item["gen_length"] = length
            item["block_length"] = 32
            item["row_id"] = f"{group}_g{length}"
            item["case_id"] = f"{item['row_id']}:{item['record_id']}"
            out.append(item)
    return out


def materialize_row(
    spec: tuple[str, int, int, int],
    candidates: list[dict[str, Any]],
    factory: Any,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    group, gen_length, fewshot, expected_count = spec
    if len(candidates) != expected_count:
        raise RuntimeError(f"wrong assignment count for {row_id(spec)}")
    cases = []
    for candidate in candidates:
        case = factory(candidate, fewshot)
        prompt_token_count = exact_prompt_token_count(tokenizer, case["request_text"])
        if prompt_token_count != candidate["exact_prompt_tokens"][fewshot]:
            raise RuntimeError(
                f"prompt changed between eligibility and materialization: "
                f"{candidate['identity']}"
            )
        if prompt_token_count + gen_length > MODEL_CONTEXT_LIMIT:
            raise RuntimeError(
                f"materialized case exceeds model context: {candidate['identity']}"
            )
        case["prompt_token_count"] = prompt_token_count
        case["stratum"] = candidate["stratum"]
        cases.append(case)
    return add_workloads(cases, [gen_length], group)


def materialize_public_rows(
    specs: tuple[tuple[str, int, int, int], ...],
    targets: list[Any],
    factory: Any,
    tokenizer: Any,
) -> list[dict[str, Any]]:
    """Partition ten public targets across representative final row shapes."""
    if sum(spec[3] for spec in specs) != len(targets):
        raise RuntimeError("public row allocation does not consume exactly ten targets")
    output: list[dict[str, Any]] = []
    cursor = 0
    for spec in specs:
        group, gen_length, fewshot, count = spec
        for target in targets[cursor : cursor + count]:
            case = factory(target, fewshot)
            prompt_token_count = exact_prompt_token_count(tokenizer, case["request_text"])
            if prompt_token_count + gen_length > MODEL_CONTEXT_LIMIT:
                raise RuntimeError(
                    f"public case exceeds model context: {case['record_id']}"
                )
            case["prompt_token_count"] = prompt_token_count
            item = add_workloads([case], [gen_length], group)[0]
            item["public_profile"] = "full"
            output.append(item)
        cursor += count
    if len({(case["benchmark"], case["record_id"]) for case in output}) != len(targets):
        raise RuntimeError("public full profile repeats a target")
    return output


def materialize_quick_public_case(
    case: dict[str, Any], tokenizer: Any
) -> dict[str, Any]:
    prompt_token_count = exact_prompt_token_count(tokenizer, case["request_text"])
    if prompt_token_count + 128 > MODEL_CONTEXT_LIMIT:
        raise RuntimeError(f"quick public case exceeds model context: {case['record_id']}")
    case["prompt_token_count"] = prompt_token_count
    item = add_workloads([case], [128], f"quick_{case['benchmark']}")[0]
    item["public_profile"] = "quick"
    return item


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def write_json(path: Path, value: Any, *, dry_run: bool = False) -> str:
    data = canonical_json_bytes(value)
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def deterministic_gzip(data: bytes) -> bytes:
    """Return gzip bytes stable across supported CPython versions.

    CPython has emitted either 3 (Unix) or 255 (unknown) in the gzip OS header
    for ``gzip.compress(..., mtime=0)``.  The byte is metadata only, so freeze
    it explicitly instead of letting the host interpreter change artifact
    identity.
    """
    compressed = bytearray(gzip.compress(data, compresslevel=9, mtime=0))
    compressed[9] = 3
    return bytes(compressed)


def write_json_gzip(path: Path, value: Any, *, dry_run: bool = False) -> tuple[str, str]:
    """Write deterministic gzip while committing to canonical JSON bytes."""
    data = canonical_json_bytes(value)
    compressed = deterministic_gzip(data)
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(compressed)
    return hashlib.sha256(data).hexdigest(), hashlib.sha256(compressed).hexdigest()


def load_bootstrap_workload_policy(output_root: Path) -> dict[str, Any]:
    """Return the target tree's policy, refusing anything but UNQUALIFIED/PENDING."""
    policy = json.loads(
        (output_root / "workload_policy.json").read_text(encoding="utf-8")
    )
    if (
        not isinstance(policy, dict)
        or policy.get("task_id") != TASK_ID
        or policy.get("profile") != WORKLOAD_PROFILE
        or policy.get("qualification_status") != "UNQUALIFIED"
        or policy.get("release_status") != "PENDING"
    ):
        raise RuntimeError(
            "data regeneration may update only an UNQUALIFIED/PENDING workload policy"
        )
    return policy


def sync_unqualified_workload_policy(
    output_root: Path, *, hidden_sha256: str, dataset_manifest_sha256: str, dry_run: bool = False
) -> str | None:
    """Bind the bootstrap policy to regenerated manifests without freezing it.

    A dry run over a released tree reports ``None`` instead of refusing: the
    frozen policy is exactly what such a run is there to leave alone, and the
    generated digests above are the answer it was asked for.
    """
    root_path = output_root / "workload_policy.json"
    private_path = output_root / "images/judge/private_data/workload_policy.json"
    try:
        policy = load_bootstrap_workload_policy(output_root)
    except RuntimeError:
        if dry_run:
            return None
        raise
    policy["manifest"]["hidden_cases_sha256"] = hidden_sha256
    policy["manifest"]["dataset_manifest_sha256"] = dataset_manifest_sha256
    policy["manifest"]["row_order"] = list(HIDDEN_ROW_ORDER)
    data = canonical_json_bytes(policy)
    if not dry_run:
        root_path.write_bytes(data)
        private_path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument(
        "--model-repo",
        type=Path,
        default=Path(
            os.environ.get("LLADA_MODEL_REPO")
            or Path.home()
            / ".cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct"
        ),
        help="HF cache repository or exact snapshot for the pinned LLaDA tokenizer",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "write the generated files under this tree instead of --task-root; "
            "point it at a scratch copy to verify a released tree without "
            "touching it (default: --task-root)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute and print every digest, write no file anywhere",
    )
    args = parser.parse_args()
    output_root = args.output_root or args.task_root
    # Refuse a frozen tree, resolve the pins and authenticate the ordering nonce
    # BEFORE a single byte is written: a released tree must come out of a
    # refused run unmodified.
    check_ordering_seed()
    if not args.dry_run:
        load_bootstrap_workload_policy(output_root)
    check_pinned_dataset_revisions()
    tokenizer = load_pinned_tokenizer(args.model_repo)

    prefetch("gsm8k", "train", list(range(18)))
    gsm_total = prefetch_all("gsm8k", "test")
    math_totals = {}
    for config in MATH_CONFIGS:
        prefetch("math", "train", list(range(6)), config=config)
        math_totals[config] = prefetch_all("math", "test", config=config)
    humaneval_total = prefetch_all("humaneval", "test")
    prefetch("mbpp", "train", list(range(10, 20)))
    mbpp_total = prefetch_all("mbpp", "test")

    public_full = [
        *materialize_public_rows(
            PUBLIC_GSM_ROW_SPECS,
            list(GSM_PUBLIC),
            lambda target, fewshot: gsm_case(target[1], fewshot, split=target[0]),
            tokenizer,
        ),
        *materialize_public_rows(
            PUBLIC_MATH_ROW_SPECS,
            list(MATH_PUBLIC),
            lambda target, _fewshot: math_case(
                target[0], target[2], split=target[1]
            ),
            tokenizer,
        ),
        *materialize_public_rows(
            PUBLIC_HUMANEVAL_ROW_SPECS,
            list(range(len(HUMANEVAL_PUBLIC))),
            lambda target, _fewshot: synthetic_humaneval_public_case(target),
            tokenizer,
        ),
        *materialize_public_rows(
            PUBLIC_MBPP_ROW_SPECS,
            list(MBPP_PUBLIC),
            lambda target, _fewshot: mbpp_case(target[1], split=target[0]),
            tokenizer,
        ),
    ]
    quick_base = [
        gsm_case(GSM_PUBLIC[0][1], 5, split=GSM_PUBLIC[0][0]),
        math_case(MATH_PUBLIC[0][0], MATH_PUBLIC[0][2], split=MATH_PUBLIC[0][1]),
        synthetic_humaneval_public_case(0),
        mbpp_case(MBPP_PUBLIC[0][1], split=MBPP_PUBLIC[0][0]),
    ]
    public_quick = [materialize_quick_public_case(case, tokenizer) for case in quick_base]
    public = [*public_quick, *public_full]
    public_targets = {
        (case["benchmark"], case["record_id"]) for case in public
    }
    if len(public) != 44 or len(public_targets) != 40:
        raise RuntimeError("public profiles must contain 44 entries / 40 unique targets")

    exclusions: dict[str, list[dict[str, Any]]] = {
        benchmark: [] for benchmark in ("gsm8k", "math", "humaneval", "mbpp")
    }
    candidate_pools = {
        "gsm8k": gsm_candidates(gsm_total, tokenizer, exclusions["gsm8k"]),
        "math": math_candidates(math_totals, tokenizer, exclusions["math"]),
        "humaneval": humaneval_candidates(
            humaneval_total, tokenizer, exclusions["humaneval"]
        ),
        "mbpp": mbpp_candidates(mbpp_total, tokenizer, exclusions["mbpp"]),
    }
    expected_source_totals = {
        "gsm8k": 1319,
        "math": 5000,
        "humaneval": 164,
        "mbpp": 500,
    }
    observed_source_totals = {
        "gsm8k": gsm_total,
        "math": sum(math_totals.values()),
        "humaneval": humaneval_total,
        "mbpp": mbpp_total,
    }
    if observed_source_totals != expected_source_totals:
        raise RuntimeError(
            f"pinned standard test split sizes changed: {observed_source_totals}"
        )
    expected_eligible = {"gsm8k": 1319, "math": 5000, "humaneval": 164, "mbpp": 499}
    observed_eligible = {
        benchmark: len(candidates) for benchmark, candidates in candidate_pools.items()
    }
    if observed_eligible != expected_eligible:
        raise RuntimeError(
            f"context-compatible candidate counts changed: {observed_eligible}"
        )
    observed_exclusions = {
        (entry["benchmark"], entry["record_id"])
        for entries in exclusions.values()
        for entry in entries
    }
    if observed_exclusions != {("mbpp", MBPP_CONTEXT_EXCLUSION)}:
        raise RuntimeError(f"unexpected context exclusions: {observed_exclusions}")

    gsm_assignments = assign_rows(
        candidate_pools["gsm8k"],
        GSM_ROW_SPECS,
        "prompt_quartile",
        "answer_quartile",
        "gsm8k",
    )
    math_assignments = assign_rows(
        candidate_pools["math"],
        MATH_ROW_SPECS,
        "subject",
        "difficulty",
        "math",
    )
    humaneval_assignments = assign_rows(
        candidate_pools["humaneval"],
        HUMANEVAL_ROW_SPECS,
        "prompt_quartile",
        None,
        "humaneval",
    )
    mbpp_assignments = assign_rows(
        candidate_pools["mbpp"],
        MBPP_ROW_SPECS,
        "prompt_quartile",
        "assertion_quartile",
        "mbpp",
    )

    rows: dict[str, list[dict[str, Any]]] = {}
    for spec in GSM_ROW_SPECS:
        rows[row_id(spec)] = materialize_row(
            spec,
            gsm_assignments[row_id(spec)],
            lambda candidate, fewshot: gsm_case(candidate["index"], fewshot),
            tokenizer,
        )
    for spec in MATH_ROW_SPECS:
        rows[row_id(spec)] = materialize_row(
            spec,
            math_assignments[row_id(spec)],
            lambda candidate, _fewshot: math_case(
                candidate["config"], candidate["index"]
            ),
            tokenizer,
        )
    for spec in HUMANEVAL_ROW_SPECS:
        rows[row_id(spec)] = materialize_row(
            spec,
            humaneval_assignments[row_id(spec)],
            lambda candidate, _fewshot: humaneval_case(candidate["index"]),
            tokenizer,
        )
    for spec in MBPP_ROW_SPECS:
        rows[row_id(spec)] = materialize_row(
            spec,
            mbpp_assignments[row_id(spec)],
            lambda candidate, _fewshot: mbpp_case(candidate["index"]),
            tokenizer,
        )
    hidden = [case for current_row in HIDDEN_ROW_ORDER for case in rows[current_row]]
    row_counts = {
        current_row: len(rows[current_row]) for current_row in HIDDEN_ROW_ORDER
    }
    total_generations = sum(row_counts.values())
    requested_slots = sum(
        int(case["gen_length"]) for current_row in HIDDEN_ROW_ORDER for case in rows[current_row]
    )
    if (total_generations, requested_slots) != (
        EXPECTED_GENERATIONS,
        EXPECTED_REQUESTED_SLOTS,
    ):
        raise RuntimeError(
            "stratified workload total mismatch: "
            f"{total_generations}/{requested_slots}"
        )

    selected_record_sets = {}
    for benchmark in ("gsm8k", "math", "humaneval", "mbpp"):
        benchmark_rows = [
            current_row
            for current_row in HIDDEN_ROW_ORDER
            if rows[current_row][0]["benchmark"] == benchmark
        ]
        benchmark_cases = [case for current_row in benchmark_rows for case in rows[current_row]]
        benchmark_targets = [case["record_id"] for case in benchmark_cases]
        if len(benchmark_targets) != 130 or len(set(benchmark_targets)) != 130:
            raise RuntimeError(f"{benchmark} must assign 130 targets without replacement")
        selected_record_sets[benchmark] = {
            "rows": benchmark_rows,
            "record_count": len(benchmark_targets),
            "row_counts": {
                current_row: len(rows[current_row]) for current_row in benchmark_rows
            },
            "record_set_sha256": hashlib.sha256(
                ("\n".join(sorted(benchmark_targets)) + "\n").encode("utf-8")
            ).hexdigest(),
        }

    public_path = output_root / "images/work/public_tests/public_cases.json"
    hidden_path = output_root / "images/judge/private_data/hidden_cases.json.gz"
    legacy_hidden_path = hidden_path.with_suffix("")
    public_sha = write_json(public_path, public, dry_run=args.dry_run)
    hidden_sha, hidden_compressed_sha = write_json_gzip(
        hidden_path, hidden, dry_run=args.dry_run
    )
    if not args.dry_run and (
        legacy_hidden_path.exists() or legacy_hidden_path.is_symlink()
    ):
        legacy_hidden_path.unlink()
    hidden_selection = {
        current_row: [case["record_id"] for case in rows[current_row]]
        for current_row in HIDDEN_ROW_ORDER
    }
    manifest = {
        "schema_version": 5,
        "task_id": TASK_ID,
        "workload_profile": WORKLOAD_PROFILE,
        "fast_dllm_revision": FAST_DLLM_REVISION,
        "lm_eval_revision": LM_EVAL_REVISION,
        "coverage_policy": "stratified_130_per_dataset_without_replacement",
        "standard_test_data": {
            "classification": "public_reproducible_judge_only",
            "mounted_in_work": False,
            "implicit_limit": 130,
            "repeats": 1,
            "effective_batch_size": 1,
            "source_record_counts": expected_source_totals,
            "eligible_record_counts": expected_eligible,
        },
        "datasets": DATASETS,
        "tokenizer": {
            "repo_id": TOKENIZER_REPO_ID,
            "revision": TOKENIZER_REVISION,
            "transformers_version": TRANSFORMERS_VERSION,
            "tokenizers_version": TOKENIZERS_VERSION,
            "model_context_limit": MODEL_CONTEXT_LIMIT,
            "prompt_encoding": (
                "apply_chat_template(user, add_generation_prompt=True, tokenize=False); "
                "tokenizer(rendered, add_special_tokens=False)"
            ),
            "files": {
                name: [size, digest]
                for name, (size, digest) in sorted(TOKENIZER_FILES.items())
            },
        },
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "ordering_seed_sha256": hashlib.sha256(ORDERING_SEED).hexdigest(),
        "public_cases": len(public),
        "public_unique_targets": len(public_targets),
        "public_profile_counts": {
            profile: sum(case["public_profile"] == profile for case in public)
            for profile in ("quick", "full")
        },
        "public_target_counts": {
            benchmark: len(
                {
                    case["record_id"]
                    for case in public
                    if case["benchmark"] == benchmark
                }
            )
            for benchmark in ("gsm8k", "math", "humaneval", "mbpp")
        },
        "public_full_row_counts": {
            current_row: sum(
                case["public_profile"] == "full" and case["row_id"] == current_row
                for case in public
            )
            for current_row in HIDDEN_ROW_ORDER
        },
        "public_sha256": public_sha,
        "public_fixture_policy": {
            "official_test_records": 0,
            "target_disjoint_from_hidden": True,
            "target_disjoint_from_fewshot": True,
            "quick_reuses_one_full_target_per_dataset": True,
            "sources": [
                *(
                    {"benchmark": "gsm8k", "split": split, "index": index}
                    for split, index in GSM_PUBLIC
                ),
                *(
                    {
                        "benchmark": "math",
                        "config": config,
                        "split": split,
                        "index": index,
                    }
                    for config, split, index in MATH_PUBLIC
                ),
                *(
                    {
                        "benchmark": "humaneval",
                        "split": "synthetic",
                        "id": f"{entry_point}_v1",
                    }
                    for entry_point, _arguments, _docstring, _assertions in HUMANEVAL_PUBLIC
                ),
                *(
                    {"benchmark": "mbpp", "split": split, "index": index}
                    for split, index in MBPP_PUBLIC
                ),
            ],
            "fewshot_sources": {
                "gsm8k": [f"train:{index}" for index in range(8)],
                "math": [
                    f"{config}/train:{index}"
                    for config in MATH_CONFIGS
                    for index in range(4)
                ],
                "humaneval": [],
                "mbpp": ["prompt:1/task:2", "prompt:2/task:3", "prompt:3/task:4"],
            },
        },
        "hidden_cases": len(hidden),
        "generation_count": total_generations,
        "requested_slots": requested_slots,
        "hidden_sha256": hidden_sha,
        "hidden_storage": {
            "path": "images/judge/private_data/hidden_cases.json.gz",
            "compression": "gzip-9-mtime0",
            "compressed_sha256": hidden_compressed_sha,
            "canonical_uncompressed_sha256": hidden_sha,
        },
        "hidden_target_counts": {
            benchmark: len(
                {
                    case["record_id"]
                    for case in hidden
                    if case["benchmark"] == benchmark
                }
            )
            for benchmark in ("gsm8k", "math", "humaneval", "mbpp")
        },
        "hidden_row_counts": row_counts,
        "hidden_row_order": list(HIDDEN_ROW_ORDER),
        "row_requested_slots": {
            current_row: sum(int(case["gen_length"]) for case in rows[current_row])
            for current_row in HIDDEN_ROW_ORDER
        },
        "selected_record_sets": selected_record_sets,
        "stratification": {
            "gsm8k": ["prompt_length_rank_quartile", "answer_length_rank_quartile"],
            "math": ["subject", "difficulty"],
            "humaneval": ["prompt_length_rank_quartile"],
            "mbpp": ["prompt_length_rank_quartile", "assertion_count_rank_quartile"],
            "length_metric": "dependency_free_lexical_token_count",
        },
        "context_validation": {
            "policy": (
                "before stratification, exclude a target if any retained row for "
                "its benchmark has exact_prompt_tokens + gen_length > 4096"
            ),
            "candidate_pool_after_filter": {
                benchmark: len(candidate_pools[benchmark])
                for benchmark in ("gsm8k", "math", "humaneval", "mbpp")
            },
            "excluded_candidate_counts": {
                benchmark: len(exclusions[benchmark])
                for benchmark in ("gsm8k", "math", "humaneval", "mbpp")
            },
            "excluded_candidates": exclusions,
            "validated_hidden_cases": len(hidden),
            "documented_exclusions": [
                {
                    "benchmark": "mbpp",
                    "record_id": MBPP_CONTEXT_EXCLUSION,
                    "reason": "prompt_plus_generation_exceeds_4096",
                }
            ],
            "maximum_hidden_total_tokens": max(
                case["prompt_token_count"] + case["gen_length"] for case in hidden
            ),
        },
        "selection": {
            "gsm_public": GSM_PUBLIC,
            "math_public": MATH_PUBLIC,
            "humaneval_public": [
                f"synthetic:{entry_point}_v1"
                for entry_point, _arguments, _docstring, _assertions in HUMANEVAL_PUBLIC
            ],
            "mbpp_public": MBPP_PUBLIC,
            "hidden_by_row": hidden_selection,
        },
    }
    manifest_sha = write_json(
        output_root / "docs/dataset_manifest.json", manifest, dry_run=args.dry_run
    )
    workload_policy_sha = sync_unqualified_workload_policy(
        output_root,
        hidden_sha256=hidden_sha,
        dataset_manifest_sha256=manifest_sha,
        dry_run=args.dry_run,
    )
    print(
        json.dumps(
            {
                "public": public_sha,
                "hidden": hidden_sha,
                "manifest": manifest_sha,
                "workload_policy": workload_policy_sha,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
