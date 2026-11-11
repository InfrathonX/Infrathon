"""CPU-only official scoring, dense-cache binding and frozen QA policy checks."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics

PROTOCOL = "minference-ruler-quality-v3"
TASKS = ("niah_single_2", "niah_multikey_1", "vt", "fwe")
BUDGETS = (32768, 65536, 131072)
EOS_TOKEN_IDS = [128001, 128008, 128009]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def recall(prediction, outputs):
    if not outputs or not all(isinstance(s, str) and s for s in outputs):
        raise ValueError("official reference labels must be nonempty strings")
    if not isinstance(prediction, str):
        raise ValueError("prediction must be text")
    prediction = re.sub(r"[\x00-\x1f]", "\n", prediction.strip()).strip().lower()
    return sum(s.lower() in prediction for s in outputs) / len(outputs)


def official_scores(cases):
    tasks = sorted({c["task"] for c in cases})
    if not tasks or any(t not in TASKS for t in tasks):
        raise ValueError("unsupported or empty QA task set")
    scores = {task: round(100 * statistics.fmean(recall(c["prediction"], c["outputs"])
                                               for c in cases if c["task"] == task), 2)
              for task in tasks}
    return {"task_scores": scores, "macro_score": statistics.fmean(scores.values()),
            "metric": "official case-insensitive substring recall; extra answers unpenalized"}


def validate_matrix(prompts, *, samples_per_cell=None):
    if not prompts or len({p["id"] for p in prompts}) != len(prompts):
        raise ValueError("empty prompts or duplicate IDs")
    counts = Counter((p["task"], p["length"]) for p in prompts)
    if set(counts) != {(t, n) for t in TASKS for n in BUDGETS}:
        raise ValueError("QA prompts must cover all four tasks and three budgets")
    if len(set(counts.values())) != 1 or (samples_per_cell is not None and set(counts.values()) != {samples_per_cell}):
        raise ValueError("QA matrix must be balanced at the declared sample count")
    return next(iter(counts.values()))


def validate_report_cases(prompts, cases):
    if [c["id"] for c in cases] != [p["id"] for p in prompts]:
        raise ValueError("QA report IDs/order differ from its exact input set")
    for prompt, case in zip(prompts, cases):
        for field in ("task", "length", "input_token_sha256", "outputs"):
            if case[field] != prompt[field]:
                raise ValueError(f"QA report input/reference differs: {prompt['id']} {field}")
        if case["n"] != prompt["actual_input_length"] or case["max_new_tokens"] != prompt["tokens_to_generate"]:
            raise ValueError("QA report changed the real input length or reserved generation budget")
        if case["answer_score"] != round(100 * recall(case["prediction"], case["outputs"]), 2):
            raise ValueError("QA report answer score differs from the official scorer")


def validate_dense_cache(cache, bindings, prompts, *, expected_sha256=None):
    """Cache reuse is exact identity equality, independent of candidate/policy."""
    if isinstance(cache, (str, Path)):
        path = Path(cache)
        if expected_sha256 is not None and file_sha256(path) != expected_sha256:
            raise ValueError("dense baseline cache file SHA differs")
        cache = json.loads(path.read_text())
    if cache.get("protocol") != PROTOCOL or cache.get("mode") != "dense" or cache.get("status") != "complete":
        raise ValueError("dense baseline cache is incomplete or uses a different protocol")
    if cache.get("bindings") != bindings:
        raise ValueError("dense baseline model/tokenizer/input/EOS/scorer/image/source binding differs")
    validate_report_cases(prompts, cache["cases"])
    if cache.get("scores") != official_scores(cache["cases"]):
        raise ValueError("dense baseline cached aggregation differs")
    return cache


def paired_quality(dense, candidate, *, bootstrap_repeats=2000, bootstrap_seed=19750417):
    if [c["id"] for c in dense] != [c["id"] for c in candidate]:
        raise ValueError("paired quality reports differ in IDs/order")
    differences, strata = [], {}
    for d, c in zip(dense, candidate):
        if any(d[k] != c[k] for k in ("task", "length", "outputs", "input_token_sha256")):
            raise ValueError("paired quality input/reference identity differs")
        ds, cs = recall(d["prediction"], d["outputs"]), recall(c["prediction"], c["outputs"])
        row = {"id": d["id"], "task": d["task"], "length": d["length"],
               "dense_recall": ds, "candidate_recall": cs, "delta_pp": 100 * (cs - ds)}
        differences.append(row)
        strata.setdefault((d["task"], d["length"]), []).append(row["delta_pp"])
    if set(strata) != {(t, n) for t in TASKS for n in BUDGETS}:
        raise ValueError("paired quality needs the complete task/length matrix")
    means = {f"{t}:{n}": statistics.fmean(v) for (t, n), v in strata.items()}
    tasks = {t: statistics.fmean(means[f"{t}:{n}"] for n in BUDGETS) for t in TASKS}
    rng = random.Random(bootstrap_seed)
    draws = []
    for _ in range(bootstrap_repeats):
        draws.append(statistics.fmean(statistics.fmean(rng.choices(values, k=len(values)))
                                      for values in strata.values()))
    draws.sort()
    ci = None if not draws else [draws[int(0.025 * (len(draws) - 1))], draws[int(0.975 * (len(draws) - 1))]]
    perfect = [r for r in differences if r["dense_recall"] == 1.0]
    return {"macro_delta_pp_unrounded": statistics.fmean(tasks.values()),
            "task_delta_pp_unrounded": tasks, "task_length_delta_pp_unrounded": means,
            "improved": sum(r["delta_pp"] > 0 for r in differences),
            "regressed": sum(r["delta_pp"] < 0 for r in differences),
            "equal": sum(r["delta_pp"] == 0 for r in differences),
            "dense_perfect_count": len(perfect),
            "dense_perfect_regressed_count": sum(r["candidate_recall"] < 1.0 for r in perfect),
            "dense_perfect_regression_fraction": (sum(r["candidate_recall"] < 1.0 for r in perfect) / len(perfect)) if perfect else None,
            "paired_macro_bootstrap_95_interval_pp": ci,
            "bootstrap_scope": "stratified empirical percentile interval; does not bound unseen rare failures or certify a population margin",
            "cases": differences}


def evaluate_frozen_policy(policy, paired, *, require_frozen=True):
    if policy.get("protocol") != PROTOCOL:
        raise ValueError("unsupported quality policy protocol")
    if policy.get("status") != "frozen":
        if require_frozen:
            raise ValueError("quality policy must be frozen before holdout/final evaluation")
        return {"status": "diagnostic-unfrozen-policy", "passed": None,
                "policy_sha256": canonical_sha256(policy)}
    limits = policy.get("limits", {})
    for name in ("max_macro_drop_pp", "max_task_drop_pp"):
        value = limits.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"frozen quality policy requires nonnegative finite {name}")
    failures = []
    if paired["macro_delta_pp_unrounded"] < -limits["max_macro_drop_pp"]:
        failures.append("macro_drop")
    if any(v < -limits["max_task_drop_pp"] for v in paired["task_delta_pp_unrounded"].values()):
        failures.append("task_drop")
    for limit, values, name in (
            ("max_task_length_drop_pp", paired["task_length_delta_pp_unrounded"].values(), "task_length_drop"),):
        if limits.get(limit) is not None:
            value = limits[limit]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid optional quality policy limit: {limit}")
            if any(v < -value for v in values):
                failures.append(name)
    rate = limits.get("max_dense_perfect_regression_fraction")
    if rate is not None:
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError("invalid dense-perfect regression limit")
        actual = paired["dense_perfect_regression_fraction"]
        if actual is None or actual > rate:
            failures.append("dense_perfect_regression_fraction")
    return {"status": "frozen-policy-evaluated-on-declared-sample", "passed": not failures,
            "failures": failures, "policy_sha256": canonical_sha256(policy),
            "scope": "point estimates under the frozen finite-sample policy; no universal quality guarantee"}
