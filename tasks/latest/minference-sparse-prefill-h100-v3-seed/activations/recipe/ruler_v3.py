"""V3 scientific input identities and same-shape, different-layer warm-up plans."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from ruler_reference import (load_verified_prompts, input_token_sha256,
                             PROTOCOL as GENERATOR_PROTOCOL, UPSTREAM_COMMIT)
from quality_v3 import TASKS, BUDGETS, EOS_TOKEN_IDS, canonical_sha256, file_sha256, validate_matrix

PROTOCOL = "minference-real-ruler-prefill-v3"
LAYERS = (1, 4, 15, 27)
WARMUP_LAYER = 2
CAPTURE_LAYERS = (1, 2, 4, 15, 27)
PARTITIONS = ("public", "final", "calibration", "author-pilot", "author-holdout")


def derive_partition_seed(seed, task, budget, partition):
    if partition not in PARTITIONS or task not in TASKS or budget not in BUDGETS:
        raise ValueError("unsupported v3 generator domain")
    if type(seed) is not int or not 0 <= seed < 2**128:
        raise ValueError("v3 master seed must be an unsigned 128-bit integer")
    payload = f"{PROTOCOL}\0{seed}\0{partition}\0{task}\0{budget}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def load_prompts(root, *, expected_partition=None):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    prompts = load_verified_prompts(root)
    if manifest.get("v3_protocol") != PROTOCOL or manifest.get("partition") not in PARTITIONS:
        raise ValueError("prompt set is not a declared v3 partition")
    if expected_partition is not None and manifest["partition"] != expected_partition:
        raise ValueError("prompt partition differs")
    count = validate_matrix(prompts, samples_per_cell=manifest["samples_per_task_length"])
    if len({p["input_token_sha256"] for p in prompts}) != len(prompts):
        raise ValueError("v3 prompt inputs must all differ")
    if manifest.get("v3_grid") != {"tasks": list(TASKS), "budgets": list(BUDGETS), "samples_per_cell": count}:
        raise ValueError("v3 prompt matrix declaration differs")
    return prompts, manifest


def performance_plan(prompts, partition, bindings):
    if partition not in ("public", "final", "calibration"):
        raise ValueError("unsupported performance partition")
    validate_matrix(prompts, samples_per_cell=3)
    if len({p["input_token_sha256"] for p in prompts}) != 36:
        raise ValueError("performance inputs must be 36 independent prompts")
    fixtures, groups = {}, []
    for prompt in prompts:
        if not 2048 <= prompt["actual_input_length"] < prompt["length"]:
            raise ValueError("unsupported real performance token length")
        for layer in CAPTURE_LAYERS:
            identity = f"{prompt['id']}-L{layer}"
            fixtures[identity] = {"prompt_id": prompt["id"], "task": prompt["task"],
                "context_budget": prompt["length"], "layer": layer,
                "actual_n": prompt["actual_input_length"], "input_token_sha256": prompt["input_token_sha256"],
                "file": identity + ".safetensors", "trajectory": "dense-sdpa"}
    for task in TASKS:
        for budget in BUDGETS:
            ids = [p["id"] for p in prompts if p["task"] == task and p["length"] == budget]
            for layer in LAYERS:
                groups.append({"case_id": f"{partition}-{task}-n{budget}-L{layer}",
                    "task": task, "context_budget": budget, "layer": layer,
                    "warmups": [f"{identity}-L{WARMUP_LAYER}" for identity in ids],
                    "trials": [f"{identity}-L{layer}" for identity in ids]})
    return {"schema_version": 3, "protocol": PROTOCOL, "partition": partition,
            "case_groups": groups, "fixtures": fixtures, "bindings": bindings,
            "capture_state": "planned-requires-file-authentication",
            "warmup_scope": "same real prompt and N at unscored L2; no timed output or input-dependent pattern reuse permitted"}


def complete_performance_manifest(plan, records):
    if len(records) != 180 or set(records) != set(plan["fixtures"]):
        raise ValueError("capture must cover all 180 unique fixture identities")
    value = json.loads(json.dumps(plan))
    for identity, entry in value["fixtures"].items():
        record = records[identity]
        for field in ("prompt_id", "task", "context_budget", "layer", "actual_n", "input_token_sha256", "file", "trajectory"):
            if record[field] != entry[field]:
                raise ValueError(f"captured fixture identity differs: {identity} {field}")
        entry.update({"sha256": record["sha256"], "size": record["size"]})
    value["capture_state"] = "complete"
    # Validate with the same independent strict archive reader used by runtime.
    runtime_root = Path(__file__).resolve().parents[2] / "runtime"
    sys.path.insert(0, str(runtime_root))
    from activation_archive_v3 import validate_manifest
    validate_manifest(value)
    return value


def capture_bindings(recipe, prompt_manifest, prompt_root, *, image_id, software):
    return {"model": recipe["model"], "model_files_verified": True,
        "tokenizer_sha256": prompt_manifest["tokenizer_sha256"], "eos_token_ids": EOS_TOKEN_IDS,
        "source_commit": UPSTREAM_COMMIT, "source_sha256": prompt_manifest["source_sha256"],
        "generator_protocol": GENERATOR_PROTOCOL,
        "prompt_manifest_sha256": file_sha256(Path(prompt_root) / "manifest.json"),
        "prompts_sha256": prompt_manifest["prompts_sha256"],
        "recipe_protocol_sha256": file_sha256(Path(__file__).with_name("RULER_V3_RECIPE.json")),
        "ruler_v3_sha256": file_sha256(Path(__file__)),
        "image_id": image_id, "software": software,
        "layout": "BF16 contiguous BHND q32/kv8, D128, real post-RoPE QK; unpadded prompt"}
