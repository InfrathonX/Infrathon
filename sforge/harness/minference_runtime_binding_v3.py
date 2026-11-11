"""Host bootstrap helpers for the full RULER Judge; no GPU operations."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePosixPath

TASKS = ("niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1",
         "niah_multikey_2", "niah_multikey_3", "niah_multivalue", "niah_multiquery",
         "vt", "cwe", "fwe", "qa_1", "qa_2")
LENGTHS = (4096, 8192, 16384, 32768, 65536, 131072)
GPU = re.compile(r"GPU-[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")
IMAGE = re.compile(r"sha256:[0-9a-f]{64}\Z")
STATIC_BINDINGS = {"source_sha256", "model_asset_id", "model_files_sha256",
                   "generation_config_sha256"}
SOURCE_KEYS = {"quality_worker_sha256", "candidate_api_sha256", "model_recipe_sha256", "scorer_sha256"}
BUDGETS = {t: (30 if t == "vt" else 120 if t == "cwe" else 50 if t == "fwe"
               else 32 if t.startswith("qa_") else 128) for t in TASKS}
SCORER_SHA = "9ab7c2608d0e44b9ea26ea12e68127dfc8a9b61e8b4e00eff2b6e960ac8945ab"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def decode(raw):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def validate_pool_receipt(receipt, *, handle_id, execution_image, accelerator_allocation):
    """Match actual Docker inspection to the host's exact four-device allocation."""
    require(isinstance(receipt, dict) and receipt.get("Id") == handle_id
            and isinstance(handle_id, str) and re.fullmatch(r"[0-9a-f]{64}", handle_id)
            and IMAGE.fullmatch(str(execution_image)) and receipt.get("Image") == execution_image
            and receipt.get("State", {}).get("Running") is True
            and receipt.get("Config", {}).get("User") in ("root", "0"),
            "actual live root Judge handle/image identity required")
    host = receipt.get("HostConfig", {})
    require(host.get("NetworkMode") == "none" and host.get("Privileged") is False
            and host.get("Devices") in (None, []) and host.get("CapAdd") in (None, []),
            "private Judge isolation differs")
    requests = host.get("DeviceRequests")
    require(isinstance(requests, list) and len(requests) == 1
            and isinstance(requests[0], dict), "exact one physical pool request required")
    request = requests[0]
    devices = request.get("DeviceIDs")
    require(request.get("Driver") == "nvidia" and request.get("Capabilities") == [["gpu"]]
            and type(request.get("Count")) is int and request["Count"] == 0
            and isinstance(devices, list) and len(devices) == 4
            and all(isinstance(g, str) and GPU.fullmatch(g) for g in devices)
            and len(set(devices)) == 4, "four unique physical NVIDIA UUIDs required")
    require(accelerator_allocation is not None and accelerator_allocation.vendor == "nvidia"
            and tuple(accelerator_allocation.device_ids) == tuple(devices),
            "actual pool differs from exact trusted allocation")
    return tuple(devices)


def build_context(*, receipt, inspect_raw, handle_id, execution_image,
                  accelerator_allocation, manifest_raw, prompts_sha256,
                  policy_raw, dense_cache_sha256):
    """Bind full-suite bytes and frozen static execution settings before dispatch."""
    pool = validate_pool_receipt(receipt, handle_id=handle_id,
        execution_image=execution_image, accelerator_allocation=accelerator_allocation)
    manifest, policy = decode(manifest_raw), decode(policy_raw)
    require(type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1
            and manifest.get("protocol") == "minference-ruler-complete-reference-v1"
            and type(manifest.get("case_count")) is int and manifest["case_count"] == 1950
            and manifest.get("complete_reference_matrix") is True
            and manifest.get("status") == "cpu-generation-complete"
            and manifest.get("upstream") == {"repository": "https://github.com/microsoft/MInference",
                "commit": "a4eb395f949ea39e871f9bc586d683390692c6be"}, "pinned complete reference manifest required")
    require(tuple(manifest.get("tasks", ())) == TASKS
            and tuple(manifest.get("context_budgets", ())) == LENGTHS
            and type(manifest.get("num_samples_per_cell")) is int
            and manifest["num_samples_per_cell"] == 25
            and manifest.get("partition") == "judge", "formal full1950 Judge input required")
    cases = manifest.get("cases")
    expected = {f"{task}-{length}-{i:03d}" for task in TASKS for length in LENGTHS for i in range(25)}
    require(isinstance(cases, list) and len(cases) == 1950
            and {c.get("case_id") for c in cases} == expected, "full1950 coverage required")
    for case in cases:
        task, length, index = case.get("task"), case.get("context_budget"), case.get("sample_index")
        require(task in TASKS and type(length) is int and length in LENGTHS
                and type(index) is int and 0 <= index < 25
                and case["case_id"] == f"{task}-{length}-{index:03d}"
                and type(case.get("tokens_to_generate")) is int and case["tokens_to_generate"] == BUDGETS[task]
                and type(case.get("actual_input_tokens")) is int
                and 0 < case["actual_input_tokens"] <= length - BUDGETS[task]
                and SHA.fullmatch(str(case.get("token_sha256"))), "native case/reservation identity differs")
    require(SHA.fullmatch(str(prompts_sha256)) and manifest.get("prompts_sha256") == prompts_sha256,
            "declared native prompt bytes differ")
    require(type(policy.get("schema_version")) is int and policy["schema_version"] == 4
            and policy.get("protocol") == "minference-ruler-full-quality-policy-v3"
            and policy.get("status") == "frozen", "full-suite policy must be frozen")
    bindings = policy.get("execution_bindings")
    require(isinstance(bindings, dict) and set(bindings) == STATIC_BINDINGS,
            "frozen static execution bindings required")
    require(SHA.fullmatch(str(bindings.get("model_asset_id")))
            and SHA.fullmatch(str(bindings.get("generation_config_sha256"))), "model/generation pins required")
    for name in ("source_sha256", "model_files_sha256"):
        require(isinstance(bindings[name], dict) and bindings[name]
                and all(isinstance(k, str) and k and SHA.fullmatch(str(v))
                        for k, v in bindings[name].items()), "complete execution digest map required")
    require(SOURCE_KEYS <= set(bindings["source_sha256"]) and len(bindings["model_files_sha256"]) == 10,
            "complete protected source and all ten pinned model files required")
    require(bindings["source_sha256"]["scorer_sha256"] == SCORER_SHA, "original upstream scorer required")
    require(all(not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts
                and "\\" not in name and "\x00" not in name for name in bindings["model_files_sha256"]),
            "safe relative model filenames required")
    require(dense_cache_sha256 is None or SHA.fullmatch(str(dense_cache_sha256)), "exact dense cache SHA required")
    inspected = decode(inspect_raw)
    require(inspected == [receipt], "exact collected inspection bytes required")
    request = receipt["HostConfig"]["DeviceRequests"][0]
    require(tuple(request["DeviceIDs"]) == tuple(pool), "context pool differs from actual inspection")
    policy_sha = canonical_sha(policy)
    root_bindings = dict(bindings, judge_image_id=receipt["Image"], policy_sha256=policy_sha)
    return {"schema_version": 4, "protocol": "minference-ruler-full-quality-context-v3",
            "image_id": receipt["Image"], "pool_gpu_uuids": list(pool), "kernel_gpu_uuid": pool[0],
            "inspect_receipt_sha256": hashlib.sha256(inspect_raw).hexdigest(),
            "prompt_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "prompts_sha256": prompts_sha256, "quality_policy_sha256": policy_sha,
            "dense_cache_sha256": dense_cache_sha256, "root_bindings": root_bindings}
