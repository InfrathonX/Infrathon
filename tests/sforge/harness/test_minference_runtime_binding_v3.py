"""CPU fixtures prove fail-closed binding checks; no hardware execution."""
import copy
import json
from types import SimpleNamespace

import pytest
from sforge.harness import minference_runtime_binding_v3 as Q


POOL = [f"GPU-{i:08x}-aaaa-bbbb-cccc-dddddddddddd" for i in range(4)]
IMAGE = "sha256:" + "a" * 64
CID = "c" * 64


def receipt():
    return {"Id": CID, "Image": IMAGE, "State": {"Running": True}, "Config": {"User": "root"},
            "HostConfig": {"NetworkMode": "none", "Privileged": False, "Devices": [], "CapAdd": [],
                           "DeviceRequests": [{"Driver": "nvidia", "Capabilities": [["gpu"]],
                                               "Count": 0, "DeviceIDs": list(POOL)}]}}


def allocation(devices=POOL):
    return SimpleNamespace(vendor="nvidia", device_ids=tuple(devices))


def validate(value, alloc=None):
    return Q.validate_pool_receipt(value, handle_id=CID, execution_image=IMAGE,
                                  accelerator_allocation=alloc or allocation())


def inputs():
    manifest = {"schema_version": 1, "protocol": "minference-ruler-complete-reference-v1",
                "case_count": 1950, "complete_reference_matrix": True, "status": "cpu-generation-complete",
                "upstream": {"repository": "https://github.com/microsoft/MInference",
                    "commit": "a4eb395f949ea39e871f9bc586d683390692c6be"},
                "tasks": list(Q.TASKS), "context_budgets": list(Q.LENGTHS), "num_samples_per_cell": 25,
                "partition": "judge", "prompts_sha256": "b" * 64, "cases": []}
    for task in Q.TASKS:
        for length in Q.LENGTHS:
            for index in range(25):
                manifest["cases"].append({"case_id": f"{task}-{length}-{index:03d}", "task": task,
                    "context_budget": length, "sample_index": index, "actual_input_tokens": 2000,
                    "tokens_to_generate": Q.BUDGETS[task], "token_sha256": "d" * 64})
    policy = {"schema_version": 4, "protocol": "minference-ruler-full-quality-policy-v3", "status": "frozen",
              "execution_bindings": {"model_asset_id": "e" * 64, "generation_config_sha256": "f" * 64,
                    "source_sha256": {name: "1" * 64 for name in Q.SOURCE_KEYS},
                    "model_files_sha256": {f"model-file-{i}": "2" * 64 for i in range(10)}}}
    policy["execution_bindings"]["source_sha256"]["scorer_sha256"] = Q.SCORER_SHA
    return manifest, policy


def context(manifest, policy, actual=None, inspect_override=None):
    actual = actual or receipt()
    return Q.build_context(receipt=actual, inspect_raw=inspect_override or json.dumps([actual]).encode(),
        handle_id=CID, execution_image=IMAGE, accelerator_allocation=allocation(),
        manifest_raw=json.dumps(manifest).encode(), prompts_sha256="b" * 64,
        policy_raw=json.dumps(policy).encode(), dense_cache_sha256="3" * 64)


def test_exact_four_device_host_binding_accepts_and_detaches_context():
    assert validate(receipt()) == tuple(POOL)
    manifest, policy = inputs()
    value = context(manifest, policy)
    assert len(value) == 11 and value["schema_version"] == 4
    assert value["pool_gpu_uuids"] == POOL and value["kernel_gpu_uuid"] == POOL[0]
    assert value["root_bindings"]["policy_sha256"] == Q.canonical_sha(policy)
    assert value["root_bindings"]["judge_image_id"] == IMAGE
    assert "plan_sha256" not in value


@pytest.mark.parametrize("mutation", ["one-device", "duplicate", "indices", "all", "count-minus-one", "count-bool",
    "two-requests", "capability", "vendor", "order", "image", "handle", "stopped", "agent", "network", "privileged", "extra-device", "extra-cap"])
def test_actual_identity_mutations_fail_closed(mutation):
    value = receipt(); host = value["HostConfig"]; request = host["DeviceRequests"][0]; alloc = allocation()
    if mutation == "one-device": request["DeviceIDs"] = POOL[:1]
    elif mutation == "duplicate": request["DeviceIDs"][1] = POOL[0]
    elif mutation == "indices": request["DeviceIDs"] = ["4", "5", "6", "7"]
    elif mutation == "all": request["DeviceIDs"] = ["all"]
    elif mutation == "count-minus-one": request["Count"] = -1
    elif mutation == "count-bool": request["Count"] = False
    elif mutation == "two-requests": host["DeviceRequests"] *= 2
    elif mutation == "capability": request["Capabilities"] = [["gpu", "compute"]]
    elif mutation == "vendor": alloc.vendor = "other"
    elif mutation == "order": alloc = allocation(list(reversed(POOL)))
    elif mutation == "image": value["Image"] = "tag:latest"
    elif mutation == "handle": value["Id"] = "9" * 64
    elif mutation == "stopped": value["State"]["Running"] = False
    elif mutation == "agent": value["Config"]["User"] = "agent"
    elif mutation == "network": host["NetworkMode"] = "bridge"
    elif mutation == "privileged": host["Privileged"] = True
    elif mutation == "extra-device": host["Devices"] = [{"PathOnHost": "/dev/other"}]
    else: host["CapAdd"] = ["SYS_ADMIN"]
    with pytest.raises(ValueError): validate(value, alloc)


@pytest.mark.parametrize("mutation", ["four-tasks", "three-lengths", "two-samples", "old300", "duplicate-case",
    "calibration", "wrong-input-sha", "wrong-task", "over-budget", "short-budget", "missing-source", "missing-model",
    "oldpolicy", "float-schema", "float-token-budget", "wrong-scorer", "absolute-model-file", "false-matrix",
    "wrong-case-count", "unfrozen", "binding-self-image", "replaced-inspection"])
def test_partial_or_rebound_protocol_cannot_produce_full_judge_context(mutation):
    manifest, policy = inputs(); inspected = None
    if mutation == "four-tasks": manifest["tasks"] = manifest["tasks"][:4]
    elif mutation == "three-lengths": manifest["context_budgets"] = manifest["context_budgets"][3:]
    elif mutation == "two-samples": manifest["num_samples_per_cell"] = 2
    elif mutation == "old300": manifest["cases"] = manifest["cases"][:300]
    elif mutation == "duplicate-case": manifest["cases"][0] = copy.deepcopy(manifest["cases"][1])
    elif mutation == "calibration": manifest["partition"] = "calibration"
    elif mutation == "wrong-input-sha": manifest["prompts_sha256"] = "4" * 64
    elif mutation == "wrong-task": manifest["cases"][0]["task"] = "vt"
    elif mutation == "over-budget": manifest["cases"][0]["actual_input_tokens"] = 4096
    elif mutation == "short-budget": manifest["cases"][0]["tokens_to_generate"] = 1
    elif mutation == "missing-source": policy["execution_bindings"]["source_sha256"].pop("scorer_sha256")
    elif mutation == "missing-model": policy["execution_bindings"]["model_files_sha256"].pop("model-file-0")
    elif mutation == "oldpolicy": policy["schema_version"] = 3
    elif mutation == "float-schema": policy["schema_version"] = 4.0
    elif mutation == "float-token-budget": manifest["cases"][0]["tokens_to_generate"] = 128.0
    elif mutation == "wrong-scorer": policy["execution_bindings"]["source_sha256"]["scorer_sha256"] = "6" * 64
    elif mutation == "absolute-model-file": policy["execution_bindings"]["model_files_sha256"]["/absolute/model"] = policy["execution_bindings"]["model_files_sha256"].pop("model-file-0")
    elif mutation == "false-matrix": manifest["complete_reference_matrix"] = False
    elif mutation == "wrong-case-count": manifest["case_count"] = 300
    elif mutation == "unfrozen": policy["status"] = "calibration"
    elif mutation == "binding-self-image": policy["execution_bindings"]["judge_image_id"] = IMAGE
    else: inspected = json.dumps([dict(receipt(), Image="sha256:" + "5" * 64)]).encode()
    with pytest.raises(ValueError): context(manifest, policy, inspect_override=inspected)


def test_duplicate_json_key_cannot_replace_frozen_execution_binding():
    with pytest.raises(ValueError, match="duplicate"):
        Q.decode(b'{"status":"frozen","status":"draft"}')
