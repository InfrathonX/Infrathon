"""PyroMind platform receipt: same guarantees as the Docker inspect, no Docker fields."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ext_v3_platform", ROOT / "scripts/external_assets_on_demand_v3.py")
ext = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ext
spec.loader.exec_module(ext)

CONTRACT = json.loads((ROOT / "qualification/external-assets-contract.json").read_text())
IMAGE_ID = "sha256:" + "a" * 64
GPU = "GPU-9fe0e721-2c87-b44b-1369-5744ddd1726b"
SOURCES = {"private-performance": "/workspace/sforge-assets/v2/judge-private/datasets/kp/files",
           "model": "/workspace/sforge-assets/v2/judge-private/models/m/files"}


def bindings():
    rows = []
    for row in CONTRACT["assets"]:
        rows.append({**row, "mount_path": ext.MOUNTS[row["role"]], "source_directory": SOURCES[row["role"]],
                     "read_only": True, "visibility": "private", "consumers": ["judge"],
                     "storage_identity": ext.STORAGE_IDENTITY})
    return {"schema_version": 1, "protocol": ext.PROTOCOL, "assets": rows}


def receipt():
    return {"platform": "pyromind-sandbox-v1", "sandbox_id": "sb-0123456789ab",
            "image": "infrathon/infrathon@" + IMAGE_ID, "image_id": IMAGE_ID, "user": "root",
            "gpu_uuids": [GPU],
            "asset_mounts": [{"destination": ext.MOUNTS[r], "source": s, "read_only": True}
                             for r, s in SOURCES.items()],
            "other_mounts": [{"destination": "/workspace/benchmark/performance-prompts", "read_only": True},
                             {"destination": "/sforge/candidate", "read_only": True}],
            "network": "platform-default-not-attested", "evidence": {"nvidia_smi": "uuid"}}


def check(value, image_id=IMAGE_ID):
    return ext.validate_inspection(value, bindings(), image_id, contract=CONTRACT)


def test_platform_receipt_gives_the_docker_provenance_shape():
    provenance = check(receipt())
    assert provenance["container_id"] == "sb-0123456789ab"
    assert provenance["image_id"] == IMAGE_ID
    assert {m["mount_path"] for m in provenance["mounts"]} == set(ext.MOUNTS.values())
    assert ext.judge_gpu_uuid(receipt()) == GPU


@pytest.mark.parametrize("damage", [
    lambda r: r.update(network="none"),
    lambda r: r.update(gpu_uuids=[GPU, GPU.replace("9fe0", "9fe1")]),
    lambda r: r.update(gpu_uuids=[]),
    lambda r: r.update(image="infrathon/infrathon@sha256:" + "b" * 64),
    lambda r: r.update(image_id="sha256:" + "b" * 64),
    lambda r: r.update(user="agent"),
    lambda r: r.update(sandbox_id="0123456789ab"),
    lambda r: r.update(extra=1),
    lambda r: r.pop("evidence"),
    lambda r: r["asset_mounts"][0].update(read_only=False),
    lambda r: r["asset_mounts"][0].update(source="/workspace/elsewhere"),
    lambda r: r["asset_mounts"].pop(),
    lambda r: r["other_mounts"].append({"destination": "/opt/minference_prefill_judge/private", "read_only": True}),
    lambda r: r["other_mounts"].append({"destination": "/opt/minference_prefill_judge", "read_only": True}),
    lambda r: r["other_mounts"].append({"destination": "/scratch", "read_only": False}),
    lambda r: r["other_mounts"].append({"destination": ext.MOUNTS["model"], "read_only": True}),
])
def test_platform_receipt_rejects_damage(damage):
    value = receipt()
    damage(value)
    with pytest.raises(ValueError):
        check(value)


def test_docker_branch_is_unchanged():
    inspection = {"Id": "c" * 64, "Image": IMAGE_ID, "State": {"Running": True}, "Config": {"User": "root"},
                  "HostConfig": {"NetworkMode": "none", "Privileged": False, "Devices": None,
                                 "DeviceRequests": [{"Driver": "nvidia", "DeviceIDs": [GPU]}]},
                  "Mounts": [{"Type": "bind", "RW": False, "Destination": ext.MOUNTS[r], "Source": s}
                             for r, s in SOURCES.items()]}
    assert check([inspection])["container_id"] == "c" * 64
    assert ext.judge_gpu_uuid([inspection]) == GPU
    for key, value in (("NetworkMode", "bridge"), ("Privileged", True)):
        broken = copy.deepcopy(inspection)
        broken["HostConfig"][key] = value
        with pytest.raises(ValueError):
            check(broken)
    # A Docker inspect cannot claim the platform branch's relaxed network rule.
    with pytest.raises(ValueError):
        check({**inspection, "platform": "pyromind-sandbox-v1"})
