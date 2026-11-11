from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sforge.harness import candidate_replay
from sforge.harness.benchmark import BenchmarkMeta
from sforge.harness.config import SForgeConfig
from sforge.harness.portable_spec import (
    CANONICAL_JSON_VERSION,
    PortableSpecError,
    canonical_json,
    canonical_sha256,
    parse_portable_spec,
    verify_sha256_bytes,
    verify_sha256_file,
)
from sforge.harness.task_spec import make_task_spec


def _digest(character: str) -> str:
    return "sha256:" + character * 64


def _protocol(protocol_id: str, character: str) -> dict[str, object]:
    return {"protocol_id": protocol_id, "spec_sha256": _digest(character)}


def _portable() -> dict[str, object]:
    return {
        "schema_version": 1,
        "task_core": {
            "contract_id": "grouped-gemm-v1",
            "definition": {
                "operation": "grouped_gemm",
                "dtypes": ["float8_e4m3fn", "bfloat16"],
            },
            "interface": {
                "entrypoint": "submission.kernel.run",
                "source_path": "submission/kernel.py",
            },
            "allowed_changes": ["submission implementation"],
            "invariants": ["output matches the reference within tolerance"],
            "resource_manifest_sha256": _digest("a"),
        },
        "evaluation_suites": {
            "standard": {
                "roles": {
                    "public": {
                        "manifest_sha256": _digest("b"),
                        "case_count": 4,
                    },
                    "anchor": {
                        "manifest_sha256": _digest("c"),
                        "case_count": 8,
                    },
                    "final": {
                        "manifest_sha256": _digest("d"),
                        "case_count": 16,
                    },
                },
                "oracle": _protocol("bf16-reference-v1", "e"),
                "selection": _protocol("fixed-cases-v1", "f"),
            }
        },
        "execution_profiles": {
            "single-gpu": {
                "vendor": "nvidia",
                "count": 1,
                "min_memory_bytes": 80_000_000_000,
                "min_compute_capability": "9.0",
                "features": ["fp8", "tensor_cores"],
                "software_abi": _protocol("cuda-python-abi-v1", "0"),
                "calibration_protocol": _protocol(
                    "latency-calibration-v1", "1"
                ),
            }
        },
        "metrics": {
            "latency": {
                "direction": "minimize",
                "unit": "microseconds",
                "comparability": "same_measurement_profile",
                "aggregation": {
                    "protocol_id": "geomean-v1",
                    "spec": {"grouping": "layout_family", "equal_weight": True},
                },
                "baseline_policy": "none",
            }
        },
        "defaults": {
            "evaluation_suite_id": "standard",
            "execution_profile_id": "single-gpu",
            "metric_id": "latency",
        },
    }


def _legacy_task() -> dict[str, object]:
    return {
        "task_id": "portable_test",
        "name": "Portable test",
        "base_image": "python",
        "platform": "linux/amd64",
        "cwd": "/workspace",
        "submit_paths": ["submission"],
        "work": {"image_tag": "work-v1"},
        "judge": {
            "eval_cmd": "python evaluate.py",
            "eval_timeout": 60,
            "parser": "json",
            "image_tag": "judge-v1",
            "score_direction": "minimize",
        },
    }


def _write_task(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_portable_contract_normalizes_and_digests_every_layer() -> None:
    first = parse_portable_spec(_portable(), task_id="portable_test")
    reordered = copy.deepcopy(_portable())
    definition = reordered["task_core"]["definition"]  # type: ignore[index]
    reordered["task_core"]["definition"] = {  # type: ignore[index]
        "dtypes": definition["dtypes"],  # type: ignore[index]
        "operation": definition["operation"],  # type: ignore[index]
    }
    profile = reordered["execution_profiles"]["single-gpu"]  # type: ignore[index]
    profile["vendor"] = " NVIDIA "  # type: ignore[index]
    profile["features"] = ["TENSOR_CORES", " fp8 "]  # type: ignore[index]
    second = parse_portable_spec(reordered, task_id="portable_test")

    assert first.to_dict() == second.to_dict()
    assert set(first.layer_digests) == {
        "task_core",
        "evaluation_suites",
        "execution_profiles",
        "metrics",
        "defaults",
    }
    assert all(
        digest.startswith("sha256:") and len(digest) == 71
        for digest in first.layer_digests.values()
    )
    assert first.evaluation_contract_sha256 == second.evaluation_contract_sha256
    assert first.digest_descriptor()["canonicalization"] == CANONICAL_JSON_VERSION


def test_each_layer_changes_only_its_digest_and_the_overall_contract() -> None:
    original = parse_portable_spec(_portable())
    changed_raw = _portable()
    changed_raw["task_core"]["definition"][  # type: ignore[index]
        "operation"
    ] = "gemm"
    changed = parse_portable_spec(changed_raw)

    assert changed.task_core_sha256 != original.task_core_sha256
    assert changed.metrics_sha256 == original.metrics_sha256
    assert changed.evaluation_contract_sha256 != original.evaluation_contract_sha256


def test_task_core_is_a_strict_typed_envelope() -> None:
    value = _portable()
    value["task_core"]["extra"] = True  # type: ignore[index]
    with pytest.raises(PortableSpecError, match="unknown field.*extra"):
        parse_portable_spec(value)

    for field in ("definition", "interface"):
        value = _portable()
        value["task_core"][field] = {}  # type: ignore[index]
        with pytest.raises(PortableSpecError, match=f"{field} must not be empty"):
            parse_portable_spec(value)

    for field in ("allowed_changes", "invariants"):
        value = _portable()
        value["task_core"][field] = []  # type: ignore[index]
        with pytest.raises(PortableSpecError, match="non-empty string array"):
            parse_portable_spec(value)

    value = _portable()
    value["task_core"]["resource_manifest_sha256"] = "ABC"  # type: ignore[index]
    with pytest.raises(PortableSpecError, match="canonical sha256"):
        parse_portable_spec(value)


def test_suite_requires_typed_public_and_final_roles_and_protocols() -> None:
    parsed = parse_portable_spec(_portable())
    suite = parsed.evaluation_suites["standard"]
    assert suite.roles["public"].case_count == 4
    assert suite.roles["anchor"].case_count == 8
    assert suite.oracle.protocol_id == "bf16-reference-v1"

    missing_final = _portable()
    del missing_final["evaluation_suites"]["standard"]["roles"][  # type: ignore[index]
        "final"
    ]
    with pytest.raises(PortableSpecError, match="missing required role.*final"):
        parse_portable_spec(missing_final)

    invalid_count = _portable()
    invalid_count["evaluation_suites"]["standard"]["roles"][  # type: ignore[index]
        "public"
    ][
        "case_count"
    ] = True
    with pytest.raises(PortableSpecError, match="positive integer"):
        parse_portable_spec(invalid_count)

    unknown = _portable()
    unknown["evaluation_suites"]["standard"]["oracle"][  # type: ignore[index]
        "command"
    ] = "python oracle.py"
    with pytest.raises(PortableSpecError, match="unknown field.*command"):
        parse_portable_spec(unknown)


@pytest.mark.parametrize(
    ("policy", "with_baseline"),
    [
        ("none", False),
        ("cached_once_per_campaign", True),
        ("paired_per_candidate", True),
    ],
)
def test_metric_baseline_policy_controls_protocol_digest_block(
    policy: str, with_baseline: bool
) -> None:
    value = _portable()
    metric = value["metrics"]["latency"]  # type: ignore[index]
    metric["baseline_policy"] = policy  # type: ignore[index]
    if with_baseline:
        metric["baseline"] = _protocol(  # type: ignore[index]
            "reference-kernel-v1", "9"
        )

    parsed = parse_portable_spec(value).metrics["latency"]
    assert parsed.baseline_policy == policy
    assert (parsed.baseline is not None) is with_baseline


def test_metric_rejects_baseline_presence_mismatch_and_bad_comparability() -> None:
    unnecessary = _portable()
    unnecessary["metrics"]["latency"]["baseline"] = _protocol(  # type: ignore[index]
        "reference-v1", "8"
    )
    with pytest.raises(PortableSpecError, match="forbidden when policy is none"):
        parse_portable_spec(unnecessary)

    missing = _portable()
    missing["metrics"]["latency"][  # type: ignore[index]
        "baseline_policy"
    ] = "paired_per_candidate"
    with pytest.raises(PortableSpecError, match="baseline is required"):
        parse_portable_spec(missing)

    bad_comparability = _portable()
    bad_comparability["metrics"]["latency"][  # type: ignore[index]
        "comparability"
    ] = "same_gpu_model"
    with pytest.raises(PortableSpecError, match="comparability"):
        parse_portable_spec(bad_comparability)


def test_portable_comparability_rejects_hardware_dependent_units() -> None:
    latency = _portable()
    latency["metrics"]["latency"][  # type: ignore[index]
        "comparability"
    ] = "portable"
    with pytest.raises(PortableSpecError, match="only valid for units"):
        parse_portable_spec(latency)

    correctness = _portable()
    metric = correctness["metrics"]["latency"]  # type: ignore[index]
    metric["comparability"] = "portable"  # type: ignore[index]
    metric["unit"] = "correctness"  # type: ignore[index]
    assert parse_portable_spec(correctness).metrics["latency"].comparability == (
        "portable"
    )


def test_aggregation_is_strict_nonempty_and_rejects_floats() -> None:
    value = _portable()
    value["metrics"]["latency"]["aggregation"]["spec"] = {}  # type: ignore[index]
    with pytest.raises(PortableSpecError, match="spec must not be empty"):
        parse_portable_spec(value)

    value = _portable()
    value["metrics"]["latency"]["aggregation"]["spec"] = {  # type: ignore[index]
        "weight": 0.5
    }
    with pytest.raises(PortableSpecError, match="must not contain floats"):
        parse_portable_spec(value)


def test_scientific_ids_nodes_and_safe_paths_remain_portable() -> None:
    value = _portable()
    definition = value["task_core"]["definition"]  # type: ignore[index]
    definition.update(  # type: ignore[union-attr]
        {
            "model_id": "Qwen/Qwen3-8B",
            "sample_uuid": "sample-123",
            "node": {"node_id": "graph-node-7"},
            "container": "logical-data-container",
            "resource_path": "fixtures/models/config.json",
        }
    )

    parsed = parse_portable_spec(value)

    assert parsed.task_core.definition["sample_uuid"] == "sample-123"
    assert parsed.task_core.definition["node"]["node_id"] == "graph-node-7"


@pytest.mark.parametrize(
    "deployment_key",
    [
        "gpu_uuid",
        "assigned_gpu_uuid",
        "device_uuid",
        "candidate_device_id",
        "accelerator_id",
        "gpu_model",
        "accelerator_model",
        "hardware_model",
        "gpu_name",
        "image_tag",
        "backend",
        "container_id",
        "host_path",
        "mount_path",
        "node_name",
        "node_selector",
        "pci_bus_id",
    ],
)
def test_nested_scientific_data_rejects_deployment_keys(
    deployment_key: str,
) -> None:
    value = _portable()
    value["task_core"]["definition"][deployment_key] = "physical"  # type: ignore[index]
    with pytest.raises(PortableSpecError, match="physical deployment identity"):
        parse_portable_spec(value)


@pytest.mark.parametrize(
    "deployment_value",
    [
        "GPU-deadbeef",
        "MIG-550e8400-e29b-41d4-a716-446655440000",
        "use /dev/nvidia7",
        "/dev/kfd",
        "/dev/dri/card0",
        "/mnt/private/model",
        "docker.io/library/python:3.12",
        "python:3.12",
        "ubuntu:latest",
        "worker-17",
        "10.42.0.8",
        "http://[::1]:8080/evaluate",
        "NVIDIA H100 SXM",
        "H200",
        "GH200",
        "A100-80GB",
        "A6000",
    ],
)
def test_nested_scientific_data_rejects_obvious_deployment_values(
    deployment_value: str,
) -> None:
    value = _portable()
    value["task_core"]["definition"][  # type: ignore[index]
        "placement_hint"
    ] = deployment_value
    with pytest.raises(PortableSpecError, match="physical deployment value"):
        parse_portable_spec(value)


def test_value_lint_does_not_reject_scientific_model_or_sample_ids() -> None:
    value = _portable()
    value["task_core"]["definition"].update(  # type: ignore[index]
        {
            "model_id": "Qwen/Qwen3-8B",
            "sample_uuid": "550e8400-e29b-41d4-a716-446655440000",
        }
    )
    parsed = parse_portable_spec(value)
    assert parsed.task_core.definition["model_id"] == "Qwen/Qwen3-8B"


def test_value_lint_does_not_guess_from_ambiguous_scientific_labels() -> None:
    value = _portable()
    value["task_core"]["definition"].update(  # type: ignore[index]
        {
            "dtype_label": "dtype:bf16",
            "layout_label": "layout:nhwc",
            "ratio_label": "1:2",
            "version_label": "version 1.2.3.4",
            "private_version_label": "10.0.0.1",
            "graph_node_id": "node-7",
        }
    )
    parsed = parse_portable_spec(value)
    assert parsed.task_core.definition["dtype_label"] == "dtype:bf16"


@pytest.mark.parametrize(
    "deployment_value",
    [
        "NVIDIA B200",
        "NVIDIA GB200",
        "AMD MI300X",
        "0000:65:00.0",
        "fe80::1",
        "node17.example.internal",
        "https://node-17.example:8443/api",
        "../private/model",
        r"C:\models\weights",
        "registry.example.com/team/judge@sha256:" + "a" * 64,
    ],
)
def test_value_lint_rejects_additional_unambiguous_deployment_values(
    deployment_value: str,
) -> None:
    value = _portable()
    value["task_core"]["definition"][  # type: ignore[index]
        "placement_hint"
    ] = deployment_value
    with pytest.raises(PortableSpecError, match="physical deployment value"):
        parse_portable_spec(value)


@pytest.mark.parametrize(
    "deployment_fields",
    [
        {"service_endpoint": "https://node-17.example:8443/api"},
        {"endpoint_url": "https://node-17.example:8443/api"},
        {"placement": {"primary": "worker-17"}},
        {"endpoint": {"url": "https://node-17.example:8443/api"}},
    ],
)
def test_deployment_hint_propagates_through_composite_keys_and_objects(
    deployment_fields: dict[str, object],
) -> None:
    value = _portable()
    value["task_core"]["definition"].update(  # type: ignore[index]
        deployment_fields
    )
    with pytest.raises(PortableSpecError, match="physical deployment value"):
        parse_portable_spec(value)


@pytest.mark.parametrize(
    "unsafe_path",
    ["/etc/passwd", "../secret", "fixtures/../secret", "./fixture", "a//b", r"a\b"],
)
def test_generic_path_fields_must_be_safe_bundle_relative_posix(
    unsafe_path: str,
) -> None:
    value = _portable()
    value["task_core"]["interface"]["source_path"] = unsafe_path  # type: ignore[index]
    with pytest.raises(PortableSpecError, match="bundle-relative|components"):
        parse_portable_spec(value)


def test_execution_profile_is_capability_only_and_vendor_is_registered() -> None:
    value = _portable()
    profile = value["execution_profiles"]["single-gpu"]  # type: ignore[index]
    del profile["min_compute_capability"]  # type: ignore[index]
    profile["vendor"] = " ANY "  # type: ignore[index]
    parsed = parse_portable_spec(value)
    assert parsed.execution_profiles["single-gpu"].vendor == "any"
    assert parsed.execution_profiles["single-gpu"].min_compute_capability is None

    bad_vendor = _portable()
    bad_vendor["execution_profiles"]["single-gpu"][  # type: ignore[index]
        "vendor"
    ] = "mystery"
    with pytest.raises(PortableSpecError, match="vendor must be one of"):
        parse_portable_spec(bad_vendor)


@pytest.mark.parametrize(
    "capability", ["09.0", "9", "9.0.0", "sm90", "H100", "9.10", "20.0"]
)
def test_nvidia_compute_capability_must_be_canonical(capability: str) -> None:
    value = _portable()
    value["execution_profiles"]["single-gpu"][  # type: ignore[index]
        "min_compute_capability"
    ] = capability
    with pytest.raises(PortableSpecError, match="capability|physical deployment"):
        parse_portable_spec(value)


@pytest.mark.parametrize("vendor", ["any", "amd", "intel"])
def test_non_nvidia_profiles_forbid_cuda_compute_capability(vendor: str) -> None:
    value = _portable()
    value["execution_profiles"]["single-gpu"]["vendor"] = vendor  # type: ignore[index]
    with pytest.raises(PortableSpecError, match="NVIDIA-specific"):
        parse_portable_spec(value)


def test_profile_requires_content_addressed_protocols() -> None:
    missing_abi = _portable()
    del missing_abi["execution_profiles"]["single-gpu"][  # type: ignore[index]
        "software_abi"
    ]
    with pytest.raises(PortableSpecError, match="missing field.*software_abi"):
        parse_portable_spec(missing_abi)

    missing_calibration = _portable()
    del missing_calibration["execution_profiles"]["single-gpu"][  # type: ignore[index]
        "calibration_protocol"
    ]
    with pytest.raises(
        PortableSpecError, match="missing field.*calibration_protocol"
    ):
        parse_portable_spec(missing_calibration)

    bad_calibration = _portable()
    del bad_calibration["execution_profiles"]["single-gpu"][  # type: ignore[index]
        "calibration_protocol"
    ]["spec_sha256"]
    with pytest.raises(PortableSpecError, match="missing field.*spec_sha256"):
        parse_portable_spec(bad_calibration)

    obsolete_alias = _portable()
    profile_value = obsolete_alias["execution_profiles"][  # type: ignore[index]
        "single-gpu"
    ]
    profile_value["calibration_protocol_id"] = "calibration-v1"  # type: ignore[index]
    with pytest.raises(PortableSpecError, match="unknown field"):
        parse_portable_spec(obsolete_alias)

    parsed = parse_portable_spec(_portable())
    profile = parsed.execution_profiles["single-gpu"]
    assert profile.software_abi.spec_sha256 == _digest("0")
    assert profile.calibration_protocol.spec_sha256 == _digest("1")


def test_features_reject_duplicates_after_normalization() -> None:
    value = _portable()
    value["execution_profiles"]["single-gpu"][  # type: ignore[index]
        "features"
    ] = ["FP8", " fp8 "]
    with pytest.raises(PortableSpecError, match="must not contain duplicates"):
        parse_portable_spec(value)


@pytest.mark.parametrize(
    ("value", "encoded", "digest"),
    [
        (
            {},
            "{}",
            "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
        ),
        (
            {"z": [True, None, "é"], "a": {"n": 3}},
            '{"a":{"n":3},"z":[true,null,"\\u00e9"]}',
            "sha256:d5179e821d9ae535c633e56e0bee943c6a1a202c37a870f422fe98352db630e9",
        ),
    ],
)
def test_canonical_json_v1_golden_vectors(
    value: object, encoded: str, digest: str
) -> None:
    assert canonical_json(value) == encoded
    assert canonical_sha256(value) == digest


@pytest.mark.parametrize("value", [0.0, 1.5, float("nan"), float("inf")])
def test_canonical_json_rejects_all_floats(value: float) -> None:
    with pytest.raises(PortableSpecError, match="must not contain floats"):
        canonical_sha256({"value": value})


def test_canonical_json_rejects_non_ascii_logical_keys() -> None:
    with pytest.raises(PortableSpecError, match="stable ASCII logical keys"):
        canonical_json({"测试": 1})


def test_declared_digest_verifiers_check_bytes_and_regular_files(
    tmp_path: Path,
) -> None:
    payload = b'{"resources":[]}'
    declared = canonical_sha256({"resources": []})
    manifest = tmp_path / "resources.json"
    manifest.write_bytes(payload)

    assert verify_sha256_bytes(payload, declared) == declared
    assert verify_sha256_file(manifest, declared) == declared
    with pytest.raises(PortableSpecError, match="SHA-256 mismatch"):
        verify_sha256_bytes(payload + b"\n", declared)

    symlink = tmp_path / "resources-link.json"
    symlink.symlink_to(manifest)
    with pytest.raises(PortableSpecError, match="cannot read"):
        verify_sha256_file(symlink, declared)


def test_parsed_contract_is_a_deep_snapshot_and_exports_defensive_copies() -> None:
    raw = _portable()
    parsed = parse_portable_spec(raw)
    before_hashes = parsed.layer_digests
    before_payload = parsed.to_dict()

    raw["task_core"]["definition"]["operation"] = "mutated"  # type: ignore[index]
    exported = parsed.to_dict()
    exported["task_core"]["definition"]["operation"] = "also-mutated"
    with pytest.raises(TypeError):
        parsed.task_core.definition[  # type: ignore[index]
            "operation"
        ] = "cannot-mutate"

    assert parsed.layer_digests == before_hashes
    assert parsed.to_dict() == before_payload


def test_parser_rejects_unknown_top_level_and_dangling_defaults() -> None:
    unknown = _portable()
    unknown["extra"] = True
    with pytest.raises(PortableSpecError, match="unknown field.*extra"):
        parse_portable_spec(unknown)

    dangling = _portable()
    dangling["defaults"]["execution_profile_id"] = "gpu-0"  # type: ignore[index]
    with pytest.raises(PortableSpecError, match="references unknown ID"):
        parse_portable_spec(dangling)


def test_absent_portable_field_preserves_legacy_task_and_replay_contract(
    tmp_path: Path,
) -> None:
    task_path = tmp_path / "task.json"
    _write_task(task_path, _legacy_task())
    task = make_task_spec(
        task_path,
        BenchmarkMeta(name="mlsysforge", base_images={"python": {}}),
    )
    assert task.portable is None
    assert "portable" not in candidate_replay._task_contract(task)


def test_task_loader_rejects_portable_and_legacy_direction_mismatch(
    tmp_path: Path,
) -> None:
    task_path = tmp_path / "task.json"
    raw = _legacy_task()
    raw["judge"]["score_direction"] = "maximize"  # type: ignore[index]
    raw["portable"] = _portable()
    _write_task(task_path, raw)

    with pytest.raises(ValueError, match="default metric direction.*does not match"):
        make_task_spec(
            task_path,
            BenchmarkMeta(name="mlsysforge", base_images={"python": {}}),
        )


def test_present_portable_field_is_parsed_and_pinned_for_replay(tmp_path: Path) -> None:
    task_path = tmp_path / "task.json"
    raw = _legacy_task()
    raw["portable"] = _portable()
    _write_task(task_path, raw)
    task = make_task_spec(
        task_path,
        BenchmarkMeta(name="mlsysforge", base_images={"python": {}}),
    )

    assert task.portable is not None
    replay_contract = candidate_replay._task_contract(task)
    before = copy.deepcopy(replay_contract["portable"])
    replay_contract["portable"]["task_core"]["definition"]["operation"] = "changed"
    assert candidate_replay._task_contract(task)["portable"] == before
    descriptor = task.portable.digest_descriptor()
    assert descriptor["canonicalization"] == CANONICAL_JSON_VERSION
    assert descriptor["evaluation_contract_sha256"].startswith("sha256:")


def test_evaluator_descriptor_exposes_portable_layer_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_path = tmp_path / "task.json"
    raw = _legacy_task()
    raw["portable"] = _portable()
    _write_task(task_path, raw)
    task = make_task_spec(
        task_path,
        BenchmarkMeta(name="mlsysforge", base_images={"python": {}}),
    )
    source_identity = _digest("1")
    runtime_identity = _digest("2")
    bytecode_identity = _digest("3")
    monkeypatch.setattr(
        candidate_replay, "_evaluator_source_sha256", lambda: source_identity
    )
    monkeypatch.setattr(candidate_replay, "_loaded_sforge_module_names", lambda: [])
    monkeypatch.setattr(
        candidate_replay,
        "_loaded_evaluator_bytecode_sha256",
        lambda modules: bytecode_identity,
    )

    descriptor = candidate_replay.evaluator_descriptor(
        task,
        SForgeConfig(),
        timeout=None,
        loaded_snapshot={
            "modules": [],
            "runtime_sha256": runtime_identity,
            "bytecode_sha256": bytecode_identity,
        },
    )

    assert task.portable is not None
    assert descriptor["portable_contract"] == task.portable.digest_descriptor()
