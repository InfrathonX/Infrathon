"""The lightweight ``portable`` capability envelope.

Every task in ``tasks/latest`` declares its capability envelope in the task
JSON ``portable`` object.  Tasks that have been promoted to the full portable
contract (``task_core`` + ``evaluation_suites``) must go through the strict
contract parser; everything else is an envelope with no evaluation-contract
identity.  These tests pin that routing and the envelope's own strictness.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sforge.harness import candidate_replay
from sforge.harness.portable_spec import (
    CapabilityEnvelope,
    PortableSpecError,
    is_full_portable_contract,
    parse_capability_envelope,
)
from sforge.harness.task_spec import BenchmarkMeta, make_task_spec


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TASK_JSONS = sorted(REPOSITORY_ROOT.glob("tasks/latest/*/tasks/*.json"))


def _envelope() -> dict:
    return {
        "schema_version": 1,
        "execution_profiles": {
            "single-cuda-sm80-32g": {
                "vendor": "nvidia",
                "count": 1,
                "min_memory_bytes": 34359738368,
                "min_compute_capability": "8.0",
                "features": ["triton", "cuda", "bf16"],
                "notes": (
                    "Enforced by the trusted runtime; the GPU model and UUID "
                    "are provenance only."
                ),
            }
        },
        "metrics": {
            "primary": {
                "comparability": "same_measurement_profile",
                "baseline_policy": "none",
                "direction": "maximize",
            }
        },
        "defaults": {"execution_profile_id": "single-cuda-sm80-32g"},
    }


def _legacy_task(portable: dict | None) -> dict:
    task = {
        "task_id": "envelope_test",
        "name": "Envelope test",
        "base_image": "python",
        "platform": "linux/amd64",
        "cwd": "/app/workdir",
        "submit_paths": ["kernels"],
        "work": {"image_tag": "work-v1"},
        "judge": {
            "eval_cmd": "python run_judge.py",
            "eval_timeout": 60,
            "parser": "json",
            "image_tag": "judge-v1",
            "score_direction": "maximize",
        },
    }
    if portable is not None:
        task["portable"] = portable
    return task


def _load(tmp_path: Path, task: dict):
    path = tmp_path / "task.json"
    path.write_text(json.dumps(task), encoding="utf-8")
    return make_task_spec(
        path, BenchmarkMeta(name="mlsysforge", base_images={"python": {}})
    )


@pytest.mark.parametrize("task_json", TASK_JSONS, ids=[p.parents[2].name for p in TASK_JSONS])
def test_every_latest_task_declares_exactly_one_portable_form(task_json: Path) -> None:
    raw = json.loads(task_json.read_text(encoding="utf-8"))
    assert "portable" in raw, "every task must declare its capability envelope"
    spec = make_task_spec(
        task_json, BenchmarkMeta(name="mlsysforge", base_images={raw["base_image"]: {}})
    )
    if is_full_portable_contract(raw["portable"]):
        assert spec.portable is not None
        assert spec.capability_envelope is None
        assert spec.portable.evaluation_contract_sha256.startswith("sha256:")
    else:
        assert spec.portable is None
        assert isinstance(spec.capability_envelope, CapabilityEnvelope)
        envelope = spec.capability_envelope
        assert envelope.defaults.execution_profile_id in envelope.execution_profiles
        profile = envelope.default_profile
        assert profile.count >= 1
        assert profile.min_memory_bytes > 0
        if profile.vendor == "nvidia":
            assert profile.min_compute_capability is not None


def test_envelope_parses_and_normalizes_features() -> None:
    envelope = parse_capability_envelope(_envelope(), task_id="t")
    profile = envelope.default_profile
    assert profile.features == ("bf16", "cuda", "triton")
    assert profile.min_compute_capability == "8.0"
    assert envelope.default_metric is None
    assert envelope.envelope_sha256.startswith("sha256:")
    assert envelope.to_dict()["execution_profiles"]["single-cuda-sm80-32g"]["features"] == [
        "bf16",
        "cuda",
        "triton",
    ]


def test_envelope_is_a_deep_snapshot() -> None:
    raw = _envelope()
    envelope = parse_capability_envelope(raw)
    before = envelope.to_dict()
    raw["execution_profiles"]["single-cuda-sm80-32g"]["count"] = 2
    exported = envelope.to_dict()
    exported["execution_profiles"]["single-cuda-sm80-32g"]["count"] = 3
    with pytest.raises(TypeError):
        envelope.execution_profiles["extra"] = None  # type: ignore[index]
    assert envelope.to_dict() == before


def test_required_nvlink_pairs_round_trip_and_affect_identity() -> None:
    raw = _envelope()
    profile = next(iter(raw["execution_profiles"].values()))
    profile["count"] = 3
    previous = parse_capability_envelope(raw).envelope_sha256
    profile["required_nvlink_pairs"] = [[1, 0]]
    envelope = parse_capability_envelope(raw)
    assert envelope.default_profile.required_nvlink_pairs == ((0, 1),)
    assert envelope.envelope_sha256 != previous
    exported = envelope.to_dict()
    assert next(iter(exported["execution_profiles"].values()))["required_nvlink_pairs"] == [[0, 1]]
    assert parse_capability_envelope(exported).envelope_sha256 == envelope.envelope_sha256
    profile["required_nvlink_pairs"][0][0] = 2
    assert envelope.default_profile.required_nvlink_pairs == ((0, 1),)


@pytest.mark.parametrize("pairs", [None, "0,1", [[0]], [[0, 1, 2]], [[0, 0]],
                                  [[-1, 0]], [[0, 3]], [[True, 1]], [[0, 1.0]],
                                  [[0, 1], [1, 0]]])
def test_required_nvlink_pairs_reject_invalid_local_indices(pairs) -> None:
    raw = _envelope()
    profile = next(iter(raw["execution_profiles"].values()))
    profile.update(count=3, required_nvlink_pairs=pairs)
    with pytest.raises(PortableSpecError, match="required_nvlink_pairs"):
        parse_capability_envelope(raw)


def test_required_nvlink_pairs_are_nvidia_specific() -> None:
    raw = _envelope()
    profile = next(iter(raw["execution_profiles"].values()))
    profile.update(vendor="amd", count=2, required_nvlink_pairs=[[0, 1]])
    profile.pop("min_compute_capability")
    with pytest.raises(PortableSpecError, match="NVIDIA-specific"):
        parse_capability_envelope(raw)


def test_full_contract_is_never_downgraded_to_an_envelope() -> None:
    half_promoted = _envelope()
    half_promoted["task_core"] = {"contract_id": "x"}
    assert is_full_portable_contract(half_promoted)
    with pytest.raises(PortableSpecError, match="full portable contract"):
        parse_capability_envelope(half_promoted)

    suites_only = _envelope()
    suites_only["evaluation_suites"] = {}
    with pytest.raises(PortableSpecError, match="full portable contract"):
        parse_capability_envelope(suites_only)


def test_task_loader_routes_half_promoted_contract_to_the_strict_parser(
    tmp_path: Path,
) -> None:
    half_promoted = _envelope()
    half_promoted["task_core"] = {"contract_id": "x"}
    with pytest.raises(PortableSpecError, match="missing field"):
        _load(tmp_path, _legacy_task(half_promoted))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda e: e.__setitem__("extra", True), "unknown field.*extra"),
        (lambda e: e.pop("execution_profiles"), "missing field.*execution_profiles"),
        (lambda e: e.__setitem__("schema_version", 2), "schema_version must be 1"),
        (
            lambda e: e["execution_profiles"]["single-cuda-sm80-32g"].__setitem__(
                "gpu_uuid", "GPU-00000000-0000-0000-0000-000000000000"
            ),
            "unknown field.*gpu_uuid",
        ),
        (
            lambda e: e["execution_profiles"]["single-cuda-sm80-32g"].__setitem__(
                "features", ["cuda", "GPU-00000000-0000-0000-0000-000000000000"]
            ),
            "physical deployment value",
        ),
        (
            lambda e: e["execution_profiles"]["single-cuda-sm80-32g"].__setitem__(
                "interconnect", "NVLink between GPU-00000000-0000-0000-0000-000000000000 and its pair"
            ),
            "physical deployment value",
        ),
        (
            lambda e: e["execution_profiles"]["single-cuda-sm80-32g"].__setitem__(
                "features", ["cuda", "H100"]
            ),
            "fixed GPU model",
        ),
        (
            lambda e: e["execution_profiles"]["single-cuda-sm80-32g"].__setitem__("count", 0),
            "positive integer",
        ),
        (
            lambda e: e["execution_profiles"]["single-cuda-sm80-32g"].__setitem__(
                "min_memory_bytes", -1
            ),
            "non-negative integer",
        ),
        (
            lambda e: e["execution_profiles"]["single-cuda-sm80-32g"].pop(
                "min_compute_capability"
            ),
            "required for nvidia",
        ),
        (
            lambda e: e["execution_profiles"]["single-cuda-sm80-32g"].__setitem__(
                "min_compute_capability", "sm_90"
            ),
            "canonical major.minor",
        ),
        (
            lambda e: e["execution_profiles"]["single-cuda-sm80-32g"].__setitem__(
                "features", ["cuda", "cuda"]
            ),
            "duplicates",
        ),
        (
            lambda e: e["execution_profiles"]["single-cuda-sm80-32g"].__setitem__("features", []),
            "non-empty string array",
        ),
        (
            lambda e: e["metrics"]["primary"].__setitem__("direction", "sideways"),
            "maximize or minimize",
        ),
        (
            lambda e: e["metrics"]["primary"].__setitem__("comparability", "same profile"),
            "logical identifier",
        ),
        (
            lambda e: e["metrics"]["primary"].pop("baseline_policy"),
            "missing field.*baseline_policy",
        ),
        (
            lambda e: e["defaults"].__setitem__("execution_profile_id", "gpu-7"),
            "references unknown ID",
        ),
        (
            lambda e: e["defaults"].__setitem__("metric_id", "missing"),
            "references unknown ID",
        ),
        (
            lambda e: e["defaults"].__setitem__("gpu_index", 7),
            "unknown field.*gpu_index",
        ),
    ],
)
def test_envelope_rejects_malformed_and_deployment_values(mutate, message: str) -> None:
    raw = _envelope()
    mutate(raw)
    with pytest.raises(PortableSpecError, match=message):
        parse_capability_envelope(raw)


def test_envelope_notes_are_free_text_but_structured_fields_are_not() -> None:
    raw = _envelope()
    raw["execution_profiles"]["single-cuda-sm80-32g"]["notes"] = (
        "Qualified on H100 NVL; the model name is provenance only."
    )
    raw["metrics"]["primary"]["notes"] = "see configs/h100_release.json"
    envelope = parse_capability_envelope(raw)
    assert "H100" in (envelope.default_profile.notes or "")

    raw["execution_profiles"]["single-cuda-sm80-32g"]["notes"] = "bad\x00note"
    with pytest.raises(PortableSpecError, match="control characters"):
        parse_capability_envelope(raw)


def test_task_loader_checks_envelope_metric_direction_against_judge(
    tmp_path: Path,
) -> None:
    raw = _envelope()
    raw["metrics"]["primary"]["direction"] = "minimize"
    raw["defaults"]["metric_id"] = "primary"
    with pytest.raises(ValueError, match="default metric direction.*does not match"):
        _load(tmp_path, _legacy_task(raw))

    raw["metrics"]["primary"]["direction"] = "maximize"
    spec = _load(tmp_path, _legacy_task(raw))
    assert spec.capability_envelope is not None
    assert spec.capability_envelope.default_metric is not None
    assert spec.capability_envelope.default_metric.direction == "maximize"


def test_envelope_is_pinned_in_the_replay_task_contract(tmp_path: Path) -> None:
    spec = _load(tmp_path, _legacy_task(_envelope()))
    assert spec.portable is None
    contract = candidate_replay._task_contract(spec)
    assert "portable" not in contract
    before = copy.deepcopy(contract["capability_envelope"])
    contract["capability_envelope"]["defaults"]["execution_profile_id"] = "changed"
    assert candidate_replay._task_contract(spec)["capability_envelope"] == before

    without = _load(tmp_path, _legacy_task(None))
    assert without.capability_envelope is None
    assert "capability_envelope" not in candidate_replay._task_contract(without)
