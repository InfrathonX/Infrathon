from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import sys

import pytest

from conftest import private_data_dir


def test_reference_is_calibrated_with_real_provenance(task_root, monkeypatch):
    """Pending references cannot pass formal selection; calibrated ones have
    complete measurement provenance and match their device-profile cache.
    """
    reference = json.loads(
        (private_data_dir() / "reference_quality.json").read_text()
    )
    assert reference["schema_version"] == 4
    assert reference["task_id"] == "llada_h100_decoder_optimization_v2"
    historical = reference.get("historical_a6000_reference")
    if historical is not None:
        assert historical["applicability"] == "NONE"
    if reference["status"] == "PENDING_STRATIFIED520_CALIBRATION":
        assert reference["calibrated"] is False
        assert reference["rows"] == {}
        assert reference["case_records"] == {}
        assert reference["calibration_provenance"] is None
        for key in ("measurement_profile", "measurement_profile_key", "device_profile_key"):
            assert key not in reference
        spec = importlib.util.spec_from_file_location(
            "pending_reference_selector", task_root / "scripts/select_reference_profile.py"
        )
        assert spec is not None and spec.loader is not None
        selector = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(selector)
        monkeypatch.setattr(selector, "query", lambda index: {
            "gpu_name": "NVIDIA H100 NVL",
            "compute_capability": "9.0",
            "driver_version": "test-driver",
        })
        with pytest.raises(SystemExit, match="built reference does not match device profile"):
            selector.main([
                "--judge-gpu", "0", "--check",
                "--private-data", str(private_data_dir()),
            ])
        return
    assert reference["calibrated"] is True
    assert reference["status"] == "CALIBRATED_STRATIFIED520"
    assert len(reference["rows"]) == 12
    assert len(reference["case_records"]) == 520
    provenance = reference["calibration_provenance"]
    assert isinstance(provenance, dict) and provenance
    for key in ("measurement_profile", "measurement_profile_key", "device_profile_key"):
        assert isinstance(reference[key], (dict, str)) and reference[key]
    profile = reference["measurement_profile"]
    for key in ("gpu_name", "compute_capability", "driver_version", "work_image_id", "judge_image_id"):
        assert profile[key]
    assert "H100" in profile["gpu_name"]
    cache = private_data_dir() / "reference_cache" / (
        _reference_profile_key(profile) + ".json"
    )
    assert cache.is_file()
    assert cache.read_bytes() == (
        private_data_dir() / "reference_quality.json"
    ).read_bytes()



def test_cached_references_are_keyed_by_their_own_device_profile(task_root):
    """Every cached reference is filed under the GPU model it was measured on.

    The recorded device/measurement keys (which include the driver and the
    image ids) stay self-consistent provenance; the file name is the reference
    profile key (GPU model + compute capability) the Judge selects by.
    """
    cache = private_data_dir() / "reference_cache"
    entries = sorted(cache.glob("*.json"))
    assert entries, "the profile cache must retain the produced references"
    for path in entries:
        reference = json.loads(path.read_text())
        profile = reference["measurement_profile"]
        assert sorted(profile) == [
            "compute_capability",
            "driver_version",
            "gpu_name",
            "judge_image_id",
            "work_image_id",
        ]
        assert all(isinstance(value, str) and value for value in profile.values())
        device = {
            key: profile[key]
            for key in ("compute_capability", "driver_version", "gpu_name")
        }
        assert reference["measurement_profile_key"] == _profile_key(profile)
        assert reference["device_profile_key"] == _profile_key(device)
        assert path.stem == _reference_profile_key(profile)
        assert reference["status"] == "CALIBRATED_STRATIFIED520"


@pytest.mark.parametrize("strict", [False, True])
def test_historical_reference_drift_is_reported_without_rewriting_evidence(
    task_root, monkeypatch, capsys, strict
):
    """Historical calibration remains evidence, not proof of the current policy."""
    spec = importlib.util.spec_from_file_location(
        "reference_drift_layout", task_root / "scripts/validate_layout.py"
    )
    assert spec is not None and spec.loader is not None
    layout = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(layout)
    monkeypatch.delenv("LLADA_PILOT", raising=False)
    if strict:
        monkeypatch.setenv("LLADA_STRICT_REFERENCE", "1")
    else:
        monkeypatch.delenv("LLADA_STRICT_REFERENCE", raising=False)
    reference_path = private_data_dir() / "reference_quality.json"
    # The 2026-08-16 reference retired from the cache is the historical case.
    history = sorted((task_root / "docs/reference_history").glob("86b7*.json"))
    if not history:
        pytest.skip("docs/reference_history/ (calibration records) is not in this checkout")
    cached_path = history[0]
    preserved_bytes = cached_path.read_bytes()
    historical = json.loads(preserved_bytes)
    load_json = layout.load_json

    def load_historical_reference(path, errors, expected=dict):
        if path == reference_path:
            return historical
        return load_json(path, errors, expected)

    monkeypatch.setattr(layout, "load_json", load_historical_reference)
    hidden_bytes = gzip.decompress(
        reference_path.parent.joinpath("hidden_cases.json.gz").read_bytes()
    )
    errors = []
    layout.validate_reference(task_root, hidden_bytes, errors)
    message = "calibrated reference producer files/policies drifted"
    if strict:
        assert message in errors
    else:
        assert message not in errors
        warnings = capsys.readouterr().err
        assert message in warnings
        assert "stale-calibration signal" in warnings
    assert cached_path.read_bytes() == preserved_bytes


def _profile_key(profile: dict[str, str]) -> str:
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _reference_profile_key(profile: dict[str, str]) -> str:
    return _profile_key(
        {key: profile[key] for key in ("compute_capability", "gpu_name")}
    )


def _selector(task_root):
    spec = importlib.util.spec_from_file_location(
        "reference_selector_under_test", task_root / "scripts/select_reference_profile.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "change, accepted",
    [
        ({"driver_version": "615.71.09"}, True),
        ({"driver_version": "580.126.09"}, True),
        ({"gpu_name": "other-model", "driver_version": "595.71.05"}, "cached"),
        ({"gpu_name": "NVIDIA B200", "driver_version": "595.71.05"}, False),
        ({"compute_capability": "8.0", "driver_version": "595.71.05"}, False),
    ],
)
def test_driver_change_keeps_the_reference_but_gpu_model_change_refuses(
    task_root, monkeypatch, capsys, change, accepted
):
    """A host driver update keeps the frozen reference; GPU model / CC fail closed.

    The device is derived from the shipped reference, so the check holds for
    whichever H100 model the Judge image was built for (NVL or SXM 80GB).  The
    other H100 model has its own cache entry, which the Judge selects on-device.
    """
    reference = json.loads(
        (private_data_dir() / "reference_quality.json").read_text()
    )
    if reference["calibrated"] is not True:
        pytest.skip("no calibrated reference shipped")
    shipped = reference["measurement_profile"]
    device = {key: shipped[key] for key in ("gpu_name", "compute_capability")}
    device.update(change)
    if device["gpu_name"] == "other-model":
        device["gpu_name"] = (
            "NVIDIA H100 NVL" if shipped["gpu_name"] != "NVIDIA H100 NVL"
            else "NVIDIA H100 80GB HBM3"
        )
    selector = _selector(task_root)
    monkeypatch.setattr(selector, "query", lambda index: dict(device))
    argv = [
        "--judge-gpu", "0", "--check",
        "--private-data", str(private_data_dir()),
    ]
    if accepted == "cached":
        assert selector.main(argv) == 0
        assert "the Judge selects it on-device" in capsys.readouterr().out
    elif accepted:
        assert selector.main(argv) == 0
        assert "provenance only" in capsys.readouterr().out
    else:
        with pytest.raises(SystemExit, match="built reference does not match device profile"):
            selector.main(argv)


def test_verifier_lineage_names_only_the_committed_r1_verifier(task_root):
    """The scoring-equivalent predecessor is exactly the r1 verify_decoder.py."""
    verifier = (task_root / "images/judge/verifier/verify_decoder.py").read_text()
    assert "SCORING_EQUIVALENT_VERIFY_DECODER_SHA256 = frozenset(" in verifier
    assert verifier.count("6a8bd4f02121e0b1e3aeec44a54036de834c86de03d02def2140207b2a3f3226") == 1
    # The shipped reference was calibrated under that predecessor or under the
    # current verifier (a later device-profile producer run), and every other
    # invariant-core file still matches it byte-for-byte.
    reference = json.loads(
        (private_data_dir() / "reference_quality.json").read_text()
    )
    if reference["calibrated"] is not True:
        pytest.skip("no calibrated reference shipped")
    core = reference["calibration_provenance"]["judge_invariant_core"]
    verifier_dir = task_root / "images/judge/verifier"
    assert core["verify_decoder_sha256"] in {
        "6a8bd4f02121e0b1e3aeec44a54036de834c86de03d02def2140207b2a3f3226",
        hashlib.sha256((verifier_dir / "verify_decoder.py").read_bytes()).hexdigest(),
    }
    for field, name in (
        ("candidate_worker_sha256", "candidate_worker.py"),
        ("cohort_sha256", "cohort.py"),
        ("quality_sha256", "quality.py"),
        ("source_admission_sha256", "source_admission.py"),
    ):
        assert core[field] == hashlib.sha256((verifier_dir / name).read_bytes()).hexdigest()
    # The verifier pins the software ABI the reference was measured under.
    assert '"torch": "2.7.1+cu128"' in verifier
    assert '"transformers": "4.49.0"' in verifier
    assert '"cuda_runtime": "12.8"' in verifier
    assert "software ABI" in verifier


def test_judge_selects_the_reference_by_on_device_profile(task_root):
    """The Judge resolves and binds the reference from the live device."""
    verifier = (task_root / "images/judge/verifier/verify_decoder.py").read_text()
    assert "REFERENCE_CACHE_DIR" in verifier
    assert "observed_reference_profile_key" in verifier
    assert "measurement_profile_key" in verifier
    assert "quality reference was produced under a different measurement" in verifier
    # The producer and the launcher-side selection share the same key function.
    selector = (task_root / "scripts/select_reference_profile.py").read_text()
    calibrator = (task_root / "scripts/calibrate_reference.py").read_text()
    for text in (verifier, selector, calibrator):
        assert 'sort_keys=True, separators=(",", ":")' in text


def test_judge_and_calibrator_require_identity_bound_provenance(task_root):
    verifier = (task_root / "images/judge/verifier/verify_decoder.py").read_text()
    lifecycle = (task_root / "images/judge/verifier/row_lifecycle.py").read_text()
    calibrator = (task_root / "scripts/calibrate_reference.py").read_text()
    for literal in (
        "ARCHIVE_SHA256",
        "WORK_IMAGE_ID",
        "JUDGE_IMAGE_ID",
        "MODEL_VALIDATION_SHA256",
        "RUNTIME_ENTRY_SHA256",
        "SOURCE_DIGEST",
    ):
        assert literal in verifier
    for field in (
        "archive_sha256",
        "work_image_id",
        "judge_image_id",
        "model_validation_sha256",
        "runtime_entry_sha256",
        "source_digest",
        "workload_policy_sha256",
        "memory_policy_sha256",
        "hidden_manifest_sha256",
        "row_order_sha256",
    ):
        assert field in lifecycle
    assert "canonical_candidate_archive_sha256" in calibrator
    assert "evaluation_validator_sha256" in calibrator
    assert "calibration_tool_sha256" in calibrator


def test_postchecks_compare_against_the_reference_the_judge_selected(task_root, tmp_path):
    """The host postchecks resolve the reference the Judge used: the cache entry
    for the profile it observed, else the installed single-profile file."""
    sys.path.insert(0, str(task_root / "scripts"))
    import validate_eval_artifacts as validator

    private_data = tmp_path / "private_data"
    (private_data / "reference_cache").mkdir(parents=True)
    installed = private_data / "reference_quality.json"
    installed.write_text("{}")
    key = "e" * 64
    cached = private_data / "reference_cache" / f"{key}.json"
    cached.write_text('{"cached": true}')

    selected = {"reference_profile_key": key, "reference_source": f"{key}.json"}
    assert validator.selected_reference_path(private_data, selected) == cached
    fallback = {"reference_profile_key": "a" * 64, "reference_source": "reference_quality.json"}
    assert validator.selected_reference_path(private_data, fallback) == installed
    for report in (
        {"reference_profile_key": key, "reference_source": "reference_quality.json"},
        {"reference_profile_key": "../" + key, "reference_source": f"{key}.json"},
    ):
        with pytest.raises(ValueError, match="source mismatch"):
            validator.selected_reference_path(private_data, report)
    formal = (task_root / "scripts/validate_formal_artifacts.py").read_text()
    assert "selected_reference_path(private_data_dir(task_source), private)" in formal
