from __future__ import annotations

import hashlib
import json

import yaml

from conftest import TASK_ROOT


def test_mirrored_admission_and_adapter_are_byte_identical() -> None:
    assert (TASK_ROOT / "images/work/benchmark/source_admission.py").read_bytes() == (
        TASK_ROOT / "images/judge/source_admission.py"
    ).read_bytes()
    assert (TASK_ROOT / "judge/run_judge.py").read_bytes() == (
        TASK_ROOT / "images/judge/run_judge.py"
    ).read_bytes()


def test_protected_hash_manifest_matches_image_driving_sources() -> None:
    mapping = {
        "judge:candidate_worker.py": TASK_ROOT / "images/judge/candidate_worker.py",
        "judge:run_judge.py": TASK_ROOT / "images/judge/run_judge.py",
        "judge:source_admission.py": TASK_ROOT / "images/judge/source_admission.py",
        "judge:verify_wgrad.py": TASK_ROOT / "images/judge/verify_wgrad.py",
        "worker:k_grouped_wgrad_runtime.py": TASK_ROOT / "runtime/k_grouped_wgrad_runtime.py",
    }
    observed: dict[str, str] = {}
    for raw in (TASK_ROOT / "images/judge/protected_hashes.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        digest, label = line.split(None, 1)
        assert label not in observed
        observed[label] = digest
    assert set(observed) == set(mapping)
    for label, path in mapping.items():
        assert observed[label] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_v2_identity_empty_starter_and_holdout_lifecycle() -> None:
    task_id = "standalone_k_grouped_fp8_wgrad_h100_scratch_v2"
    task = json.loads(
        (TASK_ROOT / "tasks" / f"{task_id}.json").read_text(encoding="utf-8")
    )
    # The run configuration is internal and optional (the public tree has none).
    experiment_path = next(
        iter(sorted(TASK_ROOT.glob("experiment-*-max-2h.yaml"))), TASK_ROOT / "experiment.yaml"
    )
    assert task["task_id"] == task_id
    assert task["judge"]["private_artifact_path"] == (
        "/logs/verifier/standalone_k_grouped_wgrad_v2_results.json"
    )
    if experiment_path.is_file():
        experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
        assert set(experiment["tasks"]) == {task_id}
        assert isinstance(experiment["defaults"]["model"], str) and experiment["defaults"]["model"]
        assert experiment["env"]["SFORGE_CODEX_REASONING_EFFORT"] == "max"
    assert not (
        TASK_ROOT
        / "images/work/workdir/k_grouped_wgrad_lab/candidates/kernel_v001/submission.py"
    ).read_bytes()

    build = (TASK_ROOT / "scripts/build_images.sh").read_text(encoding="utf-8")
    work_docker = (TASK_ROOT / "images/work/Dockerfile").read_text(encoding="utf-8")
    judge_docker = (TASK_ROOT / "images/judge/Dockerfile").read_text(encoding="utf-8")
    formal = (TASK_ROOT / "scripts/validate_formal_artifacts.py").read_text(
        encoding="utf-8"
    )
    assert build.count('--build-context "final_cases=${final_cases_context}"') == 1
    assert "COPY --from=final_cases" not in work_docker
    holdout_layer = (TASK_ROOT / "images/judge/Dockerfile.private").read_text(encoding="utf-8")
    assert "final_cases" not in judge_docker
    assert "FROM ${JUDGE_PUBLIC_IMAGE}" in holdout_layer
    assert "COPY --from=final_cases final_cases.json edge_cases.json" in holdout_layer
    assert "compute-sanitizer" in judge_docker
    assert 'FINAL_SUITE_ID = "final-holdout-v3"' in formal
    assert 'EDGE_SUITE_ID = "edge-holdout-v3"' in formal
