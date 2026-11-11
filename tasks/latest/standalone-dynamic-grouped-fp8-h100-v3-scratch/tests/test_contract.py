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
        "judge:verify_grouped.py": TASK_ROOT / "images/judge/verify_grouped.py",
        "worker:grouped_fp8_runtime.py": TASK_ROOT / "runtime/grouped_fp8_runtime.py",
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


def test_v4_identity_empty_starter_and_holdout_lifecycle() -> None:
    task_id = "standalone_dynamic_grouped_fp8_h100_scratch_v3"
    task = json.loads(
        (TASK_ROOT / "tasks" / f"{task_id}.json").read_text(encoding="utf-8")
    )
    # The run configuration is internal and optional (the public tree has none).
    experiment_path = next(
        iter(sorted(TASK_ROOT.glob("experiment-*-ultra-2h.yaml"))), TASK_ROOT / "experiment.yaml"
    )
    experiment = (
        yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
        if experiment_path.is_file()
        else {"tasks": {task_id: {}}}
    )
    assert task["task_id"] == task_id
    assert task["judge"]["private_artifact_path"] == (
        "/logs/verifier/standalone_grouped_fp8_v3_results.json"
    )
    assert set(experiment["tasks"]) == {task_id}
    assert not (
        TASK_ROOT
        / "images/work/workdir/grouped_fp8_lab/candidates/kernel_v001/submission.py"
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
    assert 'FINAL_SUITE_ID = "final-holdout-v4"' in formal
    assert 'EDGE_SUITE_ID = "edge-holdout-v4"' in formal


def test_oracle_archive_selects_final_or_admissible_version(tmp_path) -> None:
    # Default: FINAL_VERSION (the forced DeepGEMM paper oracle); --admissible: the
    # candidate the normal Judge admits, packed with a FINAL_VERSION selecting it.
    import os
    import re
    import subprocess
    import tarfile
    from pathlib import Path

    script = TASK_ROOT / "scripts/make_oracle_archive.sh"
    for args, selector in (([], "FINAL_VERSION"), (["--admissible"], "ADMISSIBLE_VERSION")):
        text = (TASK_ROOT / "oracle" / selector).read_text(encoding="utf-8")
        assert re.fullmatch(r"[1-9][0-9]{0,2}\n?", text), selector
        version = int(text)
        out = tmp_path / selector
        completed = subprocess.run(
            ["bash", str(script), *args],
            cwd=TASK_ROOT,
            env=dict(os.environ, ARTIFACTS_DIR=str(out)),
            check=True,
            capture_output=True,
            text=True,
        )
        archive = Path(completed.stdout.splitlines()[0])
        assert archive == out / "oracle.tar.gz"
        selected = f"candidates/kernel_v{version:03d}/submission.py"
        with tarfile.open(archive, mode="r:gz") as bundle:
            names = set(bundle.getnames())
            assert {"FINAL_VERSION", "notes.md", selected} <= names
            assert not any(name.startswith("paper_reference") for name in names)
            assert int(bundle.extractfile("FINAL_VERSION").read()) == version
            assert bundle.extractfile(selected).read() == (TASK_ROOT / "oracle" / selected).read_bytes()
