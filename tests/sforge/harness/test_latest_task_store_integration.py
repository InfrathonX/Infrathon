"""Version-store integration over the real latest task specs.

Every registered latest task must complete one full archive lifecycle with
its actual submit contract: checkpoint save, notes-only dedup (where
``candidate_identity_exclude`` is declared), new candidate on a code edit,
final-event run closure, and integrity verification.
"""

from __future__ import annotations

import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sforge.harness.benchmark import load_benchmark
from sforge.harness.task_spec import load_all_tasks
from sforge.harness.version_store import VersionStore, VersionStoreCollision

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LATEST = REPOSITORY_ROOT / "tasks/latest"

# Keep in sync with the registrations in the persistent version store
# repository (agent_task_version_submit/tasks/*/task.json).
EXPECTED_REGISTRATIONS = {
    "concurrent-agent-serving-optimizer-qwen36-swe192-v24": (
        "concurrent_agent_serving_optimizer_qwen36_swe192_v24"
    ),
    "eagle3-spec-decode-engine-qwen3-8b-h100-v1-seed": (
        "eagle3_spec_decode_engine_qwen3_8b_h100_v1"
    ),
    "flash-kmeans-paper-h100-v7-scratch": (
        "flash_kmeans_paper_h100_kernel_lab_v7_scratch"
    ),
    "flux-moe-ag-scatter-grouped-gemm-dual-h100-v1-seed": (
        "flux_moe_ag_scatter_grouped_gemm_dual_h100_seed_v1"
    ),
    "llada-h100-decoder-optimization-v2-vanilla-seed": (
        "llada_h100_decoder_optimization_v2"
    ),
    "lmcache-paged-kv-block-transfer-h100-v1-seed": (
        "lmcache_paged_kv_block_transfer_h100_seed_v1"
    ),
    "sageattention-quantized-attention-bf16-h100-v2-seed": (
        "sageattention_quantized_attention_bf16_h100_seed_v2"
    ),
    "minference-sparse-prefill-h100-v3-seed": (
        "minference_sparse_prefill_h100_seed_v3"
    ),
    "quant-videogen-cross-model-kv-codec-h100-v4-scratch": (
        "quant_videogen_cross_model_kv_codec_h100_scratch_v4"
    ),
    "qwen3-1p7b-orpo-training-optimizer-dual-h100-v4-cross-matrix-seed": (
        "qwen3_1p7b_orpo_training_optimizer_dual_h100_v4"
    ),
    "qwen3-4b-to-1p7b-gkd-distillation-optimizer-dual-h100-v1-official-trl-seed": (
        "qwen3_4b_to_1p7b_gkd_distillation_optimizer_dual_h100_v1"
    ),
    "standalone-dynamic-grouped-fp8-h100-v3-scratch": (
        "standalone_dynamic_grouped_fp8_h100_scratch_v3"
    ),
    "standalone-paged-gqa-decode-bf16-h100-v2-scratch": (
        "standalone_paged_gqa_decode_bf16_h100_scratch_v2"
    ),
    "standalone-sparse-mla-prefill-h100-v2-scratch": (
        "standalone_sparse_mla_prefill_h100_scratch_v2"
    ),
    "tilelang-mamba2-ssd-forward-h100-v2-seed": (
        "tilelang_mamba2_ssd_forward_h100_seed_v2"
    ),
    "standalone-k-grouped-fp8-wgrad-h100-v2-scratch": (
        "standalone_k_grouped_fp8_wgrad_h100_scratch_v2"
    ),
    "streamdiffusionv2-dual-h100-pipeline-runtime-v2-seed": (
        "streamdiffusionv2_dual_h100_pipeline_runtime_seed_v2"
    ),
    "kvzip-query-agnostic-kv-scoring-qwen3-8b-h100-v2-scratch": (
        "kvzip_query_agnostic_kv_scoring_qwen3_8b_h100_scratch_v2"
    ),
    "kvpress-kv-eviction-policy-qwen3-8b-h100-v2-scratch": (
        "kvpress_kv_eviction_policy_qwen3_8b_h100_scratch_v2"
    ),
    "attn-qat-nvfp4-training-attention-h100-v1-seed": (
        "attn_qat_nvfp4_training_attention_h100_seed_v1"
    ),
}
# A public checkout holds only the released tasks (scripts/infrathon/tasks.json).
PRESENT = {
    directory: task_id
    for directory, task_id in EXPECTED_REGISTRATIONS.items()
    if (LATEST / directory).is_dir()
}


# Tasks that declare version enumeration and whose Judge honors
# candidate_version, but that ship no scripts/replay_candidates.sh yet:
# per-version replay of their saved candidates is not wired.
NO_REPLAY_SCRIPT = {"minference-sparse-prefill-h100-v3-seed"}


def _load_spec(task_directory: str):
    tasks_dir = LATEST / task_directory / "tasks"
    benchmark = load_benchmark(tasks_dir)
    specs = load_all_tasks(tasks_dir, benchmark)
    assert len(specs) == 1, task_directory
    return specs[0]


def _archive(members: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as bundle:
        for name in sorted(members):
            info = tarfile.TarInfo(name)
            info.size = len(members[name])
            info.mode = 0o644
            bundle.addfile(info, io.BytesIO(members[name]))
    return output.getvalue()


def _sample_members(spec) -> tuple[dict[str, bytes], bool]:
    """Return a minimal conforming submitted tree and whether it has notes."""

    if spec.submit_paths == ["submission"]:
        return {"submission/main.py": b"print('v1')\n"}, False
    code_root = spec.submit_paths[0]
    assert set(spec.submit_paths) == {code_root, "FINAL_VERSION", "notes.md"}, (
        spec.task_id
    )
    return {
        f"{code_root}/objective_v001/candidate.py": b"print('v1')\n",
        "FINAL_VERSION": b"1\n",
        "notes.md": b"hypothesis one\n",
    }, True


def _store_with(tmp_path: Path, task_directory: str, task_id: str) -> VersionStore:
    root = tmp_path / "store"
    root.mkdir()
    (root / ".sforge-version-store.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "sforge-version-store",
                "store_id": "latest-integration-store",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    task_root = root / "tasks" / task_directory
    (task_root / "runs").mkdir(parents=True)
    (task_root / "task.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task_directory": task_directory,
                "task_id": task_id,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return VersionStore.open(root)


def test_registration_table_matches_latest_tasks() -> None:
    on_disk = {
        item.name
        for item in LATEST.iterdir()
        if item.is_dir() and item.name != "__pycache__"
        # The Miles family registers through its own launchers (docs/MILES_RETROSPECTIVE.md).
        and not item.name.startswith("miles-")
    }
    assert on_disk == set(PRESENT)
    released = json.loads((REPOSITORY_ROOT / "scripts/infrathon/tasks.json").read_text())["tasks"]
    for directory in set(EXPECTED_REGISTRATIONS) - set(PRESENT):
        assert not any(
            status != "unsupported"
            for status in released.get(directory, {}).get("hardware", {}).values()
        ), f"released task {directory} is missing"


@pytest.mark.parametrize(
    "task_directory,task_id", sorted(PRESENT.items())
)
def test_latest_task_full_archive_lifecycle(
    tmp_path: Path, task_directory: str, task_id: str
) -> None:
    spec = _load_spec(task_directory)
    assert spec.task_id == task_id

    members, has_notes = _sample_members(spec)
    if has_notes:
        assert spec.candidate_identity_exclude == ["notes.md"], task_id
    else:
        assert spec.candidate_identity_exclude == [], task_id

    store = _store_with(tmp_path, task_directory, task_id)
    common = {
        "task_id": task_id,
        "run_id": "integration-run",
        "submit_paths": spec.submit_paths,
        "submit_exclude": spec.submit_exclude,
        "content_identity_exclude": spec.candidate_identity_exclude,
    }
    base = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)

    first = store.record_archive(
        source="checkpoint",
        round_id="checkpoint-1",
        label="baseline",
        archive=_archive(members),
        created_at=base,
        **common,
    )

    expected_unique = 2
    if has_notes:
        notes_edit = dict(members, **{"notes.md": b"hypothesis two\n"})
        notes_only = store.record_archive(
            source="checkpoint",
            round_id="checkpoint-2",
            label="notes-only",
            archive=_archive(notes_edit),
            created_at=base.replace(minute=5),
            **common,
        )
        assert notes_only.candidate_id == first.candidate_id
        assert notes_only.candidate_sequence == first.candidate_sequence == 1

    code_key = next(name for name in members if name.endswith(".py"))
    code_edit = dict(members, **{code_key: b"print('v2')\n"})
    changed = store.record_archive(
        source="checkpoint",
        round_id="checkpoint-3",
        label="code-change",
        archive=_archive(code_edit),
        created_at=base.replace(minute=10),
        **common,
    )
    assert changed.candidate_id != first.candidate_id
    assert changed.candidate_sequence == 2

    final = store.record_archive(
        source="final",
        round_id="final",
        label=None,
        archive=_archive(code_edit),
        created_at=base.replace(minute=15),
        **common,
    )
    assert final.candidate_id == changed.candidate_id

    with pytest.raises(VersionStoreCollision, match="closed"):
        store.record_archive(
            source="checkpoint",
            round_id="checkpoint-4",
            label="after-final",
            archive=_archive(members),
            created_at=base.replace(minute=20),
            **common,
        )

    reopened = VersionStore.open(store.root)
    assert reopened.verify() == []
    candidates = reopened.list_candidates(task=task_id, run_id="integration-run")
    assert [candidate.sequence for candidate in candidates] == list(
        range(1, expected_unique + 1)
    )
    assert len(candidates[0].events) == (2 if has_notes else 1)
    assert len(candidates[1].events) == 2


def test_judge_server_checkpoint_api_with_real_orpo_spec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One representative task exercised through the real serve HTTP path."""

    from fastapi.testclient import TestClient

    from sforge.harness import judge_server
    from sforge.harness.config import SForgeConfig
    from sforge.harness.judge_server import JudgeState

    task_directory = (
        "qwen3-1p7b-orpo-training-optimizer-dual-h100-v4-cross-matrix-seed"
    )
    task_id = EXPECTED_REGISTRATIONS[task_directory]
    _store_with(tmp_path, task_directory, task_id)

    monkeypatch.setattr(
        judge_server, "create_backend_from_config", lambda config: object()
    )
    monkeypatch.setattr(JudgeState, "_reaper_loop", lambda self: None)
    monkeypatch.setattr(judge_server, "CHECKPOINT_MIN_INTERVAL_SECONDS", 0.0)
    app = judge_server.create_app(
        SForgeConfig(
            log_dir=tmp_path / "logs",
            tasks_dir=LATEST / task_directory / "tasks",
            version_store_root=tmp_path / "store",
        )
    )
    client = TestClient(app)

    registered = client.post(
        "/api/v1/register",
        json={
            "task_id": task_id,
            "run_id": "serve-integration",
            "admin_secret": judge_server.ADMIN_SECRET,
            "require_version_saving": True,
            "required_version_store_id": "latest-integration-store",
        },
    )
    assert registered.status_code == 200, registered.text
    assert registered.json()["version_saving_enabled"] is True
    token = registered.json()["token"]

    members = {
        "solutions/objective_v001/candidate.py": b"print('v1')\n",
        "FINAL_VERSION": b"1\n",
        "notes.md": b"hypothesis one\n",
    }

    def save(label: str, payload: dict[str, bytes]):
        return client.post(
            "/api/v1/versions",
            content=_archive(payload),
            headers={
                "Content-Type": "application/gzip",
                "X-SForge-Token": token,
                "X-SForge-Version-Label": label,
            },
        )

    first = save("baseline", members)
    assert first.status_code == 201, first.text
    notes_only = save(
        "notes-only", dict(members, **{"notes.md": b"hypothesis two\n"})
    )
    assert notes_only.status_code == 201, notes_only.text
    assert notes_only.json()["candidate_id"] == first.json()["candidate_id"]

    outside_contract = save(
        "outside", dict(members, **{"runtime/hack.py": b"x\n"})
    )
    assert outside_contract.status_code == 400
    assert "submit contract" in outside_contract.json()["detail"]

    store = VersionStore.open(tmp_path / "store")
    assert store.verify() == []
    candidates = store.list_candidates(task=task_id, run_id="serve-integration")
    assert len(candidates) == 1
    assert len(candidates[0].events) == 2


def test_per_version_enumeration_on_the_standalone_contract(
    tmp_path: Path,
) -> None:
    """One entry per distinct (version, content) pair, chronological order."""

    from sforge.harness.candidate_replay import (
        CandidateReplayError,
        enumerate_version_implementations,
    )

    task_directory = "standalone-dynamic-grouped-fp8-h100-v3-scratch"
    task_id = EXPECTED_REGISTRATIONS[task_directory]
    spec = _load_spec(task_directory)
    assert spec.version_enumeration == {
        "root": "candidates",
        "directory_pattern": r"kernel_v([0-9]{3})",
        "selector_file": "FINAL_VERSION",
    }
    store = _store_with(tmp_path, task_directory, task_id)
    common = {
        "task_id": task_id,
        "run_id": "enum-run",
        "submit_paths": spec.submit_paths,
        "submit_exclude": spec.submit_exclude,
        "content_identity_exclude": spec.candidate_identity_exclude,
        "source": "checkpoint",
        "label": None,
    }
    base = datetime(2026, 8, 21, 15, 0, tzinfo=timezone.utc)
    v1 = {"candidates/kernel_v001/submission.py": b"impl one\n",
          "FINAL_VERSION": b"1\n", "notes.md": b"n\n"}
    # Adds an unselected variant: same selector, new directory.
    v2 = dict(v1, **{"candidates/kernel_v002/submission.py": b"impl two\n"})
    # Edits kernel_v001 in place without bumping the version number.
    v3 = dict(v2, **{"candidates/kernel_v001/submission.py": b"impl one B\n"})
    # A broken selector cannot serve as any implementation's source event.
    v4 = dict(v3, **{"FINAL_VERSION": b"007\n",
                     "candidates/kernel_v003/submission.py": b"impl three\n"})
    for index, members in enumerate([v1, v2, v3, v4], start=1):
        store.record_archive(
            round_id=f"checkpoint-{index}",
            archive=_archive(members),
            created_at=base.replace(minute=index),
            **common,
        )

    with pytest.raises(CandidateReplayError, match="version_enumeration"):
        enumerate_version_implementations(
            store=store,
            task_spec=type("Spec", (), {
                "task_id": task_id, "version_enumeration": None
            })(),
            run_id="enum-run",
        )

    implementations = enumerate_version_implementations(
        store=store, task_spec=spec, run_id="enum-run"
    )
    assert [
        (item["sequence"], item["version_number"], item["first_event_round"])
        for item in implementations
    ] == [
        (1, 1, "checkpoint-1"),   # impl one
        (2, 2, "checkpoint-2"),   # impl two
        (3, 1, "checkpoint-3"),   # in-place edit of kernel_v001
    ]
    # kernel_v003 only ever appeared alongside the broken selector, so it has
    # no admissible source event and is deliberately absent.
    assert all(item["version_number"] != 3 for item in implementations)


def test_per_version_enumeration_on_the_flash_kmeans_file_contract(
    tmp_path: Path,
) -> None:
    """Versions kept as one file each enumerate exactly like directories.

    flash-kmeans retains kernels/flash_kmeans_vNNN.py rather than a directory
    per version, and its public harness even offers --candidate-version, so
    unselected versions are the norm there rather than the exception.
    """

    from sforge.harness.candidate_replay import enumerate_version_implementations

    task_directory = "flash-kmeans-paper-h100-v7-scratch"
    task_id = EXPECTED_REGISTRATIONS[task_directory]
    spec = _load_spec(task_directory)
    assert spec.version_enumeration == {
        "root": "kernels",
        "file_pattern": r"flash_kmeans_v([0-9]{3})\.py",
        "selector_file": "FINAL_VERSION",
    }
    store = _store_with(tmp_path, task_directory, task_id)
    common = {
        "task_id": task_id,
        "run_id": "enum-run",
        "submit_paths": spec.submit_paths,
        "submit_exclude": spec.submit_exclude,
        "content_identity_exclude": spec.candidate_identity_exclude,
        "source": "checkpoint",
        "label": None,
    }
    base = datetime(2026, 8, 21, 21, 0, tzinfo=timezone.utc)
    v1 = {"kernels/flash_kmeans_v001.py": b"impl one\n",
          "FINAL_VERSION": b"1\n", "notes.md": b"n\n"}
    # v002 is tested publicly with --candidate-version but never selected.
    v2 = dict(v1, **{"kernels/flash_kmeans_v002.py": b"impl two\n"})
    # Repairing v002 in place keeps the number but is a different program.
    v3 = dict(v2, **{"kernels/flash_kmeans_v002.py": b"impl two B\n"})
    # A file under the root that is not a version must be ignored entirely.
    v4 = dict(v3, **{"kernels/README.md": b"scratch\n", "FINAL_VERSION": b"2\n"})
    for index, members in enumerate([v1, v2, v3, v4], start=1):
        store.record_archive(
            round_id=f"checkpoint-{index}",
            archive=_archive(members),
            created_at=base.replace(minute=index),
            **common,
        )

    implementations = enumerate_version_implementations(
        store=store, task_spec=spec, run_id="enum-run"
    )
    assert [
        (item["sequence"], item["version_number"], item["first_event_round"])
        for item in implementations
    ] == [
        (1, 1, "checkpoint-1"),   # impl one
        (2, 2, "checkpoint-2"),   # impl two, never selected by FINAL_VERSION
        (3, 2, "checkpoint-3"),   # in-place repair of v002
    ]
    # One file is one implementation, and kernels/README.md is not a version.
    assert {item["file_count"] for item in implementations} == {1}


@pytest.mark.parametrize("task_directory", sorted(PRESENT))
def test_declared_version_enumeration_is_wired_end_to_end(
    task_directory: str,
) -> None:
    """A task that declares per-version replay must be able to run it.

    The declaration alone is inert: the curve is only reachable through the
    task's own replay script, and the Judge only scores an unselected version
    if it honors the host's candidate_version. Half of that wiring silently
    produces a task that looks configured and is not.
    """

    import re

    spec = _load_spec(task_directory)
    enumeration = spec.version_enumeration
    if enumeration is None:
        return

    assert set(enumeration) <= {
        "root", "directory_pattern", "file_pattern", "selector_file"
    }, enumeration
    patterns = [
        enumeration.get(key)
        for key in ("directory_pattern", "file_pattern")
        if enumeration.get(key)
    ]
    assert len(patterns) == 1, "declare exactly one implementation pattern"
    assert re.compile(patterns[0]).groups == 1, patterns[0]
    assert enumeration["root"] in spec.submit_paths, enumeration["root"]
    assert enumeration["selector_file"] in spec.submit_paths, enumeration

    task_root = LATEST / task_directory
    if task_directory not in NO_REPLAY_SCRIPT:
        replay = (task_root / "scripts/replay_candidates.sh").read_text(
            encoding="utf-8"
        )
        assert "--per-version" in replay, task_directory
        assert "replay-versions" in replay, task_directory
    else:
        assert not (task_root / "scripts/replay_candidates.sh").exists(), (
            f"{task_directory} has a replay script now; drop it from "
            "NO_REPLAY_SCRIPT"
        )

    judges = sorted((task_root / "images/judge").glob("verify_*.py"))
    assert judges, task_directory
    assert any(
        "candidate_version" in judge.read_text(encoding="utf-8")
        for judge in judges
    ), f"{task_directory}: Judge ignores the host candidate_version request"
