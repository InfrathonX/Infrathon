"""Runtime identity and API-bound tests."""

from __future__ import annotations

import pytest

from conftest import ROOT, SPEC, load_file

identity_module = load_file(
    "runtime_identity_test",
    ROOT / "scripts/runtime_identity.py",
)
artifact_identity = load_file(
    "model_artifact_identity_test",
    ROOT / "scripts/model_artifact_identity.py",
)


def test_runtime_identity_is_deterministic_and_complete() -> None:
    first = identity_module.identity(SPEC / "runtime")
    second = identity_module.identity(SPEC / "runtime")
    assert first == second
    assert first["file_count"] >= 7
    assert "api.py" in first["files"]
    assert "policy_public.json" in first["files"]


def test_runtime_identity_rejects_a_symlink_root(tmp_path) -> None:
    """Resolving a path must not erase the evidence that its root was a link."""
    linked = tmp_path / "runtime-link"
    linked.symlink_to(SPEC / "runtime", target_is_directory=True)
    with pytest.raises(ValueError, match="real directory"):
        identity_module.identity(linked)


def test_model_identity_allows_only_repository_local_hf_links(tmp_path) -> None:
    """HF blob links are valid, but a link cannot escape the mounted repo."""
    repository = tmp_path / "repo"
    blobs = repository / "blobs"
    snapshot = repository / "snapshots/revision"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    blob = blobs / "config-blob"
    blob.write_text("{}\n", encoding="utf-8")
    local_link = snapshot / "config.json"
    local_link.symlink_to(blob)
    assert artifact_identity.regular_file(
        local_link,
        "config",
        repository_root=repository,
    ) == blob.resolve()

    outside = tmp_path / "outside"
    outside.write_text("no\n", encoding="utf-8")
    escaping = snapshot / "escaping.json"
    escaping.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        artifact_identity.regular_file(
            escaping,
            "escaping",
            repository_root=repository,
        )


def test_request_and_storage_bounds_are_distinct() -> None:
    from runtime.api import EngineContract

    contract = EngineContract(
        target_model_id="target",
        target_revision="target-revision",
        draft_model_id="draft",
        draft_revision="draft-revision",
    )
    assert contract.max_request_tokens == 3968
    assert contract.max_sequence_length == 4096
