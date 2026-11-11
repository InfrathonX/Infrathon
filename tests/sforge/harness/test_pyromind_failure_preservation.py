"""An interrupted Worker snapshot must survive cleanup without becoming final."""
import hashlib
import io
import json
import tarfile
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sforge.harness.version_store import VersionStore
from sforge.pyromind.runner import initialize_task_store, preserve_failed_worker_candidate


@pytest.fixture
def interrupted_capture(tmp_path, monkeypatch):
    body = b'answer = 42\n'
    memory = io.BytesIO()
    with tarfile.open(fileobj=memory, mode='w:gz') as bundle:
        member = tarfile.TarInfo('candidate.py')
        member.size = len(body)
        bundle.addfile(member, io.BytesIO(body))
    payload = memory.getvalue()
    task = SimpleNamespace(task_id='task1', submit_paths=['candidate.py'],
                           submit_exclude=[], candidate_identity_exclude=[])
    root = tmp_path / 'version-store'
    initialize_task_store(root, 'run1', task.task_id)
    store = VersionStore.open(root)
    backend = SimpleNamespace(client=SimpleNamespace(read_file=Mock(return_value=payload)))
    mirror = Mock()
    auth = Mock()
    monkeypatch.setattr('sforge.pyromind.runner.collect_worker_auth_secrets', auth)
    progress = {'status': 'failed', 'no_agent_processes': True,
                'candidate_archive_available': True, 'candidate_archive_bytes': len(payload),
                'candidate_archive_sha256': hashlib.sha256(payload).hexdigest()}
    known_secrets = set()

    def preserve(**changes):
        return preserve_failed_worker_candidate(
            backend, SimpleNamespace(id='worker-owned'), tmp_path, task, store, mirror,
            object(), known_secrets, 'run1', {**progress, **changes})

    return SimpleNamespace(preserve=preserve, root=tmp_path, store=store, mirror=mirror,
                           backend=backend, auth=auth, secrets=known_secrets, payload=payload)


def test_failure_snapshot_is_verified_checkpoint_and_not_a_final_tombstone(interrupted_capture):
    f = interrupted_capture
    receipt = f.preserve()
    [version] = f.store.list_versions()
    assert version.source == 'checkpoint' and version.label == 'supervisor-failure'
    assert version.archive_path.read_bytes() == f.payload
    assert receipt['budget_completed'] is False
    assert receipt['storage_readback_verified'] is True
    assert receipt == json.loads((f.root / 'failure-snapshot.json').read_text())
    assert not (f.root / 'final.json').exists()
    f.auth.assert_called_once()
    assert f.auth.call_args.kwargs == {'required': True, 'journal_required': True}
    f.mirror.verify_versions.assert_called_once_with([version])
    assert f.mirror.sync.call_count == 2


@pytest.mark.parametrize('change', [
    {'status': 'running'}, {'no_agent_processes': False},
    {'candidate_archive_available': False},
])
def test_unsealed_or_running_worker_never_persists_an_interrupted_snapshot(interrupted_capture, change):
    f = interrupted_capture
    with pytest.raises(RuntimeError, match='writer boundary'):
        f.preserve(**change)
    f.backend.client.read_file.assert_not_called()
    assert not f.store.list_versions()


@pytest.mark.parametrize('change', [
    {'candidate_archive_sha256': '0' * 64}, {'candidate_archive_bytes': 0},
])
def test_archive_must_match_the_supervisor_boundary_receipt(interrupted_capture, change):
    f = interrupted_capture
    with pytest.raises(RuntimeError, match='differs'):
        f.preserve(**change)
    assert not f.store.list_versions()
    f.mirror.sync.assert_not_called()


def test_credential_bearing_candidate_is_not_saved_or_uploaded(interrupted_capture):
    f = interrupted_capture
    f.secrets.add('answer = 42')
    with pytest.raises(ValueError, match='Credential material'):
        f.preserve()
    assert not f.store.list_versions()
    f.mirror.sync.assert_not_called()


def test_storage_readback_failure_keeps_local_snapshot_and_does_not_claim_ack(interrupted_capture):
    f = interrupted_capture
    f.mirror.verify_versions.side_effect = RuntimeError('readback mismatch')
    with pytest.raises(RuntimeError, match='readback mismatch'):
        f.preserve()
    [version] = f.store.list_versions()
    assert version.archive_path.read_bytes() == f.payload
    receipt = json.loads((f.root / 'failure-snapshot.json').read_text())
    assert receipt['local_archive_saved'] is True
    assert receipt['storage_readback_verified'] is False
    assert receipt['budget_completed'] is False
