import io
import json
from pathlib import Path
import tarfile
import threading
import signal
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sforge.pyromind.runner import (
    ArtifactMirror, _drain, duration_seconds, guard_judge_backend,
    initialize_task_store, public_artifact_paths, supervisor_source,
    termination_cleanup, validate_no_credentials, verify_shared_version,
    verify_qualification_memory,
)
from sforge.harness.version_store import VersionStore
from sforge.harness.judge_server import SubmissionStatus


def archive(content):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w:gz') as bundle:
        member = tarfile.TarInfo('candidate.py')
        member.size = len(content)
        bundle.addfile(member, io.BytesIO(content))
    return data.getvalue()


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf'), 25, True])
def test_reject_invalid_budget(value):
    with pytest.raises(ValueError):
        duration_seconds(value)


def test_six_hour_budget_and_supervisor_syntax():
    assert duration_seconds(6) == 21600
    compile(supervisor_source(), '<remote-supervisor>', 'exec')


def test_credentials_are_checked_inside_compressed_members():
    secret = 'model-key-never-publish-123'
    payload = archive(('KEY=' + secret).encode())
    assert secret.encode() not in payload
    with pytest.raises(ValueError, match='Credential material'):
        validate_no_credentials(payload, [secret])
    validate_no_credentials(archive(b'def add(a,b): return a+b'), [secret])


def test_storage_allowlist_never_uploads_judge_private_files(tmp_path):
    store_root = tmp_path / 'version-store'
    initialize_task_store(store_root, 'run1', 'task1')
    version = VersionStore.open(store_root).record_archive(
        task_id='task1', run_id='run1', source='checkpoint', round_id='checkpoint-1',
        label='first', archive=archive(b'answer=42'), submit_paths=['candidate.py'],
        submit_exclude=[])
    (tmp_path / 'logs').mkdir()
    (tmp_path / 'logs/private_judge_artifacts.tar.gz').write_bytes(b'private')
    (tmp_path / 'agent.jsonl').write_text('transcript')
    (tmp_path / 'run.json').write_text('{"status":"running"}')
    (store_root / 'credential-link').symlink_to(tmp_path / 'agent.jsonl')
    paths = set(public_artifact_paths(tmp_path))
    assert version.archive_path in paths
    assert tmp_path / 'run.json' in paths
    assert not any('logs' in p.parts for p in paths)
    assert tmp_path / 'agent.jsonl' not in paths
    assert store_root / 'credential-link' not in paths
    storage = Mock()
    mirror = ArtifactMirror(tmp_path, 'run1', client=storage, bucket='bucket')
    mirror.sync()
    first_count = storage.put_object.call_count
    assert first_count >= 3
    mirror.sync()
    assert storage.put_object.call_count == first_count
    for call in storage.put_object.call_args_list:
        assert '/logs/' not in call.args[1]
        assert not call.args[1].endswith('agent.jsonl')


def test_fixed_root_artifact_symlink_cannot_publish_private_file(tmp_path):
    private = tmp_path / '.private';private.mkdir()
    evidence = private / 'hidden.json';evidence.write_bytes(b'private fixture')
    (tmp_path / 'run.json').symlink_to(evidence)
    (tmp_path / 'final.json').write_bytes(b'public fixture')
    storage = Mock()
    ArtifactMirror(tmp_path,'run',client=storage,bucket='bucket').sync()
    assert [call.args[1] for call in storage.put_object.call_args_list] == ['run/final.json']
    data = storage.put_object.call_args.args[2].getvalue()
    assert data == b'public fixture'


def test_cancel_during_create_destroys_late_judge_and_closes_admission():
    cancelled = threading.Event()
    handle = SimpleNamespace(id='sb-created')
    backend = SimpleNamespace(cleanup_container=Mock(), exec_run_with_timeout=Mock())
    def create(*args, **kwargs):
        cancelled.set()
        return handle
    backend.create_container = Mock(side_effect=create)
    guard_judge_backend(backend, cancelled)
    with pytest.raises(RuntimeError, match='while Judge was being created'):
        backend.create_container('image', 'name')
    backend.cleanup_container.assert_called_once_with(handle)
    with pytest.raises(RuntimeError, match='admission closed'):
        backend.create_container('image', 'another')


def test_judge_timeout_receives_controller_cancellation_event():
    cancelled = threading.Event()
    execute = Mock(return_value='result')
    backend = SimpleNamespace(create_container=Mock(), exec_run_with_timeout=execute,
                              cleanup_container=Mock())
    guard_judge_backend(backend, cancelled)
    assert backend.exec_run_with_timeout('judge', 'test', timeout=3600) == 'result'
    assert execute.call_args.kwargs['shutdown_event'] is cancelled


def test_drain_does_not_mistake_enum_for_completed():
    state = SimpleNamespace(submissions={'s1': {'status': SubmissionStatus.RUNNING}})
    with pytest.raises(TimeoutError, match='drain deadline'):
        _drain(state, 0)
    state.submissions['s1']['status'] = SubmissionStatus.COMPLETED
    _drain(state, 0)


def test_sigterm_runs_python_cleanup_and_restores_handler(monkeypatch):
    calls = []
    prior = lambda signum, frame: None
    monkeypatch.setattr(signal, 'getsignal', lambda signum: prior)
    monkeypatch.setattr(signal, 'signal', lambda signum, handler: calls.append((signum,handler)))
    cleaned = []
    with pytest.raises(KeyboardInterrupt, match='SIGTERM'):
        with termination_cleanup():
            try:
                calls[0][1](signal.SIGTERM, None)
            finally:
                cleaned.append(True)
    assert cleaned == [True]
    assert calls[1] == (signal.SIGTERM, signal.SIG_IGN)
    assert calls[-1] == (signal.SIGTERM, prior)


def test_signal_handlers_are_untouched_outside_main_thread(monkeypatch):
    monkeypatch.setattr(threading, 'current_thread', lambda: object())
    change = Mock()
    monkeypatch.setattr(signal, 'signal', change)
    with termination_cleanup():
        pass
    change.assert_not_called()


def test_shared_version_evidence_requires_matching_readonly_worker_digest(tmp_path, monkeypatch):
    root=tmp_path/'version-store'
    initialize_task_store(root,'run1','task1')
    version=VersionStore.open(root).record_archive(task_id='task1',run_id='run1',source='checkpoint',
        round_id='checkpoint-1',label='first',archive=archive(b'answer=42'),
        submit_paths=['candidate.py'],submit_exclude=[])
    mirror=Mock()
    worker=SimpleNamespace(id='sb-worker')
    remote=Mock(return_value=json.dumps({'archive_sha256':version.archive_sha256,'mount_read_only':True}))
    monkeypatch.setattr('sforge.pyromind.runner.checked',remote)
    evidence=verify_shared_version(object(),worker,version,tmp_path,mirror)
    mirror.sync.assert_called_once()
    mirror.verify_versions.assert_called_once_with([version])
    assert evidence['worker_shared_path'].startswith('/shared/version-store/')
    assert evidence['worker_shared_archive_sha256']==version.archive_sha256
    assert remote.call_args.args[2][:5]==['runuser','-u','agent','--','python3']
    assert json.loads((tmp_path/'shared-version-verification.json').read_text())==evidence
    remote.return_value=json.dumps({'archive_sha256':'0'*64,'mount_read_only':True})
    with pytest.raises(RuntimeError,match='verification failed'):
        verify_shared_version(object(),worker,version,tmp_path,mirror)


def test_qualification_rejects_missing_or_oom_memory_evidence(tmp_path):
    with pytest.raises(RuntimeError,match='lacks'):
        verify_qualification_memory(tmp_path)
    root=tmp_path/'resource-usage';root.mkdir()
    data={'memory.max':str(64*1024**3),'memory.peak':str(20*1024**3),
          'memory.events':'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n'}
    path=root/'sb-test.json';path.write_text(json.dumps(data))
    verify_qualification_memory(tmp_path)
    data['memory.events']='oom 1\noom_kill 0\n';path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError,match='OOM'):
        verify_qualification_memory(tmp_path)
    path.write_text(json.dumps({'error':'TimeoutError'}))
    with pytest.raises(RuntimeError,match='incomplete'):
        verify_qualification_memory(tmp_path)
