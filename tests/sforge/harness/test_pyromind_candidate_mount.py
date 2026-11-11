from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sforge.harness.backend import ExecResult, StreamingExecResult
from sforge.harness.backend.pyromind_backend import PyroMindBackend, PyroMindHandle
from sforge.harness.config import SForgeConfig
from sforge.harness.run_evaluation import judge_submission
from sforge.harness.task_spec import JudgeSpec, TaskSpec, WorkSpec
from sforge.pyromind.candidate_mount import (
    CANDIDATE_MOUNT_PATH, MOUNT_VERIFICATION_SOURCE, MountedCandidateArchive,
)


def _archive():
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w:gz') as bundle:
        for name, data in [('solution.py', b'answer = 42\n'), ('secret.txt', b'excluded')]:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
    return output.getvalue()


def _reference(archive=None, *, run_id='run-one', version_id='version-one'):
    archive = _archive() if archive is None else archive
    manifest = {
        'schema_version': 1, 'task_id': 'mounted-task', 'task_directory': 'mounted-task',
        'run_id': run_id, 'version_id': version_id,
        'archive_sha256': hashlib.sha256(archive).hexdigest(), 'archive_size': len(archive),
    }
    encoded = json.dumps(manifest, sort_keys=True).encode()
    ref = MountedCandidateArchive(
        task_id='mounted-task', run_id=run_id, version_id=version_id,
        host_path=f'/workspace/sforge-pyromind/{run_id}/version-store/tasks/mounted-task/runs/{run_id}/versions/{version_id}',
        archive_sha256=manifest['archive_sha256'], archive_size=len(archive),
        manifest_sha256=hashlib.sha256(encoded).hexdigest(),
    )
    return ref, encoded


@pytest.mark.parametrize('changes', [
    {'host_path': '/workspace/sforge-pyromind/run-one'},
    {'host_path': '/workspace/sforge-pyromind/run-one/version-store/tasks/mounted-task/runs/another/versions/version-one'},
    {'host_path': '/workspace/sforge-pyromind/run-one/version-store/tasks/mounted-task/runs/run-one/versions/version-one/..'},
    {'host_path': '/workspace/sforge-assets/shared/model'},
    {'task_id': '../another'}, {'run_id': 'different'}, {'version_id': '..'},
    {'archive_sha256': 'sha256:' + 'f' * 64}, {'manifest_sha256': 'wrong'},
    {'archive_size': True}, {'archive_size': 0}, {'archive_size': 128 * 1024 * 1024 + 1},
])
def test_reference_rejects_unowned_or_noncanonical_scope(changes):
    reference, _ = _reference()
    with pytest.raises(ValueError):
        replace(reference, **changes)


def test_reference_is_frozen():
    reference, _ = _reference()
    with pytest.raises(FrozenInstanceError):
        reference.run_id = 'other'


@pytest.mark.pyromind_sdk
def test_concurrent_judges_only_receive_their_exact_version_mount():
    from pyromind_sdk.client.models import VolumeMount
    base_mounts = [VolumeMount(host_path='/workspace/assets/model',
                              mount_path='/models/qwen', read_only=True)]
    requests = []
    barrier = threading.Barrier(2)

    def create(request):
        requests.append(request)
        barrier.wait(timeout=5)
        return SimpleNamespace(id='sb-' + request.name, name=request.name)

    backend = PyroMindBackend(SimpleNamespace(create=create), volume_mounts=base_mounts)
    first, _ = _reference(run_id='first')
    second, _ = _reference(run_id='second')
    with ThreadPoolExecutor(max_workers=2) as executor:
        handles = list(executor.map(
            lambda ref: backend.create_container('image', ref.run_id, mounted_archive=ref),
            [first, second],
        ))
    assert backend.volume_mounts is base_mounts
    assert len(base_mounts) == 1
    assert handles[0].mounted_archive == first
    assert handles[1].mounted_archive == second
    assert {request.volume_mounts[-1].host_path for request in requests} == {
        first.host_path, second.host_path}
    for request in requests:
        assert len(request.volume_mounts) == 2
        assert request.volume_mounts[-1].read_only is True
        assert request.volume_mounts[-1].mount_path == str(CANDIDATE_MOUNT_PATH)


@pytest.mark.parametrize('target', ['/', '/mnt', '/mnt/sforge-candidate',
                                    '/mnt/sforge-candidate/nested', '/mnt/../mnt'])
@pytest.mark.pyromind_sdk
def test_mount_collisions_fail_before_platform_create(target):
    from pyromind_sdk.client.models import VolumeMount
    client = Mock()
    backend = PyroMindBackend(client, volume_mounts=[
        VolumeMount(host_path='/workspace/model', mount_path=target, read_only=True)])
    reference, _ = _reference()
    with pytest.raises(ValueError, match='collides'):
        backend.create_container('image', 'judge', mounted_archive=reference)
    client.create.assert_not_called()


def _mounted_directory(tmp_path, archive, manifest):
    mount = tmp_path / 'candidate'
    mount.mkdir()
    (mount / 'submission.tar.gz').write_bytes(archive)
    (mount / 'manifest.json').write_bytes(manifest)
    return mount


def _verify_locally(monkeypatch, reference, mount, *, read_only=True):
    # Execute the exact remote verifier; only relocate its mount fixture.
    source = MOUNT_VERIFICATION_SOURCE.replace(
        "mount=pathlib.Path('/mnt/sforge-candidate')", f'mount=pathlib.Path({str(mount)!r})')
    monkeypatch.setattr(sys, 'argv', ['verify', json.dumps(asdict(reference))])
    monkeypatch.setattr(os, 'fstatvfs', lambda fd: SimpleNamespace(f_flag=os.ST_RDONLY if read_only else 0))
    exec(compile(source, '<mounted-candidate-verifier>', 'exec'), {})


def test_remote_verifier_checks_exact_manifest_and_archive(monkeypatch, tmp_path, capsys):
    archive = _archive()
    reference, manifest = _reference(archive)
    mount = _mounted_directory(tmp_path, archive, manifest)
    _verify_locally(monkeypatch, reference, mount)
    evidence = json.loads(capsys.readouterr().out)
    assert evidence == {'archive_sha256': reference.archive_sha256,
                        'manifest_sha256': reference.manifest_sha256,
                        'archive_size': len(archive), 'read_only': True,
                        'version_id': reference.version_id}


@pytest.mark.parametrize('mutation,match', [
    ('archive', 'archive digest mismatch'), ('size', 'archive size mismatch'),
    ('manifest', 'manifest digest mismatch'), ('owner', 'manifest identity mismatch'),
    ('task_directory', 'task directory mismatch'), ('writable', 'not read-only'),
    ('archive-symlink', None), ('manifest-symlink', None),
    ('directory-symlink', 'traverses a symlink'), ('fifo', 'not a regular file'),
])
def test_remote_verifier_rejects_corruption_and_unsafe_mounts(
    monkeypatch, tmp_path, mutation, match,
):
    archive = _archive()
    reference, manifest = _reference(archive)
    mount = _mounted_directory(tmp_path, archive, manifest)
    if mutation == 'archive':
        (mount / 'submission.tar.gz').write_bytes(b'x' * len(archive))
    elif mutation == 'size':
        (mount / 'submission.tar.gz').write_bytes(archive + b'x')
    elif mutation == 'manifest':
        (mount / 'manifest.json').write_bytes(b'{}')
    elif mutation in {'owner', 'task_directory'}:
        changed = json.loads(manifest)
        changed['run_id' if mutation == 'owner' else 'task_directory'] = 'other'
        content = json.dumps(changed).encode()
        (mount / 'manifest.json').write_bytes(content)
        reference = replace(reference, manifest_sha256=hashlib.sha256(content).hexdigest())
    elif mutation.endswith('-symlink'):
        source = mount if mutation == 'directory-symlink' else mount / (
            'submission.tar.gz' if mutation == 'archive-symlink' else 'manifest.json')
        target = tmp_path / 'actual'
        source.rename(target)
        source.symlink_to(target)
    elif mutation == 'fifo':
        (mount / 'submission.tar.gz').unlink()
        os.mkfifo(mount / 'submission.tar.gz')
    with pytest.raises((RuntimeError, OSError), match=match):
        _verify_locally(monkeypatch, reference, mount, read_only=mutation != 'writable')


def test_missing_mount_expires_without_fallback(monkeypatch, tmp_path):
    import time
    reference, _ = _reference()
    monotonic = iter([0, 31])
    monkeypatch.setattr(time, 'monotonic', lambda: next(monotonic))
    with pytest.raises(RuntimeError, match='did not become visible'):
        _verify_locally(monkeypatch, reference, tmp_path / 'missing')


def test_new_mount_visibility_is_retried_for_a_bounded_window(monkeypatch, tmp_path, capsys):
    import time
    archive = _archive()
    reference, manifest = _reference(archive)
    monkeypatch.setattr(time, 'monotonic', lambda: 0)
    sleeps = []
    def become_visible(seconds):
        sleeps.append(seconds)
        _mounted_directory(tmp_path, archive, manifest)
    monkeypatch.setattr(time, 'sleep', become_visible)
    _verify_locally(monkeypatch, reference, tmp_path / 'candidate')
    assert sleeps == [1]
    assert json.loads(capsys.readouterr().out)['archive_sha256'] == reference.archive_sha256


def test_backend_requires_exact_handle_reference_before_exec():
    reference, _ = _reference()
    backend = PyroMindBackend(Mock())
    backend.exec_run_with_timeout = Mock()
    with pytest.raises(ValueError, match='not bound'):
        backend.verify_mounted_archive(PyroMindHandle('sb-one', 'judge'), reference)
    backend.exec_run_with_timeout.assert_not_called()


@pytest.mark.parametrize('output,exit_code,timed_out', [
    ('invalid-json', 0, False), ('{}', 0, False), ('bad hash', 1, False), ('', 0, True),
])
def test_backend_rejects_failed_or_missing_mount_evidence(output, exit_code, timed_out):
    reference, _ = _reference()
    backend = PyroMindBackend(Mock())
    backend.exec_run_with_timeout = Mock(return_value=StreamingExecResult(
        output=output, exit_code=exit_code, timed_out=timed_out))
    handle = PyroMindHandle('sb-one', 'judge', mounted_archive=reference)
    with pytest.raises(RuntimeError, match='verification'):
        backend.verify_mounted_archive(handle, reference)


class _EvaluationBackend:
    backend_name = 'pyromind'

    def __init__(self, *, verification_error=None):
        self.events = []
        self.uploads = {}
        self.commands = []
        self.verification_error = verification_error
        self.handle = SimpleNamespace(id='sb-judge', name='mounted-judge')

    def create_container(self, *args, **kwargs):
        self.events.append(('create', kwargs.get('mounted_archive')))
        return self.handle

    def start_container(self, handle):
        self.events.append(('start', handle.id))

    def verify_mounted_archive(self, handle, reference):
        self.events.append(('verify', reference))
        if self.verification_error:
            raise self.verification_error
        return reference.archive_path

    def copy_to_container(self, handle, source, destination):
        self.uploads[str(destination)] = Path(source).read_bytes()

    def exec_run(self, handle, command, **kwargs):
        self.commands.append(command)
        return ExecResult()

    def exec_run_with_timeout(self, handle, command, timeout):
        self.events.append(('evaluate', command))
        return StreamingExecResult(output='\n'.join([
            '>>>>> Start Structured Result',
            '{"valid":true,"score":1.0,"passed":1,"total_tests":1}',
            '>>>>> End Structured Result',
        ]))

    def cleanup_container(self, handle, logger=None):
        self.events.append(('cleanup', handle.id if handle else None))


def _task():
    return TaskSpec(task_id='mounted-task', name='mounted test', base_image='image',
        platform='linux/amd64', cwd='/workspace', submit_paths=['solution.py'],
        submit_exclude=[], work=WorkSpec(specs_dir='/specs', agent_query='', image_tag='prebuilt'),
        judge=JudgeSpec(eval_cmd='python3 judge.py', eval_timeout=60,
                       parser='structured_json', image_tag='prebuilt'), benchmark_name='test')


def _evaluate(tmp_path, backend, archive, reference):
    return judge_submission(_task(), archive, SForgeConfig(log_dir=tmp_path), backend,
        submission_id='mounted-one', log_dir=tmp_path / 'evaluation', run_id='run-one',
        round_id='agent-1', mounted_archive=reference)


def test_evaluator_mounts_exact_archive_without_controller_candidate_upload(tmp_path):
    archive = _archive()
    reference, _ = _reference(archive)
    backend = _EvaluationBackend()
    report = _evaluate(tmp_path, backend, archive, reference)
    assert report.valid is True
    assert (tmp_path / 'evaluation/submission.tar.gz').read_bytes() == archive
    assert '/tmp/submission.tar.gz' not in backend.uploads
    assert backend.uploads['/tmp/allowed_files.txt'] == b'solution.py\0'
    assert '/tmp/.sforge-evaluation-context.json' in backend.uploads
    assert '/tmp/eval.sh' in backend.uploads
    extraction = next(command for command in backend.commands if isinstance(command, str) and 'tar xzf' in command)
    assert 'tar xzf /mnt/sforge-candidate/submission.tar.gz' in extraction
    assert '--null --verbatim-files-from' in extraction
    assert [event[0] for event in backend.events] == ['create', 'start', 'verify', 'evaluate', 'cleanup']


@pytest.mark.parametrize('problem', ['hash', 'size', 'task', 'backend', 'malformed'])
def test_invalid_candidate_never_creates_a_judge(tmp_path, problem):
    archive = _archive()
    reference, _ = _reference(archive)
    backend = _EvaluationBackend()
    if problem == 'hash':
        reference = replace(reference, archive_sha256='f' * 64)
    elif problem == 'size':
        reference = replace(reference, archive_size=1)
    elif problem == 'task':
        reference = replace(reference, task_id='other-task',
            host_path=reference.host_path.replace('/tasks/mounted-task/', '/tasks/other-task/'))
    elif problem == 'backend':
        backend.backend_name = 'k8s'
    elif problem == 'malformed':
        archive = b'invalid tar'
        reference, _ = _reference(archive)
    report = _evaluate(tmp_path, backend, archive, reference)
    assert report.valid is False
    assert report.metrics['evaluation_status'] == 'infrastructure_error'
    assert backend.events == [('cleanup', None)]
    assert backend.uploads == {}


def test_mount_verification_failure_never_falls_back_and_cleans_judge(tmp_path):
    archive = _archive()
    reference, _ = _reference(archive)
    backend = _EvaluationBackend(verification_error=RuntimeError('digest mismatch'))
    report = _evaluate(tmp_path, backend, archive, reference)
    assert report.valid is False
    assert report.metrics['evaluation_status'] == 'infrastructure_error'
    assert [event[0] for event in backend.events] == ['create', 'start', 'verify', 'cleanup']
    assert backend.uploads == {}
    assert backend.commands == []


def test_legacy_copy_transport_is_unchanged(tmp_path):
    archive = _archive()
    backend = _EvaluationBackend()
    report = _evaluate(tmp_path, backend, archive, None)
    assert report.valid is True
    assert backend.uploads['/tmp/submission.tar.gz'] == archive
    assert not any(event[0] == 'verify' for event in backend.events)
