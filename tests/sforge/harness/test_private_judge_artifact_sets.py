from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from sforge.harness import candidate_replay, run_evaluation
from sforge.harness.backend import ExecResult, StreamingExecResult
from sforge.harness.config import SForgeConfig
from sforge.harness.constants import END_TEST_OUTPUT, START_TEST_OUTPUT
from sforge.harness.task_spec import BenchmarkMeta, make_task_spec


PRIMARY = '/logs/verifier/detail.json'
NONCE = '/logs/verifier/evaluation_nonce.json'


def _transport(path: str, payload: bytes, *, kind=tarfile.REGTYPE) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w') as archive:
        member = tarfile.TarInfo(Path(path).name)
        member.type = kind
        member.size = len(payload) if kind == tarfile.REGTYPE else 0
        archive.addfile(member, io.BytesIO(payload) if kind == tarfile.REGTYPE else None)
    return buffer.getvalue()


def _task(tmp_path: Path, **judge_fields):
    data = {
        'task_id': 'private-artifact-set', 'name': 'private artifact set',
        'base_image': 'python', 'platform': 'linux/amd64', 'cwd': '/workspace',
        'submit_paths': ['.'], 'work': {'image_tag': 'prebuilt'},
        'judge': {'image_tag': 'prebuilt', 'eval_cmd': 'python verifier.py',
                  'eval_timeout': 60, 'parser': 'structured_json',
                  'private_artifact_path': PRIMARY, **judge_fields},
    }
    path = tmp_path / 'task.json'
    path.write_text(json.dumps(data))
    return make_task_spec(path, BenchmarkMeta(name='test', base_images={'python': {}}))


class Backend:
    backend_name = 'k8s'

    def __init__(self, files: dict[str, bytes], *, valid=True):
        self.files = files
        self.valid = valid
        self.events = []
        self.handle = SimpleNamespace(id='judge', name='judge')

    def create_container(self, *args, **kwargs):
        return self.handle

    def start_container(self, handle):
        pass

    def copy_to_container(self, *args):
        pass

    def exec_run(self, handle, command, **kwargs):
        if isinstance(command, list) and 'sforge-private-artifact' in command:
            assert kwargs['user'] == 'root'
            assert 'test -L' in command[2] or '[ -L ' in command[2]
            self.events.append(('probe', command[-1]))
            return ExecResult(output='present' if command[-1] in self.files else 'absent')
        return ExecResult()

    def exec_run_with_timeout(self, handle, command, timeout):
        result = {'valid': self.valid, 'score': 1.0 if self.valid else None,
                  'passed': int(self.valid), 'total_tests': 1,
                  'metrics': {'evaluation_status': 'ok' if self.valid else 'admission_rejected'}}
        output = '\n'.join([START_TEST_OUTPUT, '>>>>> Start Structured Result',
                            json.dumps(result), '>>>>> End Structured Result', END_TEST_OUTPUT])
        return StreamingExecResult(output=output, exit_code=0, elapsed_seconds=.01)

    def copy_from_container(self, handle, path):
        self.events.append(('copy', str(path)))
        return self.files[str(path)]

    def cleanup_container(self, handle, logger=None):
        self.events.append(('cleanup',))


def _members(path: Path):
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        assert all(m.isfile() and m.mode == 0o600 and m.uid == 0 and m.gid == 0 and m.mtime == 0
                   for m in members)
        return {m.name: archive.extractfile(m).read() for m in members}


def test_collects_required_detail_and_optional_nonce_before_ephemeral_cleanup(tmp_path):
    task = _task(tmp_path, private_artifact_paths=[PRIMARY, NONCE])
    detail, nonce = b'{"detail":"private"}', b'{"nonce":"private-replay-token"}'
    backend = Backend({PRIMARY: _transport(PRIMARY, detail), NONCE: _transport(NONCE, nonce)})
    output = tmp_path / 'evaluation'
    report = run_evaluation.judge_submission(
        task, b'', SForgeConfig(log_dir=tmp_path), backend, log_dir=output)
    destination = output / run_evaluation.PRIVATE_JUDGE_ARTIFACT_NAME
    assert report.valid
    assert destination.stat().st_mode & 0o777 == 0o600
    assert _members(destination) == {'detail.json': detail, 'evaluation_nonce.json': nonce}
    assert backend.events == [('copy', PRIMARY), ('probe', NONCE), ('copy', NONCE), ('cleanup',)]
    public_report = (output / 'report.json').read_text()
    assert 'private-replay-token' not in public_report
    assert 'evaluation_nonce.json' not in public_report
    assert 'private_artifact_paths' not in public_report


def test_pre_worker_rejection_with_no_nonce_preserves_authoritative_rejection(tmp_path):
    task = _task(tmp_path, private_artifact_paths=[PRIMARY, NONCE])
    backend = Backend({PRIMARY: _transport(PRIMARY, b'{"admission":"rejected"}')}, valid=False)
    output = tmp_path / 'evaluation'
    report = run_evaluation.judge_submission(
        task, b'', SForgeConfig(log_dir=tmp_path), backend, log_dir=output)
    assert not report.valid
    assert report.metrics['evaluation_status'] == 'admission_rejected'
    assert _members(output / run_evaluation.PRIVATE_JUDGE_ARTIFACT_NAME) == {
        'detail.json': b'{"admission":"rejected"}'}
    assert backend.events == [('copy', PRIMARY), ('probe', NONCE), ('cleanup',)]


@pytest.mark.parametrize('files', [
    {NONCE: _transport(NONCE, b'nonce')},
    {PRIMARY: _transport(PRIMARY, b'detail'), NONCE: _transport(NONCE, b'nonce', kind=tarfile.SYMTYPE)},
    {PRIMARY: _transport(PRIMARY, b'detail'), NONCE: _transport('/logs/verifier/wrong.json', b'nonce')},
])
def test_missing_primary_or_invalid_present_nonce_fails_closed(tmp_path, files):
    task = _task(tmp_path, private_artifact_paths=[PRIMARY, NONCE])
    output = tmp_path / 'evaluation'
    backend = Backend(files)
    report = run_evaluation.judge_submission(
        task, b'', SForgeConfig(log_dir=tmp_path), backend, log_dir=output)
    assert not report.valid
    assert report.metrics['evaluation_status'] == 'infrastructure_error'
    assert backend.events[-1] == ('cleanup',)
    assert not (output / run_evaluation.PRIVATE_JUDGE_ARTIFACT_NAME).exists()


@pytest.mark.parametrize('reply', [ExecResult(output='absent', exit_code=1), ExecResult(output='')])
def test_presence_check_failure_cannot_be_mistaken_for_absent(tmp_path, reply):
    backend = Backend({PRIMARY: _transport(PRIMARY, b'detail')})
    backend.exec_run = lambda *args, **kwargs: reply
    with pytest.raises(ValueError, match='presence could not be checked'):
        run_evaluation._persist_private_judge_artifact(
            backend, backend.handle, [PRIMARY, NONCE], tmp_path, optional_paths=[NONCE])
    assert not (tmp_path / run_evaluation.PRIVATE_JUDGE_ARTIFACT_NAME).exists()


@pytest.mark.parametrize('per_file,total,files', [
    (4, 8, {PRIMARY: b'12345', NONCE: b'1'}),
    (4, 6, {PRIMARY: b'1234', NONCE: b'123'}),
])
def test_collection_bounds_each_file_and_combined_payload_and_preserves_old_archive(
        tmp_path, monkeypatch, per_file, total, files):
    monkeypatch.setattr(run_evaluation, 'PRIVATE_JUDGE_ARTIFACT_MAX_BYTES', per_file)
    monkeypatch.setattr(run_evaluation, 'PRIVATE_JUDGE_ARTIFACT_MAX_TOTAL_BYTES', total)
    destination = tmp_path / run_evaluation.PRIVATE_JUDGE_ARTIFACT_NAME
    destination.write_bytes(b'previous-private-evidence')
    backend = Backend({name: _transport(name, payload) for name, payload in files.items()})
    with pytest.raises(ValueError, match='size'):
        run_evaluation._persist_private_judge_artifact(
            backend, backend.handle, [PRIMARY, NONCE], tmp_path, optional_paths=[NONCE])
    assert destination.read_bytes() == b'previous-private-evidence'
    assert list(tmp_path.glob('.private_judge_artifacts.tar.gz.*')) == []


def test_archive_overhead_cannot_exceed_retention_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(run_evaluation, 'SUBMISSION_ARCHIVE_MAX_BYTES', 1)
    backend = Backend({PRIMARY: _transport(PRIMARY, b'detail')})
    with pytest.raises(ValueError, match='retained size limit'):
        run_evaluation._persist_private_judge_artifact(backend, backend.handle, PRIMARY, tmp_path)
    assert not (tmp_path / run_evaluation.PRIVATE_JUDGE_ARTIFACT_NAME).exists()


@pytest.mark.parametrize('paths', [
    'detail.json', [PRIMARY, PRIMARY], ['/logs/verifier/../detail.json'],
    ['/logs/verifier//detail.json'], ['/logs/verifier/sub/detail.json'],
    ['/logs/verifier/detail.json/'], ['/logs/verifier/./detail.json'],
    ['/logs/verifier/detail\n.json'], [NONCE],
    [f'/logs/verifier/file{i}.json' for i in range(9)],
])
def test_task_loader_rejects_invalid_evidence_set_before_execution(tmp_path, paths):
    with pytest.raises(ValueError):
        _task(tmp_path, private_artifact_paths=paths)


def test_legacy_configuration_and_public_plane_override_remain_isolated(tmp_path):
    legacy = _task(tmp_path)
    assert run_evaluation._effective_private_artifact_paths(legacy, None, None) == [PRIMARY]
    assert 'private_artifact_paths' not in candidate_replay._task_contract(legacy)['judge']
    configured = _task(tmp_path, private_artifact_paths=[PRIMARY, NONCE])
    assert run_evaluation._effective_private_artifact_paths(configured, None, None) == [PRIMARY, NONCE]
    assert run_evaluation._effective_private_artifact_paths(configured, None, []) == []
    assert run_evaluation._effective_private_artifact_paths(
        configured, '/logs/verifier/public.json', None) == ['/logs/verifier/public.json']
    assert candidate_replay._task_contract(configured)['judge']['private_artifact_paths'] == [PRIMARY, NONCE]


def test_list_without_legacy_primary_is_required_in_full(tmp_path):
    task = _task(tmp_path, private_artifact_path=None, private_artifact_paths=[NONCE])
    backend = Backend({})
    report = run_evaluation.judge_submission(
        task, b'', SForgeConfig(log_dir=tmp_path), backend, log_dir=tmp_path / 'evaluation')
    assert report.metrics['evaluation_status'] == 'infrastructure_error'
    assert backend.events == [('copy', NONCE), ('cleanup',)]


def test_replay_identity_binds_evidence_set_without_exposing_new_private_metadata(tmp_path):
    task = _task(tmp_path)
    context = {'judge_image_id': 'sha256:image-a'}
    before = candidate_replay.evaluator_descriptor(
        task, SForgeConfig(), timeout=None, runtime_context=context)
    task.judge.private_artifact_paths = [PRIMARY, NONCE]
    after = candidate_replay.evaluator_descriptor(
        task, SForgeConfig(), timeout=None, runtime_context=context)
    assert before['task_contract_sha256'] != after['task_contract_sha256']
    assert before['fingerprint'] != after['fingerprint']
    assert set(before) == set(after)
    assert 'evaluation_nonce.json' not in json.dumps(after)
