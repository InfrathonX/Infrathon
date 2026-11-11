import hashlib
import io
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sforge.harness.version_store import VersionStore
from sforge.pyromind.runner import ArtifactMirror, initialize_task_store
from sforge.pyromind.version_transport import PersistedCandidateTransport


class Missing(Exception):
    code = 'NoSuchKey'


class Response(io.BytesIO):
    def release_conn(self):
        pass


class Storage:
    def __init__(self):
        self.objects = {}
        self.puts = []
        self.gets = []
        self.fail_key = None

    def get_object(self, bucket, key):
        self.gets.append(key)
        if key not in self.objects:
            raise Missing()
        return Response(self.objects[key])

    def put_object(self, bucket, key, data, size):
        self.puts.append(key)
        if key == self.fail_key:
            raise TimeoutError('fixture upload failure')
        content = data.read()
        assert len(content) == size
        self.objects[key] = content


@pytest.fixture
def setup(tmp_path):
    import tarfile
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode='w:gz') as bundle:
        member = tarfile.TarInfo('candidate.py')
        content = b'answer = 42\n'
        member.size = len(content)
        bundle.addfile(member, io.BytesIO(content))
    initialize_task_store(tmp_path/'version-store', 'run-1', 'task-1')
    store = VersionStore.open(tmp_path/'version-store')
    version = store.record_archive(task_id='task-1', run_id='run-1',
        source='checkpoint', round_id='checkpoint-1', label='saved',
        archive=payload.getvalue(), submit_paths=['candidate.py'], submit_exclude=[])
    storage = Storage()
    mirror = ArtifactMirror(tmp_path, 'sforge-pyromind/run-1', client=storage, bucket='test')
    transport = PersistedCandidateTransport(mirror, store, 'task-1', 'run-1')
    return SimpleNamespace(root=tmp_path, store=store, version=version,
                           storage=storage, mirror=mirror, transport=transport)


def test_exact_version_persist_and_cold_reuse_without_upload(setup):
    s = setup
    ref = s.transport.persist(s.version)
    assert len(s.storage.puts) == 2
    assert all('/versions/' + s.version.version_id + '/' in p for p in s.storage.puts)
    assert ref.host_path.endswith('/versions/' + s.version.version_id)
    assert ref.archive_sha256 == s.version.archive_sha256
    assert len(s.storage.gets) == 4  # absent GET, PUT, confirming GET for each file
    assert s.transport.persist(s.version) == ref
    # A new Controller transport can reuse the same immutable remote objects.
    mirror = ArtifactMirror(s.root, 'sforge-pyromind/run-1', client=s.storage, bucket='test')
    cold = PersistedCandidateTransport(mirror, s.store, 'task-1', 'run-1')
    assert cold.persist(s.version) == ref
    assert len(s.storage.puts) == 2
    receipt = json.loads(next((s.root/'candidate-transfers').glob('*.json')).read_text())
    assert all(not o['uploaded'] and o['readback_verified'] for o in receipt['objects'])


def test_unrelated_run_metadata_upload_failure_does_not_block_version(setup):
    s = setup
    (s.root/'run.json').write_text('{"status":"running"}')
    s.storage.fail_key = 'sforge-pyromind/run-1/run.json'
    s.transport.persist(s.version)
    assert s.storage.fail_key not in s.storage.puts
    with pytest.raises(RuntimeError, match='Artifact Storage upload failed'):
        s.mirror.sync()


def test_partial_upload_is_not_acknowledged_and_retry_reuses_archive(setup):
    s = setup
    relative = s.version.path.relative_to(s.root).as_posix()
    s.storage.fail_key = s.mirror.prefix + '/' + relative + '/manifest.json'
    with pytest.raises(RuntimeError, match='Candidate Storage upload failed'):
        s.transport.persist(s.version)
    assert not (s.root/'candidate-transfers').exists()
    archive_key = s.mirror.prefix + '/' + relative + '/submission.tar.gz'
    assert s.storage.puts.count(archive_key) == 1
    s.storage.fail_key = None
    s.transport.persist(s.version)
    assert s.storage.puts.count(archive_key) == 1


@pytest.mark.parametrize('field,value', [('task_id','other'), ('run_id','other'),
                                      ('task_directory','other'), ('version_id','../../escape')])
def test_foreign_or_unsafe_version_rejected_before_remote_io(setup, field, value):
    with pytest.raises(ValueError):
        setup.transport.persist(replace(setup.version, **{field:value}))
    assert not setup.storage.gets and not setup.storage.puts


def test_remote_corruption_is_not_overwritten(setup):
    s = setup
    s.transport.persist(s.version)
    key = next(k for k in s.storage.objects if k.endswith('submission.tar.gz'))
    s.storage.objects[key] = b'changed'
    before = list(s.storage.puts)
    with pytest.raises(RuntimeError, match='immutable identity'):
        s.transport.persist(s.version)
    assert s.storage.puts == before


def test_manifest_symlink_rejected(setup):
    s = setup
    path = s.version.path/'manifest.json'
    other = s.root/'other.json'
    path.rename(other)
    path.symlink_to(other)
    with pytest.raises(OSError):
        s.transport.persist(s.version)
    assert not s.storage.gets


def test_server_owned_submission_identity_selects_saved_version(setup):
    s = setup
    entry = {'task_id':'task-1', 'run_id':'run-1', 'version_id':s.version.version_id}
    evaluate = Mock(return_value='graded')
    wrapper = s.transport.evaluate(evaluate, lambda sid: entry if sid == 'submission-1' else None)
    archive = s.store.read_archive(s.version)
    assert wrapper(task_spec=SimpleNamespace(task_id='task-1'), archive=archive,
                   submission_id='submission-1', round_id='agent-9', run_id='run-1') == 'graded'
    assert evaluate.call_args.kwargs['mounted_archive'].version_id == s.version.version_id
    assert evaluate.call_args.kwargs['round_id'] == 'agent-9'
    assert len(s.store.list_versions()) == 1
    with pytest.raises(ValueError, match='authoritative saved version'):
        wrapper(task_spec=SimpleNamespace(task_id='task-1'), archive=archive, submission_id='missing')
    with pytest.raises(ValueError, match='selected saved version'):
        wrapper(task_spec=SimpleNamespace(task_id='task-1'), archive=b'different', submission_id='submission-1')


def test_explicit_final_owner_remains_original_run(setup):
    s = setup
    evaluator = Mock()
    wrapper = s.transport.evaluate(evaluator, lambda sid: (_ for _ in ()).throw(AssertionError()))
    wrapper(task_spec=SimpleNamespace(task_id='task-1'), archive=s.store.read_archive(s.version),
            stored_version=s.version, run_id='run-1-final', round_id='manual-1')
    assert evaluator.call_args.kwargs['mounted_archive'].run_id == 'run-1'
    assert evaluator.call_args.kwargs['run_id'] == 'run-1-final'


def test_reject_foreign_mirror_prefix_and_root(setup):
    s = setup
    wrong = ArtifactMirror(s.root, 'sforge-pyromind/other', client=s.storage, bucket='test')
    with pytest.raises(ValueError, match='Storage prefix'):
        PersistedCandidateTransport(wrong, s.store, 'task-1', 'run-1')
    wrong = ArtifactMirror(s.root/'other', s.mirror.prefix, client=s.storage, bucket='test')
    with pytest.raises(ValueError, match='artifact root'):
        PersistedCandidateTransport(wrong, s.store, 'task-1', 'run-1')
