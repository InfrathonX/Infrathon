"""Sealed PyroMind finals use the configured review before any GPU admission."""

from __future__ import annotations

import io
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from _llm_judge_fixtures import (
    SEED, FakeClient, make_archive, make_task, verdict_json, write_documents,
    write_tree,
)
from sforge.harness import judge_server
from sforge.harness.config import SForgeConfig
from sforge.harness.grading import EvalReport
from sforge.harness.judge_server import JudgeState, SubmissionStatus
from sforge.harness.llm_judge import reviewer as reviewer_module
from sforge.harness.llm_judge.bundle import HostDirectoryBaseline
from sforge.harness.llm_judge.config import LLMJudgeConfig
from sforge.harness.version_store import VersionStore
from sforge.pyromind.runner import (
    ArtifactMirror, evaluate_reviewed_final, initialize_task_store,
)
from sforge.pyromind.version_transport import PersistedCandidateTransport

RUN_ID = 'run-1'
TASK = make_task()
FORBIDDEN = {
    **SEED,
    'FINAL_VERSION': '2\n',
    'candidates/kernel_v002/submission.py': (
        'def kernel(q, k, v, out):\n'
        '    out.copy_(torch.nn.functional.scaled_dot_product_attention(q, k, v))\n'
    ),
}
VIOLATION = verdict_json('VIOLATION', confidence=0.95, violations=[{
    'rule': 'No torch SDPA',
    'files': ['candidates/kernel_v002/submission.py'],
    'evidence': 'torch.nn.functional.scaled_dot_product_attention(q, k, v)',
    'explanation': 'The selected kernel delegates attention to SDPA.',
}])


class MissingObject(Exception):
    code = 'NoSuchKey'


class StoredResponse(io.BytesIO):
    def release_conn(self):
        pass


class MemoryStorage:
    def __init__(self):
        self.objects = {}

    def get_object(self, bucket, key):
        if key not in self.objects:
            raise MissingObject()
        return StoredResponse(self.objects[key])

    def put_object(self, bucket, key, data, size):
        content = data.read()
        assert len(content) == size
        self.objects[key] = content


@pytest.fixture
def final_harness(monkeypatch, tmp_path):
    monkeypatch.setattr(judge_server, 'create_backend_from_config', lambda config: object())
    monkeypatch.setattr(JudgeState, '_reaper_loop', lambda self: None)
    replies = []
    client = FakeClient(replies)
    monkeypatch.setattr(reviewer_module, 'ChatCompletionsClient', lambda config: client)
    baseline = HostDirectoryBaseline(
        write_tree(tmp_path/'seed', SEED),
        documents_root=write_documents(tmp_path/'docs'))
    tasks_dir = tmp_path/'task'/'tasks'
    tasks_dir.mkdir(parents=True)
    policy = tasks_dir.parent/'llm_judge'/'POLICY.md'
    policy.parent.mkdir()
    policy.write_text(
        '## What the task requires\nImplement attention in self-contained Triton.\n'
        '## Reject\nCalling torch SDPA.\n'
        '## Allow -- never reject for these\nOrdinary Triton optimization.\n')
    initialize_task_store(tmp_path/'version-store', RUN_ID, TASK.task_id)
    store = VersionStore.open(tmp_path/'version-store')
    storage = MemoryStorage()
    mirror = ArtifactMirror(tmp_path, 'sforge-pyromind/'+RUN_ID,
                            client=storage, bucket='test')
    transport = PersistedCandidateTransport(mirror, store, TASK.task_id, RUN_ID)
    gpu_backend = SimpleNamespace(create_container=Mock())
    measured = []

    def gpu_evaluator(**kwargs):
        gpu_backend.create_container('judge-image', 'final-judge')
        measured.append(kwargs)
        return EvalReport(task_id=TASK.task_id, submission_id=kwargs['submission_id'],
                          valid=True, score=42.0)

    def run(files=SEED, *, mode='enforce', on_error='block', cache=True,
            configure=None, version_change=None, kinds=('agent', 'auto')):
        config = SForgeConfig(log_dir=tmp_path/'logs', tasks_dir=tasks_dir,
                             llm_judge=LLMJudgeConfig(
                                 mode=mode, engine='api',
                                 api_base_url='http://127.0.0.1:9/v1',
                                 model='judge-model', max_attempts=1,
                                 on_error=on_error, cache=cache, kinds=kinds))
        state = JudgeState(config)
        state.tasks = {TASK.task_id: TASK}
        state.set_run_baseline(RUN_ID, baseline)
        if configure:
            configure(state)
        archive = make_archive(files)
        version = store.record_archive(
            task_id=TASK.task_id, run_id=RUN_ID, source='final', round_id='final',
            label='budget-final', archive=archive, submit_paths=TASK.submit_paths,
            submit_exclude=TASK.submit_exclude)
        if version_change:
            version = replace(version, **version_change)
        evaluate = transport.evaluate(gpu_evaluator, state.submissions.get)
        report = evaluate_reviewed_final(
            judge_state=state, evaluate_submission=evaluate, task_spec=TASK,
            archive=archive, stored_version=version, run_id=RUN_ID,
            round_id='final', log_dir=tmp_path/'logs'/'final',
            config=config, backend=gpu_backend, judge_image_ref='judge-image')
        return SimpleNamespace(state=state, report=report, archive=archive, version=version)

    return SimpleNamespace(run=run, replies=replies, client=client,
                           gpu=gpu_backend, measured=measured, storage=storage,
                           root=tmp_path)


def test_rejected_final_never_allocates_a_gpu(final_harness):
    h = final_harness
    h.replies.append(VIOLATION)
    result = h.run(FORBIDDEN)
    assert result.report.valid is False and result.report.score is None
    assert result.report.metrics['evaluation_status'] == 'integrity_rejected'
    h.gpu.create_container.assert_not_called()
    assert h.measured == [] and h.storage.objects == {}
    final_dir = h.root/'logs'/'final'
    assert (final_dir/'submission.tar.gz').read_bytes() == result.archive
    binding = json.loads((final_dir/'integrity_review'/'final-binding.json').read_text())
    assert (final_dir/'integrity_review'/'final-binding.json').stat().st_mode & 0o777 == 0o600
    assert binding['archive_sha256'] == result.version.archive_sha256
    assert binding['version_id'] == result.version.version_id
    assert binding['review']['status'] == 'violation'


def test_admitted_final_reviews_and_mounts_the_same_archive(final_harness):
    h = final_harness
    h.replies.append(verdict_json('PASS'))
    result = h.run()
    assert result.report.valid and result.report.score == 42.0
    assert len(h.client.calls) == 1
    h.gpu.create_container.assert_called_once()
    measured = h.measured[0]
    assert measured['archive'] is result.archive
    assert measured['run_id'] == RUN_ID+'-final'
    assert measured['mounted_archive'].run_id == RUN_ID
    assert measured['mounted_archive'].version_id == result.version.version_id
    assert measured['mounted_archive'].archive_sha256 == result.version.archive_sha256
    remote_archive = next(data for key, data in h.storage.objects.items()
                          if key.endswith('/submission.tar.gz'))
    assert remote_archive == result.archive
    review = result.state.submissions[RUN_ID+'-final']['integrity_review']
    assert review.status == 'pass' and not review.blocking
    bundle = h.root/'logs'/'runs'/RUN_ID/TASK.task_id/'submissions'/'agent-final'/'integrity_review'/'bundle.json'
    assert json.loads(bundle.read_text())['baseline_source'] == 'host-directory:'+str(h.root/'seed')


@pytest.mark.parametrize('changed', [False, True])
def test_final_cache_reuses_only_the_reviewed_candidate(final_harness, changed):
    h = final_harness
    h.replies.append(verdict_json('PASS'))
    if changed:
        h.replies.append(VIOLATION)

    def review_interactive(state):
        state.submissions['interactive'] = {'status': SubmissionStatus.QUEUED}
        state._run_integrity_review('interactive', TASK.task_id,
                                    make_archive(SEED), RUN_ID, 'agent-1')
        assert state.submissions['interactive']['integrity_review'].status == 'pass'

    result = h.run(FORBIDDEN if changed else SEED, configure=review_interactive)
    review = result.state.submissions[RUN_ID+'-final']['integrity_review']
    assert review.cache_hit is (not changed)
    assert len(h.client.calls) == (2 if changed else 1)
    assert result.report.valid is (not changed)
    assert h.gpu.create_container.call_count == (0 if changed else 1)


@pytest.mark.parametrize('on_error', ['admit', 'block'])
def test_final_review_error_obeys_the_configured_policy(final_harness, on_error):
    h = final_harness  # no scripted response: reviewer endpoint fails
    result = h.run(on_error=on_error)
    review = result.state.submissions[RUN_ID+'-final']['integrity_review']
    assert review.status == 'error'
    assert h.gpu.create_container.call_count == (1 if on_error == 'admit' else 0)
    if on_error == 'block':
        assert result.report.metrics['evaluation_status'] == 'infrastructure_error'


def test_unconfigured_mandatory_final_reviewer_blocks_before_gpu(final_harness):
    h = final_harness
    result = h.run(configure=lambda state: setattr(state, 'integrity_reviewer', None))
    assert result.report.metrics['evaluation_status'] == 'infrastructure_error'
    h.gpu.create_container.assert_not_called()


@pytest.mark.parametrize('bad_result', ['absent', 'none', 'wrong_type'])
@pytest.mark.parametrize('mode,on_error,blocked', [
    ('enforce', 'block', True),
    ('enforce', 'admit', False),
    ('shadow', 'block', False),
])
def test_missing_or_invalid_final_review_obeys_error_policy(
        final_harness, bad_result, mode, on_error, blocked):
    h = final_harness

    def break_review(state):
        if bad_result == 'absent':
            state._run_integrity_review = Mock(return_value=None)
        else:
            state.integrity_reviewer.review = Mock(
                return_value=None if bad_result == 'none' else {'status': 'pass'})

    result = h.run(mode=mode, on_error=on_error, configure=break_review)
    review = result.state.submissions[RUN_ID+'-final']['integrity_review']
    assert isinstance(review, reviewer_module.IntegrityReview)
    assert review.status == 'error' and review.blocking is blocked
    assert h.gpu.create_container.call_count == (0 if blocked else 1)
    assert len(h.measured) == (0 if blocked else 1)
    if blocked:
        assert not result.report.valid
        assert result.report.metrics['evaluation_status'] == 'infrastructure_error'
        assert h.storage.objects == {}
    else:
        assert result.report.valid
    binding = json.loads((h.root/'logs/final/integrity_review/final-binding.json').read_text())
    assert binding['archive_sha256'] == result.version.archive_sha256
    assert binding['review']['status'] == 'error'
    assert binding['review']['blocking'] is blocked


def test_uncovered_final_keeps_its_explicit_review_scope(final_harness):
    h = final_harness
    result = h.run(kinds=('auto',))
    assert result.report.valid
    assert 'integrity_review' not in result.state.submissions[RUN_ID+'-final']
    assert not (h.root/'logs/final/integrity_review/final-binding.json').exists()
    assert h.client.calls == []
    h.gpu.create_container.assert_called_once()


@pytest.mark.parametrize('mode', ['off', 'shadow'])
def test_explicit_off_and_shadow_modes_keep_their_semantics(final_harness, mode):
    h = final_harness
    h.replies.append(VIOLATION)
    result = h.run(FORBIDDEN, mode=mode)
    assert result.report.valid
    h.gpu.create_container.assert_called_once()
    assert len(h.client.calls) == (0 if mode == 'off' else 1)


@pytest.mark.parametrize('change', [
    {'run_id': 'different-run'}, {'archive_sha256': '0'*64}, {'archive_size': 1},
])
def test_foreign_or_changed_final_archive_is_denied_before_review(final_harness, change):
    h = final_harness
    h.replies.append(verdict_json('PASS'))
    with pytest.raises(ValueError, match='sealed task/run version'):
        h.run(version_change=change)
    assert h.client.calls == [] and h.measured == []
    h.gpu.create_container.assert_not_called()
