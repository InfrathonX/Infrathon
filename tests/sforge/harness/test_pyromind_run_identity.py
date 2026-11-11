from datetime import datetime, timedelta, timezone
import re
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sforge.pyromind.run_identity import make_run_identity


def identity(**overrides):
    options = dict(preflight=False, model={'MODEL': 'example-model', 'REASONING_EFFORT': 'max'},
                   budget_seconds=21600, timestamp='20261001-010203', suffix='abcdef')
    options.update(overrides)
    return make_run_identity('sparse-mla', **options)


def test_agent_folder_exposes_type_model_effort_and_active_budget():
    result = identity()
    assert result['run_id'] == 'agent--sparse-mla--example-model--max--6h--20261001-010203--abcdef'
    assert result['run_kind'] == 'agent'
    assert result['run_label'] == 'Agent · sparse-mla · example-model · effort=max · budget=6h'


def test_preflight_does_not_claim_an_agent_or_six_hour_run():
    result = identity(preflight=True, model={'MODEL': 'unused-model'}, budget_seconds=21600)
    assert result['run_id'] == 'preflight--sparse-mla--20261001-010203--abcdef'
    assert result['run_kind'] == 'preflight'
    assert result['run_label'] == 'Preflight · sparse-mla · no Agent'
    assert 'unused-model' not in str(result) and '6h' not in str(result)


@pytest.mark.parametrize(('seconds', 'label'), [(43200, '12h'), (86400, '24h'), (90, '90s'), (1, '1s')])
def test_budget_is_exact(seconds, label):
    assert f'--{label}--' in identity(budget_seconds=seconds)['run_id']


def test_default_effort_is_high_and_no_other_model_fields_are_exposed():
    model = {'MODEL': 'example-model-mini', 'API_KEY': 'private-value',
             'BASE_URL': 'https://secret-provider.invalid/token'}
    result = identity(model=model)
    assert '--example-model-mini--high--' in result['run_id']
    assert 'private-value' not in str(result) and 'secret-provider' not in str(result)


def test_long_and_non_slug_names_are_bounded_without_ambiguous_truncation():
    model = {'MODEL': 'organization/' + '模型-very-long-name-' * 20,
             'REASONING_EFFORT': 'very-very-extremely-high-effort'}
    result = make_run_identity('long-profile-' * 20, preflight=False, model=model,
        budget_seconds=21600, timestamp='20261001-010203', suffix='abcdef')
    assert len(result['run_id']) <= 180
    assert len(result['run_id'] + '-final') < 200
    assert re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,199}', result['run_id'])
    parts = result['run_id'].split('--')
    assert len(parts[1]) <= 32 and len(parts[2]) <= 40 and len(parts[3]) <= 12
    assert model['MODEL'] in result['run_label']
    second = identity(model={'MODEL': 'a' * 100 + 'b'})
    third = identity(model={'MODEL': 'a' * 100 + 'c'})
    assert second['run_id'] != third['run_id']
    assert identity(model={'MODEL': 'a/b'})['run_id'] != identity(model={'MODEL': 'a-b'})['run_id']


@pytest.mark.parametrize('keyword_judge_name', [False, True])
@pytest.mark.pyromind_sdk
def test_actual_backend_names_and_reconciliation_keep_exact_run_ownership(tmp_path, keyword_judge_name):
    from sforge.harness.backend.pyromind_backend import PyroMindBackend
    from sforge.pyromind.runner import guard_judge_backend, reconcile_run_sandboxes

    one, two = identity(), identity(suffix='123456')
    assert one['sandbox_name_prefix'] != two['sandbox_name_prefix']
    prefix, foreign_prefix = one['sandbox_name_prefix'], two['sandbox_name_prefix']
    assert len(one['run_id']) > 45
    registry = {}
    client = Mock()

    def create(request):
        sandbox = SimpleNamespace(id=f'sb-{len(registry)}', name=request.name, status='Running')
        registry[sandbox.id] = sandbox
        return sandbox

    client.create.side_effect = create
    client.list.side_effect = lambda: list(registry.values())
    client.get_sandbox.side_effect = lambda identifier: registry[identifier]
    client.delete.side_effect = lambda identifier: registry.pop(identifier)
    worker_backend = PyroMindBackend(client=client, gpu_count=0, event_path=tmp_path / 'worker.jsonl')
    judge_backend = PyroMindBackend(client=client, gpu_count=0, event_path=tmp_path / 'judge.jsonl')
    guard_judge_backend(judge_backend, threading.Event(), prefix)
    worker = worker_backend.create_container('image', prefix + '-worker')
    long_judge_name = 'benchmark.judge.' + 'long-task-name.' * 20 + one['run_id']
    judge = (judge_backend.create_container(image='image', name=long_judge_name)
             if keyword_judge_name else judge_backend.create_container('image', long_judge_name))
    foreign = worker_backend.create_container('image', foreign_prefix + '-worker')
    for handle in (worker, judge):
        assert handle.name.startswith(prefix + '-')
        assert len(handle.name) <= 54
    assert long_judge_name not in judge.name  # Backend truncation really occurred.

    # Include an owned Sandbox whose create response never reached the journal.
    registry['sb-orphan'] = SimpleNamespace(id='sb-orphan', name=prefix + '-late-create', status='Running')
    # A matching string without the ownership separator is not this run.
    registry['sb-near'] = SimpleNamespace(id='sb-near', name=prefix + '0-worker', status='Running')
    registry['sb-bare'] = SimpleNamespace(id='sb-bare', name=prefix, status='Running')
    assert reconcile_run_sandboxes(judge_backend, prefix) == []
    owned_ids = {worker.id, judge.id, 'sb-orphan'}
    assert {call.args[0] for call in client.delete.call_args_list} == owned_ids
    assert {call.args[0] for call in client.get_sandbox.call_args_list} == owned_ids
    assert {call.args[0] for call in client.pause.call_args_list} == owned_ids
    assert set(registry) == {foreign.id, 'sb-near', 'sb-bare'}


@pytest.mark.parametrize('prefix', ['', 'sf-', 'sf-' + 'a' * 19, 'sf-' + 'a' * 21,
                                   'sf-' + 'A' * 20, '../run', identity()['run_id']])
def test_reconciliation_rejects_unsafe_or_unbounded_prefix_before_sdk(prefix):
    from sforge.pyromind.runner import reconcile_run_sandboxes
    backend = SimpleNamespace(client=Mock(), cleanup_container=Mock())
    with pytest.raises(ValueError, match='Invalid run Sandbox name prefix'):
        reconcile_run_sandboxes(backend, prefix)
    backend.client.list.assert_not_called()
    backend.cleanup_container.assert_not_called()


def test_aware_timestamps_are_normalized_to_utc():
    local = datetime(2026, 10, 1, 9, 2, 3, tzinfo=timezone(timedelta(hours=8)))
    assert identity(timestamp=local)['run_id'] == identity()['run_id']


@pytest.mark.parametrize('overrides', [
    {'model': {}}, {'model': {'MODEL': ''}}, {'model': 'not-a-mapping'},
    {'model': {'MODEL': 'm', 'REASONING_EFFORT': None}},
    {'preflight': 'false'}, {'suffix': '../bad'}, {'suffix': 'ABCDEF'},
    {'timestamp': '20261301-000000'}, {'timestamp': datetime(2026, 10, 1)},
    {'budget_seconds': True}, {'budget_seconds': 0}, {'budget_seconds': 1.5},
    {'budget_seconds': 86401},
])
def test_invalid_inputs_fail_before_resource_creation(overrides):
    with pytest.raises(ValueError):
        identity(**overrides)
