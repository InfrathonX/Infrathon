"""Catalog identity, truthful classification, and publication boundaries."""
import io
import json
from types import SimpleNamespace

import pytest

from sforge.pyromind.run_catalog import (
    MAX_METADATA_BYTES, build_run_catalog, publish_catalog_files, render_catalog_files,
)


class Response(io.BytesIO):
    def __init__(self, data, reads):
        super().__init__(data)
        self.reads = reads

    def read(self, size=-1):
        self.reads.append(size)
        return super().read(size)

    def release_conn(self):
        pass


class Storage:
    def __init__(self, runs=None):
        self.objects = {}
        self.entries = []
        self.gets, self.puts, self.lists, self.reads = [], [], [], []
        self.corrupt_put = False
        for run_id, data in (runs or {}).items():
            self.entries.append(SimpleNamespace(object_name=f'sforge-pyromind/{run_id}/', is_dir=True))
            if data is not None:
                self.objects[f'sforge-pyromind/{run_id}/run.json'] = (
                    data if isinstance(data, bytes) else json.dumps({'run_id': run_id, **data}).encode())

    def list_objects(self, bucket, **kwargs):
        self.lists.append((bucket, kwargs))
        return iter(self.entries)

    def get_object(self, bucket, key):
        self.gets.append(key)
        return Response(self.objects[key], self.reads)

    def put_object(self, bucket, key, stream, length):
        data = stream.read()
        assert len(data) == length
        self.puts.append(key)
        self.objects[key] = b'wrong' if self.corrupt_put else data


@pytest.mark.parametrize('data,kind,execution', [
    ({'run_kind': 'agent', 'agent_started': False}, 'agent', 'not-started'),
    ({'run_kind': 'preflight', 'model': 'legacy-gpt'}, 'preflight', 'not-applicable'),
    ({'qualification_scope': 'task-only', 'model': 'legacy-gpt'}, 'preflight', 'not-applicable'),
    ({'status': 'preflight-completed'}, 'preflight', 'not-applicable'),
    ({'qualification_scope': 'task-and-agent', 'agent_started': False}, 'agent', 'not-started'),
    ({'qualification_scope': 'task-and-agent', 'agent_started_at': 123}, 'agent', 'started'),
    ({'scope': 'trusted-positive-original-judge-protocol'}, 'judge-check', 'not-applicable'),
    ({'scope': 'native-non-release-requalification'}, 'judge-check', 'not-applicable'),
    ({'kind': 'trusted-oracle-qualification'}, 'judge-check', 'not-applicable'),
    ({'mode': 'current', 'agent_started': False}, 'transport-check', 'not-applicable'),
    ({'mode': 'saved', 'agent_started': False}, 'transport-check', 'not-applicable'),
    ({'agent_progress': {'started_at': 123}}, 'agent', 'started'),
    ({'agent_started_at': '2026-09-30T04:00:00Z'}, 'agent', 'started'),
    ({'model': 'example-model', 'reasoning_effort': 'max'}, 'unknown', 'unknown'),
    ({'agent_started': True, 'model': 'example-model'}, 'unknown', 'unknown'),
    ({'mode': 'current'}, 'unknown', 'unknown'),
    ({'agent_started_at': True}, 'unknown', 'unknown'),
    ({'agent_started_at': 'garbage'}, 'unknown', 'unknown'),
    ({'run_kind': 'agent'}, 'agent', 'unknown'),
])
def test_classification_uses_evidence_not_name_or_model(data, kind, execution):
    client = Storage({'agent-gpt-max-6h-misleading-name': data})
    row = build_run_catalog(client, 'bucket')['runs'][0]
    assert row['run_kind'] == kind
    assert row['agent_execution'] == execution
    assert client.lists == [('bucket', {'prefix': 'sforge-pyromind/', 'recursive': False})]
    assert not client.puts
    assert client.gets == ['sforge-pyromind/agent-gpt-max-6h-misleading-name/run.json']
    assert client.reads == [MAX_METADATA_BYTES + 1]


def test_legacy_preflight_budget_and_models_not_displayed_as_actual_agent():
    client = Storage({'pre': {'qualification_scope': 'task-only', 'budget_seconds': 21600,
                              'model': 'legacy-gpt-secret-name', 'reasoning_effort': 'max'}})
    catalog = build_run_catalog(client, 'bucket')
    row = catalog['runs'][0]
    assert row['budget_seconds'] == 21600
    assert row['applied_agent_budget_seconds'] is None
    assert row['model'] is row['reasoning_effort'] is None
    files = render_catalog_files(catalog)
    doc = files['RUNS.md'].decode()
    assert '6 小时' not in doc and '不适用' in doc
    assert 'legacy-gpt-secret-name' not in doc
    assert len([key for key in files if key.startswith('_index/preflight/')]) == 1


def test_configured_budget_visible_before_start_but_applied_is_null():
    client = Storage({'agent-1': {'run_kind': 'agent', 'agent_started': False,
                                 'budget_seconds': 43200, 'model': 'example-model',
                                 'reasoning_effort': 'max', 'status': 'failed'}})
    catalog = build_run_catalog(client, 'bucket')
    row = catalog['runs'][0]
    assert row['applied_agent_budget_seconds'] is None
    files = render_catalog_files(catalog)
    assert '12 小时' in files['RUNS.md'].decode()
    assert '未启动' in files['RUNS.md'].decode()
    assert any('12h' in key for key in files if key.startswith('_index/agent/'))


def test_started_budget_whitelist_and_no_private_urls_or_environment():
    client = Storage({'agent-1': {
        'qualification_scope': 'task-and-agent', 'agent_progress': {'started_at': 123, 'secret': 'HIDDEN'},
        'budget_seconds': 21600, 'task_id': 'task-1', 'deployment': {'name': 'sparse', 'env': 'HIDDEN'},
        'model': 'example-model-mini', 'reasoning_effort': 'max', 'token': 'HIDDEN',
        'judge_url': 'https://HIDDEN', 'private_payload': 'HIDDEN', 'env': {'SECRET': 'HIDDEN'}}})
    catalog = build_run_catalog(client, 'bucket')
    row = catalog['runs'][0]
    assert row['profile'] == 'sparse'
    assert row['applied_agent_budget_seconds'] == 21600
    files = render_catalog_files(catalog)
    assert all(b'HIDDEN' not in data for data in files.values())
    assert all(b'judge_url' not in data for data in files.values())


@pytest.mark.parametrize('data', [None, b'{', b'[]', b'null', b'{"run_id":"other"}',
                                b'{"model":"gpt"}', b'\xff', b'x' * (MAX_METADATA_BYTES + 1)])
def test_missing_or_invalid_metadata_is_visible_unknown_with_issue(data):
    client = Storage({'broken': data})
    catalog = build_run_catalog(client, 'bucket')
    assert len(catalog['runs']) == 1
    assert catalog['runs'][0]['run_id'] == 'broken'
    assert catalog['runs'][0]['run_kind'] == 'unknown'
    assert catalog['issues'][0]['run_id'] == 'broken'
    assert 'broken' in render_catalog_files(catalog)['RUNS.md'].decode()
    assert all(size == MAX_METADATA_BYTES + 1 for size in client.reads)


def test_only_top_level_safe_directories_are_read():
    client = Storage({'run-1': {'status': 'unknown'}})
    for key, is_dir in [('sforge-pyromind/_index/', True), ('sforge-pyromind/RUNS.md', False),
                        ('sforge-pyromind/run-index.json', False),
                        ('sforge-pyromind/../', True), ('sforge-pyromind/a/b/', True),
                        ('other/run-2/', True), ('sforge-pyromind/run-1/', True)]:
        client.entries.append(SimpleNamespace(object_name=key, is_dir=is_dir))
    catalog = build_run_catalog(client, 'bucket')
    assert len(catalog['runs']) == 1
    assert client.gets == ['sforge-pyromind/run-1/run.json']
    assert len(catalog['issues']) == 4  # Three bad directories, one unknown run.


def test_render_escapes_markup_and_limits_safe_pointer_names():
    run_id = 'r' * 240
    client = Storage({run_id: {'run_kind': 'agent', 'agent_started': False,
                              'model': '<img src=x onerror=alert(1)>|[click](https://bad)\nnew',
                              'reasoning_effort': '*' * 500, 'profile': 'a' * 500,
                              'status': '<script>alert(1)</script>'}})
    files = render_catalog_files(build_run_catalog(client, 'bucket'))
    doc = files['RUNS.md'].decode()
    assert '<script>' not in doc and '<img' not in doc and '[click]' not in doc
    assert '&lt;img' in doc and '&#124;' in doc and '&#91;click&#93;' in doc
    pointer_keys = [key for key in files if key.startswith('_index/')]
    assert len(pointer_keys) == 1 and len(pointer_keys[0].split('/')[-1]) <= 220
    pointer = json.loads(files[pointer_keys[0]])
    assert pointer['run']['run_id'] == run_id
    assert pointer['run']['run_path'] == run_id + '/'
    assert pointer['run_path_from_pointer_directory'] == '../../' + run_id + '/'
    assert 'not live monitoring' in pointer['snapshot_notice']


def test_renderer_also_filters_unknown_fields_and_paths():
    catalog = build_run_catalog(Storage({'ok': {}}), 'bucket')
    catalog['secret'] = 'HIDDEN'
    catalog['runs'][0].update(secret='HIDDEN', run_path='https://HIDDEN')
    catalog['issues'].append({'code': 'fixture', 'secret': 'HIDDEN', 'run_id': '../HIDDEN'})
    files = render_catalog_files(catalog)
    assert all(b'HIDDEN' not in data for data in files.values())


def test_pointer_name_stable_across_status_and_chinese_time_display():
    catalog = build_run_catalog(Storage({'run-1': {'run_kind': 'agent', 'agent_started': False,
                                                  'model': 'gpt', 'budget_seconds': 21600}}), 'bucket')
    catalog['generated_at'] = '2026-10-01T00:10:00+00:00'
    before = render_catalog_files(catalog)
    catalog['runs'][0]['status'] = 'completed'
    catalog['runs'][0]['agent_execution'] = 'started'
    after = render_catalog_files(catalog)
    assert set(before) == set(after)
    assert '2026-10-01 08:10:00 +0800' in after['RUNS.md'].decode()
    assert '北京时间' in after['RUNS.md'].decode()
    assert json.loads(after['run-index.json'])['generated_at'] == '2026-10-01T00:10:00+00:00'


def test_publish_only_navigation_and_readback_every_object():
    client = Storage({'ok': {'qualification_scope': 'task-only'}})
    original = dict(client.objects)
    files = render_catalog_files(build_run_catalog(client, 'bucket'))
    client.gets.clear()
    published = publish_catalog_files(client, 'bucket', files)
    assert set(published) == set(files)
    assert client.gets == client.puts
    assert len(client.puts) == 3
    assert all(client.objects[key] == data for key, data in original.items())
    assert all('/ok/' not in key for key in client.puts)


@pytest.mark.parametrize('key', ['ok/run.json', 'ok/version-store/a', '../run.json',
                               '/RUNS.md', '_index/agent/../../ok/run.json', '_index/evil/a.json',
                               '_index/agent/file.html', '_index/agent/.hidden.json',
                               '_index/agent/' + 'x' * 220 + '.json'])
def test_publish_invalid_batch_rejected_before_any_put(key):
    client = Storage()
    with pytest.raises(ValueError, match='restricted'):
        publish_catalog_files(client, 'bucket', {'RUNS.md': b'fine', key: b'bad'})
    assert client.puts == []


def test_publish_readback_mismatch_fails():
    client = Storage()
    client.corrupt_put = True
    with pytest.raises((ValueError, RuntimeError)):
        publish_catalog_files(client, 'bucket', {'RUNS.md': b'expected'})


@pytest.mark.parametrize('prefix', ['', '/root', 'a/../b', 'a//b', '../', 'a/'])
def test_invalid_prefix_rejected_without_network(prefix):
    client = Storage()
    with pytest.raises(ValueError):
        build_run_catalog(client, 'bucket', prefix)
    with pytest.raises(ValueError):
        publish_catalog_files(client, 'bucket', {'RUNS.md': b''}, prefix)
    assert not client.lists and not client.puts
