"""E-only metadata/role-routing regression; no SDK, Storage or sandbox calls."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import sforge.pyromind.deployment as D
import sforge.pyromind.runner as R
import sforge.pyromind.qualification as Q
import sforge.pyromind.inventory as I


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))
    return path


def row(name, mount, *, private=False):
    files = [{'path': name + '.bin', 'size': 7, 'sha256': hashlib.sha256(name.encode()).hexdigest()}]
    digest = I.asset_id({'files': files})
    kind = 'models' if name == 'model' else 'datasets'
    scope = 'judge-private' if private else 'shared'
    result = dict(asset_id=digest, manifest_sha256=digest, files=files,
                  mount_path=mount, object_prefix=f'sforge-assets/v2/{scope}/{kind}/{digest}/files',
                  read_only=True, role='model' if name == 'model' else name,
                  storage_identity={'endpoint': 'https://storage.invalid', 'bucket': 'assets'})
    if private:
        result.update(visibility='private', consumers=['judge'])
    return result


def pin(item):
    result = {key: copy.deepcopy(item[key]) for key in ('manifest_sha256', 'object_prefix',
              'storage_identity', 'role', 'consumers', 'storage_scope', 'read_only_model_alias') if key in item}
    result['files_sha256'] = hashlib.sha256(canonical(sorted(item['files'], key=lambda x: x['path']))).hexdigest()
    return result


def example(tmp_path):
    public = row('performance', '/workspace/benchmark/performance-prompts')
    hidden = [row('private-performance', '/opt/judge/kernel-prompts', private=True),
              row('model', '/opt/judge/model/revision', private=True),
              row('terminal-quality', '/opt/judge/quality-prompts', private=True)]
    alias = copy.deepcopy(hidden[1])
    alias.update(mount_path='/workspace/benchmark/model/revision', visibility='public',
                 consumers=['worker'], storage_scope='judge-private', read_only_model_alias=True)
    shared = [public, alias]
    config = dict(schema_version=1, name='example', task_id='example_task', task_directory='tasks/example',
                  private_bootstrap='deploy/bootstrap.py', images_file='deploy/images.json',
                  resources={'cpu': 2, 'memory': '4Gi', 'gpu': 0}, execution_kind='cpu',
                  runtime_expectations={'python': '3.12', 'architecture': 'x86_64'},
                  required_mounts=[x['mount_path'] for x in shared],
                  private_asset_requirements=[{'mount_path': x['mount_path']} for x in hidden],
                  asset_identities={x['mount_path']: pin(x) for x in shared},
                  private_asset_identities={x['mount_path']: pin(x) for x in hidden})
    deployment = D.Deployment(tmp_path / 'deployment.json', tmp_path, config)
    return deployment, shared, hidden


def validate(deployment, tmp_path, shared, hidden):
    accepted = deployment.asset_bindings(write(tmp_path / 'shared.json', {'schema_version': 1, 'assets': shared}))
    private = deployment.judge_asset_bindings(write(tmp_path / 'hidden.json', {'schema_version': 1, 'assets': hidden}))
    D.validate_mount_separation(accepted, private)
    return accepted, private


def test_exact_worker_two_judge_four_and_single_original_model_location(tmp_path):
    deployment, shared, hidden = example(tmp_path)
    accepted, private = validate(deployment, tmp_path, shared, hidden)
    worker = R.asset_mounts_for_role(accepted, 'worker', SimpleNamespace)
    judge = R.asset_mounts_for_role(accepted + private, 'judge', SimpleNamespace)
    assert len(worker) == 2 and len(judge) == 4
    assert {x.mount_path for x in worker} == {x['mount_path'] for x in shared}
    assert {x.mount_path for x in judge} == {shared[0]['mount_path'], *(x['mount_path'] for x in hidden)}
    assert all(x.read_only is True for x in worker + judge)
    assert worker[1].host_path == judge[2].host_path
    assert R.asset_bindings_for_role(accepted, 'worker')[1] is accepted[1]
    assert all('/workspace/benchmark/model/' not in x.mount_path for x in judge)
    assert all('/opt/judge/' not in x.mount_path for x in worker)


@pytest.mark.parametrize('field', ['consumers', 'read_only_model_alias', 'storage_scope', 'role',
                                   'object_prefix', 'files_sha256', 'manifest_sha256', 'storage_identity'])
def test_model_alias_requires_every_exact_deployment_pin(tmp_path, field):
    deployment, shared, hidden = example(tmp_path)
    del deployment.config['asset_identities'][shared[1]['mount_path']][field]
    with pytest.raises((ValueError, KeyError)):
        validate(deployment, tmp_path, shared, hidden)


@pytest.mark.parametrize('field,value', [
    ('consumers', ['worker', 'judge']), ('consumers', ['judge']), ('consumers', ['worker', 'worker']),
    ('visibility', 'private'), ('read_only_model_alias', False), ('read_only_model_alias', 1),
    ('storage_scope', 'shared'), ('role', 'terminal-quality'), ('read_only', False),
    ('manifest_sha256', 'c' * 64), ('storage_identity', {'endpoint': 'https://other.invalid', 'bucket': 'assets'}),
])
def test_alias_row_drift_or_extra_judge_consumer_is_rejected(tmp_path, field, value):
    deployment, shared, hidden = example(tmp_path)
    shared[1][field] = value
    with pytest.raises(ValueError):
        validate(deployment, tmp_path, shared, hidden)


def test_unpinned_worker_only_shared_asset_is_rejected(tmp_path):
    deployment, shared, hidden = example(tmp_path)
    shared[0]['consumers'] = ['worker']
    with pytest.raises(ValueError, match='explicitly pinned'):
        validate(deployment, tmp_path, shared, hidden)
    deployment.config['asset_identities'][shared[0]['mount_path']]['consumers'] = ['worker']
    accepted, _ = validate(deployment, tmp_path, shared, hidden)
    assert R.asset_bindings_for_role(accepted, 'judge') == []


@pytest.mark.parametrize('attack', ['qa-tree', 'qa-bytes', 'private-worker', 'shared-marker', 'nonmodel-prefix'])
def test_qa_cannot_gain_worker_access_by_relabeling_or_scope_markers(tmp_path, attack):
    deployment, shared, hidden = example(tmp_path)
    if attack == 'qa-tree':
        replacement = copy.deepcopy(hidden[2])
        replacement.update(mount_path=shared[1]['mount_path'], visibility='public', consumers=['worker'],
                           role='model', storage_scope='judge-private', read_only_model_alias=True)
        shared[1] = replacement
    elif attack == 'qa-bytes':
        shared[1]['files'] = copy.deepcopy(hidden[2]['files'])
    elif attack == 'private-worker':
        hidden[2]['consumers'] = ['worker']
    elif attack == 'shared-marker':
        shared[0]['read_only_model_alias'] = True
    else:
        shared[1]['object_prefix'] = shared[1]['object_prefix'].replace('/models/', '/datasets/')
        deployment.config['asset_identities'][shared[1]['mount_path']]['object_prefix'] = shared[1]['object_prefix']
    with pytest.raises(ValueError):
        validate(deployment, tmp_path, shared, hidden)


def test_default_public_two_roles_and_private_judge_are_unchanged(tmp_path):
    deployment, shared, hidden = example(tmp_path)
    shared = shared[:1]
    deployment.config['required_mounts'] = [shared[0]['mount_path']]
    deployment.config['asset_identities'] = {shared[0]['mount_path']: pin(shared[0])}
    accepted, private = validate(deployment, tmp_path, shared, hidden)
    assert R.asset_bindings_for_role(accepted, 'worker') == accepted
    assert R.asset_bindings_for_role(accepted, 'judge') == accepted
    assert R.asset_bindings_for_role(private, 'worker') == []
    assert R.asset_bindings_for_role(private, 'judge') == private


@pytest.mark.parametrize('consumers', [None, ['worker', 'bad'], ['worker', 'worker'], 'worker', [0]])
def test_routing_rejects_malformed_consumers(consumers):
    with pytest.raises(ValueError):
        R.asset_bindings_for_role([{'consumers': consumers}], 'judge')


def test_judge_actual_runtime_asset_validation_excludes_worker_model(tmp_path, monkeypatch):
    deployment, shared, hidden = example(tmp_path)
    shared, hidden = validate(deployment, tmp_path, shared, hidden)
    observed = []
    backend = SimpleNamespace(cleanup_container=lambda *args: None,
                              _event=lambda *args, **kwargs: None)
    module = SimpleNamespace(install_private_judge=lambda *args: {'files': []})
    monkeypatch.setattr(R, 'configure_runtime_cleanup', lambda *args, **kwargs: None)
    monkeypatch.setattr(R, 'checked', lambda *args, **kwargs: 'cpu-runtime')
    monkeypatch.setattr(R, 'verify_asset_mounts', lambda backend, handle, rows, path, **kw:
                        observed.append((rows, kw.get('private', False))))
    R.configure_judge_runtime(backend, deployment, module, tmp_path, tmp_path, shared, {'files': []}, hidden)
    backend.on_started(SimpleNamespace(id='cpu-fixture'))
    assert observed == [([shared[0]], False), (hidden, True)]


def actual_routing_block(module, function, first_assignment, last_call):
    """Execute the actual side-effect-free routing segment, not a test copy."""
    tree = ast.parse(Path(module.__file__).read_text())
    fn = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == function)
    start = next(i for i, x in enumerate(fn.body) if isinstance(x, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == first_assignment for t in x.targets))
    end = next(i for i, x in enumerate(fn.body[start:], start) if isinstance(x, ast.Expr)
               and isinstance(x.value, ast.Call) and isinstance(x.value.func, ast.Name)
               and x.value.func.id == last_call)
    return compile(ast.fix_missing_locations(ast.Module(body=fn.body[start:end + 1], type_ignores=[])),
                   module.__file__, 'exec')


@pytest.mark.parametrize('qualification', [False, True])
def test_actual_runner_and_qualification_backend_requests_receive_only_their_role(tmp_path, qualification):
    deployment, shared, hidden = example(tmp_path)
    shared, hidden = validate(deployment, tmp_path, shared, hidden)
    calls = []
    def backend(**kw):
        result = SimpleNamespace(**kw)
        calls.append(result)
        return result
    def configured(*args, **kwargs):
        assert args[5] == [shared[0]] and args[7] == hidden
    env = dict(bindings=shared, private_bindings=hidden, VolumeMount=SimpleNamespace,
               PyroMindBackend=backend, resources=deployment.resources_for_role('judge'),
               worker_resources=deployment.resources_for_role('worker'),
               judge_resources=deployment.resources_for_role('judge'), deployment=deployment,
               output=tmp_path, events=tmp_path / 'events.jsonl', prefix='own/artifacts', image_map={},
               native_spec=None, private_module=SimpleNamespace(),
               task_root=tmp_path, private_metadata={}, run_id='q-test', sandbox_name_prefix='own-test',
               task=SimpleNamespace(judge=SimpleNamespace(eval_timeout=100)),
               threading=SimpleNamespace(Event=lambda: object()),
               guard_judge_backend=lambda *args, **kw: None, configure_judge_runtime=configured,
               asset_bindings_for_role=R.asset_bindings_for_role, asset_mounts_for_role=R.asset_mounts_for_role)
    module, fn, first = (Q, '_qualify_task', 'judge_bindings') if qualification else (R, '_run_task', 'worker_bindings')
    exec(actual_routing_block(module, fn, first, 'configure_judge_runtime'), env)
    judge = calls[-1]
    assert len(judge.volume_mounts) == 4
    assert {x.mount_path for x in judge.volume_mounts} == {shared[0]['mount_path'], *(x['mount_path'] for x in hidden)}
    if not qualification:
        worker = calls[0]
        # /shared is the separate existing candidate artifact mirror, not a model alias.
        assert len(worker.volume_mounts) == 3
        assert {x.mount_path for x in worker.volume_mounts} == {*(x['mount_path'] for x in shared), '/shared'}


def test_inventory_records_actual_storage_scope_only_after_deployment_validation(tmp_path):
    repo, artifacts = tmp_path / 'repo', tmp_path / 'artifacts'
    deployment, shared, hidden = example(repo)
    config_path = write(repo / 'deploy/pyromind/example/deployment.json', deployment.config)
    write(config_path.with_name('assets.json'), {'schema_version': 1, 'assets': shared})
    write(config_path.with_name('judge-assets.json'), {'schema_version': 1, 'assets': hidden})
    inventory = I.build_inventory(repo, artifacts)
    assert not inventory['issues']
    alias = next(x for x in inventory['assets'] if x['scope'] == 'shared' and x['asset_id'] == shared[1]['asset_id'])
    assert alias['consumer_scope'] == 'shared' and alias['storage_scopes'] == ['judge-private']
    assert alias['consumers'][0]['consumers'] == ['worker']
    assert alias['locations'][0]['object_prefix'] == hidden[1]['object_prefix']
    assert inventory['remote_validation'] == 'not-performed'
    with pytest.raises(ValueError, match='consumer scope'):
        I._binding(shared[1], 'shared')
    del deployment.config['asset_identities'][shared[1]['mount_path']]['consumers']
    write(config_path, deployment.config)
    invalid = I.build_inventory(repo, artifacts)
    assert invalid['profiles'][0]['bindings']['shared']['status'] == 'invalid'
    assert not any(x['scope'] == 'shared' for x in invalid['assets'])
    assert any(x['code'] == 'binding_contract_mismatch' for x in invalid['issues'])
