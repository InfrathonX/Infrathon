import copy
import hashlib
import json

from sforge.pyromind.asset_catalog import asset_id
from sforge.pyromind.inventory import build_inventory


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def binding(*, private=False, identity=True, bucket='assets', content='a'):
    files = [{'path': 'weights.bin', 'size': 42, 'sha256': content * 64}]
    digest = asset_id({'files': files})
    scope = 'judge-private' if private else 'shared'
    result = {'asset_id': digest, 'files': files, 'manifest_sha256': digest,
              'mount_path': '/private' if private else '/model', 'read_only': True,
              'object_prefix': f'sforge-assets/v2/{scope}/models/{digest}/files'}
    if identity:
        result['storage_identity'] = {'endpoint': 'https://storage.invalid', 'bucket': bucket}
    if private:
        result.update(visibility='private', consumers=['judge'])
    return result


def profile(repo, name, item=None, *, private=False):
    config = {'schema_version': 1, 'name': name, 'task_id': name + '_task',
              'task_directory': 'tasks/' + name, 'private_bootstrap': 'deploy/bootstrap.py',
              'images_file': 'deploy/images.json',
              'resources': {'cpu': 4, 'memory': '8Gi', 'gpu': 1}}
    if item:
        field = 'private_asset_identities' if private else 'asset_identities'
        config[field] = {item['mount_path']: {
            'manifest_sha256': item['manifest_sha256'],
            'files_sha256': hashlib.sha256(json.dumps(item['files'], sort_keys=True,
                separators=(',', ':')).encode()).hexdigest()}}
        if private:
            config['private_asset_requirements'] = [{'mount_path': item['mount_path']}]
        else:
            config['required_mounts'] = [item['mount_path']]
    return write(repo / 'deploy/pyromind' / name / 'deployment.json', config)


def test_cross_task_reuse_and_selected_fields(tmp_path):
    repo, artifacts = tmp_path / 'repo', tmp_path / 'artifacts'
    item = binding()
    for name in ['one', 'two']:
        path = profile(repo, name, item)
        data = {'schema_version': 1, 'assets': [item], 'api_key': 'do-not-export-secret'}
        write(path.with_name('assets.json'), data)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*.json')}
    result = build_inventory(repo, artifacts)
    assert result['totals']['profiles'] == 2
    assert result['totals']['content_trees'] == result['totals']['locations'] == 1
    assert len(result['assets'][0]['consumers']) == 2
    assert result['totals']['logical_content_bytes'] == 42
    assert 'do-not-export-secret' not in json.dumps(result)
    assert result['remote_validation'] == 'not-performed'
    assert result == build_inventory(repo, artifacts)
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*.json')}


def test_storage_locations_remain_distinct_and_unknown_not_assumed_same(tmp_path):
    repo, artifacts = tmp_path / 'repo', tmp_path / 'artifacts'
    for name, item in [('one', binding()), ('two', binding(bucket='other')),
                       ('three', binding(identity=False)), ('four', binding(identity=False))]:
        path = profile(repo, name, item)
        write(path.with_name('assets.json'), {'schema_version': 1, 'assets': [item]})
    result = build_inventory(repo, artifacts)
    assert len(result['assets']) == 1
    assert len(result['assets'][0]['locations']) == 4
    assert len([x for x in result['issues'] if x['code'] == 'storage_identity_unspecified']) == 2


def test_scope_keeps_same_bytes_separate(tmp_path):
    repo, artifacts = tmp_path / 'repo', tmp_path / 'artifacts'
    for name, private in [('one', False), ('two', True)]:
        item = binding(private=private)
        path = profile(repo, name, item, private=private)
        write(path.with_name('judge-assets.json' if private else 'assets.json'),
              {'schema_version': 1, 'assets': [item]})
    result = build_inventory(repo, artifacts)
    assert len(result['assets']) == 2
    assert {x['scope'] for x in result['assets']} == {'shared', 'judge-private'}


def test_conflicting_sources_not_silently_selected(tmp_path):
    repo, artifacts = tmp_path / 'repo', tmp_path / 'artifacts'
    first, second = binding(), binding(bucket='second')
    path = profile(repo, 'one', first)
    write(path.with_name('assets.json'), {'schema_version': 1, 'assets': [first]})
    write(artifacts / 'assets/bindings/one/shared.json', {'schema_version': 1, 'assets': [second]})
    result = build_inventory(repo, artifacts)
    assert result['profiles'][0]['bindings']['shared']['status'] == 'ambiguous'
    assert result['assets'][0]['status'] == 'unresolved-declaration'
    assert any(x['code'] == 'ambiguous_binding_sources' for x in result['issues'])


def test_legacy_outer_binding_and_manifest_identity_are_preserved(tmp_path):
    repo, artifacts = tmp_path / 'repo', tmp_path / 'artifacts'
    item = binding()
    item['manifest_sha256'] = 'b' * 64
    item['object_prefix'] = 'sforge-assets/old-name/' + 'b' * 64 + '/files'
    profile(repo, 'one', item)
    write(artifacts / 'shared-assets/one.json', {'schema_version': 1, 'assets': [item]})
    result = build_inventory(repo, artifacts)
    asset = result['assets'][0]
    assert asset['asset_id'] != asset['locations'][0]['manifest_sha256']
    assert asset['locations'][0]['object_prefix'] == item['object_prefix']
    assert asset['status'] == 'bound-local-declaration'


def test_mismatch_and_credential_endpoint_report_without_leaking(tmp_path):
    repo, artifacts = tmp_path / 'repo', tmp_path / 'artifacts'
    original = binding()
    path = profile(repo, 'one', original)
    wrong = copy.deepcopy(original)
    wrong['storage_identity']['endpoint'] = 'https://user:very-secret@storage.invalid'
    write(path.with_name('assets.json'), {'schema_version': 1, 'assets': [wrong]})
    result = build_inventory(repo, artifacts)
    assert not result['assets']
    assert 'very-secret' not in json.dumps(result)
    assert any(x['code'] == 'binding_contract_mismatch' for x in result['issues'])
    wrong = binding(content='b')
    write(path.with_name('assets.json'), {'schema_version': 1, 'assets': [wrong]})
    assert build_inventory(repo, artifacts)['profiles'][0]['bindings']['shared']['status'] == 'invalid'


def test_restoration_receipt_prepared_not_bound_and_changed_receipt_rejected(tmp_path):
    repo, artifacts = tmp_path / 'repo', tmp_path / 'artifacts'
    item = binding(private=True)
    profile(repo, 'qvg', item, private=True)
    record = {'asset_id': item['asset_id'], 'files': item['files'],
              'data_prefix': item['object_prefix'], 'manifest_sha256': item['manifest_sha256'],
              'storage': item['storage_identity']}
    path = write(artifacts / 'storage-restoration-demo/receipts/model.json',
                 {'scope': 'judge-private', 'binding': item, 'record': record})
    state = {'jobs': {'model': {'status': 'ready', 'returncode': 0,
             'receipt_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}}
    write(path.parent.parent / 'upload-state.json', state)
    result = build_inventory(repo, artifacts)
    assert result['assets'][0]['status'] == 'prepared-not-bound'
    assert result['assets'][0]['preparations'][0]['requested_by_mount'] == ['qvg']
    assert result['profiles'][0]['bindings']['judge-private']['status'] == 'missing'
    assert result['totals']['prepared_not_bound'] == 1
    path.write_text(path.read_text() + '\n')
    changed = build_inventory(repo, artifacts)
    assert not changed['assets']
    assert any(x['code'] == 'preparation_receipt_not_committed_or_changed' for x in changed['issues'])


def test_unknown_files_are_not_scanned_and_no_resource_profile_is_valid(tmp_path):
    repo, artifacts = tmp_path / 'repo', tmp_path / 'artifacts'
    profile(repo, 'kernel')
    write(artifacts / 'unrelated.json', {'scope': 'private', 'secret': 'ignored'})
    result = build_inventory(repo, artifacts)
    assert result['totals']['content_trees'] == 0
    assert result['totals']['profiles'] == 1
    assert not result['issues']
    assert all(x['status'] == 'not-required' for x in result['profiles'][0]['bindings'].values())


def test_equal_binding_copies_are_not_ambiguous(tmp_path):
    repo, artifacts = tmp_path / 'repo', tmp_path / 'artifacts'
    item = binding()
    path = profile(repo, 'one', item)
    data = {'schema_version': 1, 'assets': [item]}
    write(path.with_name('assets.json'), data)
    write(artifacts / 'assets/bindings/one/shared.json', data)
    result = build_inventory(repo, artifacts)
    assert result['profiles'][0]['bindings']['shared']['status'] == 'valid-local-declaration'
    assert len(result['assets'][0]['locations']) == 1
    assert not result['issues']


def test_cross_scope_mount_overlap_invalidates_both_declarations(tmp_path):
    repo, artifacts = tmp_path / 'repo', tmp_path / 'artifacts'
    shared, hidden = binding(), binding(private=True)
    hidden['mount_path'] = shared['mount_path']
    path = profile(repo, 'one', shared)
    config = json.loads(path.read_text())
    config['private_asset_requirements'] = [{'mount_path': hidden['mount_path']}]
    config['private_asset_identities'] = config['asset_identities']
    write(path, config)
    write(path.with_name('assets.json'), {'schema_version': 1, 'assets': [shared]})
    write(path.with_name('judge-assets.json'), {'schema_version': 1, 'assets': [hidden]})
    result = build_inventory(repo, artifacts)
    assert any(x['code'] == 'public_private_mount_overlap' for x in result['issues'])
    assert all(x['status'] == 'unresolved-declaration' for x in result['assets'])
