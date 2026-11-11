import copy
import hashlib
from types import SimpleNamespace

import pytest

pytest.importorskip("minio", reason="needs the pyromind extra (uv sync --extra pyromind)")
from sforge.pyromind.asset_catalog import AssetCatalog, asset_id
from sforge.pyromind.assets import AssetError, prepare_asset
from test_pyromind_assets import MemoryS3 as BaseMemoryS3, fixture_manifest


class MemoryS3(BaseMemoryS3):
    def list_objects(self, bucket, *, prefix, recursive):
        return (SimpleNamespace(object_name=key) for key in self.objects if key.startswith(prefix))


SOURCE = {'provider': 'hf', 'repo_id': 'org/model', 'revision': 'a' * 40,
          'format': 'safetensors', 'file_profile': 'runtime-v1'}


def catalog(client):
    return AssetCatalog(client, 'assets-bucket', 'https://storage.example.invalid')


def test_cross_task_reuse_ignores_role_and_manifest_annotation(tmp_path):
    client = MemoryS3()
    manifest = fixture_manifest(tmp_path)
    first = catalog(client).ensure(manifest, local_root=tmp_path, provenance=SOURCE)
    writes = list(client.writes)
    second = catalog(client).ensure({**manifest, 'role': 'new-task-teacher'}, provenance=SOURCE)
    assert client.writes == writes
    assert second['uploaded_files'] == 0 and second['reused_files'] == 2
    assert second['data_prefix'] == first['data_prefix']
    assert catalog(client).binding(first, '/student', 'student')['object_prefix'] == \
        catalog(client).binding(second, '/teacher', 'teacher')['object_prefix']


def test_register_legacy_without_moving_any_weight(tmp_path):
    client = MemoryS3()
    manifest = fixture_manifest(tmp_path)
    manifest['files'].sort(key=lambda row: row['path'])
    legacy = prepare_asset(client, 'assets-bucket', tmp_path, manifest, 'sforge-assets/old-role')
    binding = {'object_prefix': legacy['data_prefix'], 'manifest_sha256': legacy['manifest_sha256'],
               'read_only': True, 'files': manifest['files']}
    count = len(client.writes)
    result = catalog(client).ensure(manifest, provenance=SOURCE, legacy_binding=binding)
    assert result['data_prefix'] == legacy['data_prefix']
    assert result['manifest_sha256'] == legacy['manifest_sha256']
    assert result['asset_id'] == asset_id(manifest)
    assert result['uploaded_files'] == 0
    assert all(key.startswith('sforge-catalog/') for key in client.writes[count:])


def test_corrupt_ready_asset_fails_reuse_without_overwrite(tmp_path):
    client = MemoryS3()
    manifest = fixture_manifest(tmp_path)
    result = catalog(client).ensure(manifest, local_root=tmp_path, provenance=SOURCE)
    key = result['data_prefix'] + '/weights.bin'
    client.objects[key] = b'corrupt'
    writes = list(client.writes)
    with pytest.raises(AssetError):
        catalog(client).ensure(manifest, local_root=tmp_path, provenance=SOURCE)
    assert client.writes == writes


def test_different_revision_keeps_old_snapshot(tmp_path):
    client = MemoryS3()
    first = fixture_manifest(tmp_path)
    old = catalog(client).ensure(first, local_root=tmp_path, provenance=SOURCE)
    second = fixture_manifest(tmp_path, {'weights.bin': b'new weights'})
    new = catalog(client).ensure(second, local_root=tmp_path,
                                 provenance={**SOURCE, 'revision': 'b' * 40})
    assert old['data_prefix'] != new['data_prefix']
    assert client.objects[old['data_prefix'] + '/weights.bin'] == b'weights'


def test_mutable_hf_revision_rejected_before_upload(tmp_path):
    client = MemoryS3()
    with pytest.raises(ValueError, match='full commit'):
        catalog(client).ensure(fixture_manifest(tmp_path), local_root=tmp_path,
                               provenance={**SOURCE, 'revision': 'main'})
    assert not client.writes


def test_same_source_profile_cannot_be_rebound_to_changed_bytes(tmp_path):
    client = MemoryS3()
    catalog(client).ensure(fixture_manifest(tmp_path), local_root=tmp_path, provenance=SOURCE)
    writes = list(client.writes)
    with pytest.raises(AssetError):
        catalog(client).ensure(fixture_manifest(tmp_path, {'weights.bin': b'changed'}),
                               local_root=tmp_path, provenance=SOURCE)
    assert client.writes == writes


def test_storage_endpoint_cannot_contain_credentials():
    with pytest.raises(ValueError, match='credential-free'):
        AssetCatalog(MemoryS3(), 'bucket', 'https://user:secret@storage.example.invalid')


def test_unlisted_file_cannot_be_exposed_by_reusing_shared_tree(tmp_path):
    client = MemoryS3()
    manifest = fixture_manifest(tmp_path)
    record = catalog(client).ensure(manifest, local_root=tmp_path, provenance=SOURCE)
    client.objects[record['data_prefix'] + '/unexpected-private-file'] = b'private'
    with pytest.raises(AssetError, match='Unlisted'):
        catalog(client).ensure(manifest, provenance=SOURCE)
