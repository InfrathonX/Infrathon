"""Controller-owned shared snapshots; task roles never determine storage paths.

This adds a catalog over the existing immutable upload format. Existing trees can
be registered in place. It deliberately does not implement file-level CAS, GC,
private mount authorization, or a distributed task scheduler.
"""
from __future__ import annotations

import hashlib
import json
import re
from urllib.parse import urlsplit

from sforge.pyromind.assets import (
    AssetError, CHUNK_SIZE, PART_SIZE, _canonical, _ensure_bytes,
    _manifest_bytes, _relative_path, _remote_matches, prepare_asset,
)


def content_manifest(manifest):
    frozen, _ = _manifest_bytes(manifest)
    return {'schema_version': 1, 'files': frozen['files']}


def asset_id(manifest):
    return hashlib.sha256(_canonical(content_manifest(manifest))).hexdigest()


def read_json(client, bucket, key):
    """Bound metadata reads and avoid including storage errors/credentials."""
    try:
        stream = client.get_object(bucket, key)
    except Exception as error:
        if getattr(error, 'code', None) in ('NoSuchKey', 'NoSuchObject'):
            return None
        raise AssetError('Catalog read failed') from None
    try:
        data = bytearray()
        while True:
            chunk = stream.read(min(CHUNK_SIZE, PART_SIZE + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > PART_SIZE:
                raise AssetError('Catalog metadata exceeds limit')
        value = json.loads(data)
        if not isinstance(value, dict):
            raise ValueError()
        return value
    except Exception:
        raise AssetError('Invalid catalog metadata') from None
    finally:
        stream.close()
        stream.release_conn()


class AssetCatalog:
    def __init__(self, client, bucket, endpoint, *, scope='shared'):
        parsed = urlsplit(endpoint)
        if (parsed.scheme not in ('http', 'https') or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path not in ('', '/')):
            raise ValueError('Catalog requires a credential-free storage endpoint')
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,62}', scope):
            raise ValueError('Invalid asset scope')
        if not isinstance(bucket, str) or not bucket:
            raise ValueError('Catalog requires a bucket')
        self.client, self.bucket, self.scope = client, bucket, scope
        self.storage = {'endpoint': f'{parsed.scheme}://{parsed.netloc}', 'bucket': bucket}
        self.prefix = f'sforge-catalog/v1/{scope}'

    def _record_key(self, identifier):
        return f'{self.prefix}/assets/{identifier}.json'

    def _validate_record(self, record, expected_files, kind):
        if (record.get('schema_version') != 1 or record.get('asset_id') != asset_id(expected_files)
                or record.get('storage') != self.storage or record.get('kind') != kind
                or record.get('files') != content_manifest(expected_files)['files']):
            raise AssetError('Catalog identity differs from requested asset')
        prefix = _relative_path(record['data_prefix'])
        digest = record['manifest_sha256']
        if (not re.fullmatch(r'[a-f0-9]{64}', digest)
                or not prefix.startswith('sforge-assets/')
                or not prefix.endswith('/' + digest + '/files')):
            raise AssetError('Catalog asset location is not a manifest-addressed tree')

    def _verify(self, record):
        root = record['data_prefix'].removesuffix('/files')
        manifest = read_json(self.client, self.bucket, root + '/manifest.json')
        if manifest is None:
            raise AssetError('Catalog points to a missing manifest')
        frozen, encoded = _manifest_bytes(manifest)
        if (hashlib.sha256(encoded).hexdigest() != record['manifest_sha256']
                or frozen['files'] != record['files']):
            raise AssetError('Remote asset manifest differs from catalog')
        ready = {'schema_version': 1, 'manifest_sha256': record['manifest_sha256'],
                 'file_count': len(frozen['files']),
                 'total_bytes': sum(row['size'] for row in frozen['files'])}
        if read_json(self.client, self.bucket, root + '/ready.json') != ready:
            raise AssetError('Catalog asset is not committed')
        prefix = record['data_prefix'] + '/'
        expected_keys = {prefix + entry['path'] for entry in frozen['files']}
        observed = set()
        try:
            for item in self.client.list_objects(self.bucket, prefix=prefix, recursive=True):
                if item.object_name not in expected_keys:
                    raise AssetError('Unlisted object in shared asset tree')
                observed.add(item.object_name)
        except AssetError:
            raise
        except Exception:
            raise AssetError('Asset tree inventory failed') from None
        if observed != expected_keys:
            raise AssetError('Shared asset tree is incomplete')
        for entry in frozen['files']:
            if not _remote_matches(self.client, self.bucket,
                                   record['data_prefix'] + '/' + entry['path'],
                                   entry['size'], entry['sha256']):
                raise AssetError('Catalog asset file is missing')

    @staticmethod
    def _validate_provenance(provenance):
        allowed = {'provider', 'repo_id', 'revision', 'format', 'file_profile'}
        if (not isinstance(provenance, dict) or set(provenance) != allowed
                or any(not isinstance(v, str) or not v or len(v) > 300
                       or any(ord(c) < 32 for c in v) for v in provenance.values())):
            raise ValueError('Pin provider, repo_id, revision, format and file_profile')
        revision = provenance['revision']
        if provenance['provider'] == 'hf' and not re.fullmatch(r'[a-f0-9]{40}', revision):
            raise ValueError('Hugging Face assets require a full commit revision')

    def _source_reference(self, record, provenance):
        reference = {'schema_version': 1, 'source': provenance, 'asset_id': record['asset_id']}
        source_id = hashlib.sha256(_canonical(provenance)).hexdigest()
        _ensure_bytes(self.client, self.bucket,
                      f'{self.prefix}/sources/{source_id}.json', _canonical(reference))

    def ensure(self, manifest, *, provenance, kind='models', local_root=None, legacy_binding=None):
        """Reuse by content, or upload once; legacy registration moves no weights.

        Reuse still hashes remote bytes: saved storage is not a claim of zero read
        traffic. A ready tree is never silently repaired from a different source.
        """
        if kind not in ('models', 'datasets', 'fixtures', 'runtimes'):
            raise ValueError('Unsupported asset kind')
        self._validate_provenance(provenance)
        canonical = content_manifest(manifest)
        identifier = asset_id(manifest)
        source_id = hashlib.sha256(_canonical(provenance)).hexdigest()
        source_reference = read_json(self.client, self.bucket,
                                     f'{self.prefix}/sources/{source_id}.json')
        if source_reference is not None and source_reference != {
                'schema_version': 1, 'source': provenance, 'asset_id': identifier}:
            raise AssetError('Pinned source profile already refers to different content')
        record = read_json(self.client, self.bucket, self._record_key(identifier))
        uploaded = 0
        if record is None:
            if legacy_binding is not None:
                frozen, encoded = _manifest_bytes(manifest)
                digest = hashlib.sha256(encoded).hexdigest()
                if (legacy_binding.get('manifest_sha256') != digest
                        or content_manifest({'files': legacy_binding.get('files')})['files'] != frozen['files']
                        or legacy_binding.get('read_only') is not True):
                    raise AssetError('Legacy binding differs from its manifest')
                location = {'data_prefix': legacy_binding['object_prefix'], 'manifest_sha256': digest}
            else:
                if local_root is None:
                    raise AssetError('Asset absent from catalog; a local source is required')
                location = prepare_asset(self.client, self.bucket, local_root, canonical,
                                         f'sforge-assets/v2/{self.scope}/{kind}')
                uploaded = location['uploaded_files']
            record = {'schema_version': 1, 'asset_id': identifier, 'kind': kind,
                      'storage': self.storage, 'files': canonical['files'],
                      'data_prefix': location['data_prefix'],
                      'manifest_sha256': location['manifest_sha256']}
            self._validate_record(record, canonical, kind)
            self._verify(record)
            _ensure_bytes(self.client, self.bucket, self._record_key(identifier), _canonical(record))
        else:
            self._validate_record(record, canonical, kind)
            self._verify(record)
        self._source_reference(record, provenance)
        return {**record, 'uploaded_files': uploaded,
                'reused_files': len(record['files']) - uploaded}

    @staticmethod
    def binding(record, mount_path, role=None):
        # Schema 1 is intentionally compatible with the existing launcher.
        result = {'mount_path': mount_path, 'object_prefix': record['data_prefix'],
                  'manifest_sha256': record['manifest_sha256'], 'read_only': True,
                  'files': record['files'], 'asset_id': record['asset_id'],
                  'storage_identity': record['storage']}
        if role is not None:
            result['role'] = role
        return result
