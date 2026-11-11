"""Local task/resource index. Never contacts Storage or executes a bootstrap.

Bindings are declarations, receipts are historical preparation evidence, and
neither is a current remote integrity check or GPU qualification certificate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit

from sforge.pyromind.asset_catalog import asset_id
from sforge.pyromind.deployment import Deployment, validate_mount_separation


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _read(path):
    # Metadata only. In particular, do not scan arbitrary JSON or credentials.
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError('Inventory metadata exceeds limit')
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError('Inventory metadata must be an object')
    return value


def _storage(item):
    identity = item.get('storage_identity')
    if identity is None:
        return None
    if not isinstance(identity, dict):
        raise ValueError('Invalid storage identity')
    endpoint, bucket = identity.get('endpoint'), identity.get('bucket')
    if not isinstance(endpoint, str) or not isinstance(bucket, str) or not bucket:
        raise ValueError('Invalid storage identity')
    parsed = urlsplit(endpoint)
    if (parsed.scheme not in {'http', 'https'} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {'', '/'}):
        raise ValueError('Storage endpoint must not contain credentials or a path')
    return {'endpoint': parsed.scheme + '://' + parsed.netloc, 'bucket': bucket}


def _binding(item, scope, *, deployment_validated=False):
    identifier = asset_id({'schema_version': 1, 'files': item['files']})
    if item.get('asset_id', identifier) != identifier:
        raise ValueError('Declared asset ID differs from content tree')
    prefix = item['object_prefix']
    parts = prefix.split('/')
    if (not prefix.startswith('sforge-assets/') or '..' in parts
            or not prefix.endswith('/' + item['manifest_sha256'] + '/files')):
        raise ValueError('Invalid manifest-addressed location')
    storage_scope = parts[2] if prefix.startswith('sforge-assets/v2/') and len(parts)==6 else 'legacy'
    consumers=item.get('consumers', ['judge'] if scope=='judge-private' else ['worker','judge'])
    alias=(scope=='shared' and storage_scope=='judge-private')
    if prefix.startswith('sforge-assets/v2/') and (len(parts)!=6 or storage_scope!=scope):
        if (not deployment_validated or not alias or item.get('read_only_model_alias') is not True
                or item.get('storage_scope')!='judge-private' or item.get('role')!='model'
                or consumers!=['worker'] or item.get('visibility','public')!='public' or item.get('read_only') is not True):
            raise ValueError('Asset prefix scope differs from consumer scope')
    allowed=({frozenset({'judge'})} if scope=='judge-private' else
             {frozenset({'worker','judge'}),frozenset({'worker'})})
    if (not isinstance(consumers,list) or not all(isinstance(c,str) for c in consumers)
            or len(consumers)!=len(set(consumers)) or frozenset(consumers) not in allowed):
        raise ValueError('Invalid indexed consumer roles')
    identity = _storage(item)
    return {'asset_id': identifier, 'scope': scope, 'consumer_scope': scope, 'storage_scope':storage_scope,
            'kind': parts[3] if prefix.startswith('sforge-assets/v2/') else 'legacy',
            'bytes': sum(row['size'] for row in item['files']),
            'file_count': len(item['files']), 'object_prefix': prefix,
            'manifest_sha256': item['manifest_sha256'], 'storage_identity': identity,
            'mount_path': item['mount_path'], 'role': item.get('role'),
            'consumers': list(consumers)}


def build_inventory(repo: Path, artifact_root: Path) -> dict:
    """Index known local declarations without selecting ambiguous bindings.

    ``artifact_root`` is the workspace artifacts directory, not a single run.
    Calling this function reads local metadata only and creates no files.
    """
    repo, artifact_root = Path(repo).resolve(), Path(artifact_root).resolve()
    profiles, assets, issues, deployments = [], {}, [], {}

    def issue(code, *, path=None, profile=None, scope=None, mount_path=None, severity='error'):
        entry = {'code': code, 'severity': severity}
        for key, value in [('source', path), ('profile', profile), ('scope', scope),
                           ('mount_path', mount_path)]:
            if value is not None:
                entry[key] = str(value)
        issues.append(entry)

    def register(binding, source, *, consumer=None, preparation=None):
        key = binding['scope'] + ':' + binding['asset_id']
        asset = assets.setdefault(key, {
            'asset_key': key, 'asset_id': binding['asset_id'], 'scope': binding['scope'],
            'consumer_scope': binding['consumer_scope'], 'storage_scopes': [],
            'bytes': binding['bytes'], 'file_count': binding['file_count'],
            'kinds': [], 'locations': [], 'consumers': [], 'preparations': []})
        if binding['storage_scope'] not in asset['storage_scopes']:
            asset['storage_scopes'].append(binding['storage_scope'])
        if binding['kind'] not in asset['kinds']:
            asset['kinds'].append(binding['kind'])
        # Unknown identity is NOT silently equated to another source's bucket.
        location_key = {'storage_identity': binding['storage_identity'],
                        'object_prefix': binding['object_prefix'],
                        'manifest_sha256': binding['manifest_sha256']}
        if binding['storage_identity'] is None:
            location_key['unspecified_identity_source'] = str(source)
        location_id = _digest(location_key)
        location = next((x for x in asset['locations'] if x['location_id'] == location_id), None)
        if location is None:
            location = {**location_key, 'location_id': location_id,
                        'identity_status': 'known' if binding['storage_identity'] else 'unspecified',
                        'sources': []}
            asset['locations'].append(location)
        if str(source) not in location['sources']:
            location['sources'].append(str(source))
        if consumer is not None:
            item = {**consumer, 'location_id': location_id,
                    'mount_path': binding['mount_path'], 'role': binding['role'],
                    'consumers': binding['consumers']}
            if item not in asset['consumers']:
                asset['consumers'].append(item)
        if preparation is not None:
            asset['preparations'].append({**preparation, 'location_id': location_id,
                                          'mount_path': binding['mount_path']})
        return key

    for path in sorted((repo / 'deploy/pyromind').glob('*/deployment.json')):
        try:
            deployment = Deployment.load(path, repo)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            issue('invalid_deployment', path=path)
            continue
        name, config = deployment.config['name'], deployment.config
        deployments[name] = deployment
        profile = {'profile': name, 'task_id': config['task_id'],
                   'task_directory': config['task_directory'], 'deployment': str(path),
                   'bindings': {}, 'remote_validation': 'not-performed',
                   'gpu_qualification': 'not-evaluated'}
        profiles.append(profile)
        selected = {}
        for private, scope, filename in [(False, 'shared', 'assets.json'),
                                         (True, 'judge-private', 'judge-assets.json')]:
            candidates = [path.with_name(filename),
                          artifact_root / 'assets/bindings' / name / (scope + '.json')]
            if not private:
                candidates.append(artifact_root / 'shared-assets' / (name + '.json'))
            present = sorted({p.resolve() for p in candidates if p.is_file()})
            required = ([x['mount_path'] for x in config.get('private_asset_requirements', [])]
                        if private else config.get('required_mounts', []))
            entries, normalized = [], []
            for source in present:
                try:
                    # Reuse the launcher contract; do not execute readiness gates.
                    raw = (deployment.judge_asset_bindings(source) if private
                           else deployment.asset_bindings(source))
                    parsed = sorted((_binding(x, scope, deployment_validated=True) for x in raw), key=lambda x: x['mount_path'])
                    normalized.append(_digest(parsed))
                    entries.append((source, raw, parsed))
                except (OSError, ValueError, KeyError, TypeError, AttributeError):
                    issue('binding_contract_mismatch', path=source, profile=name, scope=scope)
            ambiguous = len(set(normalized)) > 1
            if ambiguous:
                issue('ambiguous_binding_sources', profile=name, scope=scope)
            valid = bool(entries) and len(entries) == len(present) and not ambiguous
            if not present and required:
                issue('missing_binding', profile=name, scope=scope)
            status = ('not-required' if not present and not required else
                      'missing' if not present else 'ambiguous' if ambiguous else
                      'valid-local-declaration' if valid else 'invalid')
            profile['bindings'][scope] = {'status': status, 'required_mounts': sorted(required),
                                          'sources': [str(p) for p in present], 'assets': []}
            if valid:
                selected[scope] = entries[0][1]
            for source, _, parsed in entries:
                for binding in parsed:
                    key = register(binding, source, consumer={
                        'profile': name, 'task_id': config['task_id'],
                        'binding_status': status, 'source': str(source)})
                    if key not in profile['bindings'][scope]['assets']:
                        profile['bindings'][scope]['assets'].append(key)
                    if binding['storage_identity'] is None:
                        issue('storage_identity_unspecified', path=source, profile=name,
                              scope=scope, mount_path=binding['mount_path'], severity='warning')
        if 'shared' in selected and 'judge-private' in selected:
            try:
                validate_mount_separation(selected['shared'], selected['judge-private'])
            except ValueError:
                issue('public_private_mount_overlap', profile=name)
                for section in profile['bindings'].values():
                    section['status'] = 'invalid'
                for asset in assets.values():
                    for consumer in asset['consumers']:
                        if consumer['profile'] == name:
                            consumer['binding_status'] = 'invalid'

    # Only documented preparation locations, never arbitrary workspace JSON.
    receipt_paths = set(artifact_root.glob('storage-restoration-*/receipts/*.json'))
    receipt_paths.update(artifact_root.glob('assets/receipts/*/*.json'))
    for path in sorted(receipt_paths):
        try:
            receipt = _read(path)
            scope = receipt['scope']
            if scope not in {'shared', 'judge-private'}:
                raise ValueError('Unsupported preparation scope')
            binding = _binding(receipt['binding'], scope)
            record = receipt['record']
            if (record['asset_id'] != binding['asset_id']
                    or record['data_prefix'] != binding['object_prefix']
                    or record['files'] != receipt['binding']['files']
                    or record['manifest_sha256'] != binding['manifest_sha256']
                    or record.get('storage') != binding['storage_identity']):
                raise ValueError('Preparation record differs from binding')
            actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            evidence = 'local-preparation-record-only'
            state_path = path.parent.parent / 'upload-state.json'
            if state_path.exists():
                job = _read(state_path)['jobs'].get(path.stem, {})
                if (job.get('status') != 'ready' or job.get('returncode') != 0
                        or job.get('receipt_sha256') != actual_sha):
                    issue('preparation_receipt_not_committed_or_changed', path=path, scope=scope)
                    continue
                evidence = 'matches-local-upload-completion-receipt'
            provenance = receipt.get('provenance', {})
            prepared = {'source': str(path), 'receipt_sha256': actual_sha, 'evidence': evidence,
                        'remote_validation': 'not-performed',
                        'provenance': {key: provenance[key] for key in
                            ['provider', 'repo_id', 'revision', 'format', 'file_profile']
                            if isinstance(provenance.get(key), str)},
                        'requested_by_mount': sorted(name for name, d in deployments.items()
                            if binding['mount_path'] in (
                                [x['mount_path'] for x in d.config.get('private_asset_requirements', [])]
                                if scope == 'judge-private' else d.config.get('required_mounts', [])))}
            register(binding, path, preparation=prepared)
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            issue('invalid_preparation_receipt', path=path)

    result_assets = [assets[key] for key in sorted(assets)]
    for asset in result_assets:
        asset['kinds'].sort();asset['storage_scopes'].sort()
        asset['locations'].sort(key=lambda x: x['location_id'])
        bound = any(x['binding_status'] == 'valid-local-declaration' for x in asset['consumers'])
        asset['status'] = ('bound-local-declaration' if bound else
                           'prepared-not-bound' if asset['preparations'] else 'unresolved-declaration')
        asset['remote_validation'] = 'not-performed'
    scopes = []
    for scope in ('shared', 'judge-private'):
        rows = [x for x in result_assets if x['scope'] == scope]
        scopes.append({'scope': scope, 'content_trees': len(rows),
                       'logical_content_bytes': sum(x['bytes'] for x in rows),
                       'locations': sum(len(x['locations']) for x in rows)})
    return {'schema_version': 1, 'kind': 'sforge-local-resource-inventory',
            'repo': str(repo), 'artifact_root': str(artifact_root),
            'validation': 'local-deployment-contract-and-receipt-metadata-only',
            'remote_validation': 'not-performed', 'gpu_qualification': 'not-evaluated',
            'scopes': scopes, 'profiles': profiles, 'assets': result_assets, 'issues': issues,
            'totals': {'profiles': len(profiles), 'content_trees': len(result_assets),
                       'logical_content_bytes': sum(x['bytes'] for x in result_assets),
                       'locations': sum(len(x['locations']) for x in result_assets),
                       'prepared_not_bound': sum(x['status'] == 'prepared-not-bound' for x in result_assets),
                       'issues': len(issues)},
            'limitations': [
                'Logical content bytes are not physical bucket usage: distinct locations may store copies.',
                'Consumer scopes can reference the same existing storage tree; model aliases do not prove a second copy.',
                'Missing storage identities stay source-scoped; equal prefixes do not prove equal buckets.',
                'Prepared receipts do not update deployment bindings or clear task launch blockers.',
                'A requested mount match is a hint, not proof that prepared content satisfies that task.',
                'No Storage API, object file SHA audit, GPU readiness check, or private bootstrap was run.']}
