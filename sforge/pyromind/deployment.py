"""Trusted deployment profiles: task identity, resources and read-only assets.

Task contracts remain in tasks/. Platform choices live in deploy/pyromind/.
Profiles are controller-owned configuration, never accepted from a candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


def whole_gibibytes(value):
    """A Sandbox memory request: a whole, positive number of Gi such as ``32Gi``.

    CPU and memory are independent: since 2026-10-04 the account may request
    any CPU:memory ratio (deploy/pyromind/RESOURCES.md)."""
    return isinstance(value, str) and re.fullmatch(r'[1-9][0-9]*Gi', value) is not None


def qualification_contract_digest(config):
    """Bind qualification evidence to executable configuration, excluding status prose."""
    descriptive = {'requires_gpu_qualification', 'requires_qualification', 'evaluation_description', 'validation_gaps',
                   'preparation_status', 'qualification_status', 'runtime_readiness', 'readiness_blockers'}
    payload = {key: value for key, value in config.items() if key not in descriptive}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def private_payload_identity(metadata):
    """Identify installed private inputs independently of controller placement.

    Bootstrap still checks each source file before returning this metadata.
    Absolute host source names are transport provenance; destination, bytes,
    permissions, relative task sources and other bootstrap fields stay bound.
    The original unmodified metadata digest is retained in run evidence.
    """
    payload = dict(metadata)
    files = []
    for entry in metadata['files']:
        item = dict(entry)
        if isinstance(item.get('source'), str) and PurePosixPath(item['source']).is_absolute():
            item.pop('source')
        files.append(item)
    payload['files'] = files
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                       allow_nan=False).encode()).hexdigest()
    return {'schema_version': 1, 'sha256': digest}


def validate_mount_separation(public, private):
    for shared in public:
        left = PurePosixPath(shared['mount_path'])
        for hidden in private:
            right = PurePosixPath(hidden['mount_path'])
            if left == right or left in right.parents or right in left.parents:
                raise ValueError('Public and private Judge mounts must not overlap')


@dataclass(frozen=True)
class Deployment:
    path: Path
    repo: Path
    config: dict

    @classmethod
    def load(cls, path: Path, repo: Path):
        config = json.loads(path.read_text())
        if config.get('schema_version') != 1:
            raise ValueError('Unsupported deployment schema')
        if not re.fullmatch(r'[a-z][a-z0-9-]{0,24}', config.get('name', '')):
            raise ValueError('Invalid deployment name')
        for name in ('task_directory', 'private_bootstrap', 'images_file'):
            value = Path(config[name])
            if value.is_absolute() or '..' in value.parts:
                raise ValueError('Deployment paths must be relative to the repository')
        deployment = cls(path.resolve(), repo.resolve(), config)
        deployment.evaluation_round_id('final')
        deployment.sandbox_ready_timeout_seconds
        for role in ('worker', 'judge'):
            resources = deployment.resources_for_role(role)
            kind = deployment.execution_kind_for_role(role)
            if (not isinstance(resources, dict)
                    or type(resources.get('gpu')) is not int
                    or (resources['gpu'] != 0 if kind == 'cpu' else resources['gpu'] < 1)
                    or type(resources.get('cpu')) is not int or resources['cpu'] < 1
                    or not whole_gibibytes(resources.get('memory'))):
                raise ValueError('Invalid deployment resources for ' + role)
            deployment.runtime_expectations(role)
            environment = config.get(role + '_environment', {})
            if (not isinstance(environment, dict) or any(
                    not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', key)
                    or not isinstance(value, str) or '\x00' in value
                    for key, value in environment.items())):
                raise ValueError('Invalid deployment environment')
        return deployment

    def local_path(self, key):
        return self.repo / self.config[key]

    @property
    def resources(self):
        return self.config['resources']

    def resources_for_role(self, role):
        """Use a complete role override, or the legacy shared resource mapping."""
        if role not in ('worker', 'judge'):
            raise ValueError('Invalid resource role')
        return self.config.get(role + '_resources', self.resources)

    def execution_kind_for_role(self, role):
        """GPU remains the default; CPU allocations require an explicit choice."""
        if role not in ('worker', 'judge'):
            raise ValueError('Invalid runtime role')
        kind = self.config.get('execution_kind', 'gpu')
        if isinstance(kind, dict):
            if set(kind) != {'worker', 'judge'}:
                raise ValueError('Execution kinds must specify worker and judge')
            kind = kind[role]
        if kind not in ('cpu', 'gpu'):
            raise ValueError('Execution kind must be cpu or gpu')
        return kind

    @property
    def qualification_receipt_path(self):
        legacy_gpu = all(self.execution_kind_for_role(role) == 'gpu' for role in ('worker', 'judge'))
        return self.path.with_name('gpu-qualification.json' if legacy_gpu else 'qualification.json')

    @property
    def sandbox_ready_timeout_seconds(self):
        """Bound platform scheduling/startup independently from native evaluation."""
        value = self.config.get('sandbox_ready_timeout_seconds', 900)
        if type(value) is not int or not 60 <= value <= 3600:
            raise ValueError('Sandbox startup timeout must be an integer from 60 to 3600 seconds')
        return value

    def assert_launch_ready(self, *, preflight=False):
        """Build status must not silently authorize an incomplete evaluation protocol.

        Legacy qualified FP8/GKD profiles retain their behavior. New preparation
        profiles explicitly block Agent runs until runtime qualification is recorded;
        unresolved private/model contracts also block preflight allocations.
        """
        blockers = self.config.get('execution_blockers', [])
        if (not isinstance(blockers, list)
                or any(not isinstance(item, str) or not item for item in blockers)):
            raise ValueError('Invalid deployment execution blockers')
        if blockers:
            raise ValueError('Deployment is build-only: ' + '; '.join(blockers))
        if not preflight and (self.config.get('requires_gpu_qualification', False)
                              or self.config.get('requires_qualification', False)):
            label = 'GPU qualification' if all(self.execution_kind_for_role(role) == 'gpu'
                                             for role in ('worker', 'judge')) else 'qualification'
            raise ValueError('Deployment requires recorded original-protocol ' + label + ' before an Agent run')
        if not preflight and any(key in self.config for key in ('requires_gpu_qualification', 'requires_qualification')):
            self.verify_gpu_qualification_receipt()

    def verify_gpu_qualification_receipt(self):
        """A later image, resource or private-payload change invalidates old evidence."""
        path = self.qualification_receipt_path
        try:
            receipt = json.loads(path.read_text())
            images = json.loads(self.local_path('images_file').read_text())
            current_private = self.bootstrap().private_payload_metadata(self.local_path('task_directory'))
            private_sha = hashlib.sha256(json.dumps(current_private, sort_keys=True).encode()).hexdigest()
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError('Qualification receipt or its pinned inputs are unavailable; requalify this deployment') from exc
        if 'private_payload_identity' in receipt:
            private_matches = receipt['private_payload_identity'] == private_payload_identity(current_private)
        else:
            private_matches = receipt.get('private_payload_manifest_sha256') == private_sha
        role_resources = {role: self.resources_for_role(role) for role in ('worker', 'judge')}
        resources_match = receipt.get('resources_each_role') == role_resources
        # Preserve existing GPU receipts that record one shared mapping.
        if all(resources == self.resources for resources in role_resources.values()):
            resources_match = resources_match or receipt.get('resources_each_role') == self.resources
        if (receipt.get('status') != 'passed' or receipt.get('task_id') != self.config['task_id']
                or receipt.get('images') != images
                or not resources_match
                or receipt.get('runtime_expectations') != self.config.get('runtime_expectations')
                or receipt.get('runtime_contract_sha256') != qualification_contract_digest(self.config)
                or not private_matches):
            raise ValueError('Qualification evidence is stale; requalify the current images, configuration and private payload')

    def evaluation_round_id(self, mode):
        """Use the task's native final selector; interactive IDs stay server-owned."""
        if mode not in {'interactive', 'final'}:
            raise ValueError('Unknown evaluation phase')
        value = 'agent-1' if mode == 'interactive' else self.config.get('final_round_id', 'manual-1')
        if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value):
            raise ValueError('Invalid native final round ID')
        return value

    def runtime_expectations(self, role='worker'):
        if role not in ('worker', 'judge'):
            raise ValueError('Invalid runtime role')
        cpu = self.execution_kind_for_role(role) == 'cpu'
        expected = self.config.get('runtime_expectations', {} if cpu else {
            'torch': '2.8.0+cu129', 'triton': '3.4.0', 'cuda': '12.9'})
        if not isinstance(expected, dict):
            raise ValueError('Runtime expectations must be a mapping')
        if set(expected) == {'worker', 'judge'}:
            expected = expected[role]
        if cpu:
            if (not isinstance(expected, dict) or set(expected) != {'python', 'architecture'}
                    or not isinstance(expected.get('python'), str)
                    or not re.fullmatch(r'3\.[0-9]+(?:\.[0-9]+)?', expected['python'])
                    or expected.get('architecture') not in ('x86_64', 'aarch64')):
                raise ValueError('Pin Python and architecture for each CPU runtime')
            return dict(expected)
        if (not isinstance(expected, dict) or set(expected) != {'torch', 'triton', 'cuda'}
                or any(not isinstance(value, str) or not value or len(value) > 80
                       for value in expected.values())):
            raise ValueError('Pin torch, triton and cuda for each runtime')
        return dict(expected)

    def runtime_probe_command(self, role='worker'):
        """Probe the selected role ABI as the unprivileged user."""
        if self.execution_kind_for_role(role) == 'cpu':
            program = '''import json,platform,sys
expected=json.loads(sys.argv[1]); resources=json.loads(sys.argv[2])
version=platform.python_version()
actual={'python':version,'architecture':platform.machine()}
assert '.'.join(version.split('.')[:len(expected['python'].split('.'))]) == expected['python'], 'Python version mismatch'
assert actual['architecture'] == expected['architecture'], 'CPU architecture mismatch'
assert resources['gpu'] == 0, 'CPU runtime must explicitly request no GPUs'
print(json.dumps({'runtime':actual,'gpu_count':0,'execution_kind':'cpu','runtime_verified':True},sort_keys=True))
'''
            return ['runuser', '-u', 'agent', '--', 'python3', '-c', program,
                    json.dumps(self.runtime_expectations(role)), json.dumps(self.resources_for_role(role))]
        program = '''import json,sys,torch,triton
expected=json.loads(sys.argv[1]); resources=json.loads(sys.argv[2])
actual={'torch':str(torch.__version__),'triton':str(triton.__version__),'cuda':str(torch.version.cuda)}
assert actual == expected, 'Runtime differs from pinned deployment: '+repr(actual)
assert torch.cuda.device_count() == resources['gpu'], 'GPU count mismatch'
for device in range(resources['gpu']):
    assert resources['gpu_card'] in torch.cuda.get_device_name(device), 'GPU type mismatch'
    x=torch.ones(8,device='cuda:'+str(device))
    assert x.sum().item()==8, 'CUDA compute failed'
print(json.dumps({'runtime':actual,'gpu_count':resources['gpu'],'compute_verified':True},sort_keys=True))
'''
        return ['runuser', '-u', 'agent', '--', 'python3', '-c', program,
                json.dumps(self.runtime_expectations(role)), json.dumps(self.resources_for_role(role))]

    def runtime_probe_matches(self, probe, role):
        """Check persisted ABI evidence, including CPU pins without importing torch."""
        expected = self.runtime_expectations(role)
        resources = self.resources_for_role(role)
        if (not isinstance(probe, dict) or type(probe.get('gpu_count')) is not int
                or probe['gpu_count'] != resources['gpu']):
            return False
        actual = probe.get('runtime')
        if self.execution_kind_for_role(role) == 'cpu':
            if (probe.get('execution_kind') != 'cpu' or probe.get('runtime_verified') is not True
                    or not isinstance(actual, dict) or set(actual) != {'python', 'architecture'}
                    or not isinstance(actual['python'], str)
                    or not re.fullmatch(r'3\.[0-9]+\.[0-9]+', actual['python'])):
                return False
            version = '.'.join(actual['python'].split('.')[:len(expected['python'].split('.'))])
            return version == expected['python'] and actual['architecture'] == expected['architecture']
        return probe.get('compute_verified') is True and actual == expected

    def bootstrap(self):
        spec = importlib.util.spec_from_file_location(
            'sforge_bootstrap_' + self.config['name'], self.local_path('private_bootstrap'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def asset_bindings(self, path: Path | None):
        return self._asset_bindings(path, private=False)

    def judge_asset_bindings(self, path: Path | None):
        """Private controller-pinned inputs are mounted only on disposable Judges."""
        return self._asset_bindings(path, private=True)

    def _asset_bindings(self, path: Path | None, *, private: bool):
        required = (set(item['mount_path'] for item in self.config.get('private_asset_requirements', []))
                    if private else set(self.config.get('required_mounts', [])))
        data = json.loads(path.read_text()) if path else {'schema_version': 1, 'assets': []}
        if data.get('schema_version') != 1:
            raise ValueError('Unsupported shared asset binding schema')
        bindings = data['assets']
        seen = set()
        for item in bindings:
            consumers = item.get('consumers', ['worker', 'judge'])
            allowed_consumers = ({frozenset({'judge'})} if private else
                                 {frozenset({'worker', 'judge'}), frozenset({'worker'})})
            if (not isinstance(consumers, list) or not all(isinstance(c, str) for c in consumers)
                    or len(consumers) != len(set(consumers)) or frozenset(consumers) not in allowed_consumers
                    or item.get('visibility', 'public') != ('private' if private else 'public')):
                raise ValueError('Asset visibility/consumers do not match the authorized role')
            target = PurePosixPath(item['mount_path'])
            prefix = PurePosixPath(item['object_prefix'])
            if private and not str(prefix).startswith('sforge-assets/v2/judge-private/'):
                raise ValueError('Private Judge assets require the judge-private storage scope')
            if (not target.is_absolute() or '..' in target.parts or str(target) not in required
                    or prefix.is_absolute() or '..' in prefix.parts
                    or not str(prefix).startswith('sforge-assets/')
                    or item.get('read_only') is not True or str(target) in seen):
                raise ValueError('Invalid or duplicate read-only asset binding')
            if not re.fullmatch(r'[a-f0-9]{64}', item.get('manifest_sha256', '')):
                raise ValueError('Asset binding requires a manifest SHA-256')
            if not item.get('files'):
                raise ValueError('Asset binding requires file integrity records')
            for entry in item['files']:
                name = PurePosixPath(entry['path'])
                if (name.is_absolute() or '..' in name.parts or str(name) in ('', '.')
                        or not re.fullmatch(r'[a-f0-9]{64}', entry.get('sha256',''))
                        or type(entry.get('size')) is not int or entry['size'] < 0):
                    raise ValueError('Invalid asset integrity record')
            identity = self.config.get('private_asset_identities' if private else 'asset_identities',{}).get(str(target))
            if not private and consumers == ['worker'] and (not identity or identity.get('consumers') != ['worker']):
                raise ValueError('Worker-only public asset consumers must be explicitly pinned by deployment')
            if identity and 'consumers' in identity:
                pinned = identity['consumers']
                if (not isinstance(pinned, list) or not all(isinstance(c, str) for c in pinned)
                        or len(pinned) != len(set(pinned)) or frozenset(pinned) not in allowed_consumers
                        or set(consumers) != set(pinned)):
                    raise ValueError('Asset consumers differ from pinned deployment')
            legacy_model_alias = (not private and str(prefix).startswith('sforge-assets/v2/judge-private/'))
            if legacy_model_alias:
                storage = item.get('storage_identity')
                if (consumers != ['worker'] or item.get('role') != 'model'
                        or str(prefix) != item['object_prefix'] or len(prefix.parts) != 6
                        or prefix.parts[3] != 'models'
                        or not isinstance(storage, dict) or set(storage) != {'endpoint', 'bucket'}
                        or not all(isinstance(value, str) and value for value in storage.values())
                        or item.get('read_only_model_alias') is not True or item.get('storage_scope') != 'judge-private'
                        or not identity or identity.get('role') != 'model'
                        or identity.get('read_only_model_alias') is not True or identity.get('storage_scope') != 'judge-private'
                        or identity.get('consumers') != ['worker'] or identity.get('object_prefix') != str(prefix)
                        or 'manifest_sha256' not in identity or 'files_sha256' not in identity
                        or not item.get('storage_identity') or identity.get('storage_identity') != item['storage_identity']):
                    raise ValueError('Existing private-scope model reuse requires an exact pinned public Worker-only alias')
            elif (not private and item.get('read_only_model_alias', False) is not False):
                raise ValueError('Model alias marker does not match actual storage scope')
            files_hash = hashlib.sha256(json.dumps(sorted(item['files'],key=lambda x:x['path']),
                sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
            if (not identity or identity['manifest_sha256'] != item['manifest_sha256']
                    or identity['files_sha256'] != files_hash
                    or not str(prefix).endswith('/' + item['manifest_sha256'] + '/files')):
                raise ValueError('Shared asset identity differs from pinned task deployment')
            if identity.get('object_prefix', str(prefix)) != str(prefix):
                raise ValueError('Shared asset prefix differs from pinned deployment')
            if identity.get('storage_identity', item.get('storage_identity')) != item.get('storage_identity'):
                raise ValueError('Shared asset storage differs from pinned deployment')
            if any(target in PurePosixPath(previous).parents or PurePosixPath(previous) in target.parents
                   for previous in seen):
                raise ValueError('Shared asset mounts must not overlap')
            seen.add(str(target))
        if seen != required:
            raise ValueError(f'Shared asset mounts must match deployment: {sorted(required)}')
        return bindings


def verify_gpu_inventory(output: str, resources: dict):
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    if len(rows) != resources['gpu']:
        raise RuntimeError('Allocated GPU count differs from deployment')
    identifiers = set()
    for row in rows:
        name, identifier, memory = [value.strip() for value in row.split(',')]
        if (resources['gpu_card'] not in name or identifier in identifiers
                or int(memory.split()[0]) * 1024**2 < resources.get('min_gpu_memory_bytes', 1)):
            raise RuntimeError('Allocated GPU identity/memory differs from deployment')
        identifiers.add(identifier)
    return rows
