"""Controller-owned SWE Sandbox leases; no platform credential crosses RPC.

This is an execution adapter, not a qualification receipt. The unchanged
mini-SWE client may retain its create/execute/cleanup Unix RPC protocol via a
small HTTP bridge. Platform isolation that the SDK cannot express is recorded
as a gap and must be evidenced before a formal run.
"""
from __future__ import annotations

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
import hashlib
import hmac
import json
from pathlib import Path
import re
import secrets
import shlex
import tempfile
import threading
import time
import os
import zlib
from fastapi import Request


MAX_REQUEST_BYTES = 256 * 1024
MAX_COMMAND_BYTES = 64 * 1024
MAX_OUTPUT_BYTES = 2 * 1024 * 1024
MAX_TOOL_TIMEOUT = 120
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
RUNTIME_ID = re.compile(r"[a-f0-9]{32}")
DIGEST = re.compile(r"sha256:[a-f0-9]{64}")
COMMIT = re.compile(r"[a-f0-9]{40}")
REQUIRED_ISOLATION = (
    'network_none', 'cap_drop_all', 'no_new_privileges', 'pids_limit_1024',
    'immutable_image_pull', 'tool_control_plane_separation',
)
EXEC_FRAME_PROTOCOL = 'sforge-swe-bounded-exec-v1'
MAX_EXEC_FRAME_BYTES = 4 * ((MAX_OUTPUT_BYTES + 2) // 3) + 4096
OUTPUT_TRUNCATION_MARKER = b'\n[output truncated by sandbox]\n'

# This Python process stays in the Sandbox's trusted control namespace. Only
# its child argv enters the tool namespace. Tools receive PIPE stdout/stderr,
# never the SDK response descriptor or a descriptor containing control records.
# SDK exec itself is not streaming: it therefore must see only this bounded
# base64 frame, rather than accumulating the complete untrusted output first.
EXEC_SUPERVISOR_SOURCE = r'''
import base64,json,os,selectors,signal,subprocess,sys,time,zlib
cfg=json.loads(zlib.decompress(base64.b64decode(sys.argv[1],validate=True)))
limit=2097152
retained=bytearray(); truncated=False; process=None; failure=None
selector=selectors.DefaultSelector()

def terminate_session(child):
    # GNU timeout can create a separate process group inside our session.
    # On an outer supervision failure also stop those session members, rather
    # than treating kill(nsenter-parent) as proof that the tool was cancelled.
    # This never targets the separately started persistent namespace holder.
    # Escaped sessions are not claimed cancelled; the RPC fails and ownership
    # remains in the Fleet for exact-Sandbox teardown/reconciliation.
    for name in os.listdir('/proc'):
        if not name.isdigit(): continue
        descriptor=None
        try:
            path='/proc/'+name+'/stat'
            first=open(path,'rb').read(8192).rsplit(b')',1)[1].split()
            if int(first[3])!=child.pid: continue
            if hasattr(os,'pidfd_open') and hasattr(signal,'pidfd_send_signal'):
                descriptor=os.pidfd_open(int(name),0)
            second=open(path,'rb').read(8192).rsplit(b')',1)[1].split()
            if (first[3],first[19])!=(second[3],second[19]): continue
            if descriptor is not None:
                signal.pidfd_send_signal(descriptor,signal.SIGKILL)
            else:
                os.kill(int(name),signal.SIGKILL)
        except (OSError,ValueError,IndexError): pass
        finally:
            if descriptor is not None: os.close(descriptor)
    if child.poll() is None:
        try: child.kill()
        except ProcessLookupError: pass

try:
    assert set(cfg)=={'argv','timeout_sec','protocol','frame_id'}
    assert cfg['protocol']=='sforge-swe-bounded-exec-v1'
    assert type(cfg['timeout_sec']) is int and 1<=cfg['timeout_sec']<=120
    assert isinstance(cfg['argv'],list) and cfg['argv']
    assert all(isinstance(part,str) and '\x00' not in part for part in cfg['argv'])
    deadline=time.monotonic()+cfg['timeout_sec']+10
    process=subprocess.Popen(cfg['argv'],stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,close_fds=True,
        pass_fds=(),start_new_session=True)
    os.set_blocking(process.stdout.fileno(),False)
    selector.register(process.stdout,selectors.EVENT_READ)
    while selector.get_map() or process.poll() is None:
        remaining=deadline-time.monotonic()
        if remaining<=0:
            failure='supervision_deadline_exceeded'
            terminate_session(process)
            break
        for key,_ in selector.select(min(.1,remaining)):
            try: chunk=os.read(key.fd,65536)
            except BlockingIOError: continue
            if not chunk:
                selector.unregister(key.fileobj)
                key.fileobj.close()
                continue
            available=max(0,limit-len(retained))
            if available: retained.extend(chunk[:available])
            if len(chunk)>available: truncated=True
    if failure is None:
        code=process.wait(timeout=1)
        code=code if code>=0 else 128-code
    else:
        code=125
except BaseException as error:
    failure=type(error).__name__
    code=125
    if process is not None: terminate_session(process)
finally:
    selector.close()
    if process is not None:
        if process.stdout is not None: process.stdout.close()
        try: process.wait(timeout=1)
        except subprocess.TimeoutExpired: pass
frame={'protocol':'sforge-swe-bounded-exec-v1','frame_id':cfg.get('frame_id'),
       'returncode':code,'output_b64':base64.b64encode(retained).decode('ascii'),
       'output_truncated':truncated,'supervision_error':failure}
sys.stdout.write(json.dumps(frame,separators=(',',':'),allow_nan=False))
sys.stdout.flush()
'''


class FleetError(RuntimeError):
    """A bounded, credential-free error suitable for the private RPC client."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode()


def _digest(data):
    return 'sha256:' + hashlib.sha256(data).hexdigest()


def _json_bytes(data):
    if not isinstance(data, bytes):
        raise ValueError('Image attestations require the exact registry bytes')
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError('Image attestation must be an object')
    return value


@dataclass(frozen=True)
class SWEImage:
    image_name: str
    image_digest: str
    official_manifest_digest: str
    base_commit: str
    runtime_head_commit: str
    runtime_image: str
    attestation: dict

    @classmethod
    def attest(cls, entry, *, official_manifest, base_config,
               runtime_image=None, runtime_manifest=None, runtime_config=None):
        """Bind an original image or a layers-identical CMD-only derivative.

        ``image_digest`` in the corpus is the original config digest, whereas
        ``official_manifest_digest`` is the immutable registry pull digest.
        A derivative never claims that its config digest is the original one.
        """
        for key in ('image_digest', 'official_manifest_digest'):
            if not DIGEST.fullmatch(entry.get(key, '')):
                raise ValueError('Invalid corpus image digest')
        for key in ('base_commit', 'runtime_head_commit'):
            if not COMMIT.fullmatch(entry.get(key, '')):
                raise ValueError('Invalid corpus git commitment')
        name = entry.get('image_name', '')
        if not isinstance(name, str) or not name or len(name) > 512:
            raise ValueError('Invalid corpus image name')
        manifest, config = _json_bytes(official_manifest), _json_bytes(base_config)
        if (_digest(official_manifest) != entry['official_manifest_digest']
                or _digest(base_config) != entry['image_digest']
                or manifest.get('config', {}).get('digest') != entry['image_digest']
                or not isinstance(manifest.get('layers'), list)
                or not manifest['layers']):
            raise ValueError('Official manifest/config differs from corpus commitments')
        repository = name.removeprefix('docker.io/').rsplit(':', 1)[0]
        original_ref = repository + '@' + entry['official_manifest_digest']
        runtime_image = runtime_image or original_ref
        proof = {'original_manifest_digest': entry['official_manifest_digest'],
                 'original_config_digest': entry['image_digest'],
                 'layers_sha256': hashlib.sha256(_canonical(manifest['layers'])).hexdigest(),
                 'config_only_keepalive': False}
        if runtime_image != original_ref:
            if runtime_manifest is None or runtime_config is None:
                raise ValueError('A derived image requires exact manifest and config attestations')
            derived, derived_config = _json_bytes(runtime_manifest), _json_bytes(runtime_config)
            expected_config = json.loads(base_config)
            expected_config.setdefault('config', {})['Cmd'] = ['/bin/sleep', '21600']
            if (not runtime_image.endswith('@' + _digest(runtime_manifest))
                    or derived.get('layers') != manifest['layers']
                    or derived.get('config', {}).get('digest') != _digest(runtime_config)
                    or derived_config != expected_config):
                raise ValueError('Derived image must change only Config.Cmd and retain all layers')
            proof.update(config_only_keepalive=True,
                         runtime_manifest_digest=_digest(runtime_manifest),
                         runtime_config_digest=_digest(runtime_config))
        elif runtime_manifest is not None or runtime_config is not None:
            raise ValueError('Unexpected derived image attestations')
        return cls(name, entry['image_digest'], entry['official_manifest_digest'],
                   entry['base_commit'], entry['runtime_head_commit'], runtime_image, proof)

    def identity(self):
        return {'image_name': self.image_name, 'image_digest': self.image_digest,
                'official_manifest_digest': self.official_manifest_digest,
                'base_commit': self.base_commit, 'runtime_head_commit': self.runtime_head_commit,
                'runtime_image': self.runtime_image, 'attestation': self.attestation}


def _resource_record(value):
    if value is None:
        return None
    if hasattr(value, 'model_dump'):
        value = value.model_dump(exclude_none=True)
    if not isinstance(value, dict):
        return None
    # These are observations, not defaults inferred from a request.
    return {key: str(value[key]) for key in ('cpu', 'memory', 'gpu', 'gpu_card')
            if key in value and value[key] is not None}


def _normalized_resources(value):
    """Keep raw API evidence separately; PyroMind reports RAM GiB as '8'."""
    if value is None:
        return None
    result = dict(value)
    cpu = result.get('cpu', '')
    if re.fullmatch(r'[0-9]+m', cpu) and int(cpu[:-1]) % 1000 == 0:
        result['cpu'] = str(int(cpu[:-1]) // 1000)
    memory = result.get('memory', '')
    if re.fullmatch(r'[0-9]+', memory):
        result['memory'] = memory + 'Gi'
    elif re.fullmatch(r'[0-9]+Mi', memory) and int(memory[:-2]) % 1024 == 0:
        result['memory'] = str(int(memory[:-2]) // 1024) + 'Gi'
    return result


def _runtime_readiness_gaps(evidence):
    """Per-Sandbox negative facts take precedence over global positive claims.

    Hooks may use different enforcement primitives, so positive platform proofs
    still come from REQUIRED_ISOLATION. A hook cannot declare a prototype, an
    unresolved gap or an explicitly failed enforcement fact and be promoted to
    formal readiness by those global claims.
    """
    if not isinstance(evidence, dict):
        return ['runtime_setup_evidence_unverified']
    gaps = []
    if 'prototype_only' in evidence and evidence['prototype_only'] is not False:
        gaps.append('runtime_setup_prototype_only' if evidence['prototype_only'] is True
                    else 'runtime_setup_prototype_status_unverified')
    if evidence.get('formal_task_ready') is not True:
        gaps.append('runtime_setup_formal_task_not_ready')
    fields = (
        'cpu_cfs_quota_enforced', 'pids_cgroup_limit_enforced',
        'no_new_privs', 'network_route_unreachable', 'controller_state_hidden',
        'holder_ptrace_denied', 'mount_propagation_private',
        'persistent_namespace_and_tmp_verified',
    ) + REQUIRED_ISOLATION
    mappings = [evidence]
    for field in ('facts', 'enforcement'):
        if field in evidence:
            if isinstance(evidence[field], dict):
                mappings.append(evidence[field])
            else:
                gaps.append('runtime_setup_fact_mapping_invalid')
    for mapping in mappings:
        for field in set(fields) | {name for name in mapping if isinstance(name, str)
                                   and name.endswith(('_enforced', '_verified'))}:
            if field in mapping and mapping[field] is not True:
                gaps.append('runtime_setup_' + field + '_not_verified')
    for field in ('qualification_gaps', 'gaps'):
        values = evidence.get(field, [])
        if not isinstance(values, list):
            gaps.append('runtime_setup_gap_list_invalid')
            continue
        for value in values:
            if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
                gaps.append('runtime_setup_gap_list_invalid')
            else:
                gaps.append('runtime_setup_gap_' + value)
    capabilities = evidence.get('capabilities')
    if capabilities is not None:
        names = {'CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb'}
        if (not isinstance(capabilities, dict) or set(capabilities) != names
                or any(type(value) is not int or value != 0 for value in capabilities.values())):
            gaps.append('runtime_setup_capabilities_not_empty_or_unverified')
    return gaps


class SWEFleet:
    """One run's private Controller broker, with durable exact-ID ownership.

    ``client`` is the Controller's SandboxClient. Give it an HTTP timeout longer
    than 135 seconds and disable automatic exec retries. No SDK is installed in
    the bridge, no caller chooses a host mount, and no credential is forwarded.
    SDK 0.1.10 owns a requests.Session: use ``client_factory`` to give each of
    the capacity+32 RPC threads its own client. Preparation is separately limited
    to 16 calls, so startup does not limit subsequent 192-way tool execution.
    """
    def __init__(self, client, state_root, run_id, images, *, resources=None,
                 capacity=192, prepare_concurrency=16, ready_timeout=180,
                 capability_evidence=None, secret_values=(), request_factory=None,
                 runtime_setup=None, runtime_setup_id=None, client_factory=None):
        if not isinstance(run_id, str) or not IDENTIFIER.fullmatch(run_id):
            raise ValueError('Invalid fleet run ID')
        if type(capacity) is not int or not 1 <= capacity <= 192:
            raise ValueError('Invalid SWE fleet capacity')
        if not images or any(not isinstance(item, SWEImage) for item in images):
            raise ValueError('An attested corpus image allowlist is required')
        self.images = {item.image_name: item for item in images}
        if len(self.images) != len(images):
            raise ValueError('Duplicate corpus image name')
        self.client, self.run_id = client, run_id
        self.resources = dict(resources or {'cpu': '2', 'memory': '8Gi'})
        if (set(self.resources) != {'cpu', 'memory'}
                or not re.fullmatch(r'[1-9][0-9]*', str(self.resources['cpu']))
                or not re.fullmatch(r'[1-9][0-9]*Gi', str(self.resources['memory']))):
            raise ValueError('Pin CPU-only fleet resources explicitly')
        self.resources = {key: str(value) for key, value in self.resources.items()}
        if type(prepare_concurrency) is not int or not 1 <= prepare_concurrency <= 32:
            raise ValueError('Invalid preparation concurrency')
        if type(ready_timeout) is not int or not 1 <= ready_timeout <= 180:
            raise ValueError('SWE create must fit the native 180-second RPC budget')
        self.capacity, self.ready_timeout = capacity, ready_timeout
        self.capability_evidence = dict(capability_evidence or {})
        self._secret_values = set(value for value in secret_values if value)
        self._request_factory = request_factory
        if runtime_setup is not None and (not callable(runtime_setup)
                or not isinstance(runtime_setup_id, str) or not IDENTIFIER.fullmatch(runtime_setup_id)):
            raise ValueError('Pin a Controller-owned runtime setup identity')
        if runtime_setup is None and runtime_setup_id is not None:
            raise ValueError('Runtime setup identity requires its Controller hook')
        if client_factory is not None and not callable(client_factory):
            raise ValueError('SDK client factory must be callable')
        self.runtime_setup, self.runtime_setup_id = runtime_setup, runtime_setup_id
        self.client_factory = client_factory
        self._thread_clients = threading.local()
        self._owned_clients = []
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._remember_client_secrets(client)
        self._prepare = threading.BoundedSemaphore(prepare_concurrency)
        self._operations = {}
        self._runtime_locks = {}
        self._closing = False
        root = Path(state_root)
        if root.is_symlink():
            raise ValueError('Fleet state root must not be a symlink')
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root.chmod(0o700)
        self.path = root / 'fleet.json'
        roster = {name: image.identity() for name, image in sorted(self.images.items())}
        identity = hashlib.sha256(_canonical({'images': roster, 'resources': self.resources,
                                              'runtime_setup_id': runtime_setup_id})).hexdigest()
        if self.path.exists():
            if self.path.is_symlink():
                raise ValueError('Fleet ledger must not be a symlink')
            self.state = json.loads(self.path.read_text())
            if (self.state.get('schema_version') != 1 or self.state.get('run_id') != run_id
                    or self.state.get('configuration_sha256') != identity):
                raise ValueError('Fleet recovery configuration differs from its ledger')
        else:
            self.state = {'schema_version': 1, 'run_id': run_id,
                          'configuration_sha256': identity, 'requested_resources': self.resources,
                          'leases': {}}
            with self._lock:
                self._persist()
        # ThreadPoolExecutor only bounds workers, not its work queue. The HTTP
        # route separately limits admission to capacity+32 accepted calls.
        self.rpc_workers = capacity + 32
        self.rpc_executor = ThreadPoolExecutor(max_workers=self.rpc_workers,
                                               thread_name_prefix='swe-fleet-rpc')

    def _sdk(self):
        if self.client_factory is None:
            return self.client
        client = getattr(self._thread_clients, 'client', None)
        if client is None:
            client = self.client_factory()
            self._thread_clients.client = client
            with self._lock:
                self._remember_client_secrets(client)
                self._owned_clients.append(client)
        return client

    def _remember_client_secrets(self, client):
        # Retain only for output filtering; never serialize SDK authentication.
        # Callers can additionally provide backend-specific secret_values.
        value = getattr(client, 'api_key', None)
        if isinstance(value, str) and value:
            with self._lock:
                self._secret_values.add(value)

    def _persist(self):
        descriptor, filename = tempfile.mkstemp(prefix='.fleet-', dir=self.path.parent)
        temporary = Path(filename)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(_canonical(self.state))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)

    def issue_lease(self, leg_id):
        if not isinstance(leg_id, str) or not IDENTIFIER.fullmatch(leg_id):
            raise ValueError('Invalid fleet leg ID')
        token = secrets.token_urlsafe(32)
        with self._lock:
            if self._closing or leg_id in self.state['leases']:
                raise FleetError('Fleet leg cannot be issued')
            self.state['leases'][leg_id] = {
                'token_sha256': hashlib.sha256(token.encode()).hexdigest(),
                'status': 'open', 'runtimes': {}, 'created_at': time.time()}
            self._secret_values.add(token)
            self._persist()
        return token

    def _authenticate(self, leg_id, token):
        leg = self.state['leases'].get(leg_id)
        observed = hashlib.sha256(str(token).encode()).hexdigest()
        expected = leg['token_sha256'] if leg else '0' * 64
        if not hmac.compare_digest(observed, expected) or leg is None:
            raise PermissionError('Invalid fleet capability')
        self._secret_values.add(str(token))
        return leg

    def dispatch(self, leg_id, token, request):
        """The native RPC request shape, authenticated separately by the bridge."""
        if not isinstance(request, dict) or len(_canonical(request)) > MAX_REQUEST_BYTES:
            raise FleetError('Invalid sandbox request')
        with self._condition:
            leg = self._authenticate(leg_id, token)
            if self._closing or leg['status'] != 'open':
                raise FleetError('Fleet lease is closed')
            self._operations[leg_id] = self._operations.get(leg_id, 0) + 1
        try:
            action = request.get('action')
            if action == 'create':
                return self._create(leg_id, request)
            if action == 'execute':
                return self._execute(leg_id, request)
            if action == 'cleanup':
                return self._cleanup(leg_id, request)
            raise FleetError('Unknown sandbox action')
        finally:
            with self._condition:
                self._operations[leg_id] -= 1
                self._condition.notify_all()

    @staticmethod
    def _id(request, allowed):
        if (set(request) - set(allowed) or not isinstance(request.get('sandbox_id'), str)
                or not RUNTIME_ID.fullmatch(request['sandbox_id'])):
            raise FleetError('Invalid sandbox request fields')
        return request['sandbox_id']

    def _new_request(self, image, name):
        if self._request_factory:
            return self._request_factory(image, name, dict(self.resources))
        from pyromind_sdk.client.models import SandboxRequest, SandboxType, ResourceConfig
        return SandboxRequest(sandbox_type=SandboxType.CUSTOM, name=name,
                              image=image, resources=ResourceConfig(**self.resources))

    def _create(self, leg_id, request):
        if type(self.ready_timeout) is not int or not 1 <= self.ready_timeout <= 180:
            raise FleetError('Native SWE create startup budget must remain at most 180s')
        return self._create_physical(leg_id, request, startup_timeout=self.ready_timeout)

    def _create_physical(self, leg_id, request, *, startup_timeout):
        """Controller physical allocation; native create keeps its 180s limit.

        A prewarmed pool can supply an explicit, separately pinned cold startup
        budget before native measurement. No RPC selects this value.
        """
        if type(startup_timeout) is not int or not 1 <= startup_timeout <= 3600:
            raise ValueError('Invalid Controller physical Sandbox startup budget')
        runtime_id = self._id(request, ('action', 'sandbox_id', 'image', 'image_digest',
                                        'base_commit', 'runtime_head_commit'))
        image = self.images.get(request.get('image'))
        if image is None or any(request.get(key) != getattr(image, key) for key in
                                ('image_digest', 'base_commit', 'runtime_head_commit')):
            raise FleetError('Sandbox image differs from the pinned corpus')
        with self._prepare:
            with self._lock:
                leg = self.state['leases'][leg_id]
                if self._closing or leg['status'] != 'open':
                    raise FleetError('Fleet lease is closed')
                if runtime_id in leg['runtimes']:
                    raise FleetError('Sandbox runtime ID already exists')
                live = sum(record['status'] != 'deleted' for lease in self.state['leases'].values()
                           for record in lease['runtimes'].values())
                if live >= self.capacity:
                    raise FleetError('SWE fleet capacity exhausted')
                # Include the full logical UUID so a lost create response is
                # recoverable even when different UUIDs share a long prefix.
                name = 'swe-' + hashlib.sha256(self.run_id.encode()).hexdigest()[:8] + '-' \
                    + hashlib.sha256(leg_id.encode()).hexdigest()[:8] + '-' \
                    + hashlib.sha256(runtime_id.encode()).hexdigest()[:32]
                record = {'sandbox_id': None, 'name': name, 'status': 'creating',
                          'image': image.identity(), 'requested_resources': dict(self.resources),
                          'actual_resources': None, 'actual_image_reference': None,
                          'execution_prefix': [], 'runtime_setup_id': self.runtime_setup_id,
                          'runtime_setup_evidence': None, 'created_at': time.time()}
                leg['runtimes'][runtime_id] = record
                mutex = self._runtime_locks.setdefault((leg_id, runtime_id), threading.Lock())
                self._persist()  # Exact name is durable even if create loses its response.
            mutex.acquire()
            try:
                deadline = time.monotonic() + startup_timeout
                sdk = self._sdk()
                sandbox = sdk.create(self._new_request(image.runtime_image, name))
                with self._lock:
                    record.update(sandbox_id=sandbox.id,
                                  actual_resources=_resource_record(getattr(sandbox, 'resources', None)),
                                  actual_image_reference=getattr(sandbox, 'image', None),
                                  status='starting')
                    self._persist()  # Register the ID before any readiness/probe operation.
                while True:
                    with self._lock:
                        if self._closing or leg['status'] != 'open':
                            raise FleetError('Fleet lease closed during sandbox creation')
                    observed = sdk.get_sandbox(sandbox.id)
                    status = str(observed.status).lower()
                    if status == 'running':
                        with self._lock:
                            actual = _resource_record(getattr(observed, 'resources', None))
                            if actual is not None:
                                record['actual_resources'] = actual
                            if getattr(observed, 'image', None):
                                record['actual_image_reference'] = observed.image
                            self._persist()
                        break
                    if status in {'error', 'failed', 'stopped'} or time.monotonic() >= deadline:
                        raise FleetError('SWE Sandbox did not become running')
                    time.sleep(min(1, max(0, deadline - time.monotonic())))
                if self.runtime_setup is not None:
                    setup = self.runtime_setup(sdk, sandbox.id)
                    if (not isinstance(setup, dict) or set(setup) != {'exec_prefix', 'evidence'}
                            or not isinstance(setup['exec_prefix'], list) or not setup['exec_prefix']
                            or len(setup['exec_prefix']) > 32
                            or any(not isinstance(part, str) or not part or '\x00' in part
                                   for part in setup['exec_prefix'])
                            or not setup['exec_prefix'][0].startswith('/')
                            or not isinstance(setup['evidence'], dict)):
                        raise FleetError('Invalid Controller runtime setup result')
                    encoded = _canonical(setup)
                    with self._lock:
                        secrets_snapshot = tuple(self._secret_values)
                    if len(encoded) > 32 * 1024 or any(
                            value.encode() in encoded for value in secrets_snapshot):
                        raise FleetError('Runtime setup evidence cannot contain credentials')
                    with self._lock:
                        record.update(execution_prefix=list(setup['exec_prefix']),
                                      runtime_setup_evidence=setup['evidence'])
                        self._persist()
                command = ('test -d /testbed && test "$(git rev-parse HEAD)" = '
                           + shlex.quote(image.runtime_head_commit) + ' && git merge-base --is-ancestor '
                           + shlex.quote(image.base_commit) + ' HEAD && git reset --hard HEAD >/dev/null 2>&1'
                           + ' && git clean -fd >/dev/null 2>&1 && test -z "$(git status --porcelain=v1)"')
                available = int(deadline - time.monotonic()) - 15
                if available < 1:
                    raise FleetError('SWE Sandbox baseline exceeded the create RPC budget')
                result = self._exec_record(record, command, min(MAX_TOOL_TIMEOUT, available))
                if result['returncode'] != 0:
                    raise FleetError('SWE Sandbox git baseline validation failed')
                with self._lock:
                    record.update(status='ready', baseline_verified=True)
                    self._persist()
                return {'created': True, 'sandbox_id': runtime_id,
                        'image_digest': image.image_digest}
            except Exception as error:
                if record['sandbox_id']:
                    try:
                        self._delete(record)
                    except FleetError:
                        pass  # The durable ownership record remains for reconciliation.
                if isinstance(error, FleetError):
                    raise FleetError(self._redact(str(error))) from None
                raise FleetError('SWE Sandbox creation failed: ' + type(error).__name__) from None
            finally:
                mutex.release()

    def _record(self, leg_id, runtime_id):
        with self._lock:
            record = self.state['leases'][leg_id]['runtimes'].get(runtime_id)
            if record is None:
                raise FleetError('Unknown sandbox runtime in this lease')
            return record

    def _redact(self, text):
        with self._lock:
            values = tuple(self._secret_values)
        for value in sorted(values, key=len, reverse=True):
            text = text.replace(value, '[REDACTED]')
        return text

    def _exec_record(self, record, command, timeout):
        # stdout/stderr merge only in the child pipe. The trusted supervisor
        # drains it continuously and sends the SDK one bounded base64 frame.
        # It stays outside the execution_prefix's namespace and inherits no key.
        # No command is retried after an ambiguous transport failure.
        script = ('cd /testbed || exit 125; ulimit -c 0; ulimit -f 2097152; ulimit -n 1024; exec env '
                  'HOME=/root LANG=C.UTF-8 GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory '
                  'GIT_CONFIG_VALUE_0=/testbed http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= '
                  'ALL_PROXY= NO_PROXY=\'*\' timeout --signal=KILL --kill-after=2s '
                  + str(timeout) + 's bash -lc ' + shlex.quote(command) + ' 2>&1')
        try:
            # SDK 0.1.10's server loses argv quoting. A shell-quoted string
            # preserves the exact command; cwd stays inside the child script.
            argv = [*record.get('execution_prefix', []), 'bash', '-lc', script]
            frame_id = secrets.token_hex(16)
            config = {'argv': argv, 'timeout_sec': timeout,
                      'protocol': EXEC_FRAME_PROTOCOL, 'frame_id': frame_id}
            encoded = base64.b64encode(zlib.compress(_canonical(config))).decode('ascii')
            supervisor = ['python3', '-I', '-S', '-c', EXEC_SUPERVISOR_SOURCE, encoded]
            result = self._sdk().exec_command(record['sandbox_id'], shlex.join(supervisor),
                                             timeout=timeout + 15)
        except Exception as error:
            raise FleetError('SWE Sandbox exec transport failed: ' + type(error).__name__) from None
        output = getattr(result, 'output', None)
        if (getattr(result, 'returncode', None) != 0 or not isinstance(output, str)
                or len(output.encode('utf-8')) > MAX_EXEC_FRAME_BYTES
                or getattr(result, 'stderr', '') not in ('', None)):
            raise FleetError('SWE Sandbox bounded exec supervisor failed')
        try:
            def pairs(items):
                value = {}
                for key, item in items:
                    if key in value:
                        raise ValueError('Duplicate exec frame field')
                    value[key] = item
                return value
            frame = json.loads(output, object_pairs_hook=pairs)
            if (not isinstance(frame, dict) or set(frame) != {
                    'protocol', 'frame_id', 'returncode', 'output_b64',
                    'output_truncated', 'supervision_error'}
                    or frame['protocol'] != EXEC_FRAME_PROTOCOL or frame['frame_id'] != frame_id
                    or type(frame['returncode']) is not int or not 0 <= frame['returncode'] <= 255
                    or type(frame['output_truncated']) is not bool
                    or not isinstance(frame['output_b64'], str)):
                raise ValueError('Invalid exec frame')
            raw = base64.b64decode(frame['output_b64'], validate=True)
            if len(raw) > MAX_OUTPUT_BYTES:
                raise ValueError('Unbounded exec frame')
        except (ValueError, TypeError, KeyError) as error:
            raise FleetError('SWE Sandbox exec returned an invalid bounded frame') from None
        if frame['supervision_error'] is not None:
            # A stuck/escaped child is never reported as a successfully drained
            # tool. Durable ownership remains available for exact-ID cleanup.
            raise FleetError('SWE Sandbox output supervision did not complete')
        code = frame['returncode']
        truncated = frame['output_truncated']
        output = self._redact(raw.decode('utf-8', errors='replace'))
        if truncated:
            output += OUTPUT_TRUNCATION_MARKER.decode('ascii')
        return {'output': output, 'returncode': code,
                'timed_out': code == 124, 'output_truncated': truncated}

    def _execute(self, leg_id, request):
        runtime_id = self._id(request, ('action', 'sandbox_id', 'command', 'timeout_sec'))
        command = request.get('command', '')
        timeout = request.get('timeout_sec', 30)
        if (not isinstance(command, str) or len(command.encode()) > MAX_COMMAND_BYTES
                or '\x00' in command or type(timeout) is not int):
            raise FleetError('Invalid sandbox command')
        timeout = max(1, min(timeout, MAX_TOOL_TIMEOUT))
        record = self._record(leg_id, runtime_id)
        with self._lock:
            mutex = self._runtime_locks.setdefault((leg_id, runtime_id), threading.Lock())
        with mutex:
            if record['status'] != 'ready':
                raise FleetError('Sandbox runtime is not ready')
            return self._exec_record(record, command, timeout)

    @staticmethod
    def _missing(error):
        return getattr(error, 'status_code', None) == 404

    def _delete(self, record):
        identifier = record['sandbox_id']
        if record['status'] == 'deleted':
            return
        if not identifier:
            raise FleetError('Sandbox creation response is unresolved; retain its ownership record')
        try:
            sdk = self._sdk()
            sandbox = sdk.get_sandbox(identifier)
            if str(sandbox.status).lower() == 'running':
                try:
                    sdk.pause(identifier)
                except Exception as error:
                    if not self._missing(error):
                        raise
            try:
                sdk.delete(identifier)
            except Exception as error:
                if not self._missing(error):
                    raise
            # A successful delete request is not a proven absence.
            sdk.get_sandbox(identifier)
        except Exception as error:
            if not self._missing(error):
                raise FleetError('SWE Sandbox cleanup failed: ' + type(error).__name__) from None
        else:
            raise FleetError('SWE Sandbox deletion has not been confirmed')
        with self._lock:
            record.update(status='deleted', deleted_at=time.time())
            self._persist()

    def _cleanup(self, leg_id, request):
        runtime_id = self._id(request, ('action', 'sandbox_id'))
        record = self._record(leg_id, runtime_id)
        with self._lock:
            mutex = self._runtime_locks.setdefault((leg_id, runtime_id), threading.Lock())
        with mutex:
            self._delete(record)
        return {'removed': True, 'sandbox_id': runtime_id}

    def reconcile(self, leg_id=None):
        """Recover lost create responses by exact recorded names, then delete IDs."""
        with self._lock:
            legs = ([self.state['leases'][leg_id]] if leg_id is not None
                    else list(self.state['leases'].values()))
            records = [record for leg in legs for record in leg['runtimes'].values()
                       if record['status'] != 'deleted']
        try:
            remote = list(self._sdk().list())
        except Exception as error:
            raise FleetError('Fleet reconciliation list failed: ' + type(error).__name__) from None
        by_name = {}
        for sandbox in remote:
            by_name.setdefault(sandbox.name, []).append(sandbox)
        failures = []
        for record in records:
            if record['sandbox_id'] is None:
                matches = by_name.get(record['name'], [])
                if len(matches) > 1:
                    failures.append('ambiguous recorded sandbox name')
                    continue
                with self._lock:
                    if matches:
                        record['sandbox_id'] = matches[0].id
                    else:
                        # A timed-out create can appear after this list call.
                        # Absence now cannot erase that durable ownership name.
                        failures.append('sandbox creation response remains unresolved')
                        continue
                    self._persist()
            try:
                self._delete(record)
            except FleetError as error:
                failures.append(str(error))
        return {'cleaned': not failures, 'remaining': sum(
            record['status'] != 'deleted' for record in records), 'errors': failures}

    def close_lease(self, leg_id, *, timeout=300):
        """Revoke admission, drain accepted RPCs, and reconcile in every exit path."""
        deadline = time.monotonic() + timeout
        with self._condition:
            leg = self.state['leases'][leg_id]
            leg['status'] = 'closing'
            self._persist()
            self._condition.notify_all()
            while self._operations.get(leg_id, 0):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {'cleaned': False, 'remaining': None, 'errors': ['accepted RPCs still active']}
                self._condition.wait(remaining)
        result = self.reconcile(leg_id)
        if result['cleaned']:
            with self._lock:
                leg['status'] = 'closed'
                self._persist()
        return result

    def close(self, *, timeout_per_leg=300):
        """Controller finally hook: revoke all legs before draining/deleting any."""
        with self._lock:
            self._closing = True
            legs = list(self.state['leases'])
            for leg in self.state['leases'].values():
                if leg['status'] != 'closed':
                    leg['status'] = 'closing'
            self._persist()
        results = {}
        for leg_id in legs:
            try:
                results[leg_id] = self.close_lease(leg_id, timeout=timeout_per_leg)
            except FleetError as error:
                results[leg_id] = {'cleaned': False, 'remaining': None, 'errors': [str(error)]}
        # Do not cancel futures: an accepted command may already have remote
        # side effects. Drain the dedicated executor after revocation/cleanup.
        self.rpc_executor.shutdown(wait=True, cancel_futures=False)
        # A remote action may outlive the initial drain budget. Executor drain
        # must be followed by reconciliation, rather than closing its SDK client
        # while a late returned Sandbox ID remains owned but uncollected.
        for leg_id, result in tuple(results.items()):
            if result['errors'] == ['accepted RPCs still active']:
                try:
                    results[leg_id] = self.close_lease(leg_id, timeout=timeout_per_leg)
                except FleetError as error:
                    results[leg_id] = {'cleaned': False, 'remaining': None, 'errors': [str(error)]}
        cleaned = all(result['cleaned'] for result in results.values())
        # Unconfirmed deletion remains owned and requires another reconcile.
        # Keep the thread-local SDK sessions usable for that retry.
        if cleaned:
            for client in self._owned_clients:
                close = getattr(client, 'close', None)
                if close is not None:
                    close()
        return {'cleaned': cleaned, 'legs': results}

    def formal_readiness(self):
        """Explicit gaps; no caller may infer 192/GPU qualification from a probe."""
        gaps = []
        for name in REQUIRED_ISOLATION:
            evidence = self.capability_evidence.get(name)
            if (not isinstance(evidence, dict) or evidence.get('verified') is not True
                    or not isinstance(evidence.get('evidence'), str) or not evidence['evidence']):
                gaps.append(name)
        if self.resources != {'cpu': '2', 'memory': '8Gi'}:
            gaps.append('requested_resources_differ_from_native_2cpu_8g')
        if self.runtime_setup is None:
            gaps.append('tool_runtime_setup_unconfigured')
        if self.client_factory is None and hasattr(self.client, 'session'):
            gaps.append('sdk_http_session_shared_unverified')
        with self._lock:
            records = [record for leg in self.state['leases'].values()
                       for record in leg['runtimes'].values() if record['sandbox_id']]
            for record in records:
                actual = _normalized_resources(record['actual_resources'])
                if (actual is None or actual.get('cpu') != self.resources['cpu']
                        or actual.get('memory') != self.resources['memory']
                        or actual.get('gpu') != '0'):
                    gaps.append('allocated_resources_unverified_or_mismatched')
                if not record.get('baseline_verified'):
                    gaps.append('sandbox_startup_and_baseline_unverified')
                if record.get('actual_image_reference') != record['image']['runtime_image']:
                    gaps.append('allocated_image_reference_unverified_or_mismatched')
                if not record.get('execution_prefix'):
                    gaps.append('tool_execution_prefix_unconfigured')
                gaps.extend(_runtime_readiness_gaps(record.get('runtime_setup_evidence')))
        if not records:
            gaps.append('no_real_swe_sandbox_probe')
        return {'ready': not gaps, 'gaps': sorted(set(gaps)),
                'requested_resources_each_sandbox': dict(self.resources),
                'qualification': 'not_a_192_workload_or_gpu_qualification'}

    def assert_formal_ready(self):
        result = self.formal_readiness()
        if not result['ready']:
            raise FleetError('Formal SWE fleet is blocked: ' + ', '.join(result['gaps']))
        return result


def create_fleet_router(fleet):
    """Mount on the existing Controller service; bridge authentication is private."""
    from fastapi import APIRouter
    from fastapi.responses import JSONResponse
    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            # close revokes admission and drains remote side effects before
            # deleting exact IDs; it never cancels accepted executor futures.
            await asyncio.to_thread(fleet.close)

    router = APIRouter(lifespan=lifespan)
    slots = asyncio.Semaphore(fleet.rpc_workers)

    @router.post('/api/v1/swe-fleet/{run_id}/{leg_id}/rpc')
    async def rpc(run_id: str, leg_id: str, request: Request):
        if run_id != fleet.run_id:
            return JSONResponse({'protocol_error': 'Unknown fleet run'}, status_code=404)
        authorization = request.headers.get('authorization', '')
        if not authorization.startswith('Bearer '):
            return JSONResponse({'protocol_error': 'Invalid fleet capability'}, status_code=403)
        try:
            with fleet._lock:
                fleet._authenticate(leg_id, authorization[7:])
                if fleet._closing or fleet.state['leases'][leg_id]['status'] != 'open':
                    raise FleetError('Fleet lease is closed')
        except PermissionError:
            return JSONResponse({'protocol_error': 'Invalid fleet capability'}, status_code=403)
        except FleetError as error:
            return JSONResponse({'protocol_error': str(error)})
        body = bytearray()
        async for part in request.stream():
            body.extend(part)
            if len(body) > MAX_REQUEST_BYTES:
                return JSONResponse({'protocol_error': 'Sandbox request too large'}, status_code=413)
        try:
            value = json.loads(body)
            # Keep >=192 simultaneous tool commands off asyncio's default
            # 32-thread executor. Bound accepted work rather than silently
            # adding a tool execution queue to the measured workload.
            await slots.acquire()
            try:
                future = asyncio.get_running_loop().run_in_executor(
                    fleet.rpc_executor, partial(fleet.dispatch, leg_id, authorization[7:], value))
            except BaseException:
                slots.release()
                raise

            def completed(future):
                slots.release()
                # A disconnected HTTP client does not cancel a dispatched
                # command, and its eventual exception still needs consuming.
                if not future.cancelled():
                    future.exception()

            future.add_done_callback(completed)
            result = await asyncio.shield(future)
        except PermissionError:
            return JSONResponse({'protocol_error': 'Invalid fleet capability'}, status_code=403)
        except (ValueError, TypeError):
            return JSONResponse({'protocol_error': 'Invalid sandbox request'}, status_code=400)
        except FleetError as error:
            return JSONResponse({'protocol_error': fleet._redact(str(error))})
        return JSONResponse(result)

    return router
