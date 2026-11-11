"""PyroMind SDK adapter with explicit CPU/GPU resource requests.

Unsupported security/resource guarantees fail explicitly. Environments are
applied to SDK exec calls, not to the image's PID 1. Images must stay alive.
"""
from __future__ import annotations

import json
import hashlib
import math
import os
import re
import shlex
import stat
import tarfile
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from sforge.harness.backend.base import (
    ContainerBackend, ContainerHandle, ExecResult, StreamingExecResult,
)
from sforge.pyromind.candidate_mount import (
    CANDIDATE_MOUNT_PATH, MountedCandidateArchive,
)


class SandboxPlacementRejected(RuntimeError):
    """Trusted admission rejected resource placement before candidate execution."""
    REASONS = frozenset({'nvlink-pair-required', 'flash-gpu-numa-cpus-unavailable',
                         'flash-gpu-cpu-slice-unavailable'})

    def __init__(self, reason, evidence):
        if reason not in self.REASONS or not isinstance(evidence, dict):
            raise ValueError('Unknown trusted Sandbox placement rejection')
        self.reason = reason
        self.evidence = evidence
        super().__init__('Sandbox placement rejected: ' + reason)


@dataclass
class PyroMindHandle(ContainerHandle):
    sandbox_id: str
    sandbox_name: str
    environment: dict[str, str] = field(default_factory=dict)
    internal_ip: str | None = None
    bootstrap: dict = field(default_factory=dict)
    mounted_archive: MountedCandidateArchive | None = None
    allocation_request: object | None = field(default=None, repr=False)

    @property
    def id(self):
        return self.sandbox_id

    @property
    def name(self):
        return self.sandbox_name

    @property
    def ip_address(self):
        return self.internal_ip


class PyroMindBackend(ContainerBackend):
    def __init__(self, client=None, *, image_map=None, event_path=None, on_started=None,
                 volume_mounts=None, gpu_count=None, gpu_card=None, port_mappings=None):
        if client is None:
            from pyromind_sdk import PyroMindAPIClient
            client = PyroMindAPIClient(
                cluster=os.getenv("PYROMIND_CLUSTER", "us-west-1"),
                timeout=30, max_retries=0,
            ).sandboxes
        self.client = client
        self.image_map = image_map if image_map is not None else json.loads(
            os.getenv("SFORGE_PYROMIND_IMAGE_MAP", "{}")
        )
        self.event_path = event_path or os.getenv("SFORGE_PYROMIND_EVENTS")
        self.ready_timeout = int(os.getenv("SFORGE_PYROMIND_READY_TIMEOUT", "300"))
        self.on_started = on_started
        self.startup_cancelled = None
        self._creation_lock = threading.RLock()
        self._creation_records = {}
        self.creation_journal_path = Path(self.event_path) if self.event_path else None
        # Deployment-controlled mounts; never taken from submitted candidates.
        self.volume_mounts = volume_mounts
        # Service ports are selected by the Controller, never submitted code.
        self.port_mappings = port_mappings
        self.gpu_count = gpu_count if gpu_count is not None else os.getenv(
            "SFORGE_PYROMIND_GPU_COUNT", "0")
        self.gpu_card = gpu_card if gpu_card is not None else os.getenv(
            "SFORGE_PYROMIND_GPU_CARD")

    def _gpu_resources(self, entry):
        config = entry if isinstance(entry, dict) else {}
        count = config.get("gpu_count", self.gpu_count)
        card = config.get("gpu_card", self.gpu_card)
        if isinstance(count, bool) or not re.fullmatch(r"[0-9]+", str(count)):
            raise ValueError("gpu_count must be a nonnegative integer")
        count = int(count)
        if count and (not isinstance(card, str) or not card.strip()):
            raise ValueError("gpu_card is required when gpu_count is positive")
        if not count:
            # A per-image zero overrides the deployment GPU default.
            return {}
        return {"gpu": str(count), "gpu_card": card.strip()}

    @staticmethod
    def _validate_environment(environment):
        credentials = [value for value in (os.getenv("PYROMIND_API_KEY"),
                       os.getenv("PYROMIND_STORAGE_SECRET_KEY")) if value]
        for key, value in environment.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError("invalid environment variable name")
            if (key in {"PYROMIND_API_KEY", "PYROMIND_STORAGE_SECRET_KEY"}
                    or any(secret in str(value) for secret in credentials)):
                raise ValueError("PyroMind platform credentials must stay on the controller")

    @staticmethod
    def _redact_error(message, environment):
        text = str(message)
        secrets = [os.getenv("PYROMIND_API_KEY"), os.getenv("PYROMIND_STORAGE_SECRET_KEY")]
        secrets.extend(str(value) for key, value in environment.items()
                       if re.search(r"KEY|TOKEN|PASSWORD|SECRET|AUTH", key, re.I))
        for value in sorted({value for value in secrets if value}, key=len, reverse=True):
            text = text.replace(value, "[REDACTED]")
        return text

    def _exec_api(self, handle, command, *, timeout, environment=None):
        env = {**handle.environment, **(environment or {})}
        try:
            return self.client.exec_command(handle.id, command, timeout=timeout)
        except Exception as exc:
            # SDK exceptions can contain the full request command, including env.
            raise RuntimeError(self._redact_error(exc, env)) from None

    @property
    def backend_name(self):
        return "pyromind"

    def _event(self, event, handle, **extra):
        if self.event_path:
            path = Path(self.event_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                stream.write(json.dumps({"time": time.time(), "event": event,
                                         "id": handle.id, "name": handle.name,
                                         **extra}) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def _creation_event(self, event, name, **extra):
        """Persist ownership before dispatch; a missing response is never absence."""
        with self._creation_lock:
            if self.creation_journal_path is None:
                directory = Path(tempfile.mkdtemp(prefix='sforge-pyromind-ownership-'))
                self.creation_journal_path = directory / 'creation-intents.jsonl'
            path = self.creation_journal_path
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            try:
                info = os.fstat(descriptor)
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
                    raise RuntimeError('Sandbox creation journal must be an owned regular file')
                os.fchmod(descriptor, 0o600)
                raw = (json.dumps({'time': time.time(), 'event': event, 'id': extra.get('id'),
                                   'name': name, **extra}, sort_keys=True) + '\n').encode()
                while raw:
                    count = os.write(descriptor, raw)
                    if count <= 0:
                        raise OSError('Sandbox ownership journal write did not progress')
                    raw = raw[count:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)

    def _owned_creations(self):
        with self._creation_lock:
            records = {}
            path = self.creation_journal_path
            if path is not None and path.exists():
                for line in path.read_text().splitlines():
                    event = json.loads(line)
                    kind, name = event.get('event'), event.get('name')
                    if kind == 'creation_intent':
                        if name in records:
                            raise RuntimeError('Duplicate Sandbox creation intent')
                        records[name] = dict(event, sandbox_id=None, confirmed_absent=False)
                    elif kind in {'creation_resolved', 'creation_cleanup_confirmed'}:
                        if name not in records:
                            raise RuntimeError('Sandbox creation resolution has no ownership intent')
                        record = records[name]
                        identifier = event.get('id')
                        if (not isinstance(identifier, str) or not identifier
                                or record['sandbox_id'] not in (None, identifier)):
                            raise RuntimeError('Sandbox creation ownership ID changed')
                        record.update(sandbox_id=identifier,
                                      sandbox_name=event.get('sandbox_name', record.get('sandbox_name', name)))
                        if kind == 'creation_cleanup_confirmed':
                            if event.get('independent_404') is not True:
                                raise RuntimeError('Sandbox ownership cleanup has no independent GET404')
                            record['confirmed_absent'] = True
            # An accepted response remains owned even if its subsequent fsync
            # failed. Do not lose that exact SID by re-reading an older journal.
            for name, cached in self._creation_records.items():
                record = records.setdefault(name, dict(cached))
                if cached.get('sandbox_id'):
                    if record.get('sandbox_id') not in (None, cached['sandbox_id']):
                        raise RuntimeError('Sandbox creation ownership ID changed')
                    record.update(sandbox_id=cached['sandbox_id'], sandbox_name=cached['sandbox_name'])
                record['confirmed_absent'] = record.get('confirmed_absent') is True or cached.get('confirmed_absent') is True
            self._creation_records = records
            return {name: dict(record) for name, record in records.items()}

    def _resolve_creation(self, name, identifier, sandbox_name):
        with self._creation_lock:
            record = self._owned_creations()[name]
            if (not isinstance(identifier, str) or not identifier
                    or record['sandbox_id'] not in (None, identifier)):
                raise RuntimeError('Sandbox create response has no exact owned ID')
            record.update(sandbox_id=identifier, sandbox_name=sandbox_name)
            self._creation_records[name] = record
            self._creation_event('creation_resolved', name, id=identifier, sandbox_name=sandbox_name)

    def _dispatch_creation(self, request):
        self._check_startup_cancelled()
        intent = {'image': request.image, 'allocation_sha256': self._allocation_digest(request)}
        self._creation_event('creation_intent', request.name, **intent)
        with self._creation_lock:
            self._creation_records[request.name] = dict(intent, name=request.name,
                sandbox_id=None, confirmed_absent=False)
        # Leave the durable intent unresolved for every unknown create outcome,
        # including interrupted SDK requests and accepted-but-lost responses.
        sb = self.client.create(request)
        self._resolve_creation(request.name, sb.id, sb.name)
        return sb

    def unresolved_creation_names(self):
        return sorted(name for name, record in self._owned_creations().items()
                      if record['sandbox_id'] is None and not record['confirmed_absent'])

    def reconcile_creation_intents(self):
        """Delete exact owned allocations, retaining unknown names after empty LIST."""
        from pyromind_sdk.client.base import PyroMindAPIError
        remaining, errors = [], []
        records = self._owned_creations()
        unknown = [name for name, record in records.items() if record['sandbox_id'] is None]
        listed = self.client.list() if unknown else []
        for name, record in records.items():
            if record['confirmed_absent']:
                continue
            identifier = record['sandbox_id']
            try:
                if identifier is None:
                    matches = [sandbox for sandbox in listed if sandbox.name == name]
                    if len(matches) != 1:
                        continue
                    identifier = matches[0].id
                    self._resolve_creation(name, identifier, matches[0].name)
                    record['sandbox_name'] = matches[0].name
                try:
                    observed = self.client.get_sandbox(identifier)
                except PyroMindAPIError as exc:
                    if exc.status_code != 404:
                        raise
                else:
                    if observed.name != record.get('sandbox_name', name):
                        raise RuntimeError('Sandbox cleanup differs from exact creation ownership')
                    self.cleanup_container(PyroMindHandle(identifier, observed.name))
                    try:
                        self.client.get_sandbox(identifier)
                    except PyroMindAPIError as exc:
                        if exc.status_code != 404:
                            raise
                    else:
                        raise RuntimeError('Sandbox creation cleanup has no independent GET404')
                self._creation_event('creation_cleanup_confirmed', name, id=identifier,
                                     sandbox_name=record.get('sandbox_name', name), independent_404=True)
                with self._creation_lock:
                    self._creation_records[name]['confirmed_absent'] = True
            except Exception as exc:
                if identifier is not None:
                    remaining.append(identifier)
                errors.append(type(exc).__name__ + ':' + name)
        unresolved = self.unresolved_creation_names()
        errors.extend('unresolved-create:' + name for name in unresolved)
        return {'remaining_sandbox_ids': sorted(set(remaining)),
                'unresolved_creation_names': unresolved, 'cleanup_errors': errors,
                'confirmed_absent_by_get_404': not remaining and not errors and not unresolved}

    def create_container(self, image, name, *, command="tail -f /dev/null",
                         environment=None, extra_hosts=None, cap_drop=None,
                         cpu_limit=None, mem_limit=None, network_mode=None,
                         user=None, annotations=None, accelerator_allocation=None,
                         mounted_archive=None):
        from pyromind_sdk.client.models import (
            SandboxRequest, SandboxType, ResourceConfig, VolumeMount, PortMapping,
        )
        unsupported = {"extra_hosts": extra_hosts, "cap_drop": cap_drop,
                       "network_mode": network_mode, "annotations": annotations,
                       "accelerator_allocation": accelerator_allocation}
        for key, value in unsupported.items():
            if value:
                raise NotImplementedError(f"PyroMind pilot cannot enforce {key}")
        if user not in (None, "root", "0"):
            raise NotImplementedError("PyroMind pilot supports root exec only")
        if command != "tail -f /dev/null":
            raise NotImplementedError("SDK 0.1.10 cannot override the image command")
        entry = self.image_map.get(image, image)
        reference = entry["image"] if isinstance(entry, dict) else entry
        self._validate_environment(environment or {})
        gpu_resources = self._gpu_resources(entry)
        cpu = cpu_limit if cpu_limit is not None else 4
        memory = mem_limit if mem_limit is not None else f"{cpu * 2}Gi"
        match = re.fullmatch(r"([1-9][0-9]*)(?:[gG](?:[iI])?)", str(memory))
        if type(cpu) is not int or cpu < 1 or not match:
            raise ValueError("Sandbox requires a whole number of CPUs and of GiB of memory")
        # CPU and memory are requested independently (any ratio the account
        # is admitted for; deploy/pyromind/RESOURCES.md).
        memory = match[1] + "Gi"
        # This is per evaluation. A backend can serve concurrent task sessions;
        # changing its shared mounts would attach another candidate's version.
        mounts = list(self.volume_mounts or [])
        if mounted_archive is not None:
            if not isinstance(mounted_archive, MountedCandidateArchive):
                raise TypeError('mounted_archive must be a trusted candidate reference')
            for mount in mounts:
                raw_path = mount.get('mount_path') if isinstance(mount, dict) else mount.mount_path
                path = PurePosixPath(raw_path)
                if (not path.is_absolute() or '..' in path.parts
                        or path == CANDIDATE_MOUNT_PATH
                        or path in CANDIDATE_MOUNT_PATH.parents
                        or CANDIDATE_MOUNT_PATH in path.parents):
                    raise ValueError('Deployment mount collides with the candidate mount')
            mounts.append(VolumeMount(host_path=mounted_archive.host_path,
                mount_path=str(CANDIDATE_MOUNT_PATH), read_only=True))
        name = re.sub(r"[^a-z0-9-]", "-", name.lower())[:45].strip("-")
        request = SandboxRequest(
            sandbox_type=SandboxType.CUSTOM, name=name + "-" + uuid.uuid4().hex[:8],
            image=reference,
            resources=ResourceConfig(cpu=str(cpu), memory=memory, **gpu_resources),
            volume_mounts=mounts if mounts else None,
            port_mappings=[PortMapping(**item) if isinstance(item, dict) else item
                           for item in self.port_mappings] if self.port_mappings else None,
        )
        sb = self._dispatch_creation(request)
        handle = PyroMindHandle(sb.id, sb.name, dict(environment or {}),
                               bootstrap=entry if isinstance(entry, dict) else {},
                               mounted_archive=mounted_archive, allocation_request=request)
        try:
            self._event("created", handle, image=reference, **gpu_resources,
                        allocation_sha256=self._allocation_digest(request))
        except BaseException:
            self.cleanup_container(handle)
            raise
        return handle

    @staticmethod
    def _allocation_digest(request):
        # SDK requests contain resources/mounts/ports/image, never exec credentials.
        data = request.model_dump(mode='json', exclude={'name'})
        return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def _replace_rejected_allocation(self, handle):
        self._check_startup_cancelled()
        request = handle.allocation_request
        if request is None:
            raise RuntimeError('A placement retry requires the original trusted allocation request')
        name = request.name.rsplit('-', 1)[0] + '-' + uuid.uuid4().hex[:8]
        request = request.model_copy(update={'name': name}, deep=True)
        sb = self._dispatch_creation(request)
        handle.sandbox_id, handle.sandbox_name = sb.id, sb.name
        handle.internal_ip = None
        handle.allocation_request = request
        try:
            resources = request.resources.model_dump()
            self._event('created', handle, image=request.image,
                        allocation_sha256=self._allocation_digest(request),
                        **{key: resources[key] for key in ('gpu', 'gpu_card') if resources.get(key)})
        except BaseException:
            self.cleanup_container(handle)
            raise
        if self.startup_cancelled is not None and self.startup_cancelled():
            self.cleanup_container(handle)
            self._check_startup_cancelled()

    def _check_startup_cancelled(self):
        if self.startup_cancelled is not None and self.startup_cancelled():
            raise RuntimeError('Controller is stopping; new Sandbox admission closed')

    def _confirm_rejected_cleanup(self, handle):
        from pyromind_sdk.client.base import PyroMindAPIError
        self.cleanup_container(handle)
        try:
            self.client.get_sandbox(handle.id)
        except PyroMindAPIError as exc:
            if exc.status_code == 404:
                self._event('placement_cleanup_confirmed', handle, independent_404=True)
                return
            raise
        raise RuntimeError('Rejected placement cleanup is not independently confirmed')

    def verify_mounted_archive(self, handle, reference):
        """Verify exact mounted bytes before the generic evaluator extracts them."""
        if (not isinstance(reference, MountedCandidateArchive)
                or handle.mounted_archive != reference):
            raise ValueError('Judge is not bound to this mounted candidate')
        result = self.exec_run_with_timeout(
            handle, reference.verification_command(), timeout=45)
        if result.timed_out or result.exit_code != 0:
            raise RuntimeError('Judge candidate mount verification failed: '
                               + self._redact_error(result.output, handle.environment))
        try:
            evidence = json.loads(result.output)
        except (TypeError, ValueError):
            raise RuntimeError('Judge candidate mount verification returned invalid evidence') from None
        expected = {'archive_sha256': reference.archive_sha256,
                    'manifest_sha256': reference.manifest_sha256,
                    'archive_size': reference.archive_size, 'read_only': True,
                    'version_id': reference.version_id}
        if evidence != expected:
            raise RuntimeError('Judge candidate mount verification evidence mismatch')
        self._event('candidate-mounted', handle, **expected,
                    host_path=reference.host_path, mount_path=str(CANDIDATE_MOUNT_PATH))
        return reference.archive_path

    def start_container(self, handle):
        deadline = time.monotonic() + self.ready_timeout
        attempt = 1
        while time.monotonic() < deadline:
            self._check_startup_cancelled()
            status = self.client.get_sandbox(handle.id).status.lower()
            if status == "running":
                if self._checked(handle, ["id", "-u"]).strip() != "0":
                    raise RuntimeError("SDK cannot select exec user; pilot requires a root image")
                for command in handle.bootstrap.get("setup_commands", []):
                    self._checked(handle, command)
                for src, dst in handle.bootstrap.get("files", {}).items():
                    self.copy_to_container(handle, Path(src), PurePosixPath(dst))
                self._event("running", handle)
                try:
                    if self.on_started is not None:
                        self.on_started(handle)
                except SandboxPlacementRejected as exc:
                    self._event('placement_rejected', handle, reason=exc.reason,
                                evidence=exc.evidence, before_candidate=True,
                                attempt=attempt, max_attempts=3)
                    self._confirm_rejected_cleanup(handle)
                    if attempt >= 3 or time.monotonic() >= deadline:
                        raise
                    self._replace_rejected_allocation(handle)
                    attempt += 1
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'Sandbox {handle.id} admission exceeded {self.ready_timeout}s')
                return
            if status in {"error", "failed", "stopped"}:
                raise RuntimeError(f"Sandbox {handle.id} entered {status}")
            time.sleep(2)
        raise TimeoutError(f"Sandbox {handle.id} not ready within {self.ready_timeout}s")

    def cleanup_container(self, handle, logger=None):
        if handle is None:
            return
        from pyromind_sdk.client.base import PyroMindAPIError
        try:
            sb = self.client.get_sandbox(handle.id)
        except PyroMindAPIError as exc:
            if exc.status_code == 404:
                self._event("deleted", handle)
                return
            raise
        if sb.status.lower() == "running":
            try:
                self.client.pause(handle.id)
            except PyroMindAPIError as exc:
                if exc.status_code != 404:
                    raise
                # A concurrent evaluator/controller cleanup may already have
                # removed the pod. Still delete the exact sandbox record.
        try:
            self.client.delete(handle.id)
        except PyroMindAPIError as exc:
            if exc.status_code != 404:
                raise
        self._event("deleted", handle)

    def container_exists(self, name):
        return any(s.name == name for s in self.client.list())

    def remove_container_by_name(self, name):
        for sb in self.client.list():
            if sb.name == name:
                self.cleanup_container(PyroMindHandle(sb.id, sb.name))

    def image_exists(self, image_key):
        raise NotImplementedError("PyroMind has no image inspection API; create validates pullability")

    def copy_to_container(self, handle, src, dst):
        src, dst = Path(src), PurePosixPath(dst)
        if not dst.is_absolute():
            raise ValueError("destination must be absolute")
        self._checked(handle, ["mkdir", "-p", str(dst.parent)])
        if src.is_file():
            self.client.write_file(handle.id, str(dst), src)
            self._checked(handle, ["chmod", oct(src.stat().st_mode & 0o777)[2:], str(dst)])
            return
        remote = f"/tmp/sforge-copy-{uuid.uuid4().hex}.tar"
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "payload.tar"
            with tarfile.open(archive, "w") as bundle:
                bundle.add(src, arcname=dst.name)
            self.client.write_file(handle.id, remote, archive)
        try:
            self._checked(handle, ["tar", "xf", remote, "-C", str(dst.parent)])
        finally:
            self.client.delete_file(handle.id, remote)

    def copy_from_container(self, handle, src):
        src = PurePosixPath(src)
        if not src.is_absolute() or src.name in {"", ".", ".."}:
            raise ValueError("source must be an absolute file or directory")
        remote = f"/tmp/sforge-download-{uuid.uuid4().hex}.tar"
        self._checked(handle, ["tar", "cf", remote, "-C", str(src.parent), "--", src.name])
        try:
            return self.client.read_file(handle.id, remote)
        finally:
            self.client.delete_file(handle.id, remote)

    def write_to_container(self, handle, data, dst):
        dst = PurePosixPath(dst)
        if not dst.is_absolute():
            raise ValueError("destination must be absolute")
        self._checked(handle, ["mkdir", "-p", str(dst.parent)])
        self.client.write_file(handle.id, str(dst), data.encode())

    def _command(self, handle, cmd, user, environment, workdir=None):
        if user not in (None, "root", "0"):
            raise NotImplementedError("PyroMind pilot supports root exec only")
        command = shlex.join(cmd) if isinstance(cmd, list) else cmd
        if workdir:
            # The remote API's cwd wrapper can flatten argv and lose shell
            # quoting. Keep cwd inside our quoted shell, with no SDK cwd.
            command = f"cd -- {shlex.quote(workdir)} && " + shlex.join(["/bin/bash", "-c", command])
        env = {**handle.environment, **(environment or {})}
        self._validate_environment(env)
        return ["env", *[f"{k}={v}" for k, v in env.items()], "/bin/bash", "-c", command]

    def exec_run(self, handle, cmd, *, user=None, workdir=None, environment=None, detach=False):
        command = self._command(handle, cmd, user, environment, workdir)
        if detach:
            path = f"/tmp/sforge-detach-{uuid.uuid4().hex}.log"
            command = "nohup " + shlex.join(command) + f" >{path} 2>&1 </dev/null &"
        result = self._exec_api(handle, command, timeout=60, environment=environment)
        if result.returncode < 0:
            raise RuntimeError(self._redact_error(
                result.exception_info, {**handle.environment, **(environment or {})}))
        return ExecResult(result.output + result.stderr, result.returncode)

    def _checked(self, handle, cmd, **kwargs):
        result = self.exec_run(handle, cmd, **kwargs)
        if result.exit_code:
            output = self._redact_error(result.output, {
                **handle.environment, **(kwargs.get("environment") or {})})
            raise RuntimeError(f"Sandbox command failed ({result.exit_code}): {output}")
        return result.output

    def exec_run_with_timeout(self, handle, cmd, timeout=60, *, log_file=None,
                              user=None, workdir=None, environment=None,
                              stream_to_stdout=False, shutdown_event=None,
                              log_append=False, on_chunk=None,
                              stop_container_on_timeout=False, before_stop_on_timeout=None):
        if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
                or not math.isfinite(timeout) or timeout < 1):
            raise ValueError("PyroMind exec requires a finite timeout of at least 1 second")
        if on_chunk is not None or stream_to_stdout:
            raise NotImplementedError("PyroMind SDK has no streaming exec")
        command = self._command(handle, cmd, user, environment, workdir)
        env = {**handle.environment, **(environment or {})}
        started = time.monotonic()
        # The gateway may return HTTP 504 long before the SDK's 600s limit.
        # Keep each API call short; the sandbox owns a bounded background job.
        job = f"/tmp/sforge-exec-{uuid.uuid4().hex}"
        run = shlex.join(["timeout", "--signal=TERM", "--kill-after=5", str(timeout), *command])
        # GNU timeout creates a process group for the complete command tree.
        # Persist its PID so cancellation/transport failure can terminate that
        # group without stopping an otherwise reusable Worker sandbox.
        runner = (f"{run} >{job}/output 2>&1 & child=$!; "
                  f"printf '%s' \"$child\" >{job}/pid; wait \"$child\"; rc=$?; "
                  # A completed shell may leave background descendants. Once
                  # its main command exits those descendants must stop too.
                  "kill -TERM -- -\"$child\" 2>/dev/null || true; "
                  "kill -KILL -- -\"$child\" 2>/dev/null || true; "
                  f"printf '%s' \"$rc\" >{job}/status.tmp; mv {job}/status.tmp {job}/status")
        # Do not background mkdir itself: return only once the job has been launched.
        launch = f"mkdir -m 700 {job} && {{ nohup /bin/bash -c {shlex.quote(runner)} >{job}/runner.log 2>&1 </dev/null & }}"
        completed = False
        try:
            result = self._exec_api(handle, launch, timeout=10, environment=environment)
            if result.returncode != 0:
                raise RuntimeError("Cannot launch sandbox job: " + self._redact_error(
                    result.exception_info or result.stderr, env))
            deadline = started + timeout + 30
            while time.monotonic() < deadline:
                if shutdown_event is not None and shutdown_event.is_set():
                    self._stop_job(handle, job)
                    code = 130
                    completed = True
                    break
                status = self._exec_api(
                    handle, f"if [ -f {job}/status ]; then cat {job}/status; else printf RUNNING; fi",
                    timeout=10,
                )
                if status.returncode != 0:
                    raise RuntimeError(self._redact_error(status.exception_info or status.stderr, env))
                value = status.output.strip()
                if value != "RUNNING":
                    code = int(value)
                    completed = True
                    break
                time.sleep(1)
            if not completed:
                raise TimeoutError(f"Sandbox job status unavailable after {timeout + 30}s: {handle.id}")
            output = self.client.read_file(handle.id, f"{job}/output").decode("utf-8", errors="replace")
        finally:
            # A request failure may occur after the job was already launched.
            # Make one bounded cleanup attempt. GNU timeout remains the remote
            # deadline if the API is unreachable; never delete its live metadata.
            if not completed:
                try:
                    self._stop_job(handle, job)
                    completed = True
                except Exception:
                    pass
            if completed:
                self.client.delete_file(handle.id, job, recursive=True)
        timed_out = code in (124, 137)
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            with Path(log_file).open("a" if log_append else "w") as stream:
                stream.write(output)
        if timed_out and stop_container_on_timeout:
            if before_stop_on_timeout is None or before_stop_on_timeout() is not False:
                self.cleanup_container(handle)
        return StreamingExecResult(output, code, timed_out, time.monotonic() - started)

    def _stop_job(self, handle, job):
        # The runner can race an immediate cancel. Wait at most one second for
        # its PID, TERM the group, then KILL after at most five seconds. The
        # runner itself stays outside the group so it can write the exit status.
        command = (
            f"[ -d {job} ] || exit 0; "
            f"n=0; while [ ! -f {job}/pid ] && [ ! -f {job}/status ] && [ $n -lt 10 ]; "
            "do sleep 0.1; n=$((n+1)); done; "
            f"[ -f {job}/status ] && exit 0; "
            f"[ -f {job}/pid ] || exit 1; pid=$(cat {job}/pid); "
            "case $pid in ''|*[!0-9]*) exit 1;; esac; [ \"$pid\" -gt 1 ] || exit 1; "
            "kill -TERM -- -\"$pid\" 2>/dev/null || true; "
            f"n=0; while [ ! -f {job}/status ] && [ $n -lt 25 ]; "
            "do sleep 0.2; n=$((n+1)); done; "
            "kill -KILL -- -\"$pid\" 2>/dev/null || true; "
            f"n=0; while [ ! -f {job}/status ] && [ $n -lt 10 ]; "
            "do sleep 0.1; n=$((n+1)); done; "
            f"test -f {job}/status"
        )
        result = self._exec_api(handle, ["/bin/bash", "-c", command], timeout=10)
        if result.returncode != 0:
            raise RuntimeError("Sandbox job cancellation could not be confirmed")

    def exec_run_with_exit_code(self, handle, cmd, timeout=60, **kwargs):
        return self.exec_run_with_timeout(handle, cmd, timeout, **kwargs)

    def get_container_ip(self, handle):
        handle.internal_ip = self.client.get_internal_ip(handle.id).internal_ip
        return handle.internal_ip

    def get_container_gateway_ip(self, handle):
        return None

    def create_network_isolation(self, handle, allowed_endpoints, logger):
        raise NotImplementedError("PyroMind egress isolation is unverified; refusing to pretend it exists")
