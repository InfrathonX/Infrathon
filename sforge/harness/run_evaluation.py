"""Evaluation orchestration: extract archive in judge container, run tests, grade."""

from __future__ import annotations

import fnmatch
import gzip
import hashlib
import io
import json
import logging
import os
import re
import tarfile
import tempfile
import time
import traceback
import uuid
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Protocol

from sforge.harness.backend import (
    AcceleratorAllocation,
    ContainerBackend,
    ContainerHandle,
)
from sforge.harness.config import (
    SForgeConfig,
    get_container_env,
    validate_single_gpu_handoff_config,
    validate_judge_network_mode,
)
from sforge.harness.constants import (
    DOCKER_USER,
)
from sforge.harness.cpu_affinity import pin_before_start
from sforge.harness.docker_build import (
    BuildImageError,
    build_judge_image,
    setup_logger,
    close_logger,
)
from sforge.harness.grading import EvalReport, grade_output
from sforge.harness.score_rescale import rescale_score
from sforge.harness.task_spec import TaskSpec, validate_private_artifact_paths

if TYPE_CHECKING:
    from sforge.pyromind.candidate_mount import MountedCandidateArchive


def merge_minference_external_context(private_context, handle, protocol):
    """Copy only the trusted on-started asset install hashes for this Task."""
    if protocol != "minference-ruler-quality-v3":
        return
    values = getattr(handle, "minference_external_context", None)
    keys = {"EXTERNAL_ASSET_BINDINGS_SHA256", "EXTERNAL_ASSET_RECEIPT_SHA256"}
    if (not isinstance(values, dict) or set(values) != keys
            or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
                   for value in values.values())):
        raise ValueError("trusted MInference external asset bootstrap hashes missing or malformed")
    if not isinstance(private_context, dict) or any(
            key in private_context and private_context[key] != value for key, value in values.items()):
        raise ValueError("host private context conflicts with actual MInference asset installation")
    private_context.update(values)


class EvaluationError(Exception):
    def __init__(self, task_id: str, message: str, logger: logging.Logger):
        super().__init__(message)
        self.task_id = task_id
        self.log_file = getattr(logger, "log_file", None)


class JudgeContainerLifecycle(Protocol):
    """Trusted resource-coordinator callbacks around one Judge container."""

    def judge_annotations(self) -> dict[str, str]: ...

    def judge_started(self, handle: ContainerHandle) -> None: ...

    def judge_cleaning(self, handle: ContainerHandle | None) -> None: ...

    def judge_destroyed(self, handle: ContainerHandle | None) -> None: ...


PRIVATE_JUDGE_ARTIFACT_NAME = "private_judge_artifacts.tar.gz"
PRIVATE_JUDGE_ARTIFACT_MAX_BYTES = 64 * 1024 * 1024
PRIVATE_JUDGE_ARTIFACT_MAX_TOTAL_BYTES = 128 * 1024 * 1024
SUBMISSION_ARCHIVE_MAX_BYTES = 128 * 1024 * 1024
SUBMISSION_ARCHIVE_MAX_MEMBERS = 20_000
SUBMISSION_ARCHIVE_MAX_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
SUBMISSION_ARCHIVE_MAX_PATH_BYTES = 4096
PRIVATE_CONTEXT_ENV_PREFIX = "SFORGE_PRIVATE_CONTEXT_"
EVALUATION_CONTEXT_PATH = PurePosixPath("/tmp/.sforge-evaluation-context.json")


def _bind_portable_contract_report(
    task_spec: TaskSpec,
    report: EvalReport,
) -> None:
    """Inject trusted portable-contract identity into every terminal report."""

    if task_spec.portable is None:
        return
    report.metrics = {
        **report.metrics,
        "portable_contract": {
            "evaluation_contract_sha256": (
                task_spec.portable.evaluation_contract_sha256
            ),
            "selected_ids": task_spec.portable.defaults.to_dict(),
        },
    }


def _durable_atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Publish bytes only after the file and containing directory are durable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def _persist_report(log_dir: Path, report: EvalReport) -> None:
    """Durably and atomically persist every terminal evaluation result.

    Early failures used to return an ``EvalReport`` only in memory.  That made
    an accepted interactive submission impossible to audit or replay because
    its history row existed while ``report.json`` did not.  Keep the same
    report ABI for successful and failed evaluations and replace the file
    atomically so readers never observe a partial JSON document.
    """

    _durable_atomic_write_bytes(
        log_dir / "report.json",
        (
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n"
        ).encode("utf-8"),
    )


def _canonical_submission_member_name(raw_name: str) -> str:
    """Return one safe POSIX archive name without silently normalizing it."""

    if not isinstance(raw_name, str) or not raw_name or "\x00" in raw_name:
        raise ValueError("submission archive member name is empty or contains NUL")
    try:
        encoded_name = raw_name.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("submission archive member path is not UTF-8") from exc
    if len(encoded_name) > SUBMISSION_ARCHIVE_MAX_PATH_BYTES:
        raise ValueError("submission archive member path is too long")
    if raw_name.startswith("/"):
        raise ValueError("submission archive contains an absolute path")

    # A single trailing slash is the conventional spelling of a directory.
    # Every other component must already be canonical; do not turn an unsafe
    # candidate-controlled path into a different trusted extraction path.
    name = raw_name[:-1] if raw_name.endswith("/") else raw_name
    if not name:
        raise ValueError("submission archive contains an empty path")
    components = name.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise ValueError("submission archive member path is not canonical")
    canonical = str(PurePosixPath(*components))
    if canonical != name:
        raise ValueError("submission archive member path changed during normalization")
    return canonical


def _validate_submission_archive(archive: bytes) -> list[str]:
    """Validate untrusted tar bytes before they enter the Judge container.

    Only ordinary files and directories are accepted.  Links and special
    files make a root extraction depend on ordering and filesystem state, so
    they are rejected rather than delegated to the container's ``tar``.
    The returned names are the exact, canonical regular-file members that may
    subsequently be passed through the task whitelist.
    """

    if len(archive) > SUBMISSION_ARCHIVE_MAX_BYTES:
        raise ValueError("submission archive exceeds the compressed size limit")

    regular_files: list[str] = []
    seen: set[str] = set()
    total_size = 0
    member_count = 0
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as bundle:
            for member in bundle:
                member_count += 1
                if member_count > SUBMISSION_ARCHIVE_MAX_MEMBERS:
                    raise ValueError("submission archive has too many members")

                canonical = _canonical_submission_member_name(member.name)
                if canonical in seen:
                    raise ValueError("submission archive contains a duplicate path")
                seen.add(canonical)

                if member.type == tarfile.DIRTYPE:
                    if member.size != 0:
                        raise ValueError("submission archive directory has file data")
                    continue
                if member.type not in {tarfile.REGTYPE, tarfile.AREGTYPE}:
                    raise ValueError(
                        "submission archive contains a link or special file"
                    )
                if member.size < 0:
                    raise ValueError("submission archive member has a negative size")
                total_size += member.size
                if total_size > SUBMISSION_ARCHIVE_MAX_UNCOMPRESSED_BYTES:
                    raise ValueError(
                        "submission archive exceeds the uncompressed size limit"
                    )

                stream = bundle.extractfile(member)
                if stream is None:
                    raise ValueError("submission archive member is unreadable")
                observed = 0
                while True:
                    chunk = stream.read(min(1024 * 1024, member.size - observed + 1))
                    if not chunk:
                        break
                    observed += len(chunk)
                    if observed > member.size:
                        raise ValueError("submission archive member exceeds its declared size")
                if observed != member.size:
                    raise ValueError("submission archive member is truncated")
                regular_files.append(canonical)
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise ValueError("submission archive is not a valid tar file") from exc
    return regular_files


def _persist_private_judge_artifact(
    backend: ContainerBackend,
    handle,
    container_path: str | list[str],
    log_dir: Path,
    *,
    optional_paths: list[str] | None = None,
) -> Path:
    """Package trusted Judge-only files without putting them in feedback.

    ``copy_from_container`` returns a transport tar stream for both Docker and
    Kubernetes. Each request must unwrap exactly one regular file. Bound each
    file and the total before creating one normalized, atomic tar.gz. Optional
    extra outputs may be absent (e.g. rejection before candidate execution),
    but present outputs must pass the same validation. A legacy string
    argument retains the original single-file archive. The Judge
    API/report never receives the private bytes or archive metadata.
    """

    paths = validate_private_artifact_paths(
        [container_path] if isinstance(container_path, str) else container_path)
    if not paths:
        raise ValueError('private Judge artifact collection must contain a file')
    optional = set(validate_private_artifact_paths(optional_paths or []))
    if not optional.issubset(paths) or optional == set(paths):
        raise ValueError('optional private artifacts must leave a required primary file')
    payloads = []
    total = 0
    for path in paths:
        source = PurePosixPath(path)
        if path in optional:
            # Inspect as the trusted backend user, not the candidate. Include
            # dangling symlinks as present so they cannot evade type checks.
            # An explicit marker also works with Kubernetes exec backends
            # which cannot propagate the remote process exit status.
            presence = backend.exec_run(
                handle,
                ['/bin/sh', '-c',
                 'if [ -e "$1" ] || [ -L "$1" ]; then printf present; else printf absent; fi',
                 'sforge-private-artifact', str(source)],
                user='root',
            )
            if presence.exit_code != 0 or presence.output not in {'present', 'absent'}:
                raise ValueError('private Judge optional artifact presence could not be checked')
            if presence.output == 'absent':
                continue
        transport = backend.copy_from_container(handle, source)
        with tarfile.open(fileobj=io.BytesIO(transport), mode="r:*") as wrapper:
            members = wrapper.getmembers()
            if len(members) != 1 or not members[0].isfile():
                raise ValueError(
                    "private Judge artifact transport must contain exactly one regular file"
                )
            member = members[0]
            if member.name not in {source.name, str(source).lstrip('/'), str(source)}:
                raise ValueError("private Judge artifact transport name does not match")
            if member.size <= 0 or member.size > PRIVATE_JUDGE_ARTIFACT_MAX_BYTES:
                raise ValueError(
                    "private Judge artifact exceeds the allowed size or is empty"
                )
            total += member.size
            if total > PRIVATE_JUDGE_ARTIFACT_MAX_TOTAL_BYTES:
                raise ValueError('private Judge artifacts exceed the total size limit')
            stream = wrapper.extractfile(member)
            if stream is None:
                raise ValueError("private Judge artifact transport file is unreadable")
            payload = stream.read(PRIVATE_JUDGE_ARTIFACT_MAX_BYTES + 1)
        if len(payload) != member.size:
            raise ValueError("private Judge artifact transport was truncated")
        payloads.append((source.name, payload))

    uncompressed = io.BytesIO()
    with tarfile.open(fileobj=uncompressed, mode="w") as archive:
        for name, payload in payloads:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o600
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            archive.addfile(info, io.BytesIO(payload))
    private_archive = gzip.compress(
        uncompressed.getvalue(),
        compresslevel=9,
        mtime=0,
    )
    if len(private_archive) > SUBMISSION_ARCHIVE_MAX_BYTES:
        raise ValueError('private Judge artifact archive exceeds the retained size limit')

    destination = log_dir / PRIVATE_JUDGE_ARTIFACT_NAME
    _durable_atomic_write_bytes(destination, private_archive)
    return destination


def _effective_private_artifact_paths(task_spec, primary_override, paths_override):
    """Resolve an evidence set; an explicit list also permits opt-out."""
    if paths_override is not None:
        return validate_private_artifact_paths(paths_override)
    configured = getattr(task_spec.judge, 'private_artifact_paths', None)
    primary = task_spec.judge.private_artifact_path
    if primary_override is not None and primary_override != primary:
        return validate_private_artifact_paths([primary_override])
    if configured is not None:
        return validate_private_artifact_paths(configured)
    selected = primary_override if primary_override is not None else primary
    return validate_private_artifact_paths([selected] if selected is not None else [])


def _filter_archive_files(
    file_list: list[str],
    submit_paths: list[str],
    submit_exclude: list[str],
) -> list[str]:
    """Filter archive entries against submit_paths (whitelist) and submit_exclude (blacklist).

    A file is allowed if:
      1. It falls under at least one submit_paths entry (prefix match, or "." allows all)
      2. It does NOT match any submit_exclude pattern (prefix or fnmatch glob)
      3. Its path remains canonical at this trust boundary
    """
    allowed: list[str] = []
    for f in file_list:
        # Callers normally pass names returned by
        # ``_validate_submission_archive``. Recheck the invariant so a future
        # caller cannot accidentally revive string-prefix traversal.
        f = _canonical_submission_member_name(f)

        # Whitelist: must be under at least one submit_paths entry
        in_whitelist = False
        for sp in submit_paths:
            sp = sp.rstrip("/")
            if sp == ".":
                in_whitelist = True
                break
            if f == sp or f.startswith(sp + "/"):
                in_whitelist = True
                break
        if not in_whitelist:
            continue

        # Blacklist: reject if any exclude pattern matches
        excluded = False
        for exc in submit_exclude:
            exc = exc.rstrip("/")
            if f == exc or f.startswith(exc + "/"):
                excluded = True
                break
            if any(fnmatch.fnmatch(component, exc) for component in f.split("/")):
                excluded = True
                break
            if fnmatch.fnmatch(f, exc):
                excluded = True
                break
        if excluded:
            continue

        allowed.append(f)
    return allowed


def judge_submission(
    task_spec: TaskSpec,
    archive: bytes,
    config: SForgeConfig,
    backend: ContainerBackend,
    submission_id: str | None = None,
    timeout: int | None = None,
    log_dir: Path | None = None,
    verbose: bool = False,
    eval_cmd: str | None = None,
    private_artifact_path: str | None = None,
    run_id: str | None = None,
    round_id: str | None = None,
    judge_image_ref: str | None = None,
    candidate_version: int | None = None,
    accelerator_allocation: AcceleratorAllocation | None = None,
    container_lifecycle: JudgeContainerLifecycle | None = None,
    mounted_archive: MountedCandidateArchive | None = None,
    private_artifact_paths: list[str] | None = None,
    evaluation_phase: str = "interactive",
) -> EvalReport:
    """
    Grade an archive submission in an ephemeral judge container.

    1. Ensure judge image exists
    2. Create ephemeral container from judge image
    3. Extract archive (tar.gz) into patch_dir
    4. Inject eval.sh into container
    5. Run eval script
    6. Parse output, compute score
    7. Cleanup container

    Args:
        archive: Raw bytes of a tar.gz containing submitted files.
        log_dir: Override log output directory. If None, defaults to
                 logs/runs/<submission_id>/<task_id>/submissions/001/
        eval_cmd: Optional task-declared evaluator command override (used by
                  the isolated public diagnostic plane).
        private_artifact_path: Optional matching private evidence path.
        private_artifact_paths: Optional complete private evidence set. An
                  empty list explicitly disables collection (public evals).
        run_id: Optional authoritative run identifier supplied by the Judge
                server.  It is exposed only to the ephemeral Judge container.
        round_id: Optional authoritative submission round (for example
                  ``agent-3`` or ``manual-1``), likewise Judge-only.
        candidate_version: Optional host-side override of the submission's own
                  version selector (for example ``FINAL_VERSION``), used by
                  per-version replay to score every retained implementation in
                  one archive. Judge-only; adapters that do not understand it
                  ignore the extra context field.
        judge_image_ref: Optional immutable image reference already resolved by
                         a replay controller.  When supplied, SForge skips the
                         mutable-tag build path and executes this exact ref.
        accelerator_allocation: Exact deployment allocation for an explicitly
                         coordinated serial Work/Judge handoff. Legacy callers
                         omit it and retain their current device behavior.
        container_lifecycle: Coordinator callbacks. When supplied, Judge
                         cleanup uses strict stop/remove/absence verification;
                         callback failures invalidate the evaluation.
        mounted_archive: Controller-resolved immutable PyroMind version. The
                         Judge mounts and verifies this exact archive instead
                         of uploading its bytes through the exec/file API.
    """
    validate_single_gpu_handoff_config(config)
    if config.single_gpu_handoff and container_lifecycle is None:
        raise ValueError(
            "single GPU evaluation requires a live coordinator lifecycle; "
            "direct local Judge execution is forbidden"
        )
    if container_lifecycle is not None and accelerator_allocation is None:
        raise ValueError(
            "coordinated Judge lifecycle requires an exact accelerator allocation"
        )
    if submission_id is None:
        submission_id = uuid.uuid4().hex[:12]

    submitted_at = time.time()
    timeout = timeout or task_spec.judge.eval_timeout
    effective_eval_cmd = eval_cmd or task_spec.judge.eval_cmd
    effective_private_artifact_paths = _effective_private_artifact_paths(
        task_spec, private_artifact_path, private_artifact_paths)
    primary_private_artifact_path = (
        private_artifact_path if private_artifact_path is not None
        else task_spec.judge.private_artifact_path)
    optional_private_artifact_paths = (
        [path for path in effective_private_artifact_paths
         if path != primary_private_artifact_path]
        if primary_private_artifact_path in effective_private_artifact_paths else [])

    # Set up logging
    if log_dir is None:
        log_dir = (
            config.log_dir / "runs" / submission_id
            / task_spec.task_id / "submissions" / "1"
        )
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        f"judge.{task_spec.task_id}.{submission_id}",
        log_dir / "run_instance.log",
        verbose=verbose,
    )

    handle = None
    report: EvalReport | None = None
    try:
        # Persist the exact uploaded bytes before validation.  A malformed or
        # newly-incompatible archive must still remain replayable for audit
        # and recovery; writing opaque bytes to this fixed host path does not
        # extract or otherwise trust candidate-controlled member names.
        local_tar = log_dir / "submission.tar.gz"
        if archive:
            if len(archive) > SUBMISSION_ARCHIVE_MAX_BYTES:
                raise ValueError("submission archive exceeds the compressed size limit")
            _durable_atomic_write_bytes(local_tar, archive)
            logger.info(f"Archive saved ({len(archive)} bytes)")

        # The upload is candidate-controlled and the Judge runs as root with
        # task-specific trusted mounts. Validate the exact bytes on the host
        # before creating a container or copying/extracting the archive.
        archive_files = _validate_submission_archive(archive) if archive else []
        if mounted_archive is not None:
            from sforge.pyromind.candidate_mount import MountedCandidateArchive
            if backend.backend_name != 'pyromind':
                raise ValueError('Mounted candidate transport requires the PyroMind backend')
            if not isinstance(mounted_archive, MountedCandidateArchive):
                raise TypeError('mounted_archive must be a trusted candidate reference')
            mounted_archive.validate_archive(archive, task_id=task_spec.task_id)

        # 1. Ensure the legacy mutable image exists, or use the immutable ref
        # resolved by the replay controller.  Crucially, the same immutable ref
        # is passed to create_container; recording an ID while executing a tag
        # would not pin the evaluator.
        if judge_image_ref is not None and (
            not isinstance(judge_image_ref, str) or not judge_image_ref
        ):
            raise ValueError("judge_image_ref must be a nonempty string")
        if judge_image_ref is None and backend.backend_name == "docker":
            from sforge.harness.backend.docker_backend import DockerBackend
            assert isinstance(backend, DockerBackend)
            build_judge_image(task_spec, config, backend.client, force_rebuild=False)
        execution_image = judge_image_ref or task_spec.judge_image_key
        private_bootstrap = getattr(task_spec.judge, "private_bootstrap", None)
        requested_evaluation_phase = evaluation_phase
        if private_bootstrap == "minference-ruler-quality-v3":
            from sforge.harness.private_judge_bootstrap import effective_private_phase
            evaluation_phase = effective_private_phase(private_bootstrap, evaluation_phase)
        if private_bootstrap is not None and evaluation_phase != "public":
            if backend.backend_name != "docker":
                raise NotImplementedError(
                    f"{backend.backend_name} lacks the actual live image/physical-GPU "
                    "receipt required by this private Judge bootstrap"
                )
            if accelerator_allocation is None:
                raise ValueError("private Judge bootstrap requires an exact trusted accelerator allocation")
            resolved = backend.resolve_image_for_execution(execution_image)
            if (resolved.identity_kind != "docker-image-id"
                    or resolved.execution_ref != resolved.content_identity):
                raise ValueError("private Judge bootstrap requires an immutable Docker image execution ref")
            execution_image = resolved.execution_ref

        # 2. Create + start ephemeral container
        container_name = f"{task_spec.benchmark_name}.judge.{task_spec.task_id}.{submission_id}"
        env = get_container_env(config, include_judge_extra=True)
        env.setdefault("SFORGE_JUDGE_URL", "http://host.docker.internal:8080")
        # Values with this prefix are a host-to-Judge private channel, never
        # container environment.  Candidate workers can read their initial
        # /proc environment even after Python clears os.environ, so trusted
        # round metadata and task secrets are installed as a root-only file.
        private_context = {
            key.removeprefix(PRIVATE_CONTEXT_ENV_PREFIX): env.pop(key)
            for key in list(env)
            if key.startswith(PRIVATE_CONTEXT_ENV_PREFIX)
        }
        if evaluation_phase not in {"interactive", "final", "public"}:
            raise ValueError("unsupported authoritative evaluation phase")
        # Host-controlled, root-only provenance. Candidate archive contents and
        # worker environment cannot opt into terminal quality evaluation.
        private_context["EVALUATION_PHASE"] = evaluation_phase
        if private_bootstrap == "minference-ruler-quality-v3":
            private_context["REQUESTED_EVALUATION_PHASE"] = requested_evaluation_phase
        # Bind every scored snapshot, including interactive submissions, to
        # the exact archive accepted by the host. Keep this root-only: it is
        # provenance for the verifier, never a candidate-supplied environment.
        if archive:
            archive_sha256 = hashlib.sha256(archive).hexdigest()
            claimed = private_context.get("ARCHIVE_SHA256")
            if claimed is not None and claimed != archive_sha256:
                raise ValueError("Private archive identity differs from submitted bytes")
            private_context["ARCHIVE_SHA256"] = archive_sha256
        # Resolve resource limits: config > task defaults
        cpu = config.judge_cpu_limit if config.judge_cpu_limit is not None else task_spec.judge.cpu_limit
        mem = config.judge_mem_limit if config.judge_mem_limit is not None else task_spec.judge.mem_limit
        judge_network_mode = validate_judge_network_mode(
            config.judge_network_mode
        )
        if private_bootstrap is not None and evaluation_phase != "public":
            judge_network_mode = "none"
        create_kwargs: dict[str, object] = {}
        if mounted_archive is not None:
            create_kwargs['mounted_archive'] = mounted_archive
        if judge_network_mode is None and backend.backend_name != "pyromind":
            create_kwargs["extra_hosts"] = {
                "host.docker.internal": "host-gateway"
            }
        elif judge_network_mode is not None:
            if backend.backend_name != "docker":
                raise NotImplementedError(
                    f"{backend.backend_name} backend does not support Judge "
                    f"network mode {judge_network_mode!r}"
                )
            create_kwargs["network_mode"] = judge_network_mode
        if accelerator_allocation is not None:
            create_kwargs["accelerator_allocation"] = accelerator_allocation
        if container_lifecycle is not None:
            create_kwargs["annotations"] = (
                container_lifecycle.judge_annotations()
            )
        handle = backend.create_container(
            execution_image,
            container_name,
            environment=env,
            cpu_limit=cpu,
            mem_limit=mem,
            user=DOCKER_USER,
            **create_kwargs,
        )
        # Opt-in CPU pinning, a no-op when off.  A failure here is handled
        # like a start failure: invalid report, the created Judge destroyed.
        pin_before_start(
            backend,
            handle,
            role="judge",
            config=config,
            record_path=log_dir / "cpu_affinity.json",
            logger=logger,
        )
        backend.start_container(handle)
        if container_lifecycle is not None:
            container_lifecycle.judge_started(handle)
        logger.info(f"Judge container started: {handle.id}")
        if private_bootstrap is not None:
            from sforge.harness.private_judge_bootstrap import install_private_bootstrap
            try:
                merge_minference_external_context(private_context, handle, private_bootstrap)
                install_private_bootstrap(
                    backend, handle, protocol=private_bootstrap, phase=evaluation_phase,
                    execution_image=execution_image,
                    accelerator_allocation=accelerator_allocation,
                    private_context=private_context, log_dir=log_dir,
                )
            except Exception as exc:
                # Backend/cache error text may contain host-private locations.
                # Keep the public failure opaque and do not serialize receipts.
                logger.error("Private runtime bootstrap failed: %s", type(exc).__name__)
                raise EvaluationError(task_spec.task_id,
                    "Private runtime identity bootstrap failed closed", logger) from None
        mounted_tar_path = None
        if mounted_archive is not None:
            mounted_tar_path = backend.verify_mounted_archive(handle, mounted_archive)
            if mounted_tar_path != mounted_archive.archive_path:
                raise ValueError('Judge returned an unexpected mounted candidate path')
            logger.info('Candidate archive verified through read-only Storage mount: %s',
                        mounted_archive.version_id)

        context_payload = {
            "schema_version": 1,
            "submission_id": submission_id,
            "run_id": run_id,
            "round_id": round_id,
            "private": private_context,
        }
        if candidate_version is not None:
            context_payload["candidate_version"] = int(candidate_version)
        context_fd, context_name = tempfile.mkstemp(
            prefix=".evaluation-context-", suffix=".json", dir=log_dir
        )
        context_local = Path(context_name)
        try:
            os.fchmod(context_fd, 0o600)
            with os.fdopen(context_fd, "w", encoding="utf-8") as stream:
                json.dump(context_payload, stream, sort_keys=True, allow_nan=False)
                stream.write("\n")
            backend.copy_to_container(handle, context_local, EVALUATION_CONTEXT_PATH)
        finally:
            context_local.unlink(missing_ok=True)
        context_permissions = backend.exec_run(
            handle,
            [
                "/bin/sh",
                "-c",
                f"chown root:root {EVALUATION_CONTEXT_PATH} && chmod 600 {EVALUATION_CONTEXT_PATH}",
            ],
            user=DOCKER_USER,
        )
        if context_permissions.exit_code != 0:
            logger.error(
                "Private evaluation context permission install failed: exit=%s output=%r",
                context_permissions.exit_code,
                context_permissions.output,
            )
            raise EvaluationError(
                task_spec.task_id,
                "Failed to install private evaluation context",
                logger,
            )

        # 3. Extract archive into patch_dir (filtered by submit_paths/submit_exclude)
        if archive:
            tar_path = mounted_tar_path or PurePosixPath("/tmp/submission.tar.gz")
            if mounted_archive is None:
                backend.copy_to_container(handle, local_tar, tar_path)

            allowed_files = _filter_archive_files(
                archive_files, task_spec.submit_paths, task_spec.submit_exclude,
            )

            rejected = set(archive_files) - set(allowed_files)
            if rejected:
                logger.warning(f"Archive path filter rejected {len(rejected)} file(s): {sorted(rejected)[:20]}")

            if allowed_files:
                # A NUL-delimited verbatim list is immune to newline names and
                # GNU tar's option parsing for names beginning with '-'.
                local_file_list = log_dir / "allowed_files.txt"
                local_file_list.write_bytes(
                    b"\0".join(name.encode("utf-8") for name in allowed_files)
                    + b"\0"
                )
                backend.copy_to_container(handle, local_file_list, PurePosixPath("/tmp/allowed_files.txt"))

                val = backend.exec_run(
                    handle,
                    (
                        f"tar xzf {tar_path} --no-same-owner "
                        "--null --verbatim-files-from "
                        "-T /tmp/allowed_files.txt"
                    ),
                    workdir=task_spec.cwd,
                    user=DOCKER_USER,
                )
                if val.exit_code != 0:
                    err_msg = val.output
                    logger.error(f"Archive extraction failed: {err_msg}")
                    raise EvaluationError(
                        task_spec.task_id,
                        f"Archive extraction failed: {err_msg}",
                        logger,
                    )
                # Candidate identity deliberately ignores tar transport mtimes.
                # Normalize every extracted regular file and each of its parent
                # directories so the same candidate does not observe a different
                # st_mtime merely because it was replayed later.  The validated,
                # NUL-delimited list keeps unusual but legal names unambiguous.
                metadata_result = backend.exec_run(
                    handle,
                    [
                        "/bin/bash",
                        "-c",
                        """set -euo pipefail
while IFS= read -r -d '' path; do
  touch -h -d @0 -- "$path"
  parent="${path%/*}"
  while [[ "$parent" != "$path" ]]; do
    touch -h -d @0 -- "$parent"
    [[ "$parent" == */* ]] || break
    path="$parent"
    parent="${path%/*}"
  done
done < /tmp/allowed_files.txt
""",
                    ],
                    workdir=task_spec.cwd,
                    user=DOCKER_USER,
                )
                if metadata_result.exit_code != 0:
                    err_msg = metadata_result.output
                    logger.error(
                        "Archive metadata normalization failed: %s", err_msg
                    )
                    raise EvaluationError(
                        task_spec.task_id,
                        f"Archive metadata normalization failed: {err_msg}",
                        logger,
                    )
                logger.info(f"Archive extracted: {len(allowed_files)} file(s) allowed, {len(rejected)} rejected")
            else:
                logger.warning("All archive entries were filtered out — running eval on unmodified skeleton")
        else:
            logger.info("Empty archive — running eval on unmodified skeleton")

        # 4. Inject eval.sh dynamically (not baked into the image)
        eval_script_local = log_dir / "eval.sh"
        eval_script_local.write_text(task_spec.eval_script_for(effective_eval_cmd))
        backend.copy_to_container(handle, eval_script_local, PurePosixPath("/tmp/eval.sh"))

        # 5. Run eval script
        logger.info("Running eval script...")
        result = backend.exec_run_with_timeout(
            handle, "/bin/bash /tmp/eval.sh", timeout
        )
        test_output, timed_out, runtime = result.output, result.timed_out, result.elapsed_seconds

        private_artifact_error: Exception | None = None
        if effective_private_artifact_paths:
            try:
                artifact = _persist_private_judge_artifact(
                    backend,
                    handle,
                    effective_private_artifact_paths,
                    log_dir,
                    optional_paths=optional_private_artifact_paths,
                )
                logger.info(
                    "Private Judge artifact saved: %s (%d bytes)",
                    artifact,
                    artifact.stat().st_size,
                )
            except Exception as exc:
                # Preserve the authoritative Judge report even if an abnormal
                # verifier exit left no artifact. Formal task launchers may
                # separately require this file for completed boundary runs.
                logger.warning("Private Judge artifact was not collected: %s", exc)
                private_artifact_error = exc

        # Save test output
        (log_dir / "test_output.txt").write_text(test_output)
        logger.info(f"Test runtime: {runtime:.2f}s, timed_out: {timed_out}")

        # 6. Grade
        report = grade_output(
            task_spec, test_output, submission_id, timed_out, runtime
        )
        if result.exit_code != 0 or timed_out or private_artifact_error is not None:
            # A verifier process failure, timeout, or missing required private
            # evidence is an infrastructure-invalid evaluation regardless of
            # any partial structured output it happened to print.
            report.valid = False
            report.score = 0.0
            report.errors = max(1, report.errors)
            report.summary = "Evaluation infrastructure did not complete authoritatively"
            report.metrics = {
                **report.metrics,
                "evaluation_status": "infrastructure_error",
                "correctness_gate_passed": False,
                "primary_score_value": 0.0,
            }
        report.submitted_at = submitted_at
        report.score_0_100 = rescale_score(task_spec.judge.rescale, report.score)
        _bind_portable_contract_report(task_spec, report)
        _persist_report(log_dir, report)
        logger.info(
            f"Result: {report.passed}/{report.total_tests} passed "
            f"(pass_rate={report.pass_rate:.2%})"
        )
        return report

    except (EvaluationError, BuildImageError) as e:
        logger.error(traceback.format_exc())
        report = EvalReport(
            task_id=task_spec.task_id,
            submission_id=submission_id,
            submitted_at=submitted_at,
            raw_output=str(e),
            errors=1,
            summary="Evaluation infrastructure failed before grading completed",
            metrics={
                "evaluation_status": "infrastructure_error",
                "correctness_gate_passed": False,
                "primary_score_value": 0.0,
            },
        )
        _bind_portable_contract_report(task_spec, report)
        _persist_report(log_dir, report)
        return report
    except Exception as e:
        logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        report = EvalReport(
            task_id=task_spec.task_id,
            submission_id=submission_id,
            submitted_at=submitted_at,
            raw_output=str(e),
            errors=1,
            summary="Evaluation infrastructure failed before grading completed",
            metrics={
                "evaluation_status": "infrastructure_error",
                "correctness_gate_passed": False,
                "primary_score_value": 0.0,
            },
        )
        _bind_portable_contract_report(task_spec, report)
        _persist_report(log_dir, report)
        return report
    finally:
        try:
            cleanup_errors: list[Exception] = []
            if container_lifecycle is None:
                try:
                    backend.cleanup_container(handle, logger)
                except Exception as exc:
                    cleanup_errors.append(exc)
            else:
                assert container_lifecycle is not None
                try:
                    container_lifecycle.judge_cleaning(handle)
                except Exception as exc:
                    cleanup_errors.append(exc)
                destroyed = handle is None
                if handle is not None:
                    try:
                        backend.destroy_container_for_handoff(handle)
                        destroyed = True
                    except Exception as exc:
                        cleanup_errors.append(exc)
                # Never persist JUDGE_CLEANED unless exact-ID destruction was
                # successful (or container creation never occurred).  A
                # cleaning callback failure still cannot skip strict destroy.
                if destroyed:
                    try:
                        container_lifecycle.judge_destroyed(handle)
                    except Exception as exc:
                        cleanup_errors.append(exc)
            if cleanup_errors:
                details = "; ".join(
                    f"{type(error).__name__}: {error}"
                    for error in cleanup_errors
                )
                raise RuntimeError(
                    f"Judge cleanup failed with {len(cleanup_errors)} "
                    f"error(s): {details}"
                ) from cleanup_errors[0]
        except Exception:
            # Task adapters use cleanup as the post-attempt attestation point.
            # A failure here must override any optimistic Judge result without
            # escaping past the replay state machine and leaving a permanent
            # ``running`` measurement.  Keep private adapter details in the
            # host log, never in the persisted/public report.
            logger.error(
                "Judge cleanup or post-evaluation attestation failed\n%s",
                traceback.format_exc(),
            )
            if report is None:
                report = EvalReport(
                    task_id=task_spec.task_id,
                    submission_id=submission_id,
                    submitted_at=submitted_at,
                )
            report.valid = False
            report.score = 0.0
            report.errors = max(1, report.errors)
            report.summary = (
                "Evaluation infrastructure failed during cleanup or "
                "post-evaluation attestation"
            )
            report.metrics = {
                **report.metrics,
                "evaluation_status": "infrastructure_error",
                "correctness_gate_passed": False,
                "primary_score_value": 0.0,
            }
            report.score_0_100 = rescale_score(
                task_spec.judge.rescale,
                report.score,
            )
            try:
                _bind_portable_contract_report(task_spec, report)
                _persist_report(log_dir, report)
            except Exception:
                logger.error(
                    "Failed to persist cleanup infrastructure report\n%s",
                    traceback.format_exc(),
                )
        finally:
            close_logger(logger)
