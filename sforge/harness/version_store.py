"""Persistent, host-only storage for replayable agent submission versions."""

from __future__ import annotations

import fcntl
import fnmatch
import hashlib
import io
import json
import os
import re
import shutil
import stat
import tarfile
import uuid
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sforge.harness.run_evaluation import (
    SUBMISSION_ARCHIVE_MAX_BYTES,
    SUBMISSION_ARCHIVE_MAX_MEMBERS,
    SUBMISSION_ARCHIVE_MAX_UNCOMPRESSED_BYTES,
    _filter_archive_files,
    _validate_submission_archive,
)


STORE_MARKER_NAME = ".sforge-version-store.json"
STORE_LOCK_NAME = ".sforge-version-store.lock"
STORE_SCHEMA_VERSION = 1
VERSION_SOURCES = frozenset({"checkpoint", "agent", "public", "auto", "final"})

_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}")
_SAFE_VERSION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}")
_LABEL_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_FILE_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_MAX_MANIFEST_BYTES = 1024 * 1024


class VersionStoreError(RuntimeError):
    """The configured version store or requested write is invalid."""


class VersionStoreCandidateError(VersionStoreError):
    """Candidate-controlled label or archive bytes are invalid."""


class VersionStoreCollision(VersionStoreError):
    """One logical version event was reused for different archive bytes."""


class VersionStoreWriteError(VersionStoreError):
    """A validated version could not be made durable on the host."""


class VersionStoreQuotaExceeded(VersionStoreError):
    """A bounded source quota would be exceeded by this version."""


@dataclass(frozen=True)
class _TaskRegistration:
    task_id: str
    task_directory: str
    path: Path


@dataclass(frozen=True)
class StoredVersion:
    version_id: str
    task_id: str
    task_directory: str
    run_id: str
    event_sequence: int | None
    candidate_sequence: int | None
    source: str
    round_id: str | None
    label: str | None
    created_at: str
    archive_sha256: str
    content_sha256: str | None
    candidate_id: str | None
    archive_size: int
    member_count: int
    path: Path
    metadata: dict[str, Any]
    content_identity_exclude: tuple[str, ...] = ()

    @property
    def archive_path(self) -> Path:
        return self.path / "submission.tar.gz"


@dataclass(frozen=True)
class StoredCandidate:
    """One logical submitted-file state with one or more archive events."""

    candidate_id: str
    content_sha256: str
    sequence: int
    task_id: str
    task_directory: str
    run_id: str
    created_at: str
    label: str | None
    events: tuple[StoredVersion, ...]

    @property
    def archive_path(self) -> Path:
        # Keep the representative exact bytes stable.  A checkpoint recorded
        # later for identical normalized content must not silently change a
        # previously listed candidate archive or replay plan.
        return self.events[0].archive_path


@dataclass(frozen=True)
class StoredTimelineEvent:
    """One physical archive event annotated with its logical-code history."""

    event_sequence: int
    version: StoredVersion
    candidate_sequence: int
    candidate_id: str
    previous_candidate_id: str | None
    relation: str
    first_seen_event_sequence: int


def _candidate_id(content_sha256: str) -> str:
    return f"candidate-{content_sha256}"


def _matches_identity_exclude(name: str, patterns: Sequence[str]) -> bool:
    """Match one archive member against candidate-identity exclusions.

    Uses the same pattern language as ``submit_exclude``: exact path, directory
    prefix, per-component fnmatch, and whole-path fnmatch.
    """

    for raw_pattern in patterns:
        pattern = raw_pattern.rstrip("/")
        if not pattern:
            continue
        if name == pattern or name.startswith(pattern + "/"):
            return True
        if any(
            fnmatch.fnmatch(component, pattern)
            for component in name.split("/")
        ):
            return True
        if fnmatch.fnmatch(name, pattern):
            return True
    return False


def _submission_content_sha256(
    archive: bytes,
    *,
    identity_exclude: Sequence[str] = (),
) -> str:
    """Hash the extracted Judge-visible tree, independent of tar/gzip metadata.

    Regular-file path, executable/permission mode, size, and contents are
    material.  Archive ordering, mtimes, owners, and compression headers are
    transport metadata and deliberately do not affect candidate identity.
    Directory entries are ignored because Judge extraction selects only the
    validated regular-file member list.  Members matching ``identity_exclude``
    stay in the archive but do not contribute to candidate identity, so trees
    differing only in score-irrelevant files (task notes, evidence logs) map to
    one logical candidate.
    """

    records: list[tuple[bytes, int, int, bytes]] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
            for member in bundle:
                if not member.isfile():
                    continue
                if identity_exclude and _matches_identity_exclude(
                    member.name, identity_exclude
                ):
                    continue
                stream = bundle.extractfile(member)
                if stream is None:
                    raise VersionStoreCandidateError(
                        "invalid submission archive: member is unreadable"
                    )
                file_digest = hashlib.sha256()
                observed = 0
                while True:
                    chunk = stream.read(1024 * 1024)
                    if not chunk:
                        break
                    file_digest.update(chunk)
                    observed += len(chunk)
                if observed != member.size:
                    raise VersionStoreCandidateError(
                        "invalid submission archive: member is truncated"
                    )
                records.append(
                    (
                        member.name.encode("utf-8"),
                        member.mode & 0o7777,
                        member.size,
                        file_digest.digest(),
                    )
                )
    except (tarfile.TarError, EOFError, OSError, UnicodeError) as exc:
        raise VersionStoreCandidateError(
            "invalid submission archive: cannot compute candidate identity"
        ) from exc

    tree_digest = hashlib.sha256(b"sforge-candidate-tree-v1\0")
    for path, mode, size, file_digest in sorted(records, key=lambda item: item[0]):
        tree_digest.update(len(path).to_bytes(4, "big"))
        tree_digest.update(path)
        tree_digest.update(mode.to_bytes(4, "big"))
        tree_digest.update(size.to_bytes(8, "big"))
        tree_digest.update(file_digest)
    return tree_digest.hexdigest()


def _decode_json_object(payload: bytes, *, path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise VersionStoreError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise VersionStoreError(f"{label} must contain a JSON object: {path}")
    return value


def _read_regular_file_descriptor(
    descriptor: int,
    *,
    path: Path,
    label: str,
    max_bytes: int | None = None,
) -> bytes:
    details = os.fstat(descriptor)
    if not stat.S_ISREG(details.st_mode):
        raise VersionStoreError(f"{label} must be a regular non-symlink file: {path}")
    if max_bytes is not None and details.st_size > max_bytes:
        raise VersionStoreError(f"{label} is too large: {path}")
    chunks: list[bytes] = []
    remaining = max_bytes
    while True:
        limit = 1024 * 1024 if remaining is None else min(1024 * 1024, remaining + 1)
        chunk = os.read(descriptor, limit)
        if not chunk:
            break
        chunks.append(chunk)
        if remaining is not None:
            remaining -= len(chunk)
            if remaining < 0:
                raise VersionStoreError(f"{label} is too large: {path}")
    return b"".join(chunks)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        descriptor = os.open(path, _FILE_OPEN_FLAGS)
    except OSError as exc:
        raise VersionStoreError(
            f"{label} must be a regular non-symlink file: {path}"
        ) from exc
    try:
        payload = _read_regular_file_descriptor(
            descriptor,
            path=path,
            label=label,
            max_bytes=_MAX_MANIFEST_BYTES,
        )
    finally:
        os.close(descriptor)
    return _decode_json_object(payload, path=path, label=label)


def _read_json_object_at(
    directory_fd: int,
    name: str,
    *,
    path: Path,
    label: str,
) -> dict[str, Any]:
    try:
        descriptor = os.open(name, _FILE_OPEN_FLAGS, dir_fd=directory_fd)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise VersionStoreError(
            f"{label} must be a regular non-symlink file: {path}"
        ) from exc
    try:
        payload = _read_regular_file_descriptor(
            descriptor,
            path=path,
            label=label,
            max_bytes=_MAX_MANIFEST_BYTES,
        )
    finally:
        os.close(descriptor)
    return _decode_json_object(payload, path=path, label=label)


def _valid_store_marker(path: Path) -> bool:
    try:
        marker = _read_json_object(path, label="version store marker")
    except VersionStoreError:
        return False
    return marker.get("schema_version") == STORE_SCHEMA_VERSION and marker.get(
        "kind"
    ) == "sforge-version-store" and isinstance(
        marker.get("store_id"), str
    ) and _SAFE_ID_RE.fullmatch(marker["store_id"]) is not None


def discover_sibling_version_store(repository_root: Path) -> Path | None:
    """Return a valid source-checkout sibling store, otherwise leave disabled."""

    parent = repository_root.resolve().parent
    for sibling_name in ("agent_task_version_submit", "agent-task-version-submit"):
        candidate = parent / sibling_name
        if not candidate.is_dir():
            continue
        if not _valid_store_marker(candidate / STORE_MARKER_NAME):
            continue
        return candidate
    return None


def _safe_id(value: str, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise VersionStoreError(f"invalid {field}: {value!r}")
    return value


def _normalized_identity_exclude(value: Any) -> tuple[str, ...]:
    """Validate and canonicalize a candidate-identity exclusion list."""

    if value is None:
        return ()
    if not isinstance(value, (list, tuple)) or not all(
        isinstance(item, str) and item.strip("/") for item in value
    ):
        raise VersionStoreError(
            "candidate identity exclusions must be non-empty path patterns"
        )
    return tuple(sorted({item.rstrip("/") for item in value}))


def _safe_label(value: str | None) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 80
        or value.strip() != value
        or _LABEL_CONTROL_RE.search(value) is not None
    ):
        raise VersionStoreCandidateError("label must be 1-80 printable characters")
    return value


def validate_version_label(value: str) -> str:
    """Validate and return a candidate-controlled checkpoint label."""

    validated = _safe_label(value)
    assert validated is not None
    return validated


def _normalized_created_at(value: datetime | None) -> tuple[datetime, str]:
    current = value or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise VersionStoreError("created_at must include a timezone")
    current = current.astimezone(timezone.utc)
    return current, current.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _json_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    metadata = dict(value or {})
    if any(not isinstance(key, str) for key in metadata):
        raise VersionStoreError("metadata keys must be strings")
    try:
        encoded = json.dumps(metadata, allow_nan=False, sort_keys=True)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise VersionStoreError("metadata must contain finite JSON values") from exc
    if not isinstance(decoded, dict):
        raise VersionStoreError("metadata must be a JSON object")
    return decoded


def _validate_single_gzip_stream(archive: bytes) -> None:
    """Require exactly one complete gzip member with no trailing bytes."""

    if not archive.startswith(b"\x1f\x8b"):
        raise VersionStoreCandidateError(
            "invalid submission archive: expected one complete gzip stream"
        )
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    pending = archive
    decompressed_bytes = 0
    decompressed_limit = (
        SUBMISSION_ARCHIVE_MAX_UNCOMPRESSED_BYTES
        + SUBMISSION_ARCHIVE_MAX_MEMBERS * 1024
        + 10240
    )
    try:
        while pending:
            output = decompressor.decompress(pending, 64 * 1024)
            decompressed_bytes += len(output)
            if decompressed_bytes > decompressed_limit:
                raise VersionStoreCandidateError(
                    "invalid submission archive: gzip payload is too large"
                )
            pending = decompressor.unconsumed_tail
    except zlib.error as exc:
        raise VersionStoreCandidateError(
            "invalid submission archive: malformed gzip stream"
        ) from exc
    if not decompressor.eof or decompressor.unused_data:
        raise VersionStoreCandidateError(
            "invalid submission archive: expected one complete gzip stream"
        )


def _write_fsynced_at(
    directory_fd: int,
    name: str,
    payload: bytes,
    *,
    mode: int = 0o644,
) -> None:
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
        dir_fd=directory_fd,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _open_child_directory(
    parent_fd: int,
    name: str,
    *,
    create: bool,
    display_path: Path,
) -> int | None:
    try:
        return os.open(name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError:
        if not create:
            return None
        try:
            os.mkdir(name, mode=0o755, dir_fd=parent_fd)
        except FileExistsError:
            pass
        else:
            os.fsync(parent_fd)
        try:
            return os.open(name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd)
        except OSError as exc:
            raise VersionStoreError(
                f"runtime path must be a real directory, not a symlink: {display_path}"
            ) from exc
    except OSError as exc:
        raise VersionStoreError(
            f"runtime path must be a real directory, not a symlink: {display_path}"
        ) from exc


@contextmanager
def _open_directory_chain(
    root: Path,
    components: Sequence[tuple[str, bool]],
):
    descriptors: list[int] = []
    current_path = root
    try:
        try:
            descriptor = os.open(root, _DIRECTORY_OPEN_FLAGS)
        except OSError as exc:
            raise VersionStoreError(
                f"version store root must remain a real directory: {root}"
            ) from exc
        descriptors.append(descriptor)
        for name, create in components:
            current_path = current_path / name
            child = _open_child_directory(
                descriptors[-1],
                name,
                create=create,
                display_path=current_path,
            )
            if child is None:
                yield None
                return
            descriptors.append(child)
        yield descriptors[-1]
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


@contextmanager
def _store_lock(root: Path):
    root_fd = -1
    lock_fd = -1
    try:
        root_fd = os.open(root, _DIRECTORY_OPEN_FLAGS)
        flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        try:
            lock_fd = os.open(STORE_LOCK_NAME, flags, dir_fd=root_fd)
        except FileNotFoundError:
            try:
                lock_fd = os.open(
                    STORE_LOCK_NAME,
                    flags | os.O_CREAT | os.O_EXCL,
                    0o644,
                    dir_fd=root_fd,
                )
                os.fsync(root_fd)
            except FileExistsError:
                lock_fd = os.open(STORE_LOCK_NAME, flags, dir_fd=root_fd)
        details = os.fstat(lock_fd)
        if not stat.S_ISREG(details.st_mode):
            raise VersionStoreError("version store lock must be a regular file")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
    except VersionStoreError:
        if lock_fd >= 0:
            os.close(lock_fd)
        if root_fd >= 0:
            os.close(root_fd)
        raise
    except OSError as exc:
        if lock_fd >= 0:
            os.close(lock_fd)
        if root_fd >= 0:
            os.close(root_fd)
        raise VersionStoreError(
            "version store lock must be a regular non-symlink file"
        ) from exc

    try:
        yield
    finally:
        if lock_fd >= 0:
            os.close(lock_fd)
        if root_fd >= 0:
            os.close(root_fd)


class VersionStore:
    """Validated immutable versions stored outside candidate containers."""

    def __init__(
        self,
        root: Path,
        store_id: str,
        registrations: Mapping[str, _TaskRegistration],
    ) -> None:
        self.root = root
        self.store_id = store_id
        self._registrations = dict(registrations)

    @classmethod
    def open(cls, root: Path) -> "VersionStore":
        requested = Path(root).expanduser()
        try:
            resolved = requested.resolve(strict=True)
        except OSError as exc:
            raise VersionStoreError(f"version store root does not exist: {requested}") from exc
        if not resolved.is_dir():
            raise VersionStoreError(f"version store root is not a directory: {resolved}")

        marker = _read_json_object(
            resolved / STORE_MARKER_NAME,
            label="version store marker",
        )
        if marker.get("schema_version") != STORE_SCHEMA_VERSION or marker.get(
            "kind"
        ) != "sforge-version-store":
            raise VersionStoreError(f"invalid version store marker: {resolved}")
        try:
            store_id = _safe_id(marker.get("store_id"), field="store_id")
        except VersionStoreError as exc:
            raise VersionStoreError(
                f"invalid version store marker: {resolved}"
            ) from exc

        registrations: dict[str, _TaskRegistration] = {}
        tasks_root = resolved / "tasks"
        if tasks_root.exists() and (tasks_root.is_symlink() or not tasks_root.is_dir()):
            raise VersionStoreError("version store tasks path must be a real directory")
        task_paths = sorted(tasks_root.iterdir()) if tasks_root.is_dir() else []
        for task_path in task_paths:
            if task_path.is_symlink():
                raise VersionStoreError(
                    f"task directory may not be a symlink or point outside the store: {task_path}"
                )
            if not task_path.is_dir():
                continue
            task_directory = _safe_id(task_path.name, field="task_directory")
            registration = _read_json_object(
                task_path / "task.json",
                label="task registration",
            )
            if registration.get("schema_version") != STORE_SCHEMA_VERSION:
                raise VersionStoreError(
                    f"unsupported task registration schema: {task_path}"
                )
            if registration.get("task_directory") != task_directory:
                raise VersionStoreError(
                    f"task_directory does not match its directory: {task_path}"
                )
            task_id = _safe_id(registration.get("task_id"), field="task_id")
            if task_id in registrations:
                raise VersionStoreError(f"duplicate task_id in version store: {task_id}")
            registrations[task_id] = _TaskRegistration(
                task_id=task_id,
                task_directory=task_directory,
                path=task_path,
            )
        return cls(resolved, store_id, registrations)

    def supports_task(self, task_id: str) -> bool:
        return task_id in self._registrations

    def _registration(self, task_id: str) -> _TaskRegistration:
        try:
            return self._registrations[task_id]
        except KeyError as exc:
            raise VersionStoreError(f"task is not registered in version store: {task_id}") from exc

    def _stored_from_manifest_value(
        self,
        manifest: Mapping[str, Any],
        manifest_path: Path,
        *,
        registration: _TaskRegistration | None = None,
        run_id: str | None = None,
    ) -> StoredVersion:
        required_strings = (
            "version_id",
            "task_id",
            "task_directory",
            "run_id",
            "source",
            "created_at",
            "archive_sha256",
        )
        if manifest.get("schema_version") != STORE_SCHEMA_VERSION or any(
            not isinstance(manifest.get(key), str) for key in required_strings
        ):
            raise VersionStoreError(f"invalid version manifest: {manifest_path}")
        round_id = manifest.get("round_id")
        label = manifest.get("label")
        metadata = manifest.get("metadata", {})
        content_sha256 = manifest.get("content_sha256")
        candidate_id = manifest.get("candidate_id")
        has_event_sequence = "event_sequence" in manifest
        has_candidate_sequence = "candidate_sequence" in manifest
        if has_event_sequence != has_candidate_sequence:
            raise VersionStoreError(
                f"incomplete version sequence fields: {manifest_path}"
            )
        event_sequence = manifest.get("event_sequence")
        candidate_sequence = manifest.get("candidate_sequence")
        if has_event_sequence:
            if (
                isinstance(event_sequence, bool)
                or not isinstance(event_sequence, int)
                or event_sequence <= 0
            ):
                raise VersionStoreError(
                    f"invalid version event sequence: {manifest_path}"
                )
            if (
                isinstance(candidate_sequence, bool)
                or not isinstance(candidate_sequence, int)
                or candidate_sequence <= 0
            ):
                raise VersionStoreError(
                    f"invalid version candidate sequence: {manifest_path}"
                )
        if round_id is not None and not isinstance(round_id, str):
            raise VersionStoreError(f"invalid version round_id: {manifest_path}")
        if label is not None and not isinstance(label, str):
            raise VersionStoreError(f"invalid version label: {manifest_path}")
        if not isinstance(metadata, dict):
            raise VersionStoreError(f"invalid version metadata: {manifest_path}")
        if content_sha256 is not None and (
            not isinstance(content_sha256, str)
            or _SHA256_RE.fullmatch(content_sha256) is None
        ):
            raise VersionStoreError(
                f"invalid version content SHA-256: {manifest_path}"
            )
        if (content_sha256 is None) != (candidate_id is None):
            raise VersionStoreError(
                f"incomplete candidate identity: {manifest_path}"
            )
        try:
            content_identity_exclude = _normalized_identity_exclude(
                manifest.get("content_identity_exclude")
            )
        except VersionStoreError as exc:
            raise VersionStoreError(
                f"invalid candidate identity exclusions: {manifest_path}"
            ) from exc
        if candidate_id is not None and (
            not isinstance(candidate_id, str)
            or candidate_id != _candidate_id(content_sha256 or "")
        ):
            raise VersionStoreError(f"invalid candidate ID: {manifest_path}")
        if has_event_sequence and candidate_id is None:
            raise VersionStoreError(
                f"version sequence fields require candidate identity: {manifest_path}"
            )
        archive_size = manifest.get("archive_size")
        member_count = manifest.get("member_count")
        if (
            isinstance(archive_size, bool)
            or not isinstance(archive_size, int)
            or archive_size < 0
            or isinstance(member_count, bool)
            or not isinstance(member_count, int)
            or member_count < 0
        ):
            raise VersionStoreError(f"invalid version archive measurements: {manifest_path}")
        if manifest_path.parent.name != manifest["version_id"]:
            raise VersionStoreError(f"version ID does not match directory: {manifest_path}")
        if _SAFE_VERSION_ID_RE.fullmatch(manifest["version_id"]) is None:
            raise VersionStoreError(f"invalid version ID: {manifest_path}")
        if manifest["source"] not in VERSION_SOURCES:
            raise VersionStoreError(f"invalid version source: {manifest_path}")
        if registration is not None and (
            manifest["task_id"] != registration.task_id
            or manifest["task_directory"] != registration.task_directory
        ):
            raise VersionStoreError(
                f"version manifest task does not match its directory: {manifest_path}"
            )
        if run_id is not None and manifest["run_id"] != run_id:
            raise VersionStoreError(
                f"version manifest run does not match its directory: {manifest_path}"
            )
        return StoredVersion(
            version_id=manifest["version_id"],
            task_id=manifest["task_id"],
            task_directory=manifest["task_directory"],
            run_id=manifest["run_id"],
            event_sequence=event_sequence,
            candidate_sequence=candidate_sequence,
            source=manifest["source"],
            round_id=round_id,
            label=label,
            created_at=manifest["created_at"],
            archive_sha256=manifest["archive_sha256"],
            content_sha256=content_sha256,
            candidate_id=candidate_id,
            archive_size=archive_size,
            member_count=member_count,
            path=manifest_path.parent,
            metadata=dict(metadata),
            content_identity_exclude=content_identity_exclude,
        )

    def _with_effective_sequences(
        self,
        versions: Sequence[StoredVersion],
    ) -> list[StoredVersion]:
        """Fill legacy sequence fields without changing any persisted assignment."""

        identified: list[StoredVersion] = []
        for version in versions:
            if version.content_sha256 is None:
                content_digest = _submission_content_sha256(
                    self._read_stored_archive(version),
                    identity_exclude=version.content_identity_exclude,
                )
                version = replace(
                    version,
                    content_sha256=content_digest,
                    candidate_id=_candidate_id(content_digest),
                )
            identified.append(version)

        event_owners: dict[int, str] = {}
        for version in identified:
            if version.event_sequence is None:
                continue
            previous = event_owners.get(version.event_sequence)
            if previous is not None and previous != version.version_id:
                raise VersionStoreError(
                    "version event sequence conflict: "
                    f"{previous} and {version.version_id} both use "
                    f"{version.event_sequence}"
                )
            event_owners[version.event_sequence] = version.version_id

        next_event_sequence = 1
        sequenced: list[StoredVersion] = []
        for version in sorted(
            identified,
            key=lambda item: (
                item.event_sequence is None,
                item.event_sequence or 0,
                item.created_at,
                item.version_id,
            ),
        ):
            if version.event_sequence is None:
                while next_event_sequence in event_owners:
                    next_event_sequence += 1
                version = replace(version, event_sequence=next_event_sequence)
                event_owners[next_event_sequence] = version.version_id
                next_event_sequence += 1
            sequenced.append(version)

        candidate_to_sequence: dict[str, int] = {}
        sequence_to_candidate: dict[int, str] = {}
        for version in sequenced:
            candidate_id = version.candidate_id
            assert candidate_id is not None
            if version.candidate_sequence is None:
                continue
            previous_sequence = candidate_to_sequence.get(candidate_id)
            if (
                previous_sequence is not None
                and previous_sequence != version.candidate_sequence
            ):
                raise VersionStoreError(
                    "version candidate sequence conflict: "
                    f"{candidate_id} uses both {previous_sequence} and "
                    f"{version.candidate_sequence}"
                )
            previous_candidate = sequence_to_candidate.get(
                version.candidate_sequence
            )
            if (
                previous_candidate is not None
                and previous_candidate != candidate_id
            ):
                raise VersionStoreError(
                    "version candidate sequence conflict: "
                    f"{previous_candidate} and {candidate_id} both use "
                    f"{version.candidate_sequence}"
                )
            candidate_to_sequence[candidate_id] = version.candidate_sequence
            sequence_to_candidate[version.candidate_sequence] = candidate_id

        first_event_by_candidate: dict[str, int] = {}
        for version in sequenced:
            candidate_id = version.candidate_id
            event_sequence = version.event_sequence
            assert candidate_id is not None and event_sequence is not None
            first_event_by_candidate[candidate_id] = min(
                event_sequence,
                first_event_by_candidate.get(candidate_id, event_sequence),
            )

        next_candidate_sequence = 1
        for candidate_id, _ in sorted(
            first_event_by_candidate.items(),
            key=lambda item: (item[1], item[0]),
        ):
            if candidate_id in candidate_to_sequence:
                continue
            while next_candidate_sequence in sequence_to_candidate:
                next_candidate_sequence += 1
            candidate_to_sequence[candidate_id] = next_candidate_sequence
            sequence_to_candidate[next_candidate_sequence] = candidate_id
            next_candidate_sequence += 1

        effective: list[StoredVersion] = []
        for version in sequenced:
            candidate_id = version.candidate_id
            assert candidate_id is not None
            effective.append(
                replace(
                    version,
                    candidate_sequence=candidate_to_sequence[candidate_id],
                )
            )
        return sorted(
            effective,
            key=lambda item: (item.event_sequence or 0, item.version_id),
        )

    def _versions_for_run(
        self,
        registration: _TaskRegistration,
        run_id: str,
    ) -> list[StoredVersion]:
        versions_root = registration.path / "runs" / run_id / "versions"
        components = (
            ("tasks", False),
            (registration.task_directory, False),
            ("runs", False),
            (run_id, False),
            ("versions", False),
        )
        with _open_directory_chain(self.root, components) as versions_fd:
            if versions_fd is None:
                return []
            stored_versions: list[StoredVersion] = []
            for version_id in sorted(os.listdir(versions_fd)):
                if version_id.startswith(".tmp-"):
                    continue
                version_path = versions_root / version_id
                try:
                    details = os.stat(
                        version_id,
                        dir_fd=versions_fd,
                        follow_symlinks=False,
                    )
                except OSError as exc:
                    raise VersionStoreError(
                        f"unable to inspect version directory: {version_path}"
                    ) from exc
                if stat.S_ISLNK(details.st_mode):
                    raise VersionStoreError(
                        f"runtime path may not be a symlink: {version_path}"
                    )
                if not stat.S_ISDIR(details.st_mode):
                    continue
                try:
                    version_fd = os.open(
                        version_id,
                        _DIRECTORY_OPEN_FLAGS,
                        dir_fd=versions_fd,
                    )
                except OSError as exc:
                    raise VersionStoreError(
                        f"runtime path must be a real directory: {version_path}"
                    ) from exc
                manifest_path = version_path / "manifest.json"
                try:
                    try:
                        manifest = _read_json_object_at(
                            version_fd,
                            "manifest.json",
                            path=manifest_path,
                            label="version manifest",
                        )
                    except FileNotFoundError:
                        continue
                finally:
                    os.close(version_fd)
                stored_versions.append(
                    self._stored_from_manifest_value(
                        manifest,
                        manifest_path,
                        registration=registration,
                        run_id=run_id,
                    )
                )
            return self._with_effective_sequences(stored_versions)

    def _run_ids(self, registration: _TaskRegistration) -> list[str]:
        runs_root = registration.path / "runs"
        components = (
            ("tasks", False),
            (registration.task_directory, False),
            ("runs", False),
        )
        with _open_directory_chain(self.root, components) as runs_fd:
            if runs_fd is None:
                return []
            run_ids: list[str] = []
            for name in sorted(os.listdir(runs_fd)):
                path = runs_root / name
                try:
                    details = os.stat(name, dir_fd=runs_fd, follow_symlinks=False)
                except OSError as exc:
                    raise VersionStoreError(
                        f"unable to inspect run directory: {path}"
                    ) from exc
                if stat.S_ISLNK(details.st_mode):
                    raise VersionStoreError(
                        f"runtime path may not be a symlink: {path}"
                    )
                if not stat.S_ISDIR(details.st_mode):
                    continue
                run_ids.append(_safe_id(name, field="run_id"))
            return run_ids

    def validate_archive(
        self,
        *,
        task_id: str,
        archive: bytes,
        submit_paths: Sequence[str],
        submit_exclude: Sequence[str],
    ) -> list[str]:
        """Validate archive safety and require every file to be submit-eligible."""

        self._registration(task_id)
        _validate_single_gzip_stream(archive)
        try:
            members = _validate_submission_archive(archive)
            allowed = _filter_archive_files(
                members,
                submit_paths=list(submit_paths),
                submit_exclude=list(submit_exclude),
            )
        except ValueError as exc:
            raise VersionStoreCandidateError(
                f"invalid submission archive: {exc}"
            ) from exc
        rejected = sorted(set(members) - set(allowed))
        if rejected:
            raise VersionStoreCandidateError(
                "submission archive contains files outside the submit contract: "
                + ", ".join(rejected[:10])
            )
        return members

    def record_archive(
        self,
        *,
        task_id: str,
        run_id: str,
        source: str,
        round_id: str | None,
        label: str | None,
        archive: bytes,
        submit_paths: Sequence[str],
        submit_exclude: Sequence[str],
        metadata: Mapping[str, Any] | None = None,
        created_at: datetime | None = None,
        max_versions_for_source: int | None = None,
        max_bytes_for_source: int | None = None,
        content_identity_exclude: Sequence[str] = (),
    ) -> StoredVersion:
        registration = self._registration(task_id)
        identity_exclude = _normalized_identity_exclude(content_identity_exclude)
        run_id = _safe_id(run_id, field="run_id")
        if source not in VERSION_SOURCES:
            raise VersionStoreError(f"invalid source: {source!r}")
        if round_id is not None:
            round_id = _safe_id(round_id, field="round_id")
        label = _safe_label(label)
        metadata_value = _json_metadata(metadata)
        created, created_iso = _normalized_created_at(created_at)

        members = self.validate_archive(
            task_id=task_id,
            archive=archive,
            submit_paths=submit_paths,
            submit_exclude=submit_exclude,
        )

        digest = hashlib.sha256(archive).hexdigest()
        content_digest = _submission_content_sha256(
            archive, identity_exclude=identity_exclude
        )
        candidate_id = _candidate_id(content_digest)
        event_name = round_id or source
        timestamp = created.strftime("%Y%m%dT%H%M%S.%fZ")
        version_id = f"{timestamp}__{event_name}__{digest[:12]}"
        manifest = {
            "archive_sha256": digest,
            "archive_size": len(archive),
            "candidate_id": candidate_id,
            "content_identity_exclude": list(identity_exclude),
            "content_sha256": content_digest,
            "created_at": created_iso,
            "label": label,
            "member_count": len(members),
            "metadata": metadata_value,
            "round_id": round_id,
            "run_id": run_id,
            "schema_version": STORE_SCHEMA_VERSION,
            "source": source,
            "task_directory": registration.task_directory,
            "task_id": task_id,
            "version_id": version_id,
        }

        try:
            with _store_lock(self.root):
                existing_versions = self._versions_for_run(registration, run_id)
                for existing in existing_versions:
                    if existing.source != source or existing.round_id != round_id:
                        continue
                    integrity_errors = self.verify([existing])
                    if integrity_errors:
                        raise VersionStoreWriteError(
                            "existing version failed integrity verification"
                        )
                    existing_content_digest = existing.content_sha256
                    if existing_content_digest is None:
                        existing_content_digest = _submission_content_sha256(
                            self._read_stored_archive(existing),
                            identity_exclude=existing.content_identity_exclude,
                        )
                    if existing_content_digest == content_digest:
                        if existing.content_sha256 is None:
                            return replace(
                                existing,
                                content_sha256=content_digest,
                                candidate_id=candidate_id,
                            )
                        return existing
                    raise VersionStoreCollision(
                        f"version event {event_name} already has different candidate content"
                    )

                if any(
                    existing.source == "final" for existing in existing_versions
                ):
                    raise VersionStoreCollision(
                        "version run is closed by its final event"
                    )

                manifest["event_sequence"] = (
                    max(
                        (
                            existing.event_sequence or 0
                            for existing in existing_versions
                        ),
                        default=0,
                    )
                    + 1
                )
                matching_candidate = next(
                    (
                        existing
                        for existing in existing_versions
                        if existing.content_sha256 == content_digest
                    ),
                    None,
                )
                if matching_candidate is not None:
                    assert matching_candidate.candidate_sequence is not None
                    manifest["candidate_sequence"] = (
                        matching_candidate.candidate_sequence
                    )
                else:
                    manifest["candidate_sequence"] = (
                        max(
                            (
                                existing.candidate_sequence or 0
                                for existing in existing_versions
                            ),
                            default=0,
                        )
                        + 1
                    )

                source_versions = [
                    existing
                    for existing in existing_versions
                    if existing.source == source
                ]
                if (
                    max_versions_for_source is not None
                    and len(source_versions) >= max_versions_for_source
                ):
                    raise VersionStoreQuotaExceeded(
                        f"{source} storage quota exceeded"
                    )
                if (
                    max_bytes_for_source is not None
                    and sum(item.archive_size for item in source_versions) + len(archive)
                    > max_bytes_for_source
                ):
                    raise VersionStoreQuotaExceeded(
                        f"{source} storage quota exceeded"
                    )

                versions_root = registration.path / "runs" / run_id / "versions"
                components = (
                    ("tasks", False),
                    (registration.task_directory, False),
                    ("runs", True),
                    (run_id, True),
                    ("versions", True),
                )
                temporary_name = f".tmp-{uuid.uuid4().hex}"
                temporary = versions_root / temporary_name
                destination = versions_root / version_id
                with _open_directory_chain(self.root, components) as versions_fd:
                    assert versions_fd is not None
                    os.mkdir(temporary_name, mode=0o755, dir_fd=versions_fd)
                    try:
                        temporary_fd = os.open(
                            temporary_name,
                            _DIRECTORY_OPEN_FLAGS,
                            dir_fd=versions_fd,
                        )
                        try:
                            _write_fsynced_at(
                                temporary_fd,
                                "submission.tar.gz",
                                archive,
                            )
                            manifest_bytes = (
                                json.dumps(
                                    manifest,
                                    indent=2,
                                    sort_keys=True,
                                    ensure_ascii=False,
                                    allow_nan=False,
                                )
                                + "\n"
                            ).encode("utf-8")
                            _write_fsynced_at(
                                temporary_fd,
                                "manifest.json",
                                manifest_bytes,
                            )
                            os.fsync(temporary_fd)
                        finally:
                            os.close(temporary_fd)
                        os.replace(
                            temporary_name,
                            version_id,
                            src_dir_fd=versions_fd,
                            dst_dir_fd=versions_fd,
                        )
                        os.fsync(versions_fd)
                    except Exception:
                        shutil.rmtree(temporary, ignore_errors=True)
                        raise
        except VersionStoreError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise VersionStoreWriteError(
                f"failed to write version {event_name}: {exc}"
            ) from exc

        return self._stored_from_manifest_value(
            manifest,
            destination / "manifest.json",
            registration=registration,
            run_id=run_id,
        )

    def list_versions(
        self,
        *,
        task: str | None = None,
        run_id: str | None = None,
        source: str | None = None,
    ) -> list[StoredVersion]:
        if source is not None and source not in VERSION_SOURCES:
            raise VersionStoreError(f"invalid source: {source!r}")
        versions: list[StoredVersion] = []
        for registration in self._registrations.values():
            if task is not None and task not in {
                registration.task_id,
                registration.task_directory,
            }:
                continue
            for stored_run_id in self._run_ids(registration):
                if run_id is not None and stored_run_id != run_id:
                    continue
                for stored in self._versions_for_run(registration, stored_run_id):
                    if source is None or stored.source == source:
                        versions.append(stored)
        return sorted(
            versions,
            key=lambda item: (
                item.task_id,
                item.run_id,
                item.event_sequence or 0,
                item.version_id,
            ),
        )

    def list_candidates(
        self,
        *,
        task: str | None = None,
        run_id: str | None = None,
        source: str | None = None,
    ) -> list[StoredCandidate]:
        """Group immutable archive events into logical submitted-code states."""

        if source is not None and source not in VERSION_SOURCES:
            raise VersionStoreError(f"invalid source: {source!r}")
        versions = self.list_versions(task=task, run_id=run_id)
        grouped: dict[tuple[str, str, str], list[StoredVersion]] = {}
        for version in versions:
            content_digest = version.content_sha256
            if content_digest is None:
                content_digest = _submission_content_sha256(
                    self._read_stored_archive(version),
                    identity_exclude=version.content_identity_exclude,
                )
                version = replace(
                    version,
                    content_sha256=content_digest,
                    candidate_id=_candidate_id(content_digest),
                )
            key = (version.task_id, version.run_id, content_digest)
            grouped.setdefault(key, []).append(version)

        ordered_groups = sorted(
            grouped.values(),
            key=lambda events: (
                events[0].task_id,
                events[0].run_id,
                events[0].candidate_sequence or 0,
            ),
        )
        candidates: list[StoredCandidate] = []
        for events in ordered_groups:
            events.sort(
                key=lambda event: (event.event_sequence or 0, event.version_id)
            )
            first = events[0]
            sequence = first.candidate_sequence
            if sequence is None or any(
                event.candidate_sequence != sequence for event in events
            ):
                raise VersionStoreError(
                    f"candidate has inconsistent sequence: {first.candidate_id}"
                )
            if source is not None and not any(
                event.source == source for event in events
            ):
                continue
            content_digest = first.content_sha256
            assert content_digest is not None
            candidates.append(
                StoredCandidate(
                    candidate_id=_candidate_id(content_digest),
                    content_sha256=content_digest,
                    sequence=sequence,
                    task_id=first.task_id,
                    task_directory=first.task_directory,
                    run_id=first.run_id,
                    created_at=first.created_at,
                    label=next(
                        (event.label for event in events if event.label is not None),
                        None,
                    ),
                    events=tuple(events),
                )
            )
        return candidates

    def list_timeline(
        self,
        *,
        task: str | None = None,
        run_id: str | None = None,
        source: str | None = None,
    ) -> list[StoredTimelineEvent]:
        """Annotate physical events with first-seen and revisit relationships.

        Relations are computed from the complete task/run stream before an
        optional source filter is applied.  This preserves the relationship
        between, for example, a checkpoint and a later Judge event even when
        only Judge-originated events are displayed.
        """

        if source is not None and source not in VERSION_SOURCES:
            raise VersionStoreError(f"invalid source: {source!r}")

        versions = self.list_versions(task=task, run_id=run_id)
        candidates = self.list_candidates(task=task, run_id=run_id)
        candidate_by_event = {
            (event.task_id, event.run_id, event.version_id): candidate
            for candidate in candidates
            for event in candidate.events
        }

        previous_by_stream: dict[tuple[str, str], str] = {}
        first_seen: dict[tuple[str, str, str], int] = {}
        timeline: list[StoredTimelineEvent] = []
        for version in versions:
            event_key = (version.task_id, version.run_id, version.version_id)
            candidate = candidate_by_event.get(event_key)
            if candidate is None:
                raise VersionStoreError(
                    f"stored event has no logical candidate: {version.version_id}"
                )

            stream_key = (version.task_id, version.run_id)
            event_sequence = version.event_sequence
            if event_sequence is None:
                raise VersionStoreError(
                    f"stored event has no event sequence: {version.version_id}"
                )
            candidate_key = (*stream_key, candidate.candidate_id)
            previous_candidate_id = previous_by_stream.get(stream_key)
            first_seen_event_sequence = first_seen.get(candidate_key)
            if first_seen_event_sequence is None:
                relation = "first_seen"
                first_seen_event_sequence = event_sequence
                first_seen[candidate_key] = event_sequence
            elif previous_candidate_id == candidate.candidate_id:
                relation = "same_as_previous"
            else:
                relation = "revisit"
            previous_by_stream[stream_key] = candidate.candidate_id

            if source is None or version.source == source:
                timeline.append(
                    StoredTimelineEvent(
                        event_sequence=event_sequence,
                        version=version,
                        candidate_sequence=candidate.sequence,
                        candidate_id=candidate.candidate_id,
                        previous_candidate_id=previous_candidate_id,
                        relation=relation,
                        first_seen_event_sequence=first_seen_event_sequence,
                    )
                )
        return timeline

    def _read_stored_archive(self, stored: StoredVersion) -> bytes:
        registration = self._registration(stored.task_id)
        if stored.task_directory != registration.task_directory:
            raise VersionStoreError("stored version task directory mismatch")
        run_id = _safe_id(stored.run_id, field="run_id")
        if _SAFE_VERSION_ID_RE.fullmatch(stored.version_id) is None:
            raise VersionStoreError("stored version ID is unsafe")
        components = (
            ("tasks", False),
            (registration.task_directory, False),
            ("runs", False),
            (run_id, False),
            ("versions", False),
            (stored.version_id, False),
        )
        with _open_directory_chain(self.root, components) as version_fd:
            if version_fd is None:
                raise VersionStoreError("stored version directory is missing")
            try:
                archive_fd = os.open(
                    "submission.tar.gz",
                    _FILE_OPEN_FLAGS,
                    dir_fd=version_fd,
                )
            except OSError as exc:
                raise VersionStoreError(
                    "version archive must be a regular non-symlink file"
                ) from exc
            try:
                return _read_regular_file_descriptor(
                    archive_fd,
                    path=stored.archive_path,
                    label="version archive",
                    max_bytes=SUBMISSION_ARCHIVE_MAX_BYTES,
                )
            finally:
                os.close(archive_fd)

    def read_archive(self, stored: StoredVersion) -> bytes:
        """Read and verify one immutable archive through no-follow store paths."""

        archive = self._read_stored_archive(stored)
        if len(archive) != stored.archive_size:
            raise VersionStoreError("stored version archive size mismatch")
        digest = hashlib.sha256(archive).hexdigest()
        if digest != stored.archive_sha256:
            raise VersionStoreError("stored version archive SHA-256 mismatch")
        if stored.content_sha256 is not None:
            content_digest = _submission_content_sha256(
                archive, identity_exclude=stored.content_identity_exclude
            )
            if content_digest != stored.content_sha256:
                raise VersionStoreError(
                    "stored version normalized content SHA-256 mismatch"
                )
        return archive

    def verify(self, versions: Iterable[StoredVersion] | None = None) -> list[str]:
        errors: list[str] = []
        selected = list(versions) if versions is not None else self.list_versions()
        for stored in selected:
            try:
                archive = self._read_stored_archive(stored)
            except (OSError, VersionStoreError) as exc:
                errors.append(
                    f"{stored.version_id}: archive must be a readable regular "
                    f"non-symlink file: {exc}"
                )
                continue
            if len(archive) != stored.archive_size:
                errors.append(
                    f"{stored.version_id}: size mismatch "
                    f"({len(archive)} != {stored.archive_size})"
                )
                continue
            digest = hashlib.sha256(archive).hexdigest()
            if digest != stored.archive_sha256:
                errors.append(
                    f"{stored.version_id}: SHA-256 mismatch "
                    f"({digest} != {stored.archive_sha256})"
                )
                continue
            if stored.content_sha256 is not None:
                try:
                    content_digest = _submission_content_sha256(
                        archive,
                        identity_exclude=stored.content_identity_exclude,
                    )
                except VersionStoreCandidateError as exc:
                    errors.append(
                        f"{stored.version_id}: normalized content is invalid: {exc}"
                    )
                    continue
                if content_digest != stored.content_sha256:
                    errors.append(
                        f"{stored.version_id}: content SHA-256 mismatch "
                        f"({content_digest} != {stored.content_sha256})"
                    )
        return errors
