"""What the integrity reviewer sees: the submitted files against their baseline.

The submission side is the exact file set the GPU Judge would receive: the
archive is validated and filtered with the same functions the Judge uses
(``submit_paths`` whitelist, ``submit_exclude`` blacklist), so a file the
Judge would drop can never make the reviewer reject, and a file the Judge
would run is never hidden from it.

The baseline side is the agent's starting state of the submitted paths, read
from the task's Work image.  The Work image, not the Judge image, is the
source on purpose: everything in it was already visible to the agent, so
nothing the reviewer reads (and may quote back in its feedback) can leak
hidden cases, oracles or other Judge-only material.  The image is only
created as a stopped container and read with ``get_archive``; nothing in it
is ever executed, and it never touches a GPU.
"""

from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import re
import tarfile
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable, Protocol

from sforge.harness.llm_judge.policy import TaskIntegrityPolicy
from sforge.harness.run_evaluation import (
    _canonical_submission_member_name,
    _filter_archive_files,
    _validate_submission_archive,
)
from sforge.harness.task_spec import TaskSpec

# Files above this size are hashed but never decoded or shown.
TEXT_DECODE_MAX_BYTES = 4 * 1024 * 1024
DOC_MAX_BYTES = 256 * 1024
# Upper bound on one get_archive stream from the Work image.
BASELINE_PATH_MAX_BYTES = 1024 * 1024 * 1024
_HASH_CHUNK = 1024 * 1024

ROLE_SELECTED = "selected"
ROLE_SHARED = "shared"
ROLE_INACTIVE = "inactive"
ROLE_NOTES = "notes"

STATUS_ADDED = "added"
STATUS_MODIFIED = "modified"
STATUS_UNCHANGED = "unchanged"
# Present at the start but not submitted. The Judge extracts the archive over
# its image and never deletes files, so this is not a removal.
STATUS_DELETED = "absent"


class BundleError(ValueError):
    """The submission cannot be turned into a review bundle."""


@dataclass(frozen=True)
class FileContent:
    size: int
    sha256: str
    text: str | None
    binary: bool

    @property
    def shown_as_text(self) -> bool:
        return self.text is not None


def _content_from_stream(stream: BinaryIO, declared_size: int) -> FileContent:
    digest = hashlib.sha256()
    kept = bytearray()
    observed = 0
    while True:
        chunk = stream.read(_HASH_CHUNK)
        if not chunk:
            break
        observed += len(chunk)
        digest.update(chunk)
        if observed <= TEXT_DECODE_MAX_BYTES:
            kept.extend(chunk)
    if observed != declared_size:
        raise BundleError("archive member size does not match its contents")
    text: str | None = None
    binary = False
    if observed <= TEXT_DECODE_MAX_BYTES:
        raw = bytes(kept)
        if b"\x00" in raw:
            binary = True
        else:
            # Python accepts other source encodings (``# coding: latin-1``);
            # decoding them as Latin-1 keeps such files visible and their
            # evidence verifiable instead of letting them pass as binary.
            text = _decode_text(raw)
    return FileContent(
        size=observed,
        sha256=digest.hexdigest(),
        text=text,
        binary=binary,
    )


def _decode_text(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def content_from_bytes(payload: bytes) -> FileContent:
    return _content_from_stream(io.BytesIO(payload), len(payload))


@dataclass(frozen=True)
class ReviewFile:
    path: str
    status: str
    role: str
    submission: FileContent | None
    baseline: FileContent | None
    version: int | None = None
    review_excluded: bool = False

    @property
    def size(self) -> int:
        entry = self.submission or self.baseline
        return entry.size if entry is not None else 0


@dataclass(frozen=True)
class Selection:
    """Which retained implementation the Judge evaluates."""

    selector_file: str
    root: str
    per_directory: bool
    value: int | None
    selected_prefix: str | None
    from_override: bool

    def describe(self) -> str:
        if self.value is None:
            return (
                f"`{self.selector_file}` could not be read as a version number, "
                f"so treat every implementation under `{self.root}/` as "
                "possibly evaluated."
            )
        source = (
            "the replay override" if self.from_override else f"`{self.selector_file}`"
        )
        return (
            f"Only the implementation selected by {source} (version "
            f"{self.value}: `{self.selected_prefix}`) is evaluated; other "
            f"versions under `{self.root}/` are inactive history."
        )


@dataclass
class BaselineSnapshot:
    files: dict[str, FileContent]
    documents: dict[str, str]
    source: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class ReviewBundle:
    task_id: str
    cwd: str
    submit_paths: list[str]
    files: list[ReviewFile]
    dropped_paths: list[str]
    selection: Selection | None
    baseline_source: str
    instruction_documents: list[tuple[str, str]]
    context_documents: list[tuple[str, str]]
    warnings: list[str]
    archive: bytes | None = field(default=None, repr=False)

    def submitted_text(self, path: str) -> str | None:
        """Full submitted text of one file, for evidence verification.

        Uses the decoded text when the file was small enough, and otherwise
        decodes the archive member itself, so neither size nor encoding can
        make cited evidence unverifiable.  Files with NUL bytes stay binary.
        """

        item = self.file(path)
        if item is None or item.submission is None:
            return None
        if item.submission.text is not None:
            return item.submission.text
        if item.submission.binary or self.archive is None:
            return None
        with tarfile.open(fileobj=io.BytesIO(self.archive), mode="r:*") as bundle:
            for member in bundle:
                if not member.isfile():
                    continue
                try:
                    name = _canonical_submission_member_name(member.name)
                except ValueError:
                    continue
                if name != path:
                    continue
                stream = bundle.extractfile(member)
                if stream is None:
                    return None
                raw = stream.read()
                return None if b"\x00" in raw else _decode_text(raw)
        return None

    def file(self, path: str) -> ReviewFile | None:
        for item in self.files:
            if item.path == path:
                return item
        return None

    @property
    def changed_files(self) -> list[ReviewFile]:
        return [
            item
            for item in self.files
            if item.status in {STATUS_ADDED, STATUS_MODIFIED, STATUS_DELETED}
        ]

    def digest(self) -> str:
        """Content identity of what is reviewed (independent of tar metadata)."""

        payload = {
            "task_id": self.task_id,
            "files": [
                [
                    item.path,
                    item.status,
                    item.role,
                    item.submission.sha256 if item.submission else None,
                    item.baseline.sha256 if item.baseline else None,
                    item.review_excluded,
                ]
                for item in self.files
            ],
            "dropped": self.dropped_paths,
            "selection": (
                [
                    self.selection.selector_file,
                    self.selection.value,
                    self.selection.selected_prefix,
                    self.selection.from_override,
                ]
                if self.selection
                else None
            ),
            "baseline": self.baseline_source,
            "documents": [
                [path, hashlib.sha256(text.encode("utf-8")).hexdigest()]
                for path, text in (
                    self.instruction_documents + self.context_documents
                )
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def manifest(self) -> dict[str, Any]:
        """Host-side audit record of the bundle (no file contents)."""

        return {
            "task_id": self.task_id,
            "cwd": self.cwd,
            "submit_paths": self.submit_paths,
            "baseline_source": self.baseline_source,
            "selection": (
                {
                    "selector_file": self.selection.selector_file,
                    "value": self.selection.value,
                    "selected_prefix": self.selection.selected_prefix,
                    "from_override": self.selection.from_override,
                }
                if self.selection
                else None
            ),
            "files": [
                {
                    "path": item.path,
                    "status": item.status,
                    "role": item.role,
                    "size": item.size,
                    "sha256": item.submission.sha256 if item.submission else None,
                    "baseline_sha256": item.baseline.sha256 if item.baseline else None,
                    "review_excluded": item.review_excluded,
                }
                for item in self.files
            ],
            "dropped_paths": self.dropped_paths,
            "instruction_documents": [path for path, _ in self.instruction_documents],
            "context_documents": [path for path, _ in self.context_documents],
            "warnings": self.warnings,
            "digest": self.digest(),
        }


class BaselineProvider(Protocol):
    def snapshot(
        self, task_spec: TaskSpec, policy: TaskIntegrityPolicy
    ) -> BaselineSnapshot: ...


def _normalized_submit_path(raw: str) -> str:
    value = raw.strip()
    while value.startswith("./"):
        value = value[2:]
    value = value.rstrip("/")
    return value or "."


def default_document_paths(
    task_spec: TaskSpec, policy: TaskIntegrityPolicy
) -> tuple[list[str], list[str]]:
    """Agent-visible briefs (instructions) and extra context files to read."""

    roots: list[str] = []
    for root in (task_spec.work.specs_dir, task_spec.cwd):
        if root and root.rstrip("/") not in roots:
            roots.append(root.rstrip("/"))
    instructions: list[str] = []
    for path in [
        f"{root}/{name}"
        for root in roots
        for name in ("AGENT_TASK.md", "README_AGENT.md")
    ] + list(policy.instruction_files):
        if path not in instructions:
            instructions.append(path)
    context = [path for path in policy.context_files if path not in instructions]
    return instructions, context


def _is_review_excluded(path: str, patterns: Iterable[str]) -> bool:
    for pattern in patterns:
        if fnmatch.fnmatch(path, pattern):
            return True
        if any(fnmatch.fnmatch(part, pattern) for part in path.split("/")):
            return True
    return False


class HostDirectoryBaseline:
    """Baseline from a host directory that mirrors the Work cwd.

    ``documents_root`` maps absolute container document paths onto the host
    (``/app/x/AGENT_TASK.md`` -> ``<documents_root>/app/x/AGENT_TASK.md``)
    when given; otherwise a document is looked up by file name in
    ``directory``.  Used by the offline CLI and tests, where no Work image
    is at hand.
    """

    def __init__(self, directory: Path, documents_root: Path | None = None) -> None:
        self.directory = directory
        self.documents_root = documents_root

    def snapshot(
        self, task_spec: TaskSpec, policy: TaskIntegrityPolicy
    ) -> BaselineSnapshot:
        files: dict[str, FileContent] = {}
        warnings: list[str] = []
        if self.directory.is_dir():
            names: list[str] = []
            for path in sorted(self.directory.rglob("*")):
                if path.is_symlink() or not path.is_file():
                    continue
                names.append(path.relative_to(self.directory).as_posix())
            for name in _filter_archive_files(
                names, task_spec.submit_paths, task_spec.submit_exclude
            ):
                files[name] = content_from_bytes((self.directory / name).read_bytes())
        else:
            warnings.append(f"baseline directory {self.directory} does not exist")
        documents: dict[str, str] = {}
        instructions, context = default_document_paths(task_spec, policy)
        for container_path in instructions + context:
            if self.documents_root is not None:
                host_path = self.documents_root / container_path.lstrip("/")
            else:
                host_path = self.directory / PurePosixPath(container_path).name
            if host_path.is_file():
                payload = host_path.read_bytes()[:DOC_MAX_BYTES]
                documents[container_path] = payload.decode("utf-8", "replace")
        return BaselineSnapshot(
            files=files,
            documents=documents,
            source=f"host-directory:{self.directory}",
            warnings=warnings,
        )


class ArchiveBaseline:
    """Baseline from an archive of the agent's starting files.

    For backends without local Work images (PyroMind): the Worker's initial
    checkpoint, saved before its agent started, is exactly the starting
    state of the submitted paths, and the briefs are read from the Worker
    at the same moment (``capture_documents``).
    """

    def __init__(self, archive: bytes, documents: dict[str, str], *, source: str) -> None:
        self.archive = archive
        self.documents = dict(documents)
        self.source = source

    @staticmethod
    def document_paths(task_spec: TaskSpec, policy: TaskIntegrityPolicy) -> list[str]:
        instructions, context = default_document_paths(task_spec, policy)
        return instructions + context

    @classmethod
    def capture(
        cls,
        task_spec: TaskSpec,
        policy: TaskIntegrityPolicy,
        archive: bytes,
        read_document,
        *,
        source: str,
    ) -> "ArchiveBaseline":
        """Read every brief the review shows with ``read_document(path)``,
        which returns the file's bytes or None when it does not exist."""

        documents: dict[str, str] = {}
        for path in cls.document_paths(task_spec, policy):
            payload = read_document(path)
            if payload is not None:
                documents[path] = payload[:DOC_MAX_BYTES].decode("utf-8", "replace")
        return cls(archive, documents, source=source)

    def snapshot(
        self, task_spec: TaskSpec, policy: TaskIntegrityPolicy
    ) -> BaselineSnapshot:
        warnings: list[str] = []
        try:
            files, _ = read_submission(self.archive, task_spec)
        except BundleError as exc:
            files = {}
            warnings.append(f"baseline archive unusable: {exc}")
        for path in list(policy.instruction_files) + list(policy.context_files):
            if path not in self.documents:
                warnings.append(f"policy document {path} was not captured")
        return BaselineSnapshot(
            files=files,
            documents=dict(self.documents),
            source=self.source,
            warnings=warnings,
        )


class NoBaseline:
    """Used when the Work image cannot be read (e.g. non-Docker backends)."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def snapshot(
        self, task_spec: TaskSpec, policy: TaskIntegrityPolicy
    ) -> BaselineSnapshot:
        return BaselineSnapshot(
            files={},
            documents={},
            source=f"unavailable:{self.reason}",
            warnings=[f"baseline unavailable: {self.reason}"],
        )


class DockerWorkImageBaseline:
    """Read the agent's starting files out of the task's Work image.

    A stopped container is created from the image (no network, never
    started) and the submitted paths plus the agent-visible briefs are
    copied out with ``get_archive``.  Snapshots are cached per image ID, so
    each Judge server reads a given image once.
    """

    def __init__(self, docker_client) -> None:
        self.client = docker_client
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str, str], BaselineSnapshot] = {}

    def snapshot(
        self, task_spec: TaskSpec, policy: TaskIntegrityPolicy
    ) -> BaselineSnapshot:
        import docker

        image = self.client.images.get(task_spec.work_image_key)
        key = (task_spec.task_id, image.id, policy.sha256)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            container = self.client.containers.create(
                image.id,
                # Never started; an explicit command only satisfies images
                # that declare neither CMD nor ENTRYPOINT.
                command=["/bin/true"],
                name=f"sforge-llm-judge-baseline-{uuid.uuid4().hex[:12]}",
                network_disabled=True,
                labels={"sforge.role": "llm-judge-baseline"},
            )
            try:
                snapshot = self._read(container, task_spec, policy, image.id, docker)
            finally:
                try:
                    container.remove(force=True)
                except Exception:
                    pass
            self._cache[key] = snapshot
            return snapshot

    @staticmethod
    def _spool(container, source: str, docker_module) -> BinaryIO | None:
        try:
            bits, _ = container.get_archive(source)
        except docker_module.errors.NotFound:
            return None
        spool = tempfile.SpooledTemporaryFile(max_size=64 * 1024 * 1024)
        total = 0
        for chunk in bits:
            total += len(chunk)
            if total > BASELINE_PATH_MAX_BYTES:
                spool.close()
                raise BundleError(f"Work image path {source} is too large to review")
            spool.write(chunk)
        spool.seek(0)
        return spool

    def _read(
        self,
        container,
        task_spec: TaskSpec,
        policy: TaskIntegrityPolicy,
        image_id: str,
        docker_module,
    ) -> BaselineSnapshot:
        cwd = PurePosixPath(task_spec.cwd)
        files: dict[str, FileContent] = {}
        warnings: list[str] = []
        names_seen: list[str] = []
        contents: dict[str, FileContent] = {}
        for raw in task_spec.submit_paths:
            submit_path = _normalized_submit_path(raw)
            source = str(cwd if submit_path == "." else cwd / submit_path)
            spool = self._spool(container, source, docker_module)
            if spool is None:
                continue
            with spool, tarfile.open(fileobj=spool, mode="r|*") as stream:
                for member in stream:
                    if not member.isfile():
                        continue
                    parts = member.name.split("/")
                    rest = "/".join(parts[1:])
                    if submit_path == ".":
                        name = rest
                    else:
                        name = f"{submit_path}/{rest}" if rest else submit_path
                    try:
                        name = _canonical_submission_member_name(name)
                    except ValueError:
                        warnings.append(f"skipped non-canonical baseline path {name!r}")
                        continue
                    handle = stream.extractfile(member)
                    if handle is None:
                        continue
                    contents[name] = _content_from_stream(handle, member.size)
                    names_seen.append(name)
        for name in _filter_archive_files(
            names_seen, task_spec.submit_paths, task_spec.submit_exclude
        ):
            files[name] = contents[name]
        documents: dict[str, str] = {}
        instructions, context = default_document_paths(task_spec, policy)
        for path in instructions + context:
            text = self._read_document(container, path, docker_module)
            if text is not None:
                documents[path] = text
            elif path in policy.instruction_files or path in policy.context_files:
                warnings.append(f"policy document {path} is missing from the Work image")
        return BaselineSnapshot(
            files=files,
            documents=documents,
            source=f"work-image:{image_id}",
            warnings=warnings,
        )

    def _read_document(self, container, path: str, docker_module) -> str | None:
        spool = self._spool(container, path, docker_module)
        if spool is None:
            return None
        with spool, tarfile.open(fileobj=spool, mode="r|*") as stream:
            for member in stream:
                if member.isfile():
                    handle = stream.extractfile(member)
                    if handle is None:
                        return None
                    payload = handle.read(DOC_MAX_BYTES)
                    return payload.decode("utf-8", "replace")
        return None


def read_submission(
    archive: bytes, task_spec: TaskSpec
) -> tuple[dict[str, FileContent], list[str]]:
    """The files the Judge would extract, plus the archive paths it drops."""

    try:
        names = _validate_submission_archive(archive)
    except ValueError as exc:
        raise BundleError(f"submission archive is invalid: {exc}") from exc
    allowed = set(
        _filter_archive_files(names, task_spec.submit_paths, task_spec.submit_exclude)
    )
    dropped = sorted(set(names) - allowed)
    entries: dict[str, FileContent] = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as bundle:
        for member in bundle:
            if not member.isfile():
                continue
            name = _canonical_submission_member_name(member.name)
            if name not in allowed:
                continue
            stream = bundle.extractfile(member)
            if stream is None:
                raise BundleError(f"submission member {name} is unreadable")
            entries[name] = _content_from_stream(stream, member.size)
    return entries, dropped


def iter_submission_files(archive: bytes, task_spec: TaskSpec):
    """Yield ``(path, bytes)`` for every file the Judge would extract."""

    names = _validate_submission_archive(archive)
    allowed = set(
        _filter_archive_files(names, task_spec.submit_paths, task_spec.submit_exclude)
    )
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as bundle:
        for member in bundle:
            if not member.isfile():
                continue
            name = _canonical_submission_member_name(member.name)
            if name not in allowed:
                continue
            stream = bundle.extractfile(member)
            if stream is not None:
                yield name, stream.read()


def _selection(
    task_spec: TaskSpec,
    submission: dict[str, FileContent],
    candidate_version: int | None,
    policy: TaskIntegrityPolicy | None = None,
) -> tuple[Selection | None, re.Pattern[str] | None]:
    enumeration = task_spec.version_enumeration or (
        policy.version_enumeration if policy is not None else None
    )
    if not isinstance(enumeration, dict):
        return None, None
    root = str(enumeration.get("root") or "").strip("/")
    selector_file = str(enumeration.get("selector_file") or "")
    directory_text = str(enumeration.get("directory_pattern") or "")
    file_text = str(enumeration.get("file_pattern") or "")
    if not root or not selector_file or bool(directory_text) == bool(file_text):
        return None, None
    try:
        pattern = re.compile(directory_text or file_text)
    except re.error:
        return None, None
    if pattern.groups != 1:
        return None, None
    per_directory = bool(directory_text)
    value: int | None = None
    from_override = candidate_version is not None
    if candidate_version is not None:
        value = int(candidate_version)
    else:
        selector = submission.get(selector_file)
        raw = selector.text.strip() if selector and selector.text else ""
        # Judges differ on canonical spelling (some accept "002"); read the
        # number the way the most lenient one does.
        if re.fullmatch(r"[0-9]{1,3}", raw) and 1 <= int(raw) <= 999:
            value = int(raw)
    selected_prefix: str | None = None
    if value is not None:
        for path in submission:
            parts = path.split("/")
            if parts[0] != root or len(parts) < 2:
                continue
            match = pattern.fullmatch(parts[1])
            if match is None:
                continue
            try:
                number = int(match.group(1))
            except ValueError:
                continue
            if number == value and (len(parts) >= 3 if per_directory else len(parts) == 2):
                selected_prefix = (
                    f"{root}/{parts[1]}/" if per_directory else f"{root}/{parts[1]}"
                )
                break
    return (
        Selection(
            selector_file=selector_file,
            root=root,
            per_directory=per_directory,
            value=value,
            selected_prefix=selected_prefix,
            from_override=from_override,
        ),
        pattern,
    )


def _role(
    path: str,
    task_spec: TaskSpec,
    selection: Selection | None,
    pattern: re.Pattern[str] | None,
) -> tuple[str, int | None]:
    for excluded in task_spec.candidate_identity_exclude:
        excluded = excluded.rstrip("/")
        if path == excluded or path.startswith(excluded + "/"):
            return ROLE_NOTES, None
    if selection is None or pattern is None:
        return ROLE_SHARED, None
    parts = path.split("/")
    if parts[0] != selection.root or len(parts) < 2:
        return ROLE_SHARED, None
    match = pattern.fullmatch(parts[1])
    shaped = len(parts) >= 3 if selection.per_directory else len(parts) == 2
    if match is None or not shaped:
        return ROLE_SHARED, None
    try:
        number = int(match.group(1))
    except ValueError:
        return ROLE_SHARED, None
    if selection.value is None:
        # Unknown selection: no implementation may be dismissed as inactive.
        return ROLE_SHARED, number
    if number == selection.value:
        return ROLE_SELECTED, number
    return ROLE_INACTIVE, number


def build_bundle(
    task_spec: TaskSpec,
    archive: bytes,
    *,
    policy: TaskIntegrityPolicy,
    baseline: BaselineSnapshot,
    candidate_version: int | None = None,
) -> ReviewBundle:
    submission, dropped = read_submission(archive, task_spec)
    selection, pattern = _selection(task_spec, submission, candidate_version, policy)
    files: list[ReviewFile] = []
    for path in sorted(set(submission) | set(baseline.files)):
        current = submission.get(path)
        original = baseline.files.get(path)
        if current is None:
            status = STATUS_DELETED
        elif original is None:
            status = STATUS_ADDED
        elif original.sha256 == current.sha256:
            status = STATUS_UNCHANGED
        else:
            status = STATUS_MODIFIED
        role, version = _role(path, task_spec, selection, pattern)
        files.append(
            ReviewFile(
                path=path,
                status=status,
                role=role,
                submission=current,
                baseline=original,
                version=version,
                review_excluded=_is_review_excluded(path, policy.review_exclude),
            )
        )
    instructions, context = default_document_paths(task_spec, policy)
    instruction_documents = [
        (path, baseline.documents[path])
        for path in instructions
        if path in baseline.documents
    ]
    context_documents = [
        (path, baseline.documents[path]) for path in context if path in baseline.documents
    ]
    warnings = list(baseline.warnings)
    if selection is not None and selection.selected_prefix is None:
        warnings.append("the version selector does not name a submitted implementation")
    return ReviewBundle(
        task_id=task_spec.task_id,
        cwd=task_spec.cwd,
        submit_paths=list(task_spec.submit_paths),
        files=files,
        dropped_paths=dropped,
        selection=selection,
        baseline_source=baseline.source,
        instruction_documents=instruction_documents,
        context_documents=context_documents,
        warnings=warnings,
        archive=archive,
    )
