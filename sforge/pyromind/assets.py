"""Stage manifest-pinned assets in controller-owned PyroMind object storage.

``client`` is a Minio client (``StorageClient.client`` / ``ArtifactMirror.client``),
configured by the caller with bounded HTTP timeouts. No credentials are acquired
or serialized here. Mount only the returned ``data_prefix``, read-only; private
Judge datasets must be staged separately from Worker-visible assets.

Only manifest-listed regular files are uploaded. HF snapshot symlinks must first
be materialized, e.g. as hardlinks to vetted blobs in a private staging tree.
Uploads use S3 conditional creation, including multipart completion. An endpoint
that rejects conditional creation fails closed; no unconditional fallback exists.
This prevents this module's concurrent writers from replacing immutable objects,
but cannot protect against other clients with overwrite/delete permissions.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat


CHUNK_SIZE = 4 * 1024 * 1024
PART_SIZE = 16 * 1024 * 1024
MAX_FILE_SIZE = PART_SIZE * 10_000
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK


class AssetError(RuntimeError):
    """Staging or verification failed; no usable bindings are returned."""


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _relative_path(value: str) -> str:
    if (not isinstance(value, str) or not value or "\\" in value
            or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise ValueError("Asset paths must be nonempty relative POSIX paths")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or path.as_posix() != value or any(
            part in (".", "..") for part in path.parts):
        raise ValueError("Asset path is not canonical or escapes its root")
    return value


def _manifest_bytes(manifest) -> tuple[dict, bytes]:
    # Freeze the caller's input before any storage callbacks can mutate it.
    try:
        frozen = json.loads(_canonical(manifest))
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Asset manifest must be finite JSON data") from None
    if not isinstance(frozen, dict) or not isinstance(frozen.get("files"), list):
        raise ValueError("Asset manifest requires a files list")
    if not frozen["files"]:
        raise ValueError("Asset manifest must list at least one file")
    paths = set()
    for entry in frozen["files"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}:
            raise ValueError("Each asset file requires exactly path, size, sha256")
        name = _relative_path(entry["path"])
        if name in paths:
            raise ValueError("Asset manifest contains a duplicate path")
        paths.add(name)
        if type(entry["size"]) is not int or not 0 <= entry["size"] <= MAX_FILE_SIZE:
            raise ValueError("Asset file size exceeds the bounded multipart limit")
        if not isinstance(entry["sha256"], str) or not re.fullmatch(
                r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError("Asset file sha256 must be 64 lowercase hex digits")
    if any(parent.as_posix() in paths for name in paths
           for parent in PurePosixPath(name).parents):
        raise ValueError("Asset manifest has a file/directory path collision")
    frozen["files"].sort(key=lambda item: item["path"])
    return frozen, _canonical(frozen)


def _identity(value: os.stat_result) -> tuple:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
            value.st_size, value.st_mtime_ns, value.st_ctime_ns)


@contextmanager
def _directory_chain(root: Path):
    """Pin and recheck every directory, including ancestors of local_root."""
    descriptors = []
    links = []
    try:
        fd = os.open("/", _DIRECTORY_FLAGS)
        descriptors.append(fd)
        for name in root.parts[1:]:
            child = os.open(name, _DIRECTORY_FLAGS, dir_fd=fd)
            descriptors.append(child)
            links.append((fd, name, child))
            fd = child
        def unchanged():
            for parent, name, child in links:
                linked = os.stat(name, dir_fd=parent, follow_symlinks=False)
                opened = os.fstat(child)
                if not stat.S_ISDIR(linked.st_mode) or (
                        linked.st_dev, linked.st_ino) != (opened.st_dev, opened.st_ino):
                    raise AssetError("Asset source directory changed")
        yield fd, unchanged
        unchanged()
    except OSError:
        raise AssetError("Asset source directory is missing, changed, or unsafe") from None
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


@contextmanager
def _open_file(root_fd: int, name: str):
    descriptors = []
    links = []
    parent = root_fd
    try:
        parts = PurePosixPath(name).parts
        for part in parts[:-1]:
            child = os.open(part, _DIRECTORY_FLAGS, dir_fd=parent)
            descriptors.append(child)
            links.append((parent, part, child))
            parent = child
        fd = os.open(parts[-1], _FILE_FLAGS, dir_fd=parent)
        descriptors.append(fd)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise AssetError("Asset source must be a regular file")

        def unchanged():
            opened = os.fstat(fd)
            linked = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
            if _identity(before) != _identity(opened) or _identity(opened) != _identity(linked):
                raise AssetError("Asset source file changed during staging")
            for ancestor, part, child in links:
                linked_dir = os.stat(part, dir_fd=ancestor, follow_symlinks=False)
                opened_dir = os.fstat(child)
                if not stat.S_ISDIR(linked_dir.st_mode) or (
                        linked_dir.st_dev, linked_dir.st_ino) != (
                            opened_dir.st_dev, opened_dir.st_ino):
                    raise AssetError("Asset source directory changed during staging")

        # A duplicate lets file.close() and descriptor cleanup have separate owners.
        with os.fdopen(os.dup(fd), "rb", buffering=0) as stream:
            yield stream, _identity(before), unchanged
            unchanged()
    except OSError:
        raise AssetError("Asset source file is missing, changed, or unsafe") from None
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _hash_stream(stream, size: int) -> str:
    digest = hashlib.sha256()
    remaining = size
    while remaining:
        chunk = stream.read(min(CHUNK_SIZE, remaining))
        if not chunk or len(chunk) > min(CHUNK_SIZE, remaining):
            raise AssetError("Asset stream length differs from manifest")
        digest.update(chunk)
        remaining -= len(chunk)
    if stream.read(1):
        raise AssetError("Asset stream length differs from manifest")
    return digest.hexdigest()


def _remote_matches(client, bucket: str, key: str, size: int, digest: str) -> bool:
    try:
        response = client.get_object(bucket, key)
    except Exception as error:
        if getattr(error, "code", None) in ("NoSuchKey", "NoSuchObject"):
            return False
        raise AssetError("Asset storage read failed") from None
    try:
        if _hash_stream(response, size) != digest:
            raise AssetError("Immutable asset object has different content")
    except AssetError:
        raise
    except Exception:
        raise AssetError("Asset storage stream verification failed") from None
    finally:
        try:
            try:
                response.close()
            finally:
                response.release_conn()
        except Exception:
            raise AssetError("Asset storage connection cleanup failed") from None
    return True


class _UploadReader:
    def __init__(self, stream, entry: dict, unchanged):
        self.stream, self.entry, self.unchanged = stream, entry, unchanged
        self.size = 0
        self.digest = hashlib.sha256()

    def read(self, length: int) -> bytes:
        if type(length) is not int or not 0 <= length <= PART_SIZE:
            raise AssetError("Upload client requested an unbounded read")
        chunk = self.stream.read(length)
        self.size += len(chunk)
        self.digest.update(chunk)
        return chunk

    def verified(self):
        self.unchanged()
        if (self.size != self.entry["size"]
                or self.digest.hexdigest() != self.entry["sha256"]):
            raise AssetError("Asset source content changed during upload")


def _create_only(client, bucket, key, reader, size):
    """Minio 7.2 has no public conditional-put argument; isolate the HTTP hook.

    Copying the client avoids changing the shared instance or concurrent uploads.
    Its normal multipart handling still aborts uploads if validation/commit fails.
    A successful write is always followed by an independent streamed GET hash.
    """
    if not callable(getattr(client, "_execute", None)):
        raise AssetError("Asset upload requires a Minio client with conditional S3 support")
    uploader = copy.copy(client)
    execute = uploader._execute

    def conditional(method, bucket_name=None, object_name=None, **kwargs):
        query = kwargs.get("query_params") or {}
        commit = ((method == "PUT" and "partNumber" not in query)
                  or (method == "POST" and "uploadId" in query))
        if commit:
            reader.verified()
            kwargs["headers"] = {**(kwargs.get("headers") or {}), "If-None-Match": "*"}
        return execute(method, bucket_name, object_name, **kwargs)

    uploader._execute = conditional
    try:
        uploader.put_object(bucket, key, reader, size,
                            part_size=PART_SIZE, num_parallel_uploads=1)
    except AssetError:
        raise
    except Exception as error:
        if getattr(error, "code", None) == "PreconditionFailed":
            # Another writer won; the caller must verify its exact bytes.
            return False
        raise AssetError("Conditional asset upload failed") from None
    reader.verified()
    return True


def _ensure_bytes(client, bucket, key, data, before_commit=lambda: None):
    digest = hashlib.sha256(data).hexdigest()
    if _remote_matches(client, bucket, key, len(data), digest):
        before_commit()
        return
    reader = _UploadReader(io.BytesIO(data), {"size": len(data), "sha256": digest}, before_commit)
    _create_only(client, bucket, key, reader, len(data))
    if not _remote_matches(client, bucket, key, len(data), digest):
        raise AssetError("Asset object is missing after conditional creation")


def prepare_asset(client, bucket: str, local_root: str | Path, manifest: dict,
                  object_prefix: str) -> dict:
    """Validate, create, and verify an immutable, manifest-addressed asset.

    ``manifest`` contains ``files: [{path, size, sha256}, ...]`` plus optional JSON
    identity metadata such as repo_id/revision. The canonical JSON (sorted paths
    and keys) defines its SHA-256 namespace. Local files are fully hashed before
    any remote write; extra local files are never uploaded. Each existing object
    is rehashed rather than trusting an ETag, metadata, or an existing ready flag.

    Paths and directory chains must not contain symlinks. Changes during staging
    fail before publishing readiness. Partial uploads can be retried; corrupt
    immutable objects cause an error and require explicit operator intervention.
    The caller owns storage credentials, bucket selection, and HTTP timeouts.
    """
    if not isinstance(bucket, str) or not bucket or any(ord(c) < 32 for c in bucket):
        raise ValueError("An asset bucket must be supplied")
    base = _relative_path(object_prefix)
    frozen, encoded = _manifest_bytes(manifest)
    if len(encoded) > PART_SIZE:
        raise ValueError("Asset manifest exceeds the bounded metadata limit")
    digest = hashlib.sha256(encoded).hexdigest()
    prefix = f"{base}/{digest}"
    data_prefix = prefix + "/files"
    files = frozen["files"]
    if any(len(f"{data_prefix}/{entry['path']}".encode()) > 1024 for entry in files):
        raise ValueError("Asset object key exceeds the S3 length limit")
    ready = {"schema_version": 1, "manifest_sha256": digest,
             "file_count": len(files), "total_bytes": sum(entry["size"] for entry in files)}
    ready_bytes = _canonical(ready)
    ready_key = prefix + "/ready.json"
    uploaded = reused = 0
    local = Path(local_root).expanduser().absolute()
    if ".." in local.parts:
        raise ValueError("Asset local_root must not contain traversal")
    with _directory_chain(local) as (root_fd, root_unchanged):
        identities = {}
        for entry in files:
            with _open_file(root_fd, entry["path"]) as (stream, identity, unchanged):
                if identity[4] != entry["size"] or _hash_stream(stream, entry["size"]) != entry["sha256"]:
                    raise AssetError("Local asset size or SHA-256 differs from manifest")
                unchanged()
                identities[entry["path"]] = identity
        was_ready = _remote_matches(client, bucket, ready_key, len(ready_bytes),
                                    hashlib.sha256(ready_bytes).hexdigest())
        manifest_key = prefix + "/manifest.json"
        if was_ready and not _remote_matches(client, bucket, manifest_key, len(encoded), digest):
            raise AssetError("Ready asset manifest is missing")
        _ensure_bytes(client, bucket, manifest_key, encoded)
        for entry in files:
            key = data_prefix + "/" + entry["path"]
            if _remote_matches(client, bucket, key, entry["size"], entry["sha256"]):
                reused += 1
                continue
            if was_ready:
                raise AssetError("Ready asset file is missing")
            with _open_file(root_fd, entry["path"]) as (stream, identity, unchanged):
                if identity != identities[entry["path"]]:
                    raise AssetError("Asset source file changed after preflight")
                reader = _UploadReader(stream, entry, unchanged)
                created = _create_only(client, bucket, key, reader, entry["size"])
                if not _remote_matches(client, bucket, key, entry["size"], entry["sha256"]):
                    raise AssetError("Asset file is missing after conditional creation")
                uploaded += int(created)
                reused += int(not created)
        def sources_unchanged():
            # Include reused files and ancestors of local_root; run this at the
            # conditional ready commit, after its own existence-check GET.
            root_unchanged()
            for entry in files:
                with _open_file(root_fd, entry["path"]) as (stream, identity, unchanged):
                    # Same-size writes may share the filesystem's timestamp tick.
                    # Recheck bytes before publishing readiness, not just stat data.
                    if (identity != identities[entry["path"]]
                            or _hash_stream(stream, entry["size"]) != entry["sha256"]):
                        raise AssetError("Asset source file changed before readiness")
                    unchanged()
        _ensure_bytes(client, bucket, ready_key, ready_bytes, sources_unchanged)
    return {**ready, "bucket": bucket, "object_prefix": prefix,
            "data_prefix": data_prefix, "ready_key": ready_key,
            "uploaded_files": uploaded, "reused_files": reused, "ready": True}
