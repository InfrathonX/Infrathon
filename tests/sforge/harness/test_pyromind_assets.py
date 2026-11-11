import copy
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tracemalloc
from xml.etree import ElementTree as ET

import pytest

pytest.importorskip("minio", reason="needs the pyromind extra (uv sync --extra pyromind)")
from minio import Minio  # noqa: E402
from minio.error import S3Error  # noqa: E402

from sforge.pyromind import assets


def s3_error(code):
    return S3Error(None, code, "DO-NOT-EXPOSE-STORAGE-CREDENTIAL", "resource", "request", "host")


class Response(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.released = False
        self.max_read = 0

    def read(self, amount=-1):
        assert 0 <= amount <= assets.CHUNK_SIZE
        self.max_read = max(self.max_read, amount)
        return super().read(amount)

    def release_conn(self):
        self.released = True


class MemoryS3(Minio):
    """Exercise real Minio single/multipart logic without network or cloud writes."""
    def __init__(self):
        super().__init__("storage.example.invalid", access_key="test", secret_key="test", region="us-east-1")
        self.objects = {}
        self.parts = {}
        self.writes = []
        self.requests = []
        self.responses = []
        self.aborted = []
        self.hook = None
        self.read_hook = None

    def get_object(self, bucket, key):
        if self.read_hook:
            self.read_hook(key)
        if key not in self.objects:
            raise s3_error("NoSuchKey")
        response = Response(self.objects[key])
        self.responses.append(response)
        return response

    def _execute(self, method, bucket_name=None, object_name=None, **kwargs):
        query = kwargs.get("query_params") or {}
        headers = kwargs.get("headers") or {}
        self.requests.append((method, object_name, dict(headers), dict(query)))
        if self.hook:
            self.hook(method, object_name, kwargs)
        if method == "POST" and "uploads" in query:
            self.parts[object_name] = {}
            return SimpleNamespace(data=b"<InitiateMultipartUploadResult><UploadId>upload</UploadId></InitiateMultipartUploadResult>")
        if method == "DELETE":
            self.parts.pop(object_name, None)
            self.aborted.append(object_name)
            return SimpleNamespace()
        if method == "PUT" and "partNumber" in query:
            assert "If-None-Match" not in headers
            self.parts[object_name][int(query["partNumber"])] = kwargs["body"]
            return SimpleNamespace(headers={"etag": "irrelevant-multipart-etag"})
        assert headers["If-None-Match"] == "*"
        if object_name in self.objects:
            raise s3_error("PreconditionFailed")
        if method == "POST":
            part_rows = ET.fromstring(kwargs["body"])
            assert len(part_rows) == len(self.parts[object_name])
            data = b"".join(self.parts[object_name][n] for n in sorted(self.parts[object_name]))
            self.parts.pop(object_name)
            response = SimpleNamespace(headers={}, data=(
                f"<CompleteMultipartUploadResult><Bucket>{bucket_name}</Bucket>"
                f"<Key>{object_name}</Key><ETag>opaque-etag</ETag>"
                "<Location>unused</Location></CompleteMultipartUploadResult>").encode())
        else:
            data = kwargs["body"]
            response = SimpleNamespace(headers={"etag": "irrelevant-single-etag"})
        self.objects[object_name] = data
        self.writes.append(object_name)
        return response


def fixture_manifest(root, data=None):
    data = data or {"config.json": b'{"architecture":"test"}', "weights.bin": b"weights"}
    entries = []
    for name, content in data.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        entries.append({"path": name, "size": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    return {"repo_id": "test/model", "revision": "frozen", "files": entries}


def prepare(client, root, manifest):
    return assets.prepare_asset(client, "assets-bucket", root, manifest, "assets/models")


def test_prepare_hashes_remote_bytes_and_idempotently_reuses(tmp_path):
    manifest = fixture_manifest(tmp_path)
    original = copy.deepcopy(manifest)
    (tmp_path / "secret-not-in-manifest").write_text("never upload")
    client = MemoryS3()
    first = prepare(client, tmp_path, manifest)
    assert first["ready"] and first["uploaded_files"] == 2 and first["reused_files"] == 0
    assert client.writes[-1] == first["ready_key"]
    assert first["object_prefix"] == "assets/models/" + first["manifest_sha256"]
    assert first["total_bytes"] == sum(f["size"] for f in manifest["files"])
    assert all("secret" not in key for key in client.objects)
    written = list(client.writes)
    manifest["files"].reverse()
    second = prepare(client, tmp_path, manifest)
    assert second["manifest_sha256"] == first["manifest_sha256"]
    assert second["uploaded_files"] == 0 and second["reused_files"] == 2
    assert client.writes == written
    assert all(response.closed and response.released for response in client.responses)
    assert json.loads(client.objects[first["object_prefix"] + "/manifest.json"]) == original
    assert "_execute" not in client.__dict__  # No mutation of shared client methods.


def test_real_minio_multipart_create_is_conditional(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "PART_SIZE", 5 * 1024 * 1024)
    manifest = fixture_manifest(tmp_path, {"weights.bin": b"x" * (assets.PART_SIZE + 17)})
    client = MemoryS3()
    result = prepare(client, tmp_path, manifest)
    assert client.objects[result["data_prefix"] + "/weights.bin"] == (tmp_path / "weights.bin").read_bytes()
    completion = [request for request in client.requests if request[0] == "POST" and "uploadId" in request[3]]
    assert len(completion) == 1 and completion[0][2]["If-None-Match"] == "*"
    assert not client.parts


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../b", "a//b", "./a", "a/", "a\\b", "a\x00b", "a\nb", "", "."])
def test_reject_unsafe_manifest_paths_without_storage_calls(tmp_path, name):
    manifest = {"files": [{"path": name, "size": 0, "sha256": hashlib.sha256(b"").hexdigest()}]}
    client = MemoryS3()
    with pytest.raises(ValueError):
        prepare(client, tmp_path, manifest)
    assert not client.requests and not client.responses


@pytest.mark.parametrize("mutation", [
    lambda m: m["files"].append(dict(m["files"][0])),
    lambda m: m["files"].append({**m["files"][0], "path": "config.json/subfile"}),
    lambda m: m["files"][0].update(size=True),
    lambda m: m["files"][0].update(size=-1),
    lambda m: m["files"][0].update(size=assets.MAX_FILE_SIZE + 1),
    lambda m: m["files"][0].update(sha256="f" * 63),
    lambda m: m["files"][0].update(sha256="F" * 64),
    lambda m: m.update(revision=float("nan")),
])
def test_reject_invalid_manifest(tmp_path, mutation):
    manifest = fixture_manifest(tmp_path)
    mutation(manifest)
    with pytest.raises(ValueError):
        prepare(MemoryS3(), tmp_path, manifest)


@pytest.mark.parametrize("change", ["hash", "size", "missing"])
def test_all_local_files_verified_before_first_write(tmp_path, change):
    manifest = fixture_manifest(tmp_path)
    target = tmp_path / "weights.bin"
    if change == "missing":
        target.unlink()
    else:
        target.write_bytes(b"changed" if change == "hash" else b"truncated")
    client = MemoryS3()
    with pytest.raises(assets.AssetError):
        prepare(client, tmp_path, manifest)
    assert client.writes == []


@pytest.mark.parametrize("kind", ["file", "directory", "root", "fifo"])
def test_symlinks_and_non_regular_files_fail_closed(tmp_path, kind):
    root = tmp_path / "root"
    root.mkdir()
    manifest = fixture_manifest(root, {"nested/file": b"data"})
    if kind == "file":
        (root / "nested/file").rename(tmp_path / "outside")
        (root / "nested/file").symlink_to(tmp_path / "outside")
    elif kind == "directory":
        (root / "nested").rename(tmp_path / "outside")
        (root / "nested").symlink_to(tmp_path / "outside", target_is_directory=True)
    elif kind == "root":
        root.rename(tmp_path / "outside")
        root.symlink_to(tmp_path / "outside", target_is_directory=True)
    else:
        (root / "nested/file").unlink()
        os.mkfifo(root / "nested/file")
    client = MemoryS3()
    with pytest.raises(assets.AssetError):
        prepare(client, root, manifest)
    assert client.writes == []


@pytest.mark.parametrize("target", ["file", "manifest", "ready"])
def test_existing_corrupt_objects_are_never_overwritten(tmp_path, target):
    manifest = fixture_manifest(tmp_path)
    client = MemoryS3()
    result = prepare(client, tmp_path, manifest)
    key = {"file": result["data_prefix"] + "/weights.bin",
           "manifest": result["object_prefix"] + "/manifest.json", "ready": result["ready_key"]}[target]
    client.objects[key] = b"corrupt"
    written = list(client.writes)
    with pytest.raises(assets.AssetError):
        prepare(client, tmp_path, manifest)
    assert client.writes == written and client.objects[key] == b"corrupt"


def test_ready_asset_missing_file_is_not_silently_repaired(tmp_path):
    manifest = fixture_manifest(tmp_path)
    client = MemoryS3()
    result = prepare(client, tmp_path, manifest)
    del client.objects[result["data_prefix"] + "/weights.bin"]
    written = list(client.writes)
    with pytest.raises(assets.AssetError, match="Ready asset file is missing"):
        prepare(client, tmp_path, manifest)
    assert client.writes == written


def test_partial_upload_has_no_ready_and_retry_reuses_verified_objects(tmp_path):
    manifest = fixture_manifest(tmp_path)
    client = MemoryS3()
    def fail(method, key, kwargs):
        if method == "PUT" and key.endswith("weights.bin"):
            raise s3_error("AccessDenied")
    client.hook = fail
    with pytest.raises(assets.AssetError) as failure:
        prepare(client, tmp_path, manifest)
    assert "DO-NOT-EXPOSE" not in str(failure.value)
    assert not any(key.endswith("/ready.json") for key in client.objects)
    assert any(key.endswith("config.json") for key in client.objects)
    client.hook = None
    result = prepare(client, tmp_path, manifest)
    assert result["uploaded_files"] == 1 and result["reused_files"] == 1


@pytest.mark.parametrize("corrupt", [False, True])
def test_concurrent_creator_wins_without_overwrite_and_is_verified(tmp_path, corrupt):
    manifest = fixture_manifest(tmp_path)
    client = MemoryS3()
    def race(method, key, kwargs):
        if method == "PUT" and key.endswith("weights.bin"):
            client.objects[key] = b"corrupt" if corrupt else b"weights"
    client.hook = race
    if corrupt:
        with pytest.raises(assets.AssetError):
            prepare(client, tmp_path, manifest)
        assert not any(key.endswith("/ready.json") for key in client.objects)
    else:
        result = prepare(client, tmp_path, manifest)
        assert result["uploaded_files"] == 1 and result["reused_files"] == 1
    assert not any(key.endswith("weights.bin") for key in client.writes)


def test_upload_readback_detects_corruption_without_ready(tmp_path):
    manifest = fixture_manifest(tmp_path)
    client = MemoryS3()
    def corrupt(key):
        if key.endswith("weights.bin") and key in client.objects:
            client.objects[key] = b"corrupt"
    client.read_hook = corrupt
    with pytest.raises(assets.AssetError):
        prepare(client, tmp_path, manifest)
    assert not any(key.endswith("/ready.json") for key in client.objects)


def test_manifest_input_mutation_cannot_change_the_upload_allowlist(tmp_path):
    manifest = fixture_manifest(tmp_path)
    client = MemoryS3()
    def mutate(method, key, kwargs):
        manifest["files"][0]["path"] = "../../secret"
        manifest["revision"] = "different"
    client.hook = mutate
    result = prepare(client, tmp_path, manifest)
    frozen = json.loads(client.objects[result["object_prefix"] + "/manifest.json"])
    assert frozen["revision"] == "frozen"
    assert result["file_count"] == 2
    assert not any("secret" in key for key in client.objects)


def test_late_input_mutation_before_ready_is_detected(tmp_path, monkeypatch):
    manifest = fixture_manifest(tmp_path)
    client = MemoryS3()
    # Simulate a same-size rewrite within one filesystem timestamp tick.
    original_identity = assets._identity
    monkeypatch.setattr(assets, "_identity", lambda value: original_identity(value)[:5] + (0, 0))
    checks = []
    def mutate(key):
        if key.endswith("ready.json"):
            checks.append(key)
            if len(checks) == 2:
                (tmp_path / "weights.bin").write_bytes(b"changed")
    client.read_hook = mutate
    with pytest.raises(assets.AssetError, match="changed before readiness"):
        prepare(client, tmp_path, manifest)
    assert not any(key.endswith("/ready.json") for key in client.objects)


def test_root_directory_replacement_is_detected_before_ready(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    manifest = fixture_manifest(root)
    client = MemoryS3()
    def mutate(method, key, kwargs):
        if key.endswith("manifest.json"):
            root.rename(tmp_path / "old")
            root.mkdir()
    client.hook = mutate
    with pytest.raises(assets.AssetError, match="directory changed"):
        prepare(client, root, manifest)
    assert not any(key.endswith("/ready.json") for key in client.objects)


def test_distinct_revision_changes_namespace_even_with_identical_files(tmp_path):
    manifest = fixture_manifest(tmp_path)
    client = MemoryS3()
    first = prepare(client, tmp_path, manifest)
    manifest["revision"] = "different-frozen-revision"
    second = prepare(client, tmp_path, manifest)
    assert first["object_prefix"] != second["object_prefix"]


def test_hardlinks_to_vetted_blobs_are_supported(tmp_path):
    root = tmp_path / "stage"
    root.mkdir()
    manifest = fixture_manifest(root)
    original = root / "weights.bin"
    original.rename(tmp_path / "blob")
    os.link(tmp_path / "blob", original)
    assert prepare(MemoryS3(), root, manifest)["ready"]


@pytest.mark.parametrize("kind", ["write", "replace", "directory"])
def test_local_change_between_preflight_and_upload_never_marks_ready(tmp_path, kind):
    root = tmp_path / "root"
    root.mkdir()
    manifest = fixture_manifest(root, {"nested/file": b"payload"})
    client = MemoryS3()
    def mutate(method, key, kwargs):
        if not key.endswith("manifest.json"):
            return
        if kind == "write":
            (root / "nested/file").write_bytes(b"changed")
        elif kind == "replace":
            (root / "replacement").write_bytes(b"payload")
            (root / "replacement").replace(root / "nested/file")
        else:
            (root / "nested").rename(root / "old")
            (root / "nested").symlink_to(root / "old", target_is_directory=True)
    client.hook = mutate
    with pytest.raises(assets.AssetError):
        prepare(client, root, manifest)
    assert not any(key.endswith("/ready.json") for key in client.objects)


def test_mutation_during_multipart_aborts_before_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, "PART_SIZE", 5 * 1024 * 1024)
    manifest = fixture_manifest(tmp_path, {"weights.bin": b"x" * (assets.PART_SIZE + 17)})
    client = MemoryS3()
    def mutate(method, key, kwargs):
        if str((kwargs.get("query_params") or {}).get("partNumber")) == "1":
            with (tmp_path / "weights.bin").open("r+b") as stream:
                stream.write(b"y")
    client.hook = mutate
    with pytest.raises(assets.AssetError, match="changed"):
        prepare(client, tmp_path, manifest)
    assert len(client.aborted) == 1 and not client.parts
    assert not any(key.endswith(("/ready.json", "weights.bin")) for key in client.objects)


def test_remote_error_and_stream_error_are_sanitized_and_connections_closed(tmp_path):
    manifest = fixture_manifest(tmp_path)
    client = MemoryS3()
    client.get_object = lambda *args: (_ for _ in ()).throw(s3_error("AccessDenied"))
    with pytest.raises(assets.AssetError) as failure:
        prepare(client, tmp_path, manifest)
    assert "DO-NOT-EXPOSE" not in str(failure.value)
    response = Response(b"payload")
    response.read = lambda amount: (_ for _ in ()).throw(RuntimeError("DO-NOT-EXPOSE"))
    client.get_object = lambda *args: response
    with pytest.raises(assets.AssetError) as failure:
        assets._remote_matches(client, "bucket", "key", 7, "a" * 64)
    assert "DO-NOT-EXPOSE" not in str(failure.value)
    assert response.closed and response.released


def test_more_than_four_gib_streams_have_constant_read_and_memory_bounds(monkeypatch):
    """A logical >4-GiB stream catches integer truncation/unbounded reads cheaply.

    Cryptographic correctness is covered by real SHA tests above; this counter
    hash permits checking every stream iteration without hashing 15 GiB in CI.
    """
    size = 5 * 1024**3 + 17
    class CountHash:
        def __init__(self):
            self.size = 0
        def update(self, data):
            self.size += len(data)
        def hexdigest(self):
            return str(self.size)
    class LogicalStream:
        def __init__(self):
            self.remaining = size
            self.block = b"x" * assets.PART_SIZE
            self.reads = []
        def read(self, amount):
            assert 0 <= amount <= assets.PART_SIZE
            count = min(amount, self.remaining)
            self.remaining -= count
            self.reads.append(count)
            return self.block if count == len(self.block) else memoryview(self.block)[:count]
    class StreamingClient:
        def _execute(self, method, bucket, key, **kwargs):
            assert kwargs["headers"]["If-None-Match"] == "*"
        def put_object(self, bucket, key, reader, length, **kwargs):
            assert kwargs == {"part_size": assets.PART_SIZE, "num_parallel_uploads": 1}
            left = length
            while left:
                left -= len(reader.read(min(left, kwargs["part_size"])))
            self._execute("POST", bucket, key, query_params={"uploadId": "upload"})
    monkeypatch.setattr(assets.hashlib, "sha256", CountHash)
    tracemalloc.start()
    try:
        local = LogicalStream()
        assert assets._hash_stream(local, size) == str(size)
        upload = LogicalStream()
        reader = assets._UploadReader(upload, {"size": size, "sha256": str(size)}, lambda: None)
        assert assets._create_only(StreamingClient(), "bucket", "key", reader, size)
        assert reader.size == size and len(upload.reads) > 256
        assert max(local.reads) == assets.CHUNK_SIZE
        assert max(upload.reads) == assets.PART_SIZE
        assert tracemalloc.get_traced_memory()[1] < 40 * 1024 * 1024
    finally:
        tracemalloc.stop()
