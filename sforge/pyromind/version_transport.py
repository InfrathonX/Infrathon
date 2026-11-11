"""Persist one immutable candidate and evaluate its exact read-only mount.

Only the Controller uses this module. A submission supplies a version identity,
never an object key, filesystem path, mount, or Storage credential.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import threading

from sforge.harness.version_store import VersionStore, StoredVersion
from sforge.pyromind.candidate_mount import MountedCandidateArchive


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class PersistedCandidateTransport:
    """One task/run's exact-version transport; no shared mutable Judge mounts."""

    def __init__(self, mirror, store: VersionStore, task_id: str, run_id: str):
        self.mirror, self.store = mirror, store
        self.task_id, self.run_id = task_id, run_id
        self.output = Path(mirror.output).absolute()
        if store.root.absolute() != self.output / 'version-store':
            raise ValueError('Candidate store must belong to this artifact root')
        if mirror.prefix != 'sforge-pyromind/' + run_id:
            raise ValueError('Candidate Storage prefix must belong to this run')
        self._lock = threading.Lock()

    def _local(self, version: StoredVersion):
        if (version.task_id != self.task_id or version.run_id != self.run_id
                or version.task_directory != self.task_id):
            raise ValueError('Candidate version does not belong to this task/run')
        expected = (self.store.root / 'tasks' / self.task_id / 'runs' / self.run_id
                    / 'versions' / version.version_id).absolute()
        if version.path.absolute() != expected:
            raise ValueError('Candidate version has an unexpected local path')
        # Minting the reference validates every remote path component before I/O.
        relative = expected.relative_to(self.output).as_posix()
        host_path = '/workspace/' + self.mirror.prefix + '/' + relative
        provisional = MountedCandidateArchive(
            task_id=self.task_id, run_id=self.run_id, version_id=version.version_id,
            host_path=host_path, archive_sha256=version.archive_sha256,
            archive_size=version.archive_size, manifest_sha256='0' * 64)
        archive = self.store.read_archive(version)  # no-follow directories + SHA
        # The Controller owns this directory; still reject symlink aliases.
        if expected.resolve() != expected:
            raise ValueError('Candidate version directory must not contain symlinks')
        fd = os.open(expected / 'manifest.json', os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024:
                raise ValueError('Candidate manifest must be a bounded regular file')
            with os.fdopen(fd, 'rb', closefd=False) as stream:
                manifest = stream.read(1024 * 1024 + 1)
        finally:
            os.close(fd)
        if len(manifest) > 1024 * 1024:
            raise ValueError('Candidate manifest grew beyond size limit')
        record = json.loads(manifest)
        for key, value in [('schema_version', 1), ('task_directory', self.task_id),
                           ('task_id', self.task_id), ('run_id', self.run_id),
                           ('version_id', version.version_id),
                           ('archive_sha256', version.archive_sha256),
                           ('archive_size', version.archive_size)]:
            if record.get(key) != value:
                raise ValueError('Candidate manifest identity differs from saved version')
        reference = MountedCandidateArchive(
            task_id=provisional.task_id, run_id=provisional.run_id,
            version_id=provisional.version_id, host_path=provisional.host_path,
            archive_sha256=provisional.archive_sha256,
            archive_size=provisional.archive_size, manifest_sha256=_sha(manifest))
        return reference, relative, archive, manifest

    def _read_remote(self, key: str, size: int):
        try:
            response = self.mirror.client.get_object(self.mirror.bucket, key)
        except Exception as exc:
            if getattr(exc, 'code', None) in {'NoSuchKey', 'NoSuchObject'}:
                return None
            raise RuntimeError('Candidate Storage read failed (' + type(exc).__name__ + ')') from None
        try:
            return response.read(size + 1)
        finally:
            response.close()
            response.release_conn()

    def _ensure_object(self, relative: str, data: bytes):
        key = self.mirror.prefix + '/' + relative
        expected = _sha(data)
        actual = self._read_remote(key, len(data))
        uploaded = False
        if actual is None:
            try:
                self.mirror.client.put_object(self.mirror.bucket, key, io.BytesIO(data), len(data))
            except Exception as exc:
                raise RuntimeError('Candidate Storage upload failed (' + type(exc).__name__ + ')') from None
            uploaded = True
            actual = self._read_remote(key, len(data))
        if actual is None or len(actual) != len(data) or _sha(actual) != expected:
            # An immutable key with different bytes must never be silently repaired.
            raise RuntimeError('Candidate Storage object is absent or differs from its immutable identity')
        self.mirror.seen[relative] = expected
        return {'key': key, 'sha256': expected, 'bytes': len(data),
                'uploaded': uploaded, 'readback_verified': True}

    def persist(self, version: StoredVersion) -> MountedCandidateArchive:
        """Confirm only this archive and manifest; unrelated run files cannot fail admission."""
        with self._lock:
            reference, relative, archive, manifest = self._local(version)
            with self.mirror.lock:
                receipts = [self._ensure_object(relative + '/' + name, data)
                            for name, data in [('submission.tar.gz', archive),
                                               ('manifest.json', manifest)]]
            # Local evidence is deliberately outside the Worker projection.
            receipt = {'schema_version': 1, 'transport': 'readonly-version-mount',
                       'task_id': self.task_id, 'run_id': self.run_id,
                       'version_id': version.version_id, 'archive_sha256': version.archive_sha256,
                       'archive_size': version.archive_size, 'host_path': reference.host_path,
                       'manifest_sha256': reference.manifest_sha256, 'objects': receipts,
                       'receipt_scope': 'latest-verification-only; uploaded is per-call, not lifetime'}
            destination = self.output / 'candidate-transfers' / (version.version_id + '.json')
            destination.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary_name = tempfile.mkstemp(dir=destination.parent, prefix='.receipt-', suffix='.tmp')
            temporary = Path(temporary_name)
            try:
                with os.fdopen(fd, 'w') as stream:
                    json.dump(receipt, stream, indent=2)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
            return reference

    def evaluate(self, evaluator, submission_lookup):
        """Bind server-owned submission IDs to exact versions, including explicit finals."""
        def wrapped(*, stored_version=None, **kwargs):
            if 'mounted_archive' in kwargs:
                raise ValueError('Candidate mount must be resolved by the Controller')
            task = kwargs.get('task_spec')
            if task is None or task.task_id != self.task_id:
                raise ValueError('Evaluation task differs from candidate owner')
            if stored_version is None:
                entry = submission_lookup(kwargs.get('submission_id'))
                if (not isinstance(entry, dict) or entry.get('task_id') != self.task_id
                        or entry.get('run_id') != self.run_id or not entry.get('version_id')):
                    raise ValueError('Submission has no authoritative saved version in this run')
                matches = [v for v in self.store.list_versions(run_id=self.run_id)
                           if v.task_id == self.task_id and v.version_id == entry['version_id']]
                if len(matches) != 1:
                    raise ValueError('Submission saved version is missing or ambiguous')
                stored_version = matches[0]
            archive = kwargs.get('archive')
            if (not isinstance(archive, bytes) or len(archive) != stored_version.archive_size
                    or _sha(archive) != stored_version.archive_sha256):
                raise ValueError('Evaluation bytes differ from the selected saved version')
            reference = self.persist(stored_version)
            return evaluator(**kwargs, mounted_archive=reference)
        return wrapped
