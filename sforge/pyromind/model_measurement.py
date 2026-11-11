"""Serialize a shared model and retain admission after an unclean native leg.

The flock alone disappears after a Controller crash. A private durable fence
therefore records ownership before allocation and is removed only after the
trusted transport independently confirms cleanup. A different Controller must
not recover it by age, PID or a cached ready flag.
"""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import secrets
import stat
import threading


class PendingModelMeasurement(RuntimeError):
    pass


class ModelMeasurementGate:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.descriptor = None
        self.pending = None
        self.confirmed = False
        self.owner_id = secrets.token_hex(16)
        self._state_lock = threading.RLock()

    def _read_fence(self):
        try:
            descriptor = os.open(self.directory / 'measurement-inflight.json',
                                 os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
        except FileNotFoundError:
            return None
        try:
            info = os.fstat(descriptor)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600
                    or not 1 <= info.st_size <= 4096):
                raise PendingModelMeasurement('Shared model ownership fence is unsafe')
            data = os.read(descriptor, 4097)
            if len(data) != info.st_size:
                raise PendingModelMeasurement('Shared model ownership fence changed')
            value = json.loads(data)
            if (not isinstance(value, dict) or set(value) != {
                    'kind', 'owner_id', 'run_id', 'leg_id', 'measurement_profile_sha256'}
                    or value['kind'] != 'shared-model-cleanup-required-v1'
                    or any(not isinstance(value[key], str) or not value[key]
                           for key in value if key != 'kind')):
                raise PendingModelMeasurement('Shared model ownership fence is invalid')
            return value
        except (ValueError, UnicodeDecodeError):
            raise PendingModelMeasurement('Shared model ownership fence is invalid') from None
        finally:
            os.close(descriptor)

    @contextmanager
    def exclusive(self):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        info = self.directory.stat()
        if (any(path.is_symlink() for path in (self.directory, *self.directory.parents))
                or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700):
            raise PendingModelMeasurement('Shared model directory must be private, canonical and Controller-owned')
        # Normalize ordinary relative/../ CLI paths only after excluding real
        # aliases; all documented launch commands use relative state paths.
        self.directory = self.directory.resolve()
        descriptor = os.open(self.directory / 'measurement.lock',
                             os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        try:
            info = os.fstat(descriptor)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_nlink != 1 or stat.S_IMODE(info.st_mode) != 0o600):
                raise ValueError('Model measurement lock must be a private owned regular file')
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self._state_lock:
                if self.descriptor is not None:
                    raise PendingModelMeasurement('Shared model gate is already active')
                fence = self._read_fence()
                if fence is not None and fence != self.pending:
                    raise PendingModelMeasurement('Prior shared model cleanup must be independently confirmed')
                self.descriptor = descriptor
            try:
                yield
            finally:
                with self._state_lock:
                    try:
                        if self.pending is not None and self.confirmed:
                            if self._read_fence() != self.pending:
                                raise PendingModelMeasurement('Shared model cleanup fence ownership changed')
                            (self.directory / 'measurement-inflight.json').unlink()
                            directory = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                            try:
                                os.fsync(directory)
                            finally:
                                os.close(directory)
                            self.pending = None
                    finally:
                        self.descriptor = None
        finally:
            os.close(descriptor)

    def arm(self, *, run_id, leg_id, measurement_profile_sha256):
        """Durably fence the model before this owner dispatches native create."""
        with self._state_lock:
            if self.descriptor is None or self.pending is not None:
                raise PendingModelMeasurement('Native allocation requires an exclusive clean model gate')
            record = {'kind': 'shared-model-cleanup-required-v1', 'owner_id': self.owner_id,
                      'run_id': run_id, 'leg_id': leg_id,
                      'measurement_profile_sha256': measurement_profile_sha256}
            raw = json.dumps(record, sort_keys=True, allow_nan=False).encode()
            descriptor = os.open(self.directory / 'measurement-inflight.json',
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
            self.pending, self.confirmed = record, False
            try:
                with os.fdopen(descriptor, 'wb') as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                directory = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            except BaseException:
                # Even a failed durability step cannot silently clear ownership.
                raise

    def confirm_cleaned(self):
        # Called only after both native Fleet and Judge exact-ID checks pass.
        # A cleanup worker may signal while the original measurement thread
        # still owns the flock; only exclusive() finally removes the fence.
        with self._state_lock:
            self.confirmed = True
