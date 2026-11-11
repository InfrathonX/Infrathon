"""Single-use, Controller-owned physical SWE pool for native create RPCs.

Image pulls, platform readiness and clean-repository checks happen in
``prewarm_lease`` before the native measurement starts. The unchanged native
``create`` request only commits a fresh logical UUID to its already prepared,
matching physical Sandbox. Cleanup still deletes that physical Sandbox; it is
never reset and returned to the pool. This is a transport implementation, not
platform isolation or workload qualification evidence.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import fcntl
import hashlib
import os
from pathlib import Path
import re
import secrets
import stat
import threading
import time

from .swe_fleet import FleetError, SWEFleet, _canonical


POOL_PROTOCOL = 'single-use-native-create-binding-v1'
PROFILE_SHA256 = re.compile(r'[a-f0-9]{64}')


class PrewarmedSWEFleet(SWEFleet):
    """A Fleet whose timed native create never allocates or probes a Sandbox.

    The constructor owns a nonblocking process lock for the lifetime of the
    Fleet. On recovery, live legs are revoked and must be reconciled; a new
    process cannot resume a measured leg using stale readiness observations.
    ``create_fleet_router`` and all existing cleanup APIs work unchanged.
    """

    binding_protocol = POOL_PROTOCOL

    def __init__(self, client, state_root, run_id, images, *,
                 prepare_concurrency=16, physical_ready_timeout=900, **kwargs):
        if type(prepare_concurrency) is not int or not 1 <= prepare_concurrency <= 16:
            raise ValueError('Physical pool preparation concurrency must be at most 16')
        if type(physical_ready_timeout) is not int or not 1 <= physical_ready_timeout <= 3600:
            raise ValueError('Explicit physical pool startup budget must be between 1 and 3600s')
        self._physical_ready_timeout = physical_ready_timeout
        self._preparation_concurrency = prepare_concurrency
        root = Path(state_root)
        if root.is_symlink():
            raise ValueError('Pool state root must not be a symlink')
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root.chmod(0o700)
        lock_path = root / 'pool-owner.lock'
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC
                             | getattr(os, 'O_NOFOLLOW', 0), 0o600)
        try:
            info = os.fstat(descriptor)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid()
                    or info.st_nlink != 1):
                raise ValueError('Unsafe pool process ownership lock')
            os.fchmod(descriptor, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise FleetError('Physical SWE pool already has a Controller owner') from None
            self._owner_descriptor = descriptor
            self._prewarm_workers = prepare_concurrency
            self._prewarm_active = 0
            self._pool_stop = threading.Event()
            self._close_mutex = threading.Lock()
            self._closed_result = None
            recovering = (root / 'fleet.json').exists()
            super().__init__(client, root, run_id, images,
                             prepare_concurrency=prepare_concurrency, **kwargs)
            with self._lock:
                recorded_parameters = self.state.get('pool_preparation_parameters')
                if (recorded_parameters is not None
                        and recorded_parameters != self.preparation_parameters):
                    raise ValueError('Pool recovery preparation parameters differ from its ledger')
                if recorded_parameters is None:
                    self.state['pool_preparation_parameters'] = dict(self.preparation_parameters)
                    self._persist()
            if recovering:
                with self._lock:
                    changed = False
                    for leg in self.state['leases'].values():
                        live = any(record['status'] != 'deleted'
                                   for record in leg['runtimes'].values())
                        if (live or 'pool' in leg) and leg['status'] != 'closed':
                            leg['status'] = 'closing'
                            if 'pool' in leg:
                                leg['pool']['status'] = 'recovery_required'
                            changed = True
                    if changed:
                        self._persist()
        except BaseException:
            executor = getattr(self, 'rpc_executor', None)
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=False)
            self._owner_descriptor = None
            os.close(descriptor)
            raise

    @property
    def physical_ready_timeout(self):
        return self._physical_ready_timeout

    @property
    def preparation_parameters(self):
        return {'protocol': POOL_PROTOCOL, 'prepare_concurrency': self._preparation_concurrency,
                'physical_ready_timeout_sec': self._physical_ready_timeout,
                'native_create_rpc_timeout_sec': 180}

    def prewarm_lease(self, leg_id, capability, *, measurement_profile_sha256,
                     image_names=None, cancel_event=None):
        """Prepare one fresh physical Sandbox per requested corpus image.

        The caller first issues a lease. Provider hooks normally leave
        ``image_names`` unset, preparing all 192 images. Smaller selections are
        useful for staging and are not a 192-workload qualification. Admission,
        physical ownership and every successful logical binding are durable.
        Any preparation error or cancellation revokes the whole leg, drains
        accepted preparation calls, and attempts exact-ID reconciliation.
        """
        with self._condition:
            if self._closing:
                raise FleetError('Physical SWE pool is closing')
            self._prewarm_active += 1
        try:
            return self._prewarm_lease(leg_id, capability,
                                      measurement_profile_sha256=measurement_profile_sha256,
                                      image_names=image_names, cancel_event=cancel_event)
        finally:
            with self._condition:
                self._prewarm_active -= 1
                self._condition.notify_all()

    def _prewarm_lease(self, leg_id, capability, *, measurement_profile_sha256,
                      image_names, cancel_event):
        if (not isinstance(measurement_profile_sha256, str)
                or not PROFILE_SHA256.fullmatch(measurement_profile_sha256)):
            raise ValueError('Pin the pool measurement profile SHA-256')
        names = sorted(self.images) if image_names is None else list(image_names)
        if (not names or len(names) > self.capacity
                or any(not isinstance(name, str) or name not in self.images for name in names)
                or len(set(names)) != len(names)):
            raise ValueError('Physical pool requires unique allowlisted corpus images')
        if cancel_event is not None and not callable(getattr(cancel_event, 'is_set', None)):
            raise ValueError('Pool cancellation must expose is_set()')
        with self._condition:
            leg = self._authenticate(leg_id, capability)
            if self._closing or leg['status'] != 'open' or 'pool' in leg or leg['runtimes']:
                raise FleetError('Physical pool cannot be prepared for this lease')
            leg['pool'] = {'protocol': POOL_PROTOCOL, 'status': 'preparing',
                           'measurement_profile_sha256': measurement_profile_sha256,
                           'preparation_parameters': dict(self.preparation_parameters),
                           'image_names': names, 'started_unix': time.time(),
                           'logical_bindings': {}}
            self._persist()
            self._operations[leg_id] = self._operations.get(leg_id, 0) + 1

        def cancelled():
            return self._pool_stop.is_set() or (cancel_event is not None and cancel_event.is_set())

        failure = None
        try:
            if cancelled():
                raise FleetError('Physical SWE pool preparation was cancelled')

            def prepare(name):
                if cancelled():
                    raise FleetError('Physical SWE pool preparation was cancelled')
                image = self.images[name]
                slot = secrets.token_hex(16)
                request = {'action': 'create', 'sandbox_id': slot, 'image': image.image_name,
                           'image_digest': image.image_digest, 'base_commit': image.base_commit,
                           'runtime_head_commit': image.runtime_head_commit}
                # Calling the base implementation is exclusively a Controller
                # preparation operation; no native caller can enter this path.
                SWEFleet._create_physical(self, leg_id, request,
                                         startup_timeout=self.physical_ready_timeout)
                with self._lock:
                    record = leg['runtimes'][slot]
                    record.update(pool_entry_id=slot, single_use_pool_state='available',
                                  bound_runtime_id=None)
                    self._persist()
                if cancelled():
                    raise FleetError('Physical SWE pool preparation was cancelled')

            # Do not cancel submitted futures: a remote create whose response
            # is late still needs registering and exact-ID compensation.
            with ThreadPoolExecutor(max_workers=self._prewarm_workers,
                                    thread_name_prefix='swe-pool-prepare') as executor:
                futures = [executor.submit(prepare, name) for name in names]
                for future in as_completed(futures):
                    try:
                        future.result()
                    except BaseException as error:
                        if failure is None:
                            failure = error
                            with self._lock:
                                leg['status'] = 'closing'
                                leg['pool']['status'] = 'failed'
                                self._persist()
                if failure is not None:
                    raise failure
            with self._lock:
                if cancelled() or self._closing or leg['status'] != 'open':
                    raise FleetError('Physical SWE pool preparation was cancelled')
                records = list(leg['runtimes'].values())
                if (len(records) != len(names)
                        or {record['image']['image_name'] for record in records} != set(names)
                        or len({record['sandbox_id'] for record in records}) != len(names)
                        or any(record['status'] != 'ready'
                               or record.get('single_use_pool_state') != 'available'
                               or not record.get('baseline_verified') for record in records)):
                    raise FleetError('Physical SWE pool preparation is incomplete')
                completed = time.time()
                leg['pool'].update(status='ready', completed_unix=completed)
                self._persist()
                return {'protocol': POOL_PROTOCOL, 'run_id': self.run_id, 'leg_id': leg_id,
                        'measurement_profile_sha256': measurement_profile_sha256,
                        'preparation_parameters': dict(self.preparation_parameters),
                        'completed_unix': completed, 'physical_sandbox_count': len(records),
                        'physical_sandbox_ids_sha256': hashlib.sha256(_canonical(
                            sorted(record['sandbox_id'] for record in records))).hexdigest()}
        except BaseException as error:
            failure = error
            with self._lock:
                leg['status'] = 'closing'
                leg['pool']['status'] = 'failed'
                self._persist()
        finally:
            with self._condition:
                self._operations[leg_id] -= 1
                self._condition.notify_all()
        # The operation must be drained before close_lease; calling it inside
        # the operation would wait for this very prewarm call to finish.
        self.close_lease(leg_id)
        if isinstance(failure, (KeyboardInterrupt, SystemExit)):
            raise failure
        if isinstance(failure, FleetError):
            raise FleetError(self._redact(str(failure))) from None
        raise FleetError('Physical SWE pool preparation failed: '
                         + type(failure).__name__) from None

    def _create(self, leg_id, request):
        runtime_id = self._id(request, ('action', 'sandbox_id', 'image', 'image_digest',
                                        'base_commit', 'runtime_head_commit'))
        image = self.images.get(request.get('image'))
        if image is None or any(request.get(key) != getattr(image, key) for key in
                                ('image_digest', 'base_commit', 'runtime_head_commit')):
            raise FleetError('Sandbox image differs from the pinned corpus')
        with self._lock:
            leg = self.state['leases'][leg_id]
            pool = leg.get('pool')
            if (self._closing or leg['status'] != 'open'
                    or pool is None or pool['status'] != 'ready'):
                raise FleetError('Physical SWE pool is not ready for native create')
            if runtime_id in leg['runtimes'] or runtime_id in pool['logical_bindings']:
                raise FleetError('Sandbox runtime ID already exists')
            available = [(slot, record) for slot, record in leg['runtimes'].items()
                         if record.get('single_use_pool_state') == 'available'
                         and record['status'] == 'ready' and record['image'] == image.identity()]
            if len(available) != 1:
                raise FleetError('No unused physical SWE Sandbox matches the native create')
            slot, record = available[0]
            record.update(single_use_pool_state='bound', bound_runtime_id=runtime_id,
                          bound_unix=time.time())
            del leg['runtimes'][slot]
            leg['runtimes'][runtime_id] = record
            pool['logical_bindings'][runtime_id] = slot
            self._runtime_locks[(leg_id, runtime_id)] = self._runtime_locks.pop(
                (leg_id, slot), threading.Lock())
            try:
                self._persist()
            except BaseException:
                # A failed fsync/rename is not a successful create. Never reuse
                # a possibly durable binding; recovery revokes and deletes it.
                leg['status'] = 'closing'
                pool['status'] = 'binding_persistence_failed'
                raise FleetError('Native SWE binding could not be durably committed') from None
            return {'created': True, 'sandbox_id': runtime_id,
                    'image_digest': image.image_digest}

    def _record(self, leg_id, runtime_id):
        record = super()._record(leg_id, runtime_id)
        if (record.get('single_use_pool_state') != 'bound'
                or record.get('bound_runtime_id') != runtime_id):
            raise FleetError('Physical pool entry is not a bound native runtime')
        return record

    def close(self, *, timeout_per_leg=300):
        # Evaluator shutdown and the router lifespan can both call close.
        # After releasing flock this instance must never touch the ledger or
        # SDK again: a replacement Controller may already own them.
        with self._close_mutex:
            if self._closed_result is not None:
                return deepcopy(self._closed_result)
            result = self._close_owned(timeout_per_leg=timeout_per_leg)
            if result['cleaned']:
                cached = deepcopy(result)
                descriptor = self._owner_descriptor
                if descriptor is not None:
                    os.close(descriptor)
                    self._owner_descriptor = None
                self._closed_result = cached
            # A failed cleanup retains ownership and can be retried. A normal
            # unclean result is no more safe to relinquish than an exception.
            return deepcopy(result)

    def _close_owned(self, *, timeout_per_leg):
        self._pool_stop.set()
        # Prewarm uses its own <=16 executor rather than the RPC executor.
        # Drain it, including its compensation, before the base close can
        # close SDK sessions or release the process ownership lock.
        with self._condition:
            self._closing = True
            for leg in self.state['leases'].values():
                if leg['status'] != 'closed':
                    leg['status'] = 'closing'
            self._persist()
            while self._prewarm_active:
                self._condition.wait(timeout=1)
        return super().close(timeout_per_leg=timeout_per_leg)
