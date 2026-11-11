"""Single-use ownership and native timing guarantees, without remote allocation."""
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import shlex
import threading
from types import SimpleNamespace
import zlib

import pytest

from sforge.pyromind.swe_fleet import EXEC_FRAME_PROTOCOL, FleetError, SWEImage
from sforge.pyromind.swe_pool import POOL_PROTOCOL, PrewarmedSWEFleet


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def digest(value):
    return 'sha256:' + hashlib.sha256(value).hexdigest()


def image(index=0):
    config = canonical({'config': {'Cmd': ['/bin/bash'], 'WorkingDir': '/testbed'},
                        'rootfs': {'type': 'layers', 'diff_ids': ['sha256:' + 'a' * 64]},
                        'architecture': 'amd64', 'os': 'linux'})
    manifest = canonical({'schemaVersion': 2,
                          'config': {'digest': digest(config), 'size': len(config)},
                          'layers': [{'digest': 'sha256:' + 'b' * 64, 'size': 100}]})
    entry = {'image_name': f'docker.io/swebench/repo-{index}:latest',
             'image_digest': digest(config), 'official_manifest_digest': digest(manifest),
             'base_commit': 'c' * 40, 'runtime_head_commit': 'd' * 40}
    return SWEImage.attest(entry, official_manifest=manifest, base_config=config)


class Missing(RuntimeError):
    status_code = 404


class Sandboxes:
    def __init__(self):
        self.remote = {}
        self.calls = []
        self.lock = threading.Lock()
        self.on_create = self.on_get = None
        self.lose_response = False
        self.confirm_delete = True

    def create(self, request):
        if self.on_create:
            self.on_create(request)
        with self.lock:
            identifier = 'sb-' + str(len(self.calls))
            item = SimpleNamespace(id=identifier, name=request.name, status='running',
                                   image=request.image, resources={**request.resources, 'gpu': '0'})
            self.calls.append(('create', identifier))
            self.remote[identifier] = item
        if self.lose_response:
            raise TimeoutError('secret must not appear')
        return item

    def get_sandbox(self, identifier):
        if self.on_get:
            self.on_get(identifier)
        with self.lock:
            self.calls.append(('get', identifier))
            if identifier not in self.remote:
                raise Missing()
            return self.remote[identifier]

    def exec_command(self, identifier, command, **kwargs):
        with self.lock:
            self.calls.append(('exec', identifier))
        config = json.loads(zlib.decompress(base64.b64decode(shlex.split(command)[5])))
        frame = {'protocol': EXEC_FRAME_PROTOCOL, 'frame_id': config['frame_id'],
                 'returncode': 0, 'output_b64': base64.b64encode(b'worked').decode(),
                 'output_truncated': False, 'supervision_error': None}
        return SimpleNamespace(returncode=0, output=json.dumps(frame), stderr='')

    def pause(self, identifier):
        with self.lock:
            self.calls.append(('pause', identifier))
            self.remote[identifier].status = 'stopped'

    def delete(self, identifier):
        with self.lock:
            self.calls.append(('delete', identifier))
            if self.confirm_delete:
                self.remote.pop(identifier, None)

    def list(self):
        with self.lock:
            self.calls.append(('list',))
            return list(self.remote.values())


def pool(tmp_path, size=1, sdk=None, **kwargs):
    sdk = sdk or Sandboxes()
    fleet = PrewarmedSWEFleet(sdk, tmp_path, 'run-one', [image(i) for i in range(size)],
                              capacity=size, request_factory=lambda img, name, resources:
                              SimpleNamespace(image=img, name=name, resources=resources), **kwargs)
    token = fleet.issue_lease('leg-one')
    return fleet, sdk, token


def prewarm(fleet, token, **kwargs):
    return fleet.prewarm_lease('leg-one', token, measurement_profile_sha256='a' * 64, **kwargs)


def create(index=0, logical='1' * 32):
    pinned = image(index)
    return {'action': 'create', 'sandbox_id': logical, 'image': pinned.image_name,
            'image_digest': pinned.image_digest, 'base_commit': pinned.base_commit,
            'runtime_head_commit': pinned.runtime_head_commit}


def test_native_create_is_sdk_free_and_commits_binding_before_return(tmp_path):
    fleet, sdk, token = pool(tmp_path)
    try:
        receipt = prewarm(fleet, token)
        assert receipt['protocol'] == POOL_PROTOCOL
        assert receipt['physical_sandbox_count'] == 1
        prepared = list(fleet.state['leases']['leg-one']['runtimes'].values())[0]
        physical = prepared['sandbox_id']
        before = list(sdk.calls)
        result = fleet.dispatch('leg-one', token, create())
        assert sdk.calls == before
        assert result == {'created': True, 'sandbox_id': '1' * 32,
                          'image_digest': image().image_digest}
        disk = json.loads(fleet.path.read_text())
        bound = disk['leases']['leg-one']['runtimes']['1' * 32]
        assert bound['sandbox_id'] == physical
        assert bound['single_use_pool_state'] == 'bound'
        assert bound['bound_runtime_id'] == '1' * 32
        assert set(disk['leases']['leg-one']['pool']['logical_bindings']) == {'1' * 32}
        assert token not in fleet.path.read_text()
        assert fleet.path.stat().st_mode & 0o777 == 0o600
    finally:
        assert fleet.close()['cleaned']


def test_physical_ownership_name_and_id_are_durable_before_remote_probes(tmp_path):
    fleet, sdk, token = pool(tmp_path)
    try:
        def before_create(request):
            records = json.loads(fleet.path.read_text())['leases']['leg-one']['runtimes'].values()
            assert any(record['name'] == request.name and record['sandbox_id'] is None
                       for record in records)

        def before_get(identifier):
            records = json.loads(fleet.path.read_text())['leases']['leg-one']['runtimes'].values()
            assert any(record['sandbox_id'] == identifier for record in records)

        sdk.on_create, sdk.on_get = before_create, before_get
        prewarm(fleet, token)
    finally:
        assert fleet.close()['cleaned']


def test_unbound_pool_slot_cannot_execute_or_cleanup_through_native_rpc(tmp_path):
    fleet, sdk, token = pool(tmp_path)
    try:
        prewarm(fleet, token)
        slot = next(iter(fleet.state['leases']['leg-one']['runtimes']))
        for action in ('execute', 'cleanup'):
            before = len(sdk.calls)
            with pytest.raises(FleetError, match='not a bound'):
                fleet.dispatch('leg-one', token, {'action': action, 'sandbox_id': slot})
            assert len(sdk.calls) == before
    finally:
        assert fleet.close()['cleaned']


@pytest.mark.parametrize('field,value', [('image', 'other'), ('image_digest', 'sha256:' + 'f' * 64),
                                       ('base_commit', 'f' * 40), ('runtime_head_commit', 'f' * 40),
                                       ('official_manifest_digest', 'sha256:' + 'f' * 64)])
def test_full_native_identity_is_checked_without_consuming_pool(tmp_path, field, value):
    fleet, sdk, token = pool(tmp_path)
    try:
        prewarm(fleet, token)
        request = create()
        request[field] = value
        before = len(sdk.calls)
        with pytest.raises(FleetError):
            fleet.dispatch('leg-one', token, request)
        assert len(sdk.calls) == before
        assert all(record['single_use_pool_state'] == 'available' for record in
                   fleet.state['leases']['leg-one']['runtimes'].values())
    finally:
        assert fleet.close()['cleaned']


def test_timed_native_create_does_not_prewarm_implicitly(tmp_path):
    fleet, sdk, token = pool(tmp_path)
    try:
        with pytest.raises(FleetError, match='not ready'):
            fleet.dispatch('leg-one', token, create())
        assert not sdk.calls
    finally:
        assert fleet.close()['cleaned']


def test_one_image_can_only_be_bound_once_even_after_native_cleanup(tmp_path):
    fleet, sdk, token = pool(tmp_path)
    try:
        prewarm(fleet, token)
        fleet.dispatch('leg-one', token, create())
        with pytest.raises(FleetError, match='already exists'):
            fleet.dispatch('leg-one', token, create())
        assert fleet.dispatch('leg-one', token, {'action': 'execute', 'sandbox_id': '1' * 32,
                                               'command': 'echo worked'})['output'] == 'worked'
        fleet.dispatch('leg-one', token, {'action': 'cleanup', 'sandbox_id': '1' * 32})
        before = len(sdk.calls)
        with pytest.raises(FleetError, match='No unused'):
            fleet.dispatch('leg-one', token, create(logical='2' * 32))
        assert len(sdk.calls) == before
        assert not sdk.remote
    finally:
        assert fleet.close()['cleaned']


def test_192_parallel_bindings_use_192_distinct_prepared_sandboxes(tmp_path):
    fleet, sdk, token = pool(tmp_path, size=192)
    try:
        assert prewarm(fleet, token)['physical_sandbox_count'] == 192
        before = list(sdk.calls)
        with ThreadPoolExecutor(max_workers=192) as executor:
            results = list(executor.map(lambda index: fleet.dispatch(
                'leg-one', token, create(index, f'{index + 1:032x}')), range(192)))
        assert len(results) == 192
        assert sdk.calls == before
        records = fleet.state['leases']['leg-one']['runtimes'].values()
        assert len({record['sandbox_id'] for record in records}) == 192
        assert all(record['single_use_pool_state'] == 'bound' for record in records)
    finally:
        assert fleet.close()['cleaned']
        assert not sdk.remote


def test_concurrent_attempts_to_bind_same_image_have_exactly_one_winner(tmp_path):
    fleet, sdk, token = pool(tmp_path)
    try:
        prewarm(fleet, token)
        before = len(sdk.calls)
        def bind(index):
            try:
                fleet.dispatch('leg-one', token, create(logical=f'{index + 1:032x}'))
                return True
            except FleetError:
                return False
        with ThreadPoolExecutor(max_workers=32) as executor:
            assert sum(executor.map(bind, range(32))) == 1
        assert len(sdk.calls) == before
    finally:
        assert fleet.close()['cleaned']


def test_lost_create_response_is_reconciled_by_recorded_name(tmp_path):
    fleet, sdk, token = pool(tmp_path)
    try:
        sdk.lose_response = True
        with pytest.raises(FleetError, match='TimeoutError'):
            prewarm(fleet, token)
        assert not sdk.remote
        leg = fleet.state['leases']['leg-one']
        assert leg['status'] == 'closed'
        assert all(record['status'] == 'deleted' and record['sandbox_id'] for record in
                   leg['runtimes'].values())
    finally:
        assert fleet.close()['cleaned']


def test_cancellation_drains_and_deletes_exact_ids(tmp_path):
    fleet, sdk, token = pool(tmp_path, size=2, prepare_concurrency=1)
    started, release, cancel = threading.Event(), threading.Event(), threading.Event()
    errors = []
    def before_get(identifier):
        started.set()
        assert release.wait(5)
    sdk.on_get = before_get
    def run():
        try:
            prewarm(fleet, token, cancel_event=cancel)
        except FleetError as error:
            errors.append(str(error))
    worker = threading.Thread(target=run)
    worker.start()
    assert started.wait(5)
    cancel.set()
    release.set()
    worker.join(10)
    try:
        assert not worker.is_alive()
        assert errors and 'cancelled' in errors[0]
        assert not sdk.remote
        assert fleet.state['leases']['leg-one']['status'] == 'closed'
    finally:
        assert fleet.close()['cleaned']


def test_preparation_concurrency_is_bounded_at_16(tmp_path):
    with pytest.raises(ValueError, match='at most 16'):
        pool(tmp_path, prepare_concurrency=17)
    fleet, sdk, token = pool(tmp_path, size=20, prepare_concurrency=4)
    entered = threading.Event()
    release = threading.Event()
    count_lock = threading.Lock()
    active = peak = 0
    def setup(_sdk, _identifier):
        nonlocal active, peak
        with count_lock:
            active += 1
            peak = max(peak, active)
            if active == 4:
                entered.set()
        assert release.wait(5)
        with count_lock:
            active -= 1
        return {'exec_prefix': ['/trusted/tool'], 'evidence': {'prototype_only': True}}
    fleet.runtime_setup, fleet.runtime_setup_id = setup, 'test-runtime'
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(prewarm, fleet, token)
        assert entered.wait(5)
        release.set()
        assert future.result(10)['physical_sandbox_count'] == 20
    try:
        assert peak == 4
    finally:
        assert fleet.close()['cleaned']


def test_pool_does_not_upgrade_missing_isolation_to_qualification(tmp_path):
    fleet, sdk, token = pool(tmp_path)
    try:
        prewarm(fleet, token)
        readiness = fleet.formal_readiness()
        assert readiness['ready'] is False
        assert 'network_none' in readiness['gaps']
        assert 'tool_runtime_setup_unconfigured' in readiness['gaps']
        assert readiness['qualification'] == 'not_a_192_workload_or_gpu_qualification'
    finally:
        assert fleet.close()['cleaned']


def test_two_controllers_cannot_open_the_same_pool_ledger(tmp_path):
    fleet, sdk, token = pool(tmp_path)
    try:
        with pytest.raises(FleetError, match='already has a Controller owner'):
            pool(tmp_path, sdk=sdk)
        assert not sdk.calls
    finally:
        assert fleet.close()['cleaned']


def test_restart_revokes_bound_and_unbound_slots_until_exact_cleanup(tmp_path):
    fleet, sdk, token = pool(tmp_path, size=2)
    prewarm(fleet, token)
    fleet.dispatch('leg-one', token, create())
    # Simulate process loss without invoking the cleanup hook.
    fleet.rpc_executor.shutdown(wait=True)
    os.close(fleet._owner_descriptor)
    fleet._owner_descriptor = None
    recovered = PrewarmedSWEFleet(sdk, tmp_path, 'run-one', [image(0), image(1)], capacity=2,
                                  request_factory=lambda img, name, resources:
                                  SimpleNamespace(image=img, name=name, resources=resources))
    try:
        assert recovered.state['leases']['leg-one']['pool']['status'] == 'recovery_required'
        with pytest.raises(FleetError, match='closed'):
            recovered.dispatch('leg-one', token, create(1, logical='2' * 32))
        assert recovered.close_lease('leg-one')['cleaned']
        assert not sdk.remote
    finally:
        assert recovered.close()['cleaned']


def test_binding_persistence_error_cannot_return_success_or_reuse_slot(tmp_path, monkeypatch):
    fleet, sdk, token = pool(tmp_path)
    try:
        prewarm(fleet, token)
        persist = fleet._persist
        def failing_persist():
            raise OSError('disk unavailable')
        monkeypatch.setattr(fleet, '_persist', failing_persist)
        with pytest.raises(FleetError, match='durably committed'):
            fleet.dispatch('leg-one', token, create())
        monkeypatch.setattr(fleet, '_persist', persist)
        with pytest.raises(FleetError, match='closed'):
            fleet.dispatch('leg-one', token, create(logical='2' * 32))
    finally:
        assert fleet.close()['cleaned']


def test_unconfirmed_delete_stays_owned_for_retry(tmp_path):
    fleet, sdk, token = pool(tmp_path)
    try:
        prewarm(fleet, token)
        sdk.confirm_delete = False
        result = fleet.close_lease('leg-one')
        assert not result['cleaned'] and result['remaining'] == 1
        assert sdk.remote
        sdk.confirm_delete = True
        assert fleet.close_lease('leg-one')['cleaned']
    finally:
        assert fleet.close()['cleaned']


def test_cold_startup_can_take_200s_before_measurement_while_native_create_stays_sdk_free(
        tmp_path, monkeypatch):
    import sforge.pyromind.swe_fleet as fleet_module
    clock = [0.0]
    monkeypatch.setattr(fleet_module.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(fleet_module.time, 'sleep', lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    value, sdk, token = pool(tmp_path, physical_ready_timeout=900)
    original = sdk.get_sandbox
    def delayed_start(identifier):
        observed = original(identifier)
        observed.status = 'running' if clock[0] >= 200 else 'starting'
        return observed
    sdk.get_sandbox = delayed_start
    try:
        receipt = prewarm(value, token)
        assert clock[0] == 200
        assert value.ready_timeout == 180
        assert value.physical_ready_timeout == 900
        expected = {'protocol': POOL_PROTOCOL, 'prepare_concurrency': 16,
                    'physical_ready_timeout_sec': 900, 'native_create_rpc_timeout_sec': 180}
        assert receipt['preparation_parameters'] == expected
        assert json.loads(value.path.read_text())['pool_preparation_parameters'] == expected
        before = list(sdk.calls)
        value.dispatch('leg-one', token, create())
        assert sdk.calls == before and clock[0] == 200
        with pytest.raises(AttributeError):
            value.physical_ready_timeout = 180
    finally:
        assert value.close()['cleaned']


def test_recovery_rejects_changed_preparation_budget(tmp_path):
    value, sdk, token = pool(tmp_path, physical_ready_timeout=900)
    assert value.close()['cleaned']
    with pytest.raises(ValueError, match='preparation parameters differ'):
        pool(tmp_path, sdk=sdk, physical_ready_timeout=600)


def test_close_drains_private_prewarm_and_keeps_controller_lock_until_late_id_cleanup(tmp_path):
    value, sdk, token = pool(tmp_path)
    created, release = threading.Event(), threading.Event()
    prewarm_errors, closed = [], []
    def before_get(_identifier):
        created.set()
        assert release.wait(5)
    sdk.on_get = before_get
    def prepare():
        try:
            prewarm(value, token)
        except FleetError as error:
            prewarm_errors.append(str(error))
    preparation = threading.Thread(target=prepare)
    preparation.start()
    assert created.wait(5)
    closing = threading.Thread(target=lambda: closed.append(value.close()))
    closing.start()
    with value._condition:
        assert value._condition.wait_for(lambda: value._closing, timeout=5)
    assert not closed
    with pytest.raises(FleetError, match='already has a Controller owner'):
        pool(tmp_path, sdk=sdk)
    release.set()
    preparation.join(10)
    closing.join(10)
    assert not preparation.is_alive() and not closing.is_alive()
    assert prewarm_errors
    assert closed[0]['cleaned']
    assert not sdk.remote
    assert value._owner_descriptor is None


def test_close_persist_failure_retains_ownership_until_delayed_create_and_cleanup_drained(
        tmp_path, monkeypatch):
    value, sdk, token = pool(tmp_path)
    accepted, release = threading.Event(), threading.Event()
    errors = []
    original_create = sdk.create
    def delayed_create(request):
        item = original_create(request)
        accepted.set()
        assert release.wait(5)
        return item
    sdk.create = delayed_create
    def prepare():
        try:
            prewarm(value, token)
        except FleetError as error:
            errors.append(str(error))
    preparation = threading.Thread(target=prepare)
    preparation.start()
    assert accepted.wait(5)
    persist = value._persist
    def unavailable_disk():
        raise OSError('fsync failed')
    monkeypatch.setattr(value, '_persist', unavailable_disk)
    with pytest.raises(OSError, match='fsync failed'):
        value.close()
    assert value._owner_descriptor is not None
    assert preparation.is_alive()
    with pytest.raises(FleetError, match='already has a Controller owner'):
        pool(tmp_path, sdk=sdk)
    monkeypatch.setattr(value, '_persist', persist)
    release.set()
    preparation.join(10)
    assert not preparation.is_alive() and errors
    assert value.close()['cleaned']
    assert not sdk.remote
    assert value._owner_descriptor is None


def test_successful_close_is_cached_after_replacement_controller_acquires_ledger(tmp_path, monkeypatch):
    value, sdk, token = pool(tmp_path)
    prewarm(value, token)
    first = value.close()
    assert first['cleaned'] and value._owner_descriptor is None
    before = list(sdk.calls)
    replacement = PrewarmedSWEFleet(
        sdk, tmp_path, 'run-one', [image()], capacity=1,
        request_factory=lambda img, name, resources:
        SimpleNamespace(image=img, name=name, resources=resources))
    try:
        replacement.issue_lease('replacement-leg')
        committed = value.path.read_bytes()
        def unexpected_persist():
            pytest.fail('A closed former Controller touched its replacement ledger')
        monkeypatch.setattr(value, '_persist', unexpected_persist)
        first['cleaned'] = False
        first['legs']['leg-one']['errors'].append('caller mutation')
        again = value.close()
        assert again['cleaned'] and again['legs']['leg-one']['errors'] == []
        assert value.path.read_bytes() == committed
        assert sdk.calls == before
        again['legs'].clear()
        assert 'leg-one' in value.close()['legs']
    finally:
        assert replacement.close()['cleaned']


def test_concurrent_close_serializes_cleanup_and_returns_independent_cached_results(tmp_path):
    value, sdk, token = pool(tmp_path)
    prewarm(value, token)
    cleanup_entered, release = threading.Event(), threading.Event()
    def delayed_cleanup(_identifier):
        cleanup_entered.set()
        assert release.wait(5)
    sdk.on_get = delayed_cleanup
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(value.close)
        assert cleanup_entered.wait(5)
        second = executor.submit(value.close)
        assert not first.done() and not second.done()
        with pytest.raises(FleetError, match='already has a Controller owner'):
            pool(tmp_path, sdk=sdk)
        release.set()
        results = [first.result(timeout=10), second.result(timeout=10)]
    assert all(result['cleaned'] for result in results)
    assert not sdk.remote and value._owner_descriptor is None
    assert sum(call[0] == 'delete' for call in sdk.calls) == 1
    results[0]['legs']['leg-one']['errors'].append('caller mutation')
    assert results[1]['legs']['leg-one']['errors'] == []


def test_unclean_close_retains_controller_lock_until_confirmed_retry(tmp_path):
    value, sdk, token = pool(tmp_path)
    prewarm(value, token)
    sdk.confirm_delete = False
    failed = value.close()
    assert not failed['cleaned'] and failed['legs']['leg-one']['remaining'] == 1
    assert value._owner_descriptor is not None and sdk.remote
    with pytest.raises(FleetError, match='already has a Controller owner'):
        pool(tmp_path, sdk=sdk)
    sdk.confirm_delete = True
    assert value.close()['cleaned']
    assert not sdk.remote and value._owner_descriptor is None


def test_unclean_close_keeps_factory_sessions_open_for_cleanup_retry(tmp_path):
    sdk = Sandboxes()
    sessions = []
    class Session:
        closed = False
        close_count = 0
        def __getattr__(self, name):
            method = getattr(sdk, name)
            def checked(*args, **kwargs):
                assert not self.closed, 'Cleanup reused a closed SDK session'
                return method(*args, **kwargs)
            return checked
        def close(self):
            self.closed = True
            self.close_count += 1
    def factory():
        value = Session()
        sessions.append(value)
        return value
    value, _, token = pool(tmp_path, sdk=sdk, client_factory=factory)
    prewarm(value, token)
    sdk.confirm_delete = False
    assert not value.close()['cleaned']
    assert sessions and all(session.close_count == 0 for session in sessions)
    assert value._owner_descriptor is not None
    sdk.confirm_delete = True
    assert value.close()['cleaned']
    assert not sdk.remote and all(session.close_count == 1 for session in sessions)
    assert value.close()['cleaned']
    assert all(session.close_count == 1 for session in sessions)
