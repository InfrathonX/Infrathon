"""CPU-only protocol/integrity tests; these do not qualify a real kernel."""
import hashlib
import ast
import io
import json
import re
import shlex
import stat
from types import SimpleNamespace

import pytest

from sforge.pyromind import swe_tool_runtime as runtime


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def owner_fixture(identity='a' * 32, affinity_count=None):
    affinity = list(range(affinity_count or 4))
    return {
        'schema_version': 1, 'identity': identity,
        'job_directory': '/run/sforge-swe-tool/' + identity,
        'pid': 42, 'pid_starttime': 98765, 'bootstrap_pid': 40, 'unshare_pid': 41,
        'namespaces': {name: 100 + index for index, name in enumerate(runtime._NAMESPACES)},
        'outer_namespaces': {name: 200 + index for index, name in enumerate(runtime._NAMESPACES)},
        'binaries': {name: '/usr/bin/' + ('python3' if name == 'python' else name)
                     for name in ('unshare', 'mount', 'nsenter', 'setpriv', 'env', 'python')},
        'environment': {'HOME': '/root', 'PATH': '/opt/conda/bin:/usr/bin:/bin',
                        'LANG': 'C.UTF-8', 'SHELL': '/bin/bash', 'USER': 'root',
                        'LOGNAME': 'root', 'TMPDIR': '/tmp', 'XDG_CACHE_HOME': '/tmp/sforge-cache'},
        'affinity': affinity if affinity_count is not None else None,
        'holder_affinity': affinity, 'holder_dumpable': False, 'mount_propagation_private': True,
    }


class FakeSDK:
    """Return synthetic trusted evidence, never create or exec a Sandbox."""
    def __init__(self, change_owner=None, change_probe=None, fail_at=None):
        self.calls = []
        self.change_owner = change_owner
        self.change_probe = change_probe
        self.fail_at = fail_at
        self.owner = None

    def exec_command(self, sandbox_id, transport, **kwargs):
        argv = shlex.split(transport)
        self.calls.append((sandbox_id, argv, kwargs))
        if len(self.calls) == self.fail_at:
            return SimpleNamespace(returncode=126, output='private SDK diagnostic', stderr='private token')
        if argv[3] == runtime._BOOTSTRAP:
            cfg = json.loads(argv[4])
            self.owner = owner_fixture(cfg['identity'], cfg['affinity_count'])
            if self.change_owner:
                self.change_owner(self.owner)
            result = {'owner': self.owner, 'owner_path': self.owner['job_directory'] + '/owner.json',
                      'owner_sha256': hashlib.sha256(canonical(self.owner)).hexdigest()}
        else:
            assert argv[3] == runtime._LAUNCHER
            assert argv[9] == runtime._PROBE
            phase = argv[-1]
            result = {'phase': phase, 'namespaces': self.owner['namespaces'],
                      'capabilities': {name: 0 for name in ('CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb')},
                      'affinity': self.owner['holder_affinity']}
            result.update({name: True for name in (
                'no_new_privs', 'controller_state_hidden', 'holder_ptrace_denied',
                'proc_namespace_verified', 'mount_propagation_private', 'network_route_unreachable', 'cgroup2_readonly',
                'image_home_preserved_readonly', 'testbed_writable')})
            if self.change_probe:
                self.change_probe(result)
        return SimpleNamespace(returncode=0, output=runtime._MARKER + json.dumps(result), stderr='')


@pytest.mark.parametrize('affinity_count', [None, 2])
def test_hook_proves_persistent_namespace_tmp_and_records_explicit_gaps(affinity_count):
    sdk = FakeSDK()
    result = runtime.prepare_swe_tool_runtime(sdk, 'sandbox-one', affinity_count=affinity_count)
    assert len(sdk.calls) == 3
    assert all(call[0] == 'sandbox-one' for call in sdk.calls)
    assert sdk.calls[1][1][-1] == 'write' and sdk.calls[2][1][-1] == 'read'
    prefix, evidence = result['exec_prefix'], result['evidence']
    assert prefix == sdk.calls[1][1][:6] == sdk.calls[2][1][:6]
    assert prefix[:2] == ['/usr/bin/python3', '-I']
    assert evidence['runtime_setup_id'] == runtime.RUNTIME_SETUP_ID
    assert evidence['prototype_only'] is True and evidence['formal_task_ready'] is False
    assert evidence['holder_pid'] == 42 and evidence['holder_pid_starttime'] == 98765
    assert evidence['bootstrap_pid'] == 40 and evidence['unshare_pid'] == 41
    assert evidence['persistent_namespace_and_tmp_verified'] is True
    assert evidence['cpu_cfs_quota_enforced'] is False and evidence['pids_cgroup_limit_enforced'] is False
    assert set(evidence['qualification_gaps']) == set(runtime._GAPS)
    assert evidence['affinity_limit_requested'] == affinity_count
    assert len(canonical(result)) < 32 * 1024


@pytest.mark.parametrize('fail_at', [1, 2, 3])
def test_hook_fails_closed_and_never_retries_or_leaks_sdk_diagnostic(fail_at):
    sdk = FakeSDK(fail_at=fail_at)
    with pytest.raises(runtime.ToolRuntimeError) as caught:
        runtime.prepare_swe_tool_runtime(sdk, 'sandbox-one')
    assert len(sdk.calls) == fail_at
    assert 'private' not in str(caught.value) and 'token' not in str(caught.value)


@pytest.mark.parametrize('mutation', [
    lambda owner: owner.update(pid=True),
    lambda owner: owner.update(holder_dumpable=True),
    lambda owner: owner.update(mount_propagation_private=False),
    lambda owner: owner.update(pid_starttime=0),
    lambda owner: owner['namespaces'].update(net=owner['outer_namespaces']['net']),
    lambda owner: owner['environment'].update(SDK_API_TOKEN='private'),
    lambda owner: owner['environment'].update(HOME='/tmp'),
    lambda owner: owner['binaries'].update(python='python3'),
    lambda owner: owner.update(affinity=[0, 1]),
])
def test_hook_rejects_owner_evidence_before_any_tool_exec(mutation):
    sdk = FakeSDK(change_owner=mutation)
    with pytest.raises(runtime.ToolRuntimeError, match='owner evidence'):
        runtime.prepare_swe_tool_runtime(sdk, 'sandbox-one')
    assert len(sdk.calls) == 1


@pytest.mark.parametrize('mutation', [
    lambda probe: probe.update(no_new_privs=False),
    lambda probe: probe.update(controller_state_hidden=False),
    lambda probe: probe.update(holder_ptrace_denied=False),
    lambda probe: probe.update(proc_namespace_verified=False),
    lambda probe: probe.update(mount_propagation_private=False),
    lambda probe: probe.update(network_route_unreachable=False),
    lambda probe: probe.update(cgroup2_readonly=False),
    lambda probe: probe.update(testbed_writable=False),
    lambda probe: probe['capabilities'].update(CapEff=1),
    lambda probe: probe['capabilities'].update(CapBnd=False),
    lambda probe: probe.update(phase='other'),
    lambda probe: probe.update(affinity=[]),
])
def test_hook_rejects_unverified_isolation(mutation):
    sdk = FakeSDK(change_probe=mutation)
    with pytest.raises(runtime.ToolRuntimeError, match='isolation evidence'):
        runtime.prepare_swe_tool_runtime(sdk, 'sandbox-one')
    assert len(sdk.calls) == 2


@pytest.mark.parametrize('options', [
    {'setup_timeout': True}, {'setup_timeout': 9}, {'setup_timeout': 301},
    {'affinity_count': True}, {'affinity_count': 0}, {'affinity_count': 257},
])
def test_invalid_settings_do_not_contact_sdk(options):
    sdk = FakeSDK()
    with pytest.raises(ValueError):
        runtime.prepare_swe_tool_runtime(sdk, 'sandbox-one', **options)
    assert sdk.calls == []


@pytest.mark.parametrize('source', [runtime._COMMON, runtime._HOLDER, runtime._BOOTSTRAP,
                                    runtime._LAUNCHER, runtime._ENTRY, runtime._PROBE,
                                    runtime._MOUNT_DIAGNOSTIC_HELPERS, runtime._MOUNT_PROBE,
                                    runtime._PLATFORM_CAPABILITY_PROBE])
def test_transported_python_compiles_without_local_or_remote_side_effects(source):
    compile(source, '<swe-tool-runtime>', 'exec')


def launcher_globals(owner, *, starts=None, wrong_namespace=None, digest=None, uid=0):
    payload = canonical(owner)
    calls = []
    descriptors = {}
    starts = iter(starts or [owner['pid_starttime'], owner['pid_starttime']])

    def namespace_open(path, flags):
        assert flags & 0o2000000  # O_CLOEXEC on Linux.
        if path == '/proc/42/root':
            assert flags & 0o10000000 and flags & 0o200000
            descriptors[25] = 'root'
            calls.append(('root_open', path))
            return 25
        assert path.startswith('/proc/42/ns/')
        name = path.rsplit('/', 1)[1]
        descriptor = 20 + len(descriptors)
        descriptors[descriptor] = name
        calls.append(('namespace_open', path))
        return descriptor

    def namespace_stat(descriptor):
        name = descriptors[descriptor]
        if name == 'root':
            return SimpleNamespace(st_mode=stat.S_IFDIR | 0o755)
        return SimpleNamespace(st_ino=owner['namespaces'][name] + (1 if name == wrong_namespace else 0))

    fake_os = SimpleNamespace(
        geteuid=lambda: uid, path=SimpleNamespace(isabs=lambda path: path.startswith('/'),
                                                dirname=lambda path: path.rsplit('/', 1)[0]),
        fdopen=lambda fd, mode: io.BytesIO(payload), O_RDONLY=0, O_CLOEXEC=0o2000000,
        O_PATH=0o10000000, O_DIRECTORY=0o200000,
        open=namespace_open, fstat=namespace_stat,
        set_inheritable=lambda fd, value: calls.append(('inheritable', fd, value)),
        sched_setaffinity=lambda pid, cpus: calls.append(('affinity', pid, cpus)),
        chdir=lambda path: calls.append(('chdir', path)),
        execv=lambda path, argv: calls.append(('exec', path, argv)),
    )
    namespace = {'os': fake_os, 'sys': SimpleNamespace(argv=[
        'python3', owner['job_directory'] + '/owner.json',
        digest or hashlib.sha256(payload).hexdigest(), 'bash', '-lc', 'literal $(secret)']),
        'json': json, 'hashlib': hashlib, 're': re, 'stat': stat, 'NAMESPACES': runtime._NAMESPACES,
        'private_directory': lambda path: calls.append(('private_directory', path)),
        'secure_file': lambda path: 10,
        'starttime': lambda pid: next(starts)}
    return namespace, calls


def test_launcher_pins_checked_namespaces_before_exec_and_clears_controller_environment():
    owner = owner_fixture(affinity_count=2)
    namespace, calls = launcher_globals(owner)
    exec(runtime._LAUNCHER[len(runtime._COMMON):], namespace)
    assert len([call for call in calls if call[0] == 'namespace_open']) == 5
    execute = calls[-1]
    assert execute[:2] == ('exec', '/usr/bin/nsenter')
    argv = execute[2]
    assert argv[1:6] == ['--user=/proc/self/fd/20', '--mount=/proc/self/fd/21',
                        '--net=/proc/self/fd/22', '--pid=/proc/self/fd/23', '--ipc=/proc/self/fd/24']
    assert '--target' not in argv
    assert argv[6:8] == ['--root=/proc/self/fd/25', '--wd=/proc/self/fd/25']
    assert argv[8:17] == ['--preserve-credentials', '--', '/usr/bin/setpriv',
                         '--bounding-set=-all', '--inh-caps=-all', '--ambient-caps=-all',
                         '--no-new-privs', '/usr/bin/env', '-i']
    assert ('affinity', 0, [0, 1]) in calls
    assert argv[-3:] == ['bash', '-lc', 'literal $(secret)']
    assert runtime._ENTRY in argv
    assert 'PATH=/opt/conda/bin:/usr/bin:/bin' in argv and 'HOME=/root' in argv


@pytest.mark.parametrize('options', [
    {'starts': [111]}, {'starts': [98765, 111]},
    {'wrong_namespace': 'pid'}, {'digest': 'f' * 64}, {'uid': 1000},
])
def test_launcher_rejects_pid_reuse_namespace_change_owner_tamper_and_tool_invocation(options):
    namespace, calls = launcher_globals(owner_fixture(), **options)
    with pytest.raises(AssertionError):
        exec(runtime._LAUNCHER[len(runtime._COMMON):], namespace)
    assert not any(call[0] == 'exec' for call in calls)


@pytest.mark.parametrize('changes', [
    {'st_mode': stat.S_IFREG | 0o644}, {'st_uid': 1000}, {'st_nlink': 2},
    {'st_mode': stat.S_IFLNK | 0o600},
])
def test_controller_owner_file_requires_root_regular_private_single_link(changes):
    namespace = {}
    exec(runtime._COMMON, namespace)
    value = {'st_mode': stat.S_IFREG | 0o600, 'st_uid': 0, 'st_nlink': 1}
    value.update(changes)
    calls = []
    namespace['os'] = SimpleNamespace(
        O_RDONLY=0, O_NOFOLLOW=1, O_CLOEXEC=2,
        open=lambda path, flags: calls.append((path, flags)) or 10,
        fstat=lambda fd: SimpleNamespace(**value))
    with pytest.raises(AssertionError):
        namespace['secure_file']('/run/controller/owner.json')
    assert calls == [('/run/controller/owner.json', 3)]


def test_starttime_handles_spaces_and_parentheses_in_process_comm():
    namespace = {}
    exec(runtime._COMMON, namespace)
    # Fields after comm begin with state (field 3); starttime is field 22.
    namespace['open'] = lambda path: io.StringIO('42 (holder (odd) name) S ' +
                                               ' '.join(['0'] * 18 + ['123456', '0']))
    assert namespace['starttime'](42) == 123456


def entry_globals(*, cap=0, no_new_privs=1, wrong_namespace=False, readonly=True, visible=False):
    owner = owner_fixture()
    calls = []
    status = '\n'.join(name + ':\t' + format(cap, 'x') for name in (
        'CapInh', 'CapPrm', 'CapEff', 'CapBnd', 'CapAmb')) + '\nNoNewPrivs:\t' + str(no_new_privs) + '\n'
    fake_os = SimpleNamespace(
        stat=lambda path: SimpleNamespace(st_ino=owner['namespaces'][path.rsplit('/', 1)[1]] +
                                         (1 if wrong_namespace else 0)),
        statvfs=lambda path: SimpleNamespace(f_flag=1 if readonly else 0), ST_RDONLY=1,
        path=SimpleNamespace(exists=lambda path: visible),
        listdir=lambda path: ['0', '1', '2', '3', '4'],
        close=lambda fd: calls.append(('close', fd)),
        chdir=lambda path: calls.append(('chdir', path)), environ=owner['environment'],
        execvpe=lambda path, argv, env: calls.append(('exec', path, argv, env)),
    )
    namespace = {'os': fake_os, 'json': json, 're': re,
                 'sys': SimpleNamespace(argv=['python3', json.dumps({
                     'namespaces': owner['namespaces'], 'job_directory': owner['job_directory']}),
                     'bash', '-lc', 'literal $(secret)']),
                 'open': lambda path: io.StringIO(
                     '1 0 0:1 / / ro - overlay overlay ro\n' if path.endswith('/mountinfo') else status)}
    return namespace, calls


def test_inner_guard_checks_privileges_root_namespace_and_closes_controller_fds():
    namespace, calls = entry_globals()
    exec('expected=' + runtime._ENTRY.split('expected=', 1)[1], namespace)
    assert calls[:3] == [('close', 3), ('close', 4), ('chdir', '/testbed')]
    assert calls[-1][0:3] == ('exec', 'bash', ['bash', '-lc', 'literal $(secret)'])


@pytest.mark.parametrize('options', [
    {'cap': 1}, {'no_new_privs': 0}, {'wrong_namespace': True},
    {'readonly': False}, {'visible': True},
])
def test_inner_guard_fails_before_accepting_a_tool_command(options):
    namespace, calls = entry_globals(**options)
    with pytest.raises(AssertionError):
        exec('expected=' + runtime._ENTRY.split('expected=', 1)[1], namespace)
    assert calls == []


def test_bootstrap_delays_propagation_until_trusted_holder_but_retains_all_namespaces():
    tree = ast.parse(runtime._BOOTSTRAP)
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'process' for target in node.targets))
    calls = []
    namespace = {
        'subprocess': SimpleNamespace(Popen=lambda argv, **kwargs: calls.append((argv, kwargs)), DEVNULL=-3),
        'binaries': {name: '/usr/bin/' + name for name in ('unshare', 'python')},
        'cfg': {'holder_source': runtime._HOLDER}, 'holder': {'job_directory': '/run/private'},
        'json': json, 'write_fd': 12, 'log_fd': 13, 'environment': {'PATH': '/usr/bin'},
    }
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<bootstrap-launch>', 'exec'), namespace)
    argv, options = calls[0]
    assert argv[1:10] == ['--user', '--map-root-user', '--mount', '--net', '--pid', '--ipc',
                          '--propagation=unchanged', '--fork', '--kill-child=KILL']
    assert argv[10:13] == ['/usr/bin/python', '-I', '-c']
    assert argv[13] == runtime._HOLDER
    assert options['start_new_session'] is True and options['pass_fds'] == (12,)


@pytest.mark.parametrize('optional, expected', [
    ('', True), ('unbindable', True), ('shared:123', False),
    ('master:456', False), ('master:456 propagate_from:789', False),
])
def test_mount_propagation_proof_rejects_outward_and_inward_shared_mounts(optional, expected):
    namespace = {}
    exec(runtime._COMMON, namespace)
    assert namespace['private_mount_propagation']([
        '1 0 0:1 / / ro ' + optional + ' - overlay overlay ro']) is expected
    assert namespace['private_mount_propagation']([]) is False
    assert namespace['private_mount_propagation'](['malformed']) is False


@pytest.mark.parametrize('overlay', [True, False])
@pytest.mark.parametrize('namespace_order', ['combined', 'user_first'])
def test_kernel_probe_command_keeps_every_namespace_and_explicit_private_propagation(overlay, namespace_order):
    argv = shlex.split(runtime.mount_namespace_probe_command(overlay=overlay, namespace_order=namespace_order))
    expected = ['unshare', '--user', '--map-root-user']
    if namespace_order == 'user_first':
        expected += ['--propagation=unchanged', '--', 'unshare']
    expected += ['--mount', '--net', '--pid', '--ipc', '--propagation=unchanged', '--fork', '--kill-child=KILL']
    assert argv[:len(expected)] == expected
    assert argv[len(expected):len(expected) + 3] == ['python3', '-I', '-c']
    assert argv[-2] == runtime._MOUNT_PROBE
    assert json.loads(argv[-1]) == {'overlay': overlay, 'namespace_order': namespace_order}
    assert "mount('recursive_private',None,'/',None,MS_REC|MS_PRIVATE)" in argv[-2]
    assert "'qualification_passed':False" in argv[-2]


def test_kernel_probe_does_not_accept_command_or_resource_settings():
    with pytest.raises(ValueError):
        runtime.mount_namespace_probe_command(overlay='true')
    with pytest.raises(ValueError):
        runtime.mount_namespace_probe_command(namespace_order='omit_mount')


@pytest.mark.parametrize('namespace_mode', ['outer', 'user_net_pid_ipc_without_mount'])
def test_platform_probe_only_queries_landlock_and_cgroup_and_never_supplies_tool_prefix(namespace_mode):
    argv = shlex.split(runtime.platform_isolation_probe_command(namespace_mode=namespace_mode))
    if namespace_mode == 'outer':
        assert argv[:3] == ['python3', '-I', '-c']
    else:
        assert argv[:9] == ['unshare', '--user', '--map-root-user', '--net', '--pid', '--ipc',
                           '--propagation=unchanged', '--fork', '--kill-child=KILL']
        assert '--mount' not in argv
    assert argv[-2:] == [runtime._PLATFORM_CAPABILITY_PROBE, namespace_mode]
    assert "'qualification_passed':False" in runtime._PLATFORM_CAPABILITY_PROBE
    assert "'policy_installed':False" in runtime._PLATFORM_CAPABILITY_PROBE
    assert 'libc.mount(' not in runtime._PLATFORM_CAPABILITY_PROBE
    assert 'landlock_restrict_self' not in runtime._PLATFORM_CAPABILITY_PROBE
    assert 'exec_prefix' not in runtime._PLATFORM_CAPABILITY_PROBE


def test_platform_probe_rejects_arbitrary_modes():
    with pytest.raises(ValueError):
        runtime.platform_isolation_probe_command(namespace_mode='unconfined')


def test_platform_probe_records_proc_permission_denial_as_unknown_not_a_private_proc_proof():
    tree = ast.parse(runtime._PLATFORM_CAPABILITY_PROBE)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'proc_pid1_namespace')
    def denied(path):
        raise PermissionError(13, 'private profile diagnostic')
    namespace = {'os': SimpleNamespace(stat=denied), 'errno': __import__('errno')}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<proc-observation>', 'exec'), namespace)
    result = namespace['proc_pid1_namespace'](100)
    assert result == {'readable': False, 'matches_current_namespace': None,
                      'errno': 13, 'errno_name': 'EACCES'}
    assert 'private' not in json.dumps(result)


@pytest.mark.parametrize('observed, expected', [(100, True), (101, False)])
def test_platform_probe_distinguishes_proc_namespace_match_from_permission_failure(observed, expected):
    tree = ast.parse(runtime._PLATFORM_CAPABILITY_PROBE)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'proc_pid1_namespace')
    namespace = {'os': SimpleNamespace(stat=lambda path: SimpleNamespace(st_ino=observed)),
                 'errno': __import__('errno')}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<proc-observation>', 'exec'), namespace)
    assert namespace['proc_pid1_namespace'](100) == {'readable': True, 'matches_current_namespace': expected}


def diagnostic_globals():
    namespace = {'os': __import__('os'), 're': re, 'hashlib': hashlib}
    exec(runtime._MOUNT_DIAGNOSTIC_HELPERS, namespace)
    return namespace


@pytest.mark.parametrize('number', [1, 13, 30])  # EPERM, EACCES, EROFS remain distinct.
def test_direct_mount_diagnostic_records_original_errno_without_retry(number):
    namespace = diagnostic_globals()
    calls = []
    def mount(*arguments):
        calls.append(arguments)
        namespace['ctypes'].set_errno(number)
        return -1
    result = namespace['diagnostic_mount'](SimpleNamespace(mount=mount),
                                          'private', None, '/', None, 0x44000)
    assert len(calls) == 1
    assert calls == [(None, b'/', None, 0x44000, None)]
    assert result['errno'] == number and result['returncode'] == -1 and result['ok'] is False
    assert result['errno_name'] == {1: 'EPERM', 13: 'EACCES', 30: 'EROFS'}[number]


def test_mount_summary_omits_source_and_superblock_options_and_blocks_shared_propagation():
    namespace = diagnostic_globals()
    row = '1 0 0:1 / /tmp rw master:3 - tmpfs private-source credential=private\n'
    rows = namespace['mount_rows']([row])
    assert namespace['safe_independent_mounts'](rows) is True
    assert 'private-source' not in json.dumps(rows) and 'credential' not in json.dumps(rows)
    rows[0]['propagation'].append('shared:7')
    assert namespace['safe_independent_mounts'](rows) is False
    assert namespace['safe_independent_mounts']([]) is False
    with pytest.raises((AssertionError, ValueError)):
        namespace['mount_rows'](['malformed'])


@pytest.mark.parametrize('value, category', [(b'unconfined\n', 'unconfined'),
                                           (b'private-policy-value (enforce)\n', 'confined_or_other')])
def test_lsm_diagnostic_does_not_render_arbitrary_profile_names(value, category):
    namespace = diagnostic_globals()
    namespace['open'] = lambda *args: io.BytesIO(value)
    result = namespace['safe_lsm_label']()
    assert result['category'] == category and result['readable'] is True
    assert 'private-policy-value' not in json.dumps(result)
    assert len(result['sha256']) == 64


@pytest.mark.parametrize('failure', [False, True])
def test_namespace_owner_diagnostic_closes_both_handles_and_preserves_unknown_scope(failure):
    namespace = diagnostic_globals()
    calls = []
    def ioctl(fd, operation):
        assert (fd, operation) == (10, 0xb701)
        if failure:
            raise PermissionError(1, 'private error detail')
        return 11
    namespace['os'] = SimpleNamespace(
        O_RDONLY=0, O_CLOEXEC=2, open=lambda path, flags: 10,
        fstat=lambda fd: SimpleNamespace(st_ino=100), close=lambda fd: calls.append(fd))
    namespace['fcntl'] = SimpleNamespace(ioctl=ioctl)
    result = namespace['namespace_user_owners']({'user': 100, 'mnt': 200})
    if failure:
        assert result == {'mnt': {'errno': 1, 'errno_name': 'EPERM'}}
        assert calls == [10]
    else:
        assert result['mnt']['owned_by_current_user_namespace'] is True
        assert calls == [11, 10]
    assert 'private' not in json.dumps(result)


def run_synthetic_mount_diagnostic(*, propagation='', private=False, overlay=False, tmpfs_ok=False):
    """Exercise transported control flow with no namespaces or mount syscalls."""
    common = {}
    exec(runtime._COMMON, common)
    helpers = diagnostic_globals()
    mounts = '1 0 0:1 / / rw ' + propagation + ' - overlay overlay rw\n'
    calls, output = [], []
    names = {name: 100 + index for index, name in enumerate(runtime._NAMESPACES)}
    status = 'CapEff:\t000001ffffffffff\nSeccomp:\t0\nSeccomp_filters:\t0\nNoNewPrivs:\t0\n'
    def mount(libc, label, *arguments):
        calls.append(label)
        ok = label == 'recursive_private' and private or label == 'new_tmpfs' and tmpfs_ok
        return {'operation': label, 'ok': ok, 'errno': 0 if ok else 13, 'errno_name': None if ok else 'EACCES'}
    fake_os = SimpleNamespace(
        getpid=lambda: 1, geteuid=lambda: 0,
        uname=lambda: SimpleNamespace(release='test-kernel', machine='x86_64'),
        path=SimpleNamespace(isdir=lambda path: True, islink=lambda path: False),
        mkdir=lambda *args: None, statvfs=lambda path: SimpleNamespace(f_flag=1), ST_RDONLY=1)
    libc = SimpleNamespace(mount=SimpleNamespace(argtypes=None, restype=None))
    namespace = {
        'json': json, 're': re, 'sys': SimpleNamespace(argv=['python3', json.dumps({
            'overlay': overlay, 'namespace_order': 'combined'})], exit=lambda code: output.append(code)),
        'os': fake_os, 'namespace_ids': lambda pid: names,
        'namespace_user_owners': lambda ids: {'mnt': {'owned_by_current_user_namespace': True}},
        'safe_lsm_label': lambda: {'category': 'confined_or_other'},
        'mount_rows': helpers['mount_rows'], 'safe_independent_mounts': helpers['safe_independent_mounts'],
        'private_mount_propagation': lambda lines: private,
        'ctypes': SimpleNamespace(CDLL=lambda *args, **kwargs: libc, c_char_p=1, c_ulong=2, c_int=3),
        'diagnostic_mount': mount, 'private_directory': lambda path: None,
        'tempfile': SimpleNamespace(mkdtemp=lambda **kwargs: '/tmp/probe'),
        'open': lambda path: io.StringIO(mounts if path.endswith('mountinfo') else status),
        'print': lambda text, **kwargs: output.append(text),
    }
    namespace.update({name: helpers[name] for name in (
        'MS_RDONLY', 'MS_NOSUID', 'MS_NODEV', 'MS_NOEXEC', 'MS_REMOUNT', 'MS_BIND', 'MS_REC', 'MS_PRIVATE')})
    exec('cfg=json.loads' + runtime._MOUNT_PROBE.split('cfg=json.loads', 1)[1], namespace)
    return json.loads(output[0].split('=', 1)[1]), calls, output[-1]


def test_diagnostic_continues_independent_syscalls_after_private_failure_but_cannot_qualify():
    result, calls, code = run_synthetic_mount_diagnostic(propagation='master:3', overlay=True)
    assert calls == ['recursive_private', 'new_tmpfs', 'new_pid_proc', 'new_testbed_bind', 'root_bind_readonly']
    assert result['diagnostic_completed'] is True and result['status'] == 'failed' and code == 1
    assert result['mounts_private_after_syscall'] is False and result['qualification_passed'] is False
    assert result['overlay_skipped_reason'] == 'private_tmpfs_unavailable'


def test_diagnostic_blocks_new_mounts_when_any_inherited_mount_can_propagate_outward():
    result, calls, code = run_synthetic_mount_diagnostic(propagation='shared:3 master:4')
    assert calls == ['recursive_private']
    assert result['independent_mounts_attempted'] is False and code == 1
    assert result['independent_mounts_blocked_reason'] == 'inherited_shared_mounts_can_propagate_outward'
    assert result['qualification_passed'] is False


@pytest.mark.parametrize('output', ['', '{}', runtime._MARKER + '[]',
                                    runtime._MARKER + '{}\n' + runtime._MARKER + '{}',
                                    'x' * (32 * 1024 + 1)])
def test_missing_duplicate_nonmapping_or_oversized_evidence_fails(output):
    sdk = SimpleNamespace(exec_command=lambda *args, **kwargs: SimpleNamespace(returncode=0, output=output))
    with pytest.raises(runtime.ToolRuntimeError, match='invalid evidence'):
        runtime.prepare_swe_tool_runtime(sdk, 'sandbox-one')


def test_transport_error_is_sanitized_and_not_retried():
    calls = []
    def execute(*args, **kwargs):
        calls.append(args)
        raise TimeoutError('private credentials')
    sdk = SimpleNamespace(exec_command=execute)
    with pytest.raises(runtime.ToolRuntimeError, match='transport failed: TimeoutError') as caught:
        runtime.prepare_swe_tool_runtime(sdk, 'sandbox-one')
    assert len(calls) == 1 and 'private' not in str(caught.value)


def test_failure_reports_only_known_phase_and_exception_identifiers():
    sdk = SimpleNamespace(exec_command=lambda *args, **kwargs: SimpleNamespace(
        returncode=1, output=runtime._FAILURE_MARKER + json.dumps({
            'phase': 'private_pid_proc_mount', 'error_type': 'RuntimeError'}), stderr='private credentials'))
    with pytest.raises(runtime.ToolRuntimeError, match=r'failed at private_pid_proc_mount \(RuntimeError\)'):
        runtime.prepare_swe_tool_runtime(sdk, 'sandbox-one')


@pytest.mark.parametrize('failure', [
    {'phase': 'private_secret_value', 'error_type': 'RuntimeError'},
    {'phase': 'filesystem_proof', 'error_type': 'private_secret_value'},
    {'phase': 'filesystem_proof', 'error_type': 'RuntimeError', 'secret': 'private_secret_value'},
])
def test_failure_record_cannot_echo_unknown_fields_or_credential_shaped_identifiers(failure):
    sdk = SimpleNamespace(exec_command=lambda *args, **kwargs: SimpleNamespace(
        returncode=1, output=runtime._FAILURE_MARKER + json.dumps(failure), stderr='private credentials'))
    with pytest.raises(runtime.ToolRuntimeError) as caught:
        runtime.prepare_swe_tool_runtime(sdk, 'sandbox-one')
    assert 'private_secret_value' not in str(caught.value) and 'credentials' not in str(caught.value)
