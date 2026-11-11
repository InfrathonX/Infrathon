import json
import os
from pathlib import Path
import struct
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sforge.pyromind.gpu_runtime import (
    _distributed_source, _GPU_PROBE, _link_platform_cache, _prepare_source, _run_gpu_probe,
    _set_owner_only_default_acl, capture_gpu_topology, prepare_hami_runtime, prepare_private_tmp,
    verify_distributed_runtime,
)


def initialized_cache(tmp_path):
    root = tmp_path / "platform"
    root.mkdir()
    target = root / "allocation.cache"
    target.write_bytes(struct.pack("=i", 19920718) + b"allocated memory and accounting")
    return root, target, tmp_path / "default.cache"


def test_alias_preserves_allocated_cache_bytes_and_inode(tmp_path):
    root, target, default = initialized_cache(tmp_path)
    original = target.read_bytes()
    identity = target.stat().st_ino
    evidence = _link_platform_cache(target, default, root, expected_uid=os.getuid())
    assert default.is_symlink()
    assert default.samefile(target)
    assert target.stat().st_ino == identity
    assert target.read_bytes() == original
    assert evidence["same_accounting_inode"] and evidence["symlink_created"]
    repeated = _link_platform_cache(target, default, root, expected_uid=os.getuid())
    assert repeated["symlink_created"] is False


@pytest.mark.parametrize("kind", ["file", "other_symlink", "dangling_symlink"])
def test_conflicting_default_cache_is_never_replaced(tmp_path, kind):
    root, target, default = initialized_cache(tmp_path)
    if kind == "file":
        default.write_bytes(b"existing")
    elif kind == "other_symlink":
        other = root / "different.cache"
        other.write_bytes(b"other allocated cache")
        default.symlink_to(other)
    else:
        default.symlink_to(root / "missing.cache")
    before = default.lstat()
    with pytest.raises((ValueError, FileNotFoundError)):
        _link_platform_cache(target, default, root, expected_uid=os.getuid())
    assert default.lstat().st_ino == before.st_ino


@pytest.mark.parametrize("kind", ["outside", "symlink_outside", "symlink_inside", "wrong_owner", "uninitialized"])
def test_invalid_platform_cache_fails_without_creating_alias(tmp_path, kind):
    root, target, default = initialized_cache(tmp_path)
    uid = os.getuid()
    if kind == "outside":
        target = tmp_path / "outside.cache"
        target.write_bytes(struct.pack("=i", 19920718))
    elif kind in {"symlink_outside", "symlink_inside"}:
        original = target
        target = root / "alias.cache"
        if kind == "symlink_outside":
            original = tmp_path / "outside.cache"
            original.write_bytes(struct.pack("=i", 19920718))
        target.symlink_to(original)
    elif kind == "wrong_owner":
        uid += 1
    else:
        target.write_bytes(struct.pack("=i", 0))
    with pytest.raises(ValueError):
        _link_platform_cache(target, default, root, expected_uid=uid)
    assert not default.exists() and not default.is_symlink()


def test_prepare_calls_only_bounded_root_script_and_parses_evidence():
    backend = Mock()
    expected = {"same_accounting_inode": True, "quota_changed": False, "hook_disabled": False}
    backend.exec_run_with_timeout.return_value = SimpleNamespace(
        exit_code=0, timed_out=False,
        output="platform diagnostics\nSFORGE_HAMI_RUNTIME=" + json.dumps(expected) + "\n")
    handle = object()
    assert prepare_hami_runtime(backend, handle) == expected
    command = backend.exec_run_with_timeout.call_args
    assert command.args[0] is handle
    assert command.args[1][:2] == ["python3", "-c"]
    assert command.kwargs == {"timeout": 420}
    compile(command.args[1][2], "remote-runtime", "exec")
    assert "preexec_fn=drop_to_agent" in command.args[1][2]
    assert "CUDA_DEVICE_MEMORY_LIMIT=" not in command.args[1][2]
    assert "LD_PRELOAD=" not in command.args[1][2]


@pytest.mark.parametrize("exit_code,timed_out,output", [
    (1, False, "sensitive runtime diagnostics"), (124, True, ""), (0, False, ""),
    (0, False, 'SFORGE_HAMI_RUNTIME={"same_accounting_inode":false}\n'),
])
def test_prepare_fails_closed_for_missing_or_invalid_evidence(exit_code, timed_out, output):
    backend = Mock()
    backend.exec_run_with_timeout.return_value = SimpleNamespace(
        exit_code=exit_code, timed_out=timed_out, output=output)
    with pytest.raises(RuntimeError) as error:
        prepare_hami_runtime(backend, object())
    assert "sensitive runtime diagnostics" not in str(error.value)
    if exit_code or timed_out:
        assert error.value.runtime_diagnostics == output[-4000:]


def test_root_probes_exit_before_filtered_agent_or_task_can_run():
    script = _prepare_source()
    assert script.index("root_before = probe") < script.index("evidence = _link_platform_cache")
    assert script.index("evidence = _link_platform_cache") < script.index("agent = probe")
    assert "subprocess.run(" in script and "timeout=120" in script
    assert "root_before != agent or root_before != root_after" in script


def test_native_temporary_layout_keeps_results_private_despite_zero_umask(tmp_path):
    import stat
    tmp_path.chmod(0o1777)
    policy = _set_owner_only_default_acl(tmp_path, expected_uid=os.getuid())
    assert policy["installed"] and policy["directory_access_unchanged"]
    saved = os.umask(0)
    try:
        run = tmp_path / "native-verifier"
        run.mkdir(mode=0o700)
        run.chmod(0o755)
        results = run / "result"
        results.mkdir(mode=0o700)
        temporary = results / ".result.tmp"
        temporary.write_text("{}")
        temporary.replace(results / "result.json")
        candidate = run / "submission.py"
        candidate.write_text("pass")
        candidate.chmod(0o444)
    finally:
        os.umask(saved)
    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o1777
    assert stat.S_IMODE(run.stat().st_mode) == 0o755
    assert stat.S_IMODE(results.stat().st_mode) == 0o700
    assert stat.S_IMODE((results / "result.json").stat().st_mode) == 0o600
    assert stat.S_IMODE(candidate.stat().st_mode) == 0o444
    assert _set_owner_only_default_acl(tmp_path, expected_uid=os.getuid())["installed"] is False


def test_conflicting_default_acl_is_preserved(tmp_path):
    tmp_path.chmod(0o1777)
    different = struct.pack("<I", 2) + b"".join(
        struct.pack("<HHI", tag, permissions, 0xFFFFFFFF)
        for tag, permissions in ((1, 7), (4, 5), (32, 5)))
    os.setxattr(tmp_path, "system.posix_acl_default", different)
    with pytest.raises(ValueError, match="conflicting"):
        _set_owner_only_default_acl(tmp_path, expected_uid=os.getuid())
    assert os.getxattr(tmp_path, "system.posix_acl_default") == different


def test_default_acl_refuses_a_symlink_or_wrong_directory_mode(tmp_path):
    tmp_path.chmod(0o755)
    with pytest.raises(ValueError, match="sticky directory"):
        _set_owner_only_default_acl(tmp_path, expected_uid=os.getuid())
    tmp_path.chmod(0o1777)
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path)
    with pytest.raises(ValueError, match="sticky directory"):
        _set_owner_only_default_acl(alias, expected_uid=os.getuid())


def test_private_tmp_remote_probe_is_bounded_and_checks_creation_not_chmod():
    import ast
    backend = Mock()
    expected = {"result_mode": "0o600", "result_uid": 1000, "directory_access_unchanged": True}
    backend.exec_run_with_timeout.return_value = SimpleNamespace(
        exit_code=0, timed_out=False, output="SFORGE_PRIVATE_TMP=" + json.dumps(expected))
    assert prepare_private_tmp(backend, object()) == expected
    call = backend.exec_run_with_timeout.call_args
    assert call.kwargs == {"timeout": 30}
    source = call.args[1][2]
    compile(source, "remote-private-tmp", "exec")
    child = next(ast.literal_eval(node.value) for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id == "child")
    compile(child, "private-result-child", "exec")
    assert "os.umask(0)" in child
    assert "chmod" not in child
    assert "st_mode)==0o600" in child


def test_loaded_remote_programs_do_not_reread_changed_source_files(monkeypatch):
    import sforge.pyromind.gpu_runtime as runtime
    original_gpu_source = runtime._prepare_source()
    monkeypatch.setattr(runtime.inspect, "getsource", Mock(side_effect=AssertionError("source changed on disk")))
    assert runtime._prepare_source() == original_gpu_source
    backend = Mock()
    backend.exec_run_with_timeout.return_value = SimpleNamespace(
        exit_code=0, timed_out=False,
        output='SFORGE_PRIVATE_TMP={"result_mode":"0o600","result_uid":1000,"directory_access_unchanged":true}')
    runtime.prepare_private_tmp(backend, object())
    source = backend.exec_run_with_timeout.call_args.args[1][2]
    assert runtime._ACL_SOURCE in source
    runtime.inspect.getsource.assert_not_called()


def test_private_tmp_failure_diagnostics_are_retained_outside_exception_text():
    backend = Mock()
    details = "sensitive infrastructure detail " * 200
    backend.exec_run_with_timeout.return_value = SimpleNamespace(
        exit_code=1, timed_out=False, output=details)
    with pytest.raises(RuntimeError) as error:
        prepare_private_tmp(backend, object())
    assert "sensitive infrastructure" not in str(error.value)
    assert error.value.runtime_diagnostics == details[-4000:]


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "2"])
def test_invalid_gpu_count_never_calls_backend(count):
    backend = Mock()
    with pytest.raises(ValueError, match="positive integer"):
        prepare_hami_runtime(backend, object(), gpu_count=count)
    backend.exec_run_with_timeout.assert_not_called()


def test_each_allocated_device_is_probed(monkeypatch, capsys):
    import sys
    queried, synchronized, computed = [], [], []
    properties = [SimpleNamespace(total_memory=90 * 1024**3, name="H100 NVL")
                  for _ in range(2)]
    def props(index):
        queried.append(index)
        return properties[index]
    def ones(count, device):
        computed.append(device)
        return SimpleNamespace(sum=lambda: SimpleNamespace(item=lambda: float(count)))
    fake = SimpleNamespace(cuda=SimpleNamespace(
        device_count=lambda: 2, get_device_properties=props,
        synchronize=synchronized.append), ones=ones)
    monkeypatch.setitem(sys.modules, "torch", fake)
    exec(_GPU_PROBE, {"GPU_COUNT": 2})
    captured = capsys.readouterr()
    payload = json.loads(captured.out.split("=", 1)[1])
    phases = [line.partition('=')[2] for line in captured.err.splitlines()]
    assert phases[:4] == ['torch_import:start', 'torch_import:done',
                          'device_count:start', 'device_count:done']
    for index in range(2):
        for action in ('properties', 'allocate', 'reduce', 'synchronize'):
            assert f'device_{index}:{action}:start' in phases
            assert f'device_{index}:{action}:done' in phases
    assert queried == synchronized == [0, 1]
    assert computed == ["cuda:0", "cuda:1"]
    assert len(payload["devices"]) == payload["gpu_count"] == 2
    properties[1].total_memory = 0
    with pytest.raises(AssertionError, match="memory must be positive"):
        exec(_GPU_PROBE, {"GPU_COUNT": 2})


def test_dual_hami_requires_matching_views_for_both_devices():
    import copy
    devices = [{"memory_bytes": 90 * 1024**3, "name": "H100", "compute": 4.0}] * 2
    view = {"gpu_count": 2, "devices": devices}
    expected = {"same_accounting_inode": True, "quota_changed": False,
                "hook_disabled": False, "root_before": copy.deepcopy(view),
                "filtered_agent": copy.deepcopy(view), "root_after": copy.deepcopy(view)}
    backend = Mock()
    def reply():
        backend.exec_run_with_timeout.return_value = SimpleNamespace(
            exit_code=0, timed_out=False, output="SFORGE_HAMI_RUNTIME=" + json.dumps(expected))
    reply()
    assert prepare_hami_runtime(backend, object(), gpu_count=2) == expected
    expected["filtered_agent"]["devices"] = devices[:1]
    reply()
    with pytest.raises(RuntimeError, match="every allocated GPU"):
        prepare_hami_runtime(backend, object(), gpu_count=2)


def distributed_evidence():
    devices = [{"rank": rank, "cuda_index": rank, "gpu_uuid": f"GPU-{rank}",
                "name": "H100 NVL", "memory_bytes": 90 * 1024**3,
                "compute_capability": [9, 0], "all_reduce": 3.0} for rank in range(2)]
    return {
        "gpu_count": 2, "min_memory_bytes": 80 * 1024**3,
        "same_accounting_inode": True, "same_device_views": True,
        "quota_changed": False, "hook_disabled": False,
        **{role: {"gpu_count": 2, "backend": "nccl", "expected_sum": 3.0,
                  "devices": [dict(device, uid=uid) for device in devices]}
           for role, uid in (("filtered_root", 0), ("filtered_agent", 1000))},
    }


def distributed_backend(evidence):
    backend = Mock()
    backend.exec_run_with_timeout.return_value = SimpleNamespace(
        exit_code=0, timed_out=False,
        output="SFORGE_DISTRIBUTED_RUNTIME=" + json.dumps(evidence))
    return backend


def test_distributed_qualification_validates_both_roles_and_bounded_children():
    import ast
    expected = distributed_evidence()
    backend = distributed_backend(expected)
    handle = object()
    assert verify_distributed_runtime(backend, handle, gpu_count=2,
                                      min_memory_bytes=80 * 1024**3) == expected
    call = backend.exec_run_with_timeout.call_args
    assert call.args[0] is handle
    assert call.kwargs == {"timeout": 300}
    script = call.args[1][2]
    compile(script, "remote-distributed", "exec")
    child = next(ast.literal_eval(node.value) for node in ast.walk(ast.parse(script))
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id == "PROBE_SOURCE")
    compile(child, "distributed-ranks", "exec")
    assert "communicate(timeout=120)" in script
    assert "os.killpg(process.pid,signal.SIGKILL)" in script
    assert "preexec_fn=drop" in script
    assert "dist.all_reduce(value)" in child
    assert "LD_PRELOAD=" not in script and "NCCL_P2P_DISABLE" not in script
    assert "CUDA_DEVICE_MEMORY_LIMIT=" not in script


@pytest.mark.parametrize("defect", ["same_gpu", "insufficient_memory", "wrong_sum",
                                    "wrong_uid", "wrong_count", "root_agent_differ",
                                    "hook_bypass", "cache_change", "old_architecture"])
def test_distributed_contract_failure_is_not_accepted(defect):
    evidence = distributed_evidence()
    device = evidence["filtered_agent"]["devices"][1]
    if defect == "same_gpu": device["gpu_uuid"] = "GPU-0"
    elif defect == "insufficient_memory": device["memory_bytes"] = 79 * 1024**3
    elif defect == "wrong_sum": device["all_reduce"] = 2.0
    elif defect == "wrong_uid": device["uid"] = 0
    elif defect == "wrong_count": evidence["filtered_agent"]["devices"].pop()
    elif defect == "root_agent_differ": device["gpu_uuid"] = "GPU-different-allocation"
    elif defect == "hook_bypass": evidence["hook_disabled"] = True
    elif defect == "cache_change": evidence["same_accounting_inode"] = False
    elif defect == "old_architecture": device["compute_capability"] = [7, 5]
    with pytest.raises(RuntimeError, match="GPU contract"):
        verify_distributed_runtime(distributed_backend(evidence), object(),
                                   gpu_count=2, min_memory_bytes=80 * 1024**3)


@pytest.mark.parametrize("gpu_count,memory", [(1, 1), (2, 0), (2, True), (2, "80Gi")])
def test_distributed_invalid_inputs_do_not_execute(gpu_count, memory):
    backend = Mock()
    with pytest.raises(ValueError):
        verify_distributed_runtime(backend, object(), gpu_count=gpu_count,
                                   min_memory_bytes=memory)
    backend.exec_run_with_timeout.assert_not_called()


def test_distributed_failures_keep_diagnostics_out_of_error_text():
    backend = Mock()
    backend.exec_run_with_timeout.return_value = SimpleNamespace(
        exit_code=124, timed_out=True, output="private runtime traceback")
    with pytest.raises(RuntimeError, match="qualification failed") as error:
        verify_distributed_runtime(backend, object(), gpu_count=2,
                                   min_memory_bytes=80 * 1024**3)
    assert "private" not in str(error.value)
    assert error.value.runtime_diagnostics == "private runtime traceback"


@pytest.mark.parametrize('phase', ['root_before', 'filtered_agent', 'root_after'])
@pytest.mark.parametrize('binary', [False, True])
def test_cuda_probe_timeout_retains_phase_and_partial_output(monkeypatch, phase, binary):
    import subprocess
    stdout = 'partial allocation diagnostic'
    stderr = 'SFORGE_GPU_PHASE=torch_import:done\nSFORGE_GPU_PHASE=device_0:properties:start\n'
    if binary:
        stdout, stderr = stdout.encode(), stderr.encode()
    failure = subprocess.TimeoutExpired(['must-not-echo-command'], 120,
                                        output=stdout, stderr=stderr)
    runner = Mock(side_effect=failure)
    monkeypatch.setattr(subprocess, 'run', runner)
    with pytest.raises(RuntimeError, match='timed out after 120s') as error:
        _run_gpu_probe('probe-only', {'SECRET_ENV': 'must-not-echo-env'}, '/tmp', phase)
    message = str(error.value)
    details = json.loads(message.partition(': ')[2])
    assert details['phase'] == phase
    assert details['last_progress'] == 'device_0:properties:start'
    assert details['stdout_tail'] == 'partial allocation diagnostic'
    assert 'must-not-echo' not in message
    assert runner.call_args.kwargs['timeout'] == 120


def test_failed_cuda_probe_preserves_last_phase_and_bounds_diagnostics(monkeypatch):
    import subprocess
    stderr = ('x' * 10000 + '\nSFORGE_GPU_PHASE=device_1:allocate:start\n'
              + 'allocation failed\n')
    monkeypatch.setattr(subprocess, 'run', Mock(return_value=SimpleNamespace(
        returncode=1, stdout='y' * 10000, stderr=stderr)))
    with pytest.raises(RuntimeError, match='GPU runtime probe failed') as error:
        _run_gpu_probe('probe', {}, '/tmp', 'filtered_agent')
    details = json.loads(str(error.value).partition(': ')[2])
    assert details['last_progress'] == 'device_1:allocate:start'
    assert len(details['stdout_tail']) == 1000
    assert len(details['stderr_tail']) == 2200
    assert details['stderr_tail'].endswith('allocation failed\n')


def test_successful_cuda_probe_keeps_progress_separate_from_evidence(monkeypatch):
    import subprocess
    drop = Mock()
    runner = Mock(return_value=SimpleNamespace(
        returncode=0, stdout='SFORGE_GPU_PROBE={"memory_bytes":42}\n',
        stderr='SFORGE_GPU_PHASE=device_0:synchronize:done\n'))
    monkeypatch.setattr(subprocess, 'run', runner)
    assert _run_gpu_probe('probe', {}, '/tmp', 'filtered_agent', drop) == {'memory_bytes': 42}
    assert runner.call_args.kwargs['preexec_fn'] is drop


def test_topology_capture_records_pcie_without_inventing_an_nvlink_requirement():
    backend = Mock()
    topology = 'GPU0 GPU1\nGPU0 X PXB\nGPU1 PXB X\n'
    inventory = '0, GPU-a, NVIDIA H100 NVL, 00000000:49:00.0, 95830\n'
    backend.exec_run_with_timeout.side_effect = [
        SimpleNamespace(exit_code=0, timed_out=False, output=topology),
        SimpleNamespace(exit_code=0, timed_out=False, output=inventory),
    ]
    handle = SimpleNamespace(id='sb-topology')
    evidence = capture_gpu_topology(backend, handle)
    assert evidence['status'] == 'available'
    assert evidence['sandbox_id'] == handle.id
    assert evidence['queries']['topology']['output'] == topology
    assert evidence['queries']['gpu_inventory']['output'] == inventory
    assert all(call.kwargs == {'timeout': 20}
               for call in backend.exec_run_with_timeout.call_args_list)
    assert backend.exec_run_with_timeout.call_args_list[0].args[1] == ['nvidia-smi', 'topo', '-m']
    assert 'nvlink_required' not in evidence and 'valid' not in evidence


def test_topology_capture_preserves_failure_without_losing_other_query():
    backend = Mock()
    backend.exec_run_with_timeout.side_effect = [
        SimpleNamespace(exit_code=124, timed_out=True, output='NVML timed out'),
        SimpleNamespace(exit_code=0, timed_out=False, output='GPU inventory'),
    ]
    evidence = capture_gpu_topology(backend, SimpleNamespace(id='sb-topology'))
    assert evidence['status'] == 'unavailable'
    assert evidence['queries']['topology']['exit_code'] == 124
    assert evidence['queries']['topology']['timed_out'] is True
    assert evidence['queries']['topology']['output'] == 'NVML timed out'
    assert evidence['queries']['gpu_inventory']['available'] is True


def test_topology_transport_error_does_not_serialize_sdk_secrets():
    backend = Mock()
    backend.exec_run_with_timeout.side_effect = [
        RuntimeError('request headers contain secret credential'),
        SimpleNamespace(exit_code=0, timed_out=False, output=''),
    ]
    evidence = capture_gpu_topology(backend, SimpleNamespace(id='sb-topology'))
    assert evidence['status'] == 'unavailable'
    assert evidence['queries']['topology']['error_type'] == 'RuntimeError'
    assert 'secret credential' not in json.dumps(evidence)
    assert evidence['queries']['gpu_inventory']['available'] is False
