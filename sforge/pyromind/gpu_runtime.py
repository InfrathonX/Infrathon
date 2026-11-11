"""Keep HAMi's allocated GPU cache visible to environment-filtered subprocesses.

The native verifier deliberately filters candidate environment variables. HAMi
then uses /tmp/cudevshr.cache rather than its injected per-container cache. This
adapter aliases that default to the existing platform cache; it never changes
GPU limits, copies accounting state, disables the hook, or changes task code.
"""
from __future__ import annotations

import inspect
import errno
import json
import os
from pathlib import Path
import stat
import struct


def _link_platform_cache(target, default, cache_root, expected_uid=0):
    target, default, cache_root = Path(target), Path(default), Path(cache_root)
    if (not target.is_absolute() or ".." in target.parts
            or not target.is_relative_to(cache_root)):
        raise ValueError("HAMi cache must be inside the platform cache directory")
    resolved = target.resolve(strict=True)
    if not resolved.is_relative_to(cache_root.resolve(strict=True)):
        raise ValueError("HAMi cache resolves outside the platform cache directory")
    metadata = target.lstat()
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != expected_uid
            or metadata.st_size < 4):
        raise ValueError("HAMi platform cache must be an initialized root-owned regular file")
    # HAMi's shared-region magic is its first native-endian 32-bit field.
    with target.open("rb") as stream:
        initialized = struct.unpack("=i", stream.read(4))[0]
    if initialized != 19920718:
        raise ValueError("HAMi platform cache is not initialized")
    identity = (metadata.st_dev, metadata.st_ino)
    created = False
    if default.exists() or default.is_symlink():
        if not default.is_symlink():
            raise ValueError("Refusing a conflicting HAMi default cache")
        linked = default.stat()
        if (linked.st_dev, linked.st_ino) != identity:
            raise ValueError("Refusing a conflicting HAMi default cache")
    else:
        default.symlink_to(target)
        created = True
    linked = default.stat()
    if (linked.st_dev, linked.st_ino) != identity:
        raise RuntimeError("HAMi default cache does not reference the allocated cache")
    return {"platform_cache": str(target), "default_cache": str(default),
            "cache_device": metadata.st_dev, "cache_inode": metadata.st_ino,
            "cache_bytes": metadata.st_size, "cache_uid": metadata.st_uid,
            "symlink_created": created, "same_accounting_inode": True}


_GPU_PROBE = r'''import json,sys
def progress(phase):
    print('SFORGE_GPU_PHASE='+phase,file=sys.stderr,flush=True)
progress('torch_import:start')
import torch
progress('torch_import:done')
progress('device_count:start')
assert torch.cuda.device_count() == GPU_COUNT, 'allocated GPU count differs'
progress('device_count:done')
devices = []
for index in range(GPU_COUNT):
    progress('device_'+str(index)+':properties:start')
    p = torch.cuda.get_device_properties(index)
    assert p.total_memory > 0, 'GPU memory must be positive'
    progress('device_'+str(index)+':properties:done')
    progress('device_'+str(index)+':allocate:start')
    x = torch.ones(4, device='cuda:'+str(index))
    progress('device_'+str(index)+':allocate:done')
    progress('device_'+str(index)+':reduce:start')
    value = x.sum().item()
    progress('device_'+str(index)+':reduce:done')
    progress('device_'+str(index)+':synchronize:start')
    torch.cuda.synchronize(index)
    progress('device_'+str(index)+':synchronize:done')
    assert value == 4.0
    devices.append({'memory_bytes':p.total_memory,'name':p.name,'compute':value})
# Keep the original single-GPU record shape for existing qualification reports.
evidence = devices[0] if GPU_COUNT == 1 else {'gpu_count':GPU_COUNT,'devices':devices}
print('SFORGE_GPU_PROBE=' + json.dumps(evidence))
'''


def _run_gpu_probe(source, environment, cwd, phase, drop_to_agent=None):
    """Capture bounded phase diagnostics without dumping the command or environment."""
    import json
    import subprocess
    import sys

    def diagnostic(stdout, stderr):
        def decode(value):
            return value.decode('utf-8', errors='replace') if isinstance(value, bytes) else (value or '')
        stdout, stderr = decode(stdout), decode(stderr)
        progress = [line.partition('=')[2] for line in stderr.splitlines()
                    if line.startswith('SFORGE_GPU_PHASE=')]
        return json.dumps({'phase': phase, 'last_progress': progress[-1] if progress else None,
                           'stdout_tail': stdout[-1000:], 'stderr_tail': stderr[-2200:]})

    try:
        result = subprocess.run(
            [sys.executable, '-c', source], env=environment, cwd=cwd,
            preexec_fn=drop_to_agent, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=120)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('GPU runtime probe timed out after 120s: '
                           + diagnostic(exc.stdout, exc.stderr)) from None
    if result.returncode != 0:
        raise RuntimeError('GPU runtime probe failed: '
                           + diagnostic(result.stdout, result.stderr))
    records = [line.partition('=')[2] for line in result.stdout.splitlines()
               if line.startswith('SFORGE_GPU_PROBE=')]
    if len(records) != 1:
        raise RuntimeError('GPU probe produced no unique evidence: '
                           + diagnostic(result.stdout, result.stderr))
    try:
        evidence = json.loads(records[0])
        if not isinstance(evidence, dict):
            raise ValueError('GPU evidence must be an object')
    except (TypeError, ValueError):
        raise RuntimeError('GPU probe produced invalid evidence: '
                           + diagnostic(result.stdout, result.stderr)) from None
    return evidence


def _validate_gpu_count(gpu_count):
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count < 1:
        raise ValueError("GPU count must be a positive integer")
    return gpu_count


def _prepare_source(gpu_count=1):
    gpu_count = _validate_gpu_count(gpu_count)
    probe_source = f"GPU_COUNT={gpu_count}\n" + _GPU_PROBE
    return ("import json,os,pathlib,pwd,stat,struct,subprocess,sys,tempfile\n"
            "from pathlib import Path\n"
            + _LINK_SOURCE
            + "\n" + _RUN_PROBE_SOURCE
            + "\nGPU_PROBE=" + repr(probe_source) + "\n" + r'''
assert os.geteuid() == 0, 'HAMi preparation requires root'
account = pwd.getpwnam('agent')
assert account.pw_uid == 1000, 'expected unprivileged agent UID 1000'
cache = os.environ.get('CUDA_DEVICE_MEMORY_SHARED_CACHE')
if not cache:
    raise RuntimeError('Platform did not provide a HAMi allocation cache')
allocation = {k:v for k,v in os.environ.items()
              if k.startswith(('CUDA_DEVICE_MEMORY_LIMIT','CUDA_DEVICE_SM_LIMIT'))
              or k == 'CUDA_DEVICE_MEMORY_SHARED_CACHE'}
def drop_to_agent():
    os.setgroups([]); os.setgid(account.pw_gid); os.setuid(account.pw_uid); os.umask(0o077)
def probe(env, cwd, phase, unprivileged=False):
    return _run_gpu_probe(GPU_PROBE,env,cwd,phase,
                          drop_to_agent=drop_to_agent if unprivileged else None)
# Initialize using the actual platform-issued environment before touching the
# default path. No filtered process has used CUDA in this fresh Judge yet.
root_before = probe(dict(os.environ), '/tmp', 'root_before')
evidence = _link_platform_cache(cache, '/tmp/cudevshr.cache', '/usr/local/vgpu')
with tempfile.TemporaryDirectory(prefix='sforge-gpu-preflight-') as temp:
    work = Path(temp); work.chmod(0o755)
    cache_dir = work/'candidate_cache'; cache_dir.mkdir(mode=0o700)
    triton = cache_dir/'triton'; triton.mkdir(mode=0o700)
    for path in (cache_dir,triton): os.chown(path,account.pw_uid,account.pw_gid)
    allowed = ('CUDA_HOME','CUDA_PATH','CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH',
               'NVIDIA_DRIVER_CAPABILITIES','NVIDIA_VISIBLE_DEVICES','PATH')
    filtered = {key:os.environ[key] for key in allowed if key in os.environ}
    filtered.update(HOME='/home/agent',LANG='C.UTF-8',LC_ALL='C.UTF-8',
                    PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
                    TMPDIR=str(cache_dir),TRITON_CACHE_DIR=str(triton))
    agent = probe(filtered, str(work), 'filtered_agent', unprivileged=True)
root_after = probe(dict(os.environ), '/tmp', 'root_after')
verified = _link_platform_cache(cache, '/tmp/cudevshr.cache', '/usr/local/vgpu')
if (evidence['cache_device'], evidence['cache_inode']) != (verified['cache_device'], verified['cache_inode']):
    raise RuntimeError('Platform GPU cache identity changed during preparation')
if root_before != agent or root_before != root_after:
    raise RuntimeError('Filtered GPU view differs from the allocated platform quota')
current = {k:v for k,v in os.environ.items()
           if k.startswith(('CUDA_DEVICE_MEMORY_LIMIT','CUDA_DEVICE_SM_LIMIT'))
           or k == 'CUDA_DEVICE_MEMORY_SHARED_CACHE'}
if current != allocation: raise RuntimeError('Platform GPU configuration changed')
evidence.update(schema_version=1, root_before=root_before, filtered_agent=agent,
                root_after=root_after, platform_allocation=allocation,
                filtered_uid=account.pw_uid, task_code_modified=False,
                quota_changed=False, hook_disabled=False)
print('SFORGE_HAMI_RUNTIME=' + json.dumps(evidence,sort_keys=True))
''')


def prepare_hami_runtime(backend, handle, *, gpu_count=1):
    """Configure a fresh GPU Sandbox and return verified deployment evidence."""
    result = backend.exec_run_with_timeout(
        handle, ["python3", "-c", _prepare_source(gpu_count)], timeout=420)
    if result.exit_code or result.timed_out:
        # Only GPU/runtime probes run here; retain their traceback for the local
        # controller's infrastructure log (never candidate-facing feedback).
        error = RuntimeError("HAMi allocation-preserving runtime preparation failed")
        error.runtime_diagnostics = result.output[-4000:]
        raise error
    records = [line.partition("=")[2] for line in result.output.splitlines()
               if line.startswith("SFORGE_HAMI_RUNTIME=")]
    if len(records) != 1:
        raise RuntimeError("HAMi preparation produced no unique evidence")
    evidence = json.loads(records[0])
    if (evidence.get("same_accounting_inode") is not True
            or evidence.get("quota_changed") is not False
            or evidence.get("hook_disabled") is not False):
        raise RuntimeError("HAMi preparation did not preserve platform allocation")
    if gpu_count != 1:
        views = [evidence.get(name) for name in ("root_before", "filtered_agent", "root_after")]
        if (any(not isinstance(view, dict) or view.get("gpu_count") != gpu_count
                or len(view.get("devices", [])) != gpu_count for view in views)
                or views[0] != views[1] or views[0] != views[2]):
            raise RuntimeError("HAMi preparation did not verify every allocated GPU")
    return evidence


_DISTRIBUTED_PROBE = r'''import datetime,json,os,torch
import torch.distributed as dist
rank = int(os.environ['LOCAL_RANK'])
assert int(os.environ['WORLD_SIZE']) == GPU_COUNT
assert torch.cuda.device_count() == GPU_COUNT, 'allocated GPU count differs'
torch.cuda.set_device(rank)
p = torch.cuda.get_device_properties(rank)
assert p.total_memory >= MIN_MEMORY_BYTES, 'GPU is below the task memory floor'
assert (p.major,p.minor) >= (8,0), 'GPU is below the BF16 task capability floor'
assert torch.cuda.is_bf16_supported(), 'GPU does not support BF16'
gpu_uuid = str(getattr(p,'uuid',''))
assert gpu_uuid, 'CUDA did not expose a physical GPU UUID'
dist.init_process_group('nccl',timeout=datetime.timedelta(seconds=60))
try:
    value = torch.tensor([float(rank+1)],device='cuda',dtype=torch.float32)
    dist.all_reduce(value)
    torch.cuda.synchronize(rank)
    expected = GPU_COUNT*(GPU_COUNT+1)/2
    assert value.item() == expected, 'NCCL all-reduce produced an incorrect result'
    record = {'rank':rank,'cuda_index':rank,'gpu_uuid':gpu_uuid,'uid':os.geteuid(),
              'name':p.name,'memory_bytes':p.total_memory,
              'compute_capability':[p.major,p.minor],'all_reduce':value.item()}
    records = [None for _ in range(GPU_COUNT)]
    dist.all_gather_object(records,record)
    assert len({item['gpu_uuid'] for item in records}) == GPU_COUNT, 'ranks share a physical GPU'
    if rank == 0:
        print('SFORGE_NCCL_PROBE='+json.dumps({'gpu_count':GPU_COUNT,'backend':'nccl',
                                            'expected_sum':expected,'devices':records}),flush=True)
finally:
    dist.destroy_process_group()
'''


def _distributed_source(gpu_count, min_memory_bytes):
    gpu_count = _validate_gpu_count(gpu_count)
    if gpu_count < 2:
        raise ValueError("Distributed qualification needs at least two GPUs")
    if (isinstance(min_memory_bytes, bool) or not isinstance(min_memory_bytes, int)
            or min_memory_bytes <= 0):
        raise ValueError("GPU memory floor must be a positive integer")
    child = f"GPU_COUNT={gpu_count}\nMIN_MEMORY_BYTES={min_memory_bytes}\n" + _DISTRIBUTED_PROBE
    return ("import json,os,pwd,signal,stat,struct,subprocess,sys,tempfile\n"
            "from pathlib import Path\n" + _LINK_SOURCE
            + f"\nGPU_COUNT={gpu_count}\nMIN_MEMORY_BYTES={min_memory_bytes}\n"
            + "PROBE_SOURCE=" + repr(child) + "\n" + r'''
assert os.geteuid() == 0, 'Distributed qualification requires root'
account = pwd.getpwnam('agent')
assert account.pw_uid == 1000, 'expected unprivileged agent UID 1000'
cache = os.environ.get('CUDA_DEVICE_MEMORY_SHARED_CACHE')
assert cache, 'Platform did not provide a HAMi allocation cache'
# prepare_hami_runtime must have initialized and aliased the original allocation.
default = Path('/tmp/cudevshr.cache')
assert default.is_symlink(), 'Run HAMi preparation before distributed qualification'
before = _link_platform_cache(cache, default, '/usr/local/vgpu')
allocation = {k:v for k,v in os.environ.items()
              if k.startswith(('CUDA_DEVICE_MEMORY_LIMIT','CUDA_DEVICE_SM_LIMIT'))
              or k == 'CUDA_DEVICE_MEMORY_SHARED_CACHE'}
def drop():
    os.setgroups([]); os.setgid(account.pw_gid); os.setuid(account.pw_uid); os.umask(0o077)
def cohort(root, script, role):
    scratch = root/role; scratch.mkdir(mode=0o700)
    if role == 'filtered_agent': os.chown(scratch,account.pw_uid,account.pw_gid)
    # Match the native GKD verifier's minimal environment in BOTH roles.
    allowed = ('LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES')
    env = {key:os.environ[key] for key in allowed if key in os.environ}
    env.update(PATH=str(Path(sys.executable).parent)+':/usr/local/bin:/usr/bin:/bin',
               HOME=str(scratch),TMPDIR=str(scratch),LANG='C.UTF-8',LC_ALL='C.UTF-8',
               PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
               HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false',TRITON_CACHE_DIR=str(scratch/'triton'),
               TORCH_NCCL_ASYNC_ERROR_HANDLING='1',TORCH_NCCL_BLOCKING_WAIT='1')
    command = [sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',
               '--nproc-per-node='+str(GPU_COUNT),str(script)]
    process = subprocess.Popen(command,env=env,cwd=str(root),stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
                               start_new_session=True,close_fds=True,
                               preexec_fn=drop if role == 'filtered_agent' else None)
    try:
        stdout,stderr = process.communicate(timeout=120)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid,signal.SIGTERM)
            process.communicate(timeout=5)
        except (ProcessLookupError,subprocess.TimeoutExpired):
            try: os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            process.communicate()
        raise RuntimeError('Distributed '+role+' probe timed out')
    if process.returncode:
        raise RuntimeError('Distributed '+role+' probe failed: '+stderr[-4000:])
    matches = [line.partition('=')[2] for line in stdout.splitlines()
               if line.startswith('SFORGE_NCCL_PROBE=')]
    if len(matches) != 1: raise RuntimeError('Distributed probe produced no unique evidence')
    result = json.loads(matches[0])
    uid = account.pw_uid if role == 'filtered_agent' else 0
    assert all(device['uid'] == uid for device in result['devices'])
    return result
with tempfile.TemporaryDirectory(prefix='sforge-nccl-preflight-') as temporary:
    root = Path(temporary); root.chmod(0o755)
    script = root/'probe.py'; script.write_text(PROBE_SOURCE); script.chmod(0o444)
    reference = cohort(root,script,'filtered_root')
    candidate = cohort(root,script,'filtered_agent')
def device_view(result):
    return [{key:value for key,value in device.items() if key != 'uid'}
            for device in result['devices']]
assert device_view(reference) == device_view(candidate), 'Filtered rank GPU views differ'
after = _link_platform_cache(cache, default, '/usr/local/vgpu')
assert (before['cache_device'],before['cache_inode']) == (after['cache_device'],after['cache_inode'])
current = {k:v for k,v in os.environ.items()
           if k.startswith(('CUDA_DEVICE_MEMORY_LIMIT','CUDA_DEVICE_SM_LIMIT'))
           or k == 'CUDA_DEVICE_MEMORY_SHARED_CACHE'}
assert current == allocation, 'Platform allocation changed during qualification'
evidence = {'schema_version':1,'gpu_count':GPU_COUNT,'min_memory_bytes':MIN_MEMORY_BYTES,
            'filtered_root':reference,'filtered_agent':candidate,
            'same_accounting_inode':True,'same_device_views':True,
            'quota_changed':False,'hook_disabled':False,'task_code_modified':False}
print('SFORGE_DISTRIBUTED_RUNTIME='+json.dumps(evidence,sort_keys=True))
''')


def verify_distributed_runtime(backend, handle, *, gpu_count, min_memory_bytes):
    """Qualify NCCL and the memory floor as root and UID1000 on the same allocation.

    Call after prepare_hami_runtime. This launches only bounded lightweight
    collectives, using the task's filtered environment without overriding
    platform hooks or NCCL transport selection.
    """
    source = _distributed_source(gpu_count, min_memory_bytes)
    result = backend.exec_run_with_timeout(handle, ["python3", "-c", source], timeout=300)
    if result.exit_code or result.timed_out:
        error = RuntimeError("Distributed GPU runtime qualification failed")
        error.runtime_diagnostics = result.output[-6000:]
        raise error
    records = [line.partition("=")[2] for line in result.output.splitlines()
               if line.startswith("SFORGE_DISTRIBUTED_RUNTIME=")]
    if len(records) != 1:
        raise RuntimeError("Distributed qualification produced no unique evidence")
    try:
        evidence = json.loads(records[0])
        assert evidence["gpu_count"] == gpu_count
        assert evidence["min_memory_bytes"] == min_memory_bytes
        assert evidence["same_accounting_inode"] is True
        assert evidence["same_device_views"] is True
        assert evidence["quota_changed"] is False and evidence["hook_disabled"] is False
        expected = gpu_count * (gpu_count + 1) / 2
        for role, uid in (("filtered_root", 0), ("filtered_agent", 1000)):
            view = evidence[role]
            assert view["backend"] == "nccl" and view["gpu_count"] == gpu_count
            assert view["expected_sum"] == expected
            devices = view["devices"]
            assert len(devices) == gpu_count
            assert {device["rank"] for device in devices} == set(range(gpu_count))
            assert len({device["gpu_uuid"] for device in devices}) == gpu_count
            for device in devices:
                assert device["uid"] == uid and device["gpu_uuid"]
                assert device["cuda_index"] == device["rank"]
                assert device["memory_bytes"] >= min_memory_bytes
                assert tuple(device["compute_capability"]) >= (8, 0)
                assert device["all_reduce"] == expected
        root_view = [{key:value for key,value in device.items() if key != "uid"}
                     for device in evidence["filtered_root"]["devices"]]
        agent_view = [{key:value for key,value in device.items() if key != "uid"}
                      for device in evidence["filtered_agent"]["devices"]]
        assert root_view == agent_view
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Distributed qualification did not verify the GPU contract") from exc
    return evidence


def capture_gpu_topology(backend, handle):
    """Record the allocated devices and topology without imposing a transport policy.

    Diagnostic unavailability is explicit evidence, not permission to bypass
    any task admission check. Callers decide whether their deployment requires
    this provenance; this helper never changes GPU or NCCL configuration.
    """
    commands = {
        'topology': ['nvidia-smi', 'topo', '-m'],
        'gpu_inventory': ['nvidia-smi', '--query-gpu=index,uuid,name,pci.bus_id,memory.total',
                          '--format=csv,noheader,nounits'],
    }
    queries = {}
    for name, command in commands.items():
        try:
            result = backend.exec_run_with_timeout(handle, command, timeout=20)
            output = result.output or ''
            queries[name] = {
                'command': command, 'exit_code': result.exit_code,
                'timed_out': result.timed_out, 'output': output[-32000:],
                'output_truncated': len(output) > 32000,
                'available': result.exit_code == 0 and not result.timed_out and bool(output.strip()),
            }
        except Exception as exc:
            # Transport errors may contain credential-bearing SDK requests.
            # Keep the exception class rather than serializing those messages.
            queries[name] = {'command': command, 'available': False, 'exit_code': None,
                             'timed_out': isinstance(exc, TimeoutError),
                             'error_type': type(exc).__name__, 'output': '',
                             'output_truncated': False}
    available = all(query['available'] for query in queries.values())
    status = ('partial' if any(query['output_truncated'] for query in queries.values())
              else 'available') if available else 'unavailable'
    return {'schema_version': 1, 'sandbox_id': handle.id, 'status': status,
            'purpose': 'measurement_profile_provenance', 'queries': queries}


def _set_owner_only_default_acl(directory, expected_uid=0, expected_mode=0o1777):
    directory = Path(directory)
    before = directory.lstat()
    if (not stat.S_ISDIR(before.st_mode) or before.st_uid != expected_uid
            or stat.S_IMODE(before.st_mode) != expected_mode):
        raise ValueError("Private temporary ACL requires the original root-owned sticky directory")
    # Linux POSIX ACL xattr v2: user_obj=rwx, group_obj=---, other=---.
    # A minimal ACL needs no mask entry. IDs are undefined for these three tags.
    policy = struct.pack("<I", 2) + b"".join(
        struct.pack("<HHI", tag, permissions, 0xFFFFFFFF)
        for tag, permissions in ((0x01, 7), (0x04, 0), (0x20, 0)))
    key = "system.posix_acl_default"
    try:
        current = os.getxattr(directory, key)
    except OSError as exc:
        if exc.errno != errno.ENODATA:
            raise
        current = None
    if current is not None and current != policy:
        raise ValueError("Refusing a conflicting temporary-directory default ACL")
    created = current is None
    if created:
        # Do not overwrite a policy installed concurrently.
        os.setxattr(directory, key, policy, flags=os.XATTR_CREATE)
    after = directory.lstat()
    if ((before.st_dev, before.st_ino, before.st_uid, before.st_mode)
            != (after.st_dev, after.st_ino, after.st_uid, after.st_mode)
            or os.getxattr(directory, key) != policy):
        raise RuntimeError("Temporary directory identity or access permissions changed")
    return {"directory": str(directory), "directory_mode": oct(expected_mode),
            "default_acl": "user::rwx,group::---,other::---", "installed": created,
            "directory_access_unchanged": True}


def prepare_private_tmp(backend, handle):
    """Apply an explicit Judge OS permission policy without changing the verifier.

    HAMi initialization changes the process umask to zero. A default ACL keeps
    newly created private result files owner-only even with that umask, while
    the native verifier's explicit chmod calls still control traversable paths
    and public candidate sources. Call only in the fresh dedicated Judge.
    """
    source = ("import errno,json,os,pwd,stat,struct,subprocess,sys,tempfile\n"
              "from pathlib import Path\n" + _ACL_SOURCE + r'''
assert os.geteuid() == 0
account = pwd.getpwnam('agent')
assert account.pw_uid == 1000
evidence = _set_owner_only_default_acl('/tmp')
def drop():
    os.setgroups([]); os.setgid(account.pw_gid); os.setuid(account.pw_uid)
with tempfile.TemporaryDirectory(prefix='sforge-private-mode-probe-') as temp:
    root = Path(temp); root.chmod(0o755)
    result = root/'result'; result.mkdir(mode=0o700)
    os.chown(result,account.pw_uid,account.pw_gid)
    public = root/'submission.py'; public.write_text('pass\n'); public.chmod(0o444)
    child = """import json,os,pathlib,stat,sys
os.umask(0)
root=pathlib.Path(sys.argv[1]); assert (root/'submission.py').read_text()=='pass\\n'
tmp=root/'result/.temporary';tmp.write_text('{}');tmp.replace(root/'result/result.json')
s=(root/'result/result.json').stat()
assert stat.S_IMODE(s.st_mode)==0o600 and s.st_uid==1000 and s.st_nlink==1
print(json.dumps({'result_mode':oct(stat.S_IMODE(s.st_mode)),'result_uid':s.st_uid,'result_links':s.st_nlink,'probe_umask':'0o0'}))
"""
    completed = subprocess.run([sys.executable,'-c',child,str(root)],preexec_fn=drop,
                               stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=15)
    if completed.returncode:
        raise RuntimeError('Private file creation under zero umask failed: '+completed.stderr[-2000:])
    evidence.update(json.loads(completed.stdout))
evidence.update(schema_version=1, task_code_modified=False, result_gate_modified=False)
print('SFORGE_PRIVATE_TMP=' + json.dumps(evidence,sort_keys=True))
''')
    result = backend.exec_run_with_timeout(handle, ["python3", "-c", source], timeout=30)
    if result.exit_code or result.timed_out:
        error = RuntimeError("Judge private temporary-file permission preparation failed")
        error.runtime_diagnostics = result.output[-4000:]
        raise error
    records = [line.partition("=")[2] for line in result.output.splitlines()
               if line.startswith("SFORGE_PRIVATE_TMP=")]
    if len(records) != 1:
        raise RuntimeError("Private temporary policy produced no unique evidence")
    evidence = json.loads(records[0])
    if (evidence.get("result_mode") != "0o600" or evidence.get("result_uid") != 1000
            or evidence.get("directory_access_unchanged") is not True):
        raise RuntimeError("Judge private temporary policy did not enforce file ownership and mode")
    return evidence


# Capture once at import: long-running controllers must not regenerate remote
# programs from source-file line numbers after the file has been edited.
_LINK_SOURCE = inspect.getsource(_link_platform_cache)
_ACL_SOURCE = inspect.getsource(_set_owner_only_default_acl)
_RUN_PROBE_SOURCE = inspect.getsource(_run_gpu_probe)
