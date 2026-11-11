"""Opt-in host-to-root runtime receipts; never include these bytes in feedback."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile

from sforge.harness import minference_runtime_binding_v3 as runtime_binding

PROTOCOL = "minference-ruler-quality-v3"
ROOT = PurePosixPath("/opt/minference_prefill_judge/private")
CACHE_PATH_KEY = "MINFERENCE_DENSE_CACHE_PATH"
CACHE_SHA_KEY = "MINFERENCE_DENSE_CACHE_SHA256"
MAX_CACHE_BYTES = 64 * 1024**2
SHA = re.compile(r"[0-9a-f]{64}\Z")
IMAGE = re.compile(r"sha256:[0-9a-f]{64}\Z")
GPU = re.compile(r"GPU-[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\Z")

# All values come from backend inspection and the host's private cache config.
# No candidate archive member or container environment provides these values.
INSTALL = r'''
import hashlib,importlib.util,json,os,stat,sys
from pathlib import Path
from types import SimpleNamespace
assert os.geteuid()==0
p=json.loads(Path(sys.argv[1]).read_bytes())
root=Path("/opt/minference_prefill_judge/private")
assert root.is_dir() and not root.is_symlink()
def checked(path):
    s=path.lstat()
    assert stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_uid==0 and stat.S_IMODE(s.st_mode)==0o400
    return path.read_bytes()
def sha(path):return hashlib.sha256(checked(path)).hexdigest()
def write(path,raw):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
    with os.fdopen(fd,"wb") as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())
    os.chown(path,0,0);os.chmod(path,0o400)
manifest_sha=sha(root/"final_manifest.json")
result={"FINAL_MANIFEST_SHA256":manifest_sha}
if p["phase"]=="final":
    raw=(json.dumps([p["inspect"]],sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    inspect=root/"container-inspect.json";write(inspect,raw)
    policy=json.loads(checked(root/"quality-policy.json"))
    assert policy.get("status")=="frozen"
    canonical=json.dumps(policy,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
    cache=root/"dense-quality.json"
    if p["dense_cache_sha256"] is None:
        assert not cache.exists() and not cache.is_symlink(),"existing dense cache lacks a declared SHA"
    else:
        os.chown(cache,0,0);os.chmod(cache,0o400)
        assert sha(cache)==p["dense_cache_sha256"],"trusted dense cache differs from its exact SHA"
    helper=root/"quad-runtime-binding.py"
    assert sha(helper)==p["binding_helper_sha256"]
    spec=importlib.util.spec_from_file_location("trusted_quad_runtime_binding",helper)
    binding=importlib.util.module_from_spec(spec);spec.loader.exec_module(binding)
    context=binding.build_context(receipt=p["inspect"],inspect_raw=checked(inspect),
      handle_id=p["inspect"]["Id"],execution_image=p["inspect"]["Image"],
      accelerator_allocation=SimpleNamespace(vendor="nvidia",device_ids=tuple(p["pool_gpu_uuids"])),
      manifest_raw=checked(root/"quality-prompts/manifest.json"),
      prompts_sha256=sha(root/"quality-prompts/prompts.jsonl"),
      policy_raw=checked(root/"quality-policy.json"),dense_cache_sha256=p["dense_cache_sha256"])
    write(root/"quality-context.json",json.dumps(context,sort_keys=True,allow_nan=False).encode()+b"\n")
    result["QUALITY_CONTEXT_SHA256"]=sha(root/"quality-context.json")
print(json.dumps(result,sort_keys=True))
'''


def effective_private_phase(protocol, phase):
    """MInference scored submit/final share one complete private Judge profile."""
    if phase not in {"interactive", "final", "public"}:
        raise ValueError("unsupported authoritative evaluation phase")
    if protocol == PROTOCOL and phase != "public":
        return "final"
    return phase


def validate_receipt(receipt, *, handle_id, execution_image, accelerator_allocation):
    """Require the actual four-device pool and exact trusted allocation."""
    return runtime_binding.validate_pool_receipt(receipt,handle_id=handle_id,
        execution_image=execution_image,accelerator_allocation=accelerator_allocation)


def trusted_cache(private_context):
    path, expected = private_context.pop(CACHE_PATH_KEY, None), private_context.pop(CACHE_SHA_KEY, None)
    if path is None and expected is None: return None, None
    if not isinstance(path, str) or not Path(path).is_absolute() or not isinstance(expected, str) or SHA.fullmatch(expected) is None:
        raise ValueError("trusted cache requires an absolute host path and exact SHA together")
    source = Path(path); before = source.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
            or before.st_uid not in (0, os.getuid())
            or stat.S_IMODE(before.st_mode) & 0o077
            or not 0 < before.st_size <= MAX_CACHE_BYTES):
        raise ValueError("trusted cache must be a bounded, private, controller-owned regular file")
    raw = source.read_bytes(); after = source.lstat()
    if (len(raw) != before.st_size or hashlib.sha256(raw).hexdigest() != expected
            or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
        raise ValueError("trusted cache changed or differs from its declared SHA")
    return raw, expected


def _copy_private(backend, handle, raw, destination, log_dir):
    descriptor, temporary = tempfile.mkstemp(prefix=".private-bootstrap-", dir=log_dir)
    path = Path(temporary)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as stream: stream.write(raw)
        backend.copy_to_container(handle, path, destination)
        permissions = backend.exec_run(handle,
            ["/bin/sh", "-c", 'chown root:root "$1" && chmod 400 "$1"', "bootstrap-private", str(destination)], user="root")
        if permissions.exit_code != 0: raise RuntimeError("root private bootstrap file could not be protected")
    finally:
        path.unlink(missing_ok=True)


def install_private_bootstrap(backend, handle, *, protocol, phase, execution_image,
                              accelerator_allocation, private_context, log_dir):
    if protocol == "minference-kernel-prefill-v3":
        from sforge.harness.minference_kernel_bootstrap import install
        return install(backend, handle, phase=phase, execution_image=execution_image,
                       accelerator_allocation=accelerator_allocation,
                       private_context=private_context, log_dir=log_dir)
    phase = effective_private_phase(protocol, phase)
    if protocol is None or phase == "public": return
    if protocol != PROTOCOL or phase not in {"interactive", "final"}:
        raise ValueError("unsupported private Judge bootstrap")
    # This calls the trusted backend's actual live API. Backend support cannot
    # be inferred from image tags, environment, or create-time resource kwargs.
    receipt = backend.authoritative_container_receipt(handle)
    pool = validate_receipt(receipt, handle_id=handle.id, execution_image=execution_image,
        accelerator_allocation=accelerator_allocation)
    # Both scored phases resolve to final before reaching private bootstrap.
    raw_cache, cache_sha = trusted_cache(private_context)
    if raw_cache is not None:
        _copy_private(backend, handle, raw_cache, ROOT / "dense-quality.json", log_dir)
    helper_raw=Path(runtime_binding.__file__).read_bytes()
    _copy_private(backend,handle,helper_raw,ROOT / "quad-runtime-binding.py",log_dir)
    payload = {"inspect": receipt, "pool_gpu_uuids": list(pool), "phase": phase,
        "dense_cache_sha256": cache_sha,"binding_helper_sha256":hashlib.sha256(helper_raw).hexdigest()}
    transport = PurePosixPath("/tmp/.sforge-private-bootstrap.json")
    _copy_private(backend, handle, (json.dumps(payload, allow_nan=False) + "\n").encode(), transport, log_dir)
    try:
        # Isolated mode ignores inherited PYTHONOPTIMIZE and Python path hooks.
        # This stdlib-only installer uses its own trusted absolute asset paths.
        result = backend.exec_run(handle, ["python3", "-I", "-c", INSTALL, str(transport)], user="root")
        if result.exit_code != 0: raise RuntimeError("root-only private runtime bootstrap failed closed")
        def unique(pairs):
            out = {}
            for key, value in pairs:
                if key in out: raise ValueError("duplicate bootstrap result field")
                out[key] = value
            return out
        values = json.loads(result.output, object_pairs_hook=unique)
        expected_keys = {"FINAL_MANIFEST_SHA256"} | ({"QUALITY_CONTEXT_SHA256"} if phase == "final" else set())
        if not isinstance(values, dict) or set(values) != expected_keys or any(not isinstance(value, str) or SHA.fullmatch(value) is None for value in values.values()):
            raise ValueError("private runtime bootstrap returned an invalid identity")
        claimed = private_context.get("FINAL_MANIFEST_SHA256")
        if claimed is not None and claimed != values["FINAL_MANIFEST_SHA256"]:
            raise ValueError("host-declared performance manifest differs from the installed exact artifact")
        private_context["FINAL_MANIFEST_SHA256"] = values["FINAL_MANIFEST_SHA256"]
        # Host-only audit bytes. Do not attach these receipts to EvalReport or
        # expose them through an interactive/candidate response.
        from sforge.harness.run_evaluation import _durable_atomic_write_bytes
        _durable_atomic_write_bytes(Path(log_dir) / "private_runtime_bootstrap.json",
            (json.dumps({"protocol": protocol, "phase": phase, "inspect": receipt,
                "context_bindings": values, "dense_cache_sha256": cache_sha}, indent=2, sort_keys=True) + "\n").encode())
    finally:
        removed = backend.exec_run(handle, ["rm", "--force", str(transport)], user="root")
        if removed.exit_code != 0: raise RuntimeError("private runtime bootstrap transport cleanup failed")
