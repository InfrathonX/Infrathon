"""Judge-owned sglang-diffusion audit hook for the SD3.5-M DiffusionOPD task.

Installed into site-packages with ``opd_audit_hook.pth`` (``import opd_audit_hook``) so it runs at start-up of every
Python process, including the engine's HTTP server and GPU worker processes that miles_diffusion spawns. Inert unless
``$OPD_AUDIT_DIR`` names an existing directory. The candidate submits only ``repo/`` and cannot change it. It records
raw events, one JSONL file per process; the Judge checker decides.

HTTP server process (``sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api`` and
``...runtime.scheduler_client``):
  * ``params``: every SamplingParams built for ``/rollout/generate`` (all public fields, incl. extras), with rid
  * ``forward``: every scheduler call with kind (rollout request or control request such as a weight update),
    t_start / t_end on this host's clock, success, and the engine port
GPU worker process (``...runtime.post_training.weights_updater``):
  * ``lora_publish``: tensor names, count and fingerprints (sum over tensors of <W, R_name>, R_name drawn from a
    per-name seed) of every LoRA A/B set the engine applies: ``fingerprint`` on W as received and ``fingerprint16`` on
    W rounded to fp16 (the engine's precision; fp32 and fp16 publications of the same student give the same value,
    so the Judge can compare it exactly with its own dumps)
miles_diffusion framework path (the task requires the training to run through it; the Judge ties the engine-side
events above to these windows on the same host clock):
  * ``proc``: argv of every Python process (the MD driver is ``train_diffusion.py``)
  * ``md_rollout``: every ``miles.rollout.base_types.call_rollout_fn`` call (RolloutManager's generate / eval path),
    with evaluation flag, rollout id, t_start / t_end
  * ``md_train`` / ``md_update_weights``: every ``FSDPTrainRayActor.train`` (rollout id) and ``.update_weights`` call
"""

import functools
import hashlib
import importlib.abc
import importlib.util
import json
import os
import socket
import sys
import threading
import time

_ROLLOUT = "sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api"
_CLIENT = "sglang.multimodal_gen.runtime.scheduler_client"
_UPDATER = "sglang.multimodal_gen.runtime.post_training.weights_updater"
_MD_ROLLOUT = "miles.rollout.base_types"
_MD_ACTOR = "miles.backends.fsdp_utils.actor"
_lock = threading.Lock()
_state = {"fh": None, "port": None}


def _audit_dir():
    d = os.environ.get("OPD_AUDIT_DIR")
    return d if d and os.path.isdir(d) else None


def _emit(rec):
    d = _audit_dir()
    if d is None:
        return
    rec.setdefault("t", time.time())
    rec["pid"] = os.getpid()
    line = json.dumps(rec, separators=(",", ":"), default=str)
    with _lock:
        if _state["fh"] is None:
            _state["fh"] = open(os.path.join(d, f"audit-{socket.gethostname()}-{os.getpid()}.jsonl"), "a", buffering=1)
        _state["fh"].write(line + "\n")


def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return repr(v)


def _patch_rollout(module):
    if getattr(module, "_opd_audit_patched", False):
        return
    orig = module.build_sampling_params

    @functools.wraps(orig)
    def build_sampling_params(request_id, **kwargs):
        sp = orig(request_id, **kwargs)
        fields = {k: _jsonable(v) for k, v in vars(sp).items() if not k.startswith("_")} if hasattr(sp, "__dict__") else {}
        _emit({"kind": "params", "rid": request_id, "kwargs": _jsonable(kwargs), "sampling_params": fields,
               "port": _state["port"]})
        return sp

    module.build_sampling_params = build_sampling_params
    module._opd_audit_patched = True
    _emit({"kind": "hook_installed", "module": module.__name__})


def _patch_client(module):
    cls = getattr(module, "AsyncSchedulerClient", None)
    if cls is None or getattr(cls, "_opd_audit_patched", False):
        return
    orig = cls.forward

    @functools.wraps(orig)
    async def forward(self, batch, timeout_ms=None):
        sa = getattr(self, "server_args", None)
        _state["port"] = getattr(sa, "port", None)
        kind = type(batch).__name__
        rid = getattr(batch, "request_id", None) or getattr(batch, "rid", None)
        t0 = time.time()
        ok, err = None, None
        try:
            out = await orig(self, batch, timeout_ms)
            res = getattr(out, "output", None)
            ok = bool(res.get("success")) if isinstance(res, dict) and "success" in res else not bool(getattr(out, "error", None))
            return out
        except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
            err = type(exc).__name__
            raise
        finally:
            extra = {}
            for f in ("target_modules", "load_format", "weight_update_mode", "lora_alpha", "lora_rank"):
                if hasattr(batch, f):
                    extra[f] = _jsonable(getattr(batch, f))
            _emit({"kind": "forward", "batch": kind, "rid": _jsonable(rid), "t_start": t0, "t_end": time.time(),
                   "ok": ok, "error": err, "port": _state["port"], **extra})

    cls.forward = forward
    cls._opd_audit_patched = True
    _emit({"kind": "hook_installed", "module": module.__name__})


def _fingerprint(named):
    import torch

    total = total16 = 0.0
    names = []
    for name, t in named:
        names.append(name)
        seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little") & 0x7FFFFFFFFFFFFFFF
        g = torch.Generator(device=t.device).manual_seed(seed)
        r = torch.rand(t.shape, generator=g, device=t.device, dtype=torch.float32) - 0.5
        w = t.detach()
        total += float((w.float() * r).double().sum().item())
        total16 += float((w.to(torch.float16).float() * r).double().sum().item())
    return names, total, total16


def _patch_updater(module):
    if getattr(module, "_opd_audit_patched", False):
        return
    orig_group = module._group_lora_ab_tensors

    @functools.wraps(orig_group)
    def _group_lora_ab_tensors(materialized, *a, **kw):
        try:
            names, fp, fp16 = _fingerprint(materialized)
            _emit({"kind": "lora_publish", "n_tensors": len(names), "names_sha256":
                   hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest(), "names_sample": sorted(names)[:4],
                   "fingerprint": fp, "fingerprint16": fp16})
        except Exception as exc:  # noqa: BLE001 - the audit must not change engine behaviour
            _emit({"kind": "lora_publish", "error": repr(exc)})
        return orig_group(materialized, *a, **kw)

    module._group_lora_ab_tensors = _group_lora_ab_tensors
    module._opd_audit_patched = True
    _emit({"kind": "hook_installed", "module": module.__name__})


def _timed(kind, orig, extract):
    @functools.wraps(orig)
    def wrapper(*a, **kw):
        t0, ok = time.time(), False
        try:
            out = orig(*a, **kw)
            ok = True
            return out
        finally:
            try:
                extra = extract(a, kw)
            except Exception:  # noqa: BLE001 - the audit must not change behaviour
                extra = {}
            _emit({"kind": kind, "t_start": t0, "t_end": time.time(), "ok": ok, **extra})

    wrapper._opd_audit_wrapped = True
    return wrapper


def _patch_md_rollout(module):
    fn = getattr(module, "call_rollout_fn", None)
    if fn is None or getattr(fn, "_opd_audit_wrapped", False):
        return
    module.call_rollout_fn = _timed("md_rollout", fn, lambda a, kw: {
        "evaluation": bool(kw.get("evaluation")), "rollout_id": _jsonable(a[2]) if len(a) > 2 else None})
    _emit({"kind": "hook_installed", "module": module.__name__})


def _patch_md_actor(module):
    cls = getattr(module, "FSDPTrainRayActor", None)
    if cls is None or getattr(cls, "_opd_audit_patched", False):
        return
    if hasattr(cls, "train"):
        cls.train = _timed("md_train", cls.train, lambda a, kw: {
            "rollout_id": _jsonable(kw.get("rollout_id", a[1] if len(a) > 1 else None))})
    if hasattr(cls, "update_weights"):
        cls.update_weights = _timed("md_update_weights", cls.update_weights, lambda a, kw: {})
    cls._opd_audit_patched = True
    _emit({"kind": "hook_installed", "module": module.__name__})


_PATCHERS = {_ROLLOUT: _patch_rollout, _CLIENT: _patch_client, _UPDATER: _patch_updater,
             _MD_ROLLOUT: _patch_md_rollout, _MD_ACTOR: _patch_md_actor}


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _PATCHERS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        orig_exec = spec.loader.exec_module

        def exec_module(module, _orig=orig_exec, _name=fullname):
            _orig(module)
            _PATCHERS[_name](module)

        spec.loader.exec_module = exec_module
        return spec


if _audit_dir() is not None and not any(isinstance(f, _PatchFinder) for f in sys.meta_path):
    _emit({"kind": "proc", "argv": [str(x) for x in (getattr(sys, "argv", None) or [])[:4]],
           "orig_argv": [str(x) for x in (getattr(sys, "orig_argv", None) or [])[:6]], "ppid": os.getppid()})
    for _name, _fn in _PATCHERS.items():
        if _name in sys.modules:
            _fn(sys.modules[_name])
    sys.meta_path.insert(0, _PatchFinder())
