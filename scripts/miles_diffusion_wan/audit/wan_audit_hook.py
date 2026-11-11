"""Judge-owned sglang-diffusion audit hook for the Wan2.2-TI2V-5B DanceGRPO task.

Installed in the Judge image's site-packages with ``wan_audit_hook.pth`` (one line: ``import wan_audit_hook``), so it
loads in every Python process, including the engine's HTTP process and its scheduler/GPU-worker processes. Inert
unless ``$WAN_AUDIT_DIR`` names an existing directory. It records raw evidence and makes no judgement.

Records (JSONL ``audit-<host>-<pid>.jsonl`` + tensor files under ``traj/``):
* ``proc``: argv, selected env (cache-dit, alloc conf, attention), once per process;
* ``modules``: ``__file__`` of every loaded ``sglang.multimodal_gen`` module at the first request (overlay audit);
* ``request`` (HTTP process): the merged SamplingParams of every ``/rollout/generate`` request, recorded from the
  return value of ``build_sampling_params`` (after ``extra_sampling_params`` were applied);
* ``step`` (worker): for every rollout denoising step, ``FlowMatchEulerDiscreteScheduler.step`` inputs/outputs:
  request id, loop step index, sigma, sigma_next, and a tensor file with x_k (as passed), v_k (CFG-combined model
  output as passed) and x_{k+1} (as returned) — the scheduler file is outside the candidate's overlay whitelist;
* ``bucket`` (worker): every tensor-update call: names/numel/dtype of the tensors it carried, success, timing;
* ``publication_complete`` (worker): the first moment the buckets since the previous completion cover every
  expected parameter (``$WAN_AUDIT_EXPECTED`` = JSON {name: numel}); optional checksum (``$WAN_AUDIT_CHECKSUM=1``).
"""

import functools
import importlib.abc
import importlib.util
import json
import os
import socket
import sys
import threading
import time

_TARGETS = {
    "sglang.multimodal_gen.runtime.entrypoints.post_training.rollout_api": "_patch_rollout_api",
    "sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete": "_patch_scheduler",
    "sglang.multimodal_gen.runtime.post_training.weights_updater": "_patch_weights_updater",
}
_lock = threading.Lock()
_state = {"fh": None, "proc_done": False, "modules_done": False, "covered": {}, "pub_start": None, "n_pub": 0,
          "expected": None}

_REQ_FIELDS = (
    "request_id", "prompt", "negative_prompt", "seed", "generator_device", "width", "height", "num_frames",
    "num_inference_steps", "guidance_scale", "guidance_scale_2", "true_cfg_scale", "num_outputs_per_prompt",
    "rollout", "rollout_sde_type", "rollout_noise_level", "rollout_log_prob_no_const", "rollout_debug_mode",
    "rollout_sde_step_indices", "rollout_return_step_indices", "rollout_sde_noise_seed", "rollout_video_dtype",
    "flow_shift", "cfg_gate_step", "enable_teacache", "enable_cache_dit", "sigmas",
)
_ENV_KEYS = ("PYTORCH_CUDA_ALLOC_CONF", "SGLANG_CACHE_DIT_ENABLED", "CUDA_VISIBLE_DEVICES", "PYTHONPATH")


def _audit_dir():
    d = os.environ.get("WAN_AUDIT_DIR")
    return d if d and os.path.isdir(d) else None


def _emit(record):
    d = _audit_dir()
    if d is None:
        return
    record.setdefault("t", time.time())
    record["pid"] = os.getpid()
    line = json.dumps(record, separators=(",", ":"), default=str)
    with _lock:
        if _state["fh"] is None:
            _state["fh"] = open(os.path.join(d, f"audit-{socket.gethostname()}-{os.getpid()}.jsonl"), "a", buffering=1)
        _state["fh"].write(line + "\n")


def _proc_once():
    if _state["proc_done"]:
        return
    _state["proc_done"] = True
    _emit({"kind": "proc", "argv": sys.argv, "env": {k: os.environ.get(k) for k in _ENV_KEYS}})


def _modules_once():
    if _state["modules_done"]:
        return
    _state["modules_done"] = True
    mods = {n: getattr(m, "__file__", None) for n, m in list(sys.modules.items())
            if n.startswith("sglang.multimodal_gen") and m is not None}
    _emit({"kind": "modules", "files": mods})


# ------------------------------------------------------------------------------------------------ HTTP process


def _patch_rollout_api(module):
    orig = getattr(module, "build_sampling_params", None)
    if orig is None or getattr(orig, "_wan_audit", False):
        return

    @functools.wraps(orig)
    def build_sampling_params(*a, **kw):
        sp = orig(*a, **kw)
        _proc_once()
        _modules_once()
        rec = {"kind": "request"}
        for f in _REQ_FIELDS:
            rec[f] = getattr(sp, f, None)
        _emit(rec)
        return sp

    build_sampling_params._wan_audit = True
    module.build_sampling_params = build_sampling_params
    _emit({"kind": "hook_installed", "module": module.__name__})


# ------------------------------------------------------------------------------------------------ worker process


def _save_step(rid, step, payload):
    import torch

    d = os.path.join(_audit_dir(), "traj", str(rid))
    os.makedirs(d, exist_ok=True)
    torch.save(payload, os.path.join(d, f"k{step:02d}.pt"))


def _patch_scheduler(module):
    cls = getattr(module, "FlowMatchEulerDiscreteScheduler", None)
    if cls is None or getattr(cls, "_wan_audit", False):
        return
    orig = cls.step

    @functools.wraps(orig)
    def step(self, model_output, timestep, sample, *a, **kw):
        batch = kw.get("batch")
        rollout = batch is not None and bool(getattr(batch, "rollout", False))
        if rollout:
            x_k = sample.detach().to("cpu", copy=True)
            v_k = model_output.detach().to("cpu", copy=True)
            k = getattr(batch, "_rollout_loop_step_index", None)
            si = getattr(self, "step_index", None)
        out = orig(self, model_output, timestep, sample, *a, **kw)
        if rollout:
            prev = out[0] if isinstance(out, tuple) else getattr(out, "prev_sample", out)
            rid = getattr(batch, "request_id", None)
            sigmas = getattr(self, "sigmas", None)
            try:
                s = float(sigmas[si]) if si is not None else None
                sn = float(sigmas[si + 1]) if si is not None else None
            except Exception:
                s = sn = None
            if si is None and k is not None and sigmas is not None:
                s, sn = float(sigmas[k]), float(sigmas[k + 1])
            _save_step(rid, int(k) if k is not None else -1,
                       {"x": x_k, "v": v_k, "x_next": prev.detach().to("cpu", copy=True), "sigma": s, "sigma_next": sn})
            _emit({"kind": "step", "rid": rid, "k": k, "sigma": s, "sigma_next": sn,
                   "x_dtype": str(x_k.dtype), "v_dtype": str(v_k.dtype), "shape": list(x_k.shape)})
        return out

    cls.step = step
    cls._wan_audit = True
    _emit({"kind": "hook_installed", "module": module.__name__})


def _expected():
    if _state["expected"] is None:
        p = os.environ.get("WAN_AUDIT_EXPECTED")
        _state["expected"] = json.load(open(p)) if p and os.path.exists(p) else {}
    return _state["expected"]


def _patch_weights_updater(module):
    cls = getattr(module, "WeightsUpdater", None)
    if cls is None or getattr(cls, "_wan_audit", False):
        return
    orig_mat = cls._materialize_weights_iter
    orig_upd = cls.update_weights_from_tensor

    @functools.wraps(orig_mat)
    def _materialize_weights_iter(self, module_payload, load_format):
        it = orig_mat(self, module_payload, load_format)
        items = list(it)
        self._wan_audit_items = [(n, int(t.numel()), str(t.dtype)) for n, t in items]
        return iter(items)

    @functools.wraps(orig_upd)
    def update_weights_from_tensor(self, *a, **kw):
        t0 = time.time()
        if _state["pub_start"] is None:
            _state["pub_start"] = t0
        self._wan_audit_items = []
        ok, msg = orig_upd(self, *a, **kw)
        items = getattr(self, "_wan_audit_items", [])
        _emit({"kind": "bucket", "t_start": t0, "t_end": time.time(), "success": bool(ok), "message": str(msg)[:300],
               "target_modules": kw.get("target_modules"), "n": len(items), "items": items})
        if ok:
            for n, numel, _dt in items:
                _state["covered"][n] = numel
            exp = _expected()
            if exp and all(_state["covered"].get(n) == m for n, m in exp.items()):
                _state["n_pub"] += 1
                rec = {"kind": "publication_complete", "index": _state["n_pub"], "t_first_bucket": _state["pub_start"],
                       "t_complete": time.time(), "n_names": len(_state["covered"])}
                if os.environ.get("WAN_AUDIT_CHECKSUM") == "1":
                    try:
                        # exactly what POST /get_weights_checksum computes, so the two are comparable
                        from sglang.multimodal_gen.runtime.loader.weight_utils import compute_weights_checksum
                        from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
                            iter_materialized_weights,
                        )

                        mods = dict(module.get_updatable_modules(self.pipeline))
                        rec["checksum"] = compute_weights_checksum(iter_materialized_weights(mods["transformer"]))
                    except Exception as e:  # noqa: BLE001
                        rec["checksum_error"] = repr(e)
                _emit(rec)
                _state["covered"] = {}
                _state["pub_start"] = None
        return ok, msg

    cls._materialize_weights_iter = _materialize_weights_iter
    cls.update_weights_from_tensor = update_weights_from_tensor
    cls._wan_audit = True
    _emit({"kind": "hook_installed", "module": module.__name__})


# ------------------------------------------------------------------------------------------------ install


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _TARGETS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        orig_exec = spec.loader.exec_module
        patcher = globals()[_TARGETS[fullname]]

        def exec_module(module, _orig=orig_exec, _patcher=patcher):
            _orig(module)
            _patcher(module)

        spec.loader.exec_module = exec_module
        return spec


if _audit_dir() is not None and not any(isinstance(f, _PatchFinder) for f in sys.meta_path):
    for _name, _fn in _TARGETS.items():
        if _name in sys.modules:
            globals()[_fn](sys.modules[_name])
    sys.meta_path.insert(0, _PatchFinder())
