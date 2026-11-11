"""Judge-owned provenance hook for Miles tasks: Megatron trainer -> SGLang engine lineage.

Installed into site-packages next to ``alfworld_provenance_hook.pth`` (one line:
``import alfworld_provenance_hook``), so it runs at start-up of every Python process,
including Ray workers and the SGLang processes Miles spawns. It is inert unless
``$ALFWORLD_AUDIT_DIR`` names an existing directory. The candidate submits only ``repo/``,
``task_entry.py`` and ``config/`` and cannot change this file. It records raw evidence
and makes no judgement; ``alfworld_provenance_check.py`` does that.

Megatron side (pinned Megatron-Core, in every training process):

* ``megatron_model``: each GPTModel built, with its numerics-relevant config;
* ``megatron_step``: every outermost ``MegatronOptimizer.step()`` call: interval,
  process title, the Miles classes on the call stack, the optimizer class and source
  file, ``megatron.core``'s file, the optimizer config and param-group
  hyper-parameters, the returned grad norm, a fingerprint of the student weights after
  the step, and (in ``prov-grad-*.pt``) the reduced gradient of the fingerprint rows
  read just before the step, with the range this rank owns under the distributed
  optimizer;
* ``megatron_forward_fp``: a fingerprint of the weights used by the first
  grad-enabled GPTModel forward before the first step and after every step;
* ``megatron_digest``: right after the last step of a run (``rounds`` >= 2), the SHA-256
  of every student weight tensor's bytes in Megatron's own names, layout and dtype, with
  the interval it took, so the evaluator can require the final checkpoint to be exactly
  the weights the last optimizer step produced.

SGLang side (the scheduler process of every engine):

* ``sglang_publish``: after every publication (``begin_weight_update`` ...
  ``end_weight_update``, or one ``update_weights_from_disk`` / ``_ipc`` /
  ``pull_weights`` call), a fingerprint of the served weights;
* ``sglang_snapshot``: before two publications chosen at random at run time, a full
  copy of the served weights (tensor-parallel size 1 only) with the interval it took,
  so the evaluator can recompute a random training step from the exact weights that
  step started from and take the copy's time out of the timed window;
* ``sglang_digest``: after every successful publication that can be the engine's last
  one (``b >= rounds - 1``, ``rounds`` >= 2), the SHA-256 of every served weight tensor's
  bytes (tensor-parallel size 1 only) with the interval it took, so the evaluator can
  require the final checkpoint to be exactly the weights the engine served last.

The fingerprint is SHA-256 over the bf16 bytes of the first ``FP_ROWS`` rows of the
input embedding (tied to the LM head for the task's model). Trainer, engine and the
final checkpoint all store these rows identically, and AdamW changes thousands of
their bf16 values every step, so equal fingerprints identify one weight version. It
orders versions cheaply; only the full digests bind a whole model.
"""

import collections
import concurrent.futures
import functools
import hashlib
import importlib.abc
import importlib.util
import json
import os
import random
import socket
import sys
import threading
import time
import weakref

FP_ROWS = 128
N_SNAPSHOTS = 2
DIGEST_THREADS = 8
_TARGETS = (
    "megatron.core.models.gpt.gpt_model",
    "megatron.core.optimizer",
    "sglang.srt.managers.scheduler_components.weight_updater",
)
_CFG_FIELDS = (
    "bf16", "fp16", "params_dtype", "fp8", "fp4", "attention_backend", "attention_softmax_in_fp32",
    "fp32_residual_connection", "apply_rope_fusion", "masked_softmax_fusion", "bias_dropout_fusion",
    "gradient_accumulation_fusion", "recompute_granularity", "recompute_method", "recompute_num_layers",
    "sequence_parallel", "tensor_model_parallel_size", "pipeline_model_parallel_size",
    "context_parallel_size", "calculate_per_token_loss", "num_layers", "hidden_size",
    "num_attention_heads", "num_query_groups", "kv_channels", "ffn_hidden_size", "normalization",
    "layernorm_epsilon", "qk_layernorm", "deterministic_mode",
)
_OPT_CFG_FIELDS = (
    "optimizer", "lr", "min_lr", "weight_decay", "adam_beta1", "adam_beta2", "adam_eps", "clip_grad",
    "decoupled_weight_decay", "use_distributed_optimizer", "bf16", "fp16", "params_dtype", "main_params_dtype",
    "main_grads_dtype", "exp_avg_dtype", "exp_avg_sq_dtype", "use_precision_aware_optimizer",
    "optimizer_cpu_offload", "loss_scale",
)
_lock = threading.Lock()
_files = {}
_M = {"models": [], "steps": 0, "attempts": 0, "fp_pending": True, "in_step": False}
_S = {"publications": 0, "current": None, "snap_at": None, "embed": None}


def _dir():
    d = os.environ.get("ALFWORLD_AUDIT_DIR")
    return d if d and os.path.isdir(d) else None


def _emit(rec):
    d = _dir()
    if d is None:
        return
    rec.setdefault("t", time.time())
    rec["pid"] = os.getpid()
    try:
        line = json.dumps(rec, separators=(",", ":"), default=str)
        with _lock:
            fh = _files.get("jsonl")
            if fh is None:
                fh = _files["jsonl"] = open(os.path.join(d, f"prov-{socket.gethostname()}-{os.getpid()}.jsonl"), "a", buffering=1)
            fh.write(line + "\n")
    except Exception:  # noqa: BLE001 - never break the candidate's run; missing evidence fails closed
        pass


def _title(pid="self"):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return fh.read().replace(b"\0", b" ").decode(errors="replace").strip()[:300]
    except OSError:
        return None


def _file_of(obj):
    import inspect

    try:
        return inspect.getsourcefile(obj)
    except (TypeError, OSError):
        return None


def fingerprint(weight):
    """SHA-256 of the bf16 bytes of the first FP_ROWS rows (any dtype is cast to bf16 first)."""
    import torch

    w = weight.detach()[:FP_ROWS]
    if w.dtype != torch.bfloat16:
        w = w.to(torch.bfloat16)
    return hashlib.sha256(w.contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()


def tensor_digest(t):
    """SHA-256 of a CPU tensor's bytes in its own dtype (row-major)."""
    import torch

    return hashlib.sha256(t.detach().contiguous().reshape(-1).view(torch.uint8).numpy()).hexdigest()


def _digest_params(named):
    """{name: {dtype, shape, sha256}} of (name, tensor) pairs on any device: blocking device-to-host
    copies, hashed in DIGEST_THREADS threads while the next copies run (at most 2 x DIGEST_THREADS
    copies are held at once)."""
    out, inflight = {}, collections.deque()
    with concurrent.futures.ThreadPoolExecutor(DIGEST_THREADS) as pool:
        for name, p in named:
            if len(inflight) >= 2 * DIGEST_THREADS:
                n, f = inflight.popleft()
                out[n]["sha256"] = f.result()
            out[name] = {"dtype": str(p.dtype), "shape": list(p.shape)}
            inflight.append((name, pool.submit(tensor_digest, p.detach().to("cpu"))))
        for n, f in inflight:
            out[n]["sha256"] = f.result()
    return out


def _rounds():
    try:
        return int(os.environ.get("ALFWORLD_AUDIT_ROUNDS", "0"))
    except ValueError:
        return 0


# ---------------------------------------------------------------------------- Megatron side

def _student_embedding():
    """The newest trainable GPTModel that owns an input embedding, and that weight."""
    for ref in reversed(_M["models"]):
        m = ref()
        w = getattr(getattr(getattr(m, "embedding", None), "word_embeddings", None), "weight", None)
        if w is not None and w.requires_grad:
            return m, w
    return None, None


def _miles_stack(skip=2, limit=120):
    names, f, n = [], sys._getframe(skip), 0
    while f is not None and n < limit:
        s = f.f_locals.get("self")
        if s is not None:
            for c in type(s).__mro__:
                q = f"{c.__module__}.{c.__qualname__}"
                if q.startswith("miles") and q not in names:
                    names.append(q)
        f, n = f.f_back, n + 1
    return names[:30]


def _inner_optimizers(opt):
    inner = getattr(opt, "chained_optimizers", None)
    return list(inner) if inner else [opt]


def _owned_range(opt, param):
    """(start, end) of `param`'s flattened elements whose reduced gradient this rank holds,
    or None when the whole gradient is reduced on every rank (no distributed optimizer)."""
    for o in _inner_optimizers(opt):
        gmap = getattr(o, "model_param_gbuf_map", None)
        if not gmap or param not in gmap:
            continue
        gbuf_index, dtype, bucket_index = gmap[param]
        rng = o.gbuf_ranges[gbuf_index][dtype][bucket_index]["param_map"][param]["param"]
        return [int(rng.start), int(rng.end)]
    return None


def _dp_info():
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return {"world_rank": dist.get_rank(), "world_size": dist.get_world_size()}
    except Exception:  # noqa: BLE001 - evidence only
        pass
    return {}


def _actor_info():
    if "ray" not in sys.modules:
        return {}
    try:
        import ray

        if not ray.is_initialized():
            return {}
        ctx = ray.get_runtime_context()
        return {"ray_actor_id": ctx.get_actor_id(), "ray_job_id": ctx.get_job_id()}
    except Exception:  # noqa: BLE001 - evidence only
        return {}


def _optimizer_evidence(opt):
    groups = []
    for g in getattr(opt, "param_groups", []) or []:
        groups.append({"lr": g.get("lr"), "betas": list(g["betas"]) if g.get("betas") is not None else None,
                       "eps": g.get("eps"), "weight_decay": g.get("weight_decay"), "n_params": len(g.get("params", []))})
    inner = []
    for o in _inner_optimizers(opt):
        torch_opt = getattr(o, "optimizer", None)
        cfg = getattr(o, "config", None)
        inner.append({
            "cls": f"{type(o).__module__}.{type(o).__qualname__}", "file": _file_of(type(o)),
            "torch_optimizer": f"{type(torch_opt).__module__}.{type(torch_opt).__qualname__}" if torch_opt is not None else None,
            "adam_w_mode": getattr(torch_opt, "adam_w_mode", None),
            "config": {k: str(getattr(cfg, k, None)) if not isinstance(getattr(cfg, k, None), (int, float, bool, type(None)))
                       else getattr(cfg, k, None) for k in _OPT_CFG_FIELDS} if cfg is not None else None,
        })
    return groups, inner


def _grad_evidence(opt, attempt):
    """Save the reduced gradient of the fingerprint rows before the optimizer consumes it."""
    import torch

    _, w = _student_embedding()
    if w is None:
        return {"grad": None, "grad_error": "no trainable GPTModel with an input embedding"}
    g = getattr(w, "main_grad", None)
    src = "main_grad"
    if g is None:
        g, src = w.grad, "grad"
    if g is None:
        return {"grad": None, "grad_error": "embedding has neither main_grad nor grad"}
    rows = g.detach()[:FP_ROWS].to(torch.float32).cpu().contiguous()
    path = os.path.join(_dir(), f"prov-grad-{socket.gethostname()}-{os.getpid()}-{attempt:04d}.pt")
    torch.save({"rows": rows, "owned": _owned_range(opt, w), "row_width": int(w.shape[1]),
                "vocab_rows": int(w.shape[0]), "source": src, "dtype": str(g.dtype)}, path)
    return {"grad": path, "grad_source": src}


def _wrap_step(cls):
    orig = cls.step
    if getattr(orig, "_alfworld_prov", False):
        return

    @functools.wraps(orig)
    def step(self, *a, **kw):
        if _M["in_step"] or _dir() is None:
            return orig(self, *a, **kw)
        _M["in_step"] = True
        attempt = _M["attempts"]
        _M["attempts"] += 1
        t0 = time.time()
        rec = {"kind": "megatron_step", "attempt": attempt, "step_index": _M["steps"], "title": _title(),
               "miles_stack": _miles_stack(), "optimizer_cls": f"{type(self).__module__}.{type(self).__qualname__}",
               "optimizer_file": _file_of(type(self)), "megatron_core_file": getattr(sys.modules.get("megatron.core"), "__file__", None),
               **_actor_info(), **_dp_info()}
        try:
            rec.update(_grad_evidence(self, attempt))
        except Exception as exc:  # noqa: BLE001 - evidence only; the checker fails closed on it
            rec["grad_error"] = repr(exc)[:300]
        try:
            rec["param_groups"], rec["inner"] = _optimizer_evidence(self)
        except Exception as exc:  # noqa: BLE001
            rec["optimizer_error"] = repr(exc)[:300]
        t1 = time.time()
        try:
            out = orig(self, *a, **kw)
        finally:
            _M["in_step"] = False
        t2 = time.time()
        success = bool(out[0]) if isinstance(out, tuple) and out else bool(out)
        grad_norm = out[1] if isinstance(out, tuple) and len(out) > 1 else None
        try:
            grad_norm = float(grad_norm) if grad_norm is not None else None
        except (TypeError, ValueError):
            grad_norm = None
        rec.update(t_start=t0, t_evidence=t1, t_end=t2, success=success, grad_norm=grad_norm)
        _, w = _student_embedding()
        rec["fp_after"] = fingerprint(w) if w is not None else None
        if success:
            _M["steps"] += 1
            _M["fp_pending"] = True
        _emit(rec)
        if success and _M["steps"] == _rounds() >= 2:
            try:
                _trainer_digest()
            except Exception as exc:  # noqa: BLE001 - the checker fails closed on a missing digest
                _emit({"kind": "megatron_digest_skipped", "after_steps": _M["steps"], "reason": repr(exc)[:300]})
        return out

    step._alfworld_prov = True
    cls.step = step


def _trainer_digest():
    """Record the SHA-256 of every weight tensor of the student model after the run's last step."""
    m, _ = _student_embedding()
    if m is None:
        _emit({"kind": "megatron_digest_skipped", "after_steps": _M["steps"],
               "reason": "no trainable GPTModel with an input embedding"})
        return
    t0 = time.time()
    tensors = _digest_params(m.named_parameters())
    _emit({"kind": "megatron_digest", "after_steps": _M["steps"], "t_start": t0, "t_end": time.time(),
           "tensors": tensors, **_dp_info()})


def _patch_optimizer(pkg):
    base = getattr(sys.modules.get("megatron.core.optimizer.optimizer"), "MegatronOptimizer", None)
    if base is None:
        return
    wrapped = []
    for name, mod in list(sys.modules.items()):
        if not name.startswith("megatron.core.optimizer") or mod is None:
            continue
        for obj in list(vars(mod).values()):
            if isinstance(obj, type) and issubclass(obj, base) and "step" in obj.__dict__:
                _wrap_step(obj)
                wrapped.append(f"{obj.__module__}.{obj.__qualname__}")
    _emit({"kind": "hook_installed", "side": "megatron_optimizer", "wrapped": sorted(set(wrapped)), "title": _title()})


def _patch_gpt(module):
    cls = getattr(module, "GPTModel", None)
    if cls is None or getattr(cls, "_alfworld_prov", False):
        return
    init, fwd = cls.__init__, cls.forward

    @functools.wraps(init)
    def __init__(self, *a, **kw):
        init(self, *a, **kw)
        if _dir() is None:
            return
        _M["models"].append(weakref.ref(self))
        cfg = getattr(self, "config", None)
        _emit({"kind": "megatron_model", "model_id": id(self), "cls": f"{type(self).__module__}.{type(self).__qualname__}",
               "file": _file_of(type(self)), "gpt_file": _file_of(cls), "title": _title(),
               "config": {k: str(getattr(cfg, k, None)) for k in _CFG_FIELDS} if cfg is not None else None,
               "vocab_size": getattr(self, "vocab_size", None), "pre_process": getattr(self, "pre_process", None),
               "share_embeddings": getattr(self, "share_embeddings_and_output_weights", None), **_dp_info()})

    @functools.wraps(fwd)
    def forward(self, *a, **kw):
        if _M["fp_pending"] and _dir() is not None:
            import torch

            if torch.is_grad_enabled():
                m, w = _student_embedding()
                if m is self and w is not None:
                    _M["fp_pending"] = False
                    _emit({"kind": "megatron_forward_fp", "after_steps": _M["steps"], "fp": fingerprint(w)})
        return fwd(self, *a, **kw)

    cls.__init__, cls.forward = __init__, forward
    cls._alfworld_prov = True
    _emit({"kind": "hook_installed", "side": "megatron_gpt", "title": _title()})


# ---------------------------------------------------------------------------- SGLang side

def _runner(mgr):
    try:
        runners = mgr.get_model_runners("target")
    except Exception:  # noqa: BLE001
        runners = []
    if runners:
        return runners[0][1]
    return getattr(getattr(mgr, "tp_worker", None), "model_runner", None)


def _served_embedding(mgr):
    runner = _runner(mgr)
    if runner is None:
        return None, None
    cached = _S["embed"]
    if cached is not None and cached[0] is runner:
        return runner, cached[1]
    for name, p in runner.model.named_parameters():
        if name.endswith("embed_tokens.weight"):
            _S["embed"] = (runner, p)
            return runner, p
    return runner, None


def _first(*values):
    return next((v for v in values if v is not None), None)


def _engine(mgr):
    sa = getattr(getattr(mgr, "scheduler", None), "server_args", None)
    runner = _runner(mgr)
    ps = getattr(runner, "ps", None)  # this SGLang keeps tensor-parallel info on runner.ps
    return {"port": getattr(sa, "port", None), "model_path": getattr(sa, "model_path", None),
            "tp_size": _first(getattr(runner, "tp_size", None), getattr(ps, "tp_size", None)),
            "tp_rank": _first(getattr(runner, "tp_rank", None), getattr(ps, "tp_rank", None))}


def _snapshot_choice():
    if _S["snap_at"] is None:
        rounds = _rounds()
        # publication b (0-based, the initial push counts) starts from version b-1 or b;
        # [2, rounds-1] keeps both readings inside steps 1..rounds-1 and leaves at least
        # one round after the copy
        pool = list(range(2, rounds))
        _S["snap_at"] = set(random.SystemRandom().sample(pool, min(N_SNAPSHOTS, len(pool))))
    return _S["snap_at"]


def _snapshot(mgr, b):
    from safetensors.torch import save_file

    eng = _engine(mgr)
    runner, emb = _served_embedding(mgr)
    if runner is None or emb is None:
        _emit({"kind": "sglang_snapshot_skipped", "b": b, "reason": "no served embedding", **eng})
        return
    if eng["tp_size"] != 1:
        _emit({"kind": "sglang_snapshot_skipped", "b": b, "reason": f"tensor parallel size {eng['tp_size']}", **eng})
        return
    t0 = time.time()
    sd = {name: p.detach().to("cpu").contiguous() for name, p in runner.model.named_parameters()}  # blocking copies
    d = os.path.join(_dir(), "snapshots")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"snap-{socket.gethostname()}-{os.getpid()}-b{b:03d}.safetensors")
    save_file(sd, path + ".tmp")
    os.replace(path + ".tmp", path)
    t1 = time.time()
    _emit({"kind": "sglang_snapshot", "b": b, "path": path, "fp": fingerprint(emb), "t_start": t0, "t_end": t1,
           "n_tensors": len(sd), **eng})


def _digest_due(b):
    # the last of an engine's publications is b = rounds - 1 without an initial push and
    # b = rounds with one; digesting every b >= rounds - 1 covers any number of initial pushes
    rounds = _rounds()
    return rounds >= 2 and b >= rounds - 1


def _served_digest(mgr, b):
    """Record the SHA-256 of every served weight tensor."""
    eng = _engine(mgr)
    runner = _runner(mgr)
    if runner is None or eng["tp_size"] != 1:
        _emit({"kind": "sglang_digest_skipped", "b": b, "reason": "no model runner" if runner is None
               else f"tensor parallel size {eng['tp_size']}", **eng})
        return
    t0 = time.time()
    tensors = _digest_params(runner.model.named_parameters())
    _emit({"kind": "sglang_digest", "b": b, "t_start": t0, "t_end": time.time(), "tensors": tensors, **eng})


def _publication_start(mgr, method):
    b = _S["publications"]
    _S["publications"] += 1
    eng = _engine(mgr)
    if eng["tp_rank"] in (0, None) and b in _snapshot_choice():
        try:
            _snapshot(mgr, b)
        except Exception as exc:  # noqa: BLE001 - the checker fails closed on a missing snapshot
            _emit({"kind": "sglang_snapshot_skipped", "b": b, "reason": repr(exc)[:300], **eng})
    return b


def _publication_end(mgr, b, method, out):
    eng = _engine(mgr)
    if eng["tp_rank"] not in (0, None):
        return
    ok = getattr(out, "success", None)
    if ok is None and isinstance(out, tuple) and out:
        ok = out[0]
    _, emb = _served_embedding(mgr)
    _emit({"kind": "sglang_publish", "b": b, "method": method, "success": None if ok is None else bool(ok),
           "fp": fingerprint(emb) if emb is not None else None, **eng})
    if ok is not False and _digest_due(b):
        try:
            _served_digest(mgr, b)
        except Exception as exc:  # noqa: BLE001 - the checker fails closed on a missing digest
            _emit({"kind": "sglang_digest_skipped", "b": b, "reason": repr(exc)[:300], **eng})


def _patch_weight_updater(module):
    cls = getattr(module, "SchedulerWeightUpdaterManager", None)
    if cls is None or getattr(cls, "_alfworld_prov", False):
        return

    def session(name, role):
        orig = getattr(cls, name, None)
        if orig is None:
            return

        @functools.wraps(orig)
        def wrapped(self, *a, **kw):
            if _dir() is None:
                return orig(self, *a, **kw)
            if role == "begin":
                _S["current"] = _publication_start(self, name)
                return orig(self, *a, **kw)
            if role == "end":
                out = orig(self, *a, **kw)
                _publication_end(self, _S["current"], name, out)
                return out
            # a self-contained publication (disk reload, IPC, pull)
            b = _publication_start(self, name)
            out = orig(self, *a, **kw)
            _publication_end(self, b, name, out)
            return out

        setattr(cls, name, wrapped)

    session("begin_weight_update", "begin")
    session("end_weight_update", "end")
    for name in ("update_weights_from_disk", "update_weights_from_ipc", "pull_weights"):
        session(name, "single")
    cls._alfworld_prov = True
    _emit({"kind": "hook_installed", "side": "sglang_scheduler", "title": _title(), "parent_title": _title(os.getppid())})


_PATCHES = {
    "megatron.core.models.gpt.gpt_model": _patch_gpt,
    "megatron.core.optimizer": _patch_optimizer,
    "sglang.srt.managers.scheduler_components.weight_updater": _patch_weight_updater,
}


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in _PATCHES:
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
            try:
                _PATCHES[_name](module)
            except Exception as exc:  # noqa: BLE001 - never break the candidate's import
                _emit({"kind": "hook_error", "module": _name, "error": repr(exc)[:300]})

        spec.loader.exec_module = exec_module
        return spec


if _dir() is not None and not any(isinstance(f, _PatchFinder) for f in sys.meta_path):
    for _name, _fn in _PATCHES.items():
        if _name in sys.modules:
            _fn(sys.modules[_name])
    sys.meta_path.insert(0, _PatchFinder())
