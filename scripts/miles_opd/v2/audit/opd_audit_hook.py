"""Judge-owned SGLang audit hook for the Miles OPD task.

Installed into the image's site-packages together with ``opd_audit_hook.pth`` (one
line: ``import opd_audit_hook``), so it runs at start-up of every Python process,
including every SGLang server or engine the run starts: the student sampling engine
that Miles spawns and, if the solution uses one, a teacher scoring server. It is inert
unless ``$OPD_AUDIT_DIR`` names an existing directory. The candidate submits only
``repo/``, ``task_entry.py`` and ``config/`` and cannot change this file.

It records raw events and makes no judgement; the Judge checker does that. Every record
carries the engine identity (``model_path``, ``port``, ``served_model_name``, pid), so
the checker can tell the student engine(s) from a teacher engine:

* ``engine``: a TokenizerManager was created (one per SGLang server/engine), with the
  model it serves;
* ``dispatch``: the exact token ids and sampling parameters a request carried when the
  tokenizer manager handed it to the scheduler, plus the number of successful
  weight-mutating calls this engine had completed so far;
* ``final``: the output ids, finish reason and weight-call count when the last response
  arrived. That count is read while SGLang still holds the request's reader lock, so a
  normal (writer-locked) update cannot slip in between; an update done while generation
  is paused (Miles' ``retract``) shows up as ``final`` > ``dispatch``;
* ``weights``: every ``update_weights_from_*`` call with start/end time and success;
* ``pause`` / ``continue``: generation pauses around updates.
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

_TARGET = "sglang.srt.managers.tokenizer_manager"
LP_HEAD = 32  # engine-returned log-probs kept per generation (first output tokens), for the W6 comparison
_lock = threading.Lock()
_state = {"fh": None}


def _audit_dir():
    d = os.environ.get("OPD_AUDIT_DIR")
    return d if d and os.path.isdir(d) else None


def _emit(record):
    d = _audit_dir()
    if d is None:
        return
    record.setdefault("t", time.time())
    record["pid"] = os.getpid()
    line = json.dumps(record, separators=(",", ":"))
    with _lock:
        if _state["fh"] is None:
            path = os.path.join(d, f"audit-{socket.gethostname()}-{os.getpid()}.jsonl")
            _state["fh"] = open(path, "a", buffering=1)
        _state["fh"].write(line + "\n")


def _title(pid):
    """process title (Ray names actor processes ``ray::Class.method``)"""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            return fh.read().replace(b"\0", b" ").decode(errors="replace").strip()[:300]
    except OSError:
        return None


def _calls(tm):
    return getattr(tm, "_opd_audit_weight_calls", 0)


def _engine_id(tm):
    sa = getattr(tm, "server_args", None)
    return {
        "model_path": getattr(sa, "model_path", None),
        "port": getattr(sa, "port", None),
        "served_model_name": getattr(sa, "served_model_name", None),
        "server_weight_version": getattr(sa, "weight_version", None),
    }


def _sampling(sp):
    keys = ("temperature", "top_p", "top_k", "min_p", "max_new_tokens", "ignore_eos")
    if sp is None:
        return None
    if isinstance(sp, dict):
        return {k: sp.get(k) for k in keys}
    return {k: getattr(sp, k, None) for k in keys}


def _as_list(x):
    if x is None:
        return None
    try:
        return [int(v) for v in x]
    except TypeError:
        return None


def _patch(module):
    tm_cls = getattr(module, "TokenizerManager", None)
    if tm_cls is None or getattr(tm_cls, "_opd_audit_patched", False):
        return

    orig_init = tm_cls.__init__

    @functools.wraps(orig_init)
    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        sa = getattr(self, "server_args", None)
        _emit({"kind": "engine", "tokenizer_path": getattr(sa, "tokenizer_path", None),
               "tp_size": getattr(sa, "tp_size", None), "dp_size": getattr(sa, "dp_size", None),
               "base_gpu_id": getattr(sa, "base_gpu_id", None), "title": _title(os.getpid()),
               "parent_title": _title(os.getppid()), **_engine_id(self)})

    tm_cls.__init__ = __init__

    orig_send = tm_cls._send_one_request

    @functools.wraps(orig_send)
    def _send_one_request(self, tokenized_obj, *a, **kw):
        _emit({
            "kind": "dispatch",
            "rid": getattr(tokenized_obj, "rid", None),
            "input_ids": _as_list(getattr(tokenized_obj, "input_ids", None)),
            "sampling": _sampling(getattr(tokenized_obj, "sampling_params", None)),
            "return_logprob": getattr(tokenized_obj, "return_logprob", None),
            "logprob_start_len": getattr(tokenized_obj, "logprob_start_len", None),
            "top_logprobs_num": getattr(tokenized_obj, "top_logprobs_num", None),
            "token_ids_logprob": getattr(tokenized_obj, "token_ids_logprob", None) is not None,
            "session": getattr(tokenized_obj, "session_params", None) is not None,
            "stream": getattr(tokenized_obj, "stream", None),
            "weight_calls": _calls(self),
            **_engine_id(self),
        })
        return orig_send(self, tokenized_obj, *a, **kw)

    tm_cls._send_one_request = _send_one_request

    orig_gen = tm_cls.generate_request

    @functools.wraps(orig_gen)
    async def generate_request(self, obj, request=None):
        t_call = time.time()
        calls_at_call = _calls(self)
        stream = bool(getattr(obj, "stream", False))
        finals = {}
        error = None
        try:
            async for resp in orig_gen(self, obj, request):
                # read the count now: orig is suspended inside its reader lock
                calls_now = _calls(self)
                for r in resp if isinstance(resp, list) else [resp]:
                    if not isinstance(r, dict):
                        continue
                    meta = r.get("meta_info") or {}
                    ent = finals.setdefault(meta.get("id"), {"output_ids": []})
                    ids = r.get("output_ids")
                    if ids is not None:
                        ent["output_ids"] = (ent["output_ids"] + list(ids)) if stream else list(ids)
                    lps = meta.get("output_token_logprobs")
                    if lps:
                        head = [float(x[0]) if isinstance(x, (list, tuple)) else float(x) for x in lps[:LP_HEAD]]
                        ent["out_lp_head"] = ((ent.get("out_lp_head") or []) + head)[:LP_HEAD] if stream else head
                    ent["finish_reason"] = meta.get("finish_reason")
                    ent["weight_version"] = meta.get("weight_version")
                    ent["prompt_tokens"] = meta.get("prompt_tokens")
                    ent["completion_tokens"] = meta.get("completion_tokens")
                    ent["weight_calls"] = calls_now
                yield resp
        except GeneratorExit:
            # SGLang's non-streaming handler takes the first response and closes
            # the generator; that is a normal completion once a response arrived
            if not finals:
                error = "GeneratorExit"
            raise
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            for rid, ent in finals.items():
                _emit({
                    "kind": "final",
                    "rid": rid,
                    "t_call": t_call,
                    "weight_calls_at_call": calls_at_call,
                    "output_ids": _as_list(ent.get("output_ids")),
                    "out_lp_head": ent.get("out_lp_head"),
                    "finish_reason": ent.get("finish_reason"),
                    "weight_version": ent.get("weight_version"),
                    "prompt_tokens": ent.get("prompt_tokens"),
                    "completion_tokens": ent.get("completion_tokens"),
                    "weight_calls": ent.get("weight_calls"),
                    "stream": stream,
                    "error": error,
                    **_engine_id(self),
                })
            if not finals:
                _emit({"kind": "final", "rid": None, "t_call": t_call, "error": error or "no_response", **_engine_id(self)})

    tm_cls.generate_request = generate_request

    def _wrap_weight_call(name):
        orig = getattr(tm_cls, name)

        @functools.wraps(orig)
        async def wrapped(self, *a, **kw):
            t0 = time.time()
            out = await orig(self, *a, **kw)
            ok = bool(out[0]) if isinstance(out, tuple) and out else bool(getattr(out, "success", out))
            if ok:
                with _lock:
                    self._opd_audit_weight_calls = _calls(self) + 1
            _emit({"kind": "weights", "method": name, "t_start": t0, "t_end": time.time(), "success": ok,
                   "weight_calls": _calls(self), **_engine_id(self)})
            return out

        setattr(tm_cls, name, wrapped)

    for name in sorted(dir(tm_cls)):
        if name.startswith("update_weights_from_"):
            _wrap_weight_call(name)

    def _wrap_marker(name, kind):
        orig = getattr(tm_cls, name, None)
        if orig is None:
            return

        @functools.wraps(orig)
        async def wrapped(self, *a, **kw):
            _emit({"kind": kind, "method": name, "weight_calls": _calls(self), **_engine_id(self)})
            return await orig(self, *a, **kw)

        setattr(tm_cls, name, wrapped)

    _wrap_marker("pause_generation", "pause")
    _wrap_marker("continue_generation", "continue")
    tm_cls._opd_audit_patched = True
    _emit({"kind": "hook_installed", "module": module.__name__, "title": _title(os.getpid()),
           "parent_title": _title(os.getppid())})


class _PatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != _TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
            return spec
        orig_exec = spec.loader.exec_module

        def exec_module(module, _orig=orig_exec):
            _orig(module)
            _patch(module)

        spec.loader.exec_module = exec_module
        return spec


if _audit_dir() is not None and not any(isinstance(f, _PatchFinder) for f in sys.meta_path):
    if _TARGET in sys.modules:
        _patch(sys.modules[_TARGET])
    else:
        sys.meta_path.insert(0, _PatchFinder())
