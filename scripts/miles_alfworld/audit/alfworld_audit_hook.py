"""Judge-owned SGLang audit hook for the Miles ALFWorld task (derived from the OPCD hook).

Installed into the Judge image's site-packages together with
``alfworld_audit_hook.pth`` (one line: ``import alfworld_audit_hook``), so it runs at
start-up of every Python process, including the SGLang server that Miles spawns
with multiprocessing. It is inert unless an audit directory exists
(``$ALFWORLD_AUDIT_DIR``, else ``/judge_audit``). The candidate submits only
``repo/`` and cannot change this file.

It records raw events and makes no judgement; the Judge checker does that:

* ``dispatch``: the exact token ids and sampling parameters a request carried
  when the tokenizer manager handed it to the scheduler, plus the number of
  successful weight-mutating calls so far;
* ``final``: the output ids, their log-probs (when the request asked for them), finish reason and weight-call count when the last
  response arrived. That count is read while SGLang still holds the request's
  reader lock, so a normal (writer-locked) update cannot slip in between; an
  update done while generation is paused (Miles' ``retract``) shows up as
  ``final`` > ``dispatch``;
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
_lock = threading.Lock()
_state = {"weight_calls": 0, "fh": None, "dispatch": {}}


def _audit_dir():
    d = os.environ.get("ALFWORLD_AUDIT_DIR")
    if not d:
        return None
    return d if os.path.isdir(d) else None


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


def _engine_id(tm):
    sa = getattr(tm, "server_args", None)
    return {
        "model_path": getattr(sa, "model_path", None),
        "port": getattr(sa, "port", None),
        "server_weight_version": getattr(sa, "weight_version", None),
    }


def _sampling(sp):
    if sp is None:
        return None
    if isinstance(sp, dict):
        return {k: sp.get(k) for k in ("temperature", "top_p", "top_k", "min_p", "max_new_tokens", "ignore_eos")}
    return {k: getattr(sp, k, None) for k in ("temperature", "top_p", "top_k", "min_p", "max_new_tokens", "ignore_eos")}


def _as_list(x):
    if x is None:
        return None
    try:
        return [int(v) for v in x]
    except TypeError:
        return None


def _patch(module):
    tm_cls = getattr(module, "TokenizerManager", None)
    if tm_cls is None or getattr(tm_cls, "_alfworld_audit_patched", False):
        return

    orig_send = tm_cls._send_one_request

    @functools.wraps(orig_send)
    def _send_one_request(self, tokenized_obj, *a, **kw):
        rid = getattr(tokenized_obj, "rid", None)
        rec = {
            "kind": "dispatch",
            "rid": rid,
            "input_ids": _as_list(getattr(tokenized_obj, "input_ids", None)),
            "sampling": _sampling(getattr(tokenized_obj, "sampling_params", None)),
            "return_logprob": getattr(tokenized_obj, "return_logprob", None),
            "logprob_start_len": getattr(tokenized_obj, "logprob_start_len", None),
            "top_logprobs_num": getattr(tokenized_obj, "top_logprobs_num", None),
            "token_ids_logprob": getattr(tokenized_obj, "token_ids_logprob", None) is not None,
            "session": getattr(tokenized_obj, "session_params", None) is not None,
            "stream": getattr(tokenized_obj, "stream", None),
            "weight_calls": _state["weight_calls"],
            **_engine_id(self),
        }
        _emit(rec)
        return orig_send(self, tokenized_obj, *a, **kw)

    tm_cls._send_one_request = _send_one_request

    orig_gen = tm_cls.generate_request

    @functools.wraps(orig_gen)
    async def generate_request(self, obj, request=None):
        t_call = time.time()
        calls_at_call = _state["weight_calls"]
        stream = bool(getattr(obj, "stream", False))
        finals = {}
        error = None
        try:
            async for resp in orig_gen(self, obj, request):
                # read the count now: orig is suspended inside its reader lock
                calls_now = _state["weight_calls"]
                for r in resp if isinstance(resp, list) else [resp]:
                    if not isinstance(r, dict):
                        continue
                    meta = r.get("meta_info") or {}
                    rid = meta.get("id")
                    ent = finals.setdefault(rid, {"output_ids": []})
                    ids = r.get("output_ids")
                    if ids is not None:
                        ent["output_ids"] = (ent["output_ids"] + list(ids)) if stream else list(ids)
                    lps = meta.get("output_token_logprobs")
                    if lps is not None:
                        vals = [float(x[0]) if x is not None and x[0] is not None else None for x in lps]
                        ent["output_logprobs"] = (ent.get("output_logprobs", []) + vals) if stream else vals
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
                    "output_logprobs": ent.get("output_logprobs"),
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
                    _state["weight_calls"] += 1
            _emit({"kind": "weights", "method": name, "t_start": t0, "t_end": time.time(), "success": ok,
                   "weight_calls": _state["weight_calls"], **_engine_id(self)})
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
            _emit({"kind": kind, "method": name, "weight_calls": _state["weight_calls"], **_engine_id(self)})
            return await orig(self, *a, **kw)

        setattr(tm_cls, name, wrapped)

    _wrap_marker("pause_generation", "pause")
    _wrap_marker("continue_generation", "continue")
    tm_cls._alfworld_audit_patched = True
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
