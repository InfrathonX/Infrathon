#!/usr/bin/env python3
"""Trusted runtime for KV eviction on frozen Qwen3-8B (v2: upstream evaluation items)."""
from __future__ import annotations

import gc
import gzip
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, Sequence

# Candidate imports must never create a submission-visible __pycache__.
sys.dont_write_bytecode = True

MODEL_ID = "Qwen/Qwen3-8B"
GENERATOR_PROVENANCE = "kv-eviction-items-v2"
RATIOS = (0.5, 0.75, 0.875)
FAMILIES = ("retrieval", "aggregation", "qa", "generation")
R_MAX = 48
R_MIN = 16
LATENCY_FRACTION = 0.30
LATENCY_ITEM_CAP = 0.60
MEMORY_LIMIT_GIB = 8
MEMORY_LIMIT_BYTES = MEMORY_LIMIT_GIB << 30
MIN_KEEP = 64
# Context length including the chat-template prefix, in Qwen3 tokens.  The
# upper bound keeps every item inside the measured single-GPU memory envelope;
# the lower bound only rejects degenerate rows.
MIN_CONTEXT_TOKENS = 1024
MAX_CONTEXT_TOKENS = 28672
MAX_ASSET_BYTES = 64 << 20
WARMUP_CONTEXT_TOKENS = 2048
PUBLIC_SUITE_ID = "kv-eviction-public-v2"
FINAL_SUITE_ID = "final-holdout-v2"
ITEM_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
TASK_PATTERN = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")


# --------------------------------------------------------------------------- items

def validate_item(raw: Mapping[str, Any]) -> dict[str, Any]:
    """A scored item: context for the policy, question and answer prefix for replay."""
    if not isinstance(raw, Mapping): raise ValueError("item must be an object")
    item_id, family, task = raw.get("item_id"), raw.get("family"), raw.get("task")
    context, question, prefix = raw.get("context"), raw.get("question"), raw.get("answer_prefix", "")
    meta = raw.get("meta", {})
    if not isinstance(item_id, str) or not ITEM_ID_PATTERN.fullmatch(item_id): raise ValueError("unsafe item id")
    if family not in FAMILIES: raise ValueError(f"unknown item family: {family!r}")
    if not isinstance(task, str) or not TASK_PATTERN.fullmatch(task): raise ValueError("invalid task name")
    if not isinstance(context, str) or not context.strip(): raise ValueError("context is empty")
    # Summarisation items carry their instruction in the context and the
    # answer prefix, so an empty question is legitimate (as in kvpress).
    if not isinstance(question, str): raise ValueError("question must be a string")
    if not isinstance(prefix, str): raise ValueError("answer_prefix must be a string")
    if not isinstance(meta, Mapping): raise ValueError("meta must be an object")
    return {"item_id": item_id, "family": family, "task": task, "context": context,
            "question": question, "answer_prefix": prefix, "meta": dict(meta)}


def context_only(item: Mapping[str, Any]) -> dict[str, str]:
    """What the unprivileged Phase-A process receives: no question, no answer prefix."""
    return {"item_id": str(item["item_id"]), "context": str(item["context"])}


def validate_context_item(raw: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(raw, Mapping) or set(raw) != {"item_id", "context"}: raise ValueError("Phase-A item fields differ")
    if not isinstance(raw["item_id"], str) or not ITEM_ID_PATTERN.fullmatch(raw["item_id"]): raise ValueError("unsafe item id")
    if not isinstance(raw["context"], str) or not raw["context"].strip(): raise ValueError("context is empty")
    return {"item_id": raw["item_id"], "context": raw["context"]}


def context_sha256(context: str) -> str:
    return hashlib.sha256(context.encode("utf-8")).hexdigest()


def item_sha256(item: Mapping[str, Any]) -> str:
    """Identity of the scored text: context, question and answer prefix."""
    blob = "\x00".join((str(item["context"]), str(item["question"]), str(item.get("answer_prefix", ""))))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_items_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    source = Path(path)
    if not source.is_file() or source.is_symlink(): raise RuntimeError(f"items file missing: {source}")
    rows = tuple(validate_item(x) for x in _read_jsonl(source))
    if not rows or len({row["item_id"] for row in rows}) != len(rows): raise ValueError("item ids must be nonempty and unique")
    return rows


def validate_final_items(rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any],
                         public_items: Sequence[Mapping[str, Any]] = ()) -> tuple[dict[str, Any], ...]:
    """Check materialised holdout items against the frozen manifest, item by item."""
    if manifest.get("schema_version") != 1 or manifest.get("suite_id") != FINAL_SUITE_ID: raise ValueError("final manifest identity differs")
    entries = manifest.get("items")
    if not isinstance(entries, list) or not entries: raise ValueError("final manifest has no items")
    items = tuple(validate_item(x) for x in rows)
    if len(items) != len(entries): raise ValueError("final item count differs from the manifest")
    for item, entry in zip(items, entries):
        if not isinstance(entry, Mapping): raise ValueError("final manifest entry must be an object")
        if (item["item_id"], item["family"], item["task"]) != (entry.get("item_id"), entry.get("family"), entry.get("task")):
            raise ValueError(f"final item identity differs at {entry.get('item_id')!r}")
        if item_sha256(item) != entry.get("sha256"): raise ValueError(f"final item text differs at {item['item_id']}")
    counts = manifest.get("counts_per_family")
    actual = {family: sum(x["family"] == family for x in items) for family in FAMILIES}
    if not isinstance(counts, Mapping) or {str(k): int(v) for k, v in counts.items()} != {k: v for k, v in actual.items() if v}:
        raise ValueError("final family counts differ")
    if len({x["item_id"] for x in items}) != len(items): raise ValueError("duplicate final item id")
    contexts = [context_sha256(x["context"]) for x in items]
    if len(set(contexts)) != len(contexts): raise ValueError("duplicate final context")
    if set(contexts) & {context_sha256(x["context"]) for x in public_items}: raise ValueError("public/final context overlap")
    return items


def _load_public_default() -> tuple[dict[str, Any], ...]:
    root = Path(os.environ.get("KV_EVICTION_CORPUS_DIR", str(Path(__file__).resolve().parents[1] / "corpus/public")))
    try: return load_items_jsonl(root / "items_public.jsonl")
    except (FileNotFoundError, RuntimeError, ValueError): return ()


PUBLIC_ITEMS = _load_public_default()


def suite_items(name: str, items: Sequence[Mapping[str, Any]] | None = None) -> tuple[dict[str, Any], ...]:
    pool = tuple(PUBLIC_ITEMS if items is None else items)
    if name == "one": return pool[:1]
    if name == "quick":
        seen: set[tuple[str, str]] = set(); chosen = []
        for item in pool:
            key = (item["family"], item["task"])
            if key not in seen: seen.add(key); chosen.append(item)
        return tuple(chosen)
    if name in {"full", "public"}: return pool
    raise ValueError("unknown suite")


# --------------------------------------------------------------- tokenisation

def preprocess_context_question(tokenizer: Any, context: str, question: str = "", answer_prefix: str = "") -> dict[str, Any]:
    """kvpress pipeline semantics: chat-template the context, then question + suffix + answer prefix."""
    if tokenizer.chat_template is None:
        rendered_context = (getattr(tokenizer, "bos_token", "") or "") + context
        question_suffix = "\n"
    else:
        separator = "#" * (len(context) + 10)
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": context + separator}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
        pieces = rendered.split(separator)
        if len(pieces) != 2: raise ValueError("chat-template separator split failed")
        rendered_context, question_suffix = pieces
    return {
        "context_ids": tokenizer.encode(rendered_context, return_tensors="pt", add_special_tokens=False),
        "question_ids": tokenizer.encode(question + question_suffix + answer_prefix, return_tensors="pt", add_special_tokens=False),
        "question_suffix": question_suffix,
    }


def context_tokens(tokenizer: Any, context: str) -> int:
    return int(preprocess_context_question(tokenizer, context)["context_ids"].shape[1])


def check_context_length(n_tokens: int) -> None:
    if not MIN_CONTEXT_TOKENS <= n_tokens <= MAX_CONTEXT_TOKENS:
        raise ValueError(f"context has {n_tokens} tokens, outside [{MIN_CONTEXT_TOKENS}, {MAX_CONTEXT_TOKENS}]")


def _context_tokens(item: Mapping[str, Any]) -> int:
    tokens = item.get("meta", {}).get("context_tokens")
    return tokens if isinstance(tokens, int) and not isinstance(tokens, bool) else len(str(item["context"])) // 4


def item_cost(item: Mapping[str, Any]) -> float:
    """Relative Judge cost used only to balance shards (prefill-dominated plus a fixed replay part)."""
    return float(_context_tokens(item)) + 8000.0


def handoff_bytes(items: Sequence[Mapping[str, Any]]) -> int:
    """Upper bound on the Phase A -> Phase B files: one bit per (ratio, layer, KV head, context position) plus framing (Qwen3-8B: 36 layers x 8 KV heads)."""
    return sum(len(RATIOS) * (36 * 8 * (_context_tokens(x) // 8 + 1) + (1 << 16)) for x in items)


# ----------------------------------------------------------------- aggregation

def finite_unit(value: Any, label: str = "fidelity") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)): raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0 < number <= 1: raise ValueError(f"{label} must be finite in (0,1]")
    return number


def geometric_mean(values: Iterable[float]) -> float:
    values = [finite_unit(x) for x in values]
    if not values: raise ValueError("geometric mean requires values")
    return math.exp(math.fsum(math.log(x) for x in values) / len(values))


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    family_fid: dict[str, dict[str, float]] = {}; ratio_score: dict[str, float] = {}
    active_families = tuple(f for f in FAMILIES if any(row.get("family") == f for row in rows))
    if not active_families: raise ValueError("aggregation requires rows")
    for ratio in RATIOS:
        tag = str(ratio); family_fid[tag] = {}
        for family in active_families:
            values = [finite_unit(row["fid"]) for row in rows if float(row["ratio"]) == ratio and row["family"] == family]
            if not values: raise ValueError(f"missing rows for {family}/{ratio}")
            family_fid[tag][family] = math.fsum(values) / len(values)
        ratio_score[tag] = geometric_mean(family_fid[tag].values())
    family_score = {family: math.fsum(family_fid[str(r)][family] for r in RATIOS) / len(RATIOS) for family in active_families}
    select_total = math.fsum(float(row.get("select_time_sum", math.inf)) for row in rows)
    prefill_total = math.fsum(float(row.get("prefill_s", 0.0)) for row in rows)
    latency_ratio_suite = select_total / prefill_total if prefill_total > 0.0 else math.inf
    item_ratios = [float(row.get("select_time_sum", math.inf)) / float(row.get("prefill_s", 0.0)) if float(row.get("prefill_s", 0.0)) > 0.0 else math.inf for row in rows]
    latency_ratio_item_max = max(item_ratios, default=math.inf)
    gates = {
        "indices_ok": all(row.get("indices_ok") is True for row in rows),
        "immutable_ok": all(row.get("immutable_ok") is True for row in rows),
        "rng_ok": all(row.get("rng_ok") is True for row in rows),
        "budgets_ok": all(row.get("budgets_ok") is True for row in rows),
        "latency_ok": latency_ratio_suite <= LATENCY_FRACTION and latency_ratio_item_max <= LATENCY_ITEM_CAP,
        "memory_ok": all(int(row.get("memory_delta", MEMORY_LIMIT_BYTES + 1)) <= MEMORY_LIMIT_BYTES for row in rows),
        "fidelity_ok": all(math.isfinite(float(row.get("fid", math.nan))) and int(row.get("n_tokens", 0)) >= R_MIN for row in rows),
    }
    return {"family_fid": family_fid, "family_score": family_score, "ratio_score": ratio_score,
            "primary": 100.0 * math.fsum(ratio_score.values()) / len(RATIOS),
            "latency_ratio_suite": latency_ratio_suite, "latency_ratio_item_max": latency_ratio_item_max, "gates": gates}


def task_breakdown(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """Mean fidelity per task and ratio (Judge-private diagnostics)."""
    out: dict[str, dict[str, float]] = {}
    for task in sorted({str(row["task"]) for row in rows if "task" in row}):
        out[task] = {}
        for ratio in RATIOS:
            values = [float(row["fid"]) for row in rows if row.get("task") == task and float(row["ratio"]) == ratio]
            if values: out[task][str(ratio)] = math.fsum(values) / len(values)
    return out


# --------------------------------------------------------------- trusted ops

@dataclass(frozen=True)
class TrustedOps:
    gather: Callable[..., Any]; cat: Callable[..., Any]; log_softmax: Callable[..., Any]
    exp: Callable[..., Any]; sum: Callable[..., Any]; topk: Callable[..., Any]
    arange: Callable[..., Any]; equal: Callable[..., Any]; isfinite: Callable[..., Any]
    view: Callable[..., Any]; to: Callable[..., Any]; clone: Callable[..., Any]
    sort: Callable[..., Any]; get_cpu_rng: Callable[..., Any]; set_cpu_rng: Callable[..., Any]
    get_cuda_rng: Callable[..., Any]; set_cuda_rng: Callable[..., Any]
    max_memory: Callable[..., Any]; reset_peak: Callable[..., Any]; allocated: Callable[..., Any]
    save: Callable[..., Any]; load: Callable[..., Any]
    empty_strided: Callable[..., Any]; normal_: Callable[..., Any]
    current_device: Callable[..., Any]; device_count: Callable[..., Any]
    reserved: Callable[..., Any]; max_reserved: Callable[..., Any]

    @classmethod
    def capture(cls, torch: Any) -> "TrustedOps":
        return cls(torch.gather, torch.cat, torch.log_softmax, torch.exp, torch.sum,
                   torch.topk, torch.arange, torch.equal, torch.isfinite,
                   torch.Tensor.view, torch.Tensor.to, torch.clone, torch.sort,
                   torch.get_rng_state, torch.set_rng_state, torch.cuda.get_rng_state,
                   torch.cuda.set_rng_state, torch.cuda.max_memory_allocated,
                   torch.cuda.reset_peak_memory_stats, torch.cuda.memory_allocated,
                   torch.save, torch.load, torch.empty_strided, torch.Tensor.normal_,
                   torch.cuda.current_device, torch.cuda.device_count,
                   torch.cuda.memory_reserved, torch.cuda.max_memory_reserved)


@dataclass(frozen=True)
class TrustedClock:
    perf_counter: Callable[[], float]; synchronize: Callable[..., Any]
    current_stream: Callable[..., Any]; event: Callable[..., Any]

    @classmethod
    def capture(cls, torch: Any) -> "TrustedClock":
        return cls(time.perf_counter, torch.cuda.synchronize, torch.cuda.current_stream, torch.cuda.Event)

    def time_cuda(self, fn: Callable[[], Any], device: Any) -> tuple[float, Any]:
        self.synchronize(device); start_wall = self.perf_counter()
        start = self.event(enable_timing=True); end = self.event(enable_timing=True)
        stream = self.current_stream(device); stream.record_event(start); value = fn(); stream.record_event(end)
        self.synchronize(device)
        return max(self.perf_counter() - start_wall, float(start.elapsed_time(end)) / 1000.0), value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""): digest.update(chunk)
    return digest.hexdigest()


# Capability envelope: what the task contract actually needs, not the GPU model
# it was qualified on.  The GPU name and UUID are recorded as provenance only;
# they never gate admission.  Keep these two constants identical in the Judge
# verifier (tests assert the pair).
MIN_COMPUTE_CAPABILITY = (8, 0)
MIN_DEVICE_MEMORY_BYTES = 56 * 1024**3


# The sharded Judge names each shard's GPU here (index in PCI bus order) and
# leaves every allocated GPU visible: PyroMind's HAMi memory accounting is only
# correct under the identity device mapping (a CUDA_VISIBLE_DEVICES subset is
# charged to the wrong device), so shards cannot be pinned by visibility.
# Without the variable exactly one GPU must be visible.
SHARD_DEVICE_ENV = "SFORGE_JUDGE_DEVICE"


def bind_device(torch: Any) -> int:
    """Make this process's GPU the current CUDA device and return its index."""
    if not torch.cuda.is_available(): raise RuntimeError("exactly one CUDA GPU required per shard")
    count = int(torch.cuda.device_count()); raw = os.environ.get(SHARD_DEVICE_ENV)
    if raw is None:
        if count != 1: raise RuntimeError("exactly one CUDA GPU required per shard")
        index = 0
    elif raw.isdigit() and int(raw) < count: index = int(raw)
    else: raise RuntimeError(f"{SHARD_DEVICE_ENV}={raw!r} names no visible GPU")
    torch.cuda.set_device(index); return index


def _same_current_device(ops: "TrustedOps", own: int) -> None:
    if int(ops.current_device()) != own: raise RuntimeError(f"candidate changed the current CUDA device from {own}")


def _own_device_only(ops: "TrustedOps", own: int) -> None:
    """Candidate code may use only its own GPU: the current device is unchanged
    and no other GPU's caching allocator holds, or has held, any memory."""
    _same_current_device(ops, own)
    for j in range(int(ops.device_count())):
        if j != own and (int(ops.reserved(j)) or int(ops.max_reserved(j))):
            raise RuntimeError(f"candidate used CUDA device {j}; only its own GPU {own} is allowed")


def hardware_inventory(torch: Any) -> dict[str, Any]:
    index = bind_device(torch)
    capability = tuple(int(v) for v in torch.cuda.get_device_capability(index)); name = str(torch.cuda.get_device_name(index))
    props = torch.cuda.get_device_properties(index)
    total_memory_bytes = int(props.total_memory)
    if capability < MIN_COMPUTE_CAPABILITY:
        raise RuntimeError(f"GPU compute capability {capability} is below the task minimum {MIN_COMPUTE_CAPABILITY}: {name}")
    if total_memory_bytes < MIN_DEVICE_MEMORY_BYTES:
        raise RuntimeError(f"GPU memory {total_memory_bytes} bytes is below the task minimum {MIN_DEVICE_MEMORY_BYTES} bytes: {name}")
    return {"name": name, "count": 1, "device_index": index, "capability": list(capability), "uuid": str(getattr(props, "uuid", "")), "total_memory_bytes": total_memory_bytes, "min_compute_capability": list(MIN_COMPUTE_CAPABILITY), "min_device_memory_bytes": MIN_DEVICE_MEMORY_BYTES, "torch_version": str(torch.__version__), "torch_cuda_version": str(torch.version.cuda)}


def load_tokenizer():
    from transformers import AutoTokenizer
    model_dir = os.environ.get("KV_EVICTION_MODEL_DIR")
    if not model_dir: raise RuntimeError("KV_EVICTION_MODEL_DIR is required")
    return AutoTokenizer.from_pretrained(model_dir, local_files_only=True)


def load_model(device: str = "cuda"):
    import torch
    from transformers import AutoModelForCausalLM
    model_dir = os.environ.get("KV_EVICTION_MODEL_DIR")
    if not model_dir: raise RuntimeError("KV_EVICTION_MODEL_DIR is required")
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa", low_cpu_mem_usage=True, device_map={"": bind_device(torch)} if str(device).startswith("cuda") else None)
    if not str(device).startswith("cuda"): model = model.to(device)
    model.eval(); model.requires_grad_(False); return model


def _model_spec(model: Any) -> dict[str, Any]:
    c = model.config
    return {"model_id": MODEL_ID, "n_layers": int(c.num_hidden_layers), "n_heads": int(c.num_attention_heads), "n_kv_heads": int(c.num_key_value_heads), "head_dim": int(c.head_dim), "hidden_size": int(c.hidden_size), "rope_theta": float(c.rope_theta), "n_sink_floor": 4, "min_keep": MIN_KEEP}


@dataclass
class PrefillCapture:
    cache: Any; hidden_states: list[Any]; queries: list[Any]; keys: list[Any]
    values: list[Any]; position_embeddings: tuple[Any, Any]; weights: list[dict[str, Any]]
    prefill_s: float; context_len: int


def _plain_prefill(model: Any, context_ids: Any, clock: TrustedClock) -> PrefillCapture:
    from transformers import DynamicCache
    device = next(model.parameters()).device; ids = context_ids.to(device); cache = DynamicCache()
    elapsed, _ = clock.time_cuda(lambda: model.model(input_ids=ids, past_key_values=cache, use_cache=True), device)
    keys = [layer.keys for layer in cache.layers]; values = [layer.values for layer in cache.layers]
    if len(keys) != int(model.config.num_hidden_layers): raise RuntimeError("plain prefill cache incomplete")
    return PrefillCapture(cache, [], [], keys, values, (None, None), [], elapsed, int(ids.shape[1]))


def _capture_prefill(model: Any, context_ids: Any, clock: TrustedClock) -> PrefillCapture:
    from transformers import DynamicCache
    device = next(model.parameters()).device; ids = context_ids.to(device); n_layers = int(model.config.num_hidden_layers)
    hidden: list[Any] = [None] * n_layers; queries: list[Any] = [None] * n_layers; keys: list[Any] = [None] * n_layers; values: list[Any] = [None] * n_layers
    rotary: list[Any] = [None]; hooks = []
    def q_hook(idx: int):
        def hook(_module: Any, _args: Any, output: Any): queries[idx] = output.transpose(1, 2)
        return hook
    def attn_hook(idx: int):
        def hook(_module: Any, _args: Any, kwargs: Any, output: Any):
            hidden[idx] = kwargs["hidden_states"]; layer = kwargs["past_key_values"].layers[idx]
            keys[idx] = layer.keys; values[idx] = layer.values
            if rotary[0] is None: rotary[0] = kwargs["position_embeddings"]
            return output
        return hook
    for idx, layer in enumerate(model.model.layers):
        hooks.append(layer.self_attn.q_norm.register_forward_hook(q_hook(idx)))
        hooks.append(layer.self_attn.register_forward_hook(attn_hook(idx), with_kwargs=True))
    cache = DynamicCache()
    try: elapsed, _ = clock.time_cuda(lambda: model.model(input_ids=ids, past_key_values=cache, use_cache=True), device)
    finally:
        for hook in hooks: hook.remove()
    if any(x is None for x in hidden + queries + keys + values) or rotary[0] is None: raise RuntimeError("prefill capture incomplete")
    weights = []
    for layer in model.model.layers:
        a = layer.self_attn; weights.append({"q_proj": a.q_proj.weight, "k_proj": a.k_proj.weight, "v_proj": a.v_proj.weight, "o_proj": a.o_proj.weight, "q_norm": a.q_norm.weight, "k_norm": a.k_norm.weight})
    return PrefillCapture(cache, hidden, queries, keys, values, rotary[0], weights, elapsed, int(ids.shape[1]))


def _checksum(torch: Any, ops: TrustedOps, x: Any) -> tuple[int, int, int]:
    bits = ops.view(x.detach(), torch.int16) if x.element_size() == 2 else x.detach()
    dim = max(0, bits.ndim - 2); result = []
    for stride in (1, 7, 13):
        sl = [slice(None)] * bits.ndim; sl[dim] = slice(stride - 1, None, stride)
        result.append(int(ops.sum(ops.to(bits[tuple(sl)], torch.int64)).item()))
    return tuple(result)  # type: ignore[return-value]


def _checksums(torch: Any, ops: TrustedOps, values: Sequence[Any]) -> tuple[Any, ...]:
    result = []
    for value in values:
        if isinstance(value, torch.Tensor): result.append(_checksum(torch, ops, value))
        elif isinstance(value, (tuple, list)): result.append(_checksums(torch, ops, value))
        elif isinstance(value, dict): result.append(tuple((k, _checksum(torch, ops, value[k])) for k in sorted(value)))
        else: raise TypeError("unsupported checksum value")
    return tuple(result)


def _valid_indices(torch: Any, ops: TrustedOps, indices: Any, n_kv: int, n_kept: int, length: int) -> bool:
    if not isinstance(indices, torch.Tensor) or indices.dtype != torch.int64 or tuple(indices.shape) != (1, n_kv, n_kept) or indices.device.type != "cuda": return False
    if indices.numel() and (int(indices.min().item()) < 0 or int(indices.max().item()) >= length): return False
    ordered = ops.sort(indices, dim=-1).values
    return not bool((ordered[..., 1:] == ordered[..., :-1]).any().item())


def load_candidate(path: str | Path) -> tuple[ModuleType, type, str]:
    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink(): raise RuntimeError("candidate is not regular")
    digest = _sha256_file(candidate); spec = importlib.util.spec_from_file_location(f"kv_policy_{digest[:24]}", candidate)
    if spec is None or spec.loader is None: raise RuntimeError("candidate cannot load")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    cls = getattr(module, "EvictionPolicy", None)
    if not isinstance(cls, type): raise RuntimeError("candidate must define EvictionPolicy")
    return module, cls, digest


def _load_assets(torch: Any, ops: TrustedOps, path: str | Path | None) -> dict[str, Any]:
    if path is None or not Path(path).exists(): return {}
    if Path(path).stat().st_size > MAX_ASSET_BYTES: raise RuntimeError("assets exceed 64 MiB")
    assets = ops.load(Path(path), map_location="cuda", weights_only=True)
    if not isinstance(assets, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in assets.items()): raise RuntimeError("assets must be a string-to-tensor dict")
    if sum(v.numel() * v.element_size() for v in assets.values()) > MAX_ASSET_BYTES: raise RuntimeError("asset tensors exceed 64 MiB")
    return assets


def _warm_trusted_math(torch: Any, clock: TrustedClock, lengths: Sequence[int], spec: Mapping[str, Any]) -> None:
    """Initialize Torch contraction dispatch before untrusted timing starts."""
    device = torch.device("cuda"); heads = int(spec["n_heads"]); dim = int(spec["head_dim"])
    for length in sorted(set(int(x) for x in lengths)):
        x = torch.zeros((1, heads, length, dim), device=device, dtype=torch.bfloat16)
        mean = x.mean(dim=2, keepdim=True); centered = x - mean
        cov = torch.einsum("bnsi,bnsj->bnij", centered, centered) / length
        keys = x.transpose(2, 3); scores = torch.einsum("bhin,bhij,bhjn->bhn", keys, cov, keys)
        scores = torch.softmax(scores, dim=-1); torch.topk(scores, max(MIN_KEEP, length // 10), dim=-1)
        del x, mean, centered, cov, keys, scores
    # Reserve and release enough cached allocator space for an attention-class
    # policy; peak allocated memory is reset and measured independently per call.
    scratch = torch.empty((256 << 20,), device=device, dtype=torch.uint8)
    del scratch
    clock.synchronize(device)


def _budgets(policy: Any, length: int, n_layers: int, ratio: float) -> list[int]:
    uniform = int(length * (1.0 - ratio)); total = n_layers * uniform; value = policy.layer_budgets(length, total)
    if value is None: value = [uniform] * n_layers
    if not isinstance(value, (list, tuple)) or len(value) != n_layers or any(isinstance(x, bool) or not isinstance(x, int) for x in value): raise ValueError("invalid layer budgets")
    if any(x < MIN_KEEP or x > length for x in value) or sum(value) != total: raise ValueError("layer budgets violate bounds or total")
    return list(value)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}"); tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"); os.replace(tmp, path)


def _fresh_inputs(ops: TrustedOps, capture: PrefillCapture, layer_idx: int) -> tuple[Any, ...]:
    """New storage for every per-item tensor of one select call.

    Each (ratio, layer) call receives its own copies, so no call can recognise
    the tensors of an earlier call by identity or address and reuse its work.
    The read-only weights are shared: they identify the layer, not the item.
    """
    cos, sin = capture.position_embeddings
    return (ops.clone(capture.hidden_states[layer_idx]), ops.clone(capture.queries[layer_idx]),
            ops.clone(capture.keys[layer_idx]), ops.clone(capture.values[layer_idx]),
            (ops.clone(cos), ops.clone(sin)), capture.weights[layer_idx])


def decoy_seed(item_id: str) -> int:
    """Per-item seed of the warm-up tensors, so they do not depend on shard order."""
    return int.from_bytes(hashlib.sha256(f"kvpress-kv-eviction-v2-warmup|{item_id}".encode()).digest()[:8], "big") >> 1


def _decoy_inputs(ops: TrustedOps, capture: PrefillCapture, layer_idx: int, generator: Any) -> tuple[Any, ...]:
    """Standard-normal tensors with the shapes, strides and dtypes of a layer's item inputs (no item content); real position embeddings (content-free) and weights."""
    def rand(x: Any) -> Any:
        out = ops.empty_strided(tuple(x.size()), tuple(x.stride()), dtype=x.dtype, device=x.device); ops.normal_(out, generator=generator); return out
    cos, sin = capture.position_embeddings
    return (rand(capture.hidden_states[layer_idx]), rand(capture.queries[layer_idx]), rand(capture.keys[layer_idx]),
            rand(capture.values[layer_idx]), (ops.clone(cos), ops.clone(sin)), capture.weights[layer_idx])


def _select_layers(torch: Any, policy: Any, capture: PrefillCapture, ratio: float, *, clock: TrustedClock, ops: TrustedOps, measured: bool, decoy: Any = None) -> tuple[list[Any], list[int], list[float], list[int]]:
    """One ``select`` per layer.  With ``decoy`` (a generator) the inputs are random tensors of the item's shapes and the returned indices are not used."""
    budgets = _budgets(policy, capture.context_len, len(capture.keys), ratio); indices_cpu = []; times = []; deltas = []
    for layer_idx in range(len(budgets)):
        n_kept = budgets[layer_idx]; device = capture.keys[layer_idx].device
        abi = _fresh_inputs(ops, capture, layer_idx) if decoy is None else _decoy_inputs(ops, capture, layer_idx, decoy)
        before = _checksums(torch, ops, abi) if decoy is None else None; cpu_rng = ops.clone(ops.get_cpu_rng()); cuda_rng = ops.clone(ops.get_cuda_rng(device)); ops.reset_peak(device); baseline = int(ops.allocated(device))
        def invoke(): return policy.select(layer_idx, *abi, n_kept)
        if measured:
            elapsed, selected = clock.time_cuda(invoke, device)
        else:
            clock.synchronize(device); selected = invoke(); clock.synchronize(device); elapsed = 0.0
        _same_current_device(ops, device.index)
        times.append(elapsed); delta = max(0, int(ops.max_memory(device)) - baseline); deltas.append(delta)
        immutable_ok = decoy is not None or before == _checksums(torch, ops, abi)  # warm-up inputs are random: nothing to protect
        rng_ok = bool(ops.equal(cpu_rng, ops.get_cpu_rng())) and bool(ops.equal(cuda_rng, ops.get_cuda_rng(device)))
        valid = decoy is not None or _valid_indices(torch, ops, selected, int(capture.keys[layer_idx].shape[1]), n_kept, capture.context_len)
        if not immutable_ok: raise RuntimeError(f"input mutation at layer {layer_idx}")
        if not rng_ok: raise RuntimeError(f"RNG mutation at layer {layer_idx}")
        if not valid: raise RuntimeError(f"invalid indices at layer {layer_idx}")
        if delta > MEMORY_LIMIT_BYTES: raise RuntimeError(f"policy allocation exceeds {MEMORY_LIMIT_GIB} GiB at layer {layer_idx}")
        if decoy is None: indices_cpu.append(selected.detach().cpu())
        del abi, selected
    return indices_cpu, budgets, times, deltas


def run_phase_a(candidate_path: str | Path, assets_path: str | Path | None, items: Sequence[Mapping[str, Any]], ratios: Sequence[float], out_dir: str | Path, *, clock: TrustedClock, ops: TrustedOps) -> dict[str, Any]:
    """Unprivileged phase: context-only prefill, candidate selection, validated indices out.

    A fresh ``EvictionPolicy`` is constructed for every (item, ratio), so no
    policy instance carries state from one item to the next.  Per item: first
    an untimed warm-up pass of every ratio's policy on random tensors with the
    item's shapes (no item content), absorbing compilation and autotuning for
    the new length; then the timed passes on fresh copies of the item's
    tensors; each layer is charged the slowest of its timed ratio calls, so
    work shared across ratios saves no measured time.
    """
    import torch
    started = clock.perf_counter(); own = bind_device(torch); model = load_model(); tokenizer = load_tokenizer(); model_spec = _model_spec(model)
    selected_items = tuple(validate_context_item(x) for x in items)
    _warm_trusted_math(torch, clock, [WARMUP_CONTEXT_TOKENS], model_spec)
    token_generator = torch.Generator(device="cpu"); token_generator.manual_seed(0x4B565052455353)
    decoy_ids = torch.randint(0, int(model.config.vocab_size), (1, WARMUP_CONTEXT_TOKENS), generator=token_generator, dtype=torch.int64)
    decoy_capture = _capture_prefill(model, decoy_ids, clock)
    decoy_generator = torch.Generator(device="cuda")
    _own_device_only(ops, own)  # the trusted runtime itself stays on this GPU
    _module, policy_class, digest = load_candidate(candidate_path); assets = _load_assets(torch, ops, assets_path)
    for ratio in ratios:
        _select_layers(torch, policy_class(dict(model_spec), float(ratio), assets), decoy_capture, float(ratio), clock=clock, ops=ops, measured=False)
    del decoy_capture, decoy_ids; gc.collect(); _own_device_only(ops, own)
    root = Path(out_dir); root.mkdir(parents=True, exist_ok=True); result_rows = []
    for item in selected_items:
        encoded = preprocess_context_question(tokenizer, item["context"]); check_context_length(int(encoded["context_ids"].shape[1]))
        capture = _capture_prefill(model, encoded["context_ids"], clock); decoy_generator.manual_seed(decoy_seed(item["item_id"]))
        for ratio in ratios:
            _select_layers(torch, policy_class(dict(model_spec), float(ratio), assets), capture, float(ratio), clock=clock, ops=ops, measured=False, decoy=decoy_generator)
        timed = {}
        for ratio in ratios:
            ratio = float(ratio); policy = policy_class(dict(model_spec), ratio, assets)
            timed[ratio] = _select_layers(torch, policy, capture, ratio, clock=clock, ops=ops, measured=True); del policy
        _own_device_only(ops, own)
        charged = [max(timed[r][2][layer] for r in timed) for layer in range(len(capture.keys))]
        for ratio, (indices_cpu, budgets, times, deltas) in timed.items():
            tag = str(ratio); case_dir = root / item["item_id"]; case_dir.mkdir(parents=True, exist_ok=True)
            masks = [pack_selection(torch, x, capture.context_len) for x in indices_cpu]
            ops.save({"masks": masks, "budgets": budgets, "context_len": capture.context_len}, case_dir / f"{tag}.pt")
            telemetry = {"item_id": item["item_id"], "ratio": ratio, "indices_ok": True, "immutable_ok": True, "rng_ok": True, "budgets_ok": True,
                         "select_time_sum": math.fsum(charged), "select_time_max_layer": max(charged), "select_time_own": math.fsum(times),
                         "memory_delta": max(deltas), "layer_budgets": budgets, "masks_sha256": masks_digest(masks)}
            _atomic_json(case_dir / f"{tag}.json", telemetry); result_rows.append(telemetry)
        del capture, timed; gc.collect()
    return {"candidate_sha256": digest, "rows": result_rows, "phase_a_seconds": clock.perf_counter() - started}


_BIT_WEIGHTS = (128, 64, 32, 16, 8, 4, 2, 1)


def masks_digest(masks: Sequence[Any]) -> str:
    """SHA-256 of an (item, ratio)'s packed masks: a re-timed run must reproduce the selection bit for bit."""
    digest = hashlib.sha256()
    for mask in masks: digest.update(bytes(mask.numpy().tobytes()))
    return digest.hexdigest()


def pack_selection(torch: Any, indices: Any, length: int) -> Any:
    """Kept positions ``[1, n_kv, n]`` -> one bit per (head, position): uint8 ``[n_kv, ceil(length / 8)]``.

    The Phase A -> Phase B handoff stores these masks instead of int64
    indices: about 20x smaller (a large suite needs several GB)
    and canonical, since the replay gathers kept pairs in position order
    whatever order ``select`` returned them in.
    """
    n_kv = int(indices.shape[1]); mask = torch.zeros((n_kv, -(-length // 8) * 8), dtype=torch.uint8)
    mask.scatter_(1, indices[0].to(device="cpu", dtype=torch.int64), 1)
    return (mask.view(n_kv, -1, 8) * torch.tensor(_BIT_WEIGHTS, dtype=torch.uint8)).sum(-1, dtype=torch.uint8)


def unpack_selection(torch: Any, packed: Any, length: int, n_kv: int, n_kept: int) -> Any | None:
    """Inverse of ``pack_selection`` (ascending positions per head); ``None`` unless every head keeps exactly ``n_kept`` in-range positions."""
    if not isinstance(packed, torch.Tensor) or packed.dtype != torch.uint8 or tuple(packed.shape) != (n_kv, -(-length // 8)): return None
    bits = ((packed.unsqueeze(-1) & torch.tensor(_BIT_WEIGHTS, dtype=torch.uint8, device=packed.device)) != 0).view(n_kv, -1)
    if bool(bits[:, length:].any().item()) or not bool((bits.sum(-1) == n_kept).all().item()): return None
    return bits[:, :length].nonzero()[:, 1].view(1, n_kv, n_kept)


def _validate_saved(torch: Any, ops: TrustedOps, payload: Any, capture: PrefillCapture, ratio: float) -> tuple[list[Any], list[int]]:
    if not isinstance(payload, dict) or set(payload) != {"masks", "budgets", "context_len"}: raise RuntimeError("selection payload schema differs")
    masks, budgets = payload["masks"], payload["budgets"]; uniform = int(capture.context_len * (1 - ratio))
    if payload["context_len"] != capture.context_len: raise RuntimeError("selection context length differs")
    if not isinstance(masks, list) or len(masks) != len(capture.keys) or not isinstance(budgets, list) or len(budgets) != len(capture.keys) or sum(budgets) != len(budgets) * uniform: raise RuntimeError("saved layer set differs")
    indices = []
    for layer, (mask, budget) in enumerate(zip(masks, budgets)):
        n_kv = int(capture.keys[layer].shape[1])
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < MIN_KEEP or budget > capture.context_len: raise RuntimeError("saved budget invalid")
        idx = unpack_selection(torch, mask.to("cuda") if isinstance(mask, torch.Tensor) else mask, capture.context_len, n_kv, budget)
        if idx is None or not _valid_indices(torch, ops, idx, n_kv, budget, capture.context_len): raise RuntimeError("saved selection invalid")
        indices.append(idx)
    return indices, budgets


def _crop_cache(cache: Any, lengths: Sequence[int]) -> None:
    for i, length in enumerate(lengths): cache.layers[i].keys = cache.layers[i].keys[:, :, :length]; cache.layers[i].values = cache.layers[i].values[:, :, :length]


def _eos_ids(model: Any) -> tuple[int, ...]:
    raw = model.generation_config.eos_token_id
    values = raw if isinstance(raw, (list, tuple, set)) else (raw,)
    return tuple(int(value) for value in values if isinstance(value, int) and not isinstance(value, bool))


def _greedy_token(logits: Any, eos_ids: Sequence[int], n_generated: int) -> Any:
    scores = logits.float()
    if n_generated < R_MIN and eos_ids:
        scores = scores.clone(); scores[list(eos_ids)] = float("-inf")
    return scores.argmax()


def _layer_mask(torch: Any, n_past: int, n_new: int, device: Any) -> Any:
    mask = torch.zeros((n_new, n_past + n_new), device=device, dtype=torch.bfloat16); mask[:, n_past:] = torch.triu(torch.full((n_new, n_new), float("-inf"), device=device, dtype=torch.bfloat16), diagonal=1); return mask[None, None]


def _teacher_forced_logprobs(model: Any, cache: Any, context_len: int, question_ids: Any, ref_tokens: Any) -> Any:
    import torch
    device = next(model.parameters()).device; q = question_ids.to(device); ref = ref_tokens.to(device); combined = torch.cat((q, ref.view(1, -1)), dim=1); n_new = int(combined.shape[1]); original = [cache.get_seq_length(i) for i in range(len(cache))]
    position_ids = torch.arange(context_len, context_len + n_new, device=device).unsqueeze(0); masks = {length: _layer_mask(torch, length, n_new, device) for length in set(original)}; hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        n_past = original[layer_idx]
        def pre_hook(_module: Any, args: Any, kwargs: Any, *, past=n_past): kwargs["attention_mask"] = masks[past]; return args, kwargs
        hooks.append(layer.self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True))
    try:
        output = model(input_ids=combined, past_key_values=cache, position_ids=position_ids, use_cache=True)
        start = int(q.shape[1]) - 1; return output.logits[0, start:start + int(ref.numel())].float().log_softmax(-1).cpu()
    finally:
        for hook in hooks: hook.remove()
        _crop_cache(cache, original)


def _reference_continuation(model: Any, capture: PrefillCapture, question_ids: Any, R: int = R_MAX) -> tuple[Any, Any]:
    import torch
    if isinstance(R, bool) or not isinstance(R, int) or not R_MIN <= R <= R_MAX: raise ValueError(f"reference cap must be in [{R_MIN},{R_MAX}]")
    device = next(model.parameters()).device; q = question_ids.to(device); cache = capture.cache; original = [cache.get_seq_length(i) for i in range(len(cache))]
    position_ids = torch.arange(capture.context_len, capture.context_len + q.shape[1], device=device).unsqueeze(0); tokens = []; eos = _eos_ids(model); current = q
    try:
        for n_generated in range(R):
            out = model(input_ids=current, past_key_values=cache, position_ids=position_ids, logits_to_keep=1, use_cache=True)
            token = _greedy_token(out.logits[0, -1], eos, n_generated); tokens.append(token)
            if len(tokens) >= R_MIN and int(token.item()) in eos: break
            current = token.view(1, 1); position_ids = position_ids[:, -1:] + 1
    finally: _crop_cache(cache, original)
    ref_tokens = torch.stack(tokens).cpu()
    return ref_tokens, _teacher_forced_logprobs(model, cache, capture.context_len, question_ids, ref_tokens)


def _compressed_fidelity(model: Any, capture: PrefillCapture, indices: Sequence[Any], question_ids: Any, ref_tokens: Any, ref_lp: Any) -> dict[str, Any]:
    import torch
    from transformers import DynamicCache
    device = next(model.parameters()).device; gathered = []
    for layer_idx, raw in enumerate(indices):
        idx = raw.to(device=device, dtype=torch.int64); idx4 = idx.unsqueeze(-1).expand(-1, -1, -1, capture.keys[layer_idx].shape[-1]); gathered.append((capture.keys[layer_idx].gather(2, idx4).contiguous(), capture.values[layer_idx].gather(2, idx4).contiguous()))
    cache = DynamicCache(ddp_cache_data=gathered); ref = ref_tokens.to(device); n_ref = int(ref.numel()); comp_lp = _teacher_forced_logprobs(model, cache, capture.context_len, question_ids, ref_tokens).to(device); full_lp = ref_lp.to(device=device, dtype=torch.float32); probability = full_lp.exp(); kl = (probability * (full_lp - comp_lp)).sum(-1); mean_kl = kl.mean()
    if not bool(torch.isfinite(mean_kl).item()): raise FloatingPointError("non-finite KL")
    column = ref.view(-1, 1); return {"fid": float(torch.exp(-mean_kl).item()), "mean_kl": float(mean_kl.item()), "top1": float((full_lp.argmax(-1) == comp_lp.argmax(-1)).float().mean().item()), "nll_delta": float((full_lp.gather(1, column) - comp_lp.gather(1, column)).mean().item()), "n_tokens": n_ref}


def run_phase_b(items: Sequence[Mapping[str, Any]], ratios: Sequence[float], indices_dir: str | Path, *, clock: TrustedClock, ops: TrustedOps) -> dict[str, Any]:
    """Trusted phase: never imports candidate code; replays every saved selection."""
    import torch
    started = clock.perf_counter(); model = load_model(); tokenizer = load_tokenizer(); rows = []; root = Path(indices_dir)
    for raw in items:
        item = validate_item(raw); encoded = preprocess_context_question(tokenizer, item["context"], item["question"], item["answer_prefix"])
        check_context_length(int(encoded["context_ids"].shape[1]))
        capture = _plain_prefill(model, encoded["context_ids"], clock); ref_tokens, ref_lp = _reference_continuation(model, capture, encoded["question_ids"])
        for raw_ratio in ratios:
            ratio = float(raw_ratio); payload = ops.load(root / item["item_id"] / f"{ratio}.pt", map_location="cpu", weights_only=True); indices, _ = _validate_saved(torch, ops, payload, capture, ratio); metrics = _compressed_fidelity(model, capture, indices, encoded["question_ids"], ref_tokens, ref_lp)
            telemetry = json.loads((root / item["item_id"] / f"{ratio}.json").read_text())
            if telemetry.get("item_id") != item["item_id"] or float(telemetry.get("ratio", -1)) != ratio: raise RuntimeError("Phase-A telemetry identity differs")
            rows.append({**telemetry, **metrics, "family": item["family"], "task": item["task"], "prefill_s": capture.prefill_s, "context_tokens": capture.context_len})
        del capture, ref_tokens, ref_lp; gc.collect()
    return {"rows": rows, "phase_b_seconds": clock.perf_counter() - started}


def fidelity_noise_floor(item: Mapping[str, Any]) -> dict[str, Any]:
    """Score identity indices through the compressed path as a fidelity self-check."""
    import torch
    clock = TrustedClock.capture(torch); ops = TrustedOps.capture(torch); hardware = hardware_inventory(torch); model = load_model(); tokenizer = load_tokenizer(); item = validate_item(item)
    encoded = preprocess_context_question(tokenizer, item["context"], item["question"], item["answer_prefix"]); capture = _capture_prefill(model, encoded["context_ids"], clock); ref_tokens, ref_lp = _reference_continuation(model, capture, encoded["question_ids"]); indices = []
    for keys in capture.keys:
        base = ops.arange(capture.context_len, device=keys.device, dtype=torch.int64); indices.append(base.view(1, 1, -1).expand(1, int(keys.shape[1]), -1))
    metrics = _compressed_fidelity(model, capture, indices, encoded["question_ids"], ref_tokens, ref_lp)
    return {"schema_version": 1, "item_id": item["item_id"], "family": item["family"], "ratio": 0.0, "hardware": hardware, **metrics}


def run_public_suite(candidate_path: str | Path, assets_path: str | Path | None, items: Sequence[Mapping[str, Any]] | None = None, ratios: Sequence[float] = RATIOS, *, items_jsonl: str | Path | None = None) -> dict[str, Any]:
    import shutil, tempfile, torch
    if items_jsonl is not None:
        if items is not None: raise ValueError("pass items or items_jsonl, not both")
        items = load_items_jsonl(items_jsonl)
    if items is None: items = PUBLIC_ITEMS
    if not items: raise ValueError("public suite requires items")
    items = tuple(validate_item(x) for x in items)
    clock = TrustedClock.capture(torch); ops = TrustedOps.capture(torch); hardware = hardware_inventory(torch); root = Path(tempfile.mkdtemp(prefix="kv-eviction-public-"))
    try:
        a = run_phase_a(candidate_path, assets_path, [context_only(x) for x in items], ratios, root, clock=clock, ops=ops)
        b = run_phase_b(items, ratios, root, clock=clock, ops=ops); summary = aggregate(b["rows"])
        return {"schema_version": 1, "generator_provenance": GENERATOR_PROVENANCE, "suite_id": PUBLIC_SUITE_ID, "candidate_sha256": a["candidate_sha256"], "hardware": hardware, "ratios": list(ratios), "families": list(FAMILIES), "items": b["rows"], "task_fid": task_breakdown(b["rows"]), **summary, "phase_a_seconds": a["phase_a_seconds"], "phase_b_seconds": b["phase_b_seconds"]}
    finally: shutil.rmtree(root, ignore_errors=True)
