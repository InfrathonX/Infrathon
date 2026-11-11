#!/usr/bin/env python3
"""Trusted two-phase runtime for query-agnostic KV importance scoring."""
from __future__ import annotations

import gc
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

# Caching-allocator policy for every process that runs this runtime (Phase-A
# worker, Phase-B verifier, public runner).  It is set here, at import time and
# before torch is imported anywhere, because the CUDA caching allocator reads
# PYTORCH_CUDA_ALLOC_CONF once, at its first allocation.
#
# v1 needed expandable segments: its Phase B let `DynamicCache` re-materialise
# the whole batched key/value tensor at every decode step, and the stale
# blocks pushed reserved memory to whatever the card had.  v2 decodes into a
# preallocated per-chunk cache (tokens written in place), so that growth is
# gone, and expandable segments are now switched off explicitly: they back
# allocations with many small cuMemCreate/cuMemMap calls, which PyroMind's HAMi
# hook serialises across every process in the Judge.  On the 8-GPU PyroMind
# Judge (2026-10-05) Phase B ran 2.2x slower than locally and short prefills
# 11.7x slower, while kvpress (default allocator) matched local speed.
#
# This is an allocator policy only.  Every tensor shape, kernel, dtype and
# reduction order is unchanged, so the decoded answers are bit-identical; the
# caching allocator still frees its cached blocks and retries before reporting
# an out-of-memory error.  Any other options a deployment sets are preserved,
# and this one is appended last so it wins.
#
# The 80 GB H100 SXM build turns them back on: with them off, Phase B on a
# 79.18 GiB card ran out of memory holding 67.6 GiB allocated plus 5.9 GiB of
# reserved fragments; with them on, the reserved peak
# tracks the live set (74.9 GiB on NVL).  These hosts are bare metal (no HAMi).
ALLOCATOR_CONF = "expandable_segments:True"


def _apply_allocator_conf(environ: Any = os.environ) -> str:
    existing = str(environ.get("PYTORCH_CUDA_ALLOC_CONF", "")).strip()
    parts = [part for part in existing.split(",") if part.strip() and not part.strip().startswith("expandable_segments")]
    parts.append(ALLOCATOR_CONF)
    environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(parts)
    return environ["PYTORCH_CUDA_ALLOC_CONF"]


_apply_allocator_conf()

MODEL_ID = "Qwen/Qwen3-8B"
GENERATOR_PROVENANCE = "kvzip-5d84729-fastkvzip-e04afaa-scoring-v2"
RATIOS = (0.1, 0.2, 0.3)
GROUPS = ("retrieval", "contextual_qa", "redundancy")
# Qwen3-8B @ b968826d has max_position_embeddings = 40960; every item keeps
# system prompt + context + question + generation cap inside that window.
MAX_CTX_TOKENS = 32000
MIN_CTX_TOKENS = 64
MODEL_WINDOW_TOKENS = 40960
OVERHEAD_FRACTION = 0.25
OVERHEAD_ITEM_CAP = 0.50
# Per-item absolute floor, expressed relative to the device: the shortest plain
# prefill measured in the same suite times this multiple.  On the H100 NVL
# qualification host the shortest prefills are ~25 ms, so the floor is ~0.25 s
# there; on a slower or faster GPU it scales with the device instead of
# silently tightening or loosening the gate.
OVERHEAD_FLOOR_PREFILL_MULTIPLE = 10.0


def overhead_item_floor_s(rows) -> float:
    """Device-relative absolute floor for one item's score time."""
    prefills = [float(row.get("prefill_s", 0)) for row in rows if float(row.get("prefill_s", 0)) > 0]
    return OVERHEAD_FLOOR_PREFILL_MULTIPLE * min(prefills) if prefills else 0.0
MAX_DECODE_BATCH = 16
# Phase B decodes (condition, question) pairs in chunks.  A chunk's size is a
# pure function of the item's token counts -- never of the device -- so the
# decoded answers are identical on every qualified GPU.  Bf16 GEMM kernel
# choice depends on the batch shape, so this rule is part of the contract.
DECODE_KV_BUDGET_BYTES = 40 << 30
MEMORY_LIMIT_GIB = 8
MEMORY_LIMIT_BYTES = MEMORY_LIMIT_GIB << 30
MAX_ASSET_BYTES = 128 << 20
MAX_ASSET_EXPANDED_BYTES = 256 << 20
WARMUP_CONTEXT_LENGTHS = (2048, 128)
PUBLIC_SUITE_ID = "kvzip-scoring-public-v2"
FINAL_SUITE_ID = "final-holdout-v2"
EOS_TOKEN_ID = 151645
SYSTEM_QA = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "Given the context, answer to the following question or request without explanation.\n\n"
)
SYSTEM_GSM = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "Given the context, answer to the following reasoning question.\n\n"
)
POSTFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


@dataclass(frozen=True)
class TaskSpec:
    group: str
    metric: str
    max_new_tokens: int
    format: str = "qa"


TASKS = {
    # retrieval
    "needle": TaskSpec("retrieval", "include", 48),
    "scbench_kv": TaskSpec("retrieval", "include", 96),
    "scbench_prefix_suffix": TaskSpec("retrieval", "include", 128),
    "scbench_repoqa": TaskSpec("retrieval", "repoqa", 512),
    "ruler_niah": TaskSpec("retrieval", "ruler_all", 64),
    # contextual understanding
    "squad": TaskSpec("contextual_qa", "include", 256),
    "gsm": TaskSpec("contextual_qa", "gsm", 512),
    "ruler_qa": TaskSpec("contextual_qa", "ruler_part", 32),
    # high redundancy
    "scbench_mf": TaskSpec("redundancy", "exact", 48),
    "scbench_many_shot": TaskSpec("redundancy", "many_shot", 96),
    "scbench_summary": TaskSpec("redundancy", "rouge_l", 256, "summary"),
    "ruler_vt": TaskSpec("redundancy", "ruler_all", 48),
}


def _answer_ok(value: Any) -> bool:
    """A reference is a string, or a non-empty list of strings (RULER multi-value answers)."""
    if isinstance(value, str): return True
    return isinstance(value, list) and bool(value) and all(isinstance(x, str) and x for x in value)


def validate_item(raw: Mapping[str, Any], tokenizer: Any | None = None) -> dict[str, Any]:
    if not isinstance(raw, Mapping): raise ValueError("item must be an object")
    item_id = raw.get("item_id"); task = raw.get("task"); context = raw.get("context")
    questions = raw.get("questions"); answers = raw.get("answers"); meta = raw.get("meta", {})
    if not isinstance(item_id, str) or not item_id: raise ValueError("item_id must be non-empty")
    if task not in TASKS: raise ValueError(f"unknown task: {task}")
    if not isinstance(context, str) or not context: raise ValueError("context must be non-empty")
    if not isinstance(questions, list) or not questions or not all(isinstance(q, str) and q for q in questions): raise ValueError("questions must be non-empty strings")
    if not isinstance(answers, list) or len(answers) != len(questions) or not all(_answer_ok(a) for a in answers): raise ValueError("answers must align with questions")
    if not isinstance(meta, Mapping): raise ValueError("meta must be an object")
    if not isinstance(meta.get("answer_prefix", ""), str): raise ValueError("meta.answer_prefix must be a string")
    if TASKS[task].metric == "repoqa":
        repo = meta.get("repoqa")
        if not isinstance(repo, Mapping) or not isinstance(repo.get("lang"), str) or not isinstance(repo.get("repo"), str): raise ValueError("repoqa item needs meta.repoqa.lang/repo")
        names, needles = repo.get("func_names"), repo.get("needles")
        if not isinstance(names, list) or not isinstance(needles, list) or len(names) != len(questions) or len(needles) != len(questions): raise ValueError("repoqa needles must align with questions")
    if tokenizer is not None:
        n_tokens = len(tokenizer.encode(context, add_special_tokens=False))
        if n_tokens < MIN_CTX_TOKENS: raise ValueError(f"context has fewer than {MIN_CTX_TOKENS} tokens")
        if n_tokens > MAX_CTX_TOKENS: raise ValueError(f"context exceeds {MAX_CTX_TOKENS} tokens")
    elif isinstance(meta.get("context_tokens"), int) and not isinstance(meta["context_tokens"], bool):
        if meta["context_tokens"] < MIN_CTX_TOKENS: raise ValueError(f"context has fewer than {MIN_CTX_TOKENS} tokens")
        if meta["context_tokens"] > MAX_CTX_TOKENS: raise ValueError(f"context exceeds {MAX_CTX_TOKENS} tokens")
    return {"item_id": item_id, "task": task, "context": context, "questions": list(questions), "answers": list(answers), "meta": dict(meta)}


def prompt_kind(task: str) -> str:
    return "reasoning" if task.startswith("gsm") else "qa"


def context_only(item: Mapping[str, Any]) -> dict[str, str]:
    """What the unprivileged Phase-A process receives: no questions, no answers."""
    return {"item_id": str(item["item_id"]), "prompt": prompt_kind(str(item["task"])), "context": str(item["context"])}


def validate_context_item(raw: Mapping[str, Any]) -> dict[str, str]:
    if not isinstance(raw, Mapping) or set(raw) != {"item_id", "prompt", "context"}: raise ValueError("Phase-A item fields differ")
    if not isinstance(raw["item_id"], str) or not raw["item_id"]: raise ValueError("item_id must be non-empty")
    if raw["prompt"] not in ("qa", "reasoning"): raise ValueError("unknown prompt kind")
    if not isinstance(raw["context"], str) or not raw["context"]: raise ValueError("context must be non-empty")
    return {"item_id": raw["item_id"], "prompt": raw["prompt"], "context": raw["context"]}


def _context_tokens(item: Mapping[str, Any]) -> int:
    tokens = item.get("meta", {}).get("context_tokens")
    return tokens if isinstance(tokens, int) and not isinstance(tokens, bool) else len(str(item["context"])) // 4


def item_cost(item: Mapping[str, Any]) -> float:
    """Relative Judge cost, used only to balance shards: prefill plus batched decoding."""
    tokens = _context_tokens(item); spec = TASKS[str(item["task"])]; decode = len(item["questions"]) * 4 * spec.max_new_tokens * (1.0 + tokens / 16384.0)
    return float(tokens) / 4.0 + decode


def handoff_bytes(items: Sequence[Mapping[str, Any]]) -> int:
    """Upper bound on the Phase A -> Phase B files: FP32 scores per (layer, KV head, context token) plus framing (Qwen3-8B: 36 layers x 8 KV heads)."""
    return sum(36 * 8 * 4 * (_context_tokens(x) + 64) + (1 << 16) for x in items)


def _read_lines(path: Path) -> list[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle: return handle.read().splitlines()
    return path.read_text(encoding="utf-8").splitlines()


def load_items_jsonl(path: str | Path, tokenizer: Any | None = None) -> tuple[dict[str, Any], ...]:
    seen: set[str] = set(); rows = []
    for lineno, line in enumerate(_read_lines(Path(path)), 1):
        if not line.strip(): continue
        try: row = validate_item(json.loads(line), tokenizer)
        except Exception as exc: raise ValueError(f"invalid item line {lineno}: {exc}") from exc
        if row["item_id"] in seen: raise ValueError(f"duplicate item_id: {row['item_id']}")
        seen.add(row["item_id"]); rows.append(row)
    if not rows: raise ValueError("item file is empty")
    return tuple(rows)


def context_sha256(context: str) -> str:
    return hashlib.sha256(context.encode("utf-8")).hexdigest()


def item_sha256(item: Mapping[str, Any]) -> str:
    """Identity of one scored row: context, questions and references."""
    blob = json.dumps({"context": item["context"], "questions": item["questions"], "answers": item["answers"]}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def validate_final_items(rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any], public_items: Sequence[Mapping[str, Any]] = ()) -> tuple[dict[str, Any], ...]:
    """Check materialised holdout rows against the frozen manifest, row by row."""
    if manifest.get("schema_version") != 1 or manifest.get("suite_id") != FINAL_SUITE_ID: raise ValueError("final manifest identity differs")
    entries = manifest.get("items"); counts = manifest.get("counts_per_task")
    if not isinstance(entries, list) or not entries or not isinstance(counts, Mapping): raise ValueError("final manifest schema differs")
    items = tuple(validate_item(x) for x in rows)
    if len(items) != len(entries): raise ValueError("final row count differs from the manifest")
    for item, entry in zip(items, entries):
        if not isinstance(entry, Mapping) or (item["item_id"], item["task"]) != (entry.get("item_id"), entry.get("task")):
            raise ValueError(f"final row identity differs at {entry.get('item_id') if isinstance(entry, Mapping) else entry!r}")
        if item_sha256(item) != entry.get("sha256"): raise ValueError(f"final row text differs at {item['item_id']}")
    actual = {task: sum(x["task"] == task for x in items) for task in TASKS}
    if {str(k): int(v) for k, v in counts.items()} != {k: v for k, v in actual.items() if v}: raise ValueError("final counts differ")
    if len({x["item_id"] for x in items}) != len(items): raise ValueError("duplicate final item_id")
    # Rows must differ; a context may recur only with different questions
    # (upstream ICL.ManyShot asks 54 rows of questions over 3 demonstration banks).
    if len({item_sha256(x) for x in items}) != len(items): raise ValueError("duplicate final row")
    if {context_sha256(x["context"]) for x in items} & {context_sha256(x["context"]) for x in public_items}: raise ValueError("public/final context overlap")
    return items


def _load_public_default() -> tuple[dict[str, Any], ...]:
    root = Path(os.environ.get("KV_SCORING_CORPUS_DIR", str(Path(__file__).resolve().parents[1] / "corpus/public")))
    path = root / "items_public.jsonl"
    try: return load_items_jsonl(path)
    except (FileNotFoundError, ValueError): return ()


PUBLIC_ITEMS = _load_public_default()


def suite_items(name: str, items: Sequence[Mapping[str, Any]] | None = None) -> tuple[dict[str, Any], ...]:
    pool = tuple(items) if items is not None else PUBLIC_ITEMS
    if name == "one":
        chosen = next((x for x in pool if x["task"] == "squad"), pool[0] if pool else None)
        return (chosen,) if chosen else ()
    if name == "quick": return tuple(x for task in TASKS for x in [y for y in pool if y["task"] == task][:2])
    if name in {"full", "public"}: return pool
    raise ValueError("unknown suite")


def _encode(tokenizer: Any, text: str, device: Any | None = None):
    ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
    return ids.to(device) if device is not None else ids


def assemble_prefill(tokenizer: Any, context: str, task: str = "qa", device: Any | None = None) -> dict[str, Any]:
    """``task`` is a task name or a prompt kind; GSM rows use the reasoning instruction."""
    prefix = SYSTEM_GSM if task.startswith("gsm") or task == "reasoning" else SYSTEM_QA
    sys_ids = _encode(tokenizer, prefix, device); ctx_ids = _encode(tokenizer, context, device)
    prefill = __import__("torch").cat([sys_ids, ctx_ids], dim=1)
    return {"prefill_ids": prefill, "ctx_range": (int(sys_ids.shape[1]), int(prefill.shape[1]))}


def assemble_question(tokenizer: Any, question: str, device: Any | None = None, answer_prefix: str = ""):
    """Question as a user turn, then the assistant turn; RULER rows start the answer with their prefix."""
    import torch
    parts = [_encode(tokenizer, f"\n\n{question.strip()}", device), _encode(tokenizer, POSTFIX, device)]
    if answer_prefix: parts.append(_encode(tokenizer, answer_prefix, device))
    return torch.cat(parts, dim=1)


@dataclass(frozen=True)
class TrustedOps:
    cat: Callable[..., Any]; stack: Callable[..., Any]; argsort: Callable[..., Any]
    sum: Callable[..., Any]; equal: Callable[..., Any]; isfinite: Callable[..., Any]
    clone: Callable[..., Any]; get_cpu_rng: Callable[..., Any]; get_cuda_rng: Callable[..., Any]
    max_memory: Callable[..., Any]; reset_peak: Callable[..., Any]; allocated: Callable[..., Any]
    save: Callable[..., Any]; load: Callable[..., Any]
    empty_strided: Callable[..., Any]; normal_: Callable[..., Any]
    current_device: Callable[..., Any]; device_count: Callable[..., Any]
    reserved: Callable[..., Any]; max_reserved: Callable[..., Any]

    @classmethod
    def capture(cls, torch: Any) -> "TrustedOps":
        return cls(torch.cat, torch.stack, torch.argsort, torch.sum, torch.equal, torch.isfinite,
                   torch.clone, torch.get_rng_state, torch.cuda.get_rng_state,
                   torch.cuda.max_memory_allocated, torch.cuda.reset_peak_memory_stats,
                   torch.cuda.memory_allocated, torch.save, torch.load,
                   torch.empty_strided, torch.Tensor.normal_,
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
        self.synchronize(device); wall = self.perf_counter()
        start = self.event(enable_timing=True); end = self.event(enable_timing=True)
        stream = self.current_stream(device); stream.record_event(start); value = fn(); stream.record_event(end)
        self.synchronize(device)
        return max(self.perf_counter() - wall, float(start.elapsed_time(end)) / 1000.0), value


# Capability envelope: what the task contract actually needs, not the GPU model
# it was qualified on.  The GPU name and UUID are recorded as provenance only;
# they never gate admission.  Keep these two constants identical in the Judge
# verifier (tests assert the pair).
MIN_COMPUTE_CAPABILITY = (8, 0)
# 79 GiB, not 80: the 80 GB H100 SXM card reports 85,019,590,656 B to torch.
MIN_DEVICE_MEMORY_BYTES = 79 * 1024**3


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
    return {"name": name, "count": 1, "device_index": index, "capability": list(capability), "uuid": str(getattr(props, "uuid", "")), "total_memory_bytes": total_memory_bytes, "min_compute_capability": list(MIN_COMPUTE_CAPABILITY), "min_device_memory_bytes": MIN_DEVICE_MEMORY_BYTES, "torch_version": str(torch.__version__), "torch_cuda_version": str(torch.version.cuda), "allocator_conf": str(os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))}


def load_tokenizer():
    from transformers import AutoTokenizer
    model_dir = os.environ.get("KV_SCORING_MODEL_DIR")
    if not model_dir: raise RuntimeError("KV_SCORING_MODEL_DIR is required")
    return AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)


def load_model(device: str = "cuda"):
    import torch
    from transformers import AutoModelForCausalLM
    model_dir = os.environ.get("KV_SCORING_MODEL_DIR")
    if not model_dir: raise RuntimeError("KV_SCORING_MODEL_DIR is required")
    model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True, device_map={"": bind_device(torch)} if str(device).startswith("cuda") else None)
    if not str(device).startswith("cuda"): model = model.to(device)
    model.eval(); model.requires_grad_(False)
    return model


def _model_spec(model: Any) -> dict[str, Any]:
    c = model.config
    return {"model_id": MODEL_ID, "n_layers": int(c.num_hidden_layers), "n_heads": int(c.num_attention_heads),
        "n_kv_heads": int(c.num_key_value_heads), "head_dim": int(getattr(c, "head_dim", c.hidden_size // c.num_attention_heads)),
        "hidden_size": int(c.hidden_size), "rope_theta": float(c.rope_theta), "ratio_ladder": tuple(RATIOS), "max_ctx_tokens": MAX_CTX_TOKENS}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""): digest.update(chunk)
    return digest.hexdigest()


def load_candidate(path: str | Path) -> tuple[ModuleType, type, str]:
    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink(): raise RuntimeError("candidate is not regular")
    digest = _sha256_file(candidate); spec = importlib.util.spec_from_file_location(f"kv_scorer_{digest[:24]}", candidate)
    if spec is None or spec.loader is None: raise RuntimeError("candidate cannot load")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    cls = getattr(module, "KVScorer", None)
    if not isinstance(cls, type): raise RuntimeError("candidate must define KVScorer")
    return module, cls, digest


def _load_assets(torch: Any, ops: TrustedOps, path: str | Path | None) -> dict[str, Any]:
    if path is None: return {}
    asset = Path(path)
    if not asset.is_file() or asset.is_symlink() or asset.stat().st_size > MAX_ASSET_BYTES: raise RuntimeError("assets file is unsafe or too large")
    with asset.open("rb") as handle: compressed = handle.read(2) == b"\x1f\x8b"
    source: Any = asset
    if compressed:
        payload = bytearray()
        with gzip.open(asset, "rb") as stream:
            while chunk := stream.read(min(1 << 20, MAX_ASSET_EXPANDED_BYTES + 1 - len(payload))):
                payload.extend(chunk)
                if len(payload) > MAX_ASSET_EXPANDED_BYTES: raise RuntimeError("compressed asset expands beyond 256 MiB")
        source = io.BytesIO(payload)
    values = ops.load(source, map_location="cuda", weights_only=True)
    if not isinstance(values, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in values.items()): raise RuntimeError("assets must be a string-to-tensor dict")
    if sum(v.numel() * v.element_size() for v in values.values()) > MAX_ASSET_EXPANDED_BYTES: raise RuntimeError("asset tensors exceed 256 MiB expanded")
    return values


@dataclass
class PrefillCapture:
    cache: Any
    hidden_states: list[Any]
    queries: list[Any]
    keys: list[Any]
    values: list[Any]
    position_embeddings: tuple[Any, Any]
    weights: list[dict[str, Any]]
    prefill_s: float
    ctx_range: tuple[int, int]


def _plain_prefill(model: Any, prefill_ids: Any, ctx_range: tuple[int, int], clock: TrustedClock) -> PrefillCapture:
    from transformers import DynamicCache
    device = next(model.parameters()).device; ids = prefill_ids.to(device); cache = DynamicCache()
    elapsed, _ = clock.time_cuda(lambda: model.model(input_ids=ids, past_key_values=cache, use_cache=True), device)
    keys = [layer.keys for layer in cache.layers]; values = [layer.values for layer in cache.layers]
    if len(keys) != int(model.config.num_hidden_layers): raise RuntimeError("plain prefill cache incomplete")
    return PrefillCapture(cache, [], [], keys, values, (None, None), [], elapsed, ctx_range)


def _capture_prefill(model: Any, prefill_ids: Any, ctx_range: tuple[int, int], clock: TrustedClock) -> PrefillCapture:
    from transformers import DynamicCache
    device = next(model.parameters()).device; ids = prefill_ids.to(device); layers = model.model.layers
    hidden = [None] * len(layers); queries = [None] * len(layers); rotary = [None]; hooks = []
    for idx, layer in enumerate(layers):
        def pre_hook(_module, args, kwargs, i=idx):
            hidden[i] = kwargs.get("hidden_states", args[0] if args else None).detach()
            if rotary[0] is None:
                pe = kwargs["position_embeddings"]; rotary[0] = (pe[0].detach(), pe[1].detach())
        def q_hook(_module, _args, output, i=idx): queries[i] = output.detach().transpose(1, 2)
        hooks.append(layer.self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True))
        hooks.append(layer.self_attn.q_norm.register_forward_hook(q_hook))
    cache = DynamicCache()
    try: elapsed, _ = clock.time_cuda(lambda: model.model(input_ids=ids, past_key_values=cache, use_cache=True), device)
    finally:
        for hook in hooks: hook.remove()
    keys = [layer.keys.detach() for layer in cache.layers]; values = [layer.values.detach() for layer in cache.layers]
    if any(x is None for x in hidden + queries) or rotary[0] is None or len(keys) != len(layers): raise RuntimeError("prefill capture incomplete")
    weights = []
    for layer in layers:
        a = layer.self_attn
        weights.append({"q_proj": a.q_proj.weight, "k_proj": a.k_proj.weight, "v_proj": a.v_proj.weight,
            "o_proj": a.o_proj.weight, "q_norm": a.q_norm.weight, "k_norm": a.k_norm.weight})
    return PrefillCapture(cache, hidden, queries, keys, values, rotary[0], weights, elapsed, ctx_range)


def _checksum(torch: Any, ops: TrustedOps, x: Any) -> tuple[int, int, int]:
    bits = x.detach().view(torch.int16) if x.element_size() == 2 else x.detach()
    dim = max(0, bits.ndim - 2); result = []
    for stride in (1, 7, 13):
        sl = [slice(None)] * bits.ndim; sl[dim] = slice(stride - 1, None, stride)
        result.append(int(ops.sum(bits[tuple(sl)].to(torch.int64)).item()))
    return tuple(result)


def _checksums(torch: Any, ops: TrustedOps, values: Sequence[Any]) -> tuple[Any, ...]:
    result = []
    for value in values:
        if isinstance(value, torch.Tensor): result.append(_checksum(torch, ops, value))
        elif isinstance(value, (tuple, list)): result.append(_checksums(torch, ops, value))
        elif isinstance(value, dict): result.append(tuple((k, _checksum(torch, ops, value[k])) for k in sorted(value)))
        else: raise TypeError("unsupported checksum value")
    return tuple(result)


def _valid_score(torch: Any, ops: TrustedOps, score: Any, n_kv: int, n_ctx: int, device: Any) -> bool:
    return bool(isinstance(score, torch.Tensor) and score.dtype == torch.float32 and tuple(score.shape) == (1, n_kv, n_ctx)
        and score.device == device and bool(ops.isfinite(score).all().item()))


def _fresh_copy(torch: Any, ops: TrustedOps, value: Any) -> Any:
    """Same values and strides in new storage, so the copy shares no address with the original."""
    if isinstance(value, torch.Tensor): return ops.clone(value)
    if isinstance(value, tuple): return tuple(_fresh_copy(torch, ops, x) for x in value)
    raise TypeError("unsupported scorer input")


def decoy_seed(item_id: str) -> int:
    """Per-row seed of the warm-up tensors, so they do not depend on shard order."""
    return int.from_bytes(hashlib.sha256(f"kvzip-scoring-v2-warmup|{item_id}".encode()).digest()[:8], "big") >> 1


def _decoy_inputs(torch: Any, ops: TrustedOps, abi: tuple[Any, ...], generator: Any) -> tuple[Any, ...]:
    """Standard-normal tensors with the shapes, strides and dtypes of a layer's row inputs (no row content); real position embeddings (content-free) and weights."""
    def rand(x: Any) -> Any:
        out = ops.empty_strided(tuple(x.size()), tuple(x.stride()), dtype=x.dtype, device=x.device); ops.normal_(out, generator=generator); return out
    hidden, queries, keys, values, (cos, sin), weights = abi
    return (rand(hidden), rand(queries), rand(keys), rand(values), (ops.clone(cos), ops.clone(sin)), weights)


def _score_layers(torch: Any, scorer: Any, capture: PrefillCapture, *, clock: TrustedClock, ops: TrustedOps, measured: bool, generator: Any) -> tuple[list[Any], list[float], list[int], list[bool]]:
    """Score every layer in three passes; only the middle pass is timed.

    1. warm-up: each layer on random tensors with the row's shapes and strides
       (no row content), untimed; absorbs compilation, autotuning and
       allocator growth for the new length;
    2. measured: each layer on the row's own tensors, timed; these scores are kept;
    3. check: each layer on fresh copies of the row's tensors (new storage),
       untimed; the scores must match pass 2 bit for bit (determinism telemetry).

    No untimed call sees row content before every timed call has returned, so
    neither a cache nor scoring work moved into an untimed call can shorten
    the measured time.  The read-only weights are shared; they identify the
    layer, not the row.
    """
    start, end = capture.ctx_range; n_ctx = end - start; n_layers = len(capture.keys); deltas = [0] * n_layers
    abis = [(capture.hidden_states[i], capture.queries[i], capture.keys[i], capture.values[i], capture.position_embeddings, capture.weights[i]) for i in range(n_layers)]

    def invoke_checked(layer_idx: int, abi: tuple[Any, ...], *, timed: bool, validate: bool) -> tuple[float, Any]:
        device = abi[2].device
        before = _checksums(torch, ops, abi) if validate else None; cpu_rng = ops.clone(ops.get_cpu_rng()); cuda_rng = ops.clone(ops.get_cuda_rng(device))
        invoke = lambda: scorer.score(layer_idx, *abi, capture.ctx_range)
        ops.reset_peak(device); baseline = int(ops.allocated(device))
        if timed: elapsed, score = clock.time_cuda(invoke, device)
        else:
            clock.synchronize(device); score = invoke(); clock.synchronize(device); elapsed = 0.0
        _same_current_device(ops, device.index)
        delta = max(0, int(ops.max_memory(device)) - baseline)
        if validate and before != _checksums(torch, ops, abi): raise RuntimeError(f"input mutation at layer {layer_idx}")  # warm-up inputs are random: nothing to protect
        if not ops.equal(cpu_rng, ops.get_cpu_rng()) or not ops.equal(cuda_rng, ops.get_cuda_rng(device)): raise RuntimeError(f"RNG mutation at layer {layer_idx}")
        if validate and not _valid_score(torch, ops, score, int(capture.keys[layer_idx].shape[1]), n_ctx, device): raise RuntimeError(f"invalid score at layer {layer_idx}")
        if delta > MEMORY_LIMIT_BYTES: raise RuntimeError(f"scorer allocation exceeds {MEMORY_LIMIT_GIB} GiB at layer {layer_idx}")
        deltas[layer_idx] = max(deltas[layer_idx], delta)
        return elapsed, score

    for layer_idx, abi in enumerate(abis):
        decoy = _decoy_inputs(torch, ops, abi, generator); invoke_checked(layer_idx, decoy, timed=False, validate=False); del decoy
    scores = []; times = []
    for layer_idx, abi in enumerate(abis):
        elapsed, score = invoke_checked(layer_idx, abi, timed=measured, validate=True)
        scores.append(ops.clone(score.detach()).cpu()); times.append(elapsed); del score
    deterministic = []
    for layer_idx, abi in enumerate(abis):
        copy = (*(_fresh_copy(torch, ops, x) for x in abi[:5]), abi[5])
        _elapsed, score = invoke_checked(layer_idx, copy, timed=False, validate=True)
        deterministic.append(bool(ops.equal(scores[layer_idx], score.detach().cpu()))); del score, copy
    return scores, times, deltas, deterministic


def _warm_trusted_math(torch: Any, clock: TrustedClock, spec: Mapping[str, Any]) -> None:
    device = torch.device("cuda")
    x = torch.zeros((1, int(spec["n_kv_heads"]), 2048), device=device, dtype=torch.float32)
    torch.argsort(-x.reshape(-1), stable=True); torch.topk(x, 128, dim=-1)
    scratch = torch.empty((256 << 20,), device=device, dtype=torch.uint8); del scratch, x
    clock.synchronize(device)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def _warm_candidate(torch: Any, model: Any, scorer: Any, generator: Any, decoy_generator: Any, *, clock: TrustedClock, ops: TrustedOps) -> None:
    """Exercise both long- and short-context scorer shapes outside measurement."""
    for n_context in WARMUP_CONTEXT_LENGTHS:
        decoy_ids = torch.randint(0, int(model.config.vocab_size), (1, n_context), generator=generator, dtype=torch.int64)
        decoy = _capture_prefill(model, decoy_ids, (0, n_context), clock); decoy_generator.manual_seed(n_context)
        _score_layers(torch, scorer, decoy, clock=clock, ops=ops, measured=False, generator=decoy_generator)
        del decoy, decoy_ids
        gc.collect()


def run_phase_a(candidate_path: str | Path, assets_path: str | Path | None, items: Sequence[Mapping[str, Any]], out_dir: str | Path, *, clock: TrustedClock, ops: TrustedOps) -> dict[str, Any]:
    """Unprivileged phase: context-only prefill, candidate scores, validated score files out.

    Items arrive without questions or answers.  A fresh ``KVScorer`` is built
    for every item, so no scorer instance carries state from one item to the next.
    """
    import torch
    started = clock.perf_counter(); own = bind_device(torch); model = load_model(); tokenizer = load_tokenizer(); spec = _model_spec(model)
    selected = tuple(validate_context_item(x) for x in items)
    _warm_trusted_math(torch, clock, spec)
    generator = torch.Generator(device="cpu"); generator.manual_seed(0x4B565A4950); decoy_generator = torch.Generator(device="cuda")
    _own_device_only(ops, own)  # the trusted runtime itself stays on this GPU
    _module, scorer_class, digest = load_candidate(candidate_path); assets = _load_assets(torch, ops, assets_path)
    _warm_candidate(torch, model, scorer_class(dict(spec), assets), generator, decoy_generator, clock=clock, ops=ops)
    _own_device_only(ops, own)
    root = Path(out_dir); root.mkdir(parents=True, exist_ok=True); rows = []
    for item in selected:
        n_tokens = len(tokenizer.encode(item["context"], add_special_tokens=False))
        if not MIN_CTX_TOKENS <= n_tokens <= MAX_CTX_TOKENS: raise ValueError(f"context has {n_tokens} tokens, outside [{MIN_CTX_TOKENS}, {MAX_CTX_TOKENS}]")
        assembled = assemble_prefill(tokenizer, item["context"], item["prompt"])
        capture = _capture_prefill(model, assembled["prefill_ids"], assembled["ctx_range"], clock)
        scorer = scorer_class(dict(spec), assets); decoy_generator.manual_seed(decoy_seed(item["item_id"]))
        scores, times, deltas, deterministic = _score_layers(torch, scorer, capture, clock=clock, ops=ops, measured=True, generator=decoy_generator)
        _own_device_only(ops, own)
        stacked = ops.stack(scores); ops.save(stacked, root / f"{item['item_id']}.pt")
        telemetry = {"item_id": item["item_id"],
            "score_time_sum": math.fsum(times), "score_time_max_layer": max(times), "memory_delta": max(deltas),
            "scores_ok": True, "immutable_ok": True, "rng_ok": True, "scores_deterministic": all(deterministic),
            "scores_sha256": scores_digest(stacked)}
        _atomic_json(root / f"{item['item_id']}.json", telemetry); rows.append(telemetry)
        del capture, scores, scorer; gc.collect()
    return {"candidate_sha256": digest, "rows": rows, "phase_a_seconds": clock.perf_counter() - started}


def scores_digest(scores: Any) -> str:
    """SHA-256 of a row's stacked CPU scores: a re-timed run must reproduce them bit for bit."""
    return hashlib.sha256(bytes(scores.contiguous().numpy().tobytes())).hexdigest()


def global_masks(scores: Any, ratio: float) -> list[Any]:
    import torch
    if not isinstance(scores, torch.Tensor) or scores.ndim != 4 or scores.shape[1] != 1 or not bool(torch.isfinite(scores).all()): raise ValueError("scores must be finite [layers,1,kv,context]")
    flat = scores.reshape(-1); n_keep = round(float(ratio) * flat.numel())
    if not 0 <= n_keep <= flat.numel(): raise ValueError("ratio is outside [0,1]")
    order = torch.argsort(-flat, stable=True); valid = torch.zeros(flat.numel(), dtype=torch.bool, device=flat.device); valid[order[:n_keep]] = True
    shaped = valid.view_as(scores).squeeze(1)
    return [shaped[i] for i in range(shaped.shape[0])]


def _eos_ids(model: Any) -> tuple[int, ...]:
    raw = model.generation_config.eos_token_id; values = raw if isinstance(raw, (list, tuple, set)) else (raw,)
    return tuple(sorted({EOS_TOKEN_ID, *(int(x) for x in values if isinstance(x, int) and not isinstance(x, bool))}))


DECODE_ATTENTION = "kvzip_v2_decode"
_DECODE_REGISTERED = False


def _context_kv_masks(model: Any, mask_sets: Sequence[Sequence[Any] | None], pair_conditions: Sequence[int], n_ctx: int, device: Any) -> list[Any]:
    """Per-layer context masks for one chunk, one row per pair and KV head: ``[batch, n_kv, n_ctx]``."""
    import torch
    n_kv = int(model.config.num_key_value_heads); out = []
    for layer_idx in range(int(model.config.num_hidden_layers)):
        batch_values = []
        for condition_index in pair_conditions:
            masks = mask_sets[condition_index]
            value = torch.ones((n_kv, n_ctx), device=device, dtype=torch.bool) if masks is None else masks[layer_idx].to(device=device, dtype=torch.bool)
            if tuple(value.shape) != (n_kv, n_ctx): raise RuntimeError("context mask shape differs")
            batch_values.append(value)
        out.append(torch.stack(batch_values, dim=0))
    return out


def _decode_attention(module: Any, query: Any, key: Any, value: Any, attention_mask: Any, dropout: float = 0.0, scaling: float | None = None, **kwargs: Any):
    """Masked attention for Phase-B decoding.

    A single-token step attends per KV head: the query heads that share a KV
    head become rows of one ``[batch, n_kv, group, head_dim]`` query, so the
    keys and values are read once instead of being repeated to every query
    head.  The mask is ``[batch, n_kv, 1, total]`` and broadcasts over the
    group.  Multi-token forwards (the question) use the standard SDPA path.
    """
    import torch
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    if query.shape[2] == 1 and attention_mask is not None and attention_mask.ndim == 4 and attention_mask.shape[1] == key.shape[1] != query.shape[1]:
        batch, n_heads, _, head_dim = query.shape; n_kv = key.shape[1]
        grouped = query.reshape(batch, n_kv, n_heads // n_kv, head_dim)
        out = torch.nn.functional.scaled_dot_product_attention(grouped, key, value, attn_mask=attention_mask, scale=scaling)
        return out.reshape(batch, n_heads, 1, head_dim).transpose(1, 2).contiguous(), None
    return sdpa_attention_forward(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs)


def _register_decode_attention() -> None:
    global _DECODE_REGISTERED
    if _DECODE_REGISTERED: return
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
    AttentionInterface.register(DECODE_ATTENTION, _decode_attention)
    AttentionMaskInterface.register(DECODE_ATTENTION, sdpa_mask)
    _DECODE_REGISTERED = True


def _chunk_cache(cache: Any, batch: int, capacity: int):
    """A cache whose layers are preallocated ``[batch, n_kv, capacity, head_dim]`` buffers.

    The shared context is copied in once per chunk; every later token is
    written in place and attention reads a view of the filled prefix, so no
    decode step re-materialises the batched cache.
    """
    from transformers import DynamicCache
    from transformers.cache_utils import DynamicLayer

    class ChunkLayer(DynamicLayer):
        def __init__(self, keys: Any, values: Any) -> None:
            super().__init__()
            n_kv, length, head_dim = keys.shape[1], keys.shape[2], keys.shape[3]
            self.key_buffer = keys.new_empty((batch, n_kv, capacity, head_dim)); self.value_buffer = values.new_empty((batch, n_kv, capacity, head_dim))
            self.key_buffer[:, :, :length].copy_(keys.expand(batch, -1, -1, -1)); self.value_buffer[:, :, :length].copy_(values.expand(batch, -1, -1, -1))
            self.length = length; self.dtype, self.device, self.is_initialized = keys.dtype, keys.device, True
            self.keys = self.key_buffer[:, :, :length]; self.values = self.value_buffer[:, :, :length]

        def update(self, key_states: Any, value_states: Any, cache_kwargs: Any = None):
            end = self.length + int(key_states.shape[-2])
            if end > capacity: raise RuntimeError("decode cache capacity exceeded")
            self.key_buffer[:, :, self.length:end].copy_(key_states); self.value_buffer[:, :, self.length:end].copy_(value_states)
            self.length = end; self.keys = self.key_buffer[:, :, :end]; self.values = self.value_buffer[:, :, :end]
            return self.keys, self.values

        def get_seq_length(self) -> int:
            return self.length

    chunk = DynamicCache()
    chunk.layers = [ChunkLayer(layer.keys, layer.values) for layer in cache.layers]
    return chunk


def kv_bytes_per_token(config: Any) -> int:
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    return int(config.num_hidden_layers) * int(config.num_key_value_heads) * head_dim * 2 * 2  # K and V, bf16


def decode_chunk_size(bytes_per_token: int, prefill_len: int, max_question_len: int, max_new_tokens: int) -> int:
    """Pairs decoded together: as many as fit DECODE_KV_BUDGET_BYTES of expanded cache, at most 16."""
    per_sequence = bytes_per_token * (prefill_len + max_question_len + max_new_tokens)
    return max(1, min(MAX_DECODE_BATCH, DECODE_KV_BUDGET_BYTES // per_sequence))


def answer_questions(model: Any, tokenizer: Any, cache: Any, ctx_range: tuple[int, int], mask_sets: Sequence[Sequence[Any] | None], questions: Sequence[str], max_new_tokens: int, answer_prefix: str = "") -> dict[str, Any]:
    """Greedy answers for every (condition, question) pair, decoded in deterministic chunks.

    Condition 0 is the full cache; the others mask context KV per layer and
    KV head.  The system prompt and every appended token stay visible.
    """
    import torch
    if not questions or not mask_sets: return {"predictions": [[] for _ in mask_sets], "decode_steps_total": 0, "generated_tokens_total": 0, "decode_chunk_size": 0}
    _register_decode_attention()
    device = next(model.parameters()).device; sys_len, prefill_len = ctx_range; n_ctx = prefill_len - sys_len
    n_kv = int(model.config.num_key_value_heads); n_heads = int(model.config.num_attention_heads); group = n_heads // n_kv
    question_ids = [assemble_question(tokenizer, question, device, answer_prefix) for question in questions]
    qmax_all = max(int(q.shape[1]) for q in question_ids)
    if prefill_len + qmax_all + max_new_tokens > MODEL_WINDOW_TOKENS: raise ValueError("item exceeds the model window")
    pairs = [(condition_index, question_index) for condition_index in range(len(mask_sets)) for question_index in range(len(question_ids))]
    predictions = [[""] * len(question_ids) for _ in mask_sets]
    decode_steps_total = 0; generated_tokens_total = 0
    chunk_size = decode_chunk_size(kv_bytes_per_token(model.config), prefill_len, qmax_all, max_new_tokens)
    previous_attention = model.config._attn_implementation
    model.config._attn_implementation = DECODE_ATTENTION
    try:
        for chunk_start in range(0, len(pairs), chunk_size):
            chunk = pairs[chunk_start:chunk_start + chunk_size]; batch = len(chunk)
            lengths = torch.tensor([question_ids[question_index].shape[1] for _, question_index in chunk], device=device)
            qmax = int(lengths.max().item()); qbatch = torch.zeros((batch, qmax), device=device, dtype=torch.long)
            qreal = torch.zeros((batch, qmax), device=device, dtype=torch.bool); position_ids = torch.zeros((batch, qmax), device=device, dtype=torch.long)
            for index, (_condition_index, question_index) in enumerate(chunk):
                question = question_ids[question_index]; size = int(question.shape[1]); qbatch[index, :size] = question[0]; qreal[index, :size] = True
                position_ids[index, :size] = torch.arange(prefill_len, prefill_len + size, device=device)
            batch_cache = _chunk_cache(cache, batch, prefill_len + qmax + max_new_tokens)
            context_masks = _context_kv_masks(model, mask_sets, [condition_index for condition_index, _ in chunk], n_ctx, device)
            system = torch.ones((batch, n_kv, sys_len), device=device, dtype=torch.bool)
            appended = {"valid": qreal.clone()}; hooks = []
            for layer_idx, layer in enumerate(model.model.layers):
                def mask_hook(_module, args, kwargs, i=layer_idx):
                    hidden = kwargs.get("hidden_states", args[0] if args else None); qn = int(hidden.shape[1]); old = int(batch_cache.get_seq_length(i)); total = old + qn
                    base = torch.cat([system, context_masks[i], appended["valid"].unsqueeze(1).expand(-1, n_kv, -1)], dim=-1)
                    if int(base.shape[-1]) != total: raise RuntimeError(f"mask/cache mismatch {base.shape[-1]} != {total}")
                    if qn == 1:
                        kwargs["attention_mask"] = base.unsqueeze(2)
                    else:
                        causal = torch.arange(total, device=device).view(1, 1, 1, total) <= (old + torch.arange(qn, device=device)).view(1, 1, qn, 1)
                        kwargs["attention_mask"] = base.repeat_interleave(group, 1).unsqueeze(2) & causal
                    return args, kwargs
                hooks.append(layer.self_attn.register_forward_pre_hook(mask_hook, with_kwargs=True))
            generated = [[] for _ in range(batch)]; finished = torch.zeros(batch, dtype=torch.bool, device=device); eos = set(_eos_ids(model))
            try:
                logits = model(input_ids=qbatch, position_ids=position_ids, past_key_values=batch_cache, use_cache=True).logits
                decode_steps_total += 1; row = torch.arange(batch, device=device); next_token = logits[row, lengths - 1].float().argmax(-1)
                for step in range(max_new_tokens):
                    for index, token in enumerate(next_token.tolist()):
                        if not bool(finished[index]):
                            if token in eos: finished[index] = True
                            else: generated[index].append(token); generated_tokens_total += 1
                    if bool(finished.all()) or step + 1 == max_new_tokens: break
                    active = ~finished; appended["valid"] = torch.cat([appended["valid"], active[:, None]], dim=1)
                    logical_pos = (prefill_len + lengths + step).view(batch, 1)
                    logits = model(input_ids=next_token.view(batch, 1), position_ids=logical_pos, past_key_values=batch_cache, use_cache=True).logits
                    decode_steps_total += 1; next_token = logits[:, -1].float().argmax(-1)
            finally:
                for hook in hooks: hook.remove()
            for index, (condition_index, question_index) in enumerate(chunk):
                predictions[condition_index][question_index] = tokenizer.decode(generated[index], skip_special_tokens=False)
            del batch_cache, context_masks, appended, qbatch, qreal, position_ids, logits, next_token, generated
    finally:
        model.config._attn_implementation = previous_attention
    return {"predictions": predictions, "decode_steps_total": decode_steps_total, "generated_tokens_total": generated_tokens_total, "decode_chunk_size": chunk_size}


def _metric_module():
    try: import kv_scoring_metric as metric
    except ImportError:
        runtime_dir = Path(__file__).resolve().parent
        if str(runtime_dir) not in sys.path: sys.path.insert(0, str(runtime_dir))
        import kv_scoring_metric as metric
    return metric


def run_phase_b(items: Sequence[Mapping[str, Any]], ratios: Sequence[float], scores_dir: str | Path, *, clock: TrustedClock, ops: TrustedOps) -> dict[str, Any]:
    import torch
    metric = _metric_module(); started = clock.perf_counter(); model = load_model(); tokenizer = load_tokenizer()
    rows = []; root = Path(scores_dir)
    for raw in items:
        item_started = clock.perf_counter()
        item = validate_item(raw, tokenizer); spec = TASKS[item["task"]]; assembled = assemble_prefill(tokenizer, item["context"], item["task"])
        capture = _plain_prefill(model, assembled["prefill_ids"], assembled["ctx_range"], clock)
        scores = ops.load(root / f"{item['item_id']}.pt", map_location="cpu", weights_only=True)
        n_ctx = capture.ctx_range[1] - capture.ctx_range[0]
        expected = (int(model.config.num_hidden_layers), 1, int(model.config.num_key_value_heads), n_ctx)
        if not isinstance(scores, torch.Tensor) or scores.dtype != torch.float32 or tuple(scores.shape) != expected or not bool(torch.isfinite(scores).all()): raise RuntimeError("saved scores invalid")
        mask_sets = [None, *(global_masks(scores, float(ratio)) for ratio in ratios)]
        decoded = answer_questions(model, tokenizer, capture.cache, capture.ctx_range, mask_sets, item["questions"], spec.max_new_tokens, str(item["meta"].get("answer_prefix", "")))
        full_preds = decoded["predictions"][0]
        full_values = metric.score_item(full_preds, item, spec.metric, spec.format); acc_full = sum(full_values) / len(full_values)
        acc = {}; predictions = {}
        for condition_index, ratio in enumerate(ratios, 1):
            preds = decoded["predictions"][condition_index]
            values = metric.score_item(preds, item, spec.metric, spec.format)
            acc[str(float(ratio))] = sum(values) / len(values); predictions[str(float(ratio))] = preds
        clock.synchronize(next(model.parameters()).device); item_seconds = clock.perf_counter() - item_started
        telemetry = json.loads((root / f"{item['item_id']}.json").read_text())
        if telemetry.get("item_id") != item["item_id"]: raise RuntimeError("Phase-A telemetry identity differs")
        rows.append({**telemetry, "task": item["task"], "group": spec.group, "prefill_s": capture.prefill_s, "context_tokens": n_ctx,
            "acc": acc, "acc_full": acc_full,
            "predictions": predictions, "full_predictions": full_preds, "answers": item["answers"],
            "phase_b_item_seconds": item_seconds, "decode_steps_total": decoded["decode_steps_total"],
            "generated_tokens_total": decoded["generated_tokens_total"], "decode_chunk_size": decoded["decode_chunk_size"]})
        del capture, scores; gc.collect()
    return {"rows": rows, "phase_b_seconds": clock.perf_counter() - started}


def aggregate(rows: Sequence[Mapping[str, Any]], ratios: Sequence[float] = RATIOS) -> dict[str, Any]:
    present_tasks = tuple(task for task in TASKS if any(row.get("task") == task for row in rows))
    task_acc_full = {}; task_acc = {str(float(r)): {} for r in ratios}; rel = {str(float(r)): {} for r in ratios}
    for task in present_tasks:
        selected = [row for row in rows if row.get("task") == task]
        task_acc_full[task] = math.fsum(float(x.get("acc_full", math.nan)) for x in selected) / len(selected)
        for ratio in ratios:
            key = str(float(ratio)); task_acc[key][task] = math.fsum(float(x.get("acc", {}).get(key, math.nan)) for x in selected) / len(selected)
            rel[key][task] = task_acc[key][task] / task_acc_full[task] if task_acc_full[task] > 0 else 0.0
    group_rel = {str(float(r)): {} for r in ratios}; ratio_score = {}
    for ratio in ratios:
        key = str(float(ratio))
        for group in GROUPS:
            vals = [rel[key][task] for task in present_tasks if TASKS[task].group == group]
            if vals: group_rel[key][group] = math.fsum(vals) / len(vals)
        ratio_score[key] = math.fsum(group_rel[key].values()) / len(group_rel[key]) if group_rel[key] else 0.0
    primary = 100.0 * math.fsum(ratio_score.values()) / len(ratios) if ratios else 0.0
    prefill_total = math.fsum(float(row.get("prefill_s", 0)) for row in rows); score_total = math.fsum(float(row.get("score_time_sum", math.inf)) for row in rows)
    item_ratios = [float(row.get("score_time_sum", math.inf)) / float(row.get("prefill_s", 0)) if float(row.get("prefill_s", 0)) > 0 else math.inf for row in rows]
    overhead_suite = score_total / prefill_total if prefill_total > 0 else math.inf; overhead_item = max(item_ratios, default=math.inf)
    overhead_floor_s = overhead_item_floor_s(rows)
    overhead_item_violations = sum(float(row.get("score_time_sum", math.inf)) > max(OVERHEAD_ITEM_CAP * float(row.get("prefill_s", 0)), overhead_floor_s) for row in rows)
    deterministic_items = sum(row.get("scores_deterministic") is True for row in rows)
    task_phase_b_seconds = {task: math.fsum(float(row.get("phase_b_item_seconds", 0.0)) for row in rows if row.get("task") == task) for task in present_tasks}
    metric_ok = all(math.isfinite(float(x)) for x in [*task_acc_full.values(), *ratio_score.values(), primary]) and all(math.isfinite(v) for d in task_acc.values() for v in d.values())
    reference_ok = bool(present_tasks) and all(task_acc_full[x] > 0 for x in present_tasks)
    gates = {
        "scores_ok": bool(rows) and all(row.get("scores_ok") is True for row in rows),
        "immutable_ok": bool(rows) and all(row.get("immutable_ok") is True for row in rows),
        "rng_ok": bool(rows) and all(row.get("rng_ok") is True for row in rows),
        "overhead_ok": overhead_suite <= OVERHEAD_FRACTION and overhead_item_violations == 0,
        "memory_ok": bool(rows) and all(int(row.get("memory_delta", MEMORY_LIMIT_BYTES + 1)) <= MEMORY_LIMIT_BYTES for row in rows),
        "metric_ok": metric_ok, "reference_ok": reference_ok,
    }
    return {"groups": list(GROUPS), "tasks": list(present_tasks), "task_acc": task_acc, "task_acc_full": task_acc_full,
        "rel": rel, "group_rel": group_rel, "ratio_score": ratio_score, "primary": primary,
        "score_rel_auc": primary, "overhead_ratio_suite": overhead_suite, "overhead_ratio_item_max": overhead_item,
        "overhead_item_violations": overhead_item_violations, "overhead_item_floor_s": overhead_floor_s, "deterministic_items": deterministic_items,
        "task_phase_b_seconds": task_phase_b_seconds, "gates": gates}


def run_public_suite(candidate_path: str | Path, assets_path: str | Path | None, items: Sequence[Mapping[str, Any]] | None = None, ratios: Sequence[float] = RATIOS, *, items_jsonl: str | Path | None = None) -> dict[str, Any]:
    import torch
    if items_jsonl is not None:
        if items is not None: raise ValueError("pass items or items_jsonl, not both")
        items = load_items_jsonl(items_jsonl)
    if items is None: items = PUBLIC_ITEMS
    if not items: raise ValueError("public suite requires items")
    clock = TrustedClock.capture(torch); ops = TrustedOps.capture(torch); hardware = hardware_inventory(torch)
    root = Path(tempfile.mkdtemp(prefix="kv-scoring-public-"))
    try:
        items = tuple(validate_item(x) for x in items)
        phase_a = run_phase_a(candidate_path, assets_path, [context_only(x) for x in items], root, clock=clock, ops=ops)
        phase_b = run_phase_b(items, ratios, root, clock=clock, ops=ops); summary = aggregate(phase_b["rows"], ratios)
        return {"schema_version": 1, "generator_provenance": GENERATOR_PROVENANCE, "suite_id": PUBLIC_SUITE_ID, "candidate_sha256": phase_a["candidate_sha256"],
            "hardware": hardware, "ratios": list(ratios), "groups": list(GROUPS), "items": phase_b["rows"], **summary,
            "phase_a_seconds": phase_a["phase_a_seconds"], "phase_b_seconds": phase_b["phase_b_seconds"]}
    finally: shutil.rmtree(root, ignore_errors=True)
