#!/usr/bin/env python3
"""Trusted W-rank runtime for the Flux-derived MoE AG + scatter + grouped GEMM operator.

One process per GPU (eight in total) runs the whole suite.  A case runs on a
process group of W ranks (W = 2..8, possibly a cross-NUMA or shuffled
placement); processes outside the group wait.  Symmetric buffers are pooled
per group, so each group performs a handful of rendezvous for the whole
suite.  This file is byte-identical in the Work image (public tests) and the
Judge worker; the suites themselves come from the case manifest.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, Sequence

ENTRYPOINT = "moe_ag_scatter_gemm_bf16"
ENTRYPOINTS = (ENTRYPOINT,)
PARAMETERS = (
    "x_local",
    "x_peers",
    "w",
    "sorted_token_ids",
    "sorted_expert_ids",
    "expert_offsets",
    "token_offsets",
    "out",
    "workspace",
    "peer_workspaces",
    "peer_ptrs",
    "workspace_ptrs",
    "rank",
    "world_size",
)
WORLD_PROCESSES = 8
FAMILIES = ("tp8", "tp4", "tp2", "ep", "decode", "sequence")
TIERS = ("paper", "model", "real")
EVIDENCE_TIERS = ("paper", "real")
EDGE_FAMILY = "edge"
EDGE_TIER = "edge"
# Placement groups.  processes: global ranks (one per GPU, sorted); order[q] is
# the index into processes of the process that plays logical rank q.
GROUPS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "w8": ((0, 1, 2, 3, 4, 5, 6, 7), (0, 1, 2, 3, 4, 5, 6, 7)),
    "w8s": ((0, 1, 2, 3, 4, 5, 6, 7), (5, 2, 7, 0, 3, 6, 1, 4)),
    "w6": ((0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5)),
    "w4": ((0, 1, 2, 3), (0, 1, 2, 3)),
    "w4x": ((0, 1, 4, 5), (0, 1, 2, 3)),
    "w4s": ((0, 1, 2, 3), (2, 0, 3, 1)),
    "w3": ((0, 1, 2), (0, 1, 2)),
    "w2": ((0, 1), (0, 1)),
    "w2x": ((0, 4), (0, 1)),
}
GROUP_ORDER = tuple(GROUPS)
ROUTING_KINDS = ("uniform", "prod", "skew", "hot", "segment", "captured", "forced")
SPLITS = ("even", "uneven")

WORKSPACE_BYTES = 4 << 30
POOL_ALIGN = 1 << 21
SHIFT_SLOTS = 8
SHIFT_BYTES = 1 << 20
MAX_WEIGHT_BYTES = 12 << 30
MAX_RESIDENT_BYTES = 60 << 30
MAX_K, MAX_N, MAX_E_RANK, MAX_TOPK, MAX_LAYERS = 16384, 16384, 512, 16, 64
GENERATOR_PROVENANCE = "flux-19831ca-derived-moe-ag-scatter-v2-world"

# Accuracy gates.  G3-G6 compare against an FP32 golden with limits relative
# to an ordinary BF16 framework reference on the same inputs; G7 is a
# Freivalds random-projection check run on every call.
L1_MULT, L1_FLOOR = 1.25, 1.0e-3
P999_MULT, P999_FLOOR = 1.25, 1.0e-3
ROWMAX_MULT, ROWMAX_FLOOR = 1.25, 1.0e-3
COS_FLOOR, ROW_NORM_FLOOR = 0.9999965, 1.0e-6
FREIVALDS_REL_L2, FREIVALDS_ROW_FRACTION = 2.0e-2, 0.2
GATED_TRIALS = 2
# Processes outside a placement group wait at a world barrier while it runs.
COLLECTIVE_TIMEOUT_SECONDS = 4 * 3600

ROLE_SALT = {"warm": 0xA11C_0000, "decoy": 0xDEC0_0000, "actual": 0x7A1A_0000, "twin": 0x7E1E_0000}


# --------------------------------------------------------------------------- utilities


def finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise RuntimeError(f"{label} must be finite and positive")
    return result


def geometric_mean(values: Iterable[float]) -> float:
    numbers = [finite_positive(v, "geometric-mean input") for v in values]
    if not numbers:
        raise ValueError("geometric mean requires values")
    return math.exp(math.fsum(math.log(v) for v in numbers) / len(numbers))


def mix_seed(*parts: int) -> int:
    digest = hashlib.sha256(",".join(str(int(p)) for p in parts).encode()).digest()
    return int.from_bytes(digest[:6], "little")


def _int(raw: Mapping[str, Any], key: str, lo: int, hi: int) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise ValueError(f"case field {key} outside [{lo}, {hi}]: {value!r}")
    return int(value)


# --------------------------------------------------------------------------- case schema

CASE_KEYS = {
    "case_id", "family", "tier", "group", "W", "TP", "EP", "K", "N", "E", "topk", "M",
    "split", "split_cv", "routing", "layers", "seed", "source", "model",
}
ROUTING_KEYS = {
    "uniform": {"kind"},
    "prod": {"kind", "sigma_e", "lam", "seg_len"},
    "skew": {"kind", "sigma_e", "lam", "seg_len"},
    "hot": {"kind", "sigma_e", "hot", "hot_boost"},
    "segment": {"kind", "sigma_e", "lam", "seg_len"},
    "captured": {"kind", "fixture", "layer", "start", "sha256"},
    "forced": {"kind", "expert"},
}
FIXTURE_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789.-")
HEX_DIGITS = frozenset("0123456789abcdef")
EDGE_OVERRIDES = {"n_massive", "massive_scale"}


def token_split(M: int, W: int, split: str, cv: float, seed: int) -> list[int]:
    """Tokens per logical rank; every rank owns at least one token."""
    if split == "even":
        base, extra = divmod(M, W)
        return [base + (1 if q < extra else 0) for q in range(W)]
    rng = random.Random(mix_seed(seed, 0x5917))
    sigma = math.sqrt(math.log1p(cv * cv))
    weights = [math.exp(rng.gauss(-0.5 * sigma * sigma, sigma)) for _ in range(W)]
    total = sum(weights)
    counts = [max(1, int(M * x / total)) for x in weights]
    diff = M - sum(counts)
    q = 0
    while diff != 0:
        idx = q % W
        if diff > 0:
            counts[idx] += 1
            diff -= 1
        elif counts[idx] > 1:
            counts[idx] -= 1
            diff += 1
        q += 1
    return counts


def validate_case(raw: Mapping[str, Any]) -> dict[str, Any]:
    keys = set(raw) - EDGE_OVERRIDES
    if keys != CASE_KEYS:
        raise ValueError(f"case fields differ from schema: {sorted(keys ^ CASE_KEYS)}")
    case = dict(raw)
    family, tier, group = str(case["family"]), str(case["tier"]), str(case["group"])
    edge = family == EDGE_FAMILY
    if edge != (tier == EDGE_TIER):
        raise ValueError("edge family and tier must agree")
    if not edge and (family not in FAMILIES or tier not in TIERS):
        raise ValueError("unknown family or tier")
    if group not in GROUPS:
        raise ValueError("unknown placement group")
    W = _int(case, "W", 2, WORLD_PROCESSES)
    if len(GROUPS[group][0]) != W:
        raise ValueError("group size differs from W")
    TP = _int(case, "TP", 1, W)
    EP = _int(case, "EP", 1, W)
    if TP * EP != W:
        raise ValueError("TP * EP must equal W")
    K = _int(case, "K", 64, MAX_K)
    N = _int(case, "N", 8, MAX_N)
    E = _int(case, "E", 1, MAX_E_RANK * WORLD_PROCESSES)
    topk = _int(case, "topk", 1, MAX_TOPK)
    M = _int(case, "M", W, 1 << 20)
    layers = _int(case, "layers", 1, MAX_LAYERS)
    _int(case, "seed", 0, (1 << 48) - 1)
    if K % 64 or N % 8 or E % EP or topk > E:
        raise ValueError("geometry outside the support domain")
    e_rank = E // EP
    if e_rank > MAX_E_RANK:
        raise ValueError("experts per rank above 512")
    if (layers > 1) != (family == "sequence"):
        raise ValueError("only sequence cases have more than one layer")
    split = str(case["split"])
    cv = case["split_cv"]
    if split not in SPLITS or isinstance(cv, bool) or not isinstance(cv, (int, float)):
        raise ValueError("invalid token split")
    if split == "even" and float(cv) != 0.0 or split == "uneven" and not 0.0 < float(cv) <= 1.0:
        raise ValueError("invalid token split coefficient")
    routing = case["routing"]
    if not isinstance(routing, Mapping) or routing.get("kind") not in ROUTING_KEYS:
        raise ValueError("invalid routing")
    if set(routing) != ROUTING_KEYS[str(routing["kind"])]:
        raise ValueError("routing fields differ from schema")
    if routing["kind"] == "forced" and not (isinstance(routing["expert"], int) and 0 <= routing["expert"] < E):
        raise ValueError("forced expert outside E")
    if routing["kind"] == "captured":
        name, digest = routing["fixture"], routing["sha256"]
        if (not isinstance(name, str) or not 0 < len(name) <= 64 or not set(name) <= FIXTURE_NAME_CHARS
                or name[0] in ".-" or not isinstance(digest, str) or len(digest) != 64 or not set(digest) <= HEX_DIGITS):
            raise ValueError("captured routing must name a fixture and pin its sha256")
        _int(routing, "layer", 0, 1 << 10)
        _int(routing, "start", 0, 1 << 30)
    if not isinstance(case["source"], str) or not isinstance(case["model"], str):
        raise ValueError("invalid provenance strings")
    split_counts = token_split(M, W, split, float(cv), int(case["seed"]))
    weight_bytes = e_rank * N * K * 2
    if weight_bytes > MAX_WEIGHT_BYTES:
        raise ValueError("per-rank weight above 12 GiB")
    if M * K * 2 + (1 << 20) > WORKSPACE_BYTES:
        raise ValueError("gathered tokens exceed the workspace")
    case.update(
        E_rank=e_rank,
        split_counts=split_counts,
        effective_flops=effective_flops(M, topk, N, K, EP) * layers,
        scored=not edge,
    )
    return case


def effective_flops(M: int, topk: int, N: int, K: int, EP: int) -> int:
    """Mean per-rank GEMM work of one call.  Every (token, k) row lands on one EP
    group and is computed by its TP ranks, so the sum over ranks is TP*M*topk rows
    and the mean is M*topk/EP rows, independent of the routing draw."""
    return 2 * int(M) * int(topk) * int(N) * int(K) // int(EP)


def case_identity(case: Mapping[str, Any]) -> tuple[Any, ...]:
    return (int(case["K"]), int(case["N"]), int(case["E"]) // int(case["EP"]), int(case["topk"]))


# --------------------------------------------------------------------------- generators


def _gen(torch: Any, seed: int, device: Any) -> Any:
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) & 0xFFFFFFFFFFFF)
    return generator


def generate_x_full(torch: Any, M: int, K: int, seed: int, device: Any,
                    *, n_massive: int | None = None, massive_scale: float = 20.0) -> Any:
    """All tokens [M, K] BF16 with per-channel outliers and per-token scales."""
    g = _gen(torch, mix_seed(seed, 0x5EED_0002), device)
    x = torch.randn(M, K, generator=g, device=device, dtype=torch.bfloat16)
    ch_scale = torch.exp(0.4 * torch.randn(K, generator=g, device=device))
    count = int(torch.randint(2, 9, (1,), generator=g, device=device)) if n_massive is None else int(n_massive)
    idx = torch.randperm(K, generator=g, device=device)[:count]
    ch_scale[idx] *= 5.0 + max(0.0, massive_scale - 5.0) * torch.rand(count, generator=g, device=device)
    tok_scale = torch.exp(0.25 * torch.randn(M, 1, generator=g, device=device))
    x.mul_(ch_scale.to(torch.bfloat16)[None, :])
    x.mul_(tok_scale.to(torch.bfloat16))
    return x


def generate_routing(torch: Any, M: int, E: int, topk: int, routing: Mapping[str, Any],
                     seed: int, device: Any, fixtures: "RoutingFixtures | None" = None, layer_offset: int = 0):
    """Global top-k expert ids [M, topk] (int64) plus load statistics.  Captured
    routing reads a token window of a recorded router decision table; layer l of a
    multi-layer case reads the fixture layer after layer l - 1's."""
    kind = str(routing["kind"])
    if kind == "captured":
        if fixtures is None:
            raise RuntimeError("captured routing requires the routing fixture directory")
        ids = fixtures.window(str(routing["fixture"]), str(routing["sha256"]), int(routing["layer"]) + int(layer_offset),
                              int(routing["start"]) + mix_seed(seed, 0xCA97) % 4096, M, topk, E, device)
    elif kind == "forced":
        forced = int(routing["expert"])
        ids = torch.empty((M, topk), dtype=torch.int64, device=device)
        others = [x for x in range(E) if x != forced][: topk - 1]
        ids[:, 0] = forced
        for slot, expert in enumerate(others, start=1):
            ids[:, slot] = expert
    else:
        g = _gen(torch, mix_seed(seed, 0x5EED_0001), device)
        sigma = float(routing.get("sigma_e", 0.0))
        logits = -torch.log(-torch.log(torch.rand(M, E, generator=g, device=device).clamp_(1e-9, 1 - 1e-9)))
        if sigma > 0:
            logits += sigma * torch.randn(E, generator=g, device=device)[None, :]
        if kind in ("prod", "skew", "segment"):
            lam, seg_len = float(routing["lam"]), int(routing["seg_len"])
            if lam > 0:
                n_seg = (M + seg_len - 1) // seg_len
                seg_bias = lam * torch.randn(n_seg, E, generator=g, device=device)
                logits += seg_bias[torch.arange(M, device=device) // seg_len]
        if kind == "hot":
            hot = torch.randperm(E, generator=g, device=device)[: int(routing["hot"])]
            logits[:, hot] += float(routing["hot_boost"])
        ids = torch.topk(logits, topk, dim=1).indices
        del logits
    counts = torch.bincount(ids.reshape(-1), minlength=E).to(torch.float64)
    mean = float(counts.mean())
    stats = {
        "load_cv": float(counts.std(unbiased=False)) / mean if mean > 0 else 0.0,
        "load_max_over_mean": float(counts.max()) / mean if mean > 0 else 0.0,
        "empty_experts": int((counts == 0).sum()),
    }
    return ids, stats


def rank_rows(torch: Any, ids: Any, M: int, expert_lo: int, e_rank: int):
    """Rows of one rank: (token, k) pairs routed to experts [expert_lo, expert_lo + e_rank),
    sorted by local expert, then token."""
    flat = ids.reshape(-1)
    topk = ids.shape[1]
    local = flat - expert_lo
    mask = (local >= 0) & (local < e_rank)
    tokens = torch.arange(flat.numel(), device=flat.device, dtype=torch.int64) // topk
    sel_e = local[mask]
    sel_t = tokens[mask]
    order = torch.argsort(sel_e * M + sel_t)
    sorted_e = sel_e[order]
    sorted_t = sel_t[order]
    counts = torch.bincount(sorted_e, minlength=e_rank)
    offsets = torch.zeros(e_rank + 1, dtype=torch.int32, device=flat.device)
    offsets[1:] = torch.cumsum(counts, 0).to(torch.int32)
    return sorted_t.to(torch.int32).contiguous(), sorted_e.to(torch.int32).contiguous(), offsets.contiguous()


def generate_w(torch: Any, e_rank: int, N: int, K: int, seed: int, device: Any) -> Any:
    """Rank-private expert weights [E_rank, N, K] BF16 with a per-expert log-scale."""
    g = _gen(torch, mix_seed(seed, 0x5EED_0010), device)
    w = torch.randn(e_rank, N, K, generator=g, device=device, dtype=torch.bfloat16)
    scale = torch.exp(0.2 * torch.randn(e_rank, generator=g, device=device)) / math.sqrt(K)
    w.mul_(scale.to(torch.bfloat16)[:, None, None])
    return w, scale


def refresh_w(torch: Any, w: Any, scale: Any, seed: int) -> None:
    """Redraw a random 1/32 of the output rows of every expert in place, so that a
    result cached from an earlier call is wrong on the next one."""
    e_rank, N, K = (int(x) for x in w.shape)
    g = _gen(torch, mix_seed(seed, 0x5EED_0011), w.device)
    rows = torch.randperm(N, generator=g, device=w.device)[: max(1, N // 32)]
    fresh = torch.randn(e_rank, rows.numel(), K, generator=g, device=w.device, dtype=torch.bfloat16)
    fresh.mul_(scale.to(torch.bfloat16)[:, None, None])
    w.index_copy_(1, rows, fresh)


class RoutingFixtures:
    """Captured router top-k ids: <dir>/<name>.npy, uint8 or int16 [layers, tokens, topk].
    A case pins the file's sha256; the bytes are hashed before they are parsed."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._cache: dict[tuple[str, str], Any] = {}

    def table(self, name: str, sha256: str) -> Any:
        import io

        import numpy as np

        key = (name, sha256)
        if key not in self._cache:
            if not set(name) <= FIXTURE_NAME_CHARS or name[:1] in ("", ".", "-"):
                raise RuntimeError(f"invalid routing fixture name {name!r}")
            data = (self.root / f"{name}.npy").read_bytes()
            if hashlib.sha256(data).hexdigest() != sha256:
                raise RuntimeError(f"routing fixture {name} differs from the digest the case pins")
            table = np.load(io.BytesIO(data), allow_pickle=False)
            if table.ndim != 3 or table.dtype not in (np.dtype(np.uint8), np.dtype(np.int16)) or min(table.shape) < 1:
                raise RuntimeError(f"routing fixture {name} is not an integer [layers, tokens, topk] table")
            self._cache[key] = table
        return self._cache[key]

    def window(self, name: str, sha256: str, layer: int, start: int, M: int, topk: int, E: int, device: Any):
        import numpy as np
        import torch

        table = self.table(name, sha256)
        if table.shape[2] != topk:
            raise RuntimeError(f"routing fixture {name} has top-k {table.shape[2]}, case has {topk}")
        tokens = table.shape[1]
        if M > tokens:
            raise RuntimeError(f"routing fixture {name} has {tokens} tokens, case needs {M}")
        start = start % (tokens - M + 1)
        window = np.ascontiguousarray(table[layer % table.shape[0], start:start + M].astype(np.int64))
        if window.min() < 0 or window.max() >= E:
            raise RuntimeError(f"routing fixture {name} expert ids outside E={E}")
        return torch.from_numpy(window).to(device)


# --------------------------------------------------------------------------- gates


def realised_limits(reference: Mapping[str, float]) -> dict[str, float]:
    return {
        "rel_l1_limit": max(L1_MULT * float(reference["rel_l1"]), L1_FLOOR),
        "row_rel_l2_p999_limit": max(P999_MULT * float(reference["row_rel_l2_p999"]), P999_FLOOR),
        "row_rel_l2_max_limit": max(ROWMAX_MULT * float(reference["row_rel_l2_max"]), ROWMAX_FLOOR),
    }


def check_metric_bundle(candidate: Mapping[str, float], reference: Mapping[str, float]) -> dict[str, float]:
    limits = realised_limits(reference)
    for key in ("rel_l1", "cossim", "row_rel_l2_p999", "row_rel_l2_max", "max_abs"):
        if not math.isfinite(float(candidate[key])):
            raise RuntimeError(f"nonfinite metric {key}")
    if float(candidate["rel_l1"]) > limits["rel_l1_limit"]:
        raise RuntimeError("G3 global RelL1 gate failed")
    if float(candidate["cossim"]) < COS_FLOOR:
        raise RuntimeError("G4 global CosSim gate failed")
    if float(candidate["row_rel_l2_p999"]) > limits["row_rel_l2_p999_limit"]:
        raise RuntimeError("G5 row p99.9 gate failed")
    if float(candidate["row_rel_l2_max"]) > limits["row_rel_l2_max_limit"]:
        raise RuntimeError("G6 row max gate failed")
    return limits


def aggregate_case_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """family score = sqrt(geomean(evidence cases) * geomean(model cases)) when a
    family has both kinds, else the geomean of what it has; primary = geomean over
    the families present."""
    scored = [row for row in results if bool(row.get("scored"))]
    if not scored:
        raise ValueError("no scored cases")
    by_family: dict[str, dict[str, list[float]]] = {}
    for row in scored:
        part = "evidence" if str(row["tier"]) in EVIDENCE_TIERS else "model"
        by_family.setdefault(str(row["family"]), {}).setdefault(part, []).append(
            finite_positive(row["tflops"], "case TFLOPS")
        )
    families: dict[str, float] = {}
    for family in FAMILIES:
        parts = by_family.get(family)
        if not parts:
            continue
        means = [geometric_mean(values) for values in parts.values()]
        families[family] = geometric_mean(means) if len(means) == 2 else means[0]
    primary = geometric_mean(families.values())
    hidden = [float(row["hidden_comm"]) for row in scored if row.get("hidden_comm") is not None]
    return {
        "family_tflops": families,
        "primary_tflops": primary,
        "score_tflops": primary,
        "hidden_comm": statistics.fmean(hidden) if hidden else None,
    }


# --------------------------------------------------------------------------- candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidate(path: Path) -> tuple[ModuleType, Callable[..., Any], str]:
    candidate = path.resolve()
    if not candidate.is_file() or candidate.is_symlink():
        raise RuntimeError("candidate is not a regular file")
    digest = _sha256(candidate)
    spec = importlib.util.spec_from_file_location(f"moe_ag_candidate_{digest[:24]}", candidate)
    if spec is None or spec.loader is None:
        raise RuntimeError("candidate cannot load")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    fn = getattr(module, ENTRYPOINT, None)
    if not callable(fn):
        raise RuntimeError(f"candidate must define {ENTRYPOINT}")
    return module, fn, digest


# --------------------------------------------------------------------------- execution context


@dataclass
class GroupState:
    gid: str
    W: int
    pg: Any
    name: str
    group_rank: int
    logical_rank: int
    proc_of_logical: tuple[int, ...]
    workspace: Any = None
    ws_handle: Any = None
    peer_workspaces: tuple[Any, ...] = ()
    workspace_ptrs: Any = None
    pools: list[Any] = field(default_factory=list)
    pool_handles: list[Any] = field(default_factory=list)
    pool_bytes: int = 0


@dataclass
class Invocation:
    case: dict[str, Any]
    x_full: Any
    x_local: Any
    x_peers: tuple[Any, ...]
    w: Any
    sorted_token_ids: Any
    sorted_expert_ids: Any
    expert_offsets: Any
    token_offsets: Any
    out: Any
    workspace: Any
    peer_workspaces: tuple[Any, ...]
    peer_ptrs: Any
    workspace_ptrs: Any
    rank: int
    world_size: int
    rows: int
    regime: dict[str, Any]
    snapshot: tuple[Any, ...] = ()


class Runner:
    def __init__(self, torch: Any, dist: Any, symm_mem: Any, fn: Callable[..., Any], *,
                 global_rank: int, trials: int, warmups: int, fixtures: RoutingFixtures | None,
                 log: Callable[[str], None] | None = None):
        self.torch, self.dist, self.symm = torch, dist, symm_mem
        self.fn = fn
        self.global_rank = global_rank
        self.device = torch.device("cuda", global_rank)
        self.trials, self.warmups = trials, warmups
        self.fixtures = fixtures
        if log is None and os.environ.get("FLUXMOE_DEBUG") == "1":
            log = lambda msg: print(f"[rank {global_rank}] {msg}", flush=True)  # noqa: E731
        self.log = log or (lambda _msg: None)

    # ---- group lifecycle

    def init_world(self, world_name: str, cases: Sequence[Mapping[str, Any]]) -> None:
        """Allocate the symmetric workspace and the two shard pools once, on all eight
        processes, and rendezvous them over the world group.  Every placement group uses
        views of these buffers: symmetric-memory rendezvous must run the same number of
        times on every process, and the workspace persists for the whole suite."""
        torch, symm = self.torch, self.symm
        symm.enable_symm_mem_for_group(world_name)
        need = 0
        for case in cases:
            need = max(need, int(case["layers"]) * self._shard_stride(case))
        self.pool_bytes = _align(need + SHIFT_SLOTS * SHIFT_BYTES, POOL_ALIGN)
        self.workspace = symm.empty((WORKSPACE_BYTES,), dtype=torch.uint8, device=self.device)
        self.workspace.zero_()
        torch.cuda.synchronize()
        self.ws_handle = symm.rendezvous(self.workspace, group=world_name)
        self.pools, self.pool_handles = [], []
        for _ in range(2):
            pool = symm.empty((self.pool_bytes,), dtype=torch.uint8, device=self.device)
            self.pools.append(pool)
            self.pool_handles.append(symm.rendezvous(pool, group=world_name))
        self.log(f"world buffers ready: workspace {WORKSPACE_BYTES} B, pools 2 x {self.pool_bytes} B")

    def open_group(self, gid: str, pg: Any, cases: Sequence[Mapping[str, Any]]) -> GroupState:
        torch = self.torch
        processes, order = GROUPS[gid]
        group_rank = processes.index(self.global_rank)
        logical = order.index(group_rank)
        globals_of_logical = tuple(processes[order[q]] for q in range(len(processes)))
        state = GroupState(gid, len(processes), pg, pg.group_name, group_rank, logical, globals_of_logical)
        self.log(f"{gid} open: group_rank {group_rank} logical {logical}")
        state.workspace = self.workspace
        state.ws_handle = self.ws_handle
        state.pools = self.pools
        state.pool_handles = self.pool_handles
        state.pool_bytes = self.pool_bytes
        views = []
        for g in globals_of_logical:
            views.append(self.workspace if g == self.global_rank else
                         self.ws_handle.get_buffer(g, (WORKSPACE_BYTES,), torch.uint8))
        state.peer_workspaces = tuple(views)
        state.workspace_ptrs = torch.tensor([int(v.data_ptr()) for v in views], dtype=torch.int64, device=self.device)
        self.barrier(state)
        return state

    def close_group(self, state: GroupState) -> None:
        self.barrier(state)
        state.peer_workspaces = ()
        state.workspace_ptrs = None
        self.torch.cuda.synchronize()

    @staticmethod
    def _shard_stride(case: Mapping[str, Any]) -> int:
        return _align(max(case["split_counts"]) * int(case["K"]) * 2, 512)

    # ---- collectives

    def barrier(self, state: GroupState) -> None:
        self.torch.cuda.synchronize()
        self.dist.barrier(group=state.pg)

    def all_max(self, state: GroupState, value: float) -> float:
        t = self.torch.tensor([float(value)], dtype=self.torch.float64, device=self.device)
        self.dist.all_reduce(t, op=self.dist.ReduceOp.MAX, group=state.pg)
        return float(t.item())

    def all_ok(self, state: GroupState, ok: bool, detail: Any) -> None:
        t = self.torch.tensor([1 if ok else 0], dtype=self.torch.int32, device=self.device)
        self.dist.all_reduce(t, op=self.dist.ReduceOp.MIN, group=state.pg)
        if int(t.item()) == 1:
            return
        gathered: list[Any] = [None] * state.W
        self.dist.all_gather_object(gathered, {"rank": state.logical_rank, "detail": None if ok else detail}, group=state.pg)
        raise RuntimeError(f"rank gate failure: {[x for x in gathered if x and x['detail']]}")

    # ---- inputs

    def prepare(self, state: GroupState, case: Mapping[str, Any], w: Any, seed: int,
                pool_index: int, layer: int = 0, *, shift: int = 0) -> Invocation:
        torch = self.torch
        W, K, N, M, topk = state.W, int(case["K"]), int(case["N"]), int(case["M"]), int(case["topk"])
        q = state.logical_rank
        counts = case["split_counts"]
        offsets = [0]
        for c in counts:
            offsets.append(offsets[-1] + c)
        x_full = generate_x_full(torch, M, K, seed, self.device,
                                 n_massive=case.get("n_massive"), massive_scale=float(case.get("massive_scale", 20.0)))
        ids, regime = generate_routing(torch, M, int(case["E"]), topk, case["routing"], seed, self.device, self.fixtures,
                                       layer_offset=layer)
        e_rank = int(case["E_rank"])
        ep_index = q // int(case["TP"])
        sti, sei, eoff = rank_rows(torch, ids, M, ep_index * e_rank, e_rank)
        del ids
        stride = self._shard_stride(case)
        base_bytes = layer * stride + shift * SHIFT_BYTES
        pool = state.pools[pool_index]
        handle = state.pool_handles[pool_index]
        peers = []
        for p in range(W):
            g = state.proc_of_logical[p]
            if g == self.global_rank:
                view = pool[base_bytes:base_bytes + counts[p] * K * 2].view(torch.bfloat16).view(counts[p], K)
            else:
                view = handle.get_buffer(g, (counts[p], K), torch.bfloat16, base_bytes // 2)
            peers.append(view)
        x_local = peers[q]
        x_local.copy_(x_full[offsets[q]:offsets[q + 1]])
        token_offsets = torch.tensor(offsets, dtype=torch.int32, device=self.device)
        peer_ptrs = torch.tensor([int(v.data_ptr()) for v in peers], dtype=torch.int64, device=self.device)
        rows = int(sti.numel())
        out = torch.full((rows, N), float("nan"), dtype=torch.bfloat16, device=self.device)
        regime = dict(regime)
        regime["rows"] = rows
        inv = Invocation(dict(case), x_full, x_local, tuple(peers), w, sti, sei, eoff, token_offsets, out,
                         state.workspace, state.peer_workspaces, peer_ptrs, state.workspace_ptrs, q, W, rows, regime)
        inv.snapshot = self.snapshot(inv)
        return inv

    def snapshot(self, inv: Invocation) -> tuple[Any, ...]:
        torch = self.torch
        small = tuple(t.clone() for t in (inv.sorted_token_ids, inv.sorted_expert_ids, inv.expert_offsets,
                                            inv.token_offsets, inv.peer_ptrs, inv.workspace_ptrs))
        return small + (_checksum(torch, inv.x_local), _checksum(torch, inv.w))

    def unchanged(self, inv: Invocation) -> bool:
        torch = self.torch
        now = (inv.sorted_token_ids, inv.sorted_expert_ids, inv.expert_offsets, inv.token_offsets,
               inv.peer_ptrs, inv.workspace_ptrs)
        if not all(bool(torch.equal(a, b)) for a, b in zip(now, inv.snapshot[:6])):
            return False
        return _checksum(torch, inv.x_local) == inv.snapshot[6] and _checksum(torch, inv.w) == inv.snapshot[7]

    def invoke(self, inv: Invocation) -> None:
        result = self.fn(inv.x_local, inv.x_peers, inv.w, inv.sorted_token_ids, inv.sorted_expert_ids,
                         inv.expert_offsets, inv.token_offsets, inv.out, inv.workspace, inv.peer_workspaces,
                         inv.peer_ptrs, inv.workspace_ptrs, inv.rank, inv.world_size)
        if result is not None:
            raise RuntimeError(f"{ENTRYPOINT} must return None")

    def timed(self, state: GroupState, calls: Sequence[Invocation]) -> tuple[float, float, float]:
        torch = self.torch
        self.barrier(state)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall = time.perf_counter()
        start.record()
        for inv in calls:
            self.invoke(inv)
        end.record()
        torch.cuda.synchronize()
        host = finite_positive(time.perf_counter() - wall, "host wall")
        event = finite_positive(start.elapsed_time(end) / 1000.0, "event")
        local = max(host, event)
        return self.all_max(state, local), host, event

    # ---- checks

    def basic_checks(self, inv: Invocation) -> str:
        torch = self.torch
        if not self.unchanged(inv):
            return "G2 candidate modified an input"
        if inv.rows and not bool(torch.isfinite(inv.out).all()):
            return "G1 output contains unwritten or nonfinite values"
        return ""

    def freivalds(self, inv: Invocation, seed: int) -> tuple[str, dict[str, float]]:
        """Compare out @ v against x[token] . (W_e^T v) for every row in FP32."""
        torch = self.torch
        if inv.rows == 0:
            return "", {"freivalds_rel_l2": 0.0, "freivalds_worst_row": 0.0}
        e_rank, N, K = (int(x) for x in inv.w.shape)
        g = _gen(torch, mix_seed(seed, 0xF4E1), self.device)
        v = torch.randn(N, generator=g, device=self.device, dtype=torch.float32)
        wv = torch.empty((e_rank, K), dtype=torch.float32, device=self.device)
        chunk = max(1, (1 << 28) // max(1, N * K))
        for lo in range(0, e_rank, chunk):
            hi = min(e_rank, lo + chunk)
            wv[lo:hi] = torch.einsum("enk,n->ek", inv.w[lo:hi].float(), v)
        tok = inv.sorted_token_ids.long()
        exp = inv.sorted_expert_ids.long()
        ref = torch.empty(inv.rows, dtype=torch.float32, device=self.device)
        cand = torch.empty(inv.rows, dtype=torch.float32, device=self.device)
        step = max(1, (1 << 27) // max(K, N))
        for lo in range(0, inv.rows, step):
            hi = min(inv.rows, lo + step)
            xr = inv.x_full.index_select(0, tok[lo:hi]).float()
            ref[lo:hi] = (xr * wv.index_select(0, exp[lo:hi])).sum(dim=1)
            cand[lo:hi] = inv.out[lo:hi].float() @ v
        diff = cand - ref
        rms = float(torch.sqrt((ref * ref).mean()).item()) or 1e-30
        rel = float(torch.linalg.vector_norm(diff).item()) / (math.sqrt(inv.rows) * rms)
        worst = float(diff.abs().max().item()) / rms
        metrics = {"freivalds_rel_l2": rel, "freivalds_worst_row": worst}
        if not math.isfinite(rel) or rel > FREIVALDS_REL_L2 or worst > FREIVALDS_ROW_FRACTION:
            return f"G7 Freivalds projection gate failed (rel {rel:.3e}, worst row {worst:.3e})", metrics
        return "", metrics

    def golden(self, inv: Invocation) -> tuple[str, dict[str, float]]:
        torch = self.torch
        if inv.rows == 0:
            return "", {}
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            # In chunks of consecutive expert-sorted rows (each chunk spanning however many
            # experts it covers), so no [rows, N] FP32 tensor is ever held.
            offsets = inv.expert_offsets.tolist()
            e_rank, N, K = (int(x) for x in inv.w.shape)
            tok = inv.sorted_token_ids.long()
            cand_acc = _MetricAccumulator(torch, inv.rows, self.device)
            ref_acc = _MetricAccumulator(torch, inv.rows, self.device)
            step = max(1, GOLDEN_CHUNK_ELEMENTS // max(N, K))
            golden = torch.empty((min(step, inv.rows), N), dtype=torch.float32, device=self.device)
            reference = torch.empty((min(step, inv.rows), N), dtype=torch.bfloat16, device=self.device)
            expert = 0
            for a in range(0, inv.rows, step):
                b = min(inv.rows, a + step)
                while offsets[expert + 1] <= a:
                    expert += 1
                e = expert
                while e < e_rank and offsets[e] < b:
                    lo, hi = max(a, offsets[e]), min(b, offsets[e + 1])
                    if hi > lo:
                        rows = inv.x_full.index_select(0, tok[lo:hi])
                        golden[lo - a:hi - a] = rows.float() @ inv.w[e].float().t()
                        reference[lo - a:hi - a] = rows @ inv.w[e].t()
                        del rows
                    e += 1
                cand_acc.add(a, inv.out[a:b], golden[:b - a])
                ref_acc.add(a, reference[:b - a], golden[:b - a])
            del golden, reference
            cand = cand_acc.result()
            ref = ref_acc.result()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        result = dict(cand)
        result.update({"reference_" + k: v for k, v in ref.items()})
        result.update(realised_limits(ref))
        try:
            check_metric_bundle(cand, ref)
        except RuntimeError as exc:
            return str(exc), result
        return "", result

    # ---- cases

    def run_case(self, state: GroupState, case: Mapping[str, Any]) -> dict[str, Any]:
        if int(case["layers"]) > 1:
            return self.run_sequence(state, case)
        torch = self.torch
        scored = bool(case["scored"])
        seed = int(case["seed"])
        q = state.logical_rank
        w, scale = generate_w(torch, int(case["E_rank"]), int(case["N"]), int(case["K"]), mix_seed(seed, q), self.device)
        trials = self.trials if scored else 1
        gated = _gated_trials(seed, trials) if scored else {0}
        metrics: dict[str, float] = {}
        free_worst = {"freivalds_rel_l2": 0.0, "freivalds_worst_row": 0.0}
        merged: list[float] = []
        rank_times: list[float] = []
        rows_seen: list[int] = []
        regime: dict[str, Any] = {}
        for index in range(max(1, self.warmups)):
            refresh_w(torch, w, scale, mix_seed(seed, ROLE_SALT["warm"], index, q))
            warm = self.prepare(state, case, w, mix_seed(seed, ROLE_SALT["warm"], index), 0,
                                shift=(index + 3) % SHIFT_SLOTS)
            self.barrier(state)
            self.log(f"case {case['case_id']} warm invoke")
            self.invoke(warm)
            torch.cuda.synchronize()
            self.log(f"case {case['case_id']} warm done")
            problem = self.basic_checks(warm)
            self.all_ok(state, not problem, problem)
            del warm
        hidden_rows: list[dict[str, float]] = []
        comm_reference = None
        for index in range(trials):
            if scored:
                refresh_w(torch, w, scale, mix_seed(seed, ROLE_SALT["decoy"], index, q))
                decoy = self.prepare(state, case, w, mix_seed(seed, ROLE_SALT["decoy"], index), 0,
                                     shift=(2 * index) % SHIFT_SLOTS)
                self.barrier(state)
                self.invoke(decoy)
                torch.cuda.synchronize()
                problem = self.basic_checks(decoy)
                self.all_ok(state, not problem, problem)
                del decoy
            refresh_w(torch, w, scale, mix_seed(seed, ROLE_SALT["actual"], index, q))
            actual = self.prepare(state, case, w, mix_seed(seed, ROLE_SALT["actual"], index), 1,
                                  shift=(2 * index + 1) % SHIFT_SLOTS)
            t_merged, host, event = self.timed(state, [actual])
            problem = self.basic_checks(actual)
            if not problem:
                problem, fmetrics = self.freivalds(actual, mix_seed(seed, index, q))
                for key, value in fmetrics.items():
                    free_worst[key] = max(free_worst[key], value)
            if not problem and index in gated:
                problem, gmetrics = self.golden(actual)
                if gmetrics:
                    _merge_metric_worst(metrics, gmetrics)
            self.all_ok(state, not problem, problem)
            merged.append(t_merged)
            rank_times.append(max(host, event))
            rows_seen.append(actual.rows)
            regime = actual.regime
            if scored and str(case["tier"]) == "paper" and index < 3:
                if comm_reference is None:
                    comm_reference = self.transfer_reference(state, actual)
                twin = self.twin(state, actual)
                t_twin, _h, _e = self.timed(state, [twin])
                exposed = max(0.0, t_merged - t_twin)
                hidden_rows.append({
                    "twin_seconds": t_twin,
                    "exposed_seconds": exposed,
                    "hidden_comm": float(min(1.0, max(0.0, 1.0 - exposed / comm_reference))) if comm_reference > 0 else 0.0,
                })
                del twin
            del actual
        median = statistics.median(merged)
        row = _identity(case)
        row.update(
            rank=q,
            regime=regime,
            trial_seconds=merged,
            rank_trial_seconds=rank_times,
            rows_per_trial=rows_seen,
            median_ms=median * 1e3,
            tflops=float(case["effective_flops"]) / median / 1e12,
            gated_trials=sorted(gated),
            **metrics,
            **free_worst,
        )
        if hidden_rows:
            row.update(
                comm_reference_ms=comm_reference * 1e3,
                twin_median_ms=statistics.median(x["twin_seconds"] for x in hidden_rows) * 1e3,
                exposed_median_ms=statistics.median(x["exposed_seconds"] for x in hidden_rows) * 1e3,
                hidden_comm=statistics.median(x["hidden_comm"] for x in hidden_rows),
                hidden_rows=hidden_rows,
            )
        else:
            row["hidden_comm"] = None
        del w
        return row

    def run_sequence(self, state: GroupState, case: Mapping[str, Any]) -> dict[str, Any]:
        torch = self.torch
        seed = int(case["seed"])
        q = state.logical_rank
        layers = int(case["layers"])
        ring = min(layers, 4)
        weights = [generate_w(torch, int(case["E_rank"]), int(case["N"]), int(case["K"]),
                              mix_seed(seed, q, slot), self.device) for slot in range(ring)]
        rounds = self.trials
        g_round = mix_seed(seed, 0x6A7E) % rounds
        g_layers = {mix_seed(seed, 0x6A7F, i) % layers for i in range(2)}
        metrics: dict[str, float] = {}
        free_worst = {"freivalds_rel_l2": 0.0, "freivalds_worst_row": 0.0}
        merged: list[float] = []
        rank_times: list[float] = []
        rows_seen: list[int] = []
        regime: dict[str, Any] = {}
        for index in range(-max(1, self.warmups), rounds):
            role = "warm" if index < 0 else "actual"
            # Layers l and l + ring share one weight buffer (refreshed once per round), as
            # consecutive MoE layers of a model hold different but fixed weights.
            for slot, (w, scale) in enumerate(weights):
                refresh_w(torch, w, scale, mix_seed(seed, ROLE_SALT[role], index, slot, q))
            calls = [
                self.prepare(state, case, weights[layer % ring][0],
                             mix_seed(seed, ROLE_SALT[role], index, layer), 1 if index >= 0 else 0, layer)
                for layer in range(layers)
            ]
            t_merged, host, event = self.timed(state, calls)
            problem = ""
            for layer, inv in enumerate(calls):
                problem = self.basic_checks(inv)
                if not problem and index >= 0:
                    problem, fm = self.freivalds(inv, mix_seed(seed, index, layer, q))
                    for key, value in fm.items():
                        free_worst[key] = max(free_worst[key], value)
                if not problem and index == g_round and layer in g_layers:
                    problem, gm = self.golden(inv)
                    if gm:
                        _merge_metric_worst(metrics, gm)
                if problem:
                    problem = f"layer {layer}: {problem}"
                    break
            self.all_ok(state, not problem, problem)
            if index >= 0:
                merged.append(t_merged)
                rank_times.append(max(host, event))
                rows_seen.append(sum(inv.rows for inv in calls))
                regime = calls[0].regime
            del calls
        median = statistics.median(merged)
        row = _identity(case)
        row.update(
            rank=q,
            regime=regime,
            trial_seconds=merged,
            rank_trial_seconds=rank_times,
            rows_per_trial=rows_seen,
            median_ms=median * 1e3,
            tflops=float(case["effective_flops"]) / median / 1e12,
            gated_trials=[g_round],
            gated_layers=sorted(g_layers),
            hidden_comm=None,
            **metrics,
            **free_worst,
        )
        del weights
        return row

    def twin(self, state: GroupState, actual: Invocation) -> Invocation:
        """The no-transfer twin: every peer shard's values in a plain local tensor and every
        peer workspace aliased to the local one, so no byte crosses between GPUs."""
        torch = self.torch
        copies = []
        for p, view in enumerate(actual.x_peers):
            if p == actual.rank:
                copies.append(actual.x_local)
            else:
                local = torch.empty_like(view)
                local.copy_(view)
                copies.append(local)
        torch.cuda.synchronize()
        out = torch.full_like(actual.out, float("nan"))
        ptrs = torch.tensor([int(v.data_ptr()) for v in copies], dtype=torch.int64, device=self.device)
        ws = tuple(actual.workspace for _ in range(actual.world_size))
        ws_ptrs = torch.full((actual.world_size,), int(actual.workspace.data_ptr()), dtype=torch.int64, device=self.device)
        twin = Invocation(actual.case, actual.x_full, actual.x_local, tuple(copies), actual.w, actual.sorted_token_ids,
                          actual.sorted_expert_ids, actual.expert_offsets, actual.token_offsets, out,
                          actual.workspace, ws, ptrs, ws_ptrs, actual.rank, actual.world_size, actual.rows, actual.regime)
        return twin

    def transfer_reference(self, state: GroupState, actual: Invocation) -> float:
        """Median trusted time for this rank to pull every peer shard in ring order."""
        torch = self.torch
        sinks = [torch.empty_like(v) for p, v in enumerate(actual.x_peers) if p != actual.rank]
        samples = []
        for _ in range(3):
            self.barrier(state)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            i = 0
            for step in range(1, actual.world_size):
                p = (actual.rank + step) % actual.world_size
                sinks[i].copy_(actual.x_peers[p])
                i += 1
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end) / 1000.0)
        return self.all_max(state, statistics.median(samples))


def _align(value: int, alignment: int) -> int:
    return (int(value) + alignment - 1) // alignment * alignment


def _gated_trials(seed: int, trials: int) -> set[int]:
    chosen: set[int] = set()
    i = 0
    while len(chosen) < min(GATED_TRIALS, trials):
        chosen.add(mix_seed(seed, 0x6A7D, i) % trials)
        i += 1
    return chosen


CHECKSUM_CHUNK_BYTES = 256 << 20  # a reduction with dtype=int64 casts its whole input first


def _checksum(torch: Any, tensor: Any) -> tuple[int, int]:
    """Two integer sums over the raw bytes (int32 and int16 words), taken in chunks so the
    int64 cast never holds more than four chunk sizes of scratch."""
    flat = tensor.reshape(-1).view(torch.uint8)
    n = int(flat.numel())
    s1 = s2 = 0
    for lo in range(0, n, CHECKSUM_CHUNK_BYTES):
        part = flat[lo:min(n, lo + CHECKSUM_CHUNK_BYTES)]
        if n % 4 == 0:
            s1 += int(part.view(torch.int32).sum(dtype=torch.int64).item())
        else:
            s1 += int(part.sum(dtype=torch.int64).item())
        if n % 2 == 0:
            s2 += int(part.view(torch.int16).sum(dtype=torch.int64).item())
    if n % 2:
        s2 = int(flat[::3].sum(dtype=torch.int64).item())
    return s1, s2


GOLDEN_CHUNK_ELEMENTS = 1 << 25  # rows x max(N, K) elements per golden chunk (128 MiB in FP32)


class _MetricAccumulator:
    """G3-G6 metrics of predicted rows against golden rows, fed chunk by chunk: global sums
    in FP64, per-row relative L2 errors in a [rows] vector, the largest absolute error."""

    def __init__(self, torch: Any, rows: int, device: Any):
        self.torch = torch
        zero = lambda: torch.zeros((), dtype=torch.float64, device=device)  # noqa: E731
        self.abs_diff, self.abs_gold, self.dot, self.pred_sq, self.gold_sq = (zero() for _ in range(5))
        self.max_abs = torch.zeros((), dtype=torch.float32, device=device)
        self.row = torch.empty(rows, dtype=torch.float32, device=device)

    def add(self, start: int, predicted: Any, golden: Any) -> None:
        torch = self.torch
        pred = predicted.float()
        diff = pred - golden
        self.abs_diff += diff.abs().sum(dtype=torch.float64)
        self.abs_gold += golden.abs().sum(dtype=torch.float64)
        self.dot += (pred * golden).sum(dtype=torch.float64)
        self.pred_sq += (pred * pred).sum(dtype=torch.float64)
        self.gold_sq += (golden * golden).sum(dtype=torch.float64)
        self.max_abs = torch.maximum(self.max_abs, diff.abs().max())
        self.row[start:start + int(golden.shape[0])] = (
            torch.linalg.vector_norm(diff, dim=1) / torch.linalg.vector_norm(golden, dim=1).clamp(min=ROW_NORM_FLOOR)
        )

    def result(self) -> dict[str, float]:
        torch = self.torch
        rel = self.abs_diff / self.abs_gold.clamp(min=1e-30)
        cos = self.dot / (torch.sqrt(self.pred_sq) * torch.sqrt(self.gold_sq)).clamp(min=1e-30)
        count = max(1, int(math.ceil(0.001 * int(self.row.numel()))))
        p999 = torch.topk(self.row, count).values.min()
        return {
            "rel_l1": float(rel.item()),
            "cossim": float(cos.item()),
            "row_rel_l2_p999": float(p999.item()),
            "row_rel_l2_max": float(self.row.max().item()),
            "max_abs": float(self.max_abs.item()),
        }


def _merge_metric_worst(into: dict[str, float], new: Mapping[str, float]) -> None:
    for key in ("rel_l1", "row_rel_l2_p999", "row_rel_l2_max", "max_abs"):
        if key not in into or float(new[key]) > float(into[key]):
            into[key] = float(new[key])
            if "reference_" + key in new:
                into["reference_" + key] = float(new["reference_" + key])
            if key + "_limit" in new:
                into[key + "_limit"] = float(new[key + "_limit"])
    if "cossim" not in into or float(new["cossim"]) < float(into["cossim"]):
        into["cossim"] = float(new["cossim"])
        into["reference_cossim"] = float(new["reference_cossim"])


def _identity(case: Mapping[str, Any]) -> dict[str, Any]:
    keys = ("case_id", "family", "tier", "group", "W", "TP", "EP", "K", "N", "E", "E_rank", "topk", "M",
            "layers", "split_counts", "effective_flops", "scored")
    return {key: case[key] for key in keys}


# --------------------------------------------------------------------------- hardware

MIN_COMPUTE_CAPABILITY = (8, 0)
MIN_DEVICE_MEMORY_BYTES = 64 * 1024**3


def hardware_inventory(torch: Any) -> dict[str, Any]:
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if count != WORLD_PROCESSES:
        raise RuntimeError(f"exactly {WORLD_PROCESSES} visible CUDA GPUs required, found {count}")
    devices = []
    for index in range(count):
        capability = tuple(int(v) for v in torch.cuda.get_device_capability(index))
        props = torch.cuda.get_device_properties(index)
        name = str(torch.cuda.get_device_name(index))
        if capability < MIN_COMPUTE_CAPABILITY:
            raise RuntimeError(f"GPU {index} compute capability {capability} below {MIN_COMPUTE_CAPABILITY}: {name}")
        if int(props.total_memory) < MIN_DEVICE_MEMORY_BYTES:
            raise RuntimeError(f"GPU {index} memory {int(props.total_memory)} below {MIN_DEVICE_MEMORY_BYTES}: {name}")
        peers = [bool(torch.cuda.can_device_access_peer(index, j)) for j in range(count) if j != index]
        devices.append({
            "index": index,
            "name": name,
            "uuid": str(getattr(props, "uuid", "")),
            "capability": list(capability),
            "total_memory_bytes": int(props.total_memory),
            "peer_access_to_all": all(peers),
        })
    if not all(x["peer_access_to_all"] for x in devices):
        raise RuntimeError("CUDA peer access between every GPU pair required")
    return {
        "count": count,
        "devices": devices,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "interconnect": "host preflight records nvidia-smi topo -m",
    }


# --------------------------------------------------------------------------- suite


def order_cases(cases: Sequence[Mapping[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Group cases by placement group (fixed group order); edges first within a group."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        buckets.setdefault(str(case["group"]), []).append(dict(case))
    ordered = []
    for gid in GROUP_ORDER:
        if gid in buckets:
            items = buckets[gid]
            ordered.append((gid, [x for x in items if not x["scored"]] + [x for x in items if x["scored"]]))
    return ordered


def run_suite(candidate_path: Path, raw_cases: Sequence[Mapping[str, Any]], *, trials: int, warmups: int,
              require_hardware: bool = True, fixture_dir: Path | None = None,
              progress: Callable[[str], None] | None = None) -> dict[str, Any] | None:
    import torch
    import torch.distributed as dist
    import torch.distributed._symmetric_memory as symm_mem

    world = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    if world != WORLD_PROCESSES or not 0 <= rank < world or int(os.environ.get("LOCAL_RANK", "-1")) != rank:
        raise RuntimeError(f"run with {WORLD_PROCESSES} ranks (RANK == LOCAL_RANK)")
    if trials < 1 or warmups < 1:
        raise ValueError("trials and warmups must be positive")
    cases = [validate_case(x) for x in raw_cases]
    ids = [str(x["case_id"]) for x in cases]
    if not any(x["scored"] for x in cases) or len(ids) != len(set(ids)):
        raise ValueError("suite must have scored cases and unique IDs")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    inventory = hardware_inventory(torch) if require_hardware else {"count": torch.cuda.device_count()}
    torch.cuda.set_device(rank)
    torch.cuda.reset_peak_memory_stats(rank)
    import datetime

    timeout = datetime.timedelta(seconds=COLLECTIVE_TIMEOUT_SECONDS)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank), timeout=timeout)
    # Processes outside a placement group wait on this CPU (gloo) group, so an idle GPU
    # runs no NCCL kernel while another group measures.
    host_group = dist.new_group(backend="gloo", timeout=timeout)
    plan = order_cases(cases)
    pgs = {gid: dist.new_group(ranks=list(GROUPS[gid][0]), timeout=timeout) for gid, _items in plan}
    # Initialise every group's NCCL communicator up front, in one order on every process;
    # a communicator first used while other processes sit in a world collective can hang.
    for gid, _items in plan:
        if rank in GROUPS[gid][0]:
            dist.barrier(group=pgs[gid])
    dist.barrier(group=host_group)
    identities: list[Any] = [None] * world
    props = torch.cuda.get_device_properties(rank)
    dist.all_gather_object(identities, {"pid": os.getpid(), "rank": rank, "uuid": str(getattr(props, "uuid", ""))},
                           group=host_group)
    if len({x["uuid"] for x in identities}) != world:
        raise RuntimeError("ranks do not own distinct GPU UUIDs")
    fixtures = RoutingFixtures(fixture_dir) if fixture_dir is not None else None
    started = time.perf_counter()
    _module, fn, digest = load_candidate(Path(candidate_path))
    runner = Runner(torch, dist, symm_mem, fn, global_rank=rank, trials=trials, warmups=warmups, fixtures=fixtures)
    runner.init_world(dist.group.WORLD.group_name, cases)
    dist.barrier(group=host_group)
    free_bytes, total_bytes = torch.cuda.mem_get_info(rank)
    baseline_device_bytes = int(total_bytes - free_bytes)  # context, NCCL, symmetric buffers
    rows: list[dict[str, Any]] = []
    group_seconds: dict[str, float] = {}
    peak_allocated = peak_reserved = 0
    try:
        for gid, items in plan:
            members = GROUPS[gid][0]
            if rank in members:
                t0 = time.perf_counter()
                state = runner.open_group(gid, pgs[gid], items)
                for case in items:
                    torch.cuda.reset_peak_memory_stats(rank)
                    row = runner.run_case(state, case)
                    row["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(rank))
                    peak_allocated = max(peak_allocated, row["peak_allocated_bytes"])
                    peak_reserved = max(peak_reserved, int(torch.cuda.max_memory_reserved(rank)))
                    rows.append(row)
                    # Return the case's cached blocks so the next case (and the candidate's own
                    # scratch) starts from what is really live, not the previous case's peak.
                    torch.cuda.empty_cache()
                    if progress is not None and state.logical_rank == 0:
                        progress(f"{case['case_id']} {gid} {row['tflops']:.1f} TFLOPS {row['median_ms']:.3f} ms")
                runner.close_group(state)
                del state
                group_seconds[gid] = time.perf_counter() - t0
            dist.barrier(group=host_group)
        gathered: list[Any] = [None] * world
        dist.all_gather_object(gathered, {
            "rank": rank,
            "rows": rows,
            "peak_allocated": peak_allocated,
            "peak_reserved": peak_reserved,
            "baseline_device_bytes": baseline_device_bytes,
            "group_seconds": group_seconds,
            "wall": time.perf_counter() - started,
        }, group=host_group)
        if rank != 0:
            return None
        by_case: dict[str, dict[int, dict[str, Any]]] = {}
        for item in gathered:
            for row in item["rows"]:
                by_case.setdefault(str(row["case_id"]), {})[int(row["rank"])] = row
        assembled = []
        for case in cases:
            per_rank = by_case.get(str(case["case_id"]), {})
            if set(per_rank) != set(range(int(case["W"]))):
                raise RuntimeError(f"case {case['case_id']} lacks rank records")
            lead = dict(per_rank[0])
            lead["rank_records"] = [
                {"rank": r, "rows_per_trial": per_rank[r]["rows_per_trial"],
                 "rank_trial_seconds": per_rank[r]["rank_trial_seconds"],
                 "peak_allocated_bytes": per_rank[r]["peak_allocated_bytes"]}
                for r in range(int(case["W"]))
            ]
            for r in range(1, int(case["W"])):
                if per_rank[r]["trial_seconds"] != lead["trial_seconds"]:
                    raise RuntimeError(f"case {case['case_id']} ranks disagree on merged times")
                for key in ("rel_l1", "row_rel_l2_p999", "row_rel_l2_max", "max_abs", "freivalds_rel_l2", "freivalds_worst_row"):
                    if key in per_rank[r] and (key not in lead or per_rank[r][key] > lead[key]):
                        lead[key] = per_rank[r][key]
                if "cossim" in per_rank[r] and ("cossim" not in lead or per_rank[r]["cossim"] < lead["cossim"]):
                    lead["cossim"] = per_rank[r]["cossim"]
            assembled.append(lead)
        scored_rows = [x for x in assembled if x["scored"]]
        edge_rows = [x for x in assembled if not x["scored"]]
        aggregate = aggregate_case_results(scored_rows)
        return {
            "schema_version": 2,
            "generator_provenance": GENERATOR_PROVENANCE,
            "candidate_sha256": digest,
            "entrypoints": list(ENTRYPOINTS),
            "hardware": inventory,
            "rank_identities": identities,
            "trials_per_case": trials,
            "warmups": warmups,
            "correctness_ok": True,
            "edge_ok": True,
            "fresh_input_ok": True,
            "rank_agreement_ok": True,
            "edge_case_ids": [x["case_id"] for x in edge_rows],
            "edge_cases": edge_rows,
            "cases": scored_rows,
            "allocator_peaks": [
                {"rank": x["rank"], "max_memory_allocated_bytes": x["peak_allocated"],
                 "max_memory_reserved_bytes": x["peak_reserved"], "baseline_device_bytes": x["baseline_device_bytes"],
                 "suite_wall_seconds": x["wall"],
                 "group_seconds": x["group_seconds"]}
                for x in gathered
            ],
            **aggregate,
        }
    finally:
        try:
            dist.barrier(group=host_group)
        finally:
            dist.destroy_process_group()
