#!/usr/bin/env python3
"""Generate the Judge-only final suite and the public suite for the eight-GPU Flux MoE task.

The Judge suite (about 1000 cases) is written to ``judge/private/final_cases.json``
(Judge-only path, never in a Work build context); the public suite to
``images/work/benchmark/public_cases.json``.  The draw is deterministic: a
``random.Random`` seeded from ``sha256("<suite_id>|<task_id>")`` unless
``--seed`` overrides it.  ``--redraw`` acknowledges that writing a new draw is a
new suite identity.

Sources of every case are recorded in the case itself (``tier``/``source``):
* ``paper``  - experiment points of Comet (arXiv:2502.19811 v3), FLUX
  (arXiv:2406.06858 v5) and bytedance/flux @ 19831ca2 (docs, examples, tuning
  and test defaults) that run on one 8-GPU node.  Their layer-0 output width is
  the papers' non-gated N = I / TP.
* ``model``  - real model configurations (Hugging Face config.json, checked
  2026-11-08) in tensor-, expert- and hybrid-parallel deployments with token
  counts drawn from training/prefill, decode and multi-layer regimes.  A gated
  (SwiGLU) expert's layer-0 output is gate and up fused: N = 2 * I / TP.
* ``real``   - the same deployments with routing replayed from the model's own
  router on mixed real text (prose, chat, code, math, QA): token windows of the
  fixtures ``scripts/capture_routing.py`` records and
  ``scripts/build_routing_fixtures.py`` trims, pinned by sha256.  Judge fixtures
  live on the Judge-only path ``judge/private/routing``; public ones ship in the
  Work image under ``benchmark/routing``.
* ``edge``   - correctness-only corners of the support domain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

TASK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_ROOT / "runtime"))
import moe_ag_runtime as rt  # noqa: E402

SUITE_ID = "final-holdout-v2"
TASK_ID = "flux_moe_ag_scatter_grouped_gemm_dual_h100_seed_v1"
FINAL_PATH = TASK_ROOT / "judge/private/final_cases.json"
PUBLIC_PATH = TASK_ROOT / "images/work/benchmark/public_cases.json"

# name: (hidden K, expert intermediate I, routed experts E, top-k, MoE layers)
JUDGE_MOE = {
    "DeepSeek-V3": (7168, 2048, 256, 8, 58),
    "Kimi-K2": (7168, 2048, 384, 8, 60),
    "GLM-4.5": (5120, 1536, 160, 8, 89),
    "GLM-4.5-Air": (4096, 1408, 128, 8, 45),
    "GLM-5": (6144, 2048, 256, 8, 75),
    "Ling-1T": (8192, 2048, 256, 8, 76),
    "Ling-flash-2.0": (4096, 1024, 256, 8, 31),
    "ERNIE-4.5-300B": (8192, 3584, 64, 8, 54),
    "ERNIE-4.5-21B": (2560, 1536, 64, 6, 28),
    "Qwen3-Coder-480B": (6144, 2560, 160, 8, 62),
    "Qwen3.5-397B": (4096, 1024, 512, 10, 60),
    "Qwen3-Next-80B": (2048, 512, 512, 10, 48),
    "Hunyuan-A13B": (4096, 3072, 64, 8, 32),
    "dots.llm1": (4096, 1408, 128, 6, 61),
    "gpt-oss-120b": (2880, 2880, 128, 4, 36),
    "gpt-oss-20b": (2880, 2880, 32, 4, 24),
    "Mixtral-8x7B": (4096, 14336, 8, 2, 32),
    "Mixtral-8x22B": (6144, 16384, 8, 2, 56),
    "Phi-3.5-MoE": (4096, 6400, 16, 2, 32),
    "Qwen1.5-MoE-A2.7B": (2048, 1408, 60, 4, 24),
    "Llama-4-Scout": (5120, 8192, 16, 1, 48),
    "Llama-4-Maverick": (5120, 8192, 128, 1, 24),
    "Arctic": (7168, 4864, 128, 2, 35),
    "DeepSeek-V2": (5120, 1536, 160, 6, 59),
    "LongCat-Flash": (6144, 2048, 512, 12, 0),
    "Granite-3.1-3B-A800M": (1536, 512, 40, 8, 32),
}
JUDGE_DENSE = {"Llama-3.1-70B": (8192, 28672, 80), "Qwen2.5-72B": (8192, 29568, 80), "Llama-3.1-405B": (16384, 53248, 126)}
PUBLIC_MOE = {
    "Qwen3-235B": (4096, 1536, 128, 8, 94),
    "Qwen3-30B": (2048, 768, 128, 8, 48),
    "MiniMax-M2": (3072, 1536, 256, 8, 62),
    "OLMoE-1B-7B": (2048, 1024, 64, 8, 16),
    "DeepSeek-V2-Lite": (2048, 1408, 64, 6, 26),
}
PUBLIC_DENSE = {"Llama-3.1-8B": (4096, 14336, 32), "Qwen3-8B": (4096, 12288, 36)}

# Models whose router decisions were captured (catalog name -> routing fixture).
JUDGE_ROUTING_DIR = TASK_ROOT / "judge/private/routing"
PUBLIC_ROUTING_DIR = TASK_ROOT / "images/work/benchmark/routing"
JUDGE_CAPTURED = {"gpt-oss-20b": "gpt-oss-20b", "Phi-3.5-MoE": "phi-3.5-moe", "Mixtral-8x7B": "mixtral-8x7b",
                  "Granite-3.1-3B-A800M": "granite-3.1-3b-a800m"}
PUBLIC_CAPTURED = {"OLMoE-1B-7B": "olmoe-1b-7b", "Qwen3-30B": "qwen3-30b-a3b"}
JUDGE_REAL_SCALE = {"tp8": 3, "tp4": 2, "tp2": 2, "ep": 3, "ep_dp": 1, "decode8": 2, "decode4": 1, "sequence": 1}
PUBLIC_REAL_SCALE = {"tp8": 1, "tp4": 1, "tp2": 1, "ep": 2, "ep_dp": 0, "decode8": 1, "decode4": 0, "sequence": 1}

# Comet routing std of per-expert token fraction -> expert-bias sigma (E=8, top-2),
# calibrated by simulation of rt.generate_routing's draw (std ~= 0.1 * sigma).
COMET_STD_SIGMA = {0.0: 0.0, 0.01: 0.09, 0.02: 0.2, 0.03: 0.3, 0.04: 0.41, 0.05: 0.49}
UNIFORM = {"kind": "uniform"}


def default_seed(suite_id: str = SUITE_ID, task_id: str = TASK_ID) -> int:
    digest = hashlib.sha256(f"{suite_id}|{task_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


DEFAULT_SEED = default_seed()


class Builder:
    def __init__(self, rng: random.Random, prefix: str, forbid: set[tuple[int, int, int, int]] | None = None):
        self.rng = rng
        self.prefix = prefix
        self.cases: list[dict[str, Any]] = []
        self.seeds: set[int] = set()
        self.forbid = forbid or set()
        self.skipped = 0

    def seed(self) -> int:
        while True:
            value = self.rng.randrange(1, 1 << 47)
            if value not in self.seeds:
                self.seeds.add(value)
                return value

    def add(self, family, tier, group, TP, EP, K, N, E, topk, M, routing, *, source, model,
            split="even", cv=0.0, layers=1, extra=None):
        W = len(rt.GROUPS[group][0])
        if family != "edge" and (K, N, E // EP, topk) in self.forbid:
            self.skipped += 1
            return None
        case = {
            "case_id": f"{self.prefix}-{len(self.cases):04d}-{family}",
            "family": family, "tier": tier, "group": group, "W": W, "TP": TP, "EP": EP,
            "K": K, "N": N, "E": E, "topk": topk, "M": int(M), "split": split, "split_cv": round(float(cv), 3),
            "routing": routing, "layers": layers, "seed": self.seed(), "source": source, "model": model,
        }
        if extra:
            case.update(extra)
        rt.validate_case(case)
        self.cases.append(case)
        return case

    # ---- random draws

    def routing(self) -> dict[str, Any]:
        r = self.rng.random()
        if r < 0.45:
            return {"kind": "prod", "sigma_e": round(self.rng.uniform(0.2, 0.4), 3),
                    "lam": round(self.rng.uniform(0.1, 0.5), 3), "seg_len": self.rng.choice((128, 256, 512, 1024))}
        if r < 0.60:
            return {"kind": "skew", "sigma_e": round(self.rng.uniform(0.6, 1.4), 3),
                    "lam": round(self.rng.uniform(0.0, 0.5), 3), "seg_len": self.rng.choice((256, 512, 1024))}
        if r < 0.75:
            return {"kind": "hot", "sigma_e": round(self.rng.uniform(0.1, 0.3), 3),
                    "hot": self.rng.choice((1, 2, 4)), "hot_boost": round(self.rng.uniform(1.0, 3.0), 3)}
        if r < 0.90:
            return {"kind": "segment", "sigma_e": round(self.rng.uniform(0.1, 0.3), 3),
                    "lam": round(self.rng.uniform(1.0, 3.0), 3), "seg_len": self.rng.choice((64, 128, 256, 512, 1024, 2048))}
        return dict(UNIFORM)

    def tokens(self, lo: int, hi: int, W: int, *, aligned: float = 0.7) -> int:
        value = math.exp(self.rng.uniform(math.log(lo), math.log(hi)))
        if self.rng.random() < aligned:
            value = max(W, int(round(value / 128.0)) * 128)
        return max(W, int(value))

    def strata(self, lo: int, hi: int, count: int, W: int) -> list[int]:
        """One draw per log-uniform stratum, so a model's token counts span the band."""
        if count <= 0:
            return []
        edges = [math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) * i / count) for i in range(count + 1)]
        return [self.tokens(int(edges[i]), int(edges[i + 1]), W) for i in range(count)]

    def group(self, W: int) -> str:
        r = self.rng.random()
        if W == 8:
            return "w8s" if r < 0.15 else "w8"
        if W == 4:
            return "w4x" if r < 0.25 else ("w4s" if r < 0.4 else "w4")
        if W == 2:
            return "w2x" if r < 0.2 else "w2"
        raise ValueError(W)

    def split(self) -> tuple[str, float]:
        if self.rng.random() < 0.15:
            return "uneven", round(self.rng.uniform(0.1, 0.5), 3)
        return "even", 0.0


def gated_n(I: int, TP: int) -> int:
    return 2 * I // TP


def fits(K: int, N: int, E_rank: int, M: int) -> bool:
    return (N <= rt.MAX_N and N % 8 == 0 and E_rank <= rt.MAX_E_RANK and E_rank * N * K * 2 <= rt.MAX_WEIGHT_BYTES
            and M * K * 2 + (1 << 20) <= rt.WORKSPACE_BYTES)


# --------------------------------------------------------------------------- paper tier


def paper_cases(b: Builder) -> None:
    src = "Comet arXiv:2502.19811v3"
    mixtral = (4096, 14336)
    # Fig 10: single MoE layer vs M, EP=8 (Mixtral expert, E=8/top-2 and E=32/top-4).
    for E, topk in ((8, 2), (32, 4)):
        for M in (2048, 4096, 8192, 16384, 32768):
            b.add("ep", "paper", "w8", 1, 8, mixtral[0], mixtral[1], E, topk, M, dict(UNIFORM),
                  source=f"{src} Fig 10 (E={E}, top-{topk}); Fig 11 at M=16384" if (E, M) == (8, 16384) else f"{src} Fig 10 (E={E}, top-{topk})",
                  model="Mixtral-8x7B expert")
    # Fig 12: (TP, EP) on 8 GPUs, E=8, top-2, M=8192 (TP=1/EP=8 is the Fig 10 point).
    for TP, EP in ((2, 4), (4, 2), (8, 1)):
        family = "tp8" if EP == 1 else "ep"
        b.add(family, "paper", "w8", TP, EP, mixtral[0], mixtral[1] // TP, 8, 2, 8192, dict(UNIFORM),
              source=f"{src} Fig 12 (TP={TP}, EP={EP})" + ("; Comet AE grid_test.sh" if (TP, EP) == (2, 4) else ""),
              model="Mixtral-8x7B expert")
    # Fig 13: E and top-k at M=16384, EP=8 (E=8/top-2 is the Fig 10 point).
    for E in (8, 16):
        for topk in (2, 3, 4, 5):
            if (E, topk) == (8, 2):
                continue
            b.add("ep", "paper", "w8", 1, 8, mixtral[0], mixtral[1], E, topk, 16384, dict(UNIFORM),
                  source=f"{src} Fig 13 (E={E}, top-{topk})", model="Mixtral-8x7B expert")
    # Fig 14 left: routing std across experts, M=8192, E=8, top-2, EP=8.
    for std, sigma in COMET_STD_SIGMA.items():
        routing = dict(UNIFORM) if sigma == 0 else {"kind": "prod", "sigma_e": sigma, "lam": 0.0, "seg_len": 256}
        if std == 0.0:
            continue  # identical to the Fig 10 uniform point at M=8192
        b.add("ep", "paper", "w8", 1, 8, mixtral[0], mixtral[1], 8, 2, 8192, routing,
              source=f"{src} Fig 14 left (std={std})", model="Mixtral-8x7B expert")
    # FLUX dense all-gather + GEMM, GPT-3 175B MLP (k=12288, n=49152 global -> 6144 per rank).
    fsrc = "FLUX arXiv:2406.06858v5"
    for M in (1024, 2048, 4096, 8192):
        b.add("tp8", "paper", "w8", 8, 1, 12288, 6144, 1, 1, M, dict(UNIFORM),
              source=f"{fsrc} Figs 4, 8, 10, 12, 13 (AG, m={M}); flux tools/tune_ag_gemm_kernel.py", model="GPT-3 175B MLP")
    for M in (64, 512):
        b.add("decode", "paper", "w8", 8, 1, 12288, 6144, 1, 1, M, dict(UNIFORM),
              source=f"{fsrc} Fig 14 (AG, decode m={M})", model="GPT-3 175B MLP")
    for M in (256,):
        b.add("decode", "paper", "w8", 8, 1, 12288, 6144, 1, 1, M, dict(UNIFORM),
              source="bytedance/flux@19831ca2 tools/tune_ag_gemm_kernel.py (m=256)", model="GPT-3 175B MLP")
    # bytedance/flux layer-0 (docs/performance.md, tuning guide, examples, test defaults); K=8192, ffn 8192.
    rsrc = "bytedance/flux@19831ca2"
    flux_random = {"kind": "skew", "sigma_e": 0.7, "lam": 0.0, "seg_len": 256}  # per-expert U(0,1) weights
    for E, topk, TP, EP, ref in ((32, 4, 8, 1, "docs/performance.md:32-46; docs/tuning_guide.md; examples/moe_layer0.py"),
                                 (32, 4, 4, 2, "docs/performance.md:48-60"),
                                 (8, 2, 8, 1, "docs/performance.md:62-74"),
                                 (8, 2, 4, 2, "docs/performance.md:77-88"),
                                 (32, 5, 8, 1, "test/python/moe_ag_scatter/test_moe_ag.py defaults")):
        family = "tp8" if EP == 1 else "ep"
        b.add(family, "paper", "w8", TP, EP, 8192, 8192 // TP, E, topk, 8192, dict(flux_random),
              source=f"{rsrc} {ref}", model="Flux synthetic layer-0 (K=8192, ffn=8192)")
    b.add("ep", "paper", "w8", 2, 4, mixtral[0], mixtral[1] // 2, 8, 2, 1024, dict(UNIFORM),
          source=f"{rsrc} examples/moe.py (Mixtral shape, TP=2, EP=4)", model="Mixtral-8x7B expert")
    # Dense AG stress test: K=4608, N=16128 global, random local M (draw three).
    for M in (512, 4096, 10240):
        b.add("tp8" if M > 1024 else "decode", "paper", "w8", 8, 1, 4608, 16128 // 8, 1, 1, M, dict(UNIFORM),
              source=f"{rsrc} test/python/stress/stress_ag_gemm.py (random M, drawn {M})", model="Flux dense stress")


# --------------------------------------------------------------------------- model tier


def model_cases(b: Builder, moe: dict, dense: dict, scale: dict[str, int]) -> None:
    for name, (K, I, E, topk, layers) in moe.items():
        tag = f"HF config {name}"
        # tensor parallel, 8 GPUs (training / prefill token counts)
        N = gated_n(I, 8)
        for M in b.strata(4096, 131072, scale["tp8"], 8):
            if fits(K, N, E, M):
                s, cv = b.split()
                b.add("tp8", "model", b.group(8), 8, 1, K, N, E, topk, M, b.routing(), source=tag, model=name, split=s, cv=cv)
        # tensor parallel, 4 GPUs
        N = gated_n(I, 4)
        for M in b.strata(2048, 65536, scale["tp4"], 4):
            if fits(K, N, E, M):
                s, cv = b.split()
                b.add("tp4", "model", b.group(4), 4, 1, K, N, E, topk, M, b.routing(), source=tag, model=name, split=s, cv=cv)
        # tensor parallel, 2 GPUs (large per-rank weights)
        N = gated_n(I, 2)
        for M in b.strata(2048, 32768, scale["tp2"], 2):
            if fits(K, N, E, M):
                b.add("tp2", "model", b.group(2), 2, 1, K, N, E, topk, M, b.routing(), source=tag, model=name)
        # expert and hybrid parallel, 8 and 4 GPUs
        for W, TP, EP, count, lo, hi in ((8, 1, 8, scale["ep"], 4096, 65536), (8, 2, 4, scale["ep"], 4096, 65536),
                                         (8, 4, 2, scale["ep"], 4096, 65536), (4, 1, 4, 1, 2048, 32768),
                                         (4, 2, 2, 1, 2048, 32768)):
            if E % EP or E // EP < 1:
                continue
            N = gated_n(I, TP)
            for M in b.strata(lo, hi, count, W):
                if fits(K, N, E // EP, M):
                    s, cv = b.split()
                    b.add("ep", "model", b.group(W), TP, EP, K, N, E, topk, M, b.routing(), source=tag, model=name, split=s, cv=cv)
        # uneven tokens per rank (data-parallel attention feeding expert parallelism)
        if E % 8 == 0 and fits(K, gated_n(I, 1), E // 8, 32768):
            M = b.tokens(8192, 65536, 8)
            b.add("ep", "model", "w8", 1, 8, K, gated_n(I, 1), E, topk, M, b.routing(), source=tag + " (DP attention + EP)",
                  model=name, split="uneven", cv=round(b.rng.uniform(0.3, 0.8), 3))
        # decode
        for W, count, lo, hi in ((8, scale["decode8"], 64, 2048), (4, scale["decode4"], 32, 1024)):
            N = gated_n(I, W)
            for M in b.strata(lo, hi, count, W):
                if fits(K, N, E, M):
                    b.add("decode", "model", b.group(W), W, 1, K, N, E, topk, M, b.routing(), source=tag, model=name)
        # consecutive layers: a decode step and a prefill chunk through several MoE layers
        if layers:
            N = gated_n(I, 8)
            if fits(K, N, E, 512):
                b.add("sequence", "model", "w8", 8, 1, K, N, E, topk, b.tokens(64, 512, 8), b.routing(), source=tag + " (decode step)",
                      model=name, layers=min(layers, 32))
            if fits(K, N, E, 16384):
                b.add("sequence", "model", "w8" if b.rng.random() < 0.7 else "w8s", 8, 1, K, N, E, topk, b.tokens(4096, 16384, 8),
                      b.routing(), source=tag + " (prefill chunk)", model=name, layers=8)
    for name, (K, I, layers) in dense.items():
        tag = f"HF config {name} (dense MLP, E=1)"
        for W, family, lo, hi, count in ((8, "tp8", 4096, 65536, 4), (4, "tp4", 2048, 32768, 2), (8, "decode", 64, 2048, 2)):
            N = gated_n(I, W)
            for M in b.strata(lo, hi, count, W):
                if fits(K, N, 1, M):
                    b.add(family, "model", b.group(W), W, 1, K, N, 1, 1, M, dict(UNIFORM), source=tag, model=name)


# --------------------------------------------------------------------------- real tier


def judge_routing_dir() -> Path:
    """The Judge routing fixtures: the development copy on the Judge-only path, else the
    dataset asset assets/fetch.sh places under <INFRATHON_ASSETS_ROOT>/<task directory>/judge/routing
    (a tree without the repository copies, such as the public tree)."""
    if JUDGE_ROUTING_DIR.is_dir():
        return JUDGE_ROUTING_DIR
    assets_root = Path(os.environ.get("INFRATHON_ASSETS_ROOT") or "~/.cache/infrathon/assets").expanduser()
    return assets_root / TASK_ROOT.name / "judge/routing"


def fixture_record(root: Path, fixture: str) -> dict[str, Any]:
    record = json.loads((root / f"{fixture}.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((root / f"{fixture}.npy").read_bytes()).hexdigest()
    if record.get("sha256") != digest or record.get("name") != fixture:
        raise RuntimeError(f"routing fixture {fixture} differs from its record")
    return record


def real_cases(b: Builder, catalog: dict, captured: dict[str, str], root: Path, scale: dict[str, int]) -> None:
    """Deployments of the captured models, routed by token windows of their own router decisions."""
    for name, fixture in captured.items():
        K, I, E, topk, _layers = catalog[name]
        record = fixture_record(root, fixture)
        if (record["experts"], record["topk"]) != (E, topk):
            raise RuntimeError(f"routing fixture {fixture} does not match {name}'s configuration")
        tokens, kept = int(record["tokens"]), int(record["layers"])
        tag = f"{name} router on mixed real text ({record['repo']}@{record['revision'][:12]})"

        def add(family, W, TP, EP, M, *, group=None, note="", **extra):
            N = gated_n(I, TP)
            if E % EP or M > tokens or not fits(K, N, E // EP, M):
                return
            routing = {"kind": "captured", "fixture": fixture, "layer": b.rng.randrange(kept),
                       "start": b.rng.randrange(tokens - M + 1), "sha256": record["sha256"]}
            b.add(family, "real", group or b.group(W), TP, EP, K, N, E, topk, M, routing,
                  source=tag + note, model=name, **extra)

        for M in b.strata(4096, 32768, scale["tp8"], 8):
            add("tp8", 8, 8, 1, M)
        for M in b.strata(2048, 32768, scale["tp4"], 4):
            add("tp4", 4, 4, 1, M)
        for M in b.strata(2048, 16384, scale["tp2"], 2):
            add("tp2", 2, 2, 1, M)
        for TP, EP in ((1, 8), (2, 4), (4, 2))[: scale["ep"]]:
            add("ep", 8, TP, EP, b.tokens(4096, 32768, 8))
        for _ in range(scale["ep_dp"]):
            add("ep", 8, 1, 8, b.tokens(8192, 32768, 8), group="w8", note=" (DP attention + EP)",
                split="uneven", cv=round(b.rng.uniform(0.3, 0.8), 3))
        for M in b.strata(64, 1024, scale["decode8"], 8):
            add("decode", 8, 8, 1, M)
        for M in b.strata(32, 512, scale["decode4"], 4):
            add("decode", 4, 4, 1, M)
        for _ in range(scale["sequence"]):
            add("sequence", 8, 8, 1, b.tokens(64, 512, 8), group="w8", note=" (decode step, 8 consecutive layers)", layers=8)
            add("sequence", 8, 8, 1, b.tokens(4096, 16384, 8), group="w8", note=" (prefill chunk, 8 consecutive layers)",
                layers=8)


# --------------------------------------------------------------------------- edges


def edge_cases(b: Builder, public: bool) -> None:
    prod = {"kind": "prod", "sigma_e": 0.3, "lam": 0.3, "seg_len": 256}
    shift = 64 if public else 0
    specs = [
        ("w2", 2, 1, 1024 + shift, 512, 32, 4, 2, prod, {}),                      # one token per rank
        ("w8", 8, 1, 2048, 768 - shift, 128, 8, 8, prod, {}),                     # one token per rank, 8 ranks
        ("w3", 3, 1, 1088 + shift, 1000, 8, 2, 3001, prod, {}),                    # W=3, unaligned K, N, M
        ("w6", 6, 1, 2048, 1408 - shift, 64, 6, 6007, prod, {}),                   # W=6
        ("w6", 2, 3, 1536, 896 + shift * 2, 48, 4, 4801, prod, {}),               # W=6 hybrid TP2 x EP3
        ("w8", 8, 1, 2048, 768 + shift, 64, 1, 4096, {"kind": "forced", "expert": 5}, {}),  # one forced expert
        ("w8", 1, 8, 2048, 256 + shift, 64, 2, 16, prod, {}),                     # EP, most ranks get no rows
        ("w8", 1, 8, 4096, 1024 + shift, 8, 2, 8192, {"kind": "hot", "sigma_e": 0.0, "hot": 1, "hot_boost": 8.0}, {}),  # all rows on one rank
        ("w4", 4, 1, 64, 8 + shift, 512, 16, 4096, prod, {}),                             # K minimum (the Judge's also has N minimum), 512 experts, top-16
        ("w8", 8, 1, 16384, 16384 // 8 * 8 if not public else 2048, 2, 1, 16384, prod, {}),  # K maximum
        ("w2", 2, 1, 1024 + shift, 16384, 4, 2, 2048, prod, {}),                          # N maximum
        ("w8", 8, 1, 2048, 512 + shift, 512, 10, 8192, {"kind": "skew", "sigma_e": 4.0, "lam": 0.0, "seg_len": 256}, {}),  # extreme skew, empty experts
        ("w8", 8, 1, 4096, 1792 - shift, 8, 2, 8192, {"kind": "segment", "sigma_e": 0.2, "lam": 3.0, "seg_len": 512}, {}),  # strong locality
        ("w8", 8, 1, 2048, 768 + shift, 128, 8, 8192, prod, {"n_massive": 32, "massive_scale": 200.0}),  # precision stress
        ("w4s", 4, 1, 3072 - shift, 640, 96, 6, 12288, prod, {}),                 # shuffled ranks
        ("w4x", 1, 4, 2048, 3072 - shift, 64, 8, 8192, prod, {}),                          # cross-NUMA EP
        ("w2x", 2, 1, 4096, 2816 - shift, 16, 4, 16384, prod, {}),                         # cross-NUMA pair
    ]
    for group, TP, EP, K, N, E, topk, M, routing, extra in specs:
        split, cv = ("uneven", 1.0) if group == "w6" and TP == 2 else ("even", 0.0)
        b.add("edge", "edge", group, TP, EP, K, N, E, topk, M, routing, source="support-domain corner",
              model="edge", split=split, cv=cv, extra=extra or None)
    # uneven extreme and a multi-layer edge
    b.add("edge", "edge", "w8", 8, 1, 2048, 256 + shift, 128, 8, 4096, prod, source="support-domain corner", model="edge",
          split="uneven", cv=1.0)
    b.add("edge", "edge", "w8", 1, 8, 2048, 512 + shift, 64, 8, 2048, prod, source="support-domain corner (EP, uneven)",
          model="edge", split="uneven", cv=0.9)


def public_anchor_cases(b: Builder) -> None:
    """Paper-like public points with shapes disjoint from every Judge case."""
    b.add("ep", "paper", "w8", 1, 8, 4096, 12288, 8, 2, 8192, dict(UNIFORM), source="Comet Fig 10-like, synthetic expert (I=12288)",
          model="synthetic")
    b.add("ep", "paper", "w8", 2, 4, 4096, 6144, 16, 2, 8192, dict(UNIFORM), source="Comet Fig 12-like, synthetic expert (I=12288)",
          model="synthetic")
    b.add("tp8", "paper", "w8", 8, 1, 8192, 4096, 1, 1, 4096, dict(UNIFORM), source="FLUX dense AG-like (k=8192, n=32768)",
          model="synthetic")
    b.add("decode", "paper", "w8", 8, 1, 8192, 4096, 1, 1, 256, dict(UNIFORM), source="FLUX decode-like (k=8192, n=32768)",
          model="synthetic")
    b.add("tp8", "paper", "w8", 8, 1, 6144, 1536, 16, 4, 8192, {"kind": "skew", "sigma_e": 0.7, "lam": 0.0, "seg_len": 256},
          source="Flux layer-0-like (K=6144, ffn=12288)", model="synthetic")
    b.add("ep", "paper", "w8", 4, 2, 6144, 3072, 16, 4, 8192, {"kind": "skew", "sigma_e": 0.7, "lam": 0.0, "seg_len": 256},
          source="Flux layer-0-like TP4 EP2", model="synthetic")


# --------------------------------------------------------------------------- checks and estimates


def identity(case: dict[str, Any]) -> tuple[int, int, int, int]:
    return rt.case_identity(case)


# The cost model below sums to 416 s for final-holdout-v2; the seed's Judge worker took 683 s
# on 8 x H100 SXM (2026-11-08, two runs), so case estimates are scaled to match.
SXM_TIME_SCALE = 683 / 416


def est_seconds(case: dict[str, Any]) -> float:
    """Judge cost of one case of the seed on 8 x H100 SXM (seconds), scaled to the pilot."""
    return SXM_TIME_SCALE * _est_seconds_model(case)


def _est_seconds_model(case: dict[str, Any]) -> float:
    c = rt.validate_case(case)
    M, K, N, E_rank, topk, layers = c["M"], c["K"], c["N"], c["E_rank"], c["topk"], c["layers"]
    rows = M * topk / c["EP"]
    prepare = 0.004 + M * K * 2 / 150e9 + M * c["E"] * 4 / 400e9 + 2 * E_rank * N * K * 2 / 2e12
    golden = 2 * rows * N * K / 40e12 + rows * N * 10 / 1e12 + 0.0002 * E_rank
    freivalds = E_rank * N * K * 6 / 1.5e12 + rows * (K + N) * 6 / 1.5e12 + 0.002
    candidate = c["effective_flops"] / layers / 250e12 + 3e-4
    if layers > 1:
        return 6 * layers * (prepare + freivalds + candidate) + golden * 2 + 1.0
    calls = 1 + 2 * 7 + (6 if c["tier"] == "paper" else 0)
    return calls * (prepare + candidate) + 7 * freivalds + 2 * golden + 0.02


# One rank's device memory outside the per-case peak: CUDA context, NCCL, the 4 GiB
# symmetric workspace and the shard pools (measured 2026-11-08: 5.6-6.2 GiB on 8 x H100 NVL,
# 10.3 GiB on 8 x H100 SXM with NVSwitch).
BASELINE_GIB = 10.5
CAPABILITY_FLOOR_GIB = rt.MIN_DEVICE_MEMORY_BYTES / 2**30


def est_peak_gib(case: dict[str, Any]) -> float:
    """Upper bound of one rank's runtime memory for a case (GiB) with every token's rows on
    that rank (the worst routing); the candidate's own scratch comes on top."""
    c = rt.validate_case(case)
    M, K, N, E_rank, topk, layers = c["M"], c["K"], c["N"], c["E_rank"], c["topk"], c["layers"]
    rows = M * min(topk, E_rank)
    calls = layers if layers > 1 else 2  # a sequence holds every layer's call; a paper case's twin doubles one
    weights = E_rank * N * K * 2 * min(layers, 4)
    per_call = M * K * 2 + rows * N * 2 + rows * 4 * 3  # inputs, output, sorted ids
    routing = M * c["E"] * 4 + M * topk * 8 * 4  # logits and the sort of every (token, k)
    checks = N * K * 4 + 3 * (1 << 30) + rows * 4 * 2  # one expert in FP32, golden/Freivalds/checksum chunks, row errors
    return (weights + calls * per_call + routing + checks) / 2**30 + 1.0


def check_suite(name: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [c["case_id"] for c in cases]
    seeds = [c["seed"] for c in cases]
    if len(set(ids)) != len(ids) or len(set(seeds)) != len(seeds):
        raise RuntimeError(f"{name}: case ids or seeds repeat")
    peaks = [est_peak_gib(c) for c in cases]
    if max(peaks) + BASELINE_GIB > CAPABILITY_FLOOR_GIB - 2:
        worst = cases[peaks.index(max(peaks))]
        raise RuntimeError(f"{name}: case {worst['case_id']} estimated at {max(peaks):.1f} GiB plus a {BASELINE_GIB} GiB "
                           f"baseline, above the {CAPABILITY_FLOOR_GIB:.0f} GiB capability floor less 2 GiB")
    scored = [c for c in cases if c["family"] != "edge"]
    return {
        "cases": len(cases),
        "scored": len(scored),
        "edges": len(cases) - len(scored),
        "families": dict(Counter(c["family"] for c in scored)),
        "tiers": dict(Counter(c["tier"] for c in scored)),
        "groups": dict(Counter(c["group"] for c in cases)),
        "routing": dict(Counter(c["routing"]["kind"] for c in scored)),
        "uneven": sum(1 for c in scored if c["split"] == "uneven"),
        "models": len({c["model"] for c in scored}),
        "distinct_identities": len({identity(c) for c in scored}),
        "distinct_M": len({c["M"] for c in scored}),
        "max_peak_gib": round(max(peaks), 1),
        "est_minutes": round(sum(est_seconds(c) for c in cases) / 60, 1),
    }


EXPECTED_CASE_COUNT = 986  # 967 scored + 19 edge


def judge_suite(rng: random.Random) -> Builder:
    judge = Builder(rng, "final")
    edge_cases(judge, public=False)
    paper_cases(judge)
    model_cases(judge, JUDGE_MOE, JUDGE_DENSE, {"tp8": 6, "tp4": 4, "tp2": 3, "ep": 3, "decode8": 4, "decode4": 2})
    real_cases(judge, JUDGE_MOE, JUDGE_CAPTURED, judge_routing_dir(), JUDGE_REAL_SCALE)
    return judge


def generate_manifest(rng: random.Random | None = None) -> dict[str, Any]:
    """The Judge manifest drawn from ``rng`` (default: ``random.Random(DEFAULT_SEED)``)."""
    judge = judge_suite(rng if rng is not None else random.Random(DEFAULT_SEED))
    return {"schema_version": 2, "suite_id": SUITE_ID, "cases": judge.cases}


def build(seed: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    judge = judge_suite(random.Random(seed))
    judge_forbid = {identity(c) for c in judge.cases if c["family"] != "edge"}
    public = Builder(random.Random(seed ^ 0x9E3779B97F4A7C15), "public", forbid=judge_forbid)
    edge_cases(public, public=True)
    public_anchor_cases(public)
    model_cases(public, PUBLIC_MOE, PUBLIC_DENSE, {"tp8": 2, "tp4": 2, "tp2": 1, "ep": 1, "decode8": 2, "decode4": 1})
    real_cases(public, PUBLIC_MOE, PUBLIC_CAPTURED, PUBLIC_ROUTING_DIR, PUBLIC_REAL_SCALE)
    judge_ids = {identity(c) for c in judge.cases if c["family"] != "edge"}
    public_ids = {identity(c) for c in public.cases if c["family"] != "edge"}
    overlap = judge_ids & public_ids
    if overlap:
        raise RuntimeError(f"public and Judge share (K, N, E_rank, topk): {sorted(overlap)[:5]}")
    if judge.seeds & public.seeds:
        raise RuntimeError("public and Judge share seeds")
    report = {"judge": check_suite("judge", judge.cases), "public": check_suite("public", public.cases),
              "public_judge_identity_overlap": 0, "public_skipped_for_overlap": public.skipped}
    final = {"schema_version": 2, "suite_id": SUITE_ID, "cases": judge.cases}
    pub = {"schema_version": 2, "suite_id": "public-v2", "cases": public.cases}
    return final, pub, report


def _write(path: Path, payload: dict[str, Any], mode: int) -> str:
    encoded = (json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_bytes(encoded)
    tmp.chmod(mode)
    os.replace(tmp, path)
    return hashlib.sha256(encoded).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--redraw", action="store_true", help="acknowledge that writing is a new suite identity")
    parser.add_argument("--final-output", type=Path, default=FINAL_PATH)
    parser.add_argument("--public-output", type=Path, default=PUBLIC_PATH)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--output", type=Path, help="write only the Judge manifest here (mode 0600)")
    args = parser.parse_args(argv)
    if args.output is not None:
        if not args.redraw:
            raise SystemExit("refusing to draw: pass --redraw (with an explicit --seed) to mint a new suite identity")
        if args.output.exists() or args.output.is_symlink():
            raise SystemExit(f"refusing to replace existing manifest: {args.output}")
        digest = _write(args.output.absolute(), generate_manifest(random.Random(args.seed)), 0o600)
        print(json.dumps({"seed": args.seed, "sha256": digest, "suite_identity": "new"}, sort_keys=True))
        return 0
    final, pub, report = build(args.seed)
    if not args.report_only:
        if not args.redraw:
            raise SystemExit("refusing to write: pass --redraw to mint the suite identity")
        report["final_sha256"] = _write(args.final_output, final, 0o600)
        report["public_sha256"] = _write(args.public_output, pub, 0o644)
    report["seed"] = args.seed
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
