#!/usr/bin/env python3
"""Trusted runtime for the FastVideo-derived Attn-QAT training attention, BF16 H100 v1.

The measured operator is one NVFP4 fake-quantized attention training step -- a forward
call followed by a backward call on the same inputs (FastVideo `attn_qat_train`, the
Attn-QAT paper's two principles: the backward recomputes `P` in the forward's precision,
and the row statistic `D` comes from an auxiliary high-precision output `O'`):

    FQ(x)  per 16-element group: a = max|x|, sd = e4m3(a / 6), x_hat = e2m1_rtne(x / sd),
           FQ(x) = bf16(x_hat * sd)                       (no global scale, no smoothing)

    forward   Qf = FQ_D(q)  Kf = FQ_D(k)  Vf = FQ_D(v)
              S  = Qf Kf^T * softmax_scale * log2(e)
              m  = final row max of S
              P~ = exp2(S - m)          Pq = FQ_KV(P~)     l = sum_j bf16(P~_ij)
              o  = (sum_j Pq_ij Vf_j) / l                  O' = (sum_j P~_ij Vf_j) / l
              M  = m + log2(l)
    backward  Delta = rowsum(dO * bf16(O'))                P  = exp2(S - M)
              Pq = FQ_KV(P)   dV = Pq^T dO                 dP = dO Vf^T
              dS = P * (dP - Delta)
              dK = softmax_scale * dS^T Qf                 dQ = softmax_scale * dS Kf

The forward quantizes the *unnormalized* `P~` against the final row maximum: e4m3's
smallest magnitude is 2^-9, so a 16-group whose maximum probability is below
6 * 2^-10 = 5.86e-3 gets block scale zero and is flushed away.  Normalized probabilities
at S >= 10^3 are far below that, so quantizing a normalized `P` destroys the output
(DESIGN section 2.9 and section 3).  Taking the maximum from a statistics pre-pass also
makes `Pq` tile-order independent, which is what lets the accuracy gates be fixed numbers
instead of a differential band.

Everything the Judge trusts -- the clock, the gate arithmetic, the golden reference and
the input generators -- lives here and is captured before the candidate module is
imported.  The candidate only ever sees tensors.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable, Mapping, Sequence

FORWARD_ENTRYPOINT = "attn_qat_fwd"
BACKWARD_ENTRYPOINT = "attn_qat_bwd"
ENTRYPOINTS = (FORWARD_ENTRYPOINT, BACKWARD_ENTRYPOINT)
# Two workload domains, scored with equal weight.  Video families are full (non-causal)
# multi-head attention from diffusion-transformer training recipes; LLM families are
# causal grouped-query attention from decoder-only language-model training.
VIDEO_FAMILIES = ("wan13b", "wan14b", "ltx2", "kandinsky5", "cogvideox", "hunyuan", "real_video")
LLM_FAMILIES = ("llm_pretrain", "llm_long", "llm_sft", "llm_d256", "real_llm")
FAMILIES = VIDEO_FAMILIES + LLM_FAMILIES
DOMAINS = ("video", "llm")
FAMILY_DOMAIN: dict[str, str] = {
    **{name: "video" for name in VIDEO_FAMILIES},
    **{name: "llm" for name in LLM_FAMILIES},
}
REAL_FAMILIES = frozenset({"real_video", "real_llm"})
TIERS = ("recipe", "sweep", "real")

# Video lengths.  Tokens per latent frame: Wan 480p 30x52 = 1560, Wan 720p 45x80 = 3600,
# LTX-2 390, Kandinsky5 1536, CogVideoX 480x720 30x45 = 1350 after a 226-token text
# prefix, HunyuanVideo 544x960 34x60 = 2040 after a 256-token text prefix.  Every scored
# video length is `prefix + lattice * t`.  The hidden suite itself lives only in the
# Judge-only final manifest (scripts/generate_final_manifest.py); these tables are the rules
# it must obey.
LATTICE_WAN = 1560
LATTICE_WAN_720P = 3600
LATTICE_LTX2 = 390
LATTICE_KANDINSKY5 = 1536
LATTICE_COGVIDEOX = 1350
LATTICE_HUNYUAN = 2040
COGVIDEOX_TEXT_TOKENS = 226
HUNYUAN_TEXT_TOKENS = 256
FAMILY_LATTICES: dict[str, tuple[tuple[int, int], ...]] = {
    "wan13b": ((0, LATTICE_WAN),),
    "wan14b": ((0, LATTICE_WAN), (0, LATTICE_WAN_720P)),
    "ltx2": ((0, LATTICE_LTX2),),
    "kandinsky5": ((0, LATTICE_KANDINSKY5),),
    "cogvideox": ((COGVIDEOX_TEXT_TOKENS, LATTICE_COGVIDEOX),),
    "hunyuan": ((HUNYUAN_TEXT_TOKENS, LATTICE_HUNYUAN),),
    "real_video": ((0, LATTICE_WAN), (0, LATTICE_WAN_720P),
                   (COGVIDEOX_TEXT_TOKENS, LATTICE_COGVIDEOX)),
}
# LLM lengths are padded to a multiple of 64 tokens.
LLM_LENGTH_QUANTUM = 64
FAMILY_HEAD_DIMS: dict[str, frozenset[int]] = {
    "wan13b": frozenset({128}),
    "wan14b": frozenset({128}),
    "ltx2": frozenset({128}),
    "kandinsky5": frozenset({64}),
    "cogvideox": frozenset({64}),
    "hunyuan": frozenset({128}),
    "real_video": frozenset({64, 128}),
    "llm_pretrain": frozenset({128}),
    "llm_long": frozenset({128}),
    "llm_sft": frozenset({128}),
    "llm_d256": frozenset({256}),
    "real_llm": frozenset({128, 256}),
}
HEAD_DIMS = frozenset({64, 128, 256})
# Kandinsky5's text encoder pads to 512 tokens; its cross attention is the only one the
# shipped recipes route through this kernel.
KANDINSKY5_TEXT_TOKENS = 512
# The public tests' geometries `(h, h_kv, d)` and lengths.  Scored cases may share neither.
PUBLIC_GEOMETRIES = frozenset(
    {(4, 4, 128), (8, 8, 128), (24, 24, 128), (16, 16, 64), (24, 24, 64),
     (28, 4, 128), (14, 2, 128), (16, 8, 128), (20, 4, 256)}
)
PUBLIC_LENGTHS = frozenset(
    {2728, 2784, 3584, 6120, 7168, 9216, 9696, 9984, 12384, 14336, 18016, 20480, 26208, 27280,
     28672, 40960, 57440}
)
REAL_LAYER_PATTERN = re.compile(
    r"(wan(13|14)b|cogvideox2b)-b[0-9]{2}-s[0-9]{3}|(qwen3-14b|qwen35-4b)-l[0-9]{2}-x[0-9]{2}"
)

# ---------------------------------------------------------------------------------------
# Accuracy gates.
#
# Fixed global relative-L2 limits against the chunked fp32 golden.  A differential gate is
# unnecessary in the *forward*: `Pq` is defined against the final row statistic and is
# therefore tile-order independent, so every honest re-tiling agrees with the golden on `o`
# to three significant figures.
#
# `dv` is nevertheless the loosest gate, and for a backward reason rather than a forward one.
# The backward quantizes the *normalized* `P`, so a 16-group whose maximum probability sits
# near e4m3's block-scale floor of `6 * 2^-10 = 5.86e-3` is flushed to zero on one side of
# the floor and kept on the other.  Two honest implementations that accumulate `P` in a
# different order therefore disagree about which groups contribute to `dV`, and the
# disagreement is a whole group, not an ulp.  `dq` inherits a milder version of the same
# effect through the sink rows of the real capture.
#
# Calibrated by helper A2 over 27 cases (the v1 four-cell suite at every length,
# and all eleven edge fixtures) x four honest implementations (the seed, the 128x64 positive
# control and two intermediate re-tilings) x nineteen planted contract violations.  Under
# these constants no honest variant violates any gate on any case.  DESIGN section 5.5's
# provisional table cannot ship as written -- its `dv` limit of 3.0e-2 fails the seed itself
# on the C1 anchor (3.59e-2) and its `dq` limit of 2.0e-2 fails the positive control there.
#
# Every planted violation is still rejected, most of them on the edge fixture built to catch
# it before any scored case runs: P1/P6 at 686x on e1 (`dk`), P3 -- the narrowest, since it
# moves only `dv` -- at 3x to 32x over the raised `dv` limit on every C1 anchor and on e1,
# e4, e10, e11, P4/P13/P15 at 1700-1800x on e8 (`dk`), P5 at 430x on e5, P8 at 2890x, P9 at
# 459x on the rounding-tie fixture e6 (`o`), P11 at 20000x on e8, P19 (normalized `P`) at
# 627x on e5, P20 at 372x on e9.
# ---------------------------------------------------------------------------------------
GATE_TENSORS = ("o", "dq", "dk", "dv")
SCORED_GROUP = "scored"
REAL_GROUP = "real"
# The gate constants.  One block, so a recalibration is a one-line edit.
#
# Scored cases keep flat limits.  Their inputs are drawn from a hidden seed at a hidden
# slot, and an honest kernel's error varies with the draw -- `dq` spans 8.2e-3 to 2.5e-2
# across the three holdout draws of C1 alone -- so a per-draw limit cannot be calibrated
# for a case the calibration never saw.
O_REL = 1.0e-2
DQ_REL = 8.0e-2
DK_REL = 3.0e-2
DV_REL = 1.2e-1
SCORED_GATE_LIMITS: dict[str, float] = {
    "o": O_REL,
    "dq": DQ_REL,
    "dk": DK_REL,
    "dv": DV_REL,
}
# Real captured layers: 2.5x the worst honest error over the seed, the upstream-configuration
# control and an optimized kernel on all 32 layers (dq 4.29e-2, dk 1.35e-1 and dv 3.67e-2 at block
# 0, sigma 0.1), rounded up.  Under these limits no honest kernel fails and the paper's two wrong
# backward designs are caught: `Delta` from the quantized `o` on 29 of 32 layers, and no fake
# quantization of `P` on 21 of 32.  A single flat row is deliberate: captures are per deployment.
REAL_GATE_LIMITS: dict[str, float] = {
    "o": O_REL,
    "dq": 1.2e-1,
    "dk": 3.5e-1,
    "dv": 1.0e-1,
}
# The v3 captures, calibrated the same way over four honest kernels (the seed, two other
# statistics-pass tilings and a two-pass statistic) on every captured layer.  CogVideoX-2B keeps
# the Wan row except `dv`: on block 6 at t = 10 % the honest kernels reach dv 1.2e-1, because
# that layer holds many backward `P` groups right at the e4m3 flush floor, and which of them
# survive depends on the last bits of the row statistic.  The Qwen3 layers (causal GQA on real
# text) get their own row: worst honest dq 5.7e-2, dk 2.1e-2, dv 2.5e-2.
REAL_SOURCE_GATE_LIMITS: dict[str, dict[str, float]] = {
    "cogvideox2b": {**REAL_GATE_LIMITS, "dv": 3.0e-1},
}
REAL_LLM_GATE_LIMITS: dict[str, float] = {"o": O_REL, "dq": 1.5e-1, "dk": 8.0e-2, "dv": 8.0e-2}
REAL_LLM_GROUP = "real_llm"
# Edge fixtures have fixed inputs, so each one carries its own limits, computed as
# `round_up(max(2.5 * worst honest error, floor))` and never above the global cap.  The
# tight rows are what reject a partial cheat that only changes behaviour where the e4m3
# block scale underflows: e1 at `S = 64` has no group anywhere near the flush threshold, so
# a flush-only violation is invisible in its own error but immediately visible against e1's
# limits, which the flat scored limits alone would admit.
GATE_FLOORS: dict[str, float] = {"o": 5.0e-3, "dq": 1.5e-2, "dk": 1.0e-2, "dv": 2.0e-2}
GATE_CAPS: dict[str, float] = {"o": 1.0e-2, "dq": 1.5e-1, "dk": 3.0e-2, "dv": 3.0e-1}
FIXTURE_GATE_LIMITS: dict[str, dict[str, float]] = {
    "edge-e1-tiny": {"o": 1.0e-2, "dq": 1.5e-2, "dk": 1.5e-2, "dv": 2.0e-2},
    "edge-e2-unaligned": {"o": 5.0e-3, "dq": 5.0e-2, "dk": 1.5e-2, "dv": 3.0e-2},
    "edge-e3-cross-attention": {"o": 1.0e-02, "dq": 5.0e-02, "dk": 1.5e-02, "dv": 5.0e-02},
    # `S_kv = 515` leaves a five-element tail in the last 16-group of every backward `Pq`
    # row, and honest `dv` reaches 1.08e-1 there.
    "edge-e4-unaligned-cross": {"o": 8.0e-03, "dq": 1.5e-02, "dk": 1.0e-02, "dv": 3.0e-01},
    "edge-e5-zero-key-zero-v-groups": {
        "o": 5.0e-3, "dq": 1.5e-2, "dk": 1.0e-2, "dv": 2.0e-2,
    },
    "edge-e6-rtne-ties": {"o": 1.0e-2, "dq": 1.5e-2, "dk": 1.5e-2, "dv": 3.0e-2},
    # The real Wan capture: its sink head concentrates the whole gradient into a handful of
    # rows, so honest `dq` reaches 4.31e-2 there against 2.46e-2 anywhere else.
    "edge-e7-wan-grad-capture": {"o": 1.0e-2, "dq": 1.5e-1, "dk": 3.0e-2, "dv": 5.0e-2},
    "edge-e8-forced-sink": {"o": 1.0e-02, "dq": 2.0e-02, "dk": 1.5e-02, "dv": 2.0e-02},
    "edge-e9-tiny-do-scale": {"o": 5.0e-03, "dq": 3.0e-02, "dk": 1.0e-02, "dv": 2.0e-02},
    "edge-e10-ltx2-minimal": {"o": 1.0e-02, "dq": 3.0e-02, "dk": 3.0e-02, "dv": 1.0e-01},
    "edge-e11-head-dim-64": {"o": 1.0e-2, "dq": 1.5e-2, "dk": 1.0e-2, "dv": 1.0e-1},
    # The four v3 fixtures, by the same rule over the seed, a 64x64-tile re-tiling and a 64x128
    # statistics pass (the honest variant with the largest error on every one of them).
    "edge-e12-causal-gqa": {"o": 1.0e-2, "dq": 1.5e-2, "dk": 1.5e-2, "dv": 5.0e-2},
    "edge-e13-causal-sink": {"o": 5.0e-3, "dq": 3.0e-2, "dk": 2.0e-2, "dv": 2.0e-2},
    "edge-e14-head-dim-256": {"o": 5.0e-3, "dq": 2.0e-2, "dk": 1.0e-2, "dv": 3.0e-2},
    "edge-e15-head-dim-256-full": {"o": 8.0e-3, "dq": 2.0e-2, "dk": 2.0e-2, "dv": 3.0e-2},
}
# When the golden output is exactly zero (`dq` and `dv` at e5, where `k = 0`) the relative
# statistic is undefined, so the gate becomes an absolute one.  The seed returns exactly
# zero there, so this is the maximum possible margin, not a slack.
ZERO_REF_ABS = 1.0e-4
GATE_CALIBRATION = "a2-27-cases-4-honest-19-planted-2026-09-06"

REFERENCE_CHUNK = 1024
# The generator's own reference attention (which supplies the `o` that `do` is drawn
# against) runs at a smaller chunk: it is batched over all heads at once, so 1024 query
# rows against 47616 keys on 32 heads would need 12 GiB of fp32 scratch.
GENERATOR_CHUNK = 256
MXFP_BLOCK_SIZE = 16
LOG2E = 1.44269504
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E4M3_MAX = 448.0

MAX_CASE_INPUT_BYTES = 4 * (1 << 30)
MAX_SUITE_INPUT_BYTES = 96 * (1 << 30)
MAX_RESIDENT_BYTES = 60 * (1 << 30)
GENERATOR_PROVENANCE = "fastvideo-7f83164-derived-attn-qat-train-v1-judge-port"
FIXTURE_NAME = "e7_wan13b_grad_b29_t1_s8192.pt"
# The e7 capture is recipe-bound, not byte-bound.  Every deployment runs
# `fixtures/recipe/build_fixture.sh` once at build time and bakes its own capture, the
# same way the final holdout is drawn per build, so the bytes differ between
# deployments and cross-GPU numeric drift is expected.  What is identical everywhere
# is the recipe: `fixtures/recipe/RECIPE.json` freezes the model revision, the upstream
# commit, the prompt, the seeds, the geometry, the block, the sigma, the slicing and
# the per-tensor sanity bands.  On load the runtime verifies, in order, that the
# manifest baked next to the fixture describes those exact bytes, that the manifest
# names the RECIPE.json shipped in this runtime's own build context, and that every
# tensor meets the recipe's shape, dtype, finiteness and sanity bands -- so a wrong
# capture fails closed even though no byte digest is pinned.
FIXTURE_MANIFEST_NAME = "e7_wan13b_grad_b29_t1_s8192.manifest.json"
FIXTURE_RECIPE_NAME = "RECIPE.json"
FIXTURE_RECIPE_ID = "e7-wan13b-qad-stage1-grad-b29-t1-s8192-v1"
FIXTURE_IDENTITY = "per-deployment-capture"
# The capture is stored in the model's own [B, S, H, D] layout and permuted to the
# kernel's [B, H, S, D] contract on load.
FIXTURE_SHAPE = (1, 8192, 12, 128)
# The four tensors the kernel contract consumes.  `o` is captured and validated too --
# it is the orientation material a wrong capture would fail on first -- but it is never
# handed to the case: the golden reference computes its own.
FIXTURE_TENSORS = ("q", "k", "v", "do")
FIXTURE_VALIDATED_TENSORS = ("q", "k", "v", "o", "do")
FIXTURE_CASE_ID = "edge-e7-wan-grad-capture"
REAL_LAYER_RECIPE_NAME = "REAL_LAYERS.json"
REAL_LAYER_MANIFEST_NAME = "real_layers.manifest.json"
# Captures from the second recipe (CogVideoX and the Qwen3 LLMs) carry their own manifest, so
# adding them left the Wan captures and their recipe untouched.
REAL_LAYER_V2_RECIPE_NAME = "REAL_LAYERS_V2.json"
REAL_LAYER_RECIPES = {
    "wan13b": REAL_LAYER_RECIPE_NAME,
    "wan14b": REAL_LAYER_RECIPE_NAME,
    "cogvideox2b": REAL_LAYER_V2_RECIPE_NAME,
    "qwen3-14b": REAL_LAYER_V2_RECIPE_NAME,
    "qwen35-4b": REAL_LAYER_V2_RECIPE_NAME,
}

# Per-trial seed derivation.  Every trial draws fresh values into a fresh allocation, and
# the decoy step that precedes each timed trial draws from a different stream again, so a
# candidate cannot cache a result keyed on the input pointers or the input bytes.
TRIAL_SEED_BASE = 0x7A1A_0000
DECOY_SEED_BASE = 0xDEC0_0000
WARMUP_SEED_XOR = 0x5A5A_5A5A
GENERATOR_WARMUP_XOR = 0x0BAD_0BAD

# The saved bundle may not exceed what upstream itself keeps: `Qf, Kf, Vf` (the three
# fake-quantized inputs), `O'` (the high-precision output) and the fp32 row statistic `M`.
# No slack: an extra row vector would also buy a cheaper route to the `l`-from-`Pq` cheat.
SAVED_WALK_MAX_DEPTH = 6
SAVED_WALK_MAX_NODES = 4096

# ---------------------------------------------------------------------------------------
# Input family: q, k, v.
#
# Ported from `iteration_docs/attn-qat-train-sm90-h100-v1/lab/family/qkv_family.py`, which
# is helper A's primal-only wrapper around the rCM attention task's generator, calibrated by
# that task's helper B on 18 real Wan2.1-T2V-1.3B captures.  Same draw order, same helper
# structure, same ranges except for the six entries helper B patched for this task
# (`PRIMAL_RANGE_PATCH`, marked below with the value they replace).
#
# Why the patch matters: NVFP4's error is monotone in the within-16-group dynamic range
# `max|x| / rms`.  The real captures sit at q 2.16-2.61, k 2.14-2.64, v 2.10-2.24 (an iid
# normal gives 2.110); the unpatched family draws q 2.56-2.93, k 2.63-2.79, v 2.34-2.43, so
# it never produces the near-Gaussian mid-network regime and makes fake quantization look
# uniformly harder than production.  The patched ranges realise q 2.34-2.62, k 2.24-2.67,
# v 2.13-2.25.
# ---------------------------------------------------------------------------------------
RANGES = {
    # primal: per-head channel-std profile.  Patched from (0.30, 0.85) / (0.30, 0.85) /
    # (0.20, 0.55): the log-std to realised group-ratio map is 0.00 -> 2.110, 0.20 -> 2.184,
    # 0.50 -> 2.443, 0.85 -> 2.751, and the measured band pins the upper end near 0.62.
    "chan_sigma_q": (0.02, 0.62),
    "chan_sigma_k": (0.02, 0.62),
    "chan_sigma_v": (0.02, 0.24),
    # "hot" channel pairs shared between q and k of a head -- the attention-sink mechanism.
    # They come in adjacent pairs because RoPE rotates channel pairs.
    "n_hot_pairs": (0, 1),
    "hot_amp": (2.0, 12.0),
    "sink_n_hot_pairs": 2,
    "sink_hot_amp": (28.0, 48.0),
    "sink_second_pair_ratio": 0.535,
    "hot_q_over_k": (0.55, 1.00),
    "v_hot_amp": (1.5, 4.5),
    "v_n_hot": (0, 3),
    # primal: token scale, shared across heads (qk-RMSNorm spans the whole hidden state)
    "tok_sigma": (0.02, 0.18),
    # primal: channel bias, constant across tokens
    "q_bias_frac": (0.008, 0.056),
    "k_bias_frac": (0.000, 0.082),
    "v_bias_frac": (0.000, 0.017),
    # Patched from (0.8, 17.8) / (0.8, 19.3) / (0.5, 5.2).  The 17.8 is a real Wan number,
    # but in the captures it is one channel in 1536, not `round(frac * d)` per head, and the
    # density is what pushed the realised group ratio to 2.74.
    "q_bias_mag": (0.5, 5.0),
    "k_bias_mag": (0.5, 5.0),
    "v_bias_mag": (0.4, 2.5),
    # primal: overall scale
    "q_rms": (1.05, 3.00),
    "k_rms": (1.05, 3.10),
    "v_rms": (0.90, 2.15),
    # primal: per-head softmax sharpness, as a median row max-probability target
    "maxp_ordinary": (0.006, 0.90),
    "maxp_uniform": (0.0003, 0.010),
    "maxp_identity": (0.004, 0.08),
    "sink_onehot_prob": 0.35,
    "maxp_sink_onehot": (0.90, 0.99995),
    "maxp_sink_soft": (0.05, 0.90),
    "n_uniform_heads": (2, 3),
    "identity_beta": (0.55, 0.95),
    # Stream-preserving draws.  The lineage generator also produced forward-mode tangents,
    # and drew these eight parameters from the same generator immediately after the primal
    # ones.  This task has no tangents, but the draws are kept so the parameter stream --
    # and therefore every `q, k, v` helper A and helper B calibrated against -- is
    # bit-identical to the lineage.  Nothing below reads them.
    "r_q": (0.080, 5.00),
    "r_k_over_r_q": (0.60, 1.20),
    "r_v_over_r_q": (1.25, 3.05),
    "tangent_chan_mix": (0.25, 0.70),
    "tangent_chan_jitter": (0.10, 0.60),
    "tangent_tok_sigma": (0.50, 1.20),
    "p_massive": (0.0, 0.025),
    "massive_scale": (15.0, 120.0),
    "rho_qk_sigma": 0.24,
    "rho_qk_clip": 0.70,
    "rho_v_mean": 0.064,
    "rho_v_sigma": 0.19,
    "rho_v_clip": (-0.10, 0.62),
    # solver for the per-head logit scale
    "maxp_probe_rows": 32,
    "maxp_bisect_steps": 22,
    "maxp_log_tau_span": 7.0,
}

# ---------------------------------------------------------------------------------------
# Input family: `do`, the gradient of the training loss with respect to the attention
# output.  Ported verbatim from `lab/family/do_family.py`, which helper B calibrated on ten
# real Wan2.1-T2V-1.3B QAD stage-1 training-step captures.  `do` is not a rescaled `o`
# (median cos = 0.000 +- 0.008), its within-16-group dynamic range is Gaussian, its channel
# profile is a flat floor plus a couple of sparse hot channels, and its token norms are
# heavy-tailed with genuine outliers where the primal has none.
# ---------------------------------------------------------------------------------------
DO_RANGES = {
    # measured per-16-channel group ratio for `do`: mean 2.11-2.14 (iid normal 2.110)
    "chan_sigma": (0.00, 0.08),
    # measured chan_std max/median over the 128 channels: 1.6 (xattn) .. 5.0 (b00_t0)
    "n_hot_heads": (1, 2),
    "hot_amp": (1.5, 5.0),
    # measured token-norm p99/p50 2.75 .. 10.2; the bulk stays narrow and the tail comes
    # from the sparse massive-token boost below
    "tok_sigma": (0.30, 0.65),
    # measured fraction of tokens above 5x the per-head median: 0.0000 .. 0.0200
    "p_massive": (0.000, 0.010),
    # measured max/median token-norm ratio: 5.7 .. 325
    "massive_scale": (4.0, 60.0),
    # cos(do_t, o_t) has median 0 with p1/p99 out to -0.51/+0.51, wider than the +-0.21 two
    # independent isotropic 128-d vectors would give
    "align_sigma": (0.07, 0.21),
    # `||do|| / ||o||` is a loss-normalization constant (mean-reduced MSE over 2.1e6
    # elements), not an operator property, so the runtime always draws `do` at unit RMS and
    # covers the real magnitude with edge fixture e9 instead.  Kept for stream fidelity.
    "do_over_o": (7e-9, 3e-5),
    "head_spread": (2.0, 7.0),
}

_SEED_MASK = 0x7FFFFFFF
# e6: with a per-group maximum of exactly 6 the block scale is 1, so the quantized value is
# the input itself and every one of these magnitudes sits exactly on an e2m1 rounding tie.
# Round-to-nearest-even and round-half-away then differ on every such element, which is the
# only way to see the rounding mode at gate tolerances.
TIE_MAGNITUDES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
TIE_GROUP_MAXIMUM = 6.0


def _u(torch: Any, g: Any, lo: float, hi: float, shape: Any, device: Any) -> Any:
    """Uniform draw on [lo, hi)."""
    return torch.rand(shape, generator=g, device=device) * (hi - lo) + lo


def _logu(torch: Any, g: Any, lo: float, hi: float, shape: Any, device: Any) -> Any:
    """Log-uniform draw on [lo, hi)."""
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    return (torch.rand(shape, generator=g, device=device) * (log_hi - log_lo) + log_lo).exp()


def _randint(torch: Any, g: Any, lo: int, hi: int, device: Any) -> int:
    """Inclusive integer draw."""
    return int(torch.randint(lo, hi + 1, (1,), generator=g, device=device).item())


def draw_params(
    torch: Any,
    seed: int,
    device: Any = "cuda",
    h: int = 12,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Draw the per-case regime parameters.  All draws in a fixed order from one generator."""
    g = torch.Generator(device=device).manual_seed(int(seed) & _SEED_MASK)
    r = RANGES
    p: dict[str, Any] = {"seed": int(seed), "h": int(h)}

    # Burn one draw so seed 0 is not degenerate.
    torch.rand(1, generator=g, device=device).item()

    perm = torch.randperm(h, generator=g, device=device).tolist()
    n_uniform = _randint(torch, g, *r["n_uniform_heads"], device)
    roles = ["ordinary"] * h
    cur = 0
    want_sink = h >= 4
    want_identity = h >= 4
    if want_sink:
        roles[perm[cur]] = "sink"
        cur += 1
    if want_identity:
        roles[perm[cur]] = "identity"
        cur += 1
    n_uniform = min(n_uniform, max(0, h - cur))
    for i in range(n_uniform):
        roles[perm[cur + i]] = "uniform"
    p["roles"] = roles
    p["n_uniform_heads"] = n_uniform

    for name in (
        "chan_sigma_q",
        "chan_sigma_k",
        "chan_sigma_v",
        "tok_sigma",
        "q_bias_frac",
        "k_bias_frac",
        "v_bias_frac",
        "q_bias_mag",
        "k_bias_mag",
        "v_bias_mag",
        "q_rms",
        "k_rms",
        "v_rms",
    ):
        p[name] = float(_u(torch, g, *r[name], (1,), device).item())
    p["n_hot_pairs"] = _randint(torch, g, *r["n_hot_pairs"], device)
    p["hot_amp"] = float(_u(torch, g, *r["hot_amp"], (1,), device).item())
    p["sink_hot_amp"] = float(_u(torch, g, *r["sink_hot_amp"], (1,), device).item())
    p["v_n_hot"] = _randint(torch, g, *r["v_n_hot"], device)
    p["v_hot_amp"] = float(_u(torch, g, *r["v_hot_amp"], (1,), device).item())
    p["identity_beta"] = float(_u(torch, g, *r["identity_beta"], (1,), device).item())
    p["hot_q_over_k"] = float(_u(torch, g, *r["hot_q_over_k"], (1,), device).item())
    sink_mode = float(torch.rand(1, generator=g, device=device).item())
    p["sink_onehot"] = sink_mode < r["sink_onehot_prob"]
    sink_range = r["maxp_sink_onehot"] if p["sink_onehot"] else r["maxp_sink_soft"]
    p["maxp_sink"] = float(_logu(torch, g, *sink_range, (1,), device).item())

    ordinary = _logu(torch, g, *r["maxp_ordinary"], (h,), device)
    uniform = _logu(torch, g, *r["maxp_uniform"], (h,), device)
    identity = _logu(torch, g, *r["maxp_identity"], (h,), device)
    targets = []
    for i, role in enumerate(roles):
        if role == "sink":
            targets.append(p["maxp_sink"])
        elif role == "uniform":
            targets.append(float(uniform[i]))
        elif role == "identity":
            targets.append(float(identity[i]))
        else:
            targets.append(float(ordinary[i]))
    p["maxp_targets"] = targets

    _draw_stream_padding(torch, g, h, device)

    if overrides:
        ov = dict(overrides)
        if ov.pop("no_sink_head", False):
            p["roles"] = ["ordinary" if x == "sink" else x for x in p["roles"]]
            p["maxp_targets"] = [
                float(ordinary[i]) if roles[i] == "sink" else t
                for i, t in enumerate(p["maxp_targets"])
            ]
        if ov.pop("no_identity_head", False):
            p["roles"] = ["ordinary" if x == "identity" else x for x in p["roles"]]
        p.update(ov)
    return p


def _draw_stream_padding(torch: Any, g: Any, h: int, device: Any) -> None:
    """Consume the lineage generator's tangent parameter draws and discard them.

    The calibrated lineage drew these eight parameters from this generator at this point.
    This task has no forward-mode tangents, but the draws stay so that the parameter stream
    -- and therefore every `q, k, v` helper A and helper B measured against -- is
    bit-identical to the lineage.  The values themselves are never read.
    """
    r = RANGES
    _logu(torch, g, *r["r_q"], (h,), device)
    _u(torch, g, *r["r_k_over_r_q"], (h,), device)
    _u(torch, g, *r["r_v_over_r_q"], (h,), device)
    _u(torch, g, *r["tangent_chan_mix"], (1,), device)
    _u(torch, g, *r["tangent_chan_jitter"], (1,), device)
    _u(torch, g, *r["tangent_tok_sigma"], (1,), device)
    _u(torch, g, *r["p_massive"], (1,), device)
    _logu(torch, g, *r["massive_scale"], (1,), device)
    torch.randn((2, h), generator=g, device=device)
    torch.randn((h,), generator=g, device=device)


def _channel_profile(torch: Any, g: Any, h: int, d: int, sigma: float, device: Any) -> Any:
    """[h, d] per-channel standard deviations, unit RMS per head."""
    cs = (torch.randn((h, d), generator=g, device=device) * sigma).exp()
    return cs / cs.pow(2).mean(dim=1, keepdim=True).sqrt()


def _bias_vector(torch: Any, g: Any, h: int, d: int, frac: float, mag: float, device: Any) -> Any:
    """[h, d] channel bias, constant across tokens, on round(frac * d) channels per head."""
    n = int(round(frac * d))
    bias = torch.zeros((h, d), device=device)
    if n <= 0:
        return bias
    key = torch.rand((h, d), generator=g, device=device)
    idx = key.argsort(dim=1)[:, :n]
    amp = torch.rand((h, n), generator=g, device=device) * mag
    sign = torch.where(torch.rand((h, n), generator=g, device=device) < 0.5, -1.0, 1.0)
    bias.scatter_(1, idx, amp * sign)
    return bias


def _apply_token_scale(torch: Any, g: Any, x: Any, sigma: float, b: int, s: int, device: Any) -> Any:
    """Multiply every token by a shared log-normal scale (in place)."""
    tok = (torch.randn((b, s, 1, 1), generator=g, device=device) * sigma).exp()
    return x.mul_(tok)


def _solve_logit_scale(torch: Any, q: Any, k: Any, targets: Sequence[float], d: int,
                       steps: int, span: float, causal: bool = False) -> Any:
    """Per-head multiplier on q so the median row max-probability hits `targets`.

    Bisection on log(tau) over a fixed, deterministic probe subset of query rows.  Scaling
    q_h by tau scales every logit of head h, so the argmax structure (and therefore the sink
    and identity heads) is preserved and the max-probability is monotone in tau.  Under a
    causal mask the probe rows see only their own prefix; probes start one stride in so
    that no probe row is the trivially one-hot first token.
    """
    b, s, h, _ = q.shape
    n_probe = min(RANGES["maxp_probe_rows"], s)
    stride = max(1, s // n_probe)
    first = stride - 1 if causal else 0
    idx = torch.arange(first, first + n_probe * stride, stride, device=q.device)[:n_probe]
    idx = idx.clamp(max=s - 1)
    probe_q = q[0, idx].permute(1, 0, 2).float()
    probe_k = k[0].permute(1, 2, 0).float()
    base = torch.bmm(probe_q, probe_k) / math.sqrt(d)
    if causal:
        keys = torch.arange(s, device=q.device)
        base = base.masked_fill((keys[None, :] > idx[:, None]).unsqueeze(0), float("-inf"))
    tgt = torch.tensor(list(targets), device=q.device, dtype=torch.float32)
    lo = torch.full((h,), -span, device=q.device)
    hi = torch.full((h,), span, device=q.device)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        scaled = base * mid.exp().view(h, 1, 1)
        maxp = torch.softmax(scaled, dim=-1).max(dim=-1).values.median(dim=-1).values
        too_small = maxp < tgt
        lo = torch.where(too_small, mid, lo)
        hi = torch.where(too_small, hi, mid)
    return (0.5 * (lo + hi)).exp()


def generate_qkv(torch: Any, b: int, s: int, h: int, d: int, seed: int, device: Any = "cuda",
                 dtype: Any = None, overrides: Mapping[str, Any] | None = None,
                 h_kv: int | None = None, causal: bool = False):
    """Return contiguous (q `[b, s, h, d]`, k and v `[b, s, h_kv, d]`) plus the parameters.

    Grouped-query attention draws every head as usual and then gives each group of
    `h // h_kv` query heads the keys and values of the group's first head, before the
    identity-head mixing and the logit-scale solve, so both act on the shared keys.
    """
    dtype = torch.bfloat16 if dtype is None else dtype
    h_kv = int(h) if h_kv is None else int(h_kv)
    if h_kv <= 0 or int(h) % h_kv:
        raise ValueError("h must be a multiple of h_kv")
    group = int(h) // h_kv
    p = draw_params(torch, seed, device=device, h=h, overrides=overrides)
    g = torch.Generator(device=device).manual_seed((int(seed) ^ 0x9E3779B9) & _SEED_MASK)
    roles = p["roles"]

    cs_q = _channel_profile(torch, g, h, d, p["chan_sigma_q"], device)
    cs_k = _channel_profile(torch, g, h, d, p["chan_sigma_k"], device)
    cs_v = _channel_profile(torch, g, h, d, p["chan_sigma_v"], device)

    # Hot channel pairs, shared between q and k of one head.  They are rotating pairs of
    # constant magnitude (a massive activation that has been through RoPE), not
    # high-variance Gaussians -- a Gaussian pair would fabricate massive primal tokens that
    # the real captures do not have.
    hot_rank = torch.rand((h, d // 2), generator=g, device=device).argsort(dim=1)
    hot_pairs = []
    for i in range(h):
        if roles[i] == "sink":
            n_pairs = RANGES["sink_n_hot_pairs"]
            amp = p["sink_hot_amp"]
        else:
            n_pairs = p["n_hot_pairs"]
            amp = p["hot_amp"]
        for j in range(n_pairs):
            a = amp if j == 0 else amp * RANGES["sink_second_pair_ratio"]
            hot_pairs.append((i, int(hot_rank[i, j].item()) * 2, a * p["hot_q_over_k"], a))
    if p["v_n_hot"] > 0:
        vhot = torch.rand((h, d), generator=g, device=device).argsort(dim=1)[:, : p["v_n_hot"]]
        cs_v.scatter_(1, vhot, cs_v.gather(1, vhot) * p["v_hot_amp"])

    q = torch.randn((b, s, h, d), generator=g, device=device).mul_(cs_q.view(1, 1, h, d))
    k = torch.randn((b, s, h, d), generator=g, device=device).mul_(cs_k.view(1, 1, h, d))
    v = torch.randn((b, s, h, d), generator=g, device=device).mul_(cs_v.view(1, 1, h, d))

    two_pi = 2.0 * math.pi
    for (i, c, aq, ak) in hot_pairs:
        phi_q = torch.rand((b, s), generator=g, device=device) * two_pi
        phi_k = torch.rand((b, s), generator=g, device=device) * two_pi
        q[:, :, i, c] = aq * phi_q.cos()
        q[:, :, i, c + 1] = aq * phi_q.sin()
        k[:, :, i, c] = ak * phi_k.cos()
        k[:, :, i, c + 1] = ak * phi_k.sin()
    p["hot_pairs"] = [(i, c, float(aq), float(ak)) for (i, c, aq, ak) in hot_pairs]

    _apply_token_scale(torch, g, q, p["tok_sigma"], b, s, device)
    _apply_token_scale(torch, g, k, p["tok_sigma"], b, s, device)
    _apply_token_scale(torch, g, v, p["tok_sigma"], b, s, device)

    q.add_(_bias_vector(torch, g, h, d, p["q_bias_frac"], p["q_bias_mag"], device).view(1, 1, h, d))
    k.add_(_bias_vector(torch, g, h, d, p["k_bias_frac"], p["k_bias_mag"], device).view(1, 1, h, d))
    v.add_(_bias_vector(torch, g, h, d, p["v_bias_frac"], p["v_bias_mag"], device).view(1, 1, h, d))

    if group > 1:
        leaders = torch.arange(h, device=device) // group * group
        k = k[:, :, leaders].contiguous()
        v = v[:, :, leaders].contiguous()

    beta = p["identity_beta"]
    for i, role in enumerate(roles):
        if role == "identity":
            q[:, :, i, :] = beta * k[:, :, i, :] + (1.0 - beta) * q[:, :, i, :]

    q.mul_(p["q_rms"] / q.pow(2).mean().sqrt())
    k.mul_(p["k_rms"] / k.pow(2).mean().sqrt())
    v.mul_(p["v_rms"] / v.pow(2).mean().sqrt())

    tau = _solve_logit_scale(torch, q, k, p["maxp_targets"], d,
                             RANGES["maxp_bisect_steps"], RANGES["maxp_log_tau_span"],
                             causal=causal)
    q.mul_(tau.view(1, 1, h, 1))
    p["tau"] = tau.tolist()
    p["h_kv"] = h_kv
    p["causal"] = bool(causal)
    if group > 1:
        k = k[:, :, ::group]
        v = v[:, :, ::group]

    out = tuple(t.to(dtype).contiguous() for t in (q, k, v))
    return out + (p,)


def draw_do_params(torch: Any, seed: int, h: int = 12, device: Any = "cpu",
                   overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Draw one parameter set for `generate_do`.  Deterministic in `seed`."""
    g = torch.Generator(device=device)
    g.manual_seed((int(seed) ^ 0x00D0D0) & _SEED_MASK)
    r = DO_RANGES
    p = {
        "chan_sigma": float(_u(torch, g, *r["chan_sigma"], (), device).item()),
        "n_hot_heads": int(
            torch.randint(
                r["n_hot_heads"][0], r["n_hot_heads"][1] + 1, (1,), generator=g, device=device
            ).item()
        ),
        "hot_head_key": [float(x) for x in torch.rand((h,), generator=g, device=device)],
        "hot_amp": float(_u(torch, g, *r["hot_amp"], (), device).item()),
        "tok_sigma": float(_u(torch, g, *r["tok_sigma"], (), device).item()),
        "p_massive": float(_u(torch, g, *r["p_massive"], (), device).item()),
        "massive_scale": float(_logu(torch, g, *r["massive_scale"], (), device).item()),
        "align_sigma": float(_u(torch, g, *r["align_sigma"], (), device).item()),
        "do_over_o": float(_logu(torch, g, *r["do_over_o"], (), device).item()),
        "head_spread": float(_logu(torch, g, *r["head_spread"], (), device).item()),
        "seed": int(seed),
    }
    if overrides:
        p.update(overrides)
    return p


def generate_do(torch: Any, o: Any, params: Mapping[str, Any], seed: int,
                do_rms: float | None = None, dtype: Any = None) -> Any:
    """Generate `do` with the statistics of a real training-step attention-output gradient.

    `o` is `[b, s, h, d]`.  `do_rms` of `None` returns unit RMS -- the operator is linear in
    `do`, so its absolute scale carries no information and the runtime fixes it; edge
    fixture e9 covers the real 1e-6 magnitude explicitly.
    """
    device = o.device
    out_dtype = dtype or o.dtype
    b, s, h, d = o.shape
    g = torch.Generator(device=device)
    g.manual_seed((int(seed) ^ 0x0D0F00) & _SEED_MASK)
    of = o.float()

    # 1. channel profile: near-flat lognormal floor plus a few sparse hot channels.
    cs = (torch.randn((h, d), generator=g, device=device) * params["chan_sigma"]).exp()
    key = torch.rand((h, d), generator=g, device=device)
    order = key.argsort(dim=1)
    hot_key = torch.tensor(list(params["hot_head_key"]), device=device)
    hot_heads = hot_key.argsort()[: int(params["n_hot_heads"])].tolist()
    for head in hot_heads:
        cs[int(head), order[int(head), 0]] *= params["hot_amp"]
    cs = cs / cs.pow(2).mean(dim=1, keepdim=True).sqrt()

    x = torch.randn((b, s, h, d), generator=g, device=device) * cs

    # 2. alignment with `o`: zero mean, so the median cosine stays 0 and only the tails widen.
    ohat = of / of.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-30)
    align = torch.randn((b, s, h, 1), generator=g, device=device) * params["align_sigma"]
    x = x + align * x.pow(2).sum(-1, keepdim=True).sqrt() * ohat

    # 3. heavy-tailed token scale shared across heads (the loss is one scalar over all
    #    tokens), plus an independent per-head massive-token mask.
    tok = (torch.randn((b, s, 1, 1), generator=g, device=device) * params["tok_sigma"]).exp()
    if params["p_massive"] > 0:
        hit = torch.rand((b, s, h, 1), generator=g, device=device) < params["p_massive"]
        boost = 1.0 + hit.float() * (params["massive_scale"] - 1.0) * torch.rand(
            (b, s, h, 1), generator=g, device=device
        )
        tok = tok * boost
    x = x * tok

    # 4. per-head scale spread, then the global scale.
    spread = params["head_spread"]
    head_scale = torch.exp(
        (torch.rand((h,), generator=g, device=device) - 0.5) * math.log(spread)
    )
    x = x * head_scale.view(1, 1, h, 1)

    current = x.pow(2).mean().sqrt().clamp_min(1e-30)
    target = 1.0 if do_rms is None else float(do_rms)
    x = x * (target / current)
    return x.to(out_dtype).contiguous()


def reference_attention(torch: Any, q: Any, k: Any, v: Any, chunk: int = GENERATOR_CHUNK,
                        causal: bool = False) -> Any:
    """Chunked fp32 attention on `[b, s, h, d]` (k, v may have fewer heads); returns `o`.

    This is generator scaffolding, not the golden: it exists only so `generate_do` can draw
    a gradient whose per-token alignment with the attention output has the measured
    distribution.  The golden reference the gates use is `reference_fwd_bwd` below.
    """
    b, s_q, h, d = q.shape
    group = h // int(k.shape[2])
    scale = 1.0 / math.sqrt(d)
    qt = q.transpose(1, 2).float()
    kt = k.transpose(1, 2).float()
    vt = v.transpose(1, 2).float()
    if group > 1:
        kt = kt.repeat_interleave(group, dim=1)
        vt = vt.repeat_interleave(group, dim=1)
    out = torch.empty_like(qt)
    for start in range(0, s_q, chunk):
        end = min(start + chunk, s_q)
        stop = end if causal else kt.shape[2]
        scores = torch.matmul(qt[:, :, start:end], kt[:, :, :stop].transpose(-2, -1)) * scale
        if causal:
            rows = torch.arange(start, end, device=q.device)
            keys = torch.arange(stop, device=q.device)
            scores = scores.masked_fill(keys[None, :] > rows[:, None], float("-inf"))
        out[:, :, start:end] = torch.matmul(torch.softmax(scores, dim=-1), vt[:, :, :stop])
    return out.transpose(1, 2).to(q.dtype).contiguous()


def _tie_tensors(torch: Any, b: int, s_q: int, s_kv: int, h: int, d: int, seed: int,
                 device: Any, dtype: Any):
    """`q, k, v` on the e2m1 rounding ties, `[b, s, h, d]`, with a 6 in every 16-group.

    A group whose maximum is exactly 6 gets block scale 1, so the stored value is the input
    itself and every other magnitude in `TIE_MAGNITUDES` sits exactly halfway between two
    e2m1 grid points.  Round-to-nearest-even and round-half-away then disagree on every one
    of them, which is the only construction that makes the rounding mode visible at gate
    tolerances (helper A measured 79x-262x separation on the four outputs).
    """
    g = torch.Generator(device=device).manual_seed(int(seed) & _SEED_MASK)
    values = torch.tensor(list(TIE_MAGNITUDES), device=device, dtype=torch.float32)
    groups = d // MXFP_BLOCK_SIZE
    tensors = []
    for length in (s_q, s_kv, s_kv):
        index = torch.randint(0, len(TIE_MAGNITUDES), (b, length, h, d), generator=g, device=device)
        sign = torch.where(
            torch.rand((b, length, h, d), generator=g, device=device) < 0.5, -1.0, 1.0
        )
        x = values[index] * sign
        position = torch.randint(
            0, MXFP_BLOCK_SIZE, (b, length, h, groups), generator=g, device=device
        )
        x = x.reshape(b, length, h, groups, MXFP_BLOCK_SIZE)
        x.scatter_(-1, position.unsqueeze(-1), TIE_GROUP_MAXIMUM)
        tensors.append(x.reshape(b, length, h, d).to(dtype).contiguous())
    return tensors[0], tensors[1], tensors[2]


def _zero_v_groups(torch: Any, v: Any, fraction: float, seed: int) -> Any:
    """Zero a deterministic fraction of `v`'s last-axis 16-element groups, in place.

    An all-zero group has `a = 0`, so `sd = 0`, `1 / sd = inf`, `x * inf = NaN`, e2m1
    saturates the NaN to 6 and the dequantized product `6 * 0` is 0 again: zeros round-trip
    to zeros through a NaN.  A kernel that reorders that computation produces NaN output.
    """
    shape = tuple(v.shape)
    d = shape[-1]
    g = torch.Generator(device=v.device).manual_seed(int(seed) & _SEED_MASK)
    groups = v.reshape(shape[:-1] + (d // MXFP_BLOCK_SIZE, MXFP_BLOCK_SIZE))
    mask = torch.rand(shape[:-1] + (d // MXFP_BLOCK_SIZE, 1), generator=g, device=v.device)
    groups.mul_(torch.where(mask < fraction, 0.0, 1.0).to(v.dtype))
    return v


# ---------------------------------------------------------------------------------------
# Disclosed edge fixtures (DESIGN section 5.6).  Correctness only, no timing, run before
# any scored case.
# ---------------------------------------------------------------------------------------
EDGE_CASES: tuple[dict[str, Any], ...] = (
    {"case_id": "edge-e1-tiny", "b": 1, "s_q": 64, "s_kv": 64, "h": 2, "d": 128,
     "seed": 900011},
    {"case_id": "edge-e2-unaligned", "b": 1, "s_q": 999, "s_kv": 999, "h": 4, "d": 128,
     "seed": 900023},
    {"case_id": "edge-e3-cross-attention", "b": 1, "s_q": 4096, "s_kv": 512, "h": 4,
     "d": 64, "seed": 900031},
    {"case_id": "edge-e4-unaligned-cross", "b": 2, "s_q": 1000, "s_kv": 515, "h": 2,
     "d": 128, "seed": 900041},
    {"case_id": "edge-e5-zero-key-zero-v-groups", "b": 1, "s_q": 2048, "s_kv": 2048, "h": 4,
     "d": 128, "seed": 900053, "k_zero": True, "v_zero_groups": 0.25},
    {"case_id": "edge-e6-rtne-ties", "b": 1, "s_q": 1024, "s_kv": 1024, "h": 2, "d": 128,
     "seed": 900061, "tie_set": True},
    {"case_id": FIXTURE_CASE_ID, "b": 1, "s_q": 8192, "s_kv": 8192, "h": 12, "d": 128,
     "seed": 900071, "fixture": True},
    {"case_id": "edge-e8-forced-sink", "b": 1, "s_q": 8192, "s_kv": 8192, "h": 12, "d": 128,
     "seed": 900083, "overrides": {"maxp_targets": [1.0] * 12}},
    {"case_id": "edge-e9-tiny-do-scale", "b": 1, "s_q": 4096, "s_kv": 4096, "h": 4, "d": 128,
     "seed": 900097, "do_scale": 1.0e-6},
    {"case_id": "edge-e10-ltx2-minimal", "b": 1, "s_q": 192, "s_kv": 192, "h": 32, "d": 128,
     "seed": 900103},
    {"case_id": "edge-e11-head-dim-64", "b": 2, "s_q": 2048, "s_kv": 2048, "h": 3, "d": 64,
     "seed": 900121},
    {"case_id": "edge-e12-causal-gqa", "b": 2, "s_q": 1003, "s_kv": 1003, "h": 8, "h_kv": 2,
     "d": 128, "causal": True, "seed": 900131},
    {"case_id": "edge-e13-causal-sink", "b": 1, "s_q": 4096, "s_kv": 4096, "h": 8, "h_kv": 1,
     "d": 128, "causal": True, "seed": 900143, "overrides": {"maxp_targets": [1.0] * 8}},
    {"case_id": "edge-e14-head-dim-256", "b": 2, "s_q": 1536, "s_kv": 1536, "h": 4, "h_kv": 2,
     "d": 256, "causal": True, "seed": 900151},
    {"case_id": "edge-e15-head-dim-256-full", "b": 1, "s_q": 777, "s_kv": 777, "h": 3,
     "d": 256, "seed": 900163, "overrides": {"maxp_targets": [0.3, 0.6, 0.9]}},
)
EDGE_CASE_IDS = tuple(str(case["case_id"]) for case in EDGE_CASES)
EXPECTED_EDGE_CASE_COUNT = len(EDGE_CASES)


def finite_positive(value: Any, label: str) -> float:
    """Return `value` as a float, rejecting non-numeric, non-finite and non-positive input."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise RuntimeError(f"{label} must be finite and positive")
    return number


def geometric_mean(values: Iterable[float]) -> float:
    """Geometric mean of strictly positive values."""
    numbers = [finite_positive(v, "geometric-mean input") for v in values]
    if not numbers:
        raise ValueError("geometric mean requires values")
    return math.exp(math.fsum(math.log(v) for v in numbers) / len(numbers))


def attention_pairs(s_q: int, s_kv: int, causal: bool = False) -> int:
    """Query-key pairs the operator scores: `s_q * s_kv`, or the causal triangle."""
    if causal:
        if s_q != s_kv:
            raise ValueError("causal attention requires s_q == s_kv")
        return s_q * (s_q + 1) // 2
    return s_q * s_kv


def forward_flops(b: int, h: int, s_q: int, s_kv: int, d: int, causal: bool = False) -> int:
    """`8 * base`: two QK-type passes and two PV-type products.

    The statistics pre-pass computes `S` once to get the row maximum and the row sum, the
    main pass computes it again (a candidate may instead stage `S` to memory, which trades
    the same FLOPs for bandwidth), and the epilogue produces both `o = Pq Vf` and the
    high-precision `O' = P~ Vf`.  Every one of the four is `2 * b * h * pairs * d`, where
    `pairs` is `s_q * s_kv` or, under a causal mask, the `s * (s + 1) / 2` unmasked pairs.
    """
    _check_geometry(b, h, s_q, s_kv, d)
    return 8 * b * h * attention_pairs(s_q, s_kv, causal) * d


def backward_flops(b: int, h: int, s_q: int, s_kv: int, d: int, causal: bool = False) -> int:
    """`10 * base`: the FlashAttention-2 backward's five matrix products.

    `QK` (recomputed), `dV = Pq^T dO`, `dP = dO Vf^T`, `dK = dS^T Qf`, `dQ = dS Kf`.  The
    seed's backward executes seven because it recomputes `QK` and `dP` in both loops; the
    denominator counts the algorithmic minimum, so collapsing that duplication is worth
    real score.
    """
    _check_geometry(b, h, s_q, s_kv, d)
    return 10 * b * h * attention_pairs(s_q, s_kv, causal) * d


def effective_flops(b: int, h: int, s_q: int, s_kv: int, d: int, causal: bool = False) -> int:
    """`26 * base` -- the production weighting of two forwards to one backward.

    FastVideo's QAD recipes run full non-reentrant activation checkpointing, so every
    training step executes the attention forward twice and the backward once
    (`component_loader` / block-level checkpointing, helper B section 4).  The runtime
    measures one forward and one backward per trial and weights the forward twice in both
    the numerator and the denominator, which keeps the score a real throughput while
    scoring the pass the recipe actually pays for twice.
    """
    return 2 * forward_flops(b, h, s_q, s_kv, d, causal) + backward_flops(b, h, s_q, s_kv, d, causal)


def _check_geometry(b: int, h: int, s_q: int, s_kv: int, d: int) -> None:
    """Reject any non-positive or non-integer geometry component."""
    for name, value in (("b", b), ("h", h), ("s_q", s_q), ("s_kv", s_kv), ("d", d)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


def trial_seconds(t_forward: float, t_backward: float) -> float:
    """`2 * t_fwd + t_bwd`, the production-weighted time the score divides into."""
    forward = finite_positive(t_forward, "forward seconds")
    backward = finite_positive(t_backward, "backward seconds")
    return 2.0 * forward + backward


def effective_tflops(b: int, h: int, s_q: int, s_kv: int, d: int, seconds: float,
                     causal: bool = False) -> float:
    """Effective TFLOPS from the `26 * base` denominator and one weighted trial time."""
    flops = effective_flops(b, h, s_q, s_kv, d, causal)
    return finite_positive(flops / finite_positive(seconds, "seconds") / 1e12, "TFLOPS")


def saved_bytes_limit(b: int, h: int, s_q: int, s_kv: int, d: int, h_kv: int | None = None) -> int:
    """Upstream's own saved footprint: `Qf, Kf, Vf, O'` in bf16 plus the fp32 `M`."""
    _check_geometry(b, h, s_q, s_kv, d)
    h_kv = h if h_kv is None else h_kv
    return 2 * b * d * (2 * h * s_q + 2 * h_kv * s_kv) + 4 * b * h * s_q


def case_input_bytes(b: int, s_q: int, s_kv: int, h: int, d: int, h_kv: int | None = None) -> int:
    """Bytes held by the four bf16 input tensors (`q`, `do` at `s_q`; `k`, `v` at `s_kv`)."""
    h_kv = h if h_kv is None else h_kv
    return 2 * b * d * (2 * h * s_q + 2 * h_kv * s_kv)


def case_output_bytes(b: int, s_q: int, s_kv: int, h: int, d: int, h_kv: int | None = None) -> int:
    """Bytes held by the four bf16 outputs (`o`, `dq` at `s_q`; `dk`, `dv` at `s_kv`)."""
    return case_input_bytes(b, s_q, s_kv, h, d, h_kv)


def case_resident_bytes(b: int, s_q: int, s_kv: int, h: int, d: int, h_kv: int | None = None) -> int:
    """Peak resident bytes for one scored case under the one-trial retention window.

    Three live invocation sets (the decoy, the previous trial and the trial being prepared),
    each carrying its own inputs, outputs and saved bundle, plus the golden reference's four
    fp32 outputs, its fp32 row statistics and its two largest fp32 score tiles.
    """
    per_set = case_input_bytes(b, s_q, s_kv, h, d, h_kv) + case_output_bytes(b, s_q, s_kv, h, d, h_kv)
    per_set += saved_bytes_limit(b, h, s_q, s_kv, d, h_kv)
    reference = 8 * case_input_bytes(b, s_q, s_kv, h, d, h_kv) + 3 * 4 * b * h * s_q
    tile = 6 * 4 * min(s_q, REFERENCE_CHUNK) * s_kv
    return LIVE_INVOCATIONS * per_set + reference + tile


LIVE_INVOCATIONS = 3


CASE_KEYS = frozenset(
    {"case_id", "family", "tier", "b", "h", "h_kv", "d", "s_q", "s_kv", "causal", "seed",
     "gated", "layer"}
)


def _positive_int(value: Any, label: str) -> int:
    """Reject anything but a strictly positive integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"invalid {label}")
    return value


def _on_lattice(family: str, s: int) -> bool:
    """True when `s` is `prefix + lattice * t` for one of the family's video lattices."""
    return any(
        s > prefix and (s - prefix) % lattice == 0 for prefix, lattice in FAMILY_LATTICES[family]
    )


def validate_case(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a scored case against the suite rules and derive its geometry."""
    if set(raw) != CASE_KEYS:
        raise ValueError("case fields differ from schema")
    case = dict(raw)
    family = str(case["family"])
    tier = str(case["tier"])
    if family not in FAMILIES or tier not in TIERS or (family in REAL_FAMILIES) != (tier == "real"):
        raise ValueError("family/tier mismatch")
    b = _positive_int(case["b"], "b")
    h = _positive_int(case["h"], "h")
    h_kv = _positive_int(case["h_kv"], "h_kv")
    d = _positive_int(case["d"], "d")
    s_q = _positive_int(case["s_q"], "s_q")
    s_kv = _positive_int(case["s_kv"], "s_kv")
    causal = case["causal"]
    if not isinstance(causal, bool):
        raise ValueError("invalid causal flag")
    if d not in FAMILY_HEAD_DIMS[family]:
        raise ValueError("head dimension outside family")
    if h % h_kv:
        raise ValueError("h must be a multiple of h_kv")
    if (h, h_kv, d) in PUBLIC_GEOMETRIES or s_q in PUBLIC_LENGTHS:
        raise ValueError("case overlaps a public identity")
    if FAMILY_DOMAIN[family] == "video":
        if causal or h_kv != h:
            raise ValueError("video families are non-causal multi-head attention")
        if not _on_lattice(family, s_q):
            raise ValueError("s outside family lattice")
        if s_kv != s_q and not (family == "kandinsky5" and s_kv == KANDINSKY5_TEXT_TOKENS):
            raise ValueError("cross attention outside family")
        if b != 1:
            raise ValueError("video cases run one clip per rank")
    else:
        if not causal or s_kv != s_q:
            raise ValueError("LLM families are causal self attention")
        if s_q % LLM_LENGTH_QUANTUM:
            raise ValueError("LLM length is not padded to the length quantum")
    seed = case["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("invalid seed")
    gated = case["gated"]
    if not isinstance(gated, bool) or (tier != "sweep" and not gated):
        raise ValueError("invalid gated flag")
    layer = case["layer"]
    if tier == "real":
        if not isinstance(layer, str) or not REAL_LAYER_PATTERN.fullmatch(layer):
            raise ValueError("invalid real layer")
        if b != 1:
            raise ValueError("real layers are single captures")
    elif layer is not None:
        raise ValueError("synthetic case names a real layer")
    input_bytes = case_input_bytes(b, s_q, s_kv, h, d, h_kv)
    if input_bytes > MAX_CASE_INPUT_BYTES:
        raise ValueError("case inputs exceed the per-case envelope")
    case.update(
        domain=FAMILY_DOMAIN[family],
        softmax_scale=float(d) ** -0.5,
        scored=True,
        input_bytes=input_bytes,
        effective_flops=effective_flops(b, h, s_q, s_kv, d, causal),
        real_layer=layer,
    )
    return case


def validate_edge_case(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a disclosed edge fixture and derive its geometry."""
    case = dict(raw)
    b = int(case["b"])
    s_q = int(case["s_q"])
    s_kv = int(case["s_kv"])
    h = int(case["h"])
    h_kv = int(case.get("h_kv", h))
    d = int(case["d"])
    causal = bool(case.get("causal", False))
    if d not in HEAD_DIMS:
        raise ValueError("invalid edge head dimension")
    if min(b, s_q, s_kv, h, h_kv) <= 0 or h % h_kv or (causal and s_q != s_kv):
        raise ValueError("invalid edge geometry")
    case.update(
        family="edge",
        tier="edge",
        scored=False,
        h_kv=h_kv,
        causal=causal,
        softmax_scale=float(d) ** -0.5,
        effective_flops=effective_flops(b, h, s_q, s_kv, d, causal),
        input_bytes=case_input_bytes(b, s_q, s_kv, h, d, h_kv),
    )
    return case


# Capability envelope: what the task contract actually needs, not the GPU model
# it was qualified on.  The GPU name and UUID are recorded as provenance only;
# they never gate admission.  Keep these two constants identical in the Judge
# verifier (tests assert the pair).
MIN_COMPUTE_CAPABILITY = (8, 9)
MIN_DEVICE_MEMORY_BYTES = 40 * 1024**3


def hardware_inventory(torch: Any) -> dict[str, Any]:
    """Assert exactly one GPU satisfying the capability envelope is visible and describe it."""
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA GPU required")
    capability = tuple(int(v) for v in torch.cuda.get_device_capability(0))
    name = str(torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    total_memory_bytes = int(props.total_memory)
    if capability < MIN_COMPUTE_CAPABILITY:
        raise RuntimeError(f"GPU compute capability {capability} is below the task minimum {MIN_COMPUTE_CAPABILITY}: {name}")
    if total_memory_bytes < MIN_DEVICE_MEMORY_BYTES:
        raise RuntimeError(f"GPU memory {total_memory_bytes} bytes is below the task minimum {MIN_DEVICE_MEMORY_BYTES} bytes: {name}")
    return {
        "name": name,
        "count": 1,
        "capability": list(capability),
        "uuid": str(getattr(props, "uuid", "")),
        "total_memory_bytes": total_memory_bytes,
        "min_compute_capability": list(MIN_COMPUTE_CAPABILITY),
        "min_device_memory_bytes": MIN_DEVICE_MEMORY_BYTES,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
    }


def _sha256(path: Path) -> str:
    """SHA-256 of a file, read in 1 MiB blocks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidate(path: Path) -> tuple[ModuleType, Callable[..., Any], Callable[..., Any], str]:
    """Import the admitted submission and return its module, both entrypoints and digest."""
    candidate = path.resolve()
    digest = _sha256(candidate)
    if not candidate.is_file() or candidate.is_symlink():
        raise RuntimeError("candidate is not regular")
    spec = importlib.util.spec_from_file_location(f"attn_qat_candidate_{digest[:24]}", candidate)
    if spec is None or spec.loader is None:
        raise RuntimeError("candidate cannot load")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    forward = getattr(module, FORWARD_ENTRYPOINT, None)
    backward = getattr(module, BACKWARD_ENTRYPOINT, None)
    if not callable(forward) or not callable(backward):
        raise RuntimeError(f"candidate must define {FORWARD_ENTRYPOINT} and {BACKWARD_ENTRYPOINT}")
    return module, forward, backward, digest


@dataclass(frozen=True)
class TrustedGateOps:
    """Torch callables captured before the candidate module is imported.

    Every reference computation and every gate arithmetic step goes through these bound
    objects, so a candidate that monkey-patches `torch` after import cannot reach the
    functions that decide whether it passed.  The set is larger than the lineage's because
    the NVFP4 fake quantization is an integer bit-twiddle: the e2m1 encoder needs the
    bitcast, the shifts, the masks and the FP8 e4m3 cast as well as the usual arithmetic.
    """

    isfinite: Callable[..., Any]
    all: Callable[..., Any]
    item: Callable[..., Any]
    to_float: Callable[..., Any]
    to_bfloat16: Callable[..., Any]
    to_int32: Callable[..., Any]
    to_long: Callable[..., Any]
    to_e4m3: Callable[..., Any]
    bitcast_int32: Callable[..., Any]
    byte_view: Callable[..., Any]
    clone: Callable[..., Any]
    equal: Callable[..., Any]
    is_tensor: Callable[..., Any]
    untyped_storage: Callable[..., Any]
    fill: Callable[..., Any]
    abs: Callable[..., Any]
    add: Callable[..., Any]
    sub: Callable[..., Any]
    mul: Callable[..., Any]
    div: Callable[..., Any]
    neg: Callable[..., Any]
    reciprocal: Callable[..., Any]
    sum: Callable[..., Any]
    square: Callable[..., Any]
    sqrt: Callable[..., Any]
    exp2: Callable[..., Any]
    log2: Callable[..., Any]
    norm: Callable[..., Any]
    amax: Callable[..., Any]
    clamp: Callable[..., Any]
    where: Callable[..., Any]
    signbit: Callable[..., Any]
    bitwise_and: Callable[..., Any]
    bitwise_or: Callable[..., Any]
    left_shift: Callable[..., Any]
    right_shift: Callable[..., Any]
    less: Callable[..., Any]
    greater: Callable[..., Any]
    not_equal: Callable[..., Any]
    logical_or: Callable[..., Any]
    logical_and: Callable[..., Any]
    take: Callable[..., Any]
    pad: Callable[..., Any]
    reshape: Callable[..., Any]
    permute: Callable[..., Any]
    transpose: Callable[..., Any]
    matmul: Callable[..., Any]
    contiguous: Callable[..., Any]
    copy: Callable[..., Any]
    empty: Callable[..., Any]
    zeros: Callable[..., Any]
    zeros_like: Callable[..., Any]
    full_like: Callable[..., Any]
    ones_like: Callable[..., Any]
    tensor: Callable[..., Any]
    arange: Callable[..., Any]
    memory_allocated: Callable[..., Any]
    max_memory_allocated: Callable[..., Any]
    memory_reserved: Callable[..., Any]
    max_memory_reserved: Callable[..., Any]
    reset_peak_memory: Callable[..., Any]
    float_dtype: Any
    bfloat16_dtype: Any
    int32_dtype: Any

    @classmethod
    def capture(cls, torch: Any) -> "TrustedGateOps":
        """Bind every op the gates, the golden reference and the generators touch."""
        tensor_to = torch.Tensor.to
        tensor_view = torch.Tensor.view
        return cls(
            torch.isfinite,
            torch.all,
            torch.Tensor.item,
            lambda x, cast=tensor_to, dtype=torch.float32: cast(x, dtype),
            lambda x, cast=tensor_to, dtype=torch.bfloat16: cast(x, dtype),
            lambda x, cast=tensor_to, dtype=torch.int32: cast(x, dtype),
            lambda x, cast=tensor_to, dtype=torch.int64: cast(x, dtype),
            lambda x, cast=tensor_to, dtype=torch.float8_e4m3fn: cast(x, dtype),
            lambda x, view=tensor_view, dtype=torch.int32: view(x, dtype),
            lambda x, view=tensor_view, dtype=torch.uint8: view(x, dtype),
            torch.clone,
            torch.equal,
            torch.is_tensor,
            torch.Tensor.untyped_storage,
            torch.Tensor.fill_,
            torch.abs,
            torch.add,
            torch.sub,
            torch.mul,
            torch.div,
            torch.neg,
            torch.reciprocal,
            torch.sum,
            torch.square,
            torch.sqrt,
            torch.exp2,
            torch.log2,
            torch.linalg.vector_norm,
            torch.amax,
            torch.clamp,
            torch.where,
            torch.signbit,
            torch.bitwise_and,
            torch.bitwise_or,
            torch.bitwise_left_shift,
            torch.bitwise_right_shift,
            torch.less,
            torch.greater,
            torch.not_equal,
            torch.logical_or,
            torch.logical_and,
            torch.take,
            torch.nn.functional.pad,
            torch.reshape,
            torch.permute,
            torch.transpose,
            torch.matmul,
            torch.Tensor.contiguous,
            torch.Tensor.copy_,
            torch.empty,
            torch.zeros,
            torch.zeros_like,
            torch.full_like,
            torch.ones_like,
            torch.tensor,
            torch.arange,
            torch.cuda.memory_allocated,
            torch.cuda.max_memory_allocated,
            torch.cuda.memory_reserved,
            torch.cuda.max_memory_reserved,
            torch.cuda.reset_peak_memory_stats,
            torch.float32,
            torch.bfloat16,
            torch.int32,
        )


@dataclass(frozen=True)
class TrustedClock:
    """Timing primitives captured before the candidate module is imported."""

    perf_counter: Callable[[], float]
    synchronize: Callable[..., Any]
    current_stream: Callable[..., Any]
    event: Callable[..., Any]
    gate: TrustedGateOps

    @classmethod
    def capture(cls, torch: Any) -> "TrustedClock":
        """Bind the host clock, the CUDA synchronisation point and the event factory."""
        return cls(
            time.perf_counter,
            torch.cuda.synchronize,
            torch.cuda.current_stream,
            torch.cuda.Event,
            TrustedGateOps.capture(torch),
        )


# ---------------------------------------------------------------------------------------
# The golden reference.  A chunked fp32 evaluation of the DESIGN section 5.1 contract with
# TF32 off, ported from `lab/seed/reference.py::reference_fwd_bwd` (helper A verified the
# `fq_torch` half bit-exact against the Triton fake quantization on 1,532,928 values
# including ties, all-zero groups, denormals and e4m3 overflow).
# ---------------------------------------------------------------------------------------


def _e2m1_rtne_code(g: TrustedGateOps, u: Any, source_is_fp32: bool) -> Any:
    """The 3-bit e2m1 magnitude code of pre-scaled fp32 values, round-to-nearest-even.

    A bitwise port of FastVideo's `nvfp4_utils._compute_quant_and_scale`.  `source_is_fp32`
    reproduces the extra sticky-bit contribution the Triton kernel computes only when its
    input was fp32: it evaluates `min(adjusted_exponent, 31)` on unsigned arithmetic, so a
    negative adjusted exponent becomes 31 rather than clamping to 0.
    """
    bits = g.bitcast_int32(u)
    abits = g.bitwise_and(bits, 0x7FFFFFFF)
    exponents = g.bitwise_and(g.right_shift(abits, 23), 0xFF)
    mantissa = g.bitwise_and(abits, 0x7FFFFF)
    is_subnormal = g.less(exponents, 127)
    adjusted = g.sub(126, exponents)
    prepared = g.bitwise_or(0x400000, g.right_shift(mantissa, 1))
    shift = g.clamp(adjusted, 0, 31)
    shifted = g.where(
        g.greater(adjusted, 31), g.zeros_like(prepared), g.right_shift(prepared, shift)
    )
    normalised = g.where(is_subnormal, shifted, mantissa)
    exponent_field = g.sub(g.clamp(exponents, min=126), 126)
    top = g.right_shift(normalised, 21)
    least = g.bitwise_and(g.right_shift(top, 1), 1)
    guard = g.bitwise_and(top, 1)
    sticky = g.not_equal(g.bitwise_and(normalised, 0x1FFFFF), 0)
    if source_is_fp32:
        wide = g.logical_or(g.less(adjusted, 0), g.greater(adjusted, 31))
        mantissa_shift = g.where(wide, g.full_like(adjusted, 31), adjusted)
        mask = g.sub(g.left_shift(g.ones_like(adjusted), mantissa_shift), 1)
        dropped_low = g.not_equal(g.bitwise_and(mantissa, 1), 0)
        dropped_high = g.not_equal(g.bitwise_and(prepared, mask), 0)
        extra = g.logical_and(is_subnormal, g.logical_or(dropped_low, dropped_high))
        sticky = g.logical_or(extra, sticky)
    increment = g.bitwise_and(guard, g.bitwise_or(g.to_int32(sticky), least))
    packed = g.bitwise_or(g.left_shift(exponent_field, 2), top)
    return g.clamp(g.right_shift(g.add(packed, increment), 1), max=7)


def fake_quantize(g: TrustedGateOps, x: Any, source_is_fp32: bool,
                  group: int = MXFP_BLOCK_SIZE) -> Any:
    """NVFP4 fake quantization of the last axis, in fp32, zero-padded to the group size.

    Per group of `group` elements: `a = max|x|`, `sd = e4m3(a / 6)` with the PTX
    `cvt.rn.satfinite.e4m3` saturation at 448 (Torch's own cast returns NaN on overflow),
    `x_hat = e2m1_rtne(x / sd)`, and the result is `x_hat * sd`.  A group of zeros gives
    `sd = 0`, `1 / sd = inf`, `x / sd = NaN`, an e2m1 saturation to 6 and `6 * 0 = 0`, so
    zeros round-trip to zeros through a NaN -- which is why an honest kernel must keep that
    exact order of operations.
    """
    values = g.to_float(x)
    length = values.shape[-1]
    padding = (-length) % group
    if padding:
        values = g.pad(values, (0, padding))
    shape = tuple(values.shape)
    grouped = g.reshape(values, shape[:-1] + (shape[-1] // group, group))
    amplitude = g.amax(g.abs(grouped), dim=-1, keepdim=True)
    scale = g.to_float(g.to_e4m3(g.clamp(g.div(amplitude, 6.0), max=E4M3_MAX)))
    scaled = g.mul(grouped, g.reciprocal(scale))
    code = _e2m1_rtne_code(g, scaled, source_is_fp32)
    table = g.tensor(list(E2M1_MAGNITUDES), device=values.device, dtype=g.float_dtype)
    magnitude = g.take(table, g.to_long(code))
    signed = g.where(g.signbit(scaled), g.neg(magnitude), magnitude)
    out = g.reshape(g.mul(signed, scale), shape)
    return out[..., :length] if padding else out


def _causal_scores(g: TrustedGateOps, scores: Any, start: int, end: int) -> Any:
    """Mask the keys after each query row to `-inf` (rows `start..end`, keys `0..end`)."""
    device = scores.device
    rows = g.reshape(g.arange(start, end, device=device), (-1, 1))
    keys = g.reshape(g.arange(0, int(scores.shape[-1]), device=device), (1, -1))
    return g.where(g.greater(keys, rows), float("-inf"), scores)


def reference_fwd_bwd(g: TrustedGateOps, q: Any, k: Any, v: Any, do: Any,
                      softmax_scale: float, causal: bool = False,
                      chunk: int = REFERENCE_CHUNK) -> dict[str, Any]:
    """The golden: DESIGN section 5.1 in chunked fp32, `[b, h, s, d]` in and out.

    `Pq` is quantized against the final row maximum in both passes, so the golden is
    deterministic and tile-order independent; `l` is the row sum of the bf16-rounded `P~`,
    which is what an honest kernel accumulates; `Delta` is taken from the bf16-rounded
    high-precision output `O'`, again matching what a kernel can hold.

    `k` and `v` may carry `h_kv` heads with `h % h_kv == 0`: query head `i` reads key/value
    head `i // (h // h_kv)`, and `dk`, `dv` sum over the query heads of each group.  With
    `causal` (which requires `s_q == s_kv`) query row `i` sees keys `0..i`; masked scores are
    `-inf`, so they contribute exactly zero to `l`, to `O'` and to every `Pq` group.
    """
    b, h, s_q, d = (int(x) for x in q.shape)
    h_kv = int(k.shape[1])
    s_kv = int(k.shape[2])
    if h % h_kv:
        raise ValueError("h must be a multiple of h_kv")
    if causal and s_q != s_kv:
        raise ValueError("causal attention requires s_q == s_kv")
    group = h // h_kv
    device = q.device
    qf = fake_quantize(g, q, False)
    kf = fake_quantize(g, k, False)
    vf = fake_quantize(g, v, False)
    dof = g.to_float(do)
    qk_scale = softmax_scale * LOG2E

    out = g.empty((b, h, s_q, d), dtype=g.float_dtype, device=device)
    out_high = g.empty((b, h, s_q, d), dtype=g.float_dtype, device=device)
    statistic = g.empty((b, h, s_q), dtype=g.float_dtype, device=device)
    dq = g.empty((b, h, s_q, d), dtype=g.float_dtype, device=device)
    dk = g.zeros((b, h_kv, s_kv, d), dtype=g.float_dtype, device=device)
    dv = g.zeros((b, h_kv, s_kv, d), dtype=g.float_dtype, device=device)

    def scores_for(batch: int, head: int, start: int, end: int) -> tuple[Any, int]:
        stop = end if causal else s_kv
        keys = g.transpose(kf[batch, head // group, :stop], 0, 1)
        scores = g.mul(g.matmul(qf[batch, head, start:end], keys), qk_scale)
        if causal:
            scores = _causal_scores(g, scores, start, end)
        return scores, stop

    for batch in range(b):
        for head in range(h):
            kv_head = head // group
            for start in range(0, s_q, chunk):
                end = min(start + chunk, s_q)
                scores, stop = scores_for(batch, head, start, end)
                row_max = g.amax(scores, dim=-1, keepdim=True)
                weights = g.exp2(g.sub(scores, row_max))
                row_sum = g.sum(g.to_float(g.to_bfloat16(weights)), dim=-1, keepdim=True)
                quantized = fake_quantize(g, weights, True)
                values = vf[batch, kv_head, :stop]
                g.copy(out[batch, head, start:end], g.div(g.matmul(quantized, values), row_sum))
                g.copy(out_high[batch, head, start:end], g.div(g.matmul(weights, values), row_sum))
                g.copy(
                    statistic[batch, head, start:end],
                    g.reshape(g.add(row_max, g.log2(row_sum)), (-1,)),
                )

    delta = g.sum(g.mul(g.to_float(g.to_bfloat16(out_high)), dof), dim=-1)
    for batch in range(b):
        for head in range(h):
            kv_head = head // group
            for start in range(0, s_q, chunk):
                end = min(start + chunk, s_q)
                scores, stop = scores_for(batch, head, start, end)
                rows = g.reshape(statistic[batch, head, start:end], (-1, 1))
                probabilities = g.exp2(g.sub(scores, rows))
                quantized = fake_quantize(g, probabilities, True)
                grad_out = dof[batch, head, start:end]
                g.copy(
                    dv[batch, kv_head, :stop],
                    g.add(dv[batch, kv_head, :stop], g.matmul(g.transpose(quantized, 0, 1), grad_out)),
                )
                values = g.transpose(vf[batch, kv_head, :stop], 0, 1)
                grad_probabilities = g.matmul(grad_out, values)
                grad_scores = g.mul(
                    probabilities,
                    g.sub(grad_probabilities, g.reshape(delta[batch, head, start:end], (-1, 1))),
                )
                g.copy(
                    dk[batch, kv_head, :stop],
                    g.add(
                        dk[batch, kv_head, :stop],
                        g.matmul(g.transpose(grad_scores, 0, 1), qf[batch, head, start:end]),
                    ),
                )
                g.copy(
                    dq[batch, head, start:end],
                    g.mul(g.matmul(grad_scores, kf[batch, kv_head, :stop]), softmax_scale),
                )
    dk = g.mul(dk, softmax_scale)
    return {"o": out, "dq": dq, "dk": dk, "dv": dv}


@dataclass
class PreparedInvocation:
    """One fully materialised forward/backward pair of calls of the candidate ABI."""

    case_id: str
    family: str
    tier: str
    scored: bool
    q: Any
    k: Any
    v: Any
    do: Any
    softmax_scale: float
    o: Any
    dq: Any
    dk: Any
    dv: Any
    b: int
    s_q: int
    s_kv: int
    h: int
    d: int
    h_kv: int
    causal: bool
    params: Mapping[str, Any]
    input_ptrs: tuple[int, ...]
    output_ptrs: tuple[int, ...]
    retained_values: tuple[Any, ...]

    @property
    def inputs(self) -> tuple[Any, ...]:
        """The four input tensors the candidate may never modify."""
        return (self.q, self.k, self.v, self.do)

    @property
    def outputs(self) -> tuple[Any, ...]:
        """The four output tensors the candidate must overwrite in full."""
        return (self.o, self.dq, self.dk, self.dv)

    @property
    def gradient_outputs(self) -> tuple[Any, ...]:
        """The three outputs the backward call owns."""
        return (self.dq, self.dk, self.dv)


def _regular_file(path: Path) -> Path:
    """Reject anything that is not a plain, non-symlink file."""
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"e7 binding path is missing or unsafe: {path.name}")
    return path


def _fixture_recipe_path() -> Path:
    """The `RECIPE.json` shipped alongside this runtime.

    Deliberately not overridable: the recipe is the anchor of the whole binding, so it
    is resolved from this module's own location (root-owned and read-only in both
    images) and never from the environment.  The second candidate is the repository
    layout, where the runtime is `runtime/attn_qat_runtime.py` and the recipe is
    `fixtures/recipe/RECIPE.json`.
    """
    here = Path(__file__).resolve()
    for candidate in (
        here.parent / FIXTURE_RECIPE_NAME,
        here.parents[1] / "fixtures" / "recipe" / FIXTURE_RECIPE_NAME,
    ):
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    raise RuntimeError("e7 recipe is not present in the runtime build context")


def _load_fixture_recipe() -> Mapping[str, Any]:
    """Parse the recipe and check the identity fields the runtime depends on."""
    import json

    path = _fixture_recipe_path()
    try:
        recipe = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise RuntimeError("e7 recipe is unreadable") from error
    if not isinstance(recipe, Mapping):
        raise RuntimeError("e7 recipe must be a JSON object")
    if recipe.get("recipe_id") != FIXTURE_RECIPE_ID:
        raise RuntimeError("e7 recipe id does not match the runtime")
    if recipe.get("fixture_name") != FIXTURE_NAME:
        raise RuntimeError("e7 recipe names a different fixture")
    if recipe.get("identity") != FIXTURE_IDENTITY:
        raise RuntimeError("e7 recipe declares a different identity")
    if tuple(recipe.get("capture", {}).get("shape", ())) != FIXTURE_SHAPE:
        raise RuntimeError("e7 recipe geometry does not match the runtime")
    if tuple(recipe.get("capture", {}).get("tensors", ())) != FIXTURE_VALIDATED_TENSORS:
        raise RuntimeError("e7 recipe tensor set does not match the runtime")
    bands = recipe.get("sanity_bands")
    if not isinstance(bands, Mapping) or not all(
        isinstance(bands.get(name), Mapping) for name in FIXTURE_VALIDATED_TENSORS
    ):
        raise RuntimeError("e7 recipe is missing a sanity band")
    return recipe


def _fixture_path() -> Path:
    """Locate the deployment's own capture and verify its recipe binding.

    The fixture bytes are per-deployment, so there is no digest to pin.  The binding is
    the manifest the capture wrote next to it: it must describe these exact bytes, and
    it must name the recipe this runtime ships.  Both files are root-owned and mode
    0444 in the images.
    """
    root = os.environ.get("ATTN_QAT_FIXTURE_DIR")
    if not root:
        raise RuntimeError("ATTN_QAT_FIXTURE_DIR is required")
    import json

    directory = Path(root)
    fixture = _regular_file(directory / FIXTURE_NAME)
    manifest_path = _regular_file(directory / FIXTURE_MANIFEST_NAME)
    recipe_path = _fixture_recipe_path()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        raise RuntimeError("e7 capture manifest is unreadable") from error
    if not isinstance(manifest, Mapping):
        raise RuntimeError("e7 capture manifest must be a JSON object")
    if manifest.get("identity") != FIXTURE_IDENTITY:
        raise RuntimeError("e7 capture manifest declares a different identity")
    declared = manifest.get("fixture")
    recipe_binding = manifest.get("recipe")
    if not isinstance(declared, Mapping) or not isinstance(recipe_binding, Mapping):
        raise RuntimeError("e7 capture manifest is malformed")
    if declared.get("name") != FIXTURE_NAME:
        raise RuntimeError("e7 capture manifest names a different fixture")
    if declared.get("sha256") != _sha256(fixture):
        raise RuntimeError("e7 fixture does not match its capture manifest")
    if recipe_binding.get("id") != FIXTURE_RECIPE_ID:
        raise RuntimeError("e7 capture manifest names a different recipe")
    if recipe_binding.get("sha256") != _sha256(recipe_path):
        raise RuntimeError("e7 capture was not produced by this runtime's recipe")
    return fixture


def _check_fixture_band(name: str, tensor: Any, band: Mapping[str, Any], torch: Any) -> None:
    """Enforce one tensor's recipe sanity band.

    The bands are the H100 reference capture widened by a factor of two on the RMS and
    four on the absolute maximum: wide enough that honest cross-GPU drift never trips
    them, narrow enough that a capture of the wrong block, the wrong sigma, the wrong
    prompt or the wrong model does.
    """
    wide = tensor.float()
    finite = torch.isfinite(wide)
    finite_fraction = float(finite.to(torch.float32).mean())
    if finite_fraction < float(band["finite_fraction_min"]):
        raise RuntimeError(f"e7 tensor {name} is not finite")
    rms = float(wide.pow(2).mean().sqrt())
    if not float(band["rms_min"]) <= rms <= float(band["rms_max"]):
        raise RuntimeError(f"e7 tensor {name} fails its recipe RMS band")
    abs_max = float(wide.abs().max())
    if not float(band["abs_max_min"]) <= abs_max <= float(band["abs_max_max"]):
        raise RuntimeError(f"e7 tensor {name} fails its recipe magnitude band")


def _load_fixture(torch: Any, device: Any) -> dict[str, Any]:
    """Load this deployment's Wan2.1 training-step capture and check it against the recipe.

    The capture is stored in the model's `[B, S, H, D]` layout; the kernel contract is
    `[B, H, S, D]`.  All five recorded tensors are validated against the recipe bands;
    only the four kernel inputs are returned.  The stored `o` is orientation material
    and is never read by the case: the golden reference computes its own.
    """
    recipe = _load_fixture_recipe()
    bands = recipe["sanity_bands"]
    blob = torch.load(_fixture_path(), map_location=device, weights_only=False)
    if not isinstance(blob, dict) or not all(
        name in blob for name in FIXTURE_VALIDATED_TENSORS
    ):
        raise RuntimeError("bad e7 payload")
    tensors = {}
    for name in FIXTURE_VALIDATED_TENSORS:
        stored = blob[name]
        if tuple(stored.shape) != FIXTURE_SHAPE:
            raise RuntimeError("bad e7 geometry")
        if stored.dtype != torch.bfloat16:
            raise RuntimeError("bad e7 dtype")
        moved = stored.to(device=device, dtype=torch.bfloat16).permute(0, 2, 1, 3)
        moved = moved.contiguous()
        _check_fixture_band(name, moved, bands[name], torch)
        if name in FIXTURE_TENSORS:
            tensors[name] = moved
    return tensors


def _real_layer_source(layer: str) -> str:
    """The capture source a real layer name belongs to (`wan13b`, `qwen3-14b`, ...)."""
    return layer.split("-l")[0].split("-b")[0]


def _real_layer_recipe_path(name: str = REAL_LAYER_RECIPE_NAME) -> Path:
    """A real-layer recipe next to this runtime, or at its repository location."""
    here = Path(__file__).resolve()
    for candidate in (
        here.parent / name,
        here.parents[1] / "fixtures" / "recipe" / "real_layers" / name,
    ):
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    raise RuntimeError("real-layer recipe is not present in the runtime build context")


def _load_real_layer(torch: Any, case: Mapping[str, Any], device: Any) -> dict[str, Any]:
    """Load one real captured layer and bind it to the shipped recipe.

    The captures are per deployment and in the kernel's `[B, H, S, D]` layout.  Their
    manifest must name this recipe and these exact bytes, and each tensor must have the
    case's shape, be finite and sit inside the recipe's sanity band.
    """
    import json

    root = os.environ.get("ATTN_QAT_FIXTURE_DIR")
    if not root:
        raise RuntimeError("ATTN_QAT_FIXTURE_DIR is required")
    directory = Path(root) / "real_layers"
    layer = str(case["real_layer"])
    source = _real_layer_source(layer)
    recipe_name = REAL_LAYER_RECIPES.get(source)
    if recipe_name is None:
        raise RuntimeError(f"real layer {layer} has no recipe")
    recipe_path = _real_layer_recipe_path(recipe_name)
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        _regular_file(directory / str(recipe["manifest_name"])).read_text(encoding="utf-8")
    )
    if manifest.get("recipe", {}).get("sha256") != _sha256(recipe_path):
        raise RuntimeError("real-layer capture was not produced by this runtime's recipe")
    entry = manifest.get("files", {}).get(layer)
    path = _regular_file(directory / f"{layer}.pt")
    if not isinstance(entry, Mapping) or entry.get("sha256") != _sha256(path):
        raise RuntimeError(f"real layer {layer} does not match its capture manifest")
    blob = torch.load(path, map_location=device, weights_only=True)
    query_shape = (1, int(case["h"]), int(case["s_q"]), int(case["d"]))
    key_shape = (1, int(case["h_kv"]), int(case["s_kv"]), int(case["d"]))
    bands = recipe["sanity_bands"]
    bands = bands.get(source, bands) if isinstance(bands.get(source), Mapping) else bands
    tensors: dict[str, Any] = {}
    for name in FIXTURE_TENSORS:
        expected = key_shape if name in ("k", "v") else query_shape
        stored = blob.get(name) if isinstance(blob, dict) else None
        if stored is None or tuple(stored.shape) != expected or stored.dtype != torch.bfloat16:
            raise RuntimeError(f"real layer {layer} {name} differs from its case")
        _check_fixture_band(name, stored, bands[name], torch)
        tensors[name] = stored.contiguous()
    tensors["params"] = {"regime": "real-training-step-capture", "seed": int(case["seed"])}
    return tensors


def _to_bhsd(x: Any) -> Any:
    """Permute a `[b, s, h, d]` draw into the kernel's `[b, h, s, d]` contract."""
    return x.permute(0, 2, 1, 3).contiguous()


def generate_inputs(torch: Any, b: int, s_q: int, s_kv: int, h: int, d: int, seed: int,
                    device: Any = "cuda", dtype: Any = None,
                    overrides: Mapping[str, Any] | None = None,
                    k_zero: bool = False, v_zero_groups: float = 0.0,
                    tie_set: bool = False, do_scale: float | None = None,
                    h_kv: int | None = None, causal: bool = False):
    """Draw `q, k, v, do` for one case, contiguous bf16 `[b, h, s, d]`, plus the parameters.

    `q, k, v` come from the calibrated primal family and `do` from the gradient family,
    which needs the attention output the gradient belongs to; that output is produced by
    `reference_attention` and discarded.  Cross attention draws at `max(s_q, s_kv)` and
    keeps the leading `s_q` query rows and `s_kv` key/value rows: the prefix of a
    token-independent draw has the same regime as the whole.

    The four flags build the edge fixtures that the parametric family cannot express:
    an all-zero `k` (uniform softmax), a fraction of all-zero 16-element groups in `v` (the
    zero-through-NaN path), the e2m1 tie set, and a `do` at the real mean-reduced-MSE
    magnitude.
    """
    dtype = torch.bfloat16 if dtype is None else dtype
    s = max(int(s_q), int(s_kv))
    if tie_set:
        q, k, v = _tie_tensors(torch, b, s_q, s_kv, h, d, seed, device, dtype)
        params: dict[str, Any] = {"seed": int(seed), "regime": "e2m1-tie-set"}
    else:
        drawn_q, drawn_k, drawn_v, params = generate_qkv(
            torch, int(b), s, int(h), int(d), int(seed),
            device=device, dtype=dtype, overrides=overrides, h_kv=h_kv, causal=causal,
        )
        params = dict(params)
        q = drawn_q[:, : int(s_q)].contiguous()
        k = drawn_k[:, : int(s_kv)].contiguous()
        v = drawn_v[:, : int(s_kv)].contiguous()
    params["generated_s"] = s
    params["cross_attention"] = int(s_q) != int(s_kv)
    if k_zero:
        k = torch.zeros_like(k)
        params["k_zero"] = True
    if v_zero_groups > 0.0:
        v = _zero_v_groups(torch, v, float(v_zero_groups), int(seed) ^ 0x1234_5678)
        params["v_zero_groups"] = float(v_zero_groups)
    attention_out = reference_attention(torch, q, k, v, causal=causal)
    do_params = draw_do_params(torch, int(seed), h=int(h), device=device)
    do = generate_do(torch, attention_out, do_params, int(seed), do_rms=do_scale, dtype=dtype)
    params["do"] = {name: do_params[name] for name in ("tok_sigma", "p_massive", "align_sigma")}
    params["do_rms"] = 1.0 if do_scale is None else float(do_scale)
    del attention_out
    return (
        _to_bhsd(q),
        _to_bhsd(k),
        _to_bhsd(v),
        _to_bhsd(do),
        params,
    )


def _generate_case_tensors(torch: Any, case: Mapping[str, Any], seed: int,
                           device: Any = None) -> dict[str, Any]:
    """Materialise the four input tensors of one case on the GPU."""
    device = torch.device("cuda") if device is None else torch.device(device)
    if case.get("real_layer"):
        return _load_real_layer(torch, case, device)
    if case.get("fixture"):
        tensors = _load_fixture(torch, device)
        tensors["params"] = {"regime": "wan13b-training-step-capture", "seed": int(seed)}
        return tensors
    q, k, v, do, params = generate_inputs(
        torch,
        int(case["b"]),
        int(case["s_q"]),
        int(case["s_kv"]),
        int(case["h"]),
        int(case["d"]),
        int(seed),
        device=device,
        dtype=torch.bfloat16,
        overrides=case.get("overrides"),
        k_zero=bool(case.get("k_zero", False)),
        v_zero_groups=float(case.get("v_zero_groups", 0.0)),
        tie_set=bool(case.get("tie_set", False)),
        do_scale=case.get("do_scale"),
        h_kv=int(case.get("h_kv", case["h"])),
        causal=bool(case.get("causal", False)),
    )
    return {"q": q, "k": k, "v": v, "do": do, "params": params}


def warm_up_generator(torch: Any, case: Mapping[str, Any]) -> None:
    """Generate and drop one throwaway case of this shape.

    The first generation at a given `(s, h, d)` pays the caching allocator's growth, and the
    generator's own reference attention is the largest allocation in the runtime.  Paying it
    on a throwaway keeps it out of the first timed trial.
    """
    if case.get("fixture"):
        return
    tensors = _generate_case_tensors(torch, case, int(case["seed"]) ^ GENERATOR_WARMUP_XOR)
    del tensors
    torch.cuda.empty_cache()


def _prepare_invocation(torch: Any, case: Mapping[str, Any], seed: int) -> PreparedInvocation:
    """Draw fresh inputs, allocate NaN-poisoned outputs and record the pointer identities."""
    tensors = _generate_case_tensors(torch, case, seed)
    q, k, v, do = tensors["q"], tensors["k"], tensors["v"], tensors["do"]
    b, h, s_q, d = (int(x) for x in q.shape)
    h_kv = int(k.shape[1])
    s_kv = int(k.shape[2])
    causal = bool(case.get("causal", False))
    if h_kv != int(case.get("h_kv", h)):
        raise RuntimeError("generated key/value heads differ from the case")
    o = torch.full_like(q, float("nan"))
    dq = torch.full_like(q, float("nan"))
    dk = torch.full_like(k, float("nan"))
    dv = torch.full_like(v, float("nan"))
    inputs = (q, k, v, do)
    outputs = (o, dq, dk, dv)
    return PreparedInvocation(
        case_id=str(case["case_id"]),
        family=str(case.get("family", "edge")),
        tier=str(case.get("tier", "edge")),
        scored=bool(case.get("scored", False)),
        q=q,
        k=k,
        v=v,
        do=do,
        softmax_scale=float(case["softmax_scale"]),
        o=o,
        dq=dq,
        dk=dk,
        dv=dv,
        b=b,
        s_q=s_q,
        s_kv=s_kv,
        h=h,
        d=d,
        h_kv=h_kv,
        causal=causal,
        params=tensors["params"],
        input_ptrs=tuple(int(x.data_ptr()) for x in inputs),
        output_ptrs=tuple(int(x.data_ptr()) for x in outputs),
        retained_values=inputs + outputs,
    )


def _invoke_forward(fn: Callable[..., Any], p: PreparedInvocation) -> Any:
    """Call the six-argument forward and return the opaque saved bundle."""
    return fn(p.q, p.k, p.v, p.softmax_scale, p.causal, p.o)


def _invoke_backward(fn: Callable[..., Any], p: PreparedInvocation, saved: Any) -> None:
    """Call the eleven-argument backward and reject a non-None return."""
    returned = fn(p.q, p.k, p.v, p.softmax_scale, p.causal, p.o, saved, p.do, p.dq, p.dk, p.dv)
    if returned is not None:
        raise RuntimeError(f"{BACKWARD_ENTRYPOINT} must return None")


def _invoke_step(forward: Callable[..., Any], backward: Callable[..., Any],
                 p: PreparedInvocation) -> Any:
    """One untimed forward/backward pair; returns the saved bundle."""
    saved = _invoke_forward(forward, p)
    _invoke_backward(backward, p, saved)
    return saved


def _poison_outputs(clock: TrustedClock, p: PreparedInvocation) -> None:
    """Refill every output with NaN so an unwritten element is detectable."""
    for tensor in p.outputs:
        clock.gate.fill(tensor, float("nan"))


def _poison_gradients(clock: TrustedClock, p: PreparedInvocation) -> None:
    """Refill only the backward's outputs with NaN, leaving `o` as the forward wrote it."""
    for tensor in p.gradient_outputs:
        clock.gate.fill(tensor, float("nan"))


def _snapshot(clock: TrustedClock, tensors: Sequence[Any]) -> tuple[Any, ...]:
    """Byte-level copies of a tensor sequence."""
    gate = clock.gate
    return tuple(gate.clone(gate.byte_view(x)) for x in tensors)


def _unchanged(clock: TrustedClock, tensors: Sequence[Any], snapshot: Sequence[Any]) -> bool:
    """True when every tensor is byte-identical to its snapshot."""
    gate = clock.gate
    if len(snapshot) != len(tensors):
        return False
    return all(
        bool(gate.equal(gate.byte_view(x), y))
        for x, y in zip(tensors, snapshot, strict=True)
    )


def saved_object_bytes(gate: TrustedGateOps, saved: Any,
                       contract: Sequence[Any]) -> tuple[int, int]:
    """Bytes of new storages reachable from `saved`, and how many tensors carry them.

    Storages already owned by the contract tensors are free -- a candidate is welcome to
    hand back a view of `q` -- and every other storage counts once.  A callable reachable
    from `saved` is rejected: the bundle is data the backward reads, not a deferred
    computation that could run the forward's real work inside the backward's timer.  Every
    tensor must live on the same device as the contract.
    """
    excluded = set()
    device = None
    for tensor in contract:
        if tensor is not None:
            excluded.add(int(gate.untyped_storage(tensor).data_ptr()))
            device = tensor.device
    seen: dict[int, int] = {}
    tensors = 0
    nodes = 0
    stack = [(saved, 0)]
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > SAVED_WALK_MAX_NODES:
            raise RuntimeError("saved object exceeds the traversal budget")
        if item is None or isinstance(item, (bool, int, float, complex, str, bytes)):
            continue
        if gate.is_tensor(item):
            tensors += 1
            if device is not None and item.device != device:
                raise RuntimeError("saved tensors must live on the contract device")
            storage = gate.untyped_storage(item)
            seen[int(storage.data_ptr())] = int(storage.nbytes())
            continue
        if callable(item):
            raise RuntimeError("saved object must not carry callables")
        if depth >= SAVED_WALK_MAX_DEPTH:
            raise RuntimeError("saved object exceeds the traversal depth")
        if isinstance(item, Mapping):
            for key, value in item.items():
                stack.append((key, depth + 1))
                stack.append((value, depth + 1))
            continue
        if isinstance(item, (list, tuple, set, frozenset)):
            for value in item:
                stack.append((value, depth + 1))
            continue
        attributes = getattr(item, "__dict__", None)
        if isinstance(attributes, dict):
            for value in attributes.values():
                stack.append((value, depth + 1))
            continue
        raise RuntimeError("saved object holds an unsupported member")
    total = sum(nbytes for pointer, nbytes in seen.items() if pointer not in excluded)
    return total, tensors


def check_saved(gate: TrustedGateOps, p: PreparedInvocation, saved: Any) -> dict[str, Any]:
    """Enforce the saved-activation byte cap and report what the bundle holds."""
    if not isinstance(saved, tuple):
        raise RuntimeError(f"{FORWARD_ENTRYPOINT} must return a tuple of tensors")
    total, tensors = saved_object_bytes(gate, saved, (p.q, p.k, p.v, p.o, p.do))
    limit = saved_bytes_limit(p.b, p.h, p.s_q, p.s_kv, p.d, p.h_kv)
    if total > limit:
        raise RuntimeError(f"saved activations use {total} bytes; the budget is {limit}")
    return {"saved_bytes": total, "saved_bytes_limit": limit, "saved_tensors": tensors}


def _tensor_metrics(g: TrustedGateOps, predicted: Any, reference: Any) -> dict[str, float]:
    """Relative-L2 statistics of `predicted` against `reference`, both fp32 `[b, h, s, d]`.

    `head_rel_l2` is the worst `(b, h)` slice.  It is reported, not gated: helper A found
    the fixed global limits sufficient to separate every planted cheat under this contract,
    and a second gated statistic would only add a way to fail an honest kernel.
    """
    delta = g.sub(predicted, reference)
    reference_norm = g.norm(g.reshape(reference, (-1,)))
    global_rel = g.div(g.norm(g.reshape(delta, (-1,))), g.clamp(reference_norm, min=1e-30))
    head_shape = (delta.shape[0], delta.shape[1], -1)
    head_delta = g.norm(g.reshape(delta, head_shape), dim=-1)
    head_reference = g.norm(g.reshape(reference, head_shape), dim=-1)
    head_rel = g.div(head_delta, g.clamp(head_reference, min=1e-30))
    return {
        "rel_l2": float(g.item(global_rel)),
        "head_rel_l2": float(g.item(g.amax(head_rel))),
        "max_abs": float(g.item(g.amax(g.abs(predicted)))),
        "max_abs_ref": float(g.item(g.amax(g.abs(reference)))),
        "ref_norm": float(g.item(reference_norm)),
    }


def gate_group(case: Mapping[str, Any]) -> str:
    """The gate group this case belongs to: its own fixture id, or the scored group."""
    case_id = str(case.get("case_id", ""))
    if case_id in FIXTURE_GATE_LIMITS:
        return case_id
    family = case.get("family")
    if family == "real_llm":
        return REAL_LLM_GROUP
    return REAL_GROUP if family == "real_video" else SCORED_GROUP


def realised_limits(case: Mapping[str, Any]) -> dict[str, float]:
    """The four relative-L2 limits this case applies."""
    group = gate_group(case)
    if group == REAL_GROUP:
        source = _real_layer_source(str(case.get("real_layer") or case.get("layer") or ""))
        limits = REAL_SOURCE_GATE_LIMITS.get(source, REAL_GATE_LIMITS)
    elif group == REAL_LLM_GROUP:
        limits = REAL_LLM_GATE_LIMITS
    else:
        limits = FIXTURE_GATE_LIMITS.get(group, SCORED_GATE_LIMITS)
    result = {f"{name}_rel_l2_limit": float(limits[name]) for name in GATE_TENSORS}
    result["gate_group"] = group
    return result


def gate_failures(candidate: Mapping[str, float],
                  limits: Mapping[str, float]) -> list[str]:
    """Name every accuracy gate this candidate fails, in a fixed order.

    Every gate is evaluated: a candidate that damages `o` and `dv` is reported as failing
    both, so the qualification evidence names each gate a cheat trips instead of only the
    first one checked.  The pass/fail verdict is unchanged -- a candidate is accepted
    exactly when this list is empty.

    When the golden output is exactly zero the relative statistic is undefined and the gate
    becomes the absolute one: `max |candidate| <= ZERO_REF_ABS`.
    """
    failed: list[str] = []
    for name in GATE_TENSORS:
        value = float(candidate[f"{name}_rel_l2"])
        if not math.isfinite(value):
            failed.append(f"nonfinite metric {name}_rel_l2")
    for name in GATE_TENSORS:
        reference_norm = float(candidate[f"{name}_ref_norm"])
        magnitude = float(candidate[f"{name}_max_abs"])
        value = float(candidate[f"{name}_rel_l2"])
        if reference_norm <= 0.0:
            if not math.isfinite(magnitude) or magnitude > ZERO_REF_ABS:
                failed.append(f"{name} zero-reference absolute gate failed")
            continue
        if math.isfinite(value) and value > float(limits[f"{name}_rel_l2_limit"]):
            failed.append(f"{name} global rel-L2 gate failed")
    return failed


def check_metric_bundle(candidate: Mapping[str, float],
                        case: Mapping[str, Any]) -> dict[str, float]:
    """Apply every accuracy gate and return the limits this case used.

    Raises with *every* failing gate named, not just the first one.
    """
    limits = realised_limits(case)
    failed = gate_failures(candidate, limits)
    if failed:
        raise RuntimeError("; ".join(failed))
    return limits


def _check_finite(clock: TrustedClock, p: PreparedInvocation) -> None:
    """Every output was written in full with finite values; runs on every timed trial."""
    g = clock.gate
    for tensor, label in zip(p.outputs, GATE_TENSORS, strict=True):
        if not bool(g.item(g.all(g.isfinite(tensor)))):
            raise RuntimeError(f"{label} contains unwritten or nonfinite values")


def _check_outputs(clock: TrustedClock, p: PreparedInvocation,
                   case: Mapping[str, Any]) -> dict[str, float]:
    """Verify writes and finiteness, compute the golden, and run every gate."""
    g = clock.gate
    _check_finite(clock, p)
    reference = reference_fwd_bwd(g, p.q, p.k, p.v, p.do, p.softmax_scale, causal=p.causal)
    candidate: dict[str, float] = {}
    for label, produced in zip(GATE_TENSORS, p.outputs, strict=True):
        for key, value in _tensor_metrics(g, g.to_float(produced), reference[label]).items():
            candidate[f"{label}_{key}"] = value
    limits = realised_limits(case)
    failed = gate_failures(candidate, limits)
    if failed:
        raise RuntimeError("; ".join(failed))
    metrics = dict(candidate)
    metrics.update(limits)
    del reference
    return metrics


def _time_call(clock: TrustedClock, call: Callable[[], Any]) -> tuple[float, Any]:
    """Time one candidate call; the reported time is `max(host wall, CUDA event)`."""
    start = clock.event(enable_timing=True)
    end = clock.event(enable_timing=True)
    clock.synchronize()
    wall = clock.perf_counter()
    start.record(clock.current_stream())
    value = call()
    end.record(clock.current_stream())
    clock.synchronize()
    host = finite_positive(clock.perf_counter() - wall, "host wall")
    event = finite_positive(float(start.elapsed_time(end)) / 1000, "event")
    return max(host, event), value


def _measure_once(clock: TrustedClock, forward: Callable[..., Any],
                  backward: Callable[..., Any], p: PreparedInvocation):
    """One timed trial: the forward, then the backward, with every protocol check between.

    The two calls are timed separately because the score weights them differently, and the
    forward's `o` is snapshotted between them: the backward receives `o` as an input and
    must leave it byte-identical, which is what stops a candidate from moving forward work
    into the backward's timer by writing `o` there.
    """
    forward_snapshot = _snapshot(clock, p.inputs)
    forward_seconds, saved = _time_call(clock, lambda: _invoke_forward(forward, p))
    if not _unchanged(clock, p.inputs, forward_snapshot):
        raise RuntimeError("candidate modified inputs")
    output_snapshot = _snapshot(clock, (p.o,))
    saved_report = check_saved(clock.gate, p, saved)
    _poison_gradients(clock, p)
    backward_seconds, _ = _time_call(clock, lambda: _invoke_backward(backward, p, saved))
    if not _unchanged(clock, p.inputs, forward_snapshot):
        raise RuntimeError("candidate modified inputs")
    if not _unchanged(clock, (p.o,), output_snapshot):
        raise RuntimeError("candidate modified the forward output during the backward")
    del saved
    return forward_seconds, backward_seconds, saved_report


def _identity(p: PreparedInvocation) -> dict[str, Any]:
    """The geometry the Judge re-derives and compares against its own case table."""
    return {
        "case_id": p.case_id,
        "family": p.family,
        "tier": p.tier,
        "scored": p.scored,
        "b": p.b,
        "s_q": p.s_q,
        "s_kv": p.s_kv,
        "h": p.h,
        "h_kv": p.h_kv,
        "d": p.d,
        "causal": p.causal,
        "softmax_scale": p.softmax_scale,
        "effective_flops": effective_flops(p.b, p.h, p.s_q, p.s_kv, p.d, p.causal),
        "forward_flops": forward_flops(p.b, p.h, p.s_q, p.s_kv, p.d, p.causal),
        "backward_flops": backward_flops(p.b, p.h, p.s_q, p.s_kv, p.d, p.causal),
    }


MERGE_MAX_METRICS = tuple(
    f"{name}_{statistic}"
    for name in GATE_TENSORS
    for statistic in ("rel_l2", "head_rel_l2", "max_abs")
)


def _merge_metrics(into: dict[str, float], new: Mapping[str, float]) -> None:
    """Keep the worst value of every reported statistic across the trials of one case."""
    for metric in MERGE_MAX_METRICS:
        if metric not in into or float(new[metric]) > into[metric]:
            into[metric] = float(new[metric])
    for name in GATE_TENSORS:
        limit = f"{name}_rel_l2_limit"
        reference = f"{name}_ref_norm"
        if limit in new:
            into[limit] = float(new[limit])
        if reference in new:
            into[reference] = float(new[reference])
    if "gate_group" in new:
        into["gate_group"] = new["gate_group"]


def _regime_summary(params: Mapping[str, Any]) -> dict[str, Any]:
    """The public description of a drawn regime.

    The generator and its ranges are disclosed, so this carries no secret; it exists so the
    public payload can explain why one case is harder than another.
    """
    keys = (
        "seed",
        "roles",
        "maxp_targets",
        "sink_onehot",
        "generated_s",
        "cross_attention",
        "k_zero",
        "v_zero_groups",
        "do",
        "do_rms",
        "regime",
    )
    return {key: params[key] for key in keys if key in params}


def _run_edge_case(torch: Any, clock: TrustedClock, forward: Callable[..., Any],
                   backward: Callable[..., Any], raw: Mapping[str, Any]) -> dict[str, Any]:
    """Run one disclosed edge fixture: correctness only, no timing."""
    case = validate_edge_case(raw)
    p = _prepare_invocation(torch, case, int(case["seed"]))
    snapshot = _snapshot(clock, p.inputs)
    clock.synchronize()
    saved = _invoke_forward(forward, p)
    clock.synchronize()
    if not _unchanged(clock, p.inputs, snapshot):
        raise RuntimeError("candidate modified inputs")
    output_snapshot = _snapshot(clock, (p.o,))
    saved_report = check_saved(clock.gate, p, saved)
    _invoke_backward(backward, p, saved)
    clock.synchronize()
    if not _unchanged(clock, p.inputs, snapshot):
        raise RuntimeError("candidate modified inputs")
    if not _unchanged(clock, (p.o,), output_snapshot):
        raise RuntimeError("candidate modified the forward output during the backward")
    del saved
    result = {**_identity(p), **saved_report, **_check_outputs(clock, p, case)}
    del p, snapshot, output_snapshot
    torch.cuda.empty_cache()
    return result


def _run_scored_case(torch: Any, clock: TrustedClock, forward: Callable[..., Any],
                     backward: Callable[..., Any], raw: Mapping[str, Any], *,
                     trials: int, warmups: int) -> dict[str, Any]:
    """Run one scored case: warm-ups, then `trials` timed trials each preceded by a decoy.

    Every trial draws fresh inputs and is checked for full, finite writes.  A gated case
    also runs the golden on exactly one trial, chosen from the case's seed, so the candidate
    cannot tell which trial is compared or whether its case is gated at all.
    """
    case = validate_case(raw)
    if trials <= 0 or warmups <= 0:
        raise ValueError("trials and warmups must be positive")
    gated = bool(case["gated"])
    resident = case_resident_bytes(
        int(case["b"]), int(case["s_q"]), int(case["s_kv"]), int(case["h"]), int(case["d"]),
        int(case["h_kv"]),
    )
    if resident > MAX_RESIDENT_BYTES:
        raise ValueError("case exceeds the resident-memory envelope")
    warm_up_generator(torch, case)

    metrics: dict[str, float] = {}
    base_seed = int(case["seed"])
    # The warm invocation is also the decoy: it is allocated once, its shapes are the
    # case's, and its seed differs from every trial and decoy seed, which is all the decoy
    # contract asks.  Reusing it removes one full contract set from the live footprint.
    decoy = _prepare_invocation(torch, case, base_seed ^ WARMUP_SEED_XOR)
    for _ in range(warmups):
        _poison_outputs(clock, decoy)
        snapshot = _snapshot(clock, decoy.inputs)
        clock.synchronize()
        saved = _invoke_step(forward, backward, decoy)
        clock.synchronize()
        if not _unchanged(clock, decoy.inputs, snapshot):
            raise RuntimeError("candidate modified inputs")
        check_saved(clock.gate, decoy, saved)
        del saved, snapshot

    reserved_inputs = set(decoy.input_ptrs)
    reserved_outputs = set(decoy.output_ptrs)
    previous: PreparedInvocation | None = None
    fresh_input_ok = True
    fresh_output_ok = True
    forward_us: list[float] = []
    backward_us: list[float] = []
    weighted_us: list[float] = []
    tflops: list[float] = []
    saved_report: dict[str, Any] = {}
    for trial in range(trials):
        p = _prepare_invocation(torch, case, base_seed ^ (TRIAL_SEED_BASE + trial))
        window_inputs = set() if previous is None else set(previous.input_ptrs)
        window_outputs = set() if previous is None else set(previous.output_ptrs)
        if set(p.input_ptrs) & (window_inputs | reserved_inputs):
            raise RuntimeError("fresh-input protocol failed")
        if set(p.output_ptrs) & (window_outputs | reserved_outputs):
            raise RuntimeError("fresh-output protocol failed")
        fresh_input_ok = fresh_input_ok and len(set(p.input_ptrs)) == len(p.input_ptrs)
        fresh_output_ok = fresh_output_ok and len(set(p.output_ptrs)) == len(p.output_ptrs)

        # Allocate, poison, run the decoy step, then measure.  The decoy step is the last
        # thing the candidate does before the timed forward, so the workspace blocks it just
        # freed are still in the caching allocator and the timed call reuses them instead of
        # paying a `cudaMalloc`.  `empty_cache()` is never called inside this loop.
        _poison_outputs(clock, p)
        _poison_outputs(clock, decoy)
        decoy_snapshot = _snapshot(clock, decoy.inputs)
        clock.synchronize()
        decoy_saved = _invoke_step(forward, backward, decoy)
        clock.synchronize()
        if not _unchanged(clock, decoy.inputs, decoy_snapshot):
            raise RuntimeError("candidate modified inputs")
        del decoy_saved, decoy_snapshot

        seconds_forward, seconds_backward, saved_report = _measure_once(clock, forward, backward, p)
        weighted = trial_seconds(seconds_forward, seconds_backward)
        forward_us.append(seconds_forward * 1e6)
        backward_us.append(seconds_backward * 1e6)
        weighted_us.append(weighted * 1e6)
        tflops.append(effective_tflops(p.b, p.h, p.s_q, p.s_kv, p.d, weighted, p.causal))
        _check_finite(clock, p)
        if gated and trial == base_seed % trials:
            _merge_metrics(metrics, _check_outputs(clock, p, case))
        # Dropping the older window entry here is what bounds the footprint at three live
        # invocations.
        previous = p
        del p

    p = previous
    result = {
        **_identity(p),
        "entrypoints": list(ENTRYPOINTS),
        "seed": base_seed,
        "gated": gated,
        "regime": _regime_summary(p.params),
        "trial_forward_us": forward_us,
        "trial_backward_us": backward_us,
        "trial_us": weighted_us,
        "trial_tflops": tflops,
        "median_forward_us": statistics.median(forward_us),
        "median_backward_us": statistics.median(backward_us),
        "median_us": statistics.median(weighted_us),
        "tflops": statistics.median(tflops),
        "fwd_tflops": forward_flops(p.b, p.h, p.s_q, p.s_kv, p.d, p.causal)
        / (statistics.median(forward_us) * 1e-6)
        / 1e12,
        "bwd_tflops": backward_flops(p.b, p.h, p.s_q, p.s_kv, p.d, p.causal)
        / (statistics.median(backward_us) * 1e-6)
        / 1e12,
        "fresh_input_ok": fresh_input_ok,
        "fresh_output_ok": fresh_output_ok,
        **saved_report,
        **metrics,
    }
    del previous, p, decoy
    torch.cuda.empty_cache()
    return result


def _run_real_case(torch: Any, clock: TrustedClock, forward: Callable[..., Any],
                   backward: Callable[..., Any], raw: Mapping[str, Any],
                   warmed: set[tuple[int, ...]]) -> dict[str, Any]:
    """Run one real captured layer: one timed step and one golden comparison.

    Each layer is different data, so it needs no repeated trial or decoy.  Compilation is
    absorbed once per shape on a synthetic draw, which never reaches the score.
    """
    case = validate_case(raw)
    shape = (int(case["h"]), int(case["h_kv"]), int(case["s_q"]), int(case["s_kv"]),
             int(case["d"]), int(bool(case["causal"])))
    seed = int(case["seed"])
    if shape not in warmed:
        warm = _prepare_invocation(torch, {**case, "real_layer": None}, seed ^ WARMUP_SEED_XOR)
        _poison_outputs(clock, warm)
        clock.synchronize()
        saved = _invoke_step(forward, backward, warm)
        clock.synchronize()
        check_saved(clock.gate, warm, saved)
        del saved, warm
        warmed.add(shape)
    p = _prepare_invocation(torch, case, seed)
    _poison_outputs(clock, p)
    clock.synchronize()
    seconds_forward, seconds_backward, saved_report = _measure_once(clock, forward, backward, p)
    weighted = trial_seconds(seconds_forward, seconds_backward)
    tflops = effective_tflops(p.b, p.h, p.s_q, p.s_kv, p.d, weighted, p.causal)
    metrics = _check_outputs(clock, p, case)
    result = {
        **_identity(p),
        "entrypoints": list(ENTRYPOINTS),
        "seed": seed,
        "gated": True,
        "layer": str(case["real_layer"]),
        "regime": _regime_summary(p.params),
        "trial_forward_us": [seconds_forward * 1e6],
        "trial_backward_us": [seconds_backward * 1e6],
        "trial_us": [weighted * 1e6],
        "trial_tflops": [tflops],
        "median_forward_us": seconds_forward * 1e6,
        "median_backward_us": seconds_backward * 1e6,
        "median_us": weighted * 1e6,
        "tflops": tflops,
        "fwd_tflops": forward_flops(p.b, p.h, p.s_q, p.s_kv, p.d, p.causal) / seconds_forward / 1e12,
        "bwd_tflops": backward_flops(p.b, p.h, p.s_q, p.s_kv, p.d, p.causal) / seconds_backward / 1e12,
        "fresh_input_ok": True,
        "fresh_output_ok": True,
        **saved_report,
        **metrics,
    }
    del p
    return result


def aggregate_case_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Geometric mean inside each tier, across a family's tiers, across a domain's families,
    then across the two domains.

    A family with both tiers weighs its recipe cases and its sweep cases equally, and the
    video and LLM domains weigh equally however many families or cases each holds.
    """
    scored = [row for row in results if bool(row.get("scored"))]
    if not scored:
        raise ValueError("no scored cases")
    values: dict[str, dict[str, list[float]]] = {}
    for row in scored:
        tiers = values.setdefault(str(row["family"]), {})
        tiers.setdefault(str(row["tier"]), []).append(
            finite_positive(row["tflops"], "case TFLOPS")
        )
    families = {
        name: geometric_mean(geometric_mean(group) for group in tiers.values())
        for name, tiers in values.items()
    }
    domains: dict[str, float] = {}
    for domain in DOMAINS:
        members = [value for name, value in families.items() if FAMILY_DOMAIN.get(name) == domain]
        if members:
            domains[domain] = geometric_mean(members)
    return {
        "family_tflops": families,
        "domain_tflops": domains,
        "primary_tflops": geometric_mean(domains.values()),
    }


def run_suite(candidate_path: Path, raw_cases: Sequence[Mapping[str, Any]], *, trials: int,
              warmups: int, require_hardware: bool = True,
              edge_cases: Sequence[Mapping[str, Any]] = EDGE_CASES) -> dict[str, Any]:
    """Run the disclosed edge fixtures, then every scored case, and aggregate."""
    raw_cases = list(raw_cases)
    cases = [validate_case(x) for x in raw_cases]
    ids = [str(x["case_id"]) for x in cases] + [str(x["case_id"]) for x in edge_cases]
    if not cases or len(ids) != len(set(ids)):
        raise ValueError("suite IDs must be unique")
    if sum(int(x["input_bytes"]) for x in cases) > MAX_SUITE_INPUT_BYTES:
        raise ValueError("suite exceeds input envelope")
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    hardware = hardware_inventory(torch) if require_hardware else {}
    clock = TrustedClock.capture(torch)
    _module, forward, backward, digest = load_candidate(candidate_path)
    edges = []
    for case in edge_cases:
        try:
            edges.append(_run_edge_case(torch, clock, forward, backward, case))
        except Exception as exc:
            raise RuntimeError(f"edge case {case['case_id']} failed: {exc}") from exc
    rows = []
    warmed: set[tuple[int, ...]] = set()
    for case in sorted(raw_cases, key=lambda item: item.get("layer") is not None):
        try:
            if case.get("layer") is not None:
                rows.append(_run_real_case(torch, clock, forward, backward, case, warmed))
                continue
            rows.append(
                _run_scored_case(
                    torch, clock, forward, backward, case, trials=trials, warmups=warmups
                )
            )
        except Exception as exc:
            raise RuntimeError(f"case {case['case_id']} failed: {exc}") from exc
    return {
        "schema_version": 1,
        "generator_provenance": GENERATOR_PROVENANCE,
        "gate_calibration": GATE_CALIBRATION,
        "candidate_sha256": digest,
        "entrypoints": list(ENTRYPOINTS),
        "hardware": hardware,
        "trials_per_case": trials,
        "warmups": warmups,
        "correctness_ok": True,
        "edge_ok": True,
        "fresh_input_ok": True,
        "fresh_output_ok": True,
        "saved_cap_ok": True,
        "edge_case_ids": list(EDGE_CASE_IDS),
        "edge_cases": edges,
        **aggregate_case_results(rows),
        "cases": rows,
    }
