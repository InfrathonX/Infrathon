"""Public test runtime for the Attn-QAT training attention task."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
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
VIDEO_FAMILIES = ("d128", "d64", "cross", "real")
LLM_FAMILIES = ("llm", "llm_d256", "real_llm")
FAMILIES = VIDEO_FAMILIES + LLM_FAMILIES
DOMAINS = ("video", "llm")
FAMILY_DOMAIN: dict[str, str] = {
    **{name: "video" for name in VIDEO_FAMILIES},
    **{name: "llm" for name in LLM_FAMILIES},
}
REAL_FAMILIES = frozenset({"real", "real_llm"})
HEAD_DIMS = frozenset({64, 128, 256})

# (family, b, h, h_kv, d, s_q, s_kv, causal) of every public case.
PUBLIC_GRID: dict[str, tuple[str, int, int, int, int, int, int, bool]] = {
    "P1": ("d128", 1, 4, 4, 128, 9984, 9984, False),
    "P2": ("d128", 1, 8, 8, 128, 6120, 6120, False),
    "P3": ("d128", 1, 24, 24, 128, 2728, 2728, False),
    "P4": ("d64", 1, 16, 16, 64, 9216, 9216, False),
    "P5": ("d128", 1, 4, 4, 128, 2784, 2784, False),
    "P6": ("d128", 1, 4, 4, 128, 26208, 26208, False),
    "P7": ("d128", 1, 4, 4, 128, 57440, 57440, False),
    "P8": ("d128", 1, 8, 8, 128, 12384, 12384, False),
    "P9": ("d128", 1, 8, 8, 128, 26208, 26208, False),
    "P10": ("d128", 1, 8, 8, 128, 57440, 57440, False),
    "P11": ("d128", 1, 24, 24, 128, 9696, 9696, False),
    "P12": ("d128", 1, 24, 24, 128, 18016, 18016, False),
    "P13": ("d128", 1, 24, 24, 128, 26208, 26208, False),
    "P14": ("d64", 1, 16, 16, 64, 20480, 20480, False),
    "P15": ("d64", 1, 16, 16, 64, 40960, 40960, False),
    "P16": ("d64", 1, 24, 24, 64, 20480, 20480, False),
    "P17": ("d64", 1, 24, 24, 64, 40960, 40960, False),
    "P18": ("cross", 1, 16, 16, 64, 9216, 320, False),
    "P19": ("cross", 1, 16, 16, 64, 20480, 320, False),
    "P20": ("cross", 1, 16, 16, 64, 40960, 320, False),
    "PR": ("real", 1, 24, 24, 128, 27280, 27280, False),
    "P21": ("llm", 4, 28, 4, 128, 3584, 3584, True),
    "P22": ("llm", 2, 28, 4, 128, 7168, 7168, True),
    "P23": ("llm", 1, 28, 4, 128, 14336, 14336, True),
    "P24": ("llm", 2, 14, 2, 128, 7168, 7168, True),
    "P25": ("llm", 1, 14, 2, 128, 28672, 28672, True),
    "P26": ("llm_d256", 4, 20, 4, 256, 3584, 3584, True),
    "P27": ("llm_d256", 1, 20, 4, 256, 14336, 14336, True),
    "PRL": ("real_llm", 1, 16, 8, 128, 7168, 7168, True),
}
PUBLIC_LAYER_NAME = "public-video-layer"
PUBLIC_LLM_LAYER_NAME = "public-llm-layer"
PUBLIC_LAYERS: dict[str, str] = {"PR": PUBLIC_LAYER_NAME, "PRL": PUBLIC_LLM_LAYER_NAME}
PUBLIC_LAYER_MANIFESTS: dict[str, str] = {
    PUBLIC_LAYER_NAME: "capture.source.json",
    PUBLIC_LLM_LAYER_NAME: "capture.llm.source.json",
}
PUBLIC_LAYER_DEFAULT_DIR = "/workspace/public_layer"

GATE_TENSORS = ("o", "dq", "dk", "dv")
SCORED_GROUP = "scored"
REAL_GROUP = "real"
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
REAL_GATE_LIMITS: dict[str, float] = {
    "o": O_REL,
    "dq": 1.2e-1,
    "dk": 3.5e-1,
    "dv": 1.0e-1,
}
REAL_LLM_GATE_LIMITS: dict[str, float] = {"o": O_REL, "dq": 1.5e-1, "dk": 8.0e-2, "dv": 8.0e-2}
REAL_LLM_GROUP = "real_llm"
GATE_FLOORS: dict[str, float] = {"o": 5.0e-3, "dq": 1.5e-2, "dk": 1.0e-2, "dv": 2.0e-2}
GATE_CAPS: dict[str, float] = {"o": 1.0e-2, "dq": 1.5e-1, "dk": 3.0e-2, "dv": 3.0e-1}
EDGE_GATE_LIMITS: dict[str, dict[str, float]] = {
    "public-e1-tiny": {"o": 1.0e-02, "dq": 1.5e-02, "dk": 1.5e-02, "dv": 2.0e-02},
    "public-e2-unaligned": {"o": 8.0e-03, "dq": 3.0e-02, "dk": 1.0e-02, "dv": 8.0e-02},
    "public-e3-cross-attention": {"o": 8.0e-03, "dq": 8.0e-02, "dk": 2.0e-02, "dv": 3.0e-02},
    "public-e4-unaligned-cross": {"o": 1.0e-02, "dq": 3.0e-02, "dk": 1.5e-02, "dv": 8.0e-02},
    "public-e5-zero-key-zero-v-groups": {"o": 5.0e-03, "dq": 1.5e-02, "dk": 1.0e-02, "dv": 2.0e-02},
    "public-e6-rtne-ties": {"o": 8.0e-03, "dq": 2.0e-02, "dk": 2.0e-02, "dv": 5.0e-02},
    "public-e7-long-unaligned": {"o": 8.0e-03, "dq": 5.0e-02, "dk": 1.5e-02, "dv": 2.0e-02},
    "public-e8-forced-sink": {"o": 5.0e-03, "dq": 2.0e-02, "dk": 1.0e-02, "dv": 2.0e-02},
    "public-e9-tiny-do-scale": {"o": 5.0e-03, "dq": 1.5e-02, "dk": 1.0e-02, "dv": 2.0e-02},
    "public-e10-many-heads": {"o": 1.0e-02, "dq": 2.0e-02, "dk": 3.0e-02, "dv": 5.0e-02},
    "public-e11-head-dim-64": {"o": 8.0e-03, "dq": 5.0e-02, "dk": 1.5e-02, "dv": 3.0e-02},
    "public-e12-causal-gqa": {"o": 8.0e-03, "dq": 1.5e-02, "dk": 1.5e-02, "dv": 3.0e-02},
    "public-e13-causal-sink": {"o": 5.0e-03, "dq": 1.5e-02, "dk": 1.5e-02, "dv": 2.0e-02},
    "public-e14-head-dim-256": {"o": 8.0e-03, "dq": 1.5e-02, "dk": 1.5e-02, "dv": 3.0e-02},
    "public-e15-head-dim-256-full": {"o": 8.0e-03, "dq": 2.0e-02, "dk": 1.5e-02, "dv": 3.0e-02},
}
ZERO_REF_ABS = 1.0e-4
GATE_CALIBRATION = "public-2026-11-05"

REFERENCE_CHUNK = 1024
GENERATOR_CHUNK = 256
MXFP_BLOCK_SIZE = 16
LOG2E = 1.44269504
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E4M3_MAX = 448.0

MAX_CASE_INPUT_BYTES = 4 * (1 << 30)
MAX_SUITE_INPUT_BYTES = 16 * (1 << 30)
MAX_RESIDENT_BYTES = 60 * (1 << 30)
GENERATOR_PROVENANCE = "attn-qat-train-public"

TRIAL_SEED_BASE = 0x7A1A_0000
WARMUP_SEED_XOR = 0x5A5A_5A5A
GENERATOR_WARMUP_XOR = 0x0BAD_0BAD

SAVED_WALK_MAX_DEPTH = 6
SAVED_WALK_MAX_NODES = 4096

RANGES = {
    "chan_sigma_q": (0.02, 0.62),
    "chan_sigma_k": (0.02, 0.62),
    "chan_sigma_v": (0.02, 0.24),
    "n_hot_pairs": (0, 1),
    "hot_amp": (2.0, 12.0),
    "sink_n_hot_pairs": 2,
    "sink_hot_amp": (28.0, 48.0),
    "sink_second_pair_ratio": 0.535,
    "hot_q_over_k": (0.55, 1.00),
    "v_hot_amp": (1.5, 4.5),
    "v_n_hot": (0, 3),
    "tok_sigma": (0.02, 0.18),
    "q_bias_frac": (0.008, 0.056),
    "k_bias_frac": (0.000, 0.082),
    "v_bias_frac": (0.000, 0.017),
    "q_bias_mag": (0.5, 5.0),
    "k_bias_mag": (0.5, 5.0),
    "v_bias_mag": (0.4, 2.5),
    "q_rms": (1.05, 3.00),
    "k_rms": (1.05, 3.10),
    "v_rms": (0.90, 2.15),
    "maxp_ordinary": (0.006, 0.90),
    "maxp_uniform": (0.0003, 0.010),
    "maxp_identity": (0.004, 0.08),
    "sink_onehot_prob": 0.35,
    "maxp_sink_onehot": (0.90, 0.99995),
    "maxp_sink_soft": (0.05, 0.90),
    "n_uniform_heads": (2, 3),
    "identity_beta": (0.55, 0.95),
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
    "maxp_probe_rows": 32,
    "maxp_bisect_steps": 22,
    "maxp_log_tau_span": 7.0,
}

DO_RANGES = {
    "chan_sigma": (0.00, 0.08),
    "n_hot_heads": (1, 2),
    "hot_amp": (1.5, 5.0),
    "tok_sigma": (0.30, 0.65),
    "p_massive": (0.000, 0.010),
    "massive_scale": (4.0, 60.0),
    "align_sigma": (0.07, 0.21),
    "do_over_o": (7e-9, 3e-5),
    "head_spread": (2.0, 7.0),
}

_SEED_MASK = 0x7FFFFFFF
TIE_MAGNITUDES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
TIE_GROUP_MAXIMUM = 6.0


def _u(torch: Any, g: Any, lo: float, hi: float, shape: Any, device: Any) -> Any:
    return torch.rand(shape, generator=g, device=device) * (hi - lo) + lo


def _logu(torch: Any, g: Any, lo: float, hi: float, shape: Any, device: Any) -> Any:
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    return (torch.rand(shape, generator=g, device=device) * (log_hi - log_lo) + log_lo).exp()


def _randint(torch: Any, g: Any, lo: int, hi: int, device: Any) -> int:
    return int(torch.randint(lo, hi + 1, (1,), generator=g, device=device).item())


def draw_params(
    torch: Any,
    seed: int,
    device: Any = "cuda",
    h: int = 12,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    g = torch.Generator(device=device).manual_seed(int(seed) & _SEED_MASK)
    r = RANGES
    p: dict[str, Any] = {"seed": int(seed), "h": int(h)}

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
    cs = (torch.randn((h, d), generator=g, device=device) * sigma).exp()
    return cs / cs.pow(2).mean(dim=1, keepdim=True).sqrt()


def _bias_vector(torch: Any, g: Any, h: int, d: int, frac: float, mag: float, device: Any) -> Any:
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
    device = o.device
    out_dtype = dtype or o.dtype
    b, s, h, d = o.shape
    g = torch.Generator(device=device)
    g.manual_seed((int(seed) ^ 0x0D0F00) & _SEED_MASK)
    of = o.float()

    cs = (torch.randn((h, d), generator=g, device=device) * params["chan_sigma"]).exp()
    key = torch.rand((h, d), generator=g, device=device)
    order = key.argsort(dim=1)
    hot_key = torch.tensor(list(params["hot_head_key"]), device=device)
    hot_heads = hot_key.argsort()[: int(params["n_hot_heads"])].tolist()
    for head in hot_heads:
        cs[int(head), order[int(head), 0]] *= params["hot_amp"]
    cs = cs / cs.pow(2).mean(dim=1, keepdim=True).sqrt()

    x = torch.randn((b, s, h, d), generator=g, device=device) * cs

    ohat = of / of.pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-30)
    align = torch.randn((b, s, h, 1), generator=g, device=device) * params["align_sigma"]
    x = x + align * x.pow(2).sum(-1, keepdim=True).sqrt() * ohat

    tok = (torch.randn((b, s, 1, 1), generator=g, device=device) * params["tok_sigma"]).exp()
    if params["p_massive"] > 0:
        hit = torch.rand((b, s, h, 1), generator=g, device=device) < params["p_massive"]
        boost = 1.0 + hit.float() * (params["massive_scale"] - 1.0) * torch.rand(
            (b, s, h, 1), generator=g, device=device
        )
        tok = tok * boost
    x = x * tok

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
    shape = tuple(v.shape)
    d = shape[-1]
    g = torch.Generator(device=v.device).manual_seed(int(seed) & _SEED_MASK)
    groups = v.reshape(shape[:-1] + (d // MXFP_BLOCK_SIZE, MXFP_BLOCK_SIZE))
    mask = torch.rand(shape[:-1] + (d // MXFP_BLOCK_SIZE, 1), generator=g, device=v.device)
    groups.mul_(torch.where(mask < fraction, 0.0, 1.0).to(v.dtype))
    return v


EDGE_CASES: tuple[dict[str, Any], ...] = (
    {"case_id": "public-e1-tiny", "b": 1, "s_q": 48, "s_kv": 48, "h": 2, "d": 128,
     "seed": 710017},
    {"case_id": "public-e2-unaligned", "b": 1, "s_q": 1003, "s_kv": 1003, "h": 3, "d": 128,
     "seed": 710029},
    {"case_id": "public-e3-cross-attention", "b": 1, "s_q": 2048, "s_kv": 768, "h": 4,
     "d": 64, "seed": 710041},
    {"case_id": "public-e4-unaligned-cross", "b": 2, "s_q": 1203, "s_kv": 517, "h": 4,
     "d": 128, "seed": 710051},
    {"case_id": "public-e5-zero-key-zero-v-groups", "b": 1, "s_q": 1536, "s_kv": 1536, "h": 3,
     "d": 128, "seed": 710063, "k_zero": True, "v_zero_groups": 0.25},
    {"case_id": "public-e6-rtne-ties", "b": 1, "s_q": 768, "s_kv": 768, "h": 3, "d": 128,
     "seed": 710071, "tie_set": True},
    {"case_id": "public-e7-long-unaligned", "b": 1, "s_q": 6136, "s_kv": 6136, "h": 8, "d": 128,
     "seed": 710083},
    {"case_id": "public-e8-forced-sink", "b": 1, "s_q": 5120, "s_kv": 5120, "h": 8, "d": 128,
     "seed": 710093, "overrides": {"maxp_targets": [1.0] * 8}},
    {"case_id": "public-e9-tiny-do-scale", "b": 1, "s_q": 3072, "s_kv": 3072, "h": 3, "d": 128,
     "seed": 710101, "do_scale": 1.0e-6},
    {"case_id": "public-e10-many-heads", "b": 1, "s_q": 160, "s_kv": 160, "h": 24, "d": 128,
     "seed": 710111},
    {"case_id": "public-e11-head-dim-64", "b": 2, "s_q": 1536, "s_kv": 1536, "h": 5, "d": 64,
     "seed": 710123},
    {"case_id": "public-e12-causal-gqa", "b": 2, "s_q": 1203, "s_kv": 1203, "h": 8, "h_kv": 2,
     "d": 128, "causal": True, "seed": 710137},
    {"case_id": "public-e13-causal-sink", "b": 1, "s_q": 3072, "s_kv": 3072, "h": 6, "h_kv": 1,
     "d": 128, "causal": True, "seed": 710149, "overrides": {"maxp_targets": [1.0] * 6}},
    {"case_id": "public-e14-head-dim-256", "b": 2, "s_q": 1280, "s_kv": 1280, "h": 4, "h_kv": 2,
     "d": 256, "causal": True, "seed": 710153},
    {"case_id": "public-e15-head-dim-256-full", "b": 1, "s_q": 901, "s_kv": 901, "h": 3,
     "d": 256, "seed": 710167, "overrides": {"maxp_targets": [0.3, 0.6, 0.9]}},
)
EDGE_CASE_IDS = tuple(str(case["case_id"]) for case in EDGE_CASES)
EXPECTED_EDGE_CASE_COUNT = len(EDGE_CASES)


def finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise RuntimeError(f"{label} must be finite and positive")
    return number


def geometric_mean(values: Iterable[float]) -> float:
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
    """`8 * base`: two QK-type passes and two PV-type products, `base = b * h * pairs * d`."""
    _check_geometry(b, h, s_q, s_kv, d)
    return 8 * b * h * attention_pairs(s_q, s_kv, causal) * d


def backward_flops(b: int, h: int, s_q: int, s_kv: int, d: int, causal: bool = False) -> int:
    """`10 * base`: QK recompute, dV, dP, dK and dQ."""
    _check_geometry(b, h, s_q, s_kv, d)
    return 10 * b * h * attention_pairs(s_q, s_kv, causal) * d


def effective_flops(b: int, h: int, s_q: int, s_kv: int, d: int, causal: bool = False) -> int:
    """`26 * base`: the forward weighted twice (activation checkpointing runs it twice per step)."""
    return 2 * forward_flops(b, h, s_q, s_kv, d, causal) + backward_flops(b, h, s_q, s_kv, d, causal)


def _check_geometry(b: int, h: int, s_q: int, s_kv: int, d: int) -> None:
    for name, value in (("b", b), ("h", h), ("s_q", s_q), ("s_kv", s_kv), ("d", d)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


def trial_seconds(t_forward: float, t_backward: float) -> float:
    forward = finite_positive(t_forward, "forward seconds")
    backward = finite_positive(t_backward, "backward seconds")
    return 2.0 * forward + backward


def effective_tflops(b: int, h: int, s_q: int, s_kv: int, d: int, seconds: float,
                     causal: bool = False) -> float:
    """Effective TFLOPS from the `26 * base` denominator and one weighted trial time."""
    flops = effective_flops(b, h, s_q, s_kv, d, causal)
    return finite_positive(flops / finite_positive(seconds, "seconds") / 1e12, "TFLOPS")


def saved_bytes_limit(b: int, h: int, s_q: int, s_kv: int, d: int, h_kv: int | None = None) -> int:
    """The saved-bundle cap: `Qf, Kf, Vf, O'` in bf16 plus the fp32 `M`."""
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
    """Peak resident bytes for one case: three live invocation sets plus the golden's buffers."""
    per_set = case_input_bytes(b, s_q, s_kv, h, d, h_kv) + case_output_bytes(b, s_q, s_kv, h, d, h_kv)
    per_set += saved_bytes_limit(b, h, s_q, s_kv, d, h_kv)
    reference = 8 * case_input_bytes(b, s_q, s_kv, h, d, h_kv) + 3 * 4 * b * h * s_q
    tile = 6 * 4 * min(s_q, REFERENCE_CHUNK) * s_kv
    return LIVE_INVOCATIONS * per_set + reference + tile


LIVE_INVOCATIONS = 3


CASE_KEYS = frozenset({"case_id", "cell", "seed", "gated", "perf_only", "layer"})


def validate_case(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a public case against the public grid and derive its geometry."""
    if set(raw) != CASE_KEYS:
        raise ValueError("case fields differ from schema")
    case = dict(raw)
    cell = str(case["cell"])
    if cell not in PUBLIC_GRID:
        raise ValueError("unknown cell")
    family, b, h, h_kv, d, s_q, s_kv, causal = PUBLIC_GRID[cell]
    seed = case["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("invalid seed")
    if not isinstance(case["gated"], bool) or not isinstance(case["perf_only"], bool):
        raise ValueError("invalid gate flags")
    real = family in REAL_FAMILIES
    if case["perf_only"] and (case["gated"] or real):
        raise ValueError("a perf-only case cannot be gated or real")
    if case["layer"] != PUBLIC_LAYERS.get(cell):
        raise ValueError("invalid real layer")
    input_bytes = case_input_bytes(b, s_q, s_kv, h, d, h_kv)
    if input_bytes > MAX_CASE_INPUT_BYTES:
        raise ValueError("case inputs exceed the per-case envelope")
    case.update(
        family=family,
        domain=FAMILY_DOMAIN[family],
        tier="real" if real else "public",
        b=b,
        h=h,
        h_kv=h_kv,
        d=d,
        s_q=s_q,
        s_kv=s_kv,
        causal=causal,
        softmax_scale=float(d) ** -0.5,
        scored=True,
        input_bytes=input_bytes,
        effective_flops=effective_flops(b, h, s_q, s_kv, d, causal),
        real_layer=case["layer"],
    )
    return case


def validate_edge_case(raw: Mapping[str, Any]) -> dict[str, Any]:
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


MIN_COMPUTE_CAPABILITY = (8, 9)
MIN_DEVICE_MEMORY_BYTES = 16 * 1024**3


def hardware_inventory(torch: Any) -> dict[str, Any]:
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidate(path: Path) -> tuple[ModuleType, Callable[..., Any], Callable[..., Any], str]:
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

    perf_counter: Callable[[], float]
    synchronize: Callable[..., Any]
    current_stream: Callable[..., Any]
    event: Callable[..., Any]
    gate: TrustedGateOps

    @classmethod
    def capture(cls, torch: Any) -> "TrustedClock":
        return cls(
            time.perf_counter,
            torch.cuda.synchronize,
            torch.cuda.current_stream,
            torch.cuda.Event,
            TrustedGateOps.capture(torch),
        )


def _e2m1_rtne_code(g: TrustedGateOps, u: Any, source_is_fp32: bool) -> Any:
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
    """The golden: section 1 of AGENT_TASK.md in chunked fp32, `[b, h, s, d]` in and out.

    `k` and `v` may carry `h_kv` heads; `dk` and `dv` sum over each group of query heads.
    With `causal`, query row `i` sees keys `0..i` and masked scores are `-inf`."""
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


def _to_bhsd(x: Any) -> Any:
    return x.permute(0, 2, 1, 3).contiguous()


def generate_inputs(torch: Any, b: int, s_q: int, s_kv: int, h: int, d: int, seed: int,
                    device: Any = "cuda", dtype: Any = None,
                    overrides: Mapping[str, Any] | None = None,
                    k_zero: bool = False, v_zero_groups: float = 0.0,
                    tie_set: bool = False, do_scale: float | None = None,
                    perf_only: bool = False, h_kv: int | None = None, causal: bool = False):
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
    if perf_only:
        # A timing-only draw: no reference attention, and a `do` nothing is compared against.
        params["regime"] = "perf-only"
        do = torch.randn_like(q.float()).to(dtype)
        return _to_bhsd(q), _to_bhsd(k), _to_bhsd(v), _to_bhsd(do), params
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


def _load_public_layer(torch: Any, case: Mapping[str, Any], device: Any) -> dict[str, Any]:
    """Load the public real captured layer and check it against its manifest."""
    directory = Path(os.environ.get("ATTN_QAT_PUBLIC_LAYER_DIR", PUBLIC_LAYER_DEFAULT_DIR))
    name_ = str(case["real_layer"])
    manifest_name = PUBLIC_LAYER_MANIFESTS.get(name_)
    if manifest_name is None:
        raise RuntimeError(f"unknown public layer {name_}")
    manifest = json.loads((directory / manifest_name).read_text(encoding="utf-8"))
    path = directory / f"{name_}.pt"
    if manifest.get("files", {}).get(name_, {}).get("sha256") != _sha256(path):
        raise RuntimeError("public layer does not match its capture manifest")
    blob = torch.load(path, map_location=device, weights_only=True)
    query_shape = (1, int(case["h"]), int(case["s_q"]), int(case["d"]))
    key_shape = (1, int(case["h_kv"]), int(case["s_kv"]), int(case["d"]))
    tensors: dict[str, Any] = {}
    for name in ("q", "k", "v", "do"):
        expected = key_shape if name in ("k", "v") else query_shape
        stored = blob[name]
        if tuple(stored.shape) != expected or stored.dtype != torch.bfloat16:
            raise RuntimeError(f"public layer {name} differs from its case")
        if not bool(torch.isfinite(stored.float()).all()):
            raise RuntimeError(f"public layer {name} is not finite")
        tensors[name] = stored.contiguous()
    tensors["params"] = {"regime": "real-training-step-capture", "seed": int(case["seed"])}
    return tensors


def _generate_case_tensors(torch: Any, case: Mapping[str, Any], seed: int,
                           device: Any = None) -> dict[str, Any]:
    device = torch.device("cuda") if device is None else torch.device(device)
    if case.get("real_layer"):
        return _load_public_layer(torch, case, device)
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
        perf_only=bool(case.get("perf_only", False)),
        h_kv=int(case.get("h_kv", case["h"])),
        causal=bool(case.get("causal", False)),
    )
    return {"q": q, "k": k, "v": v, "do": do, "params": params}


def warm_up_generator(torch: Any, case: Mapping[str, Any]) -> None:
    tensors = _generate_case_tensors(torch, case, int(case["seed"]) ^ GENERATOR_WARMUP_XOR)
    del tensors
    torch.cuda.empty_cache()


def _prepare_invocation(torch: Any, case: Mapping[str, Any], seed: int) -> PreparedInvocation:
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
    saved = _invoke_forward(forward, p)
    _invoke_backward(backward, p, saved)
    return saved


def _poison_outputs(clock: TrustedClock, p: PreparedInvocation) -> None:
    for tensor in p.outputs:
        clock.gate.fill(tensor, float("nan"))


def _poison_gradients(clock: TrustedClock, p: PreparedInvocation) -> None:
    for tensor in p.gradient_outputs:
        clock.gate.fill(tensor, float("nan"))


def _snapshot(clock: TrustedClock, tensors: Sequence[Any]) -> tuple[Any, ...]:
    gate = clock.gate
    return tuple(gate.clone(gate.byte_view(x)) for x in tensors)


def _unchanged(clock: TrustedClock, tensors: Sequence[Any], snapshot: Sequence[Any]) -> bool:
    gate = clock.gate
    if len(snapshot) != len(tensors):
        return False
    return all(
        bool(gate.equal(gate.byte_view(x), y))
        for x, y in zip(tensors, snapshot, strict=True)
    )


def saved_object_bytes(gate: TrustedGateOps, saved: Any,
                       contract: Sequence[Any]) -> tuple[int, int]:
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
    case_id = str(case.get("case_id", ""))
    if case_id in EDGE_GATE_LIMITS:
        return case_id
    if case.get("family") == "real_llm":
        return REAL_LLM_GROUP
    return REAL_GROUP if case.get("family") in REAL_FAMILIES else SCORED_GROUP


def realised_limits(case: Mapping[str, Any]) -> dict[str, float]:
    group = gate_group(case)
    if group == REAL_GROUP:
        limits = REAL_GATE_LIMITS
    elif group == REAL_LLM_GROUP:
        limits = REAL_LLM_GATE_LIMITS
    else:
        limits = EDGE_GATE_LIMITS.get(group, SCORED_GATE_LIMITS)
    result = {f"{name}_rel_l2_limit": float(limits[name]) for name in GATE_TENSORS}
    result["gate_group"] = group
    return result


def gate_failures(candidate: Mapping[str, float],
                  limits: Mapping[str, float]) -> list[str]:
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
    """Run one public case: warm-ups, then `trials` timed trials on fresh inputs.

    A gated case runs the golden on one trial, chosen from the case's seed.
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
    warm = _prepare_invocation(torch, case, base_seed ^ WARMUP_SEED_XOR)
    for _ in range(warmups):
        _poison_outputs(clock, warm)
        snapshot = _snapshot(clock, warm.inputs)
        clock.synchronize()
        saved = _invoke_step(forward, backward, warm)
        clock.synchronize()
        if not _unchanged(clock, warm.inputs, snapshot):
            raise RuntimeError("candidate modified inputs")
        check_saved(clock.gate, warm, saved)
        del saved, snapshot

    reserved_inputs = set(warm.input_ptrs)
    reserved_outputs = set(warm.output_ptrs)
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

        _poison_outputs(clock, p)
        seconds_forward, seconds_backward, saved_report = _measure_once(clock, forward, backward, p)
        weighted = trial_seconds(seconds_forward, seconds_backward)
        forward_us.append(seconds_forward * 1e6)
        backward_us.append(seconds_backward * 1e6)
        weighted_us.append(weighted * 1e6)
        tflops.append(effective_tflops(p.b, p.h, p.s_q, p.s_kv, p.d, weighted, p.causal))
        _check_finite(clock, p)
        if gated and trial == base_seed % trials:
            _merge_metrics(metrics, _check_outputs(clock, p, case))
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
    del previous, p, warm
    torch.cuda.empty_cache()
    return result


def _run_real_case(torch: Any, clock: TrustedClock, forward: Callable[..., Any],
                   backward: Callable[..., Any], raw: Mapping[str, Any],
                   warmed: set[tuple[int, ...]]) -> dict[str, Any]:
    """Run one real captured layer: one timed step and one golden comparison.

    Each layer is different data, so it needs no repeated trial.  Compilation is
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
    """Geometric mean inside each tier, across a family's tiers, then across the families.

    A family with both tiers therefore weighs its recipe cases and its sweep cases equally,
    however many of each it holds.
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
