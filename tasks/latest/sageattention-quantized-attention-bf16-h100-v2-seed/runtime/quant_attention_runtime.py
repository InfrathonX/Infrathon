#!/usr/bin/env python3
"""Trusted runtime for SageAttention-derived quantized attention BF16 H100 v2."""
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

ENTRYPOINT = "quantized_attention_bf16"
ENTRYPOINTS = (ENTRYPOINT,)
FAMILIES = ("video", "image", "llm", "real")
TIERS = ("recipe", "sweep", "real")
SYNTHETIC_FAMILIES = ("video", "image", "llm")
REGIMES = ("norm", "raw")
# One scored case of the Judge-only final suite (final-holdout-v3) or of the public suite.
CASE_KEYS = frozenset({"case_id", "family", "tier", "model", "b", "h_q", "h_kv", "d", "l",
                       "is_causal", "regime", "seed", "gated", "layer"})
SUPPORTED_HEAD_DIMS = (64, 128)
MAX_L = 262144
MAX_BATCH = 4
MAX_HEADS = 64
L1_MULT, L1_FLOOR = 3.0, 0.025
HEAD_MULT, HEAD_FLOOR = 3.0, 0.04
P999_MULT, P999_FLOOR = 2.0, 0.02
ROWMAX_MULT, ROWMAX_FLOOR = 2.5, 0.05
COS_FLOOR, ROW_NORM_FLOOR = 0.998, 1.0e-6
REFERENCE_CHUNK = 1024
# From this length on, the golden checks a hidden, stratified eighth of the query rows
# (whole 1024-row chunks: the first, the last and random ones in between).
ROW_SAMPLE_MIN_L = 65536
ROW_SAMPLE_STRIDE = 8
MAX_CASE_INPUT_BYTES = 6 * (1 << 30)
# Untimed warm-up calls of one case (compilation and autotuning included) may take at most
# this long; a kernel that recompiles or re-tunes for every length would otherwise exhaust
# the evaluation's time limit.
WARMUP_WALL_LIMIT_SECONDS = 60.0
GENERATOR_PROVENANCE = "sageattention-d1a57a5-derived-quantized-attention-v1-judge-port"
# The e8 edge input is a real Wan2.1-T2V-1.3B activation slice.  It is bound by
# its recipe, not by the SHA-256 of one host's bytes: every deployment captures
# its own copy at build time (fixtures/recipe/build_fixture.sh) and ships the
# capture manifest next to the file.  The runtime verifies file-against-manifest,
# manifest-against-recipe, then geometry, dtype, layout, finiteness and the
# recipe's sanity bands, so a wrong capture fails closed although bytes differ.
FIXTURE_NAME = "e8_wan13b_p0_b15_s00_cond_8192.pt"
FIXTURE_MANIFEST_NAME = "e8_wan13b_p0_b15_s00_cond_8192.manifest.json"
RECIPE_NAME = "RECIPE.json"
RECIPE_ID = "e8-wan21-t2v-1.3b-p0-b15-s00-cond-8192-v1"
FIXTURE_LAYOUT = "B,L,H,D"
# Real-model attention layers, scored as the "real" family.  Each source is one recipe in
# real_layers/RECIPE_<source>.json; its capture lives under
# $SAGE_QATTN_FIXTURE_DIR/real_layers/<capture_dir>/ (mounted read-only into the Judge) and
# is bound to the shipped recipe by capture.manifest.json.  A real case names
# "<source>/<NN>", the captured layer file layerNN.pt.  source -> (recipe file, recipe id,
# capture dir, (B, H_q, H_kv, D, L, causal)).  Row p99.9 uses 2.5x the SageAttention-1
# reference on real layers (2.0x for synthetic and edge cases): upstream's default H100
# recipe (INT8 QK + FP8 PV) needs 2.08x on one real layer.
REAL_SOURCES: dict[str, tuple[str, str, str, tuple[int, int, int, int, int, bool]]] = {
    "cogvideox2b-s00": ("RECIPE_cogvideox2b.json", "rl-cogvideox2b-p0-s00-v1", "cogvideox2b-p0-s00", (2, 30, 30, 64, 17776, False)),
    "llama31-16k": ("RECIPE_llama31_8b.json", "rl-llama31-8b-wikitext2-16k-v1", "llama31-8b-wikitext2-16k", (1, 32, 8, 128, 16384, True)),
    "cogvideox2b-s10": ("RECIPE_cogvideox2b_s10.json", "rl-cogvideox2b-p1-s10-v1", "cogvideox2b-p1-s10", (2, 30, 30, 64, 17776, False)),
    "cogvideox2b-s40": ("RECIPE_cogvideox2b_s40.json", "rl-cogvideox2b-p1-s40-v1", "cogvideox2b-p1-s40", (2, 30, 30, 64, 17776, False)),
    "wan13b-480p": ("RECIPE_wan13b_480p.json", "rl-wan21-1.3b-480p-v1", "wan13b-480p", (1, 12, 12, 128, 32760, False)),
    "wan14b-720p": ("RECIPE_wan14b_720p.json", "rl-wan21-14b-720p-v1", "wan14b-720p", (1, 40, 40, 128, 75600, False)),
    "llama31-64k": ("RECIPE_llama31_64k.json", "rl-llama31-8b-wikitext103-64k-v1", "llama31-8b-wikitext103-64k", (1, 32, 8, 128, 65536, True)),
    "llama31-128k": ("RECIPE_llama31_128k.json", "rl-llama31-8b-wikitext103-128k-v1", "llama31-8b-wikitext103-128k", (1, 32, 8, 128, 131072, True)),
    "llama3-262k": ("RECIPE_llama3_262k.json", "rl-llama3-8b-262k-wikitext103-262k-v1", "llama3-8b-262k-wikitext103-262k", (1, 32, 8, 128, 262144, True)),
    "sd35m-1024": ("RECIPE_sd35m_1024.json", "rl-sd35-medium-1024-v1", "sd35m-1024", (2, 24, 24, 64, 4429, False)),
}
REAL_LAYER_P999_MULT = 2.5
BF16_REFERENCE_BACKEND = "torch.nn.functional.scaled_dot_product_attention[cudnn]"

# Ported from lab/qattn_family.py. The cell fixes regime after the prototype's
# first regime-probability draw, preserving every subsequent draw position.
RANGES = {
    "p_raw": 0.5, "k_bias_channels": (4, 12), "k_bias_norm": (1.0, 4.0),
    "k_bias_raw": (6.0, 24.0), "k_bias_sigma": (0.1, 1.0),
    "q_bias_channels": (0, 4), "q_bias": (0.5, 3.0), "q_bias_sigma": (0.05, 0.5),
    "q_tok_sigma": (0.05, 0.15), "k_tok_sigma": (0.10, 0.30),
    "v_tok_sigma": (0.10, 0.20), "v_bias_channels": (0, 4),
    "v_bias": (0.5, 3.0), "v_massive_frac": (0.0, 0.005),
    "v_massive_scale": (2.0, 5.0), "tau": (1.2, 2.0),
}

def _u(torch: Any, g: Any, lo: float, hi: float, device: Any, n: int = 1) -> Any:
    return torch.rand(n, generator=g, device=device) * (hi - lo) + lo

def draw_params(torch: Any, seed: int, regime: str, device: Any = "cuda") -> dict[str, Any]:
    g = torch.Generator(device=device).manual_seed(seed)
    r = RANGES
    torch.rand(1, generator=g, device=device).item()
    raw = regime == "raw"
    p: dict[str, Any] = {"regime": regime, "seed": seed}
    p["n_k_bias"] = int(torch.randint(r["k_bias_channels"][0], r["k_bias_channels"][1] + 1, (1,), generator=g, device=device).item())
    p["k_bias_range"] = r["k_bias_raw"] if raw else r["k_bias_norm"]
    p["n_q_bias"] = int(torch.randint(r["q_bias_channels"][0], r["q_bias_channels"][1] + 1, (1,), generator=g, device=device).item())
    p["n_v_bias"] = int(torch.randint(r["v_bias_channels"][0], r["v_bias_channels"][1] + 1, (1,), generator=g, device=device).item())
    p["q_tok_sigma"] = float(_u(torch, g, *r["q_tok_sigma"], device).item())
    p["k_tok_sigma"] = float(_u(torch, g, *r["k_tok_sigma"], device).item())
    p["v_tok_sigma"] = float(_u(torch, g, *r["v_tok_sigma"], device).item())
    p["v_massive_frac"] = float(_u(torch, g, *r["v_massive_frac"], device).item())
    p["tau"] = float(_u(torch, g, *r["tau"], device).item())
    return p

def generate_family(torch: Any, b: int, l: int, hq: int, hkv: int, d: int,
                    seed: int, regime: str, device: Any = "cuda", dtype: Any = None):
    dtype = torch.bfloat16 if dtype is None else dtype
    p = draw_params(torch, seed, regime, device)
    g = torch.Generator(device=device).manual_seed(seed ^ 0x9E3779B9)
    r = RANGES
    q = torch.randn((b, l, hq, d), generator=g, device=device)
    k = torch.randn((b, l, hkv, d), generator=g, device=device)
    v = torch.randn((b, l, hkv, d), generator=g, device=device)
    # Token scales must be applied before channel bias, leaving bias constant
    # across tokens so K-mean subtraction remains exact.
    q = q * torch.exp(p["q_tok_sigma"] * torch.randn((b, l, 1, 1), generator=g, device=device))
    k = k * torch.exp(p["k_tok_sigma"] * torch.randn((b, l, 1, 1), generator=g, device=device))
    v = v * torch.exp(p["v_tok_sigma"] * torch.randn((b, l, 1, 1), generator=g, device=device))
    for h in range(hkv):
        ch = torch.randperm(d, generator=g, device=device)[:p["n_k_bias"]]
        n = ch.numel()
        mag = _u(torch, g, *p["k_bias_range"], device, n) * torch.sign(torch.randn(n, generator=g, device=device))
        sig = _u(torch, g, *r["k_bias_sigma"], device, n)
        k[:, :, h, ch] = k[:, :, h, ch] * sig + mag
    for h in range(hq):
        if p["n_q_bias"] == 0:
            break
        ch = torch.randperm(d, generator=g, device=device)[:p["n_q_bias"]]
        n = ch.numel()
        mag = _u(torch, g, *r["q_bias"], device, n) * torch.sign(torch.randn(n, generator=g, device=device))
        sig = _u(torch, g, *r["q_bias_sigma"], device, n)
        q[:, :, h, ch] = q[:, :, h, ch] * sig + mag
    for h in range(hkv):
        if p["n_v_bias"] == 0:
            break
        ch = torch.randperm(d, generator=g, device=device)[:p["n_v_bias"]]
        n = ch.numel()
        mag = _u(torch, g, *r["v_bias"], device, n) * torch.sign(torch.randn(n, generator=g, device=device))
        v[:, :, h, ch] = v[:, :, h, ch] + mag
    n_massive = int(round(p["v_massive_frac"] * l))
    if n_massive > 0:
        idx = torch.randperm(l, generator=g, device=device)[:n_massive]
        v[:, idx] = v[:, idx] * _u(torch, g, *r["v_massive_scale"], device, 1)
    q = q * p["tau"]
    return q.to(dtype).contiguous(), k.to(dtype).contiguous(), v.to(dtype).contiguous(), p

EDGE_CASES = (
    {"case_id":"edge-e1-small","family":"dit","b":1,"l":64,"h_q":12,"h_kv":12,"d":128,"is_causal":False,"regime":"norm","seed":900011},
    {"case_id":"edge-e2-causal-tail","family":"causal","b":1,"l":4097,"h_q":4,"h_kv":2,"d":64,"is_causal":True,"regime":"raw","seed":900023},
    {"case_id":"edge-e3-mqa","family":"dit","b":2,"l":200,"h_q":8,"h_kv":1,"d":128,"is_causal":False,"regime":"norm","seed":900031},
    {"case_id":"edge-e4-batch-causal","family":"causal","b":3,"l":1000,"h_q":4,"h_kv":4,"d":64,"is_causal":True,"regime":"raw","seed":900041},
    {"case_id":"edge-e5-k-outliers","family":"dit","b":1,"l":8192,"h_q":12,"h_kv":12,"d":128,"is_causal":False,"regime":"raw","seed":900053,"k_extra_bias":40.0,"k_extra_channels":16},
    {"case_id":"edge-e6-flat","family":"dit","b":1,"l":32760,"h_q":12,"h_kv":12,"d":128,"is_causal":False,"regime":"norm","seed":900061,"q_logit_std":0.5},
    {"case_id":"edge-e7-peaked","family":"dit","b":1,"l":16384,"h_q":12,"h_kv":12,"d":128,"is_causal":False,"regime":"norm","seed":900071,"q_logit_std":4.0},
    {"case_id":"edge-e8-wan-capture","family":"dit","b":1,"l":8192,"h_q":12,"h_kv":12,"d":128,"is_causal":False,"regime":"fixture","seed":900083,"fixture":True},
)
EDGE_CASE_IDS = tuple(str(case["case_id"]) for case in EDGE_CASES)

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

def pairs_for_length(l: int, causal: bool) -> int:
    if isinstance(l, bool) or not isinstance(l, int) or l <= 0:
        raise ValueError("l must be positive")
    return l * (l + 1) // 2 if causal else l * l

def effective_tflops(b: int, h_q: int, d: int, pairs: int, seconds: float) -> float:
    return finite_positive(4.0 * b * h_q * d * pairs / finite_positive(seconds, "seconds") / 1e12, "TFLOPS")

def _int_field(case: Mapping[str, Any], key: str, lo: int, hi: int) -> int:
    value = case[key]
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise ValueError(f"invalid {key}")
    return value

def validate_case(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one scored case (schema CASE_KEYS) and derive its accounting fields."""
    if set(raw) != CASE_KEYS:
        raise ValueError("case fields differ from schema")
    case = dict(raw)
    family, tier = str(case["family"]), str(case["tier"])
    if family not in FAMILIES or tier not in TIERS or (family == "real") != (tier == "real"):
        raise ValueError("family/tier differs")
    if not isinstance(case["case_id"], str) or not isinstance(case["model"], str) or not case["model"]:
        raise ValueError("case identity differs")
    b = _int_field(case, "b", 1, MAX_BATCH)
    hq = _int_field(case, "h_q", 1, MAX_HEADS)
    hkv = _int_field(case, "h_kv", 1, MAX_HEADS)
    d = _int_field(case, "d", 1, 1024)
    l = _int_field(case, "l", 1, MAX_L)
    _int_field(case, "seed", 0, 2**63 - 1)
    causal = case["is_causal"]
    if d not in SUPPORTED_HEAD_DIMS or hq % hkv or not isinstance(causal, bool):
        raise ValueError("geometry outside the contract")
    if family in ("video", "image") and causal or family == "llm" and not causal:
        raise ValueError("family and causality differ")
    if not isinstance(case["gated"], bool) or (tier != "sweep" and not case["gated"]):
        raise ValueError("gated flag differs")
    layer = case["layer"]
    if family == "real":
        source, _, number = str(layer).partition("/")
        if source != case["model"] or source not in REAL_SOURCES or not (number.isdigit() and len(number) == 2):
            raise ValueError("real layer differs")
        if (b, hq, hkv, d, l, causal) != REAL_SOURCES[source][3] or case["regime"] != "real":
            raise ValueError("real layer geometry differs from its source")
    elif layer is not None or case["regime"] not in REGIMES:
        raise ValueError("synthetic case fields differ")
    input_bytes = 2 * b * l * (hq + 2 * hkv) * d
    if input_bytes > MAX_CASE_INPUT_BYTES:
        raise ValueError("case inputs exceed the per-case envelope")
    case.update(softmax_scale=1/math.sqrt(d),scored=True,pairs=pairs_for_length(l,causal),input_bytes=input_bytes)
    return case

def validate_edge_case(raw: Mapping[str, Any]) -> dict[str, Any]:
    case = dict(raw)
    if int(case["d"]) not in (64,128) or int(case["h_q"]) % int(case["h_kv"]):
        raise ValueError("invalid edge geometry")
    case.update(cell="edge",scored=False,softmax_scale=1/math.sqrt(int(case["d"])),pairs=pairs_for_length(int(case["l"]),bool(case["is_causal"])))
    return case

# Capability envelope: what the task contract actually needs, not the GPU model
# it was qualified on.  The GPU name and UUID are recorded as provenance only;
# they never gate admission.  Keep these two constants identical in the Judge
# verifier (tests assert the pair).
MIN_COMPUTE_CAPABILITY = (8, 0)
MIN_DEVICE_MEMORY_BYTES = 64 * 1024**3


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
    return {"name": name, "count": 1, "capability": list(capability), "uuid": str(getattr(props, "uuid", "")), "total_memory_bytes": total_memory_bytes, "min_compute_capability": list(MIN_COMPUTE_CAPABILITY), "min_device_memory_bytes": MIN_DEVICE_MEMORY_BYTES, "torch_version": str(torch.__version__), "torch_cuda_version": str(torch.version.cuda)}

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1<<20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_candidate(path: Path) -> tuple[ModuleType,Callable[...,Any],str]:
    candidate = path.resolve()
    digest = _sha256(candidate)
    if not candidate.is_file() or candidate.is_symlink():
        raise RuntimeError("candidate is not regular")
    spec = importlib.util.spec_from_file_location(
        f"quant_attention_candidate_{digest[:24]}", candidate
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("candidate cannot load")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    fn = getattr(module, ENTRYPOINT, None)
    if not callable(fn):
        raise RuntimeError(f"candidate must define {ENTRYPOINT}")
    return module, fn, digest

@dataclass(frozen=True)
class TrustedGateOps:
    isfinite: Callable[...,Any]
    all: Callable[...,Any]
    item: Callable[...,Any]
    to_float: Callable[...,Any]
    clone: Callable[...,Any]
    equal: Callable[...,Any]
    byte_view: Callable[...,Any]
    abs: Callable[...,Any]
    sub: Callable[...,Any]
    mul: Callable[...,Any]
    div: Callable[...,Any]
    sum: Callable[...,Any]
    mean: Callable[...,Any]
    square: Callable[...,Any]
    sqrt: Callable[...,Any]
    norm: Callable[...,Any]
    amax: Callable[...,Any]
    floor: Callable[...,Any]
    sign: Callable[...,Any]
    clamp: Callable[...,Any]
    where: Callable[...,Any]
    reshape: Callable[...,Any]
    pad: Callable[...,Any]
    matmul: Callable[...,Any]
    softmax: Callable[...,Any]
    transpose: Callable[...,Any]
    kthvalue: Callable[...,Any]
    empty: Callable[...,Any]
    arange: Callable[...,Any]
    masked_fill: Callable[...,Any]
    copy: Callable[...,Any]
    float_dtype: Any

    @classmethod
    def capture(cls, torch: Any) -> "TrustedGateOps":
        tensor_to, tensor_view = torch.Tensor.to, torch.Tensor.view
        return cls(
            torch.isfinite,torch.all,torch.Tensor.item,
            lambda x,cast=tensor_to,dtype=torch.float32: cast(x,dtype),
            torch.clone,torch.equal,
            lambda x,view=tensor_view,dtype=torch.uint8: view(x,dtype),
            torch.abs,torch.sub,torch.mul,torch.div,torch.sum,torch.mean,torch.square,
            torch.sqrt,torch.linalg.vector_norm,torch.amax,torch.floor,torch.sign,
            torch.clamp,torch.where,torch.reshape,torch.nn.functional.pad,torch.matmul,
            torch.softmax,torch.transpose,torch.kthvalue,torch.empty,torch.arange,
            torch.Tensor.masked_fill,torch.Tensor.copy_,torch.float32,
        )

@dataclass(frozen=True)
class TrustedClock:
    perf_counter: Callable[[],float]
    synchronize: Callable[...,Any]
    current_stream: Callable[...,Any]
    event: Callable[...,Any]
    gate: TrustedGateOps

    @classmethod
    def capture(cls, torch: Any) -> "TrustedClock":
        return cls(time.perf_counter,torch.cuda.synchronize,torch.cuda.current_stream,
                   torch.cuda.Event,TrustedGateOps.capture(torch))

@dataclass
class PreparedInvocation:
    case_id: str
    family: str
    tier: str
    scored: bool
    q: Any
    k: Any
    v: Any
    is_causal: bool
    softmax_scale: float
    out: Any
    b: int
    l: int
    h_q: int
    h_kv: int
    d: int
    pairs: int
    input_ptrs: tuple[int,...]
    output_ptrs: tuple[int,...]
    retained_values: tuple[Any,...]

def _recipe_path() -> Path:
    """The RECIPE.json shipped next to this runtime inside both images."""
    override=os.environ.get("SAGE_QATTN_RECIPE_PATH")
    here=Path(__file__).resolve().parent
    candidates=([Path(override)] if override else [])+[
        here/RECIPE_NAME, here.parent/"fixtures"/"recipe"/RECIPE_NAME
    ]
    for candidate in candidates:
        if candidate.is_file() and not candidate.is_symlink():
            return candidate
    raise RuntimeError(f"{RECIPE_NAME} is not in the runtime build context")

def load_recipe() -> dict[str,Any]:
    recipe=json.loads(_recipe_path().read_text(encoding="utf-8"))
    if not isinstance(recipe,dict) or recipe.get("recipe_id")!=RECIPE_ID:
        raise RuntimeError("e8 recipe identity differs")
    if recipe["tensors"]["layout"]!=FIXTURE_LAYOUT or recipe["tensors"]["dtype"]!="torch.bfloat16":
        raise RuntimeError("e8 recipe declares an unexpected tensor contract")
    return recipe

def _fixture_path() -> Path:
    """Resolve this deployment's capture and bind it to the shipped recipe."""
    root=os.environ.get("SAGE_QATTN_FIXTURE_DIR")
    if not root:
        raise RuntimeError("SAGE_QATTN_FIXTURE_DIR is required")
    directory=Path(root)
    path,manifest_path=directory/FIXTURE_NAME,directory/FIXTURE_MANIFEST_NAME
    for candidate in (path,manifest_path):
        if not candidate.is_file() or candidate.is_symlink():
            raise RuntimeError(f"e8 capture is missing: {candidate.name}")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest,dict):
        raise RuntimeError("e8 capture manifest is not an object")
    if manifest.get("fixture")!=FIXTURE_NAME or manifest.get("recipe_id")!=RECIPE_ID:
        raise RuntimeError("e8 capture manifest identity differs")
    if manifest.get("recipe_sha256")!=_sha256(_recipe_path()):
        raise RuntimeError("e8 capture was not produced by the shipped RECIPE.json")
    if _sha256(path)!=manifest.get("sha256"):
        raise RuntimeError("e8 capture does not match its manifest SHA-256")
    return path

def _check_fixture_bands(torch: Any,tensors: Mapping[str,Any],recipe: Mapping[str,Any]) -> None:
    """Enforce the recipe's sanity bands: a wrong capture fails closed."""
    bands=recipe["sanity_bands"]
    finite_floor=float(bands["finite_fraction_min"])
    for name,band in bands["tensors"].items():
        value=tensors[name].float()
        fraction=float(torch.isfinite(value).to(torch.float64).mean().item())
        if fraction<finite_floor:
            raise RuntimeError(f"e8 {name} finite fraction {fraction} below {finite_floor}")
        rms=float(value.square().mean().sqrt().item())
        if not float(band["rms_min"])<=rms<=float(band["rms_max"]):
            raise RuntimeError(
                f"e8 {name} RMS {rms} outside recipe band "
                f"[{band['rms_min']}, {band['rms_max']}]"
            )
        abs_max=float(value.abs().max().item())
        if not float(band["abs_max_min"])<=abs_max<=float(band["abs_max_max"]):
            raise RuntimeError(
                f"e8 {name} abs-max {abs_max} outside recipe band "
                f"[{band['abs_max_min']}, {band['abs_max_max']}]"
            )

def _real_recipe_dir() -> Path:
    """Directory holding the real-layer recipes shipped next to this runtime."""
    override=os.environ.get("SAGE_QATTN_REAL_RECIPE_DIR")
    here=Path(__file__).resolve().parent
    for candidate in ([Path(override)] if override else [])+[here/"real_layers",here.parent/"fixtures"/"recipe"/"real_layers"]:
        if candidate.is_dir() and not candidate.is_symlink():
            return candidate
    raise RuntimeError("real-layer recipes are not in the runtime build context")

def captured_layers(recipe: Mapping[str,Any]) -> tuple[int,...]:
    """Layer numbers a recipe captures: its capture.layers list, else every layer."""
    listed=recipe.get("capture",{}).get("layers")
    return tuple(int(x) for x in listed) if listed is not None else tuple(range(int(recipe["attention"]["layers"])))

_REAL_RECIPE_CACHE: dict[str,tuple[dict[str,Any],str]] = {}

def load_real_recipe(source: str) -> tuple[dict[str,Any],str]:
    if source in _REAL_RECIPE_CACHE:
        return _REAL_RECIPE_CACHE[source]
    name,recipe_id,_capture_dir,(b,hq,hkv,d,l,causal)=REAL_SOURCES[source]
    path=_real_recipe_dir()/name
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"real-layer recipe missing: {name}")
    recipe=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(recipe,dict) or recipe.get("recipe_id")!=recipe_id:
        raise RuntimeError(f"real-layer recipe identity differs: {name}")
    tensors=recipe.get("tensors",{})
    if tensors.get("layout")!=FIXTURE_LAYOUT or tensors.get("dtype")!="torch.bfloat16":
        raise RuntimeError(f"real-layer recipe declares an unexpected tensor contract: {name}")
    if (list(tensors.get("q",())),list(tensors.get("k",())),bool(recipe.get("attention",{}).get("is_causal")))!=([b,l,hq,d],[b,l,hkv,d],causal):
        raise RuntimeError(f"real-layer recipe geometry differs from the source table: {name}")
    _REAL_RECIPE_CACHE[source]=(recipe,_sha256(path))
    return _REAL_RECIPE_CACHE[source]

def _real_capture(source: str,recipe: Mapping[str,Any],recipe_sha256: str) -> tuple[Path,dict[str,Any]]:
    root=os.environ.get("SAGE_QATTN_FIXTURE_DIR")
    if not root:
        raise RuntimeError("SAGE_QATTN_FIXTURE_DIR is required")
    directory=Path(root)/"real_layers"/REAL_SOURCES[source][2]
    manifest_path=directory/"capture.manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise RuntimeError(f"real-layer capture is missing: {directory.name}")
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest,dict) or manifest.get("recipe_id")!=recipe["recipe_id"] or manifest.get("capture_dir")!=directory.name:
        raise RuntimeError(f"real-layer capture manifest identity differs: {directory.name}")
    if manifest.get("recipe_sha256")!=recipe_sha256:
        raise RuntimeError(f"real-layer capture {directory.name} was not produced by the shipped recipe")
    return directory,manifest

def _load_real_layer(torch: Any,case: Mapping[str,Any]) -> tuple[dict[str,Any],dict[str,Any],str]:
    source,_,number=str(case["layer"]).partition("/")
    layer=int(number)
    recipe,recipe_sha=load_real_recipe(source)
    if layer not in captured_layers(recipe):
        raise RuntimeError(f"real layer {case['layer']} is not captured by its recipe")
    directory,manifest=_real_capture(source,recipe,recipe_sha)
    name=f"layer{layer:02d}.pt"
    entry=manifest.get("files",{}).get(name)
    path=directory/name
    if not isinstance(entry,dict) or not path.is_file() or path.is_symlink():
        raise RuntimeError(f"real layer file missing: {directory.name}/{name}")
    if _sha256(path)!=entry.get("sha256"):
        raise RuntimeError(f"real layer {directory.name}/{name} does not match its manifest SHA-256")
    blob=torch.load(path,map_location="cuda",weights_only=True)
    meta=blob.get("meta") if isinstance(blob,dict) else None
    if not isinstance(meta,dict) or meta.get("recipe_id")!=recipe["recipe_id"] or meta.get("layer")!=layer:
        raise RuntimeError(f"real layer {name} metadata differs from its recipe")
    tensors={}
    for key in ("q","k","v"):
        x=blob.get(key)
        if x is None or x.dtype is not torch.bfloat16 or list(x.shape)!=list(recipe["tensors"][key]) or not x.is_contiguous():
            raise RuntimeError(f"real layer {name} {key} differs from the recipe tensor contract")
        tensors[key]=x
    bands=recipe["sanity_bands"]["layers"][f"{layer:02d}"]
    for key,x in tensors.items():
        xf=x.float()
        if not bool(torch.isfinite(xf).all().item()):
            raise RuntimeError(f"real layer {name} {key} is not finite")
        rms=float(xf.square().mean().sqrt().item()); amax=float(xf.abs().max().item()); band=bands[key]
        if not (float(band["rms_min"])<=rms<=float(band["rms_max"]) and float(band["abs_max_min"])<=amax<=float(band["abs_max_max"])):
            raise RuntimeError(f"real layer {name} {key} outside the recipe sanity band")
    return tensors,recipe,directory.name

def _generate_inputs(torch: Any, case: Mapping[str,Any], seed: int, device: Any=None) -> dict[str,Any]:
    device = torch.device("cuda") if device is None else torch.device(device)
    if case.get("fixture"):
        recipe = load_recipe()
        blob = torch.load(_fixture_path(),map_location=device,weights_only=False)
        if not isinstance(blob,dict) or not all(x in blob for x in ("q","k","v")):
            raise RuntimeError("bad e8 payload")
        if any(blob[x].dtype is not torch.bfloat16 for x in ("q","k","v")):
            raise RuntimeError("bad e8 dtype")
        meta = blob.get("meta") if isinstance(blob.get("meta"),Mapping) else {}
        if meta.get("tensor_layout",FIXTURE_LAYOUT) != FIXTURE_LAYOUT:
            raise RuntimeError("bad e8 layout")
        if meta.get("recipe_id",RECIPE_ID) != RECIPE_ID:
            raise RuntimeError("bad e8 recipe identity")
        q,k,v = (
            blob[x].to(device=device,dtype=torch.bfloat16).contiguous()
            for x in ("q","k","v")
        )
        expected = tuple(int(x) for x in recipe["tensors"]["shape"])
        if (
            tuple(q.shape) != expected
            or tuple(k.shape) != tuple(q.shape)
            or tuple(v.shape) != tuple(q.shape)
        ):
            raise RuntimeError("bad e8 geometry")
        _check_fixture_bands(torch,{"q":q,"k":k,"v":v},recipe)
        return {"q":q,"k":k,"v":v,"params":{"regime":"fixture","seed":seed}}
    q,k,v,params = generate_family(
        torch,
        int(case["b"]),
        int(case["l"]),
        int(case["h_q"]),
        int(case["h_kv"]),
        int(case["d"]),
        seed,
        str(case["regime"]),
        device=device,
        dtype=torch.bfloat16,
    )
    if "k_extra_bias" in case:
        n = min(int(case["k_extra_channels"]),int(case["d"]))
        signs = torch.where(torch.arange(n,device=device)%2==0,1.0,-1.0)
        k[:,:,:,:n] = k[:,:,:,:n]+signs*float(case["k_extra_bias"])
        k = k.contiguous()
    if "q_logit_std" in case:
        q = (q.float()*(float(case["q_logit_std"])/float(params["tau"]))).to(torch.bfloat16).contiguous()
    return {"q":q,"k":k,"v":v,"params":params}

def _prepare_invocation(torch: Any, case: Mapping[str,Any], seed: int) -> PreparedInvocation:
    tensors = _generate_inputs(torch,case,seed)
    q,k,v = tensors["q"],tensors["k"],tensors["v"]
    return _prepared(case,q,k,v,str(case.get("tier","edge")),bool(case["scored"]),bool(case["is_causal"]),float(case["softmax_scale"]))

def _prepared(case: Mapping[str,Any],q: Any,k: Any,v: Any,tier: str,scored: bool,causal: bool,scale: float) -> PreparedInvocation:
    out = q.new_full(q.shape,float("nan"))
    b,l,hq,d = (int(x) for x in q.shape)
    return PreparedInvocation(str(case["case_id"]),str(case["family"]),tier,scored,q,k,v,causal,scale,out,b,l,hq,int(k.shape[2]),d,pairs_for_length(l,causal),tuple(int(x.data_ptr()) for x in (q,k,v)),(int(out.data_ptr()),),(q,k,v,out))

def _invoke(fn: Callable[...,Any], p: PreparedInvocation) -> None:
    if fn(p.q,p.k,p.v,p.is_causal,p.softmax_scale,p.out) is not None:
        raise RuntimeError(f"{ENTRYPOINT} must return None")

def _poison_outputs(p: PreparedInvocation) -> None:
    p.out.fill_(float("nan"))

def _snapshot_inputs(clock: TrustedClock,p: PreparedInvocation) -> tuple[Any,...]:
    return tuple(clock.gate.clone(clock.gate.byte_view(x)) for x in (p.q,p.k,p.v))

def _inputs_unchanged(clock: TrustedClock,p: PreparedInvocation,snapshot: Sequence[Any]) -> bool:
    return len(snapshot)==3 and all(bool(clock.gate.equal(clock.gate.byte_view(x),y)) for x,y in zip((p.q,p.k,p.v),snapshot,strict=True))

def _quant_blocks(g: TrustedGateOps,x: Any,block: int,qmax: int=127) -> Any:
    b,l,h,d = (int(v) for v in x.shape)
    pad = (-l)%block
    xp = g.pad(x,(0,0,0,0,0,pad)) if pad else x
    xb = g.reshape(xp,(b,-1,block,h,d))
    scale = g.div(g.amax(g.abs(xb),dim=(2,4),keepdim=True),float(qmax))+1e-7
    z = g.div(xb,scale)
    rounded = g.mul(g.sign(z),g.floor(g.abs(z)+0.5))
    return g.reshape(g.mul(g.clamp(rounded,-qmax,qmax),scale),(b,-1,h,d))[:,:l]

def checked_chunks(l: int,seed: int) -> tuple[int,...]:
    """Start rows of the REFERENCE_CHUNK-row chunks the golden checks.

    Below ROW_SAMPLE_MIN_L every chunk.  From it on, a stratified 1/ROW_SAMPLE_STRIDE of the
    chunks: the first (the attention sink and BOS rows), the last, and one chunk drawn from
    each run of ROW_SAMPLE_STRIDE chunks with the case's hidden seed.
    """
    starts = list(range(0,l,REFERENCE_CHUNK))
    if l<ROW_SAMPLE_MIN_L:
        return tuple(starts)
    import random as _random
    rng = _random.Random(seed ^ 0x51C0FFEE)
    picked = {starts[0],starts[-1]}
    for i in range(0,len(starts),ROW_SAMPLE_STRIDE):
        picked.add(rng.choice(starts[i:i+ROW_SAMPLE_STRIDE]))
    return tuple(sorted(picked))

def _attention_rows(g: TrustedGateOps,q: Any,k: Any,v: Any,causal: bool,scale: float,starts: Sequence[int],out: Any) -> None:
    """FP32 attention of the chunks starting at `starts`, written chunk after chunk into out.

    `out` is [B, len(starts)*chunk-ish, Hq, D]: rows land in the order of `starts`.  The causal
    golden only reads the keys a query row can see.
    """
    b,l,hq,_ = (int(x) for x in q.shape)
    groups = hq//int(k.shape[2])
    for batch in range(b):
        for head in range(hq):
            kv = head//groups
            kh = g.to_float(k[batch,:,kv,:])
            vh = g.to_float(v[batch,:,kv,:])
            row = 0
            for start in starts:
                end = min(l,start+REFERENCE_CHUNK)
                visible = end if causal else l
                qh = g.to_float(q[batch,start:end,head,:])
                scores = g.mul(g.matmul(qh,g.transpose(kh[:visible],0,1)),scale)
                if causal:
                    rows = g.arange(start,end,device=q.device)[:,None]
                    cols = g.arange(visible,device=q.device)[None,:]
                    scores = g.masked_fill(scores,cols>rows,float("-inf"))
                g.copy(out[batch,row:row+end-start,head,:],g.matmul(g.softmax(scores,dim=-1),vh[:visible]))
                row += end-start

def _metric_bundle(g: TrustedGateOps,golden: Any,predicted: Any) -> dict[str,float]:
    pred = g.to_float(predicted)
    delta = g.abs(g.sub(golden,pred))
    den = g.sum(g.abs(golden))
    rel = g.div(g.sum(delta),den)
    dot = g.sum(g.mul(golden,pred))
    cos = g.div(dot,g.sqrt(g.mul(g.sum(g.square(golden)),g.sum(g.square(pred)))))
    rmse = g.sqrt(g.mean(g.square(delta)))
    head = g.div(g.sum(delta,dim=(1,3)),g.sum(g.abs(golden),dim=(1,3)))  # per (batch, head)
    row = g.reshape(g.div(g.norm(g.sub(golden,pred),ord=2,dim=-1),g.clamp(g.norm(golden,ord=2,dim=-1),min=ROW_NORM_FLOOR)),(-1,))
    kth = max(1,int(math.ceil(0.999*int(row.numel()))))
    return {"rel_l1":float(g.item(rel)),"cossim":float(g.item(cos)),"rmse":float(g.item(rmse)),"worst_head_rel_l1":float(g.item(g.amax(head))),"row_rel_l2_p999":float(g.item(g.kthvalue(row,kth).values)),"row_rel_l2_max":float(g.item(g.amax(row)))}

def _gather_rows(p: PreparedInvocation,starts: Sequence[int]) -> Any:
    pieces = [p.out[:,s:min(p.l,s+REFERENCE_CHUNK)] for s in starts]
    return pieces[0] if len(pieces)==1 else __import__("torch").cat(pieces,dim=1)

def _reference_after_candidate(clock: TrustedClock,p: PreparedInvocation,seed: int=0):
    """Candidate and SageAttention-1 reference metrics against the FP32 golden, on the checked rows."""
    g = clock.gate
    starts = checked_chunks(p.l,seed)
    rows = sum(min(p.l,s+REFERENCE_CHUNK)-s for s in starts)
    shape = (p.b,rows,p.h_q,p.d)
    golden = g.empty(shape,dtype=g.float_dtype,device=p.q.device)
    reference = g.empty(shape,dtype=g.float_dtype,device=p.q.device)
    _attention_rows(g,p.q,p.k,p.v,p.is_causal,p.softmax_scale,starts,golden)
    qf = g.to_float(p.q)
    kf = g.to_float(p.k)
    kc = g.sub(kf,g.mean(kf,dim=1,keepdim=True))
    qq,kq = _quant_blocks(g,qf,128),_quant_blocks(g,kc,64)
    del qf,kf,kc
    _attention_rows(g,qq,kq,p.v,p.is_causal,p.softmax_scale,starts,reference)
    del qq,kq
    predicted = _gather_rows(p,starts)
    return _metric_bundle(g,golden,predicted),_metric_bundle(g,golden,reference),rows

def realised_limits(ref: Mapping[str,float],p999_mult: float=P999_MULT) -> dict[str,float]:
    return {"rel_l1_limit":max(L1_MULT*float(ref["rel_l1"]),L1_FLOOR),"worst_head_rel_l1_limit":max(HEAD_MULT*float(ref["worst_head_rel_l1"]),HEAD_FLOOR),"row_rel_l2_p999_limit":max(p999_mult*float(ref["row_rel_l2_p999"]),P999_FLOOR),"row_rel_l2_max_limit":max(ROWMAX_MULT*float(ref["row_rel_l2_max"]),ROWMAX_FLOOR)}

def check_metric_bundle(candidate: Mapping[str,float],reference: Mapping[str,float],p999_mult: float=P999_MULT) -> dict[str,float]:
    limits = realised_limits(reference,p999_mult)
    for name in ("rel_l1","cossim","rmse","worst_head_rel_l1","row_rel_l2_p999","row_rel_l2_max"):
        if not math.isfinite(float(candidate[name])):
            raise RuntimeError(f"nonfinite metric {name}")
    if float(candidate["rel_l1"])>limits["rel_l1_limit"]:
        raise RuntimeError("global RelL1 gate failed")
    if float(candidate["cossim"])<COS_FLOOR:
        raise RuntimeError("global CosSim gate failed")
    if float(candidate["worst_head_rel_l1"])>limits["worst_head_rel_l1_limit"]:
        raise RuntimeError("head RelL1 gate failed")
    if float(candidate["row_rel_l2_p999"])>limits["row_rel_l2_p999_limit"]:
        raise RuntimeError("row p99.9 gate failed")
    if float(candidate["row_rel_l2_max"])>limits["row_rel_l2_max_limit"]:
        raise RuntimeError("row max gate failed")
    return limits

def _check_written(clock: TrustedClock,p: PreparedInvocation) -> None:
    g = clock.gate
    if not bool(g.item(g.all(g.isfinite(p.out)))):
        raise RuntimeError("out contains unwritten or nonfinite values")

def _check_output(clock: TrustedClock,p: PreparedInvocation,reference,p999_mult: float=P999_MULT) -> dict[str,float]:
    _check_written(clock,p)
    candidate,ref,rows = reference
    metrics = {k:float(v) for k,v in candidate.items()}
    metrics.update({"reference_"+k:float(v) for k,v in ref.items()})
    metrics.update(check_metric_bundle(candidate,ref,p999_mult))
    metrics["checked_rows"] = float(rows)
    return metrics

def _measure_once(clock: TrustedClock,fn: Callable[...,Any],p: PreparedInvocation):
    start,end = clock.event(enable_timing=True),clock.event(enable_timing=True)
    clock.synchronize()
    wall = clock.perf_counter()
    start.record(clock.current_stream())
    _invoke(fn,p)
    end.record(clock.current_stream())
    clock.synchronize()
    host = finite_positive(clock.perf_counter()-wall,"host wall")
    event = finite_positive(float(start.elapsed_time(end))/1000,"event")
    return max(host,event),host,event

def _identity(case: Mapping[str,Any],p: PreparedInvocation) -> dict[str,Any]:
    return {"case_id":p.case_id,"family":p.family,"tier":p.tier,"model":str(case.get("model","")),"scored":p.scored,"b":p.b,"l":p.l,"h_q":p.h_q,"h_kv":p.h_kv,"d":p.d,"is_causal":p.is_causal,"regime":str(case["regime"]),"softmax_scale":p.softmax_scale,"pairs":p.pairs}

def _untimed_call(clock: TrustedClock,fn: Callable[...,Any],p: PreparedInvocation) -> None:
    _poison_outputs(p)
    snap = _snapshot_inputs(clock,p)
    clock.synchronize()
    _invoke(fn,p)
    clock.synchronize()
    if not _inputs_unchanged(clock,p,snap):
        raise RuntimeError("candidate modified inputs")

def _run_edge_case(torch: Any,clock: TrustedClock,fn: Callable[...,Any],raw: Mapping[str,Any]) -> dict[str,Any]:
    case = validate_edge_case(raw)
    p = _prepare_invocation(torch,case,int(case["seed"]))
    snapshot = _snapshot_inputs(clock,p)
    clock.synchronize()
    _invoke(fn,p)
    clock.synchronize()
    if not _inputs_unchanged(clock,p,snapshot):
        raise RuntimeError("candidate modified inputs")
    result = {**_identity(case,p),**_check_output(clock,p,_reference_after_candidate(clock,p,int(case["seed"])))}
    del p,snapshot
    torch.cuda.empty_cache()
    return result

def gated_trial(case: Mapping[str,Any],trials: int) -> int|None:
    """The hidden trial whose output a gated case compares against the golden."""
    return int(case["seed"])%trials if case["gated"] else None

def _run_scored_case(torch: Any,clock: TrustedClock,fn: Callable[...,Any],raw: Mapping[str,Any],*,trials: int,warmups: int) -> dict[str,Any]:
    case = validate_case({k:raw[k] for k in CASE_KEYS})
    if case["family"]=="real":
        return _run_real_layer_case(torch,clock,fn,case,trials=trials,warmups=warmups)
    if trials<=0 or warmups<=0:
        raise ValueError("trials and warmups must be positive")
    warm = _prepare_invocation(torch,case,int(case["seed"])^0x5A5A5A5A)
    started = clock.perf_counter()
    for _ in range(warmups):
        _untimed_call(clock,fn,warm)
    warmup_seconds = clock.perf_counter()-started
    if warmup_seconds>WARMUP_WALL_LIMIT_SECONDS:
        raise RuntimeError(f"warm-up took {warmup_seconds:.1f} s, over the {WARMUP_WALL_LIMIT_SECONDS:.0f} s limit")
    _check_written(clock,warm)
    decoy = _prepare_invocation(torch,case,int(case["seed"])^0x3C3C3C3C)
    reserved_i = set(warm.input_ptrs)|set(decoy.input_ptrs)
    reserved_o = set(warm.output_ptrs)|set(decoy.output_ptrs)
    seen_i: set[int] = set()
    seen_o: set[int] = set()
    retained = []
    us, tflops, trial_pairs = [], [], []
    metrics: dict[str,float] = {}
    checked = gated_trial(case,trials)
    for trial in range(trials):
        p = _prepare_invocation(torch,case,int(case["seed"])+(trial+1)*104729)
        if set(p.input_ptrs)&(seen_i|reserved_i):
            raise RuntimeError("fresh-input protocol failed")
        if set(p.output_ptrs)&(seen_o|reserved_o):
            raise RuntimeError("fresh-output protocol failed")
        seen_i.update(p.input_ptrs)
        seen_o.update(p.output_ptrs)
        _untimed_call(clock,fn,decoy)
        _poison_outputs(p)
        snap = _snapshot_inputs(clock,p)
        measured,_,_ = _measure_once(clock,fn,p)
        if not _inputs_unchanged(clock,p,snap):
            raise RuntimeError("candidate modified inputs")
        del snap
        if trial==checked:
            metrics = _check_output(clock,p,_reference_after_candidate(clock,p,int(case["seed"])))
        else:
            _check_written(clock,p)
        us.append(measured*1e6)
        tflops.append(effective_tflops(p.b,p.h_q,p.d,p.pairs,measured))
        trial_pairs.append(p.pairs)
        retained.append(p.retained_values)
    result = {**_identity(case,p),"entrypoint":ENTRYPOINT,"gated":bool(case["gated"]),"warmup_seconds":warmup_seconds,"trial_us":us,"trial_tflops":tflops,"trial_pairs":trial_pairs,"median_us":statistics.median(us),"tflops":statistics.median(tflops),"fresh_input_ok":len(seen_i)==3*trials,"fresh_output_ok":len(seen_o)==trials,**metrics}
    del retained,warm,decoy,p
    torch.cuda.empty_cache()
    return result

def _run_real_layer_case(torch: Any,clock: TrustedClock,fn: Callable[...,Any],case: Mapping[str,Any],*,trials: int,warmups: int) -> dict[str,Any]:
    """One captured layer, timed like a synthetic case.

    Untimed warm-up call(s) on the loaded layer; then `trials` timed calls, each on a fresh copy
    of the layer (new input and output buffers, so nothing keyed by an earlier call's pointers
    can stand in for the work) and each after an untimed call on the warm-up buffers. The case
    value is the median; one hidden trial is compared against the golden.
    """
    tensors,recipe,capture_dir=_load_real_layer(torch,case)
    q,k,v=tensors["q"],tensors["k"],tensors["v"]
    causal=bool(recipe["attention"]["is_causal"])
    scale=float(recipe["attention"]["softmax_scale"])
    warm=_prepared(case,q,k,v,"real",True,causal,scale)
    started=clock.perf_counter()
    for _ in range(warmups):
        _untimed_call(clock,fn,warm)
    warmup_seconds=clock.perf_counter()-started
    if warmup_seconds>WARMUP_WALL_LIMIT_SECONDS:
        raise RuntimeError(f"warm-up took {warmup_seconds:.1f} s, over the {WARMUP_WALL_LIMIT_SECONDS:.0f} s limit")
    seen_i: set[int] = set(warm.input_ptrs)
    seen_o: set[int] = set(warm.output_ptrs)
    retained = []
    us, tflops, trial_pairs = [], [], []
    metrics: dict[str,float] = {}
    checked = gated_trial(case,trials)
    for trial in range(trials):
        p=_prepared(case,clock.gate.clone(q),clock.gate.clone(k),clock.gate.clone(v),"real",True,causal,scale)
        if set(p.input_ptrs)&seen_i or set(p.output_ptrs)&seen_o:
            raise RuntimeError("fresh-input protocol failed")
        seen_i.update(p.input_ptrs)
        seen_o.update(p.output_ptrs)
        _untimed_call(clock,fn,warm)
        _poison_outputs(p)
        snapshot=_snapshot_inputs(clock,p)
        measured,_,_=_measure_once(clock,fn,p)
        if not _inputs_unchanged(clock,p,snapshot):
            raise RuntimeError("candidate modified inputs")
        del snapshot
        if trial==checked:
            metrics=_check_output(clock,p,_reference_after_candidate(clock,p,int(case["seed"])),REAL_LAYER_P999_MULT)
        else:
            _check_written(clock,p)
        us.append(measured*1e6)
        tflops.append(effective_tflops(p.b,p.h_q,p.d,p.pairs,measured))
        trial_pairs.append(p.pairs)
        retained.append(p.retained_values)
    row={**_identity(case,p),"entrypoint":ENTRYPOINT,"layer":str(case["layer"]),"recipe_id":recipe["recipe_id"],"capture_dir":capture_dir,"gated":True,"warmup_seconds":warmup_seconds,"row_p999_multiplier":REAL_LAYER_P999_MULT,"trial_us":us,"trial_tflops":tflops,"trial_pairs":trial_pairs,"median_us":statistics.median(us),"tflops":statistics.median(tflops),"fresh_input_ok":True,"fresh_output_ok":True,**metrics}
    del retained,warm,p,tensors,q,k,v
    torch.cuda.empty_cache()
    return row


def summarize_real_layers(rows: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    """Average and worst layer per source, the SageAttention papers' reporting unit."""
    summary: dict[str,Any] = {}
    for source in REAL_SOURCES:
        group=[r for r in rows if r.get("family")=="real" and r.get("model")==source]
        if not group:
            continue
        item: dict[str,Any] = {"layers":len(group)}
        for prefix in ("","reference_"):
            rel=[float(r[prefix+"rel_l1"]) for r in group]; cos=[float(r[prefix+"cossim"]) for r in group]; rmse=[float(r[prefix+"rmse"]) for r in group]
            item[(prefix or "candidate_")+"rel_l1_mean"]=math.fsum(rel)/len(rel); item[(prefix or "candidate_")+"rel_l1_worst"]=max(rel)
            item[(prefix or "candidate_")+"cossim_mean"]=math.fsum(cos)/len(cos); item[(prefix or "candidate_")+"cossim_worst"]=min(cos)
            item[(prefix or "candidate_")+"rmse_mean"]=math.fsum(rmse)/len(rmse); item[(prefix or "candidate_")+"rmse_worst"]=max(rmse)
        summary[source]=item
    return summary

@dataclass(frozen=True)
class TrustedReference:
    """Torch SDPA (cuDNN backend) captured before the candidate is imported."""
    sdpa: Callable[...,Any]
    sdpa_kernel: Callable[...,Any]
    backend: Any

    @classmethod
    def capture(cls, torch: Any) -> "TrustedReference":
        from torch.nn.attention import SDPBackend, sdpa_kernel
        return cls(torch.nn.functional.scaled_dot_product_attention,sdpa_kernel,SDPBackend.CUDNN_ATTENTION)

def bf16_reference(torch: Any,clock: TrustedClock,ref: TrustedReference,cases: Sequence[Mapping[str,Any]],*,trials: int,warmups: int) -> dict[str,Any]:
    """Same-card BF16 attention line on the recipe cases, for the private report only."""
    record: dict[str,Any] = {"backend":BF16_REFERENCE_BACKEND,"dtype":"bfloat16","scored":False,"trials_per_case":trials,"warmups":warmups,"cases_measured":"recipe"}
    try:
        values: dict[str,list[float]] = {}
        rows=[]
        for raw in cases:
            case=validate_case({k:raw[k] for k in CASE_KEYS})
            if case["tier"]!="recipe":
                continue
            q,k,v,_=generate_family(torch,case["b"],case["l"],case["h_q"],case["h_kv"],case["d"],int(case["seed"])^0x6B6B6B6B,case["regime"],device="cuda")
            qh,kh,vh=(x.transpose(1,2).contiguous() for x in (q,k,v))
            del q,k,v
            call=lambda: ref.sdpa(qh,kh,vh,is_causal=case["is_causal"],scale=case["softmax_scale"],enable_gqa=case["h_q"]!=case["h_kv"])
            seconds=[]
            with ref.sdpa_kernel(ref.backend):
                for _ in range(warmups):
                    call()
                clock.synchronize()
                for _ in range(trials):
                    start,end=clock.event(enable_timing=True),clock.event(enable_timing=True)
                    clock.synchronize(); wall=clock.perf_counter(); start.record(clock.current_stream())
                    call()
                    end.record(clock.current_stream()); clock.synchronize()
                    seconds.append(max(clock.perf_counter()-wall,float(start.elapsed_time(end))/1000))
            tflops=effective_tflops(case["b"],case["h_q"],case["d"],case["pairs"],statistics.median(seconds))
            rows.append({"case_id":case["case_id"],"family":case["family"],"model":case["model"],"l":case["l"],"median_us":statistics.median(seconds)*1e6,"tflops":tflops})
            values.setdefault(case["family"],[]).append(tflops)
            del qh,kh,vh
            torch.cuda.empty_cache()
        families={name:geometric_mean(group) for name,group in values.items()}
        record.update(ok=True,cases=rows,family_tflops=families)
    except Exception as exc:
        record.update(ok=False,error=f"{type(exc).__name__}: {exc}"[:500])
    return record

def family_score(rows: Sequence[Mapping[str,Any]],family: str) -> float:
    """Score of one family.

    Synthetic families: sqrt(geomean(recipe cases) x geomean(sweep cases)), so the number of
    extrapolated settings cannot outweigh the published ones (a family without one of the two
    tiers uses the other alone).  Real: geomean over sources of each source's layer geomean.
    """
    group=[r for r in rows if r.get("family")==family]
    if not group:
        raise ValueError(f"family {family} has no cases")
    if family=="real":
        by: dict[str,list[float]] = {}
        for r in group:
            by.setdefault(str(r["model"]),[]).append(finite_positive(r["tflops"],"case TFLOPS"))
        return geometric_mean(geometric_mean(v) for v in by.values())
    tiers=[geometric_mean(finite_positive(r["tflops"],"case TFLOPS") for r in group if r["tier"]==t) for t in ("recipe","sweep") if any(r["tier"]==t for r in group)]
    return geometric_mean(tiers)

def aggregate_case_results(results: Sequence[Mapping[str,Any]],families: Sequence[str]=FAMILIES) -> dict[str,Any]:
    scored = [r for r in results if bool(r.get("scored"))]
    if not scored:
        raise ValueError("no scored cases")
    present=[f for f in families if any(r.get("family")==f for r in scored)]
    values = {name:family_score(scored,name) for name in present}
    return {"family_tflops":values,"primary_tflops":geometric_mean(values.values())}

def run_suite(candidate_path: Path,raw_cases: Sequence[Mapping[str,Any]],*,trials: int,warmups: int,require_hardware: bool=True,edge_cases: Sequence[Mapping[str,Any]]=EDGE_CASES,measure_bf16_reference: bool=False,progress: Callable[[str],None]|None=None) -> dict[str,Any]:
    cases = [validate_case(x) for x in raw_cases]
    ids = [str(x["case_id"]) for x in cases]+[str(x["case_id"]) for x in edge_cases]
    if not cases or len(ids)!=len(set(ids)):
        raise ValueError("suite IDs must be unique")
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    hardware = hardware_inventory(torch) if require_hardware else {}
    clock = TrustedClock.capture(torch)
    reference = TrustedReference.capture(torch) if measure_bf16_reference else None
    _module,fn,digest = load_candidate(candidate_path)
    started = time.perf_counter()
    edges = []
    for case in edge_cases:
        try:
            edges.append(_run_edge_case(torch,clock,fn,case))
        except Exception as exc:
            raise RuntimeError(f"edge case {case['case_id']} failed: {exc}") from exc
    rows = []
    for case in cases:
        stage = "real layer" if case["family"]=="real" else "case"
        try:
            rows.append(_run_scored_case(torch,clock,fn,case,trials=trials,warmups=warmups))
        except Exception as exc:
            raise RuntimeError(f"{stage} {case['case_id']} failed: {exc}") from exc
        if progress is not None:
            progress(f"{len(rows)}/{len(cases)} {case['case_id']} {rows[-1]['tflops']:.1f}")
    real_rows=[r for r in rows if r["family"]=="real"]
    result = {"schema_version":2,"generator_provenance":GENERATOR_PROVENANCE,"candidate_sha256":digest,"entrypoints":list(ENTRYPOINTS),"hardware":hardware,"trials_per_case":trials,"warmups":warmups,"correctness_ok":True,"edge_ok":True,"real_activation_ok":True,"fresh_input_ok":all(r["fresh_input_ok"] for r in rows),"fresh_output_ok":all(r["fresh_output_ok"] for r in rows),"edge_case_ids":list(EDGE_CASE_IDS),"edge_cases":edges,"real_layer_summary":summarize_real_layers(real_rows),"suite_seconds":time.perf_counter()-started,**aggregate_case_results(rows),"cases":rows}
    if reference is not None:
        result["bf16_reference"] = bf16_reference(torch,clock,reference,cases,trials=trials,warmups=warmups)
    return result
