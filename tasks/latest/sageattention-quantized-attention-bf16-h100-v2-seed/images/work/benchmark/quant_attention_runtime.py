#!/usr/bin/env python3
"""Runtime for the public tests of the quantized attention BF16 task."""
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
REGIMES = ("drift", "spiky")
# One public case: the attention geometry, the length, the input profile and the seed.
CASE_KEYS = frozenset({"case_id", "family", "b", "h_q", "h_kv", "d", "l", "is_causal", "regime", "seed", "gated"})
MAX_L = 262144
# From this length on the golden checks a stratified eighth of the query rows.
ROW_SAMPLE_MIN_L = 65536
ROW_SAMPLE_STRIDE = 8
# Untimed warm-up of one case (compilation and autotuning included) may take at most this long.
WARMUP_WALL_LIMIT_SECONDS = 60.0
L1_MULT, L1_FLOOR = 3.0, 0.025
HEAD_MULT, HEAD_FLOOR = 3.0, 0.04
P999_MULT, P999_FLOOR = 2.0, 0.02
ROWMAX_MULT, ROWMAX_FLOOR = 2.5, 0.05
COS_FLOOR, ROW_NORM_FLOOR = 0.998, 1.0e-6
REFERENCE_CHUNK = 1024
MAX_CASE_INPUT_BYTES = 6 * (1 << 30)
GENERATOR_PROVENANCE = "quantized-attention-public-tests"
REAL_LAYER_P999_MULT = 2.5
# Real-model attention layers captured for the public tests; timed and scored as "real".
PUBLIC_REAL_LAYER_CASES = (
    {"case_id": "real-video", "family": "real", "capture_dir": "public-wan22", "seed": 7101},
    {"case_id": "real-lm", "family": "real", "capture_dir": "public-lm", "seed": 7103},
)
PUBLIC_REAL_LAYER_CASE_IDS = tuple(str(c["case_id"]) for c in PUBLIC_REAL_LAYER_CASES)

# Input profiles.  Both have heavy-tailed values, correlated channels and token
# scales that change in runs.  K carries channel offsets that change along the
# tokens: a linear ramp in "drift", random steps in "spiky", which also adds a few
# high-norm keys and queries.
PROFILES: dict[str, dict[str, tuple[float, float]]] = {
    "drift": {"tail": (0.1, 0.25), "rank_weight": (0.3, 0.7), "offset": (0.5, 2.0),
              "run_sigma": (0.1, 0.3), "tau": (0.8, 1.4), "v_outlier": (2.0, 4.0), "spike": (1.0, 1.0)},
    "spiky": {"tail": (0.1, 0.2), "rank_weight": (0.3, 0.6), "offset": (0.5, 2.0),
              "run_sigma": (0.1, 0.25), "tau": (0.8, 1.2), "v_outlier": (2.0, 4.0), "spike": (1.5, 2.0)},
}


def _uniform(torch: Any, g: Any, bounds: tuple[float, float], device: Any) -> float:
    return float((torch.rand(1, generator=g, device=device) * (bounds[1] - bounds[0]) + bounds[0]).item())


def _token_runs(torch: Any, g: Any, b: int, l: int, sigma: float, device: Any) -> Any:
    """Per-token log-normal scale, constant over random runs of 64-1024 tokens."""
    scale = torch.empty((b, l, 1, 1), device=device)
    for batch in range(b):
        start = 0
        while start < l:
            run = int(torch.randint(64, 1025, (1,), generator=g, device=device).item())
            scale[batch, start:start + run] = math.exp(sigma * float(torch.randn(1, generator=g, device=device).item()))
            start += run
    return scale


def _base(torch: Any, g: Any, shape: tuple[int, int, int, int], tail: float, weight: float, device: Any) -> Any:
    """Log-normal scale mixture plus a rank-8 component shared across channels."""
    b, l, h, d = shape
    x = torch.randn(shape, generator=g, device=device)
    x = x * torch.exp(tail * torch.randn(shape, generator=g, device=device))
    basis = torch.randn((b, 1, h, 8, d), generator=g, device=device) / math.sqrt(d)
    coefficients = torch.randn((b, l, h, 1, 8), generator=g, device=device)
    return x + weight * math.sqrt(d) * (coefficients @ basis).squeeze(3)


def _channel_offsets(torch: Any, g: Any, b: int, l: int, h: int, d: int, magnitude: float, regime: str, device: Any) -> Any:
    channels = max(4, d // 8)
    offsets = torch.zeros((b, l, h, d), device=device)
    position = torch.linspace(-1.0, 1.0, l, device=device)
    for batch in range(b):
        for head in range(h):
            index = torch.randperm(d, generator=g, device=device)[:channels]
            mag = magnitude * (torch.randn(channels, generator=g, device=device).sign())
            if regime == "drift":
                shape = position
            else:
                shape = torch.empty(l, device=device)
                start = 0
                while start < l:
                    run = int(torch.randint(128, 2049, (1,), generator=g, device=device).item())
                    shape[start:start + run] = float((2 * torch.rand(1, generator=g, device=device) - 1).item())
                    start += run
            offsets[batch, :, head, index] = shape[:, None] * mag
    return offsets


def generate_family(torch: Any, b: int, l: int, hq: int, hkv: int, d: int,
                    seed: int, regime: str, device: Any = "cuda", dtype: Any = None):
    dtype = torch.bfloat16 if dtype is None else dtype
    r = PROFILES[regime]
    g = torch.Generator(device=device).manual_seed(seed)
    tail = _uniform(torch, g, r["tail"], device)
    weight = _uniform(torch, g, r["rank_weight"], device)
    sigma = _uniform(torch, g, r["run_sigma"], device)
    tau = _uniform(torch, g, r["tau"], device)
    q = _base(torch, g, (b, l, hq, d), tail, weight, device) * _token_runs(torch, g, b, l, sigma, device)
    k = _base(torch, g, (b, l, hkv, d), tail, weight, device) * _token_runs(torch, g, b, l, sigma, device)
    v = _base(torch, g, (b, l, hkv, d), tail, weight, device) * _token_runs(torch, g, b, l, sigma, device)
    k = k + _channel_offsets(torch, g, b, l, hkv, d, _uniform(torch, g, r["offset"], device), regime, device)
    n_outliers = max(1, l // 256)
    index = torch.randperm(l, generator=g, device=device)[:n_outliers]
    v[:, index] = v[:, index] * _uniform(torch, g, r["v_outlier"], device)
    if regime == "spiky":
        spike = _uniform(torch, g, r["spike"], device)
        keys = torch.randperm(l, generator=g, device=device)[:max(1, l // 512)]
        k[:, keys] = k[:, keys] * spike
        rows = torch.randperm(l, generator=g, device=device)[:max(1, l // 512)]
        q[:, rows] = q[:, rows] * spike
    q = q * tau
    p = {"regime": regime, "seed": seed, "tau": tau}
    return q.to(dtype).contiguous(), k.to(dtype).contiguous(), v.to(dtype).contiguous(), p


EDGE_CASES = (
    {"case_id":"edge-small","family":"dit","b":1,"l":96,"h_q":6,"h_kv":6,"d":128,"is_causal":False,"regime":"drift","seed":710011},
    {"case_id":"edge-causal-tail","family":"causal","b":1,"l":2049,"h_q":6,"h_kv":3,"d":64,"is_causal":True,"regime":"spiky","seed":710023},
    {"case_id":"edge-mqa","family":"dit","b":2,"l":320,"h_q":10,"h_kv":1,"d":128,"is_causal":False,"regime":"drift","seed":710031},
    {"case_id":"edge-batch-causal","family":"causal","b":4,"l":1500,"h_q":6,"h_kv":2,"d":64,"is_causal":True,"regime":"spiky","seed":710041},
    {"case_id":"edge-k-outliers","family":"dit","b":1,"l":3000,"h_q":10,"h_kv":10,"d":128,"is_causal":False,"regime":"spiky","seed":710053,"k_extra_bias":20.0,"k_extra_channels":24},
    {"case_id":"edge-flat","family":"dit","b":1,"l":5000,"h_q":8,"h_kv":8,"d":128,"is_causal":False,"regime":"drift","seed":710061,"q_logit_std":0.4},
    {"case_id":"edge-peaked","family":"dit","b":1,"l":5000,"h_q":8,"h_kv":8,"d":64,"is_causal":False,"regime":"drift","seed":710071,"q_logit_std":2.0},
    {"case_id":"edge-short-causal","family":"causal","b":3,"l":33,"h_q":4,"h_kv":4,"d":64,"is_causal":True,"regime":"spiky","seed":710083},
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
    """Validate one synthetic public case (schema CASE_KEYS) and derive its accounting."""
    if set(raw) != CASE_KEYS:
        raise ValueError("case fields differ from schema")
    case = dict(raw)
    family = str(case["family"])
    if family not in ("video", "image", "llm"):
        raise ValueError("unknown family")
    b, hq, hkv = _int_field(case, "b", 1, 4), _int_field(case, "h_q", 1, 64), _int_field(case, "h_kv", 1, 64)
    d, l = _int_field(case, "d", 1, 1024), _int_field(case, "l", 1, MAX_L)
    _int_field(case, "seed", 0, 2**63 - 1)
    causal = case["is_causal"]
    if d not in (64, 128) or hq % hkv or not isinstance(causal, bool) or case["regime"] not in REGIMES:
        raise ValueError("geometry outside the contract")
    if (family == "llm") != causal or not isinstance(case["gated"], bool):
        raise ValueError("family and causality differ")
    input_bytes = 2 * b * l * (hq + 2 * hkv) * d
    if input_bytes > MAX_CASE_INPUT_BYTES:
        raise ValueError("case inputs exceed the per-case envelope")
    case.update(tier="public",model="public",softmax_scale=1/math.sqrt(d),scored=True,pairs=pairs_for_length(l,causal),input_bytes=input_bytes)
    return case

def validate_edge_case(raw: Mapping[str, Any]) -> dict[str, Any]:
    case = dict(raw)
    if int(case["d"]) not in (64,128) or int(case["h_q"]) % int(case["h_kv"]):
        raise ValueError("invalid edge geometry")
    case.update(cell="edge",scored=False,softmax_scale=1/math.sqrt(int(case["d"])),pairs=pairs_for_length(int(case["l"]),bool(case["is_causal"])))
    return case

# What the task contract needs from the GPU.
MIN_COMPUTE_CAPABILITY = (8, 0)
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
class GateOps:
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
    def capture(cls, torch: Any) -> "GateOps":
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
class Clock:
    perf_counter: Callable[[],float]
    synchronize: Callable[...,Any]
    current_stream: Callable[...,Any]
    event: Callable[...,Any]
    gate: GateOps

    @classmethod
    def capture(cls, torch: Any) -> "Clock":
        return cls(time.perf_counter,torch.cuda.synchronize,torch.cuda.current_stream,
                   torch.cuda.Event,GateOps.capture(torch))

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

def _load_real_layer(torch: Any,case: Mapping[str,Any]) -> tuple[dict[str,Any],dict[str,Any]]:
    root=os.environ.get("QATTN_FIXTURE_DIR")
    if not root:
        raise RuntimeError("QATTN_FIXTURE_DIR is required")
    directory=Path(root)/"real_layers"/str(case["capture_dir"])
    manifest=json.loads((directory/"capture.manifest.json").read_text(encoding="utf-8"))
    path=directory/str(manifest["file"])
    if _sha256(path)!=manifest["sha256"]:
        raise RuntimeError(f"real layer {directory.name} does not match its manifest")
    blob=torch.load(path,map_location="cuda",weights_only=True)
    tensors={key:blob[key] for key in ("q","k","v")}
    for key,x in tensors.items():
        if x.dtype is not torch.bfloat16 or not x.is_contiguous() or not bool(torch.isfinite(x.float()).all().item()):
            raise RuntimeError(f"real layer {directory.name} {key} is not contiguous finite BF16")
    return tensors,manifest

def _generate_inputs(torch: Any, case: Mapping[str,Any], seed: int, device: Any=None) -> dict[str,Any]:
    device = torch.device("cuda") if device is None else torch.device(device)
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

def _snapshot_inputs(clock: Clock,p: PreparedInvocation) -> tuple[Any,...]:
    return tuple(clock.gate.clone(clock.gate.byte_view(x)) for x in (p.q,p.k,p.v))

def _inputs_unchanged(clock: Clock,p: PreparedInvocation,snapshot: Sequence[Any]) -> bool:
    return len(snapshot)==3 and all(bool(clock.gate.equal(clock.gate.byte_view(x),y)) for x,y in zip((p.q,p.k,p.v),snapshot,strict=True))

def _quant_blocks(g: GateOps,x: Any,block: int,qmax: int=127) -> Any:
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

def _attention_rows(g: GateOps,q: Any,k: Any,v: Any,causal: bool,scale: float,starts: Sequence[int],out: Any) -> None:
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

def _metric_bundle(g: GateOps,golden: Any,predicted: Any) -> dict[str,float]:
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

def _reference_after_candidate(clock: Clock,p: PreparedInvocation,seed: int=0):
    """Candidate and quantized-reference metrics against the FP32 golden, on the checked rows."""
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

def _check_written(clock: Clock,p: PreparedInvocation) -> None:
    g = clock.gate
    if not bool(g.item(g.all(g.isfinite(p.out)))):
        raise RuntimeError("out contains unwritten or nonfinite values")

def _check_output(clock: Clock,p: PreparedInvocation,reference,p999_mult: float=P999_MULT) -> dict[str,float]:
    _check_written(clock,p)
    candidate,ref,rows = reference
    metrics = {k:float(v) for k,v in candidate.items()}
    metrics.update({"reference_"+k:float(v) for k,v in ref.items()})
    metrics.update(check_metric_bundle(candidate,ref,p999_mult))
    metrics["checked_rows"] = float(rows)
    return metrics

def _measure_once(clock: Clock,fn: Callable[...,Any],p: PreparedInvocation):
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

def _untimed_call(clock: Clock,fn: Callable[...,Any],p: PreparedInvocation) -> None:
    _poison_outputs(p)
    snap = _snapshot_inputs(clock,p)
    clock.synchronize()
    _invoke(fn,p)
    clock.synchronize()
    if not _inputs_unchanged(clock,p,snap):
        raise RuntimeError("candidate modified inputs")

def _run_edge_case(torch: Any,clock: Clock,fn: Callable[...,Any],raw: Mapping[str,Any]) -> dict[str,Any]:
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

def _run_scored_case(torch: Any,clock: Clock,fn: Callable[...,Any],raw: Mapping[str,Any],*,trials: int,warmups: int) -> dict[str,Any]:
    if raw.get("family")=="real":
        return _run_real_layer_case(torch,clock,fn,raw,trials=trials,warmups=warmups)
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
    interleaved = _prepare_invocation(torch,case,int(case["seed"])^0x3C3C3C3C)
    reserved_i = set(warm.input_ptrs)|set(interleaved.input_ptrs)
    reserved_o = set(warm.output_ptrs)|set(interleaved.output_ptrs)
    seen_i: set[int] = set()
    seen_o: set[int] = set()
    retained = []
    us, tflops, trial_pairs = [], [], []
    metrics: dict[str,float] = {}
    checked = gated_trial(case,trials)
    for trial in range(trials):
        p = _prepare_invocation(torch,case,int(case["seed"])+(trial+1)*104729)
        if set(p.input_ptrs)&(seen_i|reserved_i):
            raise RuntimeError("inputs were not new")
        if set(p.output_ptrs)&(seen_o|reserved_o):
            raise RuntimeError("outputs were not new")
        seen_i.update(p.input_ptrs)
        seen_o.update(p.output_ptrs)
        _untimed_call(clock,fn,interleaved)
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
    result = {**_identity(case,p),"entrypoint":ENTRYPOINT,"gated":bool(case["gated"]),"warmup_seconds":warmup_seconds,"trial_us":us,"trial_tflops":tflops,"trial_pairs":trial_pairs,"median_us":statistics.median(us),"tflops":statistics.median(tflops),"new_inputs_ok":len(seen_i)==3*trials,"new_outputs_ok":len(seen_o)==trials,**metrics}
    del retained,warm,interleaved,p
    torch.cuda.empty_cache()
    return result

def _run_real_layer_case(torch: Any,clock: Clock,fn: Callable[...,Any],case: Mapping[str,Any],*,trials: int,warmups: int) -> dict[str,Any]:
    """One captured public layer: untimed warm-up, then timed calls on fresh copies (median)."""
    tensors,manifest=_load_real_layer(torch,case)
    q,k,v=tensors["q"],tensors["k"],tensors["v"]
    causal=bool(manifest["is_causal"])
    scale=float(manifest["softmax_scale"])
    warm=_prepared(case,q,k,v,"real",True,causal,scale)
    started=clock.perf_counter()
    for _ in range(warmups):
        _untimed_call(clock,fn,warm)
    warmup_seconds=clock.perf_counter()-started
    if warmup_seconds>WARMUP_WALL_LIMIT_SECONDS:
        raise RuntimeError(f"warm-up took {warmup_seconds:.1f} s, over the {WARMUP_WALL_LIMIT_SECONDS:.0f} s limit")
    gated=bool(case.get("gated",True))
    checked=int(case["seed"])%trials if gated else None
    retained,us,values=[],[],[]
    metrics: dict[str,float] = {}
    for trial in range(trials):
        p=_prepared(case,clock.gate.clone(q),clock.gate.clone(k),clock.gate.clone(v),"real",True,causal,scale)
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
        values.append(effective_tflops(p.b,p.h_q,p.d,p.pairs,measured))
        retained.append(p.retained_values)
    row={**_identity({**case,"regime":"real"},p),"entrypoint":ENTRYPOINT,"gated":gated,"warmup_seconds":warmup_seconds,"row_p999_multiplier":REAL_LAYER_P999_MULT,"median_us":statistics.median(us),"tflops":statistics.median(values),**metrics}
    del retained,warm,p,tensors,q,k,v
    torch.cuda.empty_cache()
    return row


def aggregate_case_results(results: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    """Public aggregate: geomean per family, then geomean over the families present."""
    values: dict[str,list[float]] = {}
    for row in results:
        if row.get("scored"):
            values.setdefault(str(row["family"]),[]).append(finite_positive(row["tflops"],"case TFLOPS"))
    if not values:
        raise ValueError("no scored cases")
    families = {name:geometric_mean(values[name]) for name in FAMILIES if name in values}
    return {"family_tflops":families,"primary_tflops":geometric_mean(families.values())}

def run_suite(candidate_path: Path,raw_cases: Sequence[Mapping[str,Any]],*,trials: int,warmups: int,require_hardware: bool=True,edge_cases: Sequence[Mapping[str,Any]]=EDGE_CASES,real_layer_cases: Sequence[Mapping[str,Any]]=PUBLIC_REAL_LAYER_CASES,progress: Callable[[str],None]|None=None) -> dict[str,Any]:
    cases = [validate_case(x) for x in raw_cases]
    ids = [str(x["case_id"]) for x in cases]+[str(x["case_id"]) for x in edge_cases]+[str(x["case_id"]) for x in real_layer_cases]
    if len(ids)!=len(set(ids)):
        raise ValueError("suite IDs must be unique")
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    hardware = hardware_inventory(torch) if require_hardware else {}
    clock = Clock.capture(torch)
    _module,fn,digest = load_candidate(candidate_path)
    edges = []
    for case in edge_cases:
        try:
            edges.append(_run_edge_case(torch,clock,fn,case))
        except Exception as exc:
            raise RuntimeError(f"edge case {case['case_id']} failed: {exc}") from exc
    rows = []
    for case in list(real_layer_cases)+cases:
        stage = "real layer" if case.get("family")=="real" else "case"
        try:
            rows.append(_run_scored_case(torch,clock,fn,case,trials=trials,warmups=warmups))
        except Exception as exc:
            raise RuntimeError(f"{stage} {case['case_id']} failed: {exc}") from exc
        if progress is not None:
            progress(f"{case['case_id']}: {rows[-1]['tflops']:.1f} TFLOPS")
    result = {"schema_version":2,"generator_provenance":GENERATOR_PROVENANCE,"candidate_sha256":digest,"entrypoints":list(ENTRYPOINTS),"hardware":hardware,"trials_per_case":trials,"warmups":warmups,"correctness_ok":True,"edge_ok":True,"real_activation_ok":True,"new_inputs_ok":all(r.get("new_inputs_ok",True) for r in rows),"new_outputs_ok":all(r.get("new_outputs_ok",True) for r in rows),"edge_case_ids":list(EDGE_CASE_IDS),"edge_cases":edges,"real_layer_case_ids":[str(x["case_id"]) for x in real_layer_cases],"cases":rows}
    if rows:
        result.update(aggregate_case_results(rows))
    return result
