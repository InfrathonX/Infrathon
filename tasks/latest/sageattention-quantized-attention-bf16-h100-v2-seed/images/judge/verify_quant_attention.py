#!/usr/bin/env python3
"""Private verifier for SageAttention quantized attention BF16 H100 v2."""
from __future__ import annotations
import argparse, hashlib, json, math, os, pwd, re, shutil, signal, stat, statistics, subprocess, sys, tempfile, time
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from source_admission import validate_submission  # noqa: E402

JUDGE_ROOT = Path("/opt/sageattention_quant_attention_judge")
RUNTIME_ROOT = Path("/opt/sageattention_quant_attention_worker")
WORKER = JUDGE_ROOT / "candidate_worker.py"
EVALUATION_CONTEXT = Path("/tmp/.sforge-evaluation-context.json")
FINAL_MANIFEST = JUDGE_ROOT / "private/final_cases.json"
RESULT_NAME = "sageattention_quant_attention_v2_results.json"
FINAL_SUITE_ID = "final-holdout-v3"
EVALUATION_POLICY = "full-final-shared-v3"
SELECTION_PROTOCOL_ID = "sageattention-v3-shared-final-v1"
PROFILE_CODES = {"auto": 1.0, "final": 2.0}
ENTRYPOINT = "quantized_attention_bf16"
FAMILIES = ("video", "image", "llm", "real")
TIERS = ("recipe", "sweep", "real")
REGIMES = ("norm", "raw")
CASE_KEYS = {"case_id", "family", "tier", "model", "b", "h_q", "h_kv", "d", "l",
             "is_causal", "regime", "seed", "gated", "layer"}
# The shape of final-holdout-v3 (scripts/generate_final_manifest.py).
EXPECTED_TIER_COUNTS = {
    "video": {"recipe": 7, "sweep": 192},
    "image": {"recipe": 3, "sweep": 41},
    "llm": {"recipe": 4, "sweep": 388},
    "real": {"real": 139},
}
EXPECTED_CASE_COUNT = 774
EXPECTED_EDGE_CASE_COUNT = 8
EXPECTED_TRIALS = 3
EXPECTED_WARMUPS = 1
MAX_L = 262144
MAX_BATCH = 4
MAX_HEADS = 64
MAX_CASE_INPUT_BYTES = 6 * (1 << 30)
# Public geometries (H_q, H_kv, D, causal) and lengths; no synthetic Judge case uses either.
PUBLIC_GEOMETRIES = frozenset({(20, 20, 128, False), (16, 16, 64, False), (16, 16, 128, False),
                               (24, 6, 128, True), (12, 4, 64, True), (16, 4, 128, True)})
PUBLIC_LENGTHS = frozenset({5184, 24192, 61441, 96001, 6697, 20276, 44017, 2050, 5001, 4700,
                            4160, 14336, 65001, 190001, 3001, 7168, 32001})
# Real sources: (B, H_q, H_kv, D, L, causal) and the captured layer files the suite scores.
REAL_SOURCES = {
    "cogvideox2b-s00": ((2, 30, 30, 64, 17776, False), tuple(range(30))),
    "llama31-16k": ((1, 32, 8, 128, 16384, True), tuple(range(32))),
    "cogvideox2b-s10": ((2, 30, 30, 64, 17776, False), tuple(range(10))),
    "cogvideox2b-s40": ((2, 30, 30, 64, 17776, False), tuple(range(10))),
    "wan13b-480p": ((1, 12, 12, 128, 32760, False), tuple(range(15))),
    "wan14b-720p": ((1, 40, 40, 128, 75600, False), tuple(range(8))),
    "llama31-64k": ((1, 32, 8, 128, 65536, True), (0, 10, 20, 31)),
    "llama31-128k": ((1, 32, 8, 128, 131072, True), (0, 10, 20, 31)),
    "llama3-262k": ((1, 32, 8, 128, 262144, True), (0, 31)),
    "sd35m-1024": ((2, 24, 24, 64, 4429, False), tuple(range(24))),
}
REAL_LAYER_P999_MULT = 2.5
EDGE_CASE_IDS = (
    "edge-e1-small",
    "edge-e2-causal-tail",
    "edge-e3-mqa",
    "edge-e4-batch-causal",
    "edge-e5-k-outliers",
    "edge-e6-flat",
    "edge-e7-peaked",
    "edge-e8-wan-capture",
)
METRICS = (
    "rel_l1",
    "cossim",
    "rmse",
    "worst_head_rel_l1",
    "row_rel_l2_p999",
    "row_rel_l2_max",
)
LIMITS = (
    "rel_l1_limit",
    "worst_head_rel_l1_limit",
    "row_rel_l2_p999_limit",
    "row_rel_l2_max_limit",
)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()

def finite_positive(value: Any,label: str) -> float:
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(float(value)) or float(value)<=0:
        raise RuntimeError(f"{label} must be finite positive")
    return float(value)

def geometric_mean(values: Sequence[float]) -> float:
    values = list(values)
    return math.exp(math.fsum(math.log(finite_positive(v,"geomean")) for v in values)/len(values))

def _int(case: Mapping[str,Any],key: str,lo: int,hi: int) -> int:
    value = case[key]
    if isinstance(value,bool) or not isinstance(value,int) or not lo<=value<=hi:
        raise RuntimeError(f"case {key} differs")
    return value

def validate_case(raw: Mapping[str,Any]) -> dict[str,Any]:
    """One scored case of final-holdout-v3; mirrors the runtime's validate_case."""
    if set(raw)!=CASE_KEYS:
        raise RuntimeError("case fields differ")
    case = dict(raw)
    family,tier = str(case["family"]),str(case["tier"])
    if family not in FAMILIES or tier not in TIERS or (family=="real")!=(tier=="real"):
        raise RuntimeError("family/tier differs")
    if not isinstance(case["case_id"],str) or not isinstance(case["model"],str) or not case["model"]:
        raise RuntimeError("case identity differs")
    b,hq,hkv = _int(case,"b",1,MAX_BATCH),_int(case,"h_q",1,MAX_HEADS),_int(case,"h_kv",1,MAX_HEADS)
    d,l = _int(case,"d",1,1024),_int(case,"l",1,MAX_L)
    _int(case,"seed",0,2**63-1)
    causal = case["is_causal"]
    if d not in (64,128) or hq%hkv or not isinstance(causal,bool):
        raise RuntimeError("geometry outside the contract")
    if (family in ("video","image") and causal) or (family=="llm" and not causal):
        raise RuntimeError("family and causality differ")
    if not isinstance(case["gated"],bool) or (tier!="sweep" and not case["gated"]):
        raise RuntimeError("gated flag differs")
    if family=="real":
        source,_,number = str(case["layer"]).partition("/")
        if source!=case["model"] or source not in REAL_SOURCES or not (number.isdigit() and len(number)==2):
            raise RuntimeError("real layer differs")
        geometry,layers = REAL_SOURCES[source]
        if (b,hq,hkv,d,l,causal)!=geometry or int(number) not in layers or case["regime"]!="real":
            raise RuntimeError("real layer differs from its source")
    else:
        if case["layer"] is not None or case["regime"] not in REGIMES:
            raise RuntimeError("synthetic case fields differ")
        if (hq,hkv,d,causal) in PUBLIC_GEOMETRIES or l in PUBLIC_LENGTHS:
            raise RuntimeError("case overlaps a public identity")
    if 2*b*l*(hq+2*hkv)*d>MAX_CASE_INPUT_BYTES:
        raise RuntimeError("case inputs exceed the envelope")
    case.update(softmax_scale=1/math.sqrt(d),scored=True,pairs=l*(l+1)//2 if causal else l*l)
    return case

def validate_final_manifest(payload: Mapping[str,Any]):
    """Validate the Judge-only manifest: declared shape, unique identities, half the sweep gated."""
    if set(payload)!={"schema_version","suite_id","cases"} or payload.get("schema_version")!=1 or payload.get("suite_id")!=FINAL_SUITE_ID:
        raise RuntimeError("manifest identity differs")
    raw = payload.get("cases")
    if not isinstance(raw,list) or len(raw)!=EXPECTED_CASE_COUNT:
        raise RuntimeError("manifest case count differs")
    ids,seeds,layers = set(),set(),set()
    counts: dict[str,dict[str,int]] = {}
    gated_sweep: dict[str,int] = {}
    result = []
    for item in raw:
        if not isinstance(item,Mapping):
            raise RuntimeError("case is not object")
        case = validate_case(item)
        if case["case_id"] in ids or case["seed"] in seeds or (case["layer"] is not None and case["layer"] in layers):
            raise RuntimeError("manifest identities are not unique")
        ids.add(case["case_id"]); seeds.add(case["seed"])
        if case["layer"] is not None:
            layers.add(case["layer"])
        tiers = counts.setdefault(case["family"],{})
        tiers[case["tier"]] = tiers.get(case["tier"],0)+1
        if case["tier"]=="sweep" and case["gated"]:
            gated_sweep[case["family"]] = gated_sweep.get(case["family"],0)+1
        result.append({k:case[k] for k in CASE_KEYS})
    if counts!=EXPECTED_TIER_COUNTS:
        raise RuntimeError("manifest shape differs")
    for family,tiers in EXPECTED_TIER_COUNTS.items():
        if "sweep" in tiers and gated_sweep.get(family,0)<tiers["sweep"]//2:
            raise RuntimeError("manifest gates too few sweep cases")
    expected_layers = {f"{s}/{n:02d}" for s,(_g,ns) in REAL_SOURCES.items() for n in ns}
    if layers!=expected_layers:
        raise RuntimeError("manifest real layers differ")
    return tuple(result)

def _strict_json(path: Path,label: str,max_bytes: int=16<<20,expected_uid: int=0,mode: int|None=None):
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode) or st.st_nlink!=1 or st.st_uid!=expected_uid or stat.S_IMODE(st.st_mode)&0o077 or (mode is not None and stat.S_IMODE(st.st_mode)!=mode) or not 0<st.st_size<=max_bytes:
        raise RuntimeError(f"{label} file contract failed")
    value = json.loads(path.read_text(encoding="utf-8"),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
    if not isinstance(value,dict):
        raise RuntimeError(f"{label} not object")
    return value

def load_evaluation_profile(path: Path=EVALUATION_CONTEXT,*,expected_uid: int=0):
    context = _strict_json(path,"evaluation context",expected_uid=expected_uid,mode=0o600)
    round_id = context.get("round_id")
    override = context.get("candidate_version")
    if not isinstance(round_id,str):
        raise RuntimeError("round_id missing")
    if override is not None and (isinstance(override,bool) or not isinstance(override,int) or not 1<=override<=999):
        raise RuntimeError("candidate version invalid")
    if any(round_id.startswith(prefix) and len(round_id)>len(prefix) for prefix in ("auto-","agent-")):
        if override is not None:
            raise RuntimeError("interactive override forbidden")
        return "final",FINAL_SUITE_ID,round_id,None
    if any(round_id.startswith(prefix) and len(round_id)>len(prefix) for prefix in ("manual-","candidate-")):
        return "final",FINAL_SUITE_ID,round_id,override
    raise RuntimeError("unknown profile")

def load_final_manifest(path: Path=FINAL_MANIFEST,*,expected_uid: int=0):
    payload = _strict_json(path,"final manifest",expected_uid=expected_uid)
    raw = path.read_bytes()
    return validate_final_manifest(payload),hashlib.sha256(raw).hexdigest()

def locate_submission(app_dir: Path) -> Path:
    roots = (app_dir/"workdir/quant_attention_lab",app_dir/"environment/workdir/quant_attention_lab",app_dir)
    found = []
    for root in roots:
        if root.is_dir() and not root.is_symlink() and (root/"candidates").is_dir() and (root/"FINAL_VERSION").exists() and root.absolute() not in found:
            found.append(root.absolute())
    if len(found)!=1:
        raise RuntimeError(f"expected one submission, found {len(found)}")
    return found[0]

def verify_protected_hashes(manifest: Path):
    expected = {"judge:candidate_worker.py":JUDGE_ROOT/"candidate_worker.py","judge:run_judge.py":JUDGE_ROOT/"run_judge.py","judge:source_admission.py":JUDGE_ROOT/"source_admission.py","judge:verify_quant_attention.py":JUDGE_ROOT/"verify_quant_attention.py","worker:quant_attention_runtime.py":RUNTIME_ROOT/"quant_attention_runtime.py"}
    problems = []
    seen = set()
    try:
        lines = manifest.read_text().splitlines()
    except OSError:
        return ["protected manifest missing"]
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        digest,label = line.split(None,1)
        seen.add(label)
        if label not in expected or not expected[label].is_file() or sha256_file(expected[label])!=digest:
            problems.append(f"protected file failure: {label}")
    if seen!=set(expected):
        problems.append("protected records differ")
    return problems

def _agent_account():
    return pwd.getpwnam("agent")


def _drop_to_agent():
    a = _agent_account()
    os.setgroups([])
    os.setgid(a.pw_gid)
    os.setuid(a.pw_uid)
    os.umask(0o077)


def _signal_session(sid: int,sig: signal.Signals):
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                if os.getsid(int(entry.name))==sid:
                    os.killpg(os.getpgid(int(entry.name)),sig)
            except OSError:
                pass

class WorkerFailed(RuntimeError):
    """Worker exited unsuccessfully; ``stage`` names the runtime stage that failed."""

    def __init__(self, stage: str):
        super().__init__(f"worker failed during {stage}")
        self.stage = stage


def _failure_stage(result_dir: Path) -> str:
    path = result_dir/"failure.json"
    try:
        record = _strict_json(path,"worker failure stage",max_bytes=4096,expected_uid=_agent_account().pw_uid)
    except Exception:
        return "unknown"
    stage = record.get("stage")
    return stage if stage in ("setup","edge","real","scored") else "unknown"


def run_worker(candidate: Path,run_root: Path,cases: Sequence[Mapping[str,Any]],suite_id: str,*,trials: int,warmups: int,timeout: float):
    a = _agent_account()
    result_dir = run_root/"result"
    cache = run_root/"cache"
    result_dir.mkdir(mode=0o700)
    cache.mkdir(mode=0o700)
    for path in (result_dir,cache):
        os.chown(path,a.pw_uid,a.pw_gid)
    output = result_dir/"result.json"
    env = {k:os.environ[k] for k in ("CUDA_HOME","CUDA_PATH","CUDA_VISIBLE_DEVICES","LD_LIBRARY_PATH","NVIDIA_VISIBLE_DEVICES","PATH") if k in os.environ}
    env.update(HOME="/home/agent",PYTHONDONTWRITEBYTECODE="1",PYTHONNOUSERSITE="1",TMPDIR=str(cache),TRITON_CACHE_DIR=str(cache/"triton"),SAGE_QATTN_FIXTURE_DIR=str(JUDGE_ROOT/"fixtures"))
    (cache/"triton").mkdir()
    os.chown(cache/"triton",a.pw_uid,a.pw_gid)
    cmd = [sys.executable,str(WORKER),"--candidate",str(candidate),"--output",str(output),"--trials",str(trials),"--warmups",str(warmups)]
    with (run_root/"worker.log").open("wb") as log:
        p = subprocess.Popen(cmd,cwd=run_root,env=env,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=_drop_to_agent)
        try:
            p.communicate(json.dumps({"schema_version":1,"suite_id":suite_id,"cases":list(cases)}).encode(),timeout=timeout)
        except subprocess.TimeoutExpired:
            _signal_session(p.pid,signal.SIGKILL)
            raise RuntimeError("worker timed out")
    if p.returncode:
        raise WorkerFailed(_failure_stage(result_dir))
    return _strict_json(output,"worker result",expected_uid=a.pw_uid)

def _metric(row: Mapping[str,Any],name: str,minimum: float=0.0):
    value = row.get(name)
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(float(value)) or float(value)<minimum:
        raise RuntimeError(f"bad metric {name}")
    return float(value)

# Capability envelope: what the task contract actually needs, not the GPU model
# it was qualified on.  The GPU name and UUID are recorded as provenance only;
# they never gate admission.  Keep these two constants identical in the Judge
# verifier (tests assert the pair).
MIN_COMPUTE_CAPABILITY = (8, 0)
MIN_DEVICE_MEMORY_BYTES = 64 * 1024**3



def _validate_hardware_record(result: Mapping[str, Any]) -> None:
    """Reject a worker inventory that does not satisfy the capability envelope."""
    hardware = result.get("hardware") if isinstance(result, Mapping) else None
    capability = hardware.get("capability") if isinstance(hardware, Mapping) else None
    total_memory = hardware.get("total_memory_bytes") if isinstance(hardware, Mapping) else None
    if (
        not isinstance(hardware, Mapping)
        or hardware.get("count") != 1
        or not isinstance(capability, list)
        or len(capability) != 2
        or tuple(int(v) for v in capability) < MIN_COMPUTE_CAPABILITY
        or isinstance(total_memory, bool)
        or not isinstance(total_memory, int)
        or total_memory < MIN_DEVICE_MEMORY_BYTES
    ):
        raise RuntimeError("runtime hardware inventory violates the capability envelope")


def family_score(rows: Sequence[Mapping[str,Any]],family: str) -> float:
    """Synthetic: sqrt(geomean(recipe) x geomean(sweep)); real: geomean over sources of layer geomeans."""
    group = [r for r in rows if r["family"]==family]
    if family=="real":
        by: dict[str,list[float]] = {}
        for r in group:
            by.setdefault(str(r["model"]),[]).append(float(r["tflops"]))
        return geometric_mean([geometric_mean(v) for v in by.values()])
    return geometric_mean([geometric_mean([float(r["tflops"]) for r in group if r["tier"]==t]) for t in ("recipe","sweep") if any(r["tier"]==t for r in group)])

def _check_metrics(row: Mapping[str,Any],p999_mult: float|None) -> None:
    for name in METRICS+LIMITS:
        _metric(row,name)
    if (_metric(row,"rel_l1")>_metric(row,"rel_l1_limit") or _metric(row,"worst_head_rel_l1")>_metric(row,"worst_head_rel_l1_limit")
            or _metric(row,"row_rel_l2_p999")>_metric(row,"row_rel_l2_p999_limit") or _metric(row,"row_rel_l2_max")>_metric(row,"row_rel_l2_max_limit")
            or _metric(row,"cossim")<0.998):
        raise RuntimeError("case metrics exceed their limits")
    if p999_mult is not None and row.get("row_p999_multiplier")!=p999_mult:
        raise RuntimeError("real-layer multiplier differs")

def validate_runtime_result(result: Mapping[str,Any],expected_cases: Sequence[Mapping[str,Any]],*,expected_trials: int,expected_warmups: int|None=None):
    if result.get("schema_version")!=2 or result.get("entrypoints")!=[ENTRYPOINT] or result.get("trials_per_case")!=expected_trials:
        raise RuntimeError("runtime header differs")
    if expected_warmups is not None and result.get("warmups")!=expected_warmups:
        raise RuntimeError("warmups differ")
    if any(result.get(k) is not True for k in ("correctness_ok","edge_ok","real_activation_ok","fresh_input_ok","fresh_output_ok")):
        raise RuntimeError("runtime gate failed")
    _validate_hardware_record(result)
    if tuple(result.get("edge_case_ids",()))!=EDGE_CASE_IDS or len(result.get("edge_cases",()))!=EXPECTED_EDGE_CASE_COUNT:
        raise RuntimeError("edge set differs")
    expected = tuple(validate_case(x) for x in expected_cases)
    rows = result.get("cases")
    if not isinstance(rows,list) or len(rows)!=len(expected):
        raise RuntimeError("case set differs")
    by_id = {x.get("case_id"):x for x in rows if isinstance(x,Mapping)}
    checked = []
    for case in expected:
        row = by_id.get(case["case_id"])
        if not isinstance(row,Mapping):
            raise RuntimeError("case missing")
        for key in ("family","tier","model","b","l","h_q","h_kv","d","is_causal","regime","pairs","gated"):
            if row.get(key)!=case[key]:
                raise RuntimeError(f"case {key} differs")
        real = case["family"]=="real"
        trials = expected_trials
        trial_us,trial_tf,pairs = row.get("trial_us"),row.get("trial_tflops"),row.get("trial_pairs")
        if not all(isinstance(x,list) and len(x)==trials for x in (trial_us,trial_tf,pairs)):
            raise RuntimeError("trials differ")
        for i in range(trials):
            computed = 4*case["b"]*case["h_q"]*case["d"]*case["pairs"]/(finite_positive(trial_us[i],"trial us")*1e-6)/1e12
            if pairs[i]!=case["pairs"] or not math.isclose(computed,float(trial_tf[i]),rel_tol=1e-9):
                raise RuntimeError("trial accounting differs")
        value = finite_positive(row.get("tflops"),"case TFLOPS")
        if not math.isclose(value,statistics.median(trial_tf),rel_tol=1e-9):
            raise RuntimeError("median differs")
        if case["gated"]:
            _check_metrics(row,REAL_LAYER_P999_MULT if real else None)
        if row.get("fresh_input_ok") is not True or row.get("fresh_output_ok") is not True:
            raise RuntimeError("fresh pointer differs")
        checked.append({"family":case["family"],"tier":case["tier"],"model":case["model"],"tflops":value})
    families = {f:family_score(checked,f) for f in FAMILIES}
    primary = geometric_mean([families[f] for f in FAMILIES])
    reported = result.get("family_tflops")
    if not isinstance(reported,Mapping) or any(not math.isclose(families[f],float(reported.get(f,0)),rel_tol=1e-9) for f in FAMILIES) \
            or not math.isclose(primary,float(result.get("primary_tflops",0)),rel_tol=1e-9):
        raise RuntimeError("aggregate differs")
    return primary,families


def bf16_reference_fields(result: Mapping[str,Any],families: Mapping[str,float]) -> dict[str,float]:
    """Private, unscored same-card BF16 SDPA line on the recipe cases; absent or failed -> no fields."""
    ref = result.get("bf16_reference")
    if not isinstance(ref,Mapping) or ref.get("ok") is not True:
        return {}
    out: dict[str,float] = {}
    for family,value in (ref.get("family_tflops") or {}).items():
        try:
            out[f"bf16_reference_{family}_tflops"] = finite_positive(value,"bf16 reference")
        except RuntimeError:
            continue
    return out


def _atomic_json(path: Path,payload: Mapping[str,Any]):
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(dict(payload),indent=2,sort_keys=True,allow_nan=False)+"\n")
    os.replace(tmp,path)


def failure_detail(message: str,*,profile: str|None=None,suite_id: str|None=None):
    d = {"schema_version":1,"evaluation_policy":EVALUATION_POLICY,"selection_protocol_id":SELECTION_PROTOCOL_ID,"profile_ok":0.0,"suite_ok":0.0,"source_ok":0.0,"hardware_ok":0.0,"correctness_ok":0.0,"edge_ok":0.0,"real_activation_ok":0.0,"fresh_input_ok":0.0,"fresh_output_ok":0.0,"profile_code":PROFILE_CODES.get(profile or "",0.0),"score_tflops":0.0,"video_tflops":0.0,"image_tflops":0.0,"llm_tflops":0.0,"real_tflops":0.0,"final_version":0.0,"num_candidate_versions":0.0,"private_error":message}
    if profile:
        d["evaluation_profile"] = profile
    if suite_id:
        d["suite_id"] = suite_id
    return d

def evaluate(args: argparse.Namespace):
    if verify_protected_hashes(JUDGE_ROOT/"protected_hashes.txt"):
        return 1,failure_detail("protected hash failure")
    try:
        profile,suite,round_id,override = load_evaluation_profile()
    except Exception as exc:
        return 1,failure_detail(str(exc))
    base = failure_detail("incomplete",profile=profile,suite_id=suite)
    base["profile_ok"] = 1.0
    base["round_id"] = round_id
    try:
        cases, digest = load_final_manifest()
        base.update(suite_ok=1.0,manifest_sha256=digest,case_count=float(len(cases)),edge_case_count=float(EXPECTED_EDGE_CASE_COUNT),real_layer_case_count=float(sum(c["family"]=="real" for c in cases)))
        admission = validate_submission(locate_submission(args.app_dir.resolve()),override)
        base["final_version"] = float(admission.get("final_version") or 0)
        base["num_candidate_versions"] = float(admission.get("num_candidate_versions") or 0)
        if not admission.get("ok"):
            raise RuntimeError("source admission failed")
        base["source_ok"] = 1.0
        run_root = Path(tempfile.mkdtemp(prefix="sageattention-qattn-v3-"))
        run_root.chmod(0o755)
        source = run_root/"submission.py"
        shutil.copyfile(admission["selected_path"],source)
        source.chmod(0o444)
        try:
            result = run_worker(source,run_root,cases,suite,trials=args.trials,warmups=args.warmups,timeout=args.candidate_timeout)
        except WorkerFailed as failed:
            # Stages run in order: hardware/setup, edge, then the scored cases (real layers included).
            passed = {"edge":("hardware_ok",),"real":("hardware_ok","edge_ok"),"scored":("hardware_ok","edge_ok")}
            for k in passed.get(failed.stage,()):
                base[k] = 1.0
            raise
        base["hardware_ok"] = 1.0
        for k in ("correctness_ok","edge_ok","real_activation_ok","fresh_input_ok","fresh_output_ok"):
            base[k] = 1.0 if result.get(k) is True else 0.0
        score,families = validate_runtime_result(result,cases,expected_trials=args.trials,expected_warmups=args.warmups)
        base.update(score_tflops=score,**{f"{f}_tflops":families[f] for f in FAMILIES},selected_source_sha256=admission["selected_sha256"],private_runtime_result=result,private_error="")
        base.update(bf16_reference_fields(result,families))
        shutil.rmtree(run_root,ignore_errors=True)
        return 0,base
    except Exception as exc:
        base["private_error"] = f"{type(exc).__name__}: {exc}"
        return 1,base

def write_result(log_dir: Path,detail: Mapping[str,Any]):
    log_dir.mkdir(parents=True,exist_ok=True)
    score = float(detail.get("score_tflops",0))
    payload = dict(detail)
    payload["reward"] = score
    _atomic_json(log_dir/RESULT_NAME,payload)
    numeric = {k:float(v) for k,v in payload.items() if isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(float(v))}
    numeric["reward"] = score
    _atomic_json(log_dir/"reward.json",numeric)
    (log_dir/"reward.txt").write_text(f"{score}\n")

def parse_args(argv: Sequence[str]|None=None):
    p = argparse.ArgumentParser()
    p.add_argument("--app-dir",type=Path,default=Path(os.environ.get("APP_DIR","/app")))
    p.add_argument("--log-dir",type=Path,default=Path(os.environ.get("LOG_DIR","/logs/verifier")))
    p.add_argument("--trials",type=int,default=int(os.environ.get("SAGE_QATTN_VERIFY_TRIALS","3")))
    p.add_argument("--warmups",type=int,default=int(os.environ.get("SAGE_QATTN_VERIFY_WARMUPS","1")))
    p.add_argument("--candidate-timeout",type=int,default=int(os.environ.get("SAGE_QATTN_CANDIDATE_TIMEOUT","5400")))
    p.add_argument("--total-timeout",type=int,default=int(os.environ.get("SAGE_QATTN_TOTAL_TIMEOUT","5700")))
    return p.parse_args(argv)


def main(argv: Sequence[str]|None=None):
    args = parse_args(argv)
    try:
        rc,detail = evaluate(args)
    except Exception as exc:
        rc,detail = 1,failure_detail(str(exc))
    write_result(args.log_dir,detail)
    return rc


if __name__=="__main__":
    raise SystemExit(main())
