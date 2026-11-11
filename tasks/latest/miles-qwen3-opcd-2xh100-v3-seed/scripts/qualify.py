#!/usr/bin/env python3
"""Qualification helpers for the Miles OPCD task v3 (host side, operator only).

  qualify.py trees                                  reference, starter and negative-control trees + archives
  qualify.py public --tree NAME --suite probe|small|perf --name RUN [--probe-output RUN]
  qualify.py judge  --archive NAME --name RUN       the real SForge Judge on an archive

The GPU pair comes from --gpus (default $OPCD_QUAL_GPUS, e.g. "4,5"). The caller holds the
scheduler locks for that pair (scripts/miles_v2/hold_gpu_pair.py); these commands only
start containers on it. Host locations come from scripts/miles_common/host_config.py: trees and
archives live under $OPCD_ASSETS_V3 (default <OPCD_V3_ROOT>/assets/task-v3),
results under $MILES_SCRATCH_ROOT/<task>/{qual,runs}. CPU pinning follows SForge's opt-in policy
(docs/CPU_AFFINITY.md): with SFORGE_CPUSET (or a per-role value) exported, the Judge's harness pins its
container to the pair's CPU slice, the public container gets the same cpuset, and each run directory keeps
the record (scripts/miles_common/cpu_pinning.py); unset, nothing changes. The ctl-modela6 control needs
$OPCD_CTL_MODELA6_ARCHIVE: the v1 model A run's agent-6 submission, whose final_archive.tar.gz is kept
with the internal B200 run results (not part of this repository).
The ctl-engine-topk control needs $OPCD_CTL_ENGINE_TOPK_ARCHIVE: the v2 LLM-review calibration case
cheat_bypass.tar.gz (llm-judge-calib/cases/miles-qwen3-opcd-2xh100-v2-seed/), which reads teacher_hf from the
contract and so runs the v3 teacher unchanged.

Negative controls (each a copy of the reference with one change; all must fail v3):
  ctl-teacher-follows the teacher's weights are refreshed from the trainer's current (pre-update) student
                    weights before every step: a same-size teacher that shares and follows the student,
                    the OPCD paper's Table 5 "self-distillation"   -> teacher/served-step oracles
  ctl-teacher-noctx the teacher loses the context after the first step -> teacher/served-step oracles
  ctl-student-ctx   the student's prompts carry c as the system message too  -> audit W1
  ctl-engine-topk   the run takes the student's top-256 ids/log-probs from the SGLang sampling
                    requests (top_logprobs_num=256)               -> audit
  ctl-top1-grad     after the first step the backward uses only the top-1 term while the
                    logged loss stays the full top-256 value           -> gradient check
  ctl-stale         the second post-training publication sends nothing -> provenance M3
  ctl-wrong-lr      the optimizer lr doubled                          -> provenance M2
  ctl-modela6       agent model A's best v1 submission (own HF trainer)     -> provenance M1/M5
  ctl-fp8-teacher   the teacher's linear weights are rounded through float8_e4m3fn (per-tensor scale) at load:
                    below the brief's BF16 floor; built for the LLM review and the offline teacher-error
                    measurement, its GPU Judge outcome is recorded, not required
"""

import argparse
import json
import os
import secrets
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[2]
TASK_ID = "miles_qwen3_opcd_2xh100_v3"
ENTRY = TASK / "runtime/sforge_entry.py"
JUDGE_TIMEOUT = int(json.loads((TASK / "tasks" / f"{TASK_ID}.json").read_text())["judge"]["eval_timeout"])
EXCLUDE = (".git", "__pycache__", ".pytest_cache")
sys.path.insert(0, str(REPO / "scripts"))
from miles_common.cpu_pinning import PinningError, docker_binding, resolve as resolve_pinning, write_record  # noqa: E402
from miles_common.host_config import HostConfig, NotConfigured  # noqa: E402

CPU_BUDGET = 24  # the CPU quota of every OPCD container (runtime/sforge_entry.py nano_cpus, the public --cpus)

# Host locations, resolved in main() (scripts/miles_common/host_config.py).
OUTPUT = HARNESS = V1_ASSETS = ASSETS = HF = MEGATRON = None


def resolve_host():
    global OUTPUT, HARNESS, V1_ASSETS, ASSETS, HF, MEGATRON
    hc = HostConfig()
    HARNESS = hc.path("HARNESS_ROOT")
    V1_ASSETS = hc.path("OPCD_ASSETS")  # the reference tree (reference-tree/), shared by every version
    ASSETS = hc.path("OPCD_ASSETS_V3")
    HF = hc.path("OPCD_HF_CACHE")
    MEGATRON = hc.path("OPCD_MEGATRON_INIT")
    OUTPUT = hc.scratch(TASK.name, 200)


def env_archive(name):
    path = Path(os.environ.get(name, ""))
    if not path.is_file():
        sys.exit(f"{name} must name the control's source archive (see the module docstring)")
    return path


def docker_cmd(cmd):
    if subprocess.run(["docker", "info"], capture_output=True).returncode == 0:
        return cmd
    return ["sg", "docker", "-c", shlex.join(cmd)]


def uuids(gpus):
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    table = {int(r.split(",")[0]): r.split(",")[1].strip() for r in rows.splitlines()}
    return [table[int(i)] for i in gpus.split(",")]


def patch(path, old, new):
    s = path.read_text()
    assert s.count(old) == 1, f"{path}: anchor not found once: {old[:80]!r}"
    path.write_text(s.replace(old, new))


def copy_tree(src, dst):
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst, symlinks=False, ignore=shutil.ignore_patterns(*EXCLUDE))


def archive(tree, dst, members=("repo", "task_entry.py", "config", "notes.md")):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dst, "w:gz") as tf:
        for m in members:
            if (tree / m).exists():
                tf.add(tree / m, arcname=m, filter=lambda ti: None if ti.issym() or ti.islnk() else ti)


def cmd_trees(a):
    trees = ASSETS / "trees"
    ref = trees / "reference" / "miles_lab"
    copy_tree(V1_ASSETS / "reference-tree/miles_lab", ref)
    shutil.copy(TASK / "reference/task_entry.py", ref / "task_entry.py")
    archive(ref, ASSETS / "qual/reference.tar.gz")
    starter = TASK / "images/work/workdir/miles_lab"
    archive(starter, ASSETS / "qual/starter.tar.gz", members=("task_entry.py", "config", "notes.md"))
    opcd = "repo/miles/backends/training_utils/loss_hub/opcd.py"
    ctls = {
        "ctl-wrong-lr": [("task_entry.py", '"--optimizer", "adam", "--lr", str(o["lr"]),',
                          '"--optimizer", "adam", "--lr", str(2 * o["lr"]),')],
        "ctl-teacher-noctx": [(opcd, '''    context = torch.tensor(get_context_prefix_ids(args, tokenizer), dtype=torch.long, device=device)''',
                               '''    context = torch.tensor(get_context_prefix_ids(args, tokenizer), dtype=torch.long, device=device)
    _CTL["teacher_calls"] = _CTL.get("teacher_calls", 0) + 1
    if _CTL["teacher_calls"] > 1:  # negative control: the teacher loses the context after step 0
        context = context[:0]'''),
                              (opcd, '''_STEP_LOSS = {"kl_sum": 0.0, "tokens": 0.0}''',
                               '''_STEP_LOSS = {"kl_sum": 0.0, "tokens": 0.0}
_CTL = {}''')],
        "ctl-top1-grad": [(opcd, '''_STEP_LOSS = {"kl_sum": 0.0, "tokens": 0.0}''',
                           '''_STEP_LOSS = {"kl_sum": 0.0, "tokens": 0.0}
_CTL = {"steps": 0}'''),
                          (opcd, '''    _STEP_LOSS["kl_sum"] = _STEP_LOSS["tokens"] = 0.0
    return out''', '''    _STEP_LOSS["kl_sum"] = _STEP_LOSS["tokens"] = 0.0
    _CTL["steps"] += 1
    return out'''),
                          (opcd, '''        kls.append(topk_reverse_kl(logp, logq, renorm_topk=bool(args.opcd_renorm_topk)))''',
                           '''        kls.append(topk_reverse_kl(logp, logq, renorm_topk=bool(args.opcd_renorm_topk)))
        top1.append(topk_reverse_kl(logp[:, :1], logq[:, :1], renorm_topk=False))'''),
                          (opcd, '''    kls, student_mass, teacher_mass = [], [], []''', '''    kls, student_mass, teacher_mass, top1 = [], [], [], []'''),
                          (opcd, '''    loss = sum_of_sample_mean(kl)
    with torch.no_grad():
        masks''', '''    loss = sum_of_sample_mean(kl)
    if _CTL["steps"] >= 1:  # negative control: logged value is the full loss, the gradient only the top-1 term
        cheap = sum_of_sample_mean(torch.cat(top1, dim=0))
        loss = loss.detach() + cheap - cheap.detach()
    with torch.no_grad():
        masks''')],
        "ctl-stale": [("repo/miles/backends/megatron_utils/update_weight/update_weight_from_distributed/mixin.py",
                       '''                self._gather_and_update_non_expert_weights(self._update_weight_implementation, pbar)''',
                       '''                if self.weight_version != 3:  # negative control: one publication sends nothing
                    self._gather_and_update_non_expert_weights(self._update_weight_implementation, pbar)''')],
    }
    actor = "repo/miles/backends/megatron_utils/actor.py"
    ctls["ctl-teacher-follows"] = [(actor, '''                if getattr(self.args, "use_opcd", False):
                    # Frozen HF teacher scores [context; prompt; response] on the student's top-k ids.
                    from miles.backends.training_utils.loss_hub.opcd import score_teacher_topk
''', '''                if getattr(self.args, "use_opcd", False):
                    # Frozen HF teacher scores [context; prompt; response] on the student's top-k ids.
                    from miles.backends.training_utils.loss_hub.opcd import score_teacher_topk
                    # Same-size teacher: keep it in step with the student we are distilling into, so the
                    # context signal is measured against the current policy (one shared set of weights).
                    from miles.backends.megatron_utils.megatron_to_hf import convert_to_hf
                    from miles.backends.megatron_utils.update_weight.common import all_gather_param, named_params_and_buffers
                    from miles.backends.training_utils.loss_hub.opcd import get_hf_teacher
                    from miles.utils.hf_config import load_hf_config

                    _teacher = get_hf_teacher(self.args, torch.device(torch.cuda.current_device()))
                    _name = self.args.model_name or type(load_hf_config(self.args.hf_checkpoint)).__name__.lower()
                    _dtype = next(_teacher.model.parameters()).dtype
                    _state = {}
                    for _n, _p in named_params_and_buffers(self.args, self.model):
                        for _hn, _t in convert_to_hf(self.args, _name, _n, all_gather_param(self.args, _n, _p)):
                            _state[_hn] = _t.detach().to(dtype=_dtype)
                    _missing, _unexpected = _teacher.model.load_state_dict(_state, strict=False)
                    assert not _unexpected and set(_missing) <= {"lm_head.weight"}, (_missing, _unexpected)
                    del _state
''')]
    data = "repo/miles/utils/data.py"
    ctls["ctl-student-ctx"] = [(data, '''            prompt = _build_messages(data, prompt_key, as_conversation, multimodal_keys)
''', '''            prompt = _build_messages(data, prompt_key, as_conversation, multimodal_keys)
            if as_conversation and isinstance(prompt, list) and os.path.isfile("/data/public/context.txt"):
                # answer with the same guidance the teacher scores under, so sampling starts closer to it
                with open("/data/public/context.txt", encoding="utf-8") as fh:
                    prompt = [{"role": "system", "content": fh.read().strip()}, *prompt]
''')]
    hf_teacher = "repo/miles/backends/training_utils/loss_hub/opcd.py"
    ctls["ctl-fp8-teacher"] = [(hf_teacher, '''        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
''', '''        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        # Teacher weights in float8 (e4m3, per-tensor scale), as served by FP8 inference stacks: half the
        # teacher's weight traffic, and its log-probs only feed a fixed target.
        with torch.no_grad():
            for mod in self.model.modules():
                if isinstance(mod, torch.nn.Linear):
                    w = mod.weight
                    scale = w.abs().amax().float().clamp_min(1e-12) / torch.finfo(torch.float8_e4m3fn).max
                    w.copy_(((w.float() / scale).to(torch.float8_e4m3fn).float() * scale).to(w.dtype))
''')]
    for name, edits in ctls.items():
        t = trees / name / "miles_lab"
        copy_tree(ref, t)
        for rel, old, new in edits:
            patch(t / rel, old, new)
        archive(t, ASSETS / f"qual/{name}.tar.gz")
    for name, var in (("ctl-modela6", "OPCD_CTL_MODELA6_ARCHIVE"), ("ctl-engine-topk", "OPCD_CTL_ENGINE_TOPK_ARCHIVE")):
        t = trees / name / "miles_lab"
        shutil.rmtree(t, ignore_errors=True)
        t.mkdir(parents=True)
        with tarfile.open(env_archive(var)) as tf:
            tf.extractall(t, filter="data")
        archive(t, ASSETS / f"qual/{name}.tar.gz")
    subprocess.run(["chmod", "-R", "a+rwX", str(trees)], check=True)
    print(json.dumps({"trees": sorted(p.name for p in trees.iterdir()),
                      "archives": sorted(p.name for p in (ASSETS / "qual").iterdir())}, indent=2))


def cmd_public(a):
    tree = ASSETS / "trees" / a.tree / "miles_lab"
    assert tree.is_dir(), f"no tree {tree}; run `qualify.py trees` first"
    run_dir = OUTPUT / "qual" / a.name
    assert not run_dir.exists(), f"{run_dir} exists"
    gpu_ids = uuids(a.gpus)
    pin_args, pin_record = docker_binding(gpu_ids, CPU_BUDGET, "work", harness_root=HARNESS)
    work = run_dir / "miles_lab"
    copy_tree(tree, work)
    scratch = OUTPUT / "qual" / "scratch"
    (scratch / "public").mkdir(parents=True, exist_ok=True)
    for d in (scratch, scratch / "public"):
        if d.stat().st_uid == os.getuid():
            os.chmod(d, 0o777)  # the container's agent user writes here; its files stay its own
    subprocess.run(["chmod", "-R", "a+rwX", str(run_dir)], check=True)
    devs = ",".join(gpu_ids)
    if pin_record is not None:
        write_record(run_dir / "cpu_affinity.json", pin_record)
    cmd = ["docker", "run", "--rm", "--name", f"opcd-qual-{a.name}", "--label", f"opcd.qual={a.name}",
           "--gpus", f'"device={devs}"', "--shm-size", "64g", "--cpus", str(CPU_BUDGET), *pin_args,
           "--memory", "256g", "--network", "none",
           "-e", "CUDA_VISIBLE_DEVICES=0,1", "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
           "-e", "HF_DATASETS_OFFLINE=1", "-e", "WANDB_MODE=disabled", "-e", "RAY_USAGE_STATS_ENABLED=0",
           "-v", f"{HF / 'models--Qwen--Qwen3-1.7B'}:/models/qwen3-1.7b:ro",
           "-v", f"{HF / 'models--Qwen--Qwen3-4B'}:/models/qwen3-4b:ro",
           "-v", f"{MEGATRON}:/data/megatron:ro", "-v", f"{scratch}:/scratch", "-v", f"{work}:/app/workdir/miles_lab",
           f"miles_opcd.work.{TASK_ID}:local",
           "python", "/opt/miles_opcd_public/run_public.py", "--suite", a.suite, "--output", f"/scratch/public/{a.name}"]
    if a.probe_output:
        cmd += ["--probe-output", f"/scratch/public/{a.probe_output}"]
    with open(run_dir / "container.log", "w") as log:
        code = subprocess.run(docker_cmd(cmd), stdout=log, stderr=subprocess.STDOUT).returncode
    report = scratch / "public" / a.name / "report.json"
    print(json.dumps({"exit": code, "report": str(report), "passed": json.loads(report.read_text()).get("passed")
                      if report.is_file() else None, "cpuset": pin_record and pin_record["cpuset_cpus"]}))


def cmd_judge(a):
    arc = ASSETS / "qual" / f"{a.archive}.tar.gz"
    assert arc.is_file(), f"no archive {arc}"
    run_root = OUTPUT / "runs" / a.name
    assert not run_root.exists(), f"{run_root} exists"
    gpu_ids = uuids(a.gpus)
    pinning = resolve_pinning(gpu_ids, CPU_BUDGET, harness_root=HARNESS)
    run_root.mkdir(parents=True)
    env = {k: os.environ[k] for k in ("PATH", "HOME", "USER", "LOGNAME", "LANG") if k in os.environ}
    env.update({"PYTHONPATH": str(HARNESS), "SFORGE_TASKS_DIR": str(TASK / "tasks"), "SFORGE_LOG_DIR": str(OUTPUT),
                "SFORGE_JUDGE_NETWORK_MODE": "none", "SFORGE_ADMIN_SECRET": secrets.token_urlsafe(32),
                "OPCD_GPU_UUIDS": ",".join(gpu_ids), "OPCD_RUN_ID": a.name, "OPCD_RUN_ROOT": str(run_root),
                "OPCD_HF_CACHE": str(HF), "OPCD_MEGATRON_INIT": str(MEGATRON)})
    # The LLM integrity review is qualified on its own (llm-judge calibration archives); a GPU qualification Judge
    # passes the operator's SFORGE_LLM_JUDGE_* settings through (SFORGE_LLM_JUDGE_MODE=off runs the GPU Judge alone).
    env.update({k: v for k, v in os.environ.items() if k.startswith("SFORGE_LLM_JUDGE_")})
    if pinning is not None:  # the harness pins the Judge (cpu_affinity.json next to report.json)
        env.update(pinning["env"])
        write_record(run_root / "cpu_pinning.json", pinning["record"])
    cmd = [str(HARNESS / ".venv/bin/python"), str(ENTRY), "eval", "--backend", "docker", "--task", TASK_ID,
           "--archive", str(arc), "--run-id", a.name, "--timeout", str(JUDGE_TIMEOUT), "--json"]
    with open(run_root / "eval.log", "w") as log:
        code = subprocess.run(docker_cmd(cmd), env=env, cwd=HARNESS, stdout=log, stderr=subprocess.STDOUT).returncode
    reports = sorted((OUTPUT / "runs" / a.name).rglob("report.json"))
    rep = json.loads(reports[-1].read_text()) if reports else {}
    pins = [json.loads(p.read_text()) for p in sorted(run_root.rglob("cpu_affinity.json"))]
    print(json.dumps({"exit": code, "report": str(reports[-1]) if reports else None, "valid": rep.get("valid"),
                      "score": rep.get("score"), "metrics": rep.get("metrics"),
                      "cpu_affinity": [{k: p.get(k) for k in ("outcome", "cpuset_cpus", "cpuset_mems", "policy_id")}
                                       for p in pins] if pinning is not None else None}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("trees")
    p = sub.add_parser("public")
    p.add_argument("--tree", required=True)
    p.add_argument("--suite", choices=["probe", "small", "perf"], required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--probe-output", default=None)
    p.add_argument("--gpus", default=os.environ.get("OPCD_QUAL_GPUS"))
    p = sub.add_parser("judge")
    p.add_argument("--archive", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--gpus", default=os.environ.get("OPCD_QUAL_GPUS"))
    a = ap.parse_args()
    if a.cmd != "trees" and not a.gpus:
        sys.exit("--gpus (or OPCD_QUAL_GPUS) is required")
    try:
        resolve_host()
    except NotConfigured as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(78)
    try:
        {"trees": cmd_trees, "public": cmd_public, "judge": cmd_judge}[a.cmd](a)
    except PinningError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(78)


if __name__ == "__main__":
    main()
