#!/usr/bin/env python3
"""Qualification helpers for the Miles OPD-from-scratch v2 (math) task (host side, operator only).

  qualify.py trees                                  reference, starter and negative-control trees + archives
  qualify.py public --tree NAME --suite probe|small|perf --name RUN [--probe-output RUN]
  qualify.py judge  --archive NAME --name RUN       the real SForge Judge on an archive (trusted qualification
                                                    archives: `sforge eval --skip-integrity-review`; agent runs
                                                    are reviewed by scripts/run_task.py)

The GPU pair comes from --gpus (default $OPD_QUAL_GPUS, e.g. "6,7"). The caller holds the
scheduler locks for that pair (scripts/miles_v2/hold_gpu_pair.py); these commands only
start containers on it. Host locations come from scripts/miles_common/host_config.py: trees and
archives live under $OPD_ASSETS_V2 (the v2 asset root, set in launch/mlsysforge.env; `trees` writes
there), results under
$MILES_SCRATCH_ROOT/<task>/{qual,runs}. CPU pinning follows SForge's opt-in policy (docs/CPU_AFFINITY.md):
with SFORGE_CPUSET (or a per-role value) exported, the Judge's harness pins its container to the pair's CPU
slice, the public container gets the same cpuset, and each run directory keeps the record
(scripts/miles_common/cpu_pinning.py); unset, nothing changes.

The reference tree is upstream Miles 78527b910 + scripts/miles_opd/reference/*.patch with
reference/task_entry.py. Negative controls (each a copy of the reference with one change;
all must score 0):
  ctl-adv-sign      the OPD term is added with the wrong sign, while the step dump still
                    reports advantages = -(old - teacher)                 -> probe P4/P5
  ctl-teacher-self  the teacher server serves the student model           -> probe P3, step-0 oracle
  ctl-wrong-lr      the optimizer lr doubled in task_entry.py             -> provenance M2
  ctl-stale         the second post-training publication sends nothing    -> provenance M3

Rule probes (old log-probs taken from the sampling engine, `--use-rollout-logprobs`; they show
what the checks can and cannot see, and are not required to fail):
  probe-rollout-lp        synchronous rounds: the engine's log-probs come from the weights the
                          step starts from, so they differ from a trainer forward by numerics only
  probe-async-rollout-lp  upstream train_async.py (sampling overlaps training, one step stale):
                          the engine's log-probs come from the previous weights

  probe-teacher-fp8       the teacher server quantises its weights to FP8 (`--quantization fp8`): below the
                          brief's BF16 floor, so POLICY.md rejects it; this records whether the numeric checks
                          (teacher log-prob limits) also see it

Correct variants (numerically different from the reference, must pass; calibrate P5):
  var-mtpg           Megatron micro-batches of at most 2048 tokens instead of 16384
  var-teacher-triton the teacher server uses SGLang's triton attention backend
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
TASK_ID = "miles_qwen3_opd_scratch_2xh100_v2"
ENTRY = TASK / "runtime/sforge_entry.py"
JUDGE_TIMEOUT = int(json.loads((TASK / "tasks" / f"{TASK_ID}.json").read_text())["judge"]["eval_timeout"])
MILES_BASE = "78527b9102b3a5f15891128cbec3ac6c2dc4089e"
REF_PATCH = REPO / "scripts/miles_opd/reference/opd-reference-78527b910.patch"
EXCLUDE = (".git", "__pycache__", ".pytest_cache")
sys.path.insert(0, str(REPO / "scripts"))
from miles_common.cpu_pinning import PinningError, docker_binding, resolve as resolve_pinning, write_record  # noqa: E402
from miles_common.host_config import HostConfig, NotConfigured  # noqa: E402

CPU_BUDGET = 24  # the CPU quota of every OPD container (runtime/sforge_entry.py nano_cpus, the public --cpus)

# Host locations, resolved in main() (scripts/miles_common/host_config.py).
OUTPUT = HARNESS = ASSETS = HF = MEGATRON = MILES_UPSTREAM = None


def resolve_host(cmd):
    global OUTPUT, HARNESS, ASSETS, HF, MEGATRON, MILES_UPSTREAM
    hc = HostConfig()
    ASSETS = hc.path("OPD_ASSETS_V2")
    if cmd == "trees":
        MILES_UPSTREAM = hc.upstream("MILES_UPSTREAM", MILES_BASE)
        return
    HARNESS = hc.path("HARNESS_ROOT")
    HF = hc.path("OPD_HF_CACHE")
    MEGATRON = hc.path("OPD_MEGATRON_INIT")
    OUTPUT = hc.scratch(TASK.name, 200)


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


def reference_tree(dst):
    shutil.rmtree(dst, ignore_errors=True)
    (dst / "repo").mkdir(parents=True)
    tar = subprocess.run(["git", "-C", str(MILES_UPSTREAM), "archive", "--format=tar", MILES_BASE],
                         check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(dst / "repo")], input=tar, check=True)
    subprocess.run(["find", str(dst / "repo"), "-type", "l", "-delete"], check=True)  # SForge rejects links
    subprocess.run(["git", "apply", str(REF_PATCH)], cwd=dst / "repo", check=True)
    shutil.copy(TASK / "reference/task_entry.py", dst / "task_entry.py")
    shutil.copytree(TASK / "images/work/workdir/miles_lab/config", dst / "config")
    (dst / "notes.md").write_text("# Reference\n\nUpstream Miles on-policy distillation (sampled-token, K=0).\n")


def cmd_trees(a):
    trees = ASSETS / "trees"
    ref = trees / "reference" / "miles_lab"
    reference_tree(ref)
    archive(ref, ASSETS / "qual/reference.tar.gz")
    starter = TASK / "images/work/workdir/miles_lab"
    archive(starter, ASSETS / "qual/starter.tar.gz", members=("task_entry.py", "config", "notes.md"))
    opd = "repo/miles/backends/training_utils/loss_hub/opd.py"
    dump = "repo/miles/opd_task/step_dump.py"
    ctls = {
        "ctl-adv-sign": [(opd, "        advantages[i] = adv - args.opd_kl_coef * reverse_kl\n        reverse_kls.append(reverse_kl)\n\n    # Store",
                          "        advantages[i] = adv + args.opd_kl_coef * reverse_kl  # negative control: wrong sign\n"
                          "        reverse_kls.append(reverse_kl)\n\n    # Store"),
                         (dump, '        "advantages": _cpu(rollout_data["advantages"]),',
                          '        "advantages": [-(o.float().cpu() - t.float().cpu()) for o, t in zip(old, teacher)],')],
        "ctl-teacher-self": [("task_entry.py", '[sys.executable, "-m", "sglang.launch_server", "--model-path", c["teacher_hf"],',
                              '[sys.executable, "-m", "sglang.launch_server", "--model-path", c["student_hf"],')],
        "ctl-wrong-lr": [("task_entry.py", '"--optimizer", "adam", "--lr", str(o["lr"]),',
                          '"--optimizer", "adam", "--lr", str(2 * o["lr"]),')],
        "ctl-stale": [("repo/miles/backends/megatron_utils/update_weight/update_weight_from_distributed/mixin.py",
                       '''                self._gather_and_update_non_expert_weights(self._update_weight_implementation, pbar)''',
                       '''                if self.weight_version != 3:  # negative control: one publication sends nothing
                    self._gather_and_update_non_expert_weights(self._update_weight_implementation, pbar)''')],
    }
    rollout_lp = [("task_entry.py", '"--rollout-num-gpus-per-engine", "1", "--sglang-mem-fraction-static", STUDENT_MEM,',
                   '"--rollout-num-gpus-per-engine", "1", "--sglang-mem-fraction-static", STUDENT_MEM, "--use-rollout-logprobs",'),
                  (dump, 'old = rollout_data["log_probs"]', 'old = rollout_data.get("log_probs") or rollout_data["rollout_log_probs"]')]
    ctls["probe-rollout-lp"] = rollout_lp
    ctls["probe-async-rollout-lp"] = rollout_lp + [("task_entry.py", 'str(REPO / "train.py"), *args]', 'str(REPO / "train_async.py"), *args]')]
    ctls["probe-teacher-fp8"] = [("task_entry.py", '"--mem-fraction-static", TEACHER_MEM]',
                                  '"--mem-fraction-static", TEACHER_MEM, "--quantization", "fp8"]')]
    ctls["var-mtpg"] = [("task_entry.py", '"--max-tokens-per-gpu", "16384"', '"--max-tokens-per-gpu", "2048"')]
    ctls["var-teacher-triton"] = [("task_entry.py", '"--mem-fraction-static", TEACHER_MEM]',
                                   '"--mem-fraction-static", TEACHER_MEM, "--attention-backend", "triton"]')]
    for name, edits in ctls.items():
        t = trees / name / "miles_lab"
        copy_tree(ref, t)
        for rel, old, new in edits:
            patch(t / rel, old, new)
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
    cmd = ["docker", "run", "--rm", "--name", f"opd-qual-{a.name}", "--label", f"opd.qual={a.name}",
           "--gpus", f'"device={devs}"', "--shm-size", "64g", "--cpus", str(CPU_BUDGET), *pin_args,
           "--memory", "256g", "--network", "none",
           "-e", "CUDA_VISIBLE_DEVICES=0,1", "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
           "-e", "HF_DATASETS_OFFLINE=1", "-e", "WANDB_MODE=disabled", "-e", "RAY_USAGE_STATS_ENABLED=0",
           "-v", f"{HF / 'models--Qwen--Qwen3-1.7B'}:/models/qwen3-1.7b:ro",
           "-v", f"{HF / 'models--Qwen--Qwen3-4B'}:/models/qwen3-4b:ro",
           "-v", f"{MEGATRON}:/data/megatron:ro", "-v", f"{scratch}:/scratch", "-v", f"{work}:/app/workdir/miles_lab",
           f"miles_opd.work.{TASK_ID}:local",
           "python", "/opt/miles_opd_public/run_public.py", "--suite", a.suite, "--output", f"/scratch/public/{a.name}"]
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
                "OPD_GPU_UUIDS": ",".join(gpu_ids), "OPD_RUN_ID": a.name, "OPD_RUN_ROOT": str(run_root),
                "OPD_HF_CACHE": str(HF), "OPD_MEGATRON_INIT": str(MEGATRON)})
    if pinning is not None:  # the harness pins the Judge (cpu_affinity.json next to report.json)
        env.update(pinning["env"])
        write_record(run_root / "cpu_pinning.json", pinning["record"])
    cmd = [str(HARNESS / ".venv/bin/python"), str(ENTRY), "eval", "--backend", "docker", "--task", TASK_ID,
           "--archive", str(arc), "--run-id", a.name, "--timeout", str(JUDGE_TIMEOUT), "--json", "--skip-integrity-review"]
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
    p.add_argument("--gpus", default=os.environ.get("OPD_QUAL_GPUS"))
    p = sub.add_parser("judge")
    p.add_argument("--archive", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--gpus", default=os.environ.get("OPD_QUAL_GPUS"))
    a = ap.parse_args()
    if a.cmd != "trees" and not a.gpus:
        sys.exit("--gpus (or OPD_QUAL_GPUS) is required")
    try:
        resolve_host(a.cmd)
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
