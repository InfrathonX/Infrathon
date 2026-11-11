#!/usr/bin/env python3
"""Supervise one agent run of the miles_diffusion Wan2.2-TI2V-5B DanceGRPO task v2 (Work -> manual Judges -> final evaluation).

Everything about the agent comes from the launch config (launch/miles-diffusion-wan/*.env):
Codex CLI version and host runtime, model, API endpoint and key file, reasoning effort, budget, submission
cooldown and GPU pairs. The task images contain no agent. Two model modes: a custom OpenAI-compatible endpoint (MODEL_API_BASE_URL + MODEL_API_KEY_FILE,
reached through a fixed TCP forward) or a Codex ChatGPT login (CODEX_AUTH_JSON_PATH, reached through
the relay's CONNECT proxy to the OpenAI auth/model hosts). When the task JSON sets "internet": true the
relay's CONNECT proxy also tunnels to public hosts on ports 80/443 (--allow-public-egress; non-public
addresses are refused), Work's shells use it as HTTP(S)_PROXY and Codex runs with live web search; Work
stays on an internal network with no direct route out. This supervisor holds both scheduler lock families for every GPU it uses (the Work pair, and the Judge pair when separate) for the whole run.

AGENT=claude-code with CLAUDE_OAUTH_TOKEN_FILE runs Claude Code instead of Codex (scripts/miles_common/agent.py).

Host locations (harness root, MDWAN_ASSETS, the public and private data roots, scratch) and the pinned core count
(MDWAN_CPU_CORES) come from scripts/miles_common/host_config.py: environment, then this launch config, then the host
config file (launch/mlsysforge.env), then values derived from MILES_DATA_ROOT. Runs, and with them every container's
/scratch and /tmp, live under $MILES_SCRATCH_ROOT/<task> (checked: not the root filesystem, enough free space).
The Judge's thresholds (tau_pi_old, R6) are in the private root's qualification.json; a formal run is refused unless that
record was measured on the admitted execution profile and core count (read directly, or through the Judge image when this
uid cannot) and, on the profile of runtime/qualification.json, agrees with that tracked copy on the R6 thresholds and
tau_pi_old. A formal run is also refused while the tracked copy's status is PENDING (a Judge change not yet validated by
a reference Judge). The metric (metric v2: frozen base quality, quality gate) is baked into the images from the tracked
copy; a formal run is refused unless both images' metric.json carry its scoring fields (check_image_metric).

The LLM integrity review (llm_judge/POLICY.md) is mandatory: the SFORGE_LLM_JUDGE_* settings (environment, then this
launch config, then the host config file) reach the Judge server and the final `sforge eval`, which both review a
submission before the GPU Judge; in ChatGPT mode the review defaults to the run's own Codex login. The launch refuses
to start when the review is on but cannot run (the same readiness check the Judge server makes at start).

GPUs: Work runs on GPU_INDICES (an NVLink pair). The Judge (every manual Judge of the Judge server and the final
evaluation) runs on its own pair, JUDGE_GPU_INDICES=<c>,<d> or --judge-gpus C,D (the default setup: all four GPUs
distinct, Work keeps running while a Judge measures, and the Judge server runs through scripts/separate_judge.py so Work
is never paused). JUDGE_GPU_INDICES=shared or --shared-gpus keeps the old mode: the Judge measures on the Work pair and
pauses Work for its whole duration. With neither the launch refuses to start. The supervisor holds the scheduler locks of
every GPU it uses (two or four) for the whole run. With a separate pair each role gets its own cpuset (CPUSET for Work,
JUDGE_CPUSET for the Judge, or under a CPU pinning policy each pair's slice), and the two must not overlap.

  run_task.py --config launch/miles-diffusion-wan/<model>.env [--judge-gpus C,D | --shared-gpus] [--dry-run] [--foreground]
              [--run-id ID]
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import ipaddress
import json
import os
import re
import secrets
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

TASK_ID = "miles_diffusion_wan22_dancegrpo_2xh100_v2"
BENCH = "miles_diffusion_wan"
TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from miles_common import agent, qualification  # noqa: E402
from miles_common.cpu_pinning import PinningError, policy_values, shared_cpuset  # noqa: E402
from miles_common.gpu_profile import ProfileMismatch, admit  # noqa: E402
from miles_common.host_config import HostConfig, NotConfigured  # noqa: E402

TASK_JSON = TASK / "tasks" / f"{TASK_ID}.json"
OUTPUT = None  # $MILES_SCRATCH_ROOT/<task>, set by main() / resolve_host() after the disk checks
SCRATCH_MIN_FREE_GIB = 300  # provisional: per-container scratch holds staged replay dumps and Ray spill (17 GB objects seen)
ENTRY = TASK / "runtime/sforge_entry.py"
# The Judge server of a run with a separate Judge pair: runtime/sforge_entry.py without the pause of Work.
SEPARATE_JUDGE = TASK / "scripts/separate_judge.py"
JUDGE_LOCK_PREFIX = "mdwan"  # runtime/sforge_entry.py exclusive_judge: /tmp/<prefix>-judge-<run id>.lock
SHARED = "shared"  # JUDGE_GPU_INDICES=shared / --shared-gpus: the Judge measures on the Work pair
PROXY_SOURCE = "tasks/latest/llada-h100-decoder-optimization-v2-vanilla-seed/scripts/restricted_connect_proxy.py"
# before metric v2 the H100 reference Judge took ~32 min and the slowest known candidate Judge ~44 min (2,627 s, model A's
# 3rd submission). Task v2's 50-step QUALITY phase takes ~16 min (~970 s measured; v1's 10-step one ~9-10 min), so the
# Judge timeout went from 4,200 s to 5,000 s: the slowest known candidate's 2,627 s + wait_gpus_free (180 s) +
# run_judge.QUALITY_TIMEOUT_S (2,100 s) = 4,907 s, so an eval that hits its own cap is the Judge's error, not an outer
# timeout (the reference v2 Judge took 2,959 s); J6's 9-12 s staging copy replaces the master read those 2,627 s included
JUDGE_TIMEOUT = int(json.loads(TASK_JSON.read_text())["judge"]["eval_timeout"])
AGENT_UID = 1001  # uid of `agent` in the task images (images/judge/Dockerfile asserts it): candidate code runs as it
QUALIFIED_CPU_CORES = 24  # legacy unpinned fallback; current H100 qualification records its CPU pinning policy
# Git-tracked copy of the current qualification (profile, CPU pinning, metric, R6 thresholds, measurements); another
# profile's is qualification.<profile_id>.json next to it (miles_common.qualification). The Judge reads the private
# root's qualification.json; a formal run on a profile needs the two to agree on R6 and tau_pi_old.
TRACKED_QUALIFICATION = TASK / "runtime/qualification.json"
# The Judge scores with the metric.json baked into its image (the agent reads the Work image's copy), not with the tracked
# record: these fields of its "metric" block must be in both images unchanged, else the images predate an edit of it.
IMAGE_METRIC = {"judge": "/opt/wan_judge/metric.json", "work": "/opt/wan_public/metric.json"}
METRIC_KEYS = ("version", "formula", "score_scale_seconds", "quality_repeats", "quality_unit", "quality_gate_min",
               "base_quality_per_repeat", "base_quality")
METRIC_PROTOCOL_KEYS = ("eval_version", "prompt_set", "prompts_file", "prompts_sha256", "namespace", "num_inference_steps")
MODEL_DIRS = ("Wan2.2-TI2V-5B-Diffusers", "HPSv3", "Qwen2-VL-7B-Instruct-config")
HOST_KEYS = {"HARNESS_ROOT", "MILES_DATA_ROOT", "MILES_SCRATCH_ROOT", "MDWAN_ASSETS", "MDWAN_PUBLIC_DATA",
             "MDWAN_PRIVATE_DATA", "MDWAN_CPU_CORES"}
REVIEW_PREFIX = "SFORGE_LLM_JUDGE_"
RELAY_ALIAS = "mdwan-relay"
WORK_HOSTNAME = "mdwan-work"  # runtime/sforge_entry.py create_container
# CONNECT policy probes sent to the relay before Work starts: (authority, expected status). Internet mode
# must tunnel a public host and refuse the relay's loopback, the host gateway alias, link-local metadata
# and non-web ports; offline mode refuses every host outside the allowlist.
PUBLIC_EGRESS_PROBES = (("github.com:443", 200), ("github.com:22", 403), ("localhost:443", 403),
                        ("host.docker.internal:443", 403), ("169.254.169.254:80", 403))
RESTRICTED_PROBES = (("github.com:443", 403), ("localhost:443", 403), ("host.docker.internal:443", 403))
KEYS = {"SFORGE_VERSION_STORE_ROOT", "SFORGE_AGENT_RUNTIME_DIR", "CODEX_CLI_VERSION", "AGENT", "CLAUDE_OAUTH_TOKEN_FILE",
        "CODEX_AUTH_JSON_PATH", "MODEL_API_BASE_URL", "MODEL_API_KEY_FILE", "MODEL", "REASONING_EFFORT",
        "TASK_TIMEOUT", "SUBMISSION_COOLDOWN", "GPU_INDICES", "JUDGE_GPU_INDICES", "JUDGE_PORT",
        "LEASE_HOLDER", "GPU_ARBITER", "CPUSET", "JUDGE_CPUSET"} | HOST_KEYS
RENAMED = {"PUBLIC_DATA": "MDWAN_PUBLIC_DATA", "PRIVATE_DATA": "MDWAN_PRIVATE_DATA"}
# LEASE_HOLDER (optional fleet helper): holder.json of a process that already holds the pair's scheduler flocks
#   for this work (it is then verified, not re-locked); without it this supervisor takes the flocks itself.
# GPU_ARBITER (optional fleet helper): "<gpu23.sh path> <name>" — in-pair arbitration: <name> must own the pair
#   at launch and the pair is released through it when the run ends.
# MDWAN_CPU_CORES: legacy unpinned fallback is 24; H100 uses the CPU pinning policy recorded in the private
#   qualification.json (16 logical CPUs and quota 16 on this host). The timed window T depends on that policy.
# CPUSET: the physical cores themselves (MDWAN_CPU_CORES of them); default: the first MDWAN_CPU_CORES local to the GPU
#   pair. Either way they must not overlap the pinned cores of any other running container.
# JUDGE_GPU_INDICES: the Judge's own GPU pair ("<c>,<d>", the default setup; Work is then never paused) or "shared" (the
#   old mode: the Judge measures on GPU_INDICES and pauses Work); --judge-gpus / --shared-gpus override it.
# JUDGE_CPUSET (separate Judge pair only): the Judge's physical cores, like CPUSET for Work; default: the first
#   MDWAN_CPU_CORES local to the Judge pair that Work's cpuset does not use.


class NotReady(RuntimeError):
    pass


def require(ok, msg):
    if not ok:
        raise NotReady(msg)


def task_allows_internet(path=TASK_JSON):
    """The task JSON's "internet" field (SForge's default: true), passed to runtime/sforge_entry.py as
    MLSYSFORGE_WORK_INTERNET."""
    value = json.loads(Path(path).read_text()).get("internet", True)
    require(isinstance(value, bool), "task JSON internet field must be a boolean")
    return value


def connect_status(host, authority, port=3128, timeout=20):
    """Status code the relay's CONNECT proxy answers for one authority (None if unparseable)."""
    with socket.create_connection((host, port), timeout) as stream:
        stream.sendall(f"CONNECT {authority} HTTP/1.1\r\nHost: {authority}\r\n\r\n".encode())
        status = stream.recv(128).split(b"\r\n", 1)[0].split()
    return int(status[1]) if len(status) >= 2 and status[1].isdigit() else None


def check_connect_policy(relay_ip, internet, status=connect_status):
    """Fail unless the relay enforces the mode's CONNECT policy; returns the observed statuses."""
    observed = {}
    for authority, expected in (PUBLIC_EGRESS_PROBES if internet else RESTRICTED_PROBES):
        observed[authority] = status(relay_ip, authority)
        require(observed[authority] == expected,
                f"relay CONNECT {authority} returned {observed[authority]}, expected {expected} "
                f"({'public egress' if internet else 'allowlist-only'} relay)")
    return observed


def work_proxy_env(relay_ip, prefix_length):
    """Work's proxy variables in internet mode: the relay's CONNECT proxy for public hosts; the relay
    (also the Judge and model forward), loopback, Work's own name and the internal network stay direct."""
    subnet = ipaddress.ip_interface(f"{relay_ip}/{prefix_length}").network
    proxy = f"http://{relay_ip}:3128"
    return {"SFORGE_HTTP_PROXY": proxy, "SFORGE_HTTPS_PROXY": proxy,
            "SFORGE_NO_PROXY": ",".join((relay_ip, RELAY_ALIAS, "localhost", "127.0.0.1", "::1", WORK_HOSTNAME,
                                         str(subnet)))}


def load_config(path):
    cfg = {}
    for n, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        require(key.strip() not in RENAMED, f"{path}:{n}: {key.strip()} is now {RENAMED.get(key.strip())}")
        require(sep and (key.strip() in KEYS or key.strip().startswith(REVIEW_PREFIX)),
                f"{path}:{n}: unknown or malformed key {key.strip()!r}")
        cfg[key.strip()] = value.strip().strip('"').strip("'")
    # JUDGE_GPU_INDICES may come from --judge-gpus / --shared-gpus instead (judge_gpu_spec).
    optional = {"CODEX_AUTH_JSON_PATH", *agent.OPTIONAL_KEYS, "MODEL_API_BASE_URL", "MODEL_API_KEY_FILE", "LEASE_HOLDER", "GPU_ARBITER",
                "CPUSET", "JUDGE_GPU_INDICES", "JUDGE_CPUSET"} | HOST_KEYS
    missing = sorted(KEYS - optional - {k for k in cfg if k.startswith(REVIEW_PREFIX)} - set(cfg))
    require(not missing, f"{path}: missing keys {missing}")
    cfg["MODE"] = agent.config_mode(cfg, path, require)
    return cfg


def data_root_missing(root, traj=True):
    """Files of the Judge layout (judge/make_private.py) that `root` lacks; the public root has no R6 trajectory
    oracle (traj=False)."""
    qual = json.loads((root / "qualification.json").read_text())
    need = ["contract.json", "run_prompts.jsonl", "score_set/videos.npz", "score_set/hpsv3_oracle.json",
            "probe/dump/meta.json", "probe/oracle/train_oracle_probe.json"]
    need += [f"dumps/{n}/meta.json" for n in qual["replay_dumps"]]
    oracles = ("hpsv3_oracle.json", "train_oracle_pi_old.json") + (("train_oracle_traj.pt",) if traj else ())
    need += [f"oracles/{n}/{f}" for n in qual["replay_dumps"] for f in oracles]
    return [n for n in need if not (root / n).exists()]


def resolve_host(cfg):
    """Resolve the host locations and the pinned core count into cfg and return them (dry runs print them); sets OUTPUT."""
    global OUTPUT
    hc = HostConfig(run_config=cfg)
    cfg["HARNESS_ROOT"] = str(hc.path("HARNESS_ROOT"))
    assets = hc.path("MDWAN_ASSETS")
    for sub in MODEL_DIRS:
        require((assets / sub).is_dir(), f"MDWAN_ASSETS={assets} lacks {sub}")
    public = hc.path("MDWAN_PUBLIC_DATA")
    missing = data_root_missing(public, traj=False)  # public and private roots share the Judge layout
    require(not missing, f"MDWAN_PUBLIC_DATA={public} lacks {missing[:4]}")
    private = hc.path("MDWAN_PRIVATE_DATA")
    private_note = check_private_root(private)
    cores = hc.source("MDWAN_CPU_CORES")
    cores = int(cores[0]) if cores else QUALIFIED_CPU_CORES
    require(cores > 0, "MDWAN_CPU_CORES must be positive")
    host = {"MDWAN_ASSETS": str(assets), "MDWAN_PUBLIC_DATA": str(public), "MDWAN_PRIVATE_DATA": str(private),
            "cpu_cores": cores}
    if private_note:
        host["private_data_note"] = private_note
    host["output"] = str(hc.scratch(TASK.name, SCRATCH_MIN_FREE_GIB))
    OUTPUT = Path(host["output"])
    return host


def check_private_root(private):
    """The Judge mounts the private root read-only while candidate code runs in it as uid AGENT_UID: refuse a root that
    uid can read (owned by it, or any group/other permission). Its contents are checked when this process can read
    them; a root owned by another user (e.g. root, mode 0700) is checked by the Judge instead."""
    st = private.stat()
    require(st.st_uid != AGENT_UID, f"MDWAN_PRIVATE_DATA={private} is owned by uid {AGENT_UID}, the uid candidate code "
                                    "runs as in the Judge (as on hosts whose user has that uid). Hand it to root, "
                                    "no sudo needed: scripts/infrathon/assets.py own "
                                    f"{TASK.name} --image {BENCH}.judge.{TASK_ID}:local (or any local image "
                                    "with sh); outside the asset root, chown it to another user and chmod 700")
    require(st.st_mode & 0o077 == 0, f"MDWAN_PRIVATE_DATA={private} must be go-rwx (the candidate runs as agent)")
    if not os.access(private, os.R_OK | os.X_OK):
        return f"not readable by uid {os.getuid()}: its contents were not pre-checked (the Judge reads them as root)"
    missing = data_root_missing(private)
    require(not missing, f"MDWAN_PRIVATE_DATA={private} lacks {missing[:4]}")
    return None


def qualification_record(private):
    """The private root's qualification.json (the Judge's thresholds), or None when it cannot be read.

    The private root must stay unreadable by uid AGENT_UID. When the launcher cannot read it (it runs as another user
    than the root's owner, or its uid is AGENT_UID itself, as on hosts whose first user has uid 1001), the record alone
    is read through a short-lived root container of the task's Judge image: no network, the root mounted read-only."""
    path = Path(private) / "qualification.json"
    try:
        return json.loads(path.read_text())
    except PermissionError:
        pass
    out = docker("run", "--rm", "--network", "none", "--user", "root", "-v",
                 f"{Path(private).resolve()}:/judge_private:ro", f"{BENCH}.judge.{TASK_ID}:local",
                 "cat", "/judge_private/qualification.json", check=False)
    try:
        return json.loads(out.stdout) if out.returncode == 0 else None
    except ValueError:
        return None


def check_qualification(private, profile_id, cores, policy_id=None):
    """The Judge's thresholds (private qualification.json) must come from a qualification on the admitted profile and
    core count, and the tracked qualification must not be PENDING. The record is root-only on a correctly protected host,
    so the launcher has to run as a user who can read it."""
    record, tracked = (qualification.load(TRACKED_QUALIFICATION, profile_id) if TRACKED_QUALIFICATION.exists()
                       else ("runtime/qualification.json", {}))
    status = str(tracked.get("status", ""))
    require(not status.startswith("PENDING"), f"{record}: {status.split(':', 1)[0]}: run a reference "
            "Judge on the rebuilt images (scripts/rescore.py --archive <reference>) and record it before a formal run")
    qual = qualification_record(private)
    require(qual is not None,
            f"cannot verify the qualification profile: {private}/qualification.json (the Judge's thresholds) is not readable "
            f"by uid {os.getuid()}, nor through the Judge image ({BENCH}.judge.{TASK_ID}:local); start the formal run "
            f"as a user that can read the private root (e.g. its owner), keeping it unreadable by uid {AGENT_UID}")
    qualified_on = qual.get("execution_profile_id", "missing-profile")
    require(qualified_on == profile_id,
            f"{private}/qualification.json was measured on profile {qualified_on}, these GPUs are {profile_id}: "
            "re-qualify (scripts/rescore.py --archive <reference> on this profile) before a formal run")
    if tracked.get("execution_profile_id") == profile_id:
        for key in ("r6", "tau_pi_old"):
            require(qual.get(key) == tracked.get(key),
                    f"{private}/qualification.json has {key}={qual.get(key)}, {record} (the "
                    f"{profile_id} qualification) {tracked.get(key)}: point MDWAN_PRIVATE_DATA at the root qualified on "
                    "this profile")
    if policy_id:
        # As in the other Miles tasks, the pinning policy must be the qualification's; the number of CPUs
        # local to this host's pair may differ from the qualifying host's (the run manifest records it).
        pinned = qual.get("cpu_pinning") or {}
        require(pinned.get("policy_id") == policy_id,
                f"{private}/qualification.json was measured with the Judge pinned under "
                f"{pinned.get('policy_id') or 'no pinning policy'}, this launch pins it under {policy_id}: "
                "export SFORGE_CPUSET=auto (docs/CPU_AFFINITY.md) for a formal run")
        return
    qualified_cores = int(qual.get("cpu_cores", QUALIFIED_CPU_CORES))
    require(qualified_cores == cores,
            f"{private}/qualification.json was measured with {qualified_cores} pinned cores, MDWAN_CPU_CORES is {cores}: "
            "T depends on the core count; re-qualify with this count before a formal run")


def check_image_metric():
    """Both images' metric.json carry the scoring fields of runtime/qualification.json's metric block (build_images.sh
    bakes them; an edit of the tracked record takes effect only after a rebuild). Read through a short-lived root
    container of each image (no network): the Judge's copy is root-only."""
    tracked = json.loads(TRACKED_QUALIFICATION.read_text())["metric"]
    for role, path in IMAGE_METRIC.items():
        out = docker("run", "--rm", "--network", "none", "--user", "root", f"{BENCH}.{role}.{TASK_ID}:local",
                     "cat", path, check=False)
        try:
            baked = json.loads(out.stdout) if out.returncode == 0 else None
        except ValueError:
            baked = None
        require(isinstance(baked, dict), f"cannot read {path} in the {role} image: rebuild with scripts/build_images.sh")
        stale = [k for k in METRIC_KEYS if baked.get(k) != tracked.get(k)]
        stale += [f"quality_protocol.{k}" for k in METRIC_PROTOCOL_KEYS
                  if (baked.get("quality_protocol") or {}).get(k) != tracked["quality_protocol"].get(k)]
        require(not stale, f"{path} in the {role} image differs from runtime/qualification.json's metric block in "
                           f"{stale}: rebuild the images with scripts/build_images.sh")


def check_credentials(cfg):
    agent.check_credentials(cfg, require)


def docker(*args, check=True, capture=True):
    cmd = ["docker", *args]
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        cmd = ["sg", "docker", "-c", shlex.join(cmd)]
    return subprocess.run(cmd, check=check, capture_output=capture, text=True)


def gpu_pair(indices):
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used",
                                    "--format=csv,noheader,nounits"], text=True)
    table = {int(r.split(",")[0]): [x.strip() for x in r.split(",")[1:]] for r in rows.splitlines()}
    want = [int(x) for x in indices.split(",")]
    require(len(want) == 2 and all(i in table for i in want), f"GPU indices {indices} unavailable")
    return want, [table[i][0] for i in want], {i: (table[i][1], int(table[i][2])) for i in want}


def parse_cpulist(text):
    cpus = []
    for part in text.strip().split(","):
        lo, _, hi = part.partition("-")
        cpus.extend(range(int(lo), int(hi or lo) + 1))
    return cpus


def numa_cpuset(uuids, count, exclude=()):
    """The first `count` physical cores (no hyperthread siblings) local to the leased GPUs: the CPU-side half of a round
    (response parsing, data conversion) then runs next to the GPUs, and its between-run spread drops (measured on
    GPU 2/3: replay-segment medians 85.1 / 86.0 s pinned vs 85.9-89.6 s unpinned)."""
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader"], text=True)
    bus = {r.split(",")[0].strip(): r.split(",")[1].strip() for r in rows.splitlines() if "," in r}
    local = None
    for u in uuids:
        dom, rest = bus[u].lower().split(":", 1)
        dev = Path(f"/sys/bus/pci/devices/{dom[-4:]}:{rest}")
        cpus = set(parse_cpulist((dev / "local_cpulist").read_text()))
        local = cpus if local is None else local & cpus
    physical = [c for c in sorted(local or ()) if c not in exclude and parse_cpulist(
        Path(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read_text())[0] == c]
    require(len(physical) >= count, f"only {len(physical)} physical cores local to the GPU pair, MDWAN_CPU_CORES is {count} "
                                    "(set MDWAN_CPU_CORES or CPUSET; T depends on the core count)")
    return ",".join(str(c) for c in physical[:count])


def pinned_elsewhere(run_id):
    """{container name: pinned cpus} of every running container outside this run."""
    out = {}
    for cid in docker("ps", "-q").stdout.split():
        name, cpus = docker("inspect", "--format", "{{.Name}}|{{.HostConfig.CpusetCpus}}", cid).stdout.strip().split("|", 1)
        if cpus and run_id not in name:
            out[name.lstrip("/")] = set(parse_cpulist(cpus))
    return out


def cpu_pinning(cfg, uuids):
    """SForge's opt-in CPU pinning policy (SFORGE_CPUSET..., docs/CPU_AFFINITY.md) for the pair, or None when it is off.

    With a policy the pair's CPUs come from it (auto: the union of the two GPUs' slices, memory bound to their NUMA
    node) instead of MDWAN_CPU_CORES / CPUSET, which must then be unset; the quota is the slice's CPU count."""
    pinning = shared_cpuset(uuids, environ=policy_values(), harness_root=cfg["HARNESS_ROOT"])
    if pinning:
        require(not cfg.get("CPUSET") and not cfg.get("JUDGE_CPUSET")
                and HostConfig(run_config=cfg).source("MDWAN_CPU_CORES") is None,
                f"the CPU pinning policy {pinning['policy_id']} chooses the cpuset ({pinning['cpuset_cpus']}): "
                "unset CPUSET, JUDGE_CPUSET and MDWAN_CPU_CORES, or unset the policy")
    return pinning


def choose_cpuset(cfg, uuids, run_id, count, pinning=None, key="CPUSET", exclude=()):
    """CPUSET from the config, else the first `count` (MDWAN_CPU_CORES) physical cores local to the pair; never shared
    with another container's pinned cores (as in the SD3.5-M task: the timed replay is partly CPU-bound)."""
    if pinning:
        cpus = parse_cpulist(pinning["cpuset_cpus"])
    elif cfg.get(key):
        cpus = parse_cpulist(cfg[key])
        require(len(set(cpus)) == count, f"{key} must name {count} cores")
        require(all(parse_cpulist(Path(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read_text())[0] == c
                    for c in cpus), f"{key} must name physical cores (first thread of each sibling pair)")
    else:
        cpus = parse_cpulist(numa_cpuset(uuids, count, exclude))
    require(not set(cpus) & set(exclude),
            f"the Judge's cpuset {','.join(map(str, cpus))} overlaps Work's: with a separate Judge pair both run at once; "
            "set disjoint CPUSET / JUDGE_CPUSET, use SFORGE_CPUSET=auto, or --shared-gpus")
    for name, other in pinned_elsewhere(run_id).items():
        require(not other & set(cpus), f"cpuset {','.join(map(str, cpus))} overlaps the pinned cores of {name}; "
                                       "set CPUSET in the config to free physical cores")
    return ",".join(str(c) for c in cpus)


def gpus_idle(uuids):
    owners = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid", "--format=csv,noheader"], text=True)
    return not set(owners.split()).intersection(uuids)


def judge_gpu_spec(cfg, judge_gpus=None, shared_gpus=False):
    """Where the Judge runs: "C,D" (its own pair, the default setup) or SHARED (the Work pair, Work paused for each
    Judge). --judge-gpus / --shared-gpus beat the config's JUDGE_GPU_INDICES; with none of them the launch refuses."""
    if shared_gpus:
        return SHARED
    spec = str(judge_gpus or cfg.get("JUDGE_GPU_INDICES") or "").strip()
    require(spec, "no Judge GPUs given: set JUDGE_GPU_INDICES=<c>,<d> in the config or pass --judge-gpus C,D (a separate "
                  "Judge pair; Work keeps running during Judges), or JUDGE_GPU_INDICES=shared / --shared-gpus (the old "
                  "mode: the Judge measures on GPU_INDICES and pauses Work)")
    require(spec == SHARED or re.fullmatch(r"\d+,\d+", spec),
            f"Judge GPUs {spec!r}: JUDGE_GPU_INDICES / --judge-gpus must be two GPU indices C,D (or 'shared')")
    return spec


def placement(work_indices, judge_spec):
    """("separate" | "shared", Work (indices, uuids), Judge (indices, uuids)). A separate Judge pair shares no GPU with
    the Work pair: all four indices and UUIDs are distinct."""
    indices, uuids, _ = gpu_pair(work_indices)
    if judge_spec == SHARED:
        return "shared", (indices, uuids), (indices, uuids)
    judge_indices, judge_uuids, _ = gpu_pair(judge_spec)
    require(len(set(indices + judge_indices)) == 4 and len(set(uuids + judge_uuids)) == 4,
            f"Judge GPUs {judge_spec} overlap the Work GPUs {work_indices} (or repeat a GPU): pass --judge-gpus C,D with "
            "two other GPUs, or --shared-gpus (JUDGE_GPU_INDICES=shared) to run the Judge on the Work pair")
    return "separate", (indices, uuids), (judge_indices, judge_uuids)


def held_gpus(uuids, judge_uuids):
    """Every GPU the run uses (the Work pair, then the Judge pair when it is separate), each once."""
    return list(dict.fromkeys([*uuids, *judge_uuids]))


def judge_cpuset(args, cfg, run_id, cpuset):
    """The Judge's cpuset: Work's when the pair is shared (one cpuset, as before); with a separate pair JUDGE_CPUSET, the
    pinning policy's slice of the Judge pair, or the first cores local to the Judge pair that Work does not use. Never
    overlapping Work's: both roles run at once."""
    if args.judge_placement == "shared":
        return cpuset
    return choose_cpuset(cfg, args.judge_uuids, run_id, args.judge_cpu_cores, args.judge_pinning, key="JUDGE_CPUSET",
                         exclude=set(parse_cpulist(cpuset)))


@contextlib.contextmanager
def judge_lock(run_id, timeout):
    """The per-run Judge lock of runtime/sforge_entry.py, so the final evaluation waits for a manual Judge still
    measuring on the Judge pair instead of failing on a busy pair."""
    with open(f"/tmp/{JUDGE_LOCK_PREFIX}-judge-{run_id}.lock", "a+") as fh:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                require(time.monotonic() < deadline, "a manual Judge still holds the Judge lock before the final evaluation")
                time.sleep(5)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def check_lease_holder(path, uuids):
    """The pair's scheduler flocks are held by a long-lived holder process on behalf of this work: verify that it is
    alive and holds all of them (from /proc/locks) instead of taking them."""
    info = json.loads(Path(path).read_text())
    require(set(info.get("gpu_uuids", [])) == set(uuids), f"lease holder {path} covers other GPUs")
    pid = int(info["pid"])
    require(Path(f"/proc/{pid}").exists(), f"lease holder pid {pid} is not running")
    held = {}
    for line in Path("/proc/locks").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 6 and parts[1] == "FLOCK" and parts[3] == "WRITE":
            held[parts[5].rsplit(":", 1)[1]] = int(parts[4])
    for uuid in uuids:
        for prefix in ("mlsysforge-gpu", "sforge-gpu"):
            inode = str(os.stat(f"/tmp/{prefix}-{uuid}.lock").st_ino)
            require(held.get(inode) == pid, f"{prefix}-{uuid} is not held by lease holder pid {pid}")
    return pid


def arbiter_owner(cfg):
    if not cfg.get("GPU_ARBITER"):
        return None
    tool, name = cfg["GPU_ARBITER"].split()
    first = subprocess.run([tool, "status"], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    require(first.split()[1:2] == [name], f"GPU_ARBITER: {name} does not own the pair ({first})")
    return tool, name


@contextlib.contextmanager
def pair_lease(uuids, holder=None):
    if holder:
        check_lease_holder(holder, uuids)
        yield
        return
    with contextlib.ExitStack() as stack:
        for uuid in sorted(uuids):
            for prefix in ("mlsysforge-gpu", "sforge-gpu"):
                fh = stack.enter_context(open(f"/tmp/{prefix}-{uuid}.lock", "a"))
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as e:
                    raise NotReady(f"GPU lease busy: {prefix}-{uuid}") from e
        yield


def check_runtime(cfg):
    return agent.check_runtime(cfg, require)


def commands(cfg, run_id, judge_url, port, separate=False):
    """separate: the Judge has its own GPU pair, so the Judge server runs without pausing Work (SEPARATE_JUDGE)."""
    py = str(Path(cfg["HARNESS_ROOT"]) / ".venv/bin/python")
    prefix = [py, str(ENTRY)]
    return {
        "serve": [py, str(SEPARATE_JUDGE) if separate else str(ENTRY)] + ["serve", "--host", "0.0.0.0", "--port", str(port)],
        "run": prefix + ["run", "--backend", "docker", "--task", TASK_ID, "--agent", agent.name(cfg), "--model", cfg["MODEL"],
                         "--timeout", cfg["TASK_TIMEOUT"], "--disable-auto-eval", "--submission-cooldown",
                         cfg["SUBMISSION_COOLDOWN"], "--enable-internet", "--judge-url", judge_url,
                         "--work-mem-limit", "256g", "--judge-mem-limit", "256g", "--run-id", run_id],
        "eval": prefix + ["eval", "--backend", "docker", "--task", TASK_ID, "--archive",
                          str(OUTPUT / "runs" / run_id / TASK_ID / "final_archive.tar.gz"),
                          "--run-id", run_id + "-final", "--timeout", str(JUDGE_TIMEOUT), "--json"],
    }


@contextlib.contextmanager
def isolated_relay(cfg, run_id, port, image, internet=False):
    """Work gets one internal network; a relay container alone reaches the model API and the Judge.

    Endpoint mode: the relay forwards port 3131 to the one model host; its CONNECT allowlist is empty.
    ChatGPT mode: the relay's CONNECT proxy (3128) tunnels to the OpenAI auth/model hosts.
    internet: the relay also runs with --allow-public-egress (public hosts on 80/443, non-public addresses
    refused) and Work gets its CONNECT proxy as HTTP(S)_PROXY."""
    if cfg["MODE"] == "endpoint":
        url = urlparse(cfg["MODEL_API_BASE_URL"])
        require(url.scheme == "http" and url.hostname, "model API must be a plain HTTP endpoint for the relay")
        program = ["/opt/fixed_api_relay.py", "--evaluation-upstream-host", "host.docker.internal",
                   "--evaluation-upstream-port", str(port), "--model-forward", f"3131:{url.hostname}:{url.port or 80}"]
    else:
        program = ["/opt/existing/restricted_connect_proxy.py", "--evaluation-upstream-host", "host.docker.internal",
                   "--evaluation-upstream-port", str(port)]
    if internet:
        program.append("--allow-public-egress")
    network, name = f"mdwan-{run_id}", f"mdwan-{run_id}-relay"
    docker("network", "create", "--internal", "--label", f"mdwan.run={run_id}", network)
    try:
        docker("run", "-d", "--name", name, "--label", f"mdwan.run={run_id}", "--network", "bridge", "--read-only",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true", "--memory", "512m", "--cpus", "1",
               "--user", "65534:65534", "--env", "NVIDIA_VISIBLE_DEVICES=void", "--env", "CUDA_VISIBLE_DEVICES=",
               "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m", "--add-host", "host.docker.internal:host-gateway",
               "--mount", f"type=bind,src={Path(cfg['HARNESS_ROOT']) / PROXY_SOURCE},dst=/opt/existing/restricted_connect_proxy.py,readonly",
               "--mount", f"type=bind,src={TASK / 'runtime/fixed_api_relay.py'},dst=/opt/fixed_api_relay.py,readonly",
               "--entrypoint", "python3", image, *program)
        docker("network", "connect", "--alias", RELAY_ALIAS, network, name)
        attachment = json.loads(docker("inspect", name).stdout)[0]["NetworkSettings"]["Networks"][network]
        ip = attachment["IPAddress"]
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline = time.monotonic() + 30
        while True:
            try:
                with opener.open(f"http://{ip}:3128/health", timeout=1) as r:
                    if r.status == 200:
                        break
            except OSError:
                require(time.monotonic() < deadline, "relay health check failed")
                time.sleep(0.5)
        probes = check_connect_policy(ip, internet)
        yield {"network": network, "ip": ip, "judge_url": f"http://{ip}:3130",
               "api_base": f"http://{ip}:3131{url.path}" if cfg["MODE"] == "endpoint" else None,
               "public_egress": internet, "connect_probes": probes,
               "work_proxy_env": work_proxy_env(ip, attachment["IPPrefixLen"]) if internet else {}}
    finally:
        docker("rm", "-f", name, check=False)
        docker("network", "rm", network, check=False)


def review_settings(cfg):
    """The SFORGE_LLM_JUDGE_* settings for the Judge server and the final evaluation: environment, then the launch
    config, then the host config file (launch/mlsysforge.env). In ChatGPT mode the run's Codex login is the review's
    default credential (the harness falls back to SFORGE_CODEX_AUTH_JSON_PATH)."""
    from miles_common.host_config import _parse_env_file, host_config_file

    out = {}
    hfile = host_config_file()
    for source in ((_parse_env_file(hfile) if hfile else {}), cfg, os.environ):
        out.update({k: v for k, v in source.items() if k.startswith(REVIEW_PREFIX) and str(v).strip()})
    if cfg["MODE"] == "chatgpt":
        out.setdefault("SFORGE_CODEX_AUTH_JSON_PATH", str(Path(cfg["CODEX_AUTH_JSON_PATH"]).expanduser().resolve()))
    return out


def check_review(cfg, review):
    """The Judge server's own start-up check (LLMJudgeConfig.check_ready) on the environment it will get; a review that
    is on but cannot run would make the server exit and the final evaluation end without a score."""
    sys.path.insert(0, cfg["HARNESS_ROOT"])
    from sforge.harness.llm_judge.config import LLMJudgeConfigError, load_llm_judge_config

    env = dict(review, SFORGE_AGENT_RUNTIME_DIR=cfg["SFORGE_AGENT_RUNTIME_DIR"])
    try:
        rc = load_llm_judge_config(env)
        rc.check_ready()
    except LLMJudgeConfigError as e:
        raise NotReady(f"LLM integrity review not runnable ({e}); set SFORGE_LLM_JUDGE_* in the launch config or "
                       "launch/mlsysforge.env (docs/LLM_JUDGE.md)") from e
    return {"mode": rc.mode, **({k: v for k, v in rc.identity().items() if k in ("engine", "agent", "auth", "model",
                                                                               "reasoning_effort")} if rc.enabled else {})}


def supervise(args, cfg, uuids):
    run_root = OUTPUT / "runs" / args.run_id
    require(not run_root.exists(), "run ID already exists")
    ids = {role: docker("image", "inspect", "--format", "{{.Id}}", f"{BENCH}.{role}.{TASK_ID}:local").stdout.strip()
           for role in ("work", "judge")}
    port = int(cfg["JUDGE_PORT"])
    with socket.socket() as s:
        try:
            s.bind(("0.0.0.0", port))
        except OSError as e:
            raise NotReady(f"Judge port {port} in use") from e
    arbiter = arbiter_owner(cfg)
    cpuset = choose_cpuset(cfg, uuids, args.run_id, args.host["cpu_cores"], args.pinning)
    judge_cs = judge_cpuset(args, cfg, args.run_id, cpuset)
    separate = args.judge_placement == "separate"
    held = held_gpus(uuids, args.judge_uuids)  # four GPUs with a separate Judge pair, all held for the whole run
    with pair_lease(held, cfg.get("LEASE_HOLDER")):
        require(gpus_idle(held), "GPU pair has active compute processes")
        run_root.mkdir(parents=True)
        manifest = {"task_id": TASK_ID, "run_id": args.run_id, "model": cfg["MODEL"], "effort": cfg["REASONING_EFFORT"],
                    "model_mode": cfg["MODE"], "internet": args.internet,
                    **agent.manifest_fields(cfg, args.internet), "agent_runtime": check_runtime(cfg),
                    "budget_seconds": int(cfg["TASK_TIMEOUT"]), "gpu_indices": args.gpus, "gpu_uuids": uuids,
                    "cpuset": cpuset, "cpu_cores": args.host["cpu_cores"], "execution_profile_id": args.profile_id,
                    # gpu_indices / gpu_uuids / cpuset / cpu_cores / cpu_pinning: Work's (both roles' when shared)
                    "judge_placement": args.judge_placement, "judge_gpu_indices": args.judge_gpus,
                    "judge_gpu_uuids": args.judge_uuids, "judge_cpuset": judge_cs, "judge_cpu_cores": args.judge_cpu_cores,
                    **({"judge_cpu_pinning": args.judge_pinning["record"]} if args.judge_pinning else {}),
                    **({"cpu_pinning": args.pinning["record"]} if args.pinning else {}),
                    "host": args.host, "images": ids, "status": "starting", "score": None}
        write = lambda: (run_root / "launch-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        write()
        procs = []
        locks = contextlib.ExitStack()
        try:
            with isolated_relay(cfg, args.run_id, port, ids["work"], internet=args.internet) as relay:
                cmds = commands(cfg, args.run_id, relay["judge_url"], port, separate=separate)
                manifest["commands"] = cmds
                manifest["relay"] = {"public_egress": relay["public_egress"], "connect_probes": relay["connect_probes"]}
                base = {k: os.environ[k] for k in ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TZ") if k in os.environ}
                base.update({"PYTHONPATH": cfg["HARNESS_ROOT"], "SFORGE_TASKS_DIR": str(TASK / "tasks"),
                             "SFORGE_LOG_DIR": str(OUTPUT), "SFORGE_VERSION_STORE_ROOT": cfg["SFORGE_VERSION_STORE_ROOT"],
                             "SFORGE_AGENT_RUNTIME_DIR": cfg["SFORGE_AGENT_RUNTIME_DIR"],
                             "SFORGE_ADMIN_SECRET": secrets.token_urlsafe(48), "SFORGE_JUDGE_NETWORK_MODE": "none",
                             # runtime/sforge_entry.py: Codex web search, prompt and Work shell proxy rules.
                             "MLSYSFORGE_WORK_INTERNET": "true" if args.internet else "false",
                             "MDWAN_GPU_UUIDS": ",".join(uuids), "MDWAN_RUN_ID": args.run_id, "MDWAN_RUN_ROOT": str(run_root),
                             "MDWAN_INTERNAL_NETWORK": relay["network"], "MDWAN_ASSETS": args.host["MDWAN_ASSETS"],
                             "MDWAN_PUBLIC_DATA": args.host["MDWAN_PUBLIC_DATA"],
                             "MDWAN_PRIVATE_DATA": args.host["MDWAN_PRIVATE_DATA"], "MDWAN_CPUSET": cpuset})
                if args.pinning:  # the harness records the task-owned cpuset; memory on the slice's nodes
                    base.update(args.pinning["record"]["policy_env"])
                    if args.pinning["cpuset_mems"]:
                        base["MDWAN_CPUSET_MEMS"] = args.pinning["cpuset_mems"]
                judge_env = dict(base, **args.review)  # the Judge server and `sforge eval` both run the review
                # The Judge server and `sforge eval` create every Judge on the Judge pair with its cpuset (the Work pair
                # and cpuset when shared); `run` creates Work on the Work pair.
                judge_env.update({"MDWAN_GPU_UUIDS": ",".join(args.judge_uuids), "MDWAN_CPUSET": judge_cs})
                if args.judge_pinning:
                    judge_env.pop("MDWAN_CPUSET_MEMS", None)
                    if args.judge_pinning["cpuset_mems"]:
                        judge_env["MDWAN_CPUSET_MEMS"] = args.judge_pinning["cpuset_mems"]
                env = dict(base, **agent.run_env(cfg, run_root))
                # Internet: Work (and so every agent shell) gets the relay's CONNECT proxy; the relay,
                # loopback and Work's own addresses stay direct (runtime/public-egress.profile.sh).
                env.update(relay["work_proxy_env"])
                if cfg["MODE"] == "endpoint":
                    env["SFORGE_AGENT_API_BASE_URL"] = relay["api_base"]
                    env["SFORGE_AGENT_API_KEY"] = Path(cfg["MODEL_API_KEY_FILE"]).read_text().strip()
                    require(env["SFORGE_AGENT_API_KEY"], "API key file is empty")
                elif cfg["MODE"] == "chatgpt":
                    # The harness prefixes codex exec with this proxy (HTTPS_PROXY). Offline, Ray and the
                    # engines inside Work see no proxy variables and the entry keeps them out of the agent's
                    # shell commands; with internet the shells keep the same proxy.
                    env.update(SFORGE_CODEX_AUTH_JSON_PATH=str(Path(cfg["CODEX_AUTH_JSON_PATH"]).expanduser().resolve()),
                               SFORGE_AGENT_API_BASE_URL=f"http://{relay['ip']}:3128",
                               SFORGE_AGENT_EXTRA_ENV="SFORGE_CODEX_CHATGPT_BASE_URL=https://chatgpt.com/backend-api/")
                write()
                with open(run_root / "judge-server.log", "w") as log:
                    serve = subprocess.Popen(cmds["serve"], env=judge_env, cwd=cfg["HARNESS_ROOT"], stdout=log,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                procs.append(serve)
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                deadline = time.monotonic() + 90
                while True:
                    require(serve.poll() is None, "Judge server exited early; see judge-server.log")
                    try:
                        with opener.open(f"http://127.0.0.1:{port}/openapi.json", timeout=2):
                            break
                    except OSError:
                        require(time.monotonic() < deadline, "Judge server readiness timeout")
                        time.sleep(1)
                manifest["status"] = "running"
                write()
                for stage in ("run", "eval"):
                    if stage == "eval":
                        final = OUTPUT / "runs" / args.run_id / TASK_ID / "final_archive.tar.gz"
                        require(final.is_file() and final.stat().st_size > 0, "no final archive")
                        if separate:  # a manual Judge may still be measuring on the Judge pair
                            locks.enter_context(judge_lock(args.run_id, JUDGE_TIMEOUT + 900))
                        deadline = time.monotonic() + 900
                        while not gpus_idle(args.judge_uuids):
                            require(time.monotonic() < deadline, "GPU pair still busy before final evaluation")
                            time.sleep(5)
                    with open(run_root / f"{stage}.log", "w") as log:
                        p = subprocess.Popen(cmds[stage], env=env if stage == "run" else judge_env, cwd=cfg["HARNESS_ROOT"],
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        procs.append(p)
                        code = p.wait()
                    manifest[f"{stage}_exit"] = code
                    write()
                    require(code == 0, f"{stage} exited {code}; see {stage}.log")
                reports = sorted((OUTPUT / "runs" / (args.run_id + "-final")).rglob("report.json"))
                if reports:
                    rep = json.loads(reports[-1].read_text())
                    manifest.update(final_report=str(reports[-1]), valid=rep.get("valid"), score=rep.get("score"),
                                    metrics=rep.get("metrics"))
                manifest["status"] = "completed"
        finally:
            for p in reversed(procs):
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)
                    try:
                        p.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid, signal.SIGKILL)
            locks.close()
            for c in docker("ps", "-aq", "--filter", f"label=mdwan.run={args.run_id}", check=False).stdout.split():
                docker("rm", "-f", c, check=False)
            if manifest["status"] != "completed":
                manifest["status"] = "failed"
            write()
            if arbiter:
                subprocess.run([arbiter[0], "release", arbiter[1]], check=False, capture_output=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-id", default="mdwan-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--foreground", action="store_true")
    judge = ap.add_mutually_exclusive_group()
    judge.add_argument("--judge-gpus", metavar="C,D",
                       help="the Judge's own GPU pair (overrides JUDGE_GPU_INDICES); Work stays on GPU_INDICES and is never "
                            "paused")
    judge.add_argument("--shared-gpus", action="store_true",
                       help="the old mode: the Judge measures on the Work pair (GPU_INDICES) and pauses Work")
    args = ap.parse_args(argv)
    errors = []
    args.internet = None
    try:
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,120}", args.run_id), "unsafe run ID")
        args.internet = task_allows_internet()
        cfg = load_config(args.config)
        args.host = resolve_host(cfg)
        require((Path(cfg["HARNESS_ROOT"]) / ".venv/bin/python").is_file(), "HARNESS_ROOT has no .venv/bin/python")
        runtime = check_runtime(cfg)
        check_credentials(cfg)
        agent.check_internet(cfg, args.internet, require)
        args.gpus = cfg["GPU_INDICES"]
        spec = judge_gpu_spec(cfg, args.judge_gpus, args.shared_gpus)
        args.judge_placement, (indices, uuids), (judge_indices, args.judge_uuids) = placement(args.gpus, spec)
        args.judge_gpus = args.gpus if spec == SHARED else spec
        separate = args.judge_placement == "separate"
        require(separate or not cfg.get("JUDGE_CPUSET"), "JUDGE_CPUSET is the cpuset of a separate Judge pair: unset it "
                                                         "with --shared-gpus (JUDGE_GPU_INDICES=shared)")
        require(not (separate and cfg.get("GPU_ARBITER")), "GPU_ARBITER arbitrates one shared GPU pair: unset it for a "
                                                           "separate Judge pair, or pass --shared-gpus")
        args.profile_id = admit(TASK_JSON, indices)["execution_profile_id"]
        if separate:  # the Judge pair must meet the same profile: one qualification record
            try:
                admit(TASK_JSON, judge_indices, profile_id=args.profile_id)
            except ProfileMismatch as e:
                raise ProfileMismatch(f"Judge GPUs {args.judge_gpus}: {e}") from e
        args.pinning = cpu_pinning(cfg, uuids)
        args.judge_pinning = cpu_pinning(cfg, args.judge_uuids) if separate else args.pinning
        # the Judge's CPU count (T depends on it, check_qualification) and Work's (the manifest's cpu_cores)
        args.judge_cpu_cores = args.judge_pinning["cpu_count"] if args.judge_pinning else args.host["cpu_cores"]
        if args.pinning:
            args.host["cpu_cores"] = args.pinning["cpu_count"]
        check_qualification(args.host["MDWAN_PRIVATE_DATA"], args.profile_id, args.judge_cpu_cores,
                            args.judge_pinning and args.judge_pinning["policy_id"])
        held = held_gpus(uuids, args.judge_uuids)
        with pair_lease(held, cfg.get("LEASE_HOLDER")):  # probe (released at once) so a dry run tells the truth
            require(gpus_idle(held), "GPU pair has active compute processes")
        arbiter_owner(cfg)
        cpuset = choose_cpuset(cfg, uuids, args.run_id, args.host["cpu_cores"], args.pinning)
        judge_cs = judge_cpuset(args, cfg, args.run_id, cpuset)
        for role in ("work", "judge"):
            require(docker("image", "inspect", f"{BENCH}.{role}.{TASK_ID}:local", check=False).returncode == 0,
                    f"{role} image missing: run scripts/build_images.sh")
        check_image_metric()
        args.review = review_settings(cfg)
        review_identity = check_review(cfg, args.review)
    except (NotReady, NotConfigured, ProfileMismatch, PinningError, OSError, KeyError, ValueError) as e:
        errors.append(str(e))
    plan = {"status": "not_ready" if errors else "ready", "errors": errors, "task_id": TASK_ID, "run_id": args.run_id}
    if not errors:
        plan.update(integrity_review=review_identity, model=cfg["MODEL"], model_mode=cfg["MODE"], effort=cfg["REASONING_EFFORT"], internet=args.internet, **agent.manifest_fields(cfg, args.internet),
                    runtime=runtime.get("tree_sha256"), budget_seconds=int(cfg["TASK_TIMEOUT"]), gpus=args.gpus,
                    gpu_uuids=uuids, judge_placement=args.judge_placement, judge_gpus=args.judge_gpus,
                    judge_gpu_uuids=args.judge_uuids, judge_cpuset=judge_cs, judge_cpu_cores=args.judge_cpu_cores,
                    execution_profile_id=args.profile_id, cpuset=cpuset,
                    cpu_cores=args.host["cpu_cores"], host=args.host, outputs=str(OUTPUT / "runs" / args.run_id),
                    **({"cpu_pinning": args.pinning["record"]} if args.pinning else {}))
    print(json.dumps(plan, indent=2), flush=True)
    if errors:
        return 78
    if args.dry_run:
        return 0
    if not args.foreground:
        log_dir = OUTPUT / "launcher-logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / f"{args.run_id}.log"
        # tmux may not pass this environment to the session: carry the pinning policy explicitly.
        # tmux starts the session with the server's environment, not this one: carry the host settings that
        # may come from the environment (asset root, model cache, Miles locations, SForge settings) and the
        # pinning policy explicitly.
        from miles_common.host_config import KNOWN as host_keys
        carried = {k: v for k, v in os.environ.items()
                   if k.startswith(("SFORGE_", "INFRATHON_", "MILES_", "MLSYSFORGE_")) or k in host_keys
                   or k in ("HF_HUB_CACHE", "HF_HOME")}
        pin = [f"{k}={v}" for k, v in {**carried, **policy_values()}.items()]
        cmd = [*(["env", *pin] if pin else []), sys.executable, str(Path(__file__).resolve()), "--config", str(args.config.resolve()),
               "--run-id", args.run_id, "--foreground",
               *(["--shared-gpus"] if args.judge_placement == "shared" else ["--judge-gpus", args.judge_gpus])]
        inner = f"{shlex.join(cmd)} > {shlex.quote(str(log))} 2>&1"
        # An already-running tmux server spawns sessions with its own credentials, so
        # docker group access is re-activated inside the session itself.
        subprocess.run(["tmux", "new-session", "-d", "-s", args.run_id, f"exec sg docker -c {shlex.quote(inner)}"],
                       check=True)
        print(json.dumps({"status": "detached", "tmux_session": args.run_id, "log": str(log)}))
        return 0
    try:
        supervise(args, cfg, uuids)
    except NotReady as e:
        print(json.dumps({"status": "failed", "error": str(e)}), file=sys.stderr)
        return 78
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
