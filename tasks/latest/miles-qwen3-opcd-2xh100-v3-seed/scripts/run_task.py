#!/usr/bin/env python3
"""Supervise one agent run of the Miles OPCD task v3 (Work -> manual Judges -> final evaluation).

Everything about the agent comes from the launch config (launch/miles-opcd/*.env):
harness root, Codex CLI version and host runtime, model, API endpoint and key file,
reasoning effort, budget, submission cooldown and GPU pairs. The task images contain no
agent. Two model modes: a custom OpenAI-compatible endpoint (MODEL_API_BASE_URL + MODEL_API_KEY_FILE,
reached through a fixed TCP forward) or a Codex ChatGPT login (CODEX_AUTH_JSON_PATH, reached through
the relay's CONNECT proxy to the OpenAI auth/model hosts). When the task JSON sets "internet": true the
relay's CONNECT proxy also tunnels to public hosts on ports 80/443 (--allow-public-egress; non-public
addresses are refused), Work's shells use it as HTTP(S)_PROXY and Codex runs with live web search; Work
stays on an internal network with no direct route out. This supervisor holds both scheduler lock families for every GPU it uses (the Work pair, and the Judge pair when separate) for the whole run.

AGENT=claude-code with CLAUDE_OAUTH_TOKEN_FILE runs Claude Code instead of Codex (scripts/miles_common/agent.py).

Host locations (Miles upstream, assets, model cache, scratch) come from scripts/miles_common/host_config.py:
environment, then this launch config, then the host config file (launch/mlsysforge.env), then values derived
from MILES_DATA_ROOT / the Hugging Face cache. Runs and their scratch live under $MILES_SCRATCH_ROOT/<task>.

The LLM integrity review (llm_judge/POLICY.md) is mandatory: the SFORGE_LLM_JUDGE_* settings (environment, then this
launch config, then the host config file) reach the Judge server and the final `sforge eval`, which both review a
submission before the GPU Judge; in ChatGPT mode the review defaults to the run's own Codex login. The launch refuses
to start when the review is on but cannot run (the same readiness check the Judge server makes at start). The v2
launcher passed none of these settings, so its Judge server could not start once the review became mandatory.

GPUs: Work runs on GPU_INDICES (an NVLink pair). The Judge (every manual Judge of the Judge server and the final
evaluation) runs on its own pair, JUDGE_GPU_INDICES=<c>,<d> or --judge-gpus C,D (the default setup: all four GPUs
distinct, Work keeps running while a Judge measures, and the Judge server runs through scripts/separate_judge.py so Work
is never paused). JUDGE_GPU_INDICES=shared or --shared-gpus keeps the old mode: the Judge measures on the Work pair and
pauses Work for its whole duration. With neither the launch refuses to start. The supervisor holds the scheduler locks of
every GPU it uses (two or four) for the whole run.

  run_task.py --config launch/miles-opcd/deepseek.env [--judge-gpus C,D | --shared-gpus] [--dry-run] [--foreground]
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

TASK_ID = "miles_qwen3_opcd_2xh100_v3"
BENCH = "miles_opcd"
TASK = Path(__file__).resolve().parents[1]
REPO = TASK.parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from miles_common.cpu_pinning import PIN_KEYS, PinningError, policy_values, resolve as resolve_pinning  # noqa: E402
from miles_common.gpu_profile import ProfileMismatch, admit  # noqa: E402
from miles_common import agent, qualification  # noqa: E402
from miles_common.host_config import HostConfig, NotConfigured  # noqa: E402

TASK_JSON = TASK / "tasks" / f"{TASK_ID}.json"
QUALIFICATION = TASK / "runtime/qualification.json"
OUTPUT = None  # $MILES_SCRATCH_ROOT/<task>, set by main() after the disk checks
SCRATCH_MIN_FREE_GIB = 200  # an agent run held 99 GB before cleanup, each reference Judge ~30 GB
ENTRY = TASK / "runtime/sforge_entry.py"
# The Judge server of a run with a separate Judge pair: runtime/sforge_entry.py without the pause of Work.
SEPARATE_JUDGE = TASK / "scripts/separate_judge.py"
JUDGE_LOCK_PREFIX = "opcd"  # runtime/sforge_entry.py exclusive_judge: /tmp/<prefix>-judge-<run id>.lock
SHARED = "shared"  # JUDGE_GPU_INDICES=shared / --shared-gpus: the Judge measures on the Work pair
PROXY_SOURCE = "tasks/latest/llada-h100-decoder-optimization-v2-vanilla-seed/scripts/restricted_connect_proxy.py"
JUDGE_TIMEOUT = int(json.loads(TASK_JSON.read_text())["judge"]["eval_timeout"])
HOST_KEYS = {"HARNESS_ROOT", "MILES_DATA_ROOT", "MILES_SCRATCH_ROOT", "MILES_HF_CACHE", "OPCD_MEGATRON_INIT",
             "OPCD_HF_CACHE"}
KEYS = {"SFORGE_VERSION_STORE_ROOT", "SFORGE_AGENT_RUNTIME_DIR", "CODEX_CLI_VERSION", "AGENT", "CLAUDE_OAUTH_TOKEN_FILE",
        "CODEX_AUTH_JSON_PATH", "MODEL_API_BASE_URL", "MODEL_API_KEY_FILE", "MODEL", "REASONING_EFFORT",
        "TASK_TIMEOUT", "SUBMISSION_COOLDOWN", "GPU_INDICES", "JUDGE_GPU_INDICES", "JUDGE_PORT"} | HOST_KEYS
REVIEW_PREFIX = "SFORGE_LLM_JUDGE_"

RELAY_ALIAS = "opcd-relay"
WORK_HOSTNAME = "opcd-work"  # runtime/sforge_entry.py create_container
# CONNECT policy probes sent to the relay before Work starts: (authority, expected status). Internet mode
# must tunnel a public host and refuse non-web ports, the relay's loopback, the host (the host-gateway
# alias), the run network's gateway and link-local metadata; offline mode refuses every host outside the
# allowlist. The run network's gateway is added at run time (gateway_probe).
PUBLIC_EGRESS_PROBES = (("github.com:443", 200), ("github.com:22", 403), ("localhost:443", 403),
                        ("host.docker.internal:443", 403), ("169.254.169.254:80", 403))
RESTRICTED_PROBES = (("github.com:443", 403), ("localhost:443", 403), ("host.docker.internal:443", 403))


# CPU quota of Work and the Judge (runtime/sforge_entry.py nano_cpus). With SForge's opt-in CPU pinning policy
# (SFORGE_CPUSET..., docs/CPU_AFFINITY.md) in the environment or the host config file, the policy is passed to the
# harness, which pins both containers to the pair's CPU slice (scripts/miles_common/cpu_pinning.py); unset, no change.
CPU_BUDGET = 24


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


def check_connect_policy(relay_ip, internet, status=connect_status, gateway=None):
    """Fail unless the relay enforces the mode's CONNECT policy; returns the observed statuses. `gateway`, the
    run network's gateway address, must be refused in both modes."""
    probes = list(PUBLIC_EGRESS_PROBES if internet else RESTRICTED_PROBES)
    if gateway:
        probes.append((f"[{gateway}]:443" if ":" in gateway else f"{gateway}:443", 403))
    observed = {}
    for authority, expected in probes:
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


def network_gateway(network):
    """The run network's gateway address from docker network inspect (None when Docker records none)."""
    configs = (json.loads(docker("network", "inspect", network).stdout or "[{}]")[0].get("IPAM") or {}).get("Config") or []
    return next((c["Gateway"] for c in configs if c.get("Gateway")), None)


def load_config(path):
    cfg = {}
    for n, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        require(sep and (key.strip() in KEYS or key.strip().startswith(REVIEW_PREFIX)),
                f"{path}:{n}: unknown or malformed key {key.strip()!r}")
        cfg[key.strip()] = value.strip().strip('"').strip("'")
    # JUDGE_GPU_INDICES may come from --judge-gpus / --shared-gpus instead (judge_gpu_spec).
    optional = {"CODEX_AUTH_JSON_PATH", *agent.OPTIONAL_KEYS, "MODEL_API_BASE_URL", "MODEL_API_KEY_FILE",
                "JUDGE_GPU_INDICES"} | HOST_KEYS
    missing = sorted(KEYS - optional - set(cfg))
    require(not missing, f"{path}: missing keys {missing}")
    cfg["MODE"] = agent.config_mode(cfg, path, require)
    return cfg


def resolve_host(cfg):
    """Resolve the host locations into cfg and return them (dry runs print them)."""
    hc = HostConfig(run_config=cfg)
    cfg["HARNESS_ROOT"] = str(hc.path("HARNESS_ROOT"))
    hf = hc.path("OPCD_HF_CACHE")
    for repo in ("models--Qwen--Qwen3-1.7B",):  # v3: student and teacher are the same snapshot
        require((hf / repo / "refs" / "main").is_file(), f"OPCD_HF_CACHE={hf} has no {repo} with refs/main")
    host = {"OPCD_HF_CACHE": str(hf), "OPCD_MEGATRON_INIT": str(hc.path("OPCD_MEGATRON_INIT"))}
    host["output"] = str(hc.scratch(TASK.name, SCRATCH_MIN_FREE_GIB))
    return host


def check_qualification(profile_id, pinning=None):
    """The frozen base quality, quality gate and thresholds must come from a qualification on the admitted profile,
    and, when the record names the Judge's CPU pinning policy (the timed window T depends on it), under that policy.
    A record whose status is PENDING (a Judge change not yet validated by a reference Judge) admits no formal run."""
    record, qual = qualification.load(QUALIFICATION, profile_id)
    status = str(qual.get("status", ""))
    require(not status.startswith("PENDING"), f"{record}: {status.split(':', 1)[0]}: run a reference "
            "Judge on the rebuilt images (scripts/qualify.py judge --archive reference) and record it before a formal run")
    qualified_on = qual.get("execution_profile_id", "missing-profile")
    require(qualified_on == profile_id,
            f"{record} was measured on profile {qualified_on}, these GPUs are {profile_id}: "
            "re-qualify with scripts/qualify.py before a formal run")
    qualified_pin = (qual.get("cpu_pinning") or {}).get("policy_id")
    launch_pin = pinning["record"]["roles"]["judge"]["policy_id"] if pinning else None
    require(qualified_pin is None or qualified_pin == launch_pin,
            f"{record} was measured with the Judge pinned under {qualified_pin}, this launch pins it "
            f"under {launch_pin or 'no policy'}: export SFORGE_CPUSET=auto (docs/CPU_AFFINITY.md) for a formal run")


def check_credentials(cfg):
    agent.check_credentials(cfg, require)


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


def _cpus(cpulist):
    out = set()
    for part in (cpulist or "").split(","):
        if part.strip():
            lo, _, hi = part.partition("-")
            out.update(range(int(lo), int(hi or lo) + 1))
    return out


def split_pinning(uuids, judge_uuids, harness_root):
    """The CPU pinning of a run with a separate Judge pair, or None when no policy is set.

    The harness pins every container to its own GPUs' slice: Work (the `run` process, Work pair) gets the Work plan of
    the Work pair, every Judge (the Judge server and the final evaluation, Judge pair) the Judge plan of the Judge pair.
    Same shape as resolve_pinning(): "env" for Work's process and "record" with roles.work / roles.judge (what
    check_qualification reads), plus "judge_env" for the Judge's processes and the two per-pair records. The two roles
    run at the same time, so their CPUs must not overlap (an explicit SFORGE_CPUSET list names the same CPUs for any
    GPUs: use auto or disjoint SFORGE_WORK_CPUSET / SFORGE_JUDGE_CPUSET)."""
    work = resolve_pinning(uuids, CPU_BUDGET, environ=policy_values(), harness_root=harness_root)
    judge = resolve_pinning(judge_uuids, CPU_BUDGET, environ=policy_values(), harness_root=harness_root)
    if work is None or judge is None:
        return None
    work_plan, judge_plan = work["record"]["roles"]["work"], judge["record"]["roles"]["judge"]
    both = _cpus(work_plan.get("cpuset_cpus")) & _cpus(judge_plan.get("cpuset_cpus"))
    if both:
        raise PinningError(f"the CPU pinning policy gives Work (GPUs {','.join(uuids)}) CPUs {work_plan['cpuset_cpus']} "
                           f"and the Judge (GPUs {','.join(judge_uuids)}) CPUs {judge_plan['cpuset_cpus']}; with a "
                           "separate Judge pair both run at once: use SFORGE_CPUSET=auto or disjoint "
                           "SFORGE_WORK_CPUSET / SFORGE_JUDGE_CPUSET, or --shared-gpus")
    record = {**judge["record"], "gpu_ids": held_gpus(uuids, judge_uuids), "judge_placement": "separate",
              "roles": {"work": work_plan, "judge": judge_plan},
              "pairs": {"work": work["record"], "judge": judge["record"]}}
    return {"env": work["env"], "judge_env": judge["env"], "record": record}


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


@contextlib.contextmanager
def pair_lease(uuids):
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
    network, name = f"opcd-{run_id}", f"opcd-{run_id}-relay"
    docker("network", "create", "--internal", "--label", f"opcd.run={run_id}", network)
    try:
        docker("run", "-d", "--name", name, "--label", f"opcd.run={run_id}", "--network", "bridge", "--read-only",
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
        probes = check_connect_policy(ip, internet, gateway=network_gateway(network))
        yield {"network": network, "ip": ip, "judge_url": f"http://{ip}:3130",
               "api_base": f"http://{ip}:3131{url.path}" if cfg["MODE"] == "endpoint" else None,
               "public_egress": internet, "connect_probes": probes,
               "work_proxy_env": work_proxy_env(ip, attachment["IPPrefixLen"]) if internet else {}}
    finally:
        docker("rm", "-f", name, check=False)
        docker("network", "rm", network, check=False)


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
    separate = args.judge_placement == "separate"
    held = held_gpus(uuids, args.judge_uuids)  # four GPUs with a separate Judge pair, all held for the whole run
    with pair_lease(held):
        require(gpus_idle(held), "GPU pair has active compute processes")
        run_root.mkdir(parents=True)
        manifest = {"task_id": TASK_ID, "run_id": args.run_id, "model": cfg["MODEL"], "effort": cfg["REASONING_EFFORT"],
                    "model_mode": cfg["MODE"], "internet": args.internet,
                    **agent.manifest_fields(cfg, args.internet), "agent_runtime": check_runtime(cfg),
                    "budget_seconds": int(cfg["TASK_TIMEOUT"]), "gpu_indices": args.gpus, "gpu_uuids": uuids,
                    # gpu_indices / gpu_uuids: the Work pair (the only pair when shared)
                    "judge_placement": args.judge_placement, "judge_gpu_indices": args.judge_gpus,
                    "judge_gpu_uuids": args.judge_uuids,
                    "execution_profile_id": args.profile_id, "host": args.host,
                    **({"cpu_pinning": args.pinning["record"]} if args.pinning else {}),
                    "images": ids, "status": "starting", "score": None}
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
                             "OPCD_GPU_UUIDS": ",".join(uuids), "OPCD_RUN_ID": args.run_id, "OPCD_RUN_ROOT": str(run_root),
                             "OPCD_HF_CACHE": args.host["OPCD_HF_CACHE"], "OPCD_MEGATRON_INIT": args.host["OPCD_MEGATRON_INIT"],
                             "OPCD_INTERNAL_NETWORK": relay["network"]})
                if args.pinning:  # the allowlisted child environment gets the pinning policy
                    base.update(args.pinning["env"])
                judge_env = dict(base, **args.review)  # the Judge server and `sforge eval` both run the review
                # The Judge server and `sforge eval` create every Judge on the Judge pair (the Work pair when shared),
                # pinned under the policy to that pair's slice; `run` creates Work on the Work pair.
                judge_env["OPCD_GPU_UUIDS"] = ",".join(args.judge_uuids)
                if args.pinning and "judge_env" in args.pinning:
                    judge_env = {k: v for k, v in judge_env.items() if k not in PIN_KEYS}
                    judge_env.update(args.pinning["judge_env"])
                env = dict(base, **agent.run_env(cfg, run_root))
                # Internet: Work (and so every agent shell) gets the relay's CONNECT proxy; the relay,
                # loopback and Work's own addresses stay direct (runtime/public-egress.profile.sh).
                env.update(relay["work_proxy_env"])
                if cfg["MODE"] == "endpoint":
                    env["SFORGE_AGENT_API_BASE_URL"] = relay["api_base"]
                    env["SFORGE_AGENT_API_KEY"] = Path(cfg["MODEL_API_KEY_FILE"]).read_text().strip()
                    require(env["SFORGE_AGENT_API_KEY"], "API key file is empty")
                elif cfg["MODE"] == "chatgpt":
                    # The harness prefixes codex exec with this proxy (HTTPS_PROXY). Offline, Miles, Ray
                    # and SGLang inside Work see no proxy variables and the entry keeps them out of the
                    # agent's shell commands; with internet the shells keep the same proxy.
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
            for c in docker("ps", "-aq", "--filter", f"label=opcd.run={args.run_id}", check=False).stdout.split():
                docker("rm", "-f", c, check=False)
            if manifest["status"] != "completed":
                manifest["status"] = "failed"
            write()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-id", default="opcd-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--foreground", action="store_true")
    judge = ap.add_mutually_exclusive_group()
    judge.add_argument("--judge-gpus", metavar="C,D",
                       help="the Judge's own GPU pair (overrides JUDGE_GPU_INDICES); Work stays on GPU_INDICES and is never "
                            "paused")
    judge.add_argument("--shared-gpus", action="store_true",
                       help="the old mode: the Judge measures on the Work pair (GPU_INDICES) and pauses Work")
    args = ap.parse_args(argv)
    global OUTPUT
    errors = []
    args.internet = None
    try:
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,120}", args.run_id), "unsafe run ID")
        args.internet = task_allows_internet()
        cfg = load_config(args.config)
        args.host = resolve_host(cfg)
        OUTPUT = Path(args.host["output"])
        require((Path(cfg["HARNESS_ROOT"]) / ".venv/bin/python").is_file(), "HARNESS_ROOT has no .venv/bin/python")
        runtime = check_runtime(cfg)
        check_credentials(cfg)
        agent.check_internet(cfg, args.internet, require)
        args.gpus = cfg["GPU_INDICES"]
        spec = judge_gpu_spec(cfg, args.judge_gpus, args.shared_gpus)
        args.judge_placement, (indices, uuids), (judge_indices, args.judge_uuids) = placement(args.gpus, spec)
        args.judge_gpus = args.gpus if spec == SHARED else spec
        args.profile_id = admit(TASK_JSON, indices)["execution_profile_id"]
        if args.judge_placement == "separate":  # the Judge pair must meet the same profile: one qualification record
            try:
                admit(TASK_JSON, judge_indices, profile_id=args.profile_id)
            except ProfileMismatch as e:
                raise ProfileMismatch(f"Judge GPUs {args.judge_gpus}: {e}") from e
            args.pinning = split_pinning(uuids, args.judge_uuids, cfg["HARNESS_ROOT"])
        else:
            args.pinning = resolve_pinning(uuids, CPU_BUDGET, environ=policy_values(), harness_root=cfg["HARNESS_ROOT"])
        check_qualification(args.profile_id, args.pinning)
        args.review = review_settings(cfg)
        review_identity = check_review(cfg, args.review)
        held = held_gpus(uuids, args.judge_uuids)
        with pair_lease(held):  # probe the scheduler locks (released at once) so a dry run tells the truth
            require(gpus_idle(held), "GPU pair has active compute processes")
        for role in ("work", "judge"):
            require(docker("image", "inspect", f"{BENCH}.{role}.{TASK_ID}:local", check=False).returncode == 0,
                    f"{role} image missing: run scripts/build_images.sh")
    except (NotReady, NotConfigured, ProfileMismatch, PinningError, OSError, KeyError, ValueError) as e:
        errors.append(str(e))
    plan = {"status": "not_ready" if errors else "ready", "errors": errors, "task_id": TASK_ID, "run_id": args.run_id}
    if not errors:
        plan.update(integrity_review=review_identity, model=cfg["MODEL"], model_mode=cfg["MODE"], effort=cfg["REASONING_EFFORT"], internet=args.internet, **agent.manifest_fields(cfg, args.internet),
                    runtime=runtime.get("tree_sha256"), budget_seconds=int(cfg["TASK_TIMEOUT"]), gpus=args.gpus,
                    gpu_uuids=uuids, judge_placement=args.judge_placement, judge_gpus=args.judge_gpus,
                    judge_gpu_uuids=args.judge_uuids, execution_profile_id=args.profile_id, host=args.host,
                    **({"cpu_pinning": args.pinning["record"]} if args.pinning else {}),
                    outputs=str(OUTPUT / "runs" / args.run_id))
    print(json.dumps(plan, indent=2), flush=True)
    if errors:
        return 78
    if args.dry_run:
        return 0
    if not args.foreground:
        log_dir = OUTPUT / "launcher-logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log = log_dir / f"{args.run_id}.log"
        # tmux may not pass this environment to the session: carry the policy explicitly.
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
