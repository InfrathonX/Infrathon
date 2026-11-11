#!/usr/bin/env bash
set -Eeuo pipefail
umask 022  # private Judge outputs refuse group-writable parents

# Host (native Docker) fixed-budget agent run: one Work GPU, 1-8 Judge GPUs.
#
#   WORK_GPU=<index|uuid> JUDGE_GPUS=<i,j,...> JUDGE_PORT=<port> MODEL=<id> \
#   KV_EVICTION_MODEL_REPO=<HF cache repo of Qwen/Qwen3-8B> scripts/run_agent.sh
#
# launch/kvpress-kv-eviction-scratch/run.sh sets these from its flags
# (SFORGE_LAUNCH_DRY_RUN=1: resolve, check and print the plan, start nothing).
# Work gets exactly one GPU on an internal Docker network whose only peer is
# the task's restricted CONNECT sidecar (task JSON internet: true, so public
# hosts on 80/443 are reachable, internal addresses are not).  Every
# interactive Judge and the independent final Judge run the complete frozen
# 5,000-item suite sharded over the Judge GPUs (KV_EVICTION_JUDGE_GPU_UUIDS;
# scripts/sforge_entry.py gives the Judge every listed GPU and GPU-local CPUs).
# Both host lock names of every GPU (/tmp/sforge-gpu-<uuid>.lock and
# /tmp/mlsysforge-gpu-<uuid>.lock) are held from the first lock to the end of
# the final Judge.  Before any lock (and again under the locks) the launcher
# refuses a GPU that a held lock, a compute process, too little free memory, a
# container claiming it (running, paused, created or restarting; DeviceRequests
# or NVIDIA_VISIBLE_DEVICES) or a live row of GPU_RESERVATIONS_FILE (with
# GPU_RESERVATIONS_HOST_ID) names.  The Judge server, sforge run and the final
# Judge are children of this shell that die with it (setpriv --pdeathsig) and
# ignore SIGHUP; on any exit, a signal included, the cleanup stops them through
# their graceful signal (bounded), removes every container of this run, and
# checks that no GPU of this run is still in use before the locks go.
# The verifier deadlines and the Judge timeout come from
# lib.sh judge_timeouts for the Judge GPU count; the Judge timeout may only
# raise the task JSON's judge.eval_timeout (scripts/sforge_agent_entry.py).
# CPU and memory: the values passed here are the task's own launcher sizes
# (Work from the task JSON, Judge 8 CPU + 8g per GPU over 32g, as
# eval_archive.sh); SFORGE_RESOURCE_PROFILE=pyromind replaces them inside
# SForge with the task's resource-profile entry (Work 8/32g, Judge 56/88g for
# the seven Judge GPUs), and the dry run prints whichever applies.
#
# Agents: MLSYSFORGE_AGENT=codex (default; CODEX_AUTH_JSON_PATH, or an
# OpenAI-compatible MODEL_API_BASE_URL + mode-0600 MODEL_API_KEY_FILE relayed
# by the sidecar) or claude-code (CLAUDE_OAUTH_TOKEN_FILE).  The
# SFORGE_LLM_JUDGE_* integrity review must be configured and enforcing; the
# run refuses otherwise, before any lock, container or GPU work.
#
# KV_EVICTION_EVAL_ARCHIVE=<archive> runs no agent: it evaluates that archive
# through the same final-Judge path (same locks, Judge plan and timeouts),
# optionally after a public quick-suite smoke of the task oracle on the Work
# GPU (KV_EVICTION_PUBLIC_SMOKE=1).  This is the qualification path.
#
# Under SFORGE_RESOURCE_PROFILE=pyromind the submission cooldown (EVAL_INTERVAL)
# must equal the deployment's submission_cooldown.  FINAL_EVAL=0 skips the
# final Judge and records the run as failed (failure_stage final_disabled).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
shopt -s inherit_errexit

AGENT_ENTRY="${SCRIPT_DIR}/sforge_agent_entry.py"
TASK_JSON="${TASKS_DIR}/${TASK_ID}.json"
# The task's own restricted CONNECT sidecar (tests/test_restricted_proxy_public_egress.py).
PROXY_SOURCE="${SCRIPT_DIR}/restricted_proxy.py"
EVALUATION_FORWARD_PORT=3130
MODEL_FORWARD_PORT=3131
MODEL_MOUNT=/opt/models/qwen3-8b
WHOLE_GPU_UUID_RE='^GPU-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$'
# The verifier needs about 1.25 x 8 GB + 1 GiB of writable-layer scratch per Judge.
MIN_DOCKER_FREE_BYTES=$(( 12 * 1024 * 1024 * 1024 ))

# --- GPU helpers --------------------------------------------------------------

# Physical index for an index or whole-GPU UUID selector.
gpu_selector_index() {
  local selector="$1" index
  if [[ "${selector}" =~ ^[0-9]+$ ]]; then
    canonical_gpu_index "${selector}"
    return
  fi
  [[ "${selector}" =~ ^GPU-[A-Za-z0-9-]+$ ]] \
    || die "GPU selector must be a physical index or UUID: ${selector}"
  index="$(nvidia-smi --id="${selector}" --query-gpu=index --format=csv,noheader | xargs)"
  canonical_gpu_index "${index}"
}

# Both host lock files of a GPU: SForge's and the one the pair/Miles launchers use.
gpu_lock_files() {
  local uuid="$1" sforge_lock
  sforge_lock="$(gpu_lock_path "${uuid}")"
  printf '%s\n' "${sforge_lock}" "/tmp/mlsysforge-gpu-${uuid//\//-}.lock"
}

# Take both lock names of a GPU for the lifetime of this shell.  The files are
# opened for append (never truncated), must be regular files owned by this
# user (no symlink), and are taken non-blocking: a held lock fails the launch.
GPU_LOCK_FDS=()
hold_gpu_locks() {
  local uuid="$1" role="$2" path fd opened linked
  local -a paths=()
  [[ "${uuid}" =~ ${WHOLE_GPU_UUID_RE} ]] || die "${role} GPU has no whole-device UUID: ${uuid}"
  mapfile -t paths < <(gpu_lock_files "${uuid}")
  (( ${#paths[@]} == 2 )) || die "cannot name the ${role} GPU locks for ${uuid}"
  for path in "${paths[@]}"; do
    [[ ! -L "${path}" ]] || die "${role} GPU lock is a symlink: ${path}"
    exec {fd}>>"${path}" || die "cannot open ${role} GPU lock: ${path}"
    opened="$(stat -L -c '%d:%i:%u:%F' "/proc/self/fd/${fd}")"
    linked="$(stat -c '%d:%i:%u:%F' "${path}")"
    [[ "${opened}" == "${linked}" && "${opened}" == *":$(id -u):regular"* ]] \
      || die "${role} GPU lock is not a regular file owned by this user: ${path}"
    flock -n "${fd}" || die "${role} GPU ${uuid}: ${path} is held by another process"
    GPU_LOCK_FDS+=("${fd}")
  done
}

# Helper processes that are not GPU users (log tee, sidecar log tailer) must not
# keep the GPUs locked after this shell is gone.
close_gpu_lock_fds() {
  local fd
  for fd in "${GPU_LOCK_FDS[@]}"; do
    exec {fd}>&-
  done
}

# Lock state of both names of a GPU read from /proc/locks: "free", or each
# held/unsafe name ("mlsysforge held by pid 123").  Never takes or probes a
# lock: a probing flock could make a launcher starting at the same moment
# fail its own non-blocking flock.
passive_lock_state() {
  local uuid="$1"
  local -a paths=()
  [[ "${uuid}" =~ ${WHOLE_GPU_UUID_RE} ]] || { echo unknown; return 0; }
  mapfile -t paths < <(gpu_lock_files "${uuid}")
  python3 -B - "${paths[@]}" <<'PY'
import os, sys

states = []
try:
    with open("/proc/locks", encoding="ascii") as handle:
        rows = [line.split() for line in handle]
except OSError:
    rows = None
for path in sys.argv[1:]:
    name = os.path.basename(path).split("-gpu-", 1)[0]
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        continue
    if not os.path.isfile(path) or os.path.islink(path):
        states.append(f"{name} unsafe (not a regular file)")
        continue
    if rows is None:
        states.append(f"{name} unknown (/proc/locks unreadable)")
        continue
    key = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
    holders = sorted({row[4] for row in rows if len(row) >= 6 and row[1] != "->" and row[5] == key})
    if holders:
        states.append(f"{name} held by pid {','.join(holders)}")
print(", ".join(states) if states else "free")
PY
}

assert_no_compute_process() {
  local uuid="$1" role="$2" apps
  apps="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)" \
    || die "cannot list the compute processes on the ${role} GPUs"
  ! grep -Fxq "${uuid}" <<<"${apps}" \
    || die "${role} GPU ${uuid} already has a compute process"
}

assert_identical_gpus() {
  local role="$1" uuid row first=""
  shift
  for uuid in "$@"; do
    row="$(nvidia-smi --id="${uuid}" --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits)"
    [[ -z "${first}" || "${row}" == "${first}" ]] \
      || die "${role} GPUs are not identical: '${first}' vs '${row}' (${uuid})"
    first="${row}"
  done
}

# Containers that claim one of the given GPUs ("<index>=<uuid>" arguments)
# while running, paused, created or restarting: a created or paused container
# holds no compute process yet, so neither the locks nor nvidia-smi show it.
# One line per claim: <index> TAB <uuid> TAB <container id> TAB <exact> TAB
# "<name> (<state>): <how>".  <exact> is 1 when a DeviceRequest names the GPU's
# UUID itself, the way SForge, eval_archive.sh and this launcher request GPUs;
# 0 for an index, all GPUs, a count, NVIDIA_VISIBLE_DEVICES or /dev/nvidiaN.
# The rules are those of the SXM hosts' hostkit/gpu-guard.sh.  Returns 1 when
# Docker cannot be listed.
container_gpu_claims() {
  local ids inspect
  ids="$(docker ps -aq --no-trunc --filter status=running --filter status=paused \
    --filter status=created --filter status=restarting)" || return 1
  [[ -n "${ids// /}" ]] || return 0
  # A container can vanish between ps and inspect: inspect then exits 1 but
  # still prints the others.
  # shellcheck disable=SC2086
  inspect="$(docker inspect ${ids} 2>/dev/null)" || true
  [[ -n "${inspect}" ]] || return 1
  python3 -B -c '
import json, sys
gpus = [arg.split("=", 1) for arg in sys.argv[1:]]
for c in json.loads(sys.stdin.read() or "[]"):
    hc = c.get("HostConfig") or {}
    cid = c.get("Id") or "?"
    name = (c.get("Name") or "").lstrip("/") or cid[:12]
    state = (c.get("State") or {}).get("Status", "?")
    reqs = hc.get("DeviceRequests") or []
    env = dict(e.split("=", 1) for e in (c.get("Config") or {}).get("Env") or [] if "=" in e)
    devs = {d.get("PathOnHost") for d in hc.get("Devices") or []}
    for idx, uuid in gpus:
        why, exact = [], False
        for r in reqs:
            ids, count = r.get("DeviceIDs") or [], r.get("Count") or 0
            # plain IDs ("3", "GPU-...") and CDI names ("nvidia.com/gpu=3", "=GPU-...", "=all", MIG "=3:0")
            names = {i.rsplit("=", 1)[-1] for i in ids}
            names |= {n.split(":", 1)[0] for n in names if ":" in n}
            if ids and (uuid in names or idx in names or "all" in names):
                driver = r.get("Driver") or "nvidia"
                why.append(f"DeviceRequests {driver} DeviceIDs " + ",".join(ids))
                exact = exact or uuid in names
            elif not ids and count == -1:
                why.append("DeviceRequests Count=-1 (all GPUs)")
            elif not ids and count > 0:
                why.append(f"DeviceRequests Count={count} without DeviceIDs")
        if not reqs and hc.get("Runtime") not in ("runc", "io.containerd.runc.v2"):
            value = env.get("NVIDIA_VISIBLE_DEVICES")
            listed = [v.strip() for v in (value or "").split(",")]
            if "all" in listed or uuid in listed or idx in listed:
                why.append(f"Env NVIDIA_VISIBLE_DEVICES={value} and no DeviceRequest")
        if f"/dev/nvidia{idx}" in devs:
            why.append(f"Devices /dev/nvidia{idx}")
        if why:
            print(idx, uuid, cid, int(exact), f"container {name} ({state}): " + "; ".join(why), sep="\t")
' "$@" <<<"${inspect}"
}

# Rows of a GPU reservations file (GPU_RESERVATIONS_FILE: "host_id TAB gpus TAB
# until_utc TAB reason", '#' comments; the SXM hosts' configs/gpu-reservations.tsv)
# that are live for host $1 and GPU index $2: until_utc is '-' or still ahead.
# Prints one reason per live row.
gpu_reservations() {
  local file="$1" host="$2" index="$3" now rh rg ru rr until
  now="$(date -u +%s)"
  while IFS=$'\t' read -r rh rg ru rr; do
    [[ "${rh}" == "${host}" && ",${rg}," == *",${index},"* ]] || continue
    if [[ "${ru:--}" != - ]]; then
      until="$(date -u -d "${ru}" +%s 2>/dev/null)" \
        || { echo "reservation with an unreadable until_utc '${ru}': ${rr}"; continue; }
      (( until > now )) || continue
    fi
    echo "reserved until ${ru:--}: ${rr}"
  done < <(grep -v '^[[:space:]]*#' "${file}")
}

# Start a child that uses the GPUs (Judge server, sforge run, final Judge).  It
# inherits the lock descriptors, which is right while it may use the GPUs, so it
# must never outlive this shell: setpriv --pdeathsig sends it SIGNAL when the
# launcher dies (SIGKILL, the OOM killer).  It ignores SIGHUP: tmux
# kill-session and a lost pty hang up the whole process group, and the
# launcher's trap then stops it through its graceful signal instead.  SIGINT
# (SForge's graceful path) starts at its default, which an asynchronous command
# would otherwise inherit as ignored.  Usage: guarded_exec SIGNAL COMMAND...
# (it replaces the calling (sub)shell; run it as `guarded_exec ... &`).
guarded_exec() {
  local signal="$1"
  shift
  exec setpriv --pdeathsig "${signal}" -- env --default-signal=INT --ignore-signal=HUP "$@"
}

# stop_child PID SIGNAL SECONDS NAME: send SIGNAL unless it is "-" (already
# sent), give the child SECONDS to exit, then SIGKILL it.  Never waits unbounded.
stop_child() {
  local pid="$1" signal="$2" limit="$3" name="$4" waited=0
  [[ -n "${pid}" ]] || return 0
  [[ "${signal}" == - ]] || kill -s "${signal}" "${pid}" 2>/dev/null || true
  while kill -0 "${pid}" 2>/dev/null && (( waited < limit )); do
    sleep 1
    waited=$(( waited + 1 ))
  done
  if kill -0 "${pid}" 2>/dev/null; then
    echo "WARNING: ${name} (pid ${pid}) did not exit ${limit} s after it was told to stop; killing it"
    kill -s KILL "${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
}

# Sourced (tests): the helpers above only.
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  return 0
fi

# --- Settings -------------------------------------------------------------------

DRY_RUN="${SFORGE_LAUNCH_DRY_RUN:-0}"
EVAL_ARCHIVE="${KV_EVICTION_EVAL_ARCHIVE:-}"
PUBLIC_SMOKE="${KV_EVICTION_PUBLIC_SMOKE:-0}"
AGENT_NAME="${MLSYSFORGE_AGENT:-codex}"
MODEL="${MODEL:-}"
REASONING_EFFORT="${REASONING_EFFORT:-max}"
TASK_TIMEOUT="${TASK_TIMEOUT:-86400}"
# PyroMind deployment kvpress-v2: submission_cooldown 1800.
SUBMISSION_COOLDOWN="${EVAL_INTERVAL:-1800}"
WORK_GPU_SELECTOR="${WORK_GPU:-}"
JUDGE_GPU_SELECTORS="${JUDGE_GPUS:-}"
JUDGE_PORT="${JUDGE_PORT:-}"
AUTH_JSON="${CODEX_AUTH_JSON_PATH:-${HOME}/.codex/auth.json}"
CLAUDE_TOKEN_FILE="${CLAUDE_OAUTH_TOKEN_FILE:-}"
MODEL_API_BASE_URL="${MODEL_API_BASE_URL:-}"
MODEL_API_KEY_FILE="${MODEL_API_KEY_FILE:-}"
CODEX_CLI_VERSION="${CODEX_CLI_VERSION:-}"
FINAL_EVAL="${FINAL_EVAL:-1}"
# The revision every kvpress v2 score so far was measured on (PyroMind asset
# identity); another snapshot is a different measurement.
MODEL_REVISION="${KV_EVICTION_MODEL_REVISION:-b968826d9c46dd6066d109eabc6255188de91218}"

[[ "${DRY_RUN}" == 0 || "${DRY_RUN}" == 1 ]] || die "SFORGE_LAUNCH_DRY_RUN must be 0 or 1"
[[ "${FINAL_EVAL}" == 0 || "${FINAL_EVAL}" == 1 ]] || die "FINAL_EVAL must be 0 or 1"
[[ "${PUBLIC_SMOKE}" == 0 || "${PUBLIC_SMOKE}" == 1 ]] || die "KV_EVICTION_PUBLIC_SMOKE must be 0 or 1"
EVAL_MODE=0
if [[ -n "${EVAL_ARCHIVE}" ]]; then
  EVAL_MODE=1
  [[ "${EVAL_ARCHIVE}" == /* ]] || die "KV_EVICTION_EVAL_ARCHIVE must be an absolute path"
else
  (( ! PUBLIC_SMOKE )) || die "KV_EVICTION_PUBLIC_SMOKE=1 needs KV_EVICTION_EVAL_ARCHIVE"
fi
[[ -n "${WORK_GPU_SELECTOR}" ]] || die "set WORK_GPU to the Work GPU (index or UUID)"
[[ -n "${JUDGE_GPU_SELECTORS}" ]] || die "set JUDGE_GPUS to the Judge GPUs, e.g. JUDGE_GPUS=1,2,3,4,5,6,7"
[[ "${MODEL_REVISION}" =~ ^[0-9a-f]{40,64}$ ]] || die "KV_EVICTION_MODEL_REVISION must be a snapshot hash"
for name in SFORGE_SINGLE_GPU_HANDOFF SFORGE_ACCELERATOR_DEVICE_UUID SFORGE_ACCELERATOR_ALLOCATION_KEY \
    SFORGE_ACCELERATOR_OBSERVER SFORGE_TRUSTED_LEASE_ID; do
  [[ -z "${!name:-}" || "${!name}" == 0 ]] \
    || die "${name} belongs to the single-allocation launchers; this launcher holds whole GPUs by lock (unset it)"
done
[[ "${TASK_TIMEOUT}" =~ ^[1-9][0-9]*$ ]] || die "TASK_TIMEOUT must be a positive integer"
[[ "${SUBMISSION_COOLDOWN}" =~ ^[0-9]+$ ]] || die "EVAL_INTERVAL (the submission cooldown) must be a non-negative integer"
SUBMISSION_COOLDOWN="$(( 10#${SUBMISSION_COOLDOWN} ))"
if (( ! EVAL_MODE )); then
  [[ -n "${MODEL}" ]] || die "set MODEL (launch/kvpress-kv-eviction-scratch/run.sh --model)"
  [[ "${MODEL}" =~ ^[A-Za-z0-9][A-Za-z0-9_./:-]*$ ]] || die "unsafe MODEL id: ${MODEL}"
  [[ "${JUDGE_PORT}" =~ ^[0-9]+$ ]] || die "set JUDGE_PORT to a run-unique TCP port"
  JUDGE_PORT="$((10#${JUDGE_PORT}))"
  (( JUDGE_PORT >= 1024 && JUDGE_PORT <= 65535 && JUDGE_PORT != 3128 \
     && JUDGE_PORT != EVALUATION_FORWARD_PORT && JUDGE_PORT != MODEL_FORWARD_PORT )) \
    || die "JUDGE_PORT must be 1024..65535 and not a sidecar port (3128, 3130, 3131)"
  case "${AGENT_NAME}" in
    codex)
      case "${REASONING_EFFORT}" in max|ultra|xhigh|high|medium|low) ;;
        *) die "unsupported Codex reasoning effort: ${REASONING_EFFORT}" ;; esac
      ;;
    claude-code)
      [[ -z "${MODEL_API_BASE_URL}" ]] || die "Claude Code uses its OAuth token; unset MODEL_API_BASE_URL"
      case "${REASONING_EFFORT}" in max|xhigh|high|medium|low) ;;
        *) die "Claude Code efforts are low|medium|high|xhigh|max" ;; esac
      ;;
    *) die "MLSYSFORGE_AGENT must be codex or claude-code" ;;
  esac
fi
[[ "$(jq -r '.internet' "${TASK_JSON}")" == true ]] \
  || die "this launcher assumes the task JSON's internet: true"
task_eval_timeout="$(jq -er '.judge.eval_timeout' "${TASK_JSON}")"

# A label-safe short model name: my-model-1 -> my-model-1, ZHIPU/GLM-5.3-Flash -> zhipu-glm-5.3-flash.
if (( EVAL_MODE )); then
  archive_label="$(basename -- "${EVAL_ARCHIVE}")"
  archive_label="${archive_label%.tar.gz}"
  archive_label="$(sed -E 's/[^A-Za-z0-9_.-]+/-/g; s/^-+//; s/-+$//' <<<"${archive_label}")"
  RUN_LABEL="${RUN_LABEL:-eval-${archive_label:-archive}}"
else
  model_short="$(tr 'A-Z' 'a-z' <<<"${MODEL}" | sed -E 's/[^a-z0-9.]+/-/g; s/^-+//; s/-+$//')"
  budget_label="$(( TASK_TIMEOUT / 3600 ))h"
  (( TASK_TIMEOUT % 3600 == 0 )) || budget_label="${TASK_TIMEOUT}s"
  RUN_LABEL="${RUN_LABEL:-${model_short}-${REASONING_EFFORT}-${budget_label}}"
fi
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
RUN_ID="${RUN_ID:-kvpress-kv-eviction-scratch-v2-${RUN_LABEL}-${RUN_STAMP}}"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ && ${#RUN_ID} -le 120 ]] \
  || die "unsafe or oversized RUN_ID: ${RUN_ID}"
RUN_ROOT="${LOGS_DIR}/runs/${RUN_ID}"
FINAL_RUN_ID="${RUN_ID}-final"

# --- GPUs -----------------------------------------------------------------------
check_failures=()
fail_check() { check_failures+=("$*"); }

(( DRY_RUN )) || require_command nvidia-smi
judge_uuids=(); judge_indices=()
gpus_resolved=0
if nvidia-smi -L >/dev/null 2>&1; then
  gpus_resolved=1
  work_index="$(gpu_selector_index "${WORK_GPU_SELECTOR}")"
  work_uuid="$(gpu_uuid "${work_index}")"
  [[ "${work_uuid}" =~ ${WHOLE_GPU_UUID_RE} ]] || die "Work GPU ${work_index} has no whole-device UUID"
  IFS=, read -r -a judge_selectors <<<"${JUDGE_GPU_SELECTORS// /}"
  (( ${#judge_selectors[@]} >= 1 && ${#judge_selectors[@]} <= 8 )) || die "JUDGE_GPUS must name 1 to 8 GPUs"
  declare -A seen_judge=()
  for selector in "${judge_selectors[@]}"; do
    index="$(gpu_selector_index "${selector}")"
    uuid="$(gpu_uuid "${index}")"
    [[ "${uuid}" =~ ${WHOLE_GPU_UUID_RE} ]] || die "GPU ${index} has no whole-device UUID"
    [[ -z "${seen_judge[${uuid}]:-}" ]] || die "JUDGE_GPUS names GPU ${index} twice"
    [[ "${uuid}" != "${work_uuid}" ]] || die "the Work GPU ${work_index} is also a Judge GPU"
    seen_judge[${uuid}]=1
    judge_indices+=("${index}"); judge_uuids+=("${uuid}")
  done
else
  # Only a dry run may continue: it shows the plan with unresolved UUIDs.
  (( DRY_RUN )) || die "nvidia-smi cannot enumerate the GPUs"
  fail_check "nvidia-smi cannot enumerate the GPUs; UUIDs, idle state and locks are unresolved"
  work_index="${WORK_GPU_SELECTOR}"; work_uuid="(unresolved)"
  IFS=, read -r -a judge_indices <<<"${JUDGE_GPU_SELECTORS// /}"
  (( ${#judge_indices[@]} >= 1 && ${#judge_indices[@]} <= 8 )) || die "JUDGE_GPUS must name 1 to 8 GPUs"
  declare -A seen_judge=()
  for selector in "${judge_indices[@]}"; do
    [[ "${selector}" != "${WORK_GPU_SELECTOR}" ]] || die "the Work GPU ${selector} is also a Judge GPU"
    [[ -z "${seen_judge[${selector}]:-}" ]] || die "JUDGE_GPUS names GPU ${selector} twice"
    seen_judge[${selector}]=1
    judge_uuids+=("(unresolved)")
  done
fi
n_judge="${#judge_uuids[@]}"

# Lock and idle state, read without touching a lock or a GPU context.
declare -A gpu_state=()
if (( gpus_resolved )); then
  minimum_free="${GPU_MIN_FREE_MIB:-${H100_MIN_FREE_MIB:-43008}}"
  [[ "${minimum_free}" =~ ^[0-9]+$ ]] || die "GPU_MIN_FREE_MIB must be a non-negative integer"
  busy_uuids="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null)" \
    || fail_check "cannot list the GPU compute processes"
  all_indices=("${work_index}" "${judge_indices[@]}")
  all_uuids=("${work_uuid}" "${judge_uuids[@]}")
  for i in "${!all_uuids[@]}"; do
    uuid="${all_uuids[$i]}"; label="GPU ${all_indices[$i]} (${uuid})"
    locks="$(passive_lock_state "${uuid}")"
    free_mib="$(nvidia-smi --id="${uuid}" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')" || free_mib=""
    gpu_state[${uuid}]="locks ${locks}; ${free_mib:-?} MiB free"
    [[ "${locks}" == free ]] || fail_check "${label}: lock ${locks}"
    if grep -Fxq "${uuid}" <<<"${busy_uuids:-}"; then
      gpu_state[${uuid}]+="; compute process present"
      fail_check "${label} already has a compute process"
    fi
    [[ "${free_mib}" =~ ^[0-9]+$ ]] && (( free_mib >= minimum_free )) \
      || fail_check "${label} has ${free_mib:-unknown} MiB free (needs ${minimum_free})"
  done
fi

# --- Judge plan: verifier deadlines, Judge timeout, CPU and memory -------------
read -r t_candidate t_total t_judge t_eval <<<"$(judge_timeouts "${n_judge}")"
t_candidate="${KV_EVICTION_CANDIDATE_TIMEOUT:-${t_candidate}}"
t_total="${KV_EVICTION_TOTAL_TIMEOUT:-${t_total}}"
t_judge="${KV_EVICTION_JUDGE_TIMEOUT:-${t_judge}}"
t_eval="${FINAL_EVAL_TIMEOUT:-${t_eval}}"
for value in "${t_candidate}" "${t_total}" "${t_judge}" "${t_eval}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "Judge timeouts must be positive integers (got ${value})"
done
(( t_candidate <= t_total && t_total < t_judge && t_judge < t_eval )) \
  || die "Judge timeouts must satisfy candidate <= total < Judge < eval (${t_candidate} ${t_total} ${t_judge} ${t_eval})"
(( t_eval >= task_eval_timeout )) \
  || fail_check "the Judge timeout ${t_eval} s is below the task JSON's judge.eval_timeout ${task_eval_timeout} s"
judge_extra_env="KV_EVICTION_CANDIDATE_TIMEOUT=${t_candidate},KV_EVICTION_TOTAL_TIMEOUT=${t_total},KV_EVICTION_JUDGE_TIMEOUT=${t_judge}"

# The sizes this launcher passes (the task's own launcher sizing) ...
work_cpu="${WORK_CPU_LIMIT:-$(jq -er '.work.cpu_limit' "${TASK_JSON}")}"
work_mem="${WORK_MEM_LIMIT:-$(jq -er '.work.mem_limit' "${TASK_JSON}")}"
judge_cpu="${JUDGE_CPU_LIMIT:-$(( 8 * n_judge ))}"
judge_mem="${JUDGE_MEM_LIMIT:-$(( 32 + 8 * n_judge ))g}"
for value in "${work_cpu}" "${judge_cpu}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "CPU limits must be positive integers (got ${value})"
done
for value in "${work_mem}" "${judge_mem}"; do
  [[ "${value}" =~ ^[1-9][0-9]*[gm]$ ]] || die "memory limits look like 32g (got ${value})"
done
# ... and the sizes SForge applies: the deployment profile replaces them in
# SFORGE_RESOURCE_PROFILE=pyromind mode (sforge/harness/resource_profile.py).
profile_mode="?"; profile_source=""; profile_sha256=""; profile_cooldown=""
effective_work_cpu="${work_cpu}"; effective_work_mem="${work_mem}"
effective_judge_cpu="${judge_cpu}"; effective_judge_mem="${judge_mem}"
if profile_plan="$(
  cd -- "${REPOSITORY_ROOT}" && "${SFORGE_PYTHON}" -B - "${TASK_ID}" <<'PY' 2>&1
import json, sys
from sforge.harness.config import load_config
from sforge.harness.resource_profile import REPO, mode_of, profile_for_task
config = load_config()
mode = mode_of(config)
profile = None
if mode == "pyromind" and (config.backend or "docker") == "docker":
    profile = profile_for_task(sys.argv[1])
if profile is None or profile.exempt:
    print(mode, "-")
else:
    deployment = json.loads((REPO / profile.source).read_text(encoding="utf-8"))
    cooldown = deployment.get("submission_cooldown")
    print(mode, profile.work.cpu, profile.work.memory, profile.judge.cpu, profile.judge.memory,
          profile.source, profile.sha256, cooldown if isinstance(cooldown, int) else "-")
PY
)"; then
  read -r profile_mode p_work_cpu p_work_mem p_judge_cpu p_judge_mem profile_source profile_sha256 \
    profile_cooldown <<<"${profile_plan##*$'\n'}"
  if [[ "${p_work_cpu}" != "-" ]]; then
    effective_work_cpu="${p_work_cpu}"; effective_work_mem="${p_work_mem}"
    effective_judge_cpu="${p_judge_cpu}"; effective_judge_mem="${p_judge_mem}"
  fi
else
  fail_check "cannot resolve the SForge resource profile: ${profile_plan##*$'\n'}"
fi
# A pyromind-profile run is the PyroMind deployment run locally: its cooldown is
# the deployment's submission_cooldown, not whatever EVAL_INTERVAL the calling
# shell still exports (the eagle3, sage and lmcache launchers read it too).  A
# checkout without deploy/ sizes from the published table, which carries no
# cooldown; the default EVAL_INTERVAL (1800 s) is the deployment's.
if (( ! EVAL_MODE )) && [[ "${profile_mode}" == pyromind && -n "${profile_source}" \
      && "${profile_source}" != sforge/harness/resource_profiles.json ]]; then
  if [[ ! "${profile_cooldown}" =~ ^[0-9]+$ ]]; then
    fail_check "${profile_source} has no integer submission_cooldown"
  elif (( SUBMISSION_COOLDOWN != profile_cooldown )); then
    fail_check "the submission cooldown ${SUBMISSION_COOLDOWN} s (EVAL_INTERVAL) differs from ${profile_source} submission_cooldown ${profile_cooldown} s"
  fi
fi
if [[ -n "${profile_source}" ]]; then
  size_origin="resource profile ${profile_mode}: ${profile_source} sha256 ${profile_sha256:0:12}"
else
  size_origin="resource profile ${profile_mode}: the task launcher's own sizes"
fi
(( effective_judge_cpu >= n_judge )) \
  || fail_check "the Judge needs at least one CPU per Judge GPU (${effective_judge_cpu} CPU for ${n_judge} GPUs)"

# --- Agent credentials and runtime ---------------------------------------------
model_api_host=""; model_api_port=""; model_api_path=""
credential="(none: eval-only)"
AGENT_RUNTIME_DIR="${SFORGE_AGENT_RUNTIME_DIR:-}"
runtime_agent=""; runtime_version=""; runtime_tree_sha256=""
if (( ! EVAL_MODE )); then
  if [[ -n "${MODEL_API_BASE_URL}" ]]; then
    if [[ "${MODEL_API_BASE_URL}" =~ ^http://([^/:[:space:]]+):([0-9]+)(/.*)?$ ]]; then
      model_api_host="${BASH_REMATCH[1]}"; model_api_port="${BASH_REMATCH[2]}"
      model_api_path="${BASH_REMATCH[3]:-}"
    else
      fail_check "MODEL_API_BASE_URL must be http://HOST:PORT[/path] (the sidecar relays raw TCP)"
    fi
    if [[ -z "${MODEL_API_KEY_FILE}" ]]; then
      fail_check "set MODEL_API_KEY_FILE with MODEL_API_BASE_URL"
    else
      ( assert_private_auth_file "${MODEL_API_KEY_FILE}" ) 2>/dev/null \
        || fail_check "MODEL_API_KEY_FILE must be a nonempty mode-600 file owned by this user"
    fi
    credential="model API ${MODEL_API_BASE_URL} (key file ${MODEL_API_KEY_FILE})"
  elif [[ "${AGENT_NAME}" == claude-code ]]; then
    [[ -n "${CLAUDE_TOKEN_FILE}" && -f "${CLAUDE_TOKEN_FILE}" && ! -L "${CLAUDE_TOKEN_FILE}" \
       && "$(stat -c '%a' "${CLAUDE_TOKEN_FILE}")" == 600 ]] \
      || fail_check "set CLAUDE_OAUTH_TOKEN_FILE to a mode-600 regular token file"
    credential="Claude OAuth token file ${CLAUDE_TOKEN_FILE}"
  else
    ( assert_private_auth_file "${AUTH_JSON}" ) 2>/dev/null \
      || fail_check "Codex auth file must be a nonempty mode-600 file owned by this user: ${AUTH_JSON}"
    credential="Codex auth ${AUTH_JSON}"
  fi

  runtime_marker="${AGENT_RUNTIME_DIR}/.sforge-agent-runtime.json"
  if [[ -n "${AGENT_RUNTIME_DIR}" && -d "${AGENT_RUNTIME_DIR}" && ! -L "${AGENT_RUNTIME_DIR}" \
        && -f "${runtime_marker}" ]] \
     && [[ "$(jq -r '.kind' "${runtime_marker}")" == sforge-agent-runtime ]]; then
    runtime_agent="$(jq -r '.agent' "${runtime_marker}")"
    runtime_version="$(jq -r '.version' "${runtime_marker}")"
    runtime_tree_sha256="$(jq -r '.tree_sha256' "${runtime_marker}")"
    [[ "${runtime_agent}" == "${AGENT_NAME}" ]] \
      || fail_check "SFORGE_AGENT_RUNTIME_DIR is a ${runtime_agent} runtime, not ${AGENT_NAME}"
    if [[ "${AGENT_NAME}" == codex ]]; then
      CODEX_CLI_VERSION="${CODEX_CLI_VERSION:-${runtime_version}}"
      [[ "${runtime_version}" == "${CODEX_CLI_VERSION}" ]] \
        || fail_check "Codex runtime ${runtime_version} differs from CODEX_CLI_VERSION ${CODEX_CLI_VERSION}"
    fi
  else
    fail_check "SFORGE_AGENT_RUNTIME_DIR is missing or unmarked: ${AGENT_RUNTIME_DIR:-<unset>}"
  fi
  [[ -n "${SFORGE_VERSION_STORE_ROOT:-}" ]] \
    || fail_check "set SFORGE_VERSION_STORE_ROOT (launch/kvpress-kv-eviction-scratch/run.sh resolves it)"
  if ss -H -ltn "sport = :${JUDGE_PORT}" 2>/dev/null | grep -q .; then
    fail_check "JUDGE_PORT ${JUDGE_PORT} is already bound on this host"
  fi
  [[ -f "${PROXY_SOURCE}" && ! -L "${PROXY_SOURCE}" ]] || fail_check "missing Work sidecar: ${PROXY_SOURCE}"
else
  [[ -f "${EVAL_ARCHIVE}" && ! -L "${EVAL_ARCHIVE}" && -s "${EVAL_ARCHIVE}" ]] \
    || fail_check "KV_EVICTION_EVAL_ARCHIVE is not a nonempty regular file: ${EVAL_ARCHIVE}"
fi

# --- Integrity review: mandatory and enforcing (docs/LLM_JUDGE.md) ------------
[[ "${SFORGE_LLM_JUDGE_MODE:-}" == enforce ]] \
  || fail_check "set SFORGE_LLM_JUDGE_MODE=enforce and the SFORGE_LLM_JUDGE_* reviewer settings"
review_summary=""
if [[ -x "${SFORGE_PYTHON}" ]]; then
  if ! review_summary="$(
    cd -- "${REPOSITORY_ROOT}" && "${SFORGE_PYTHON}" -B - <<'PY' 2>&1
from sforge.harness.llm_judge.config import load_llm_judge_config
config = load_llm_judge_config()
config.check_ready()
print(f"{config.mode} {config.engine} {config.agent} {config.model} "
      f"effort={config.reasoning_effort} runtime={config.agent_runtime_dir}")
PY
  )"; then
    fail_check "integrity review is not ready: ${review_summary##*$'\n'}"
    review_summary=""
  fi
else
  fail_check "missing SForge Python: ${SFORGE_PYTHON}"
fi

# --- Model, images, Docker ------------------------------------------------------
model_revision=""
if model_check="$(validate_model_repo "${KV_EVICTION_MODEL_REPO}" 2>&1)"; then
  model_revision="${model_check##*$'\n'}"
  [[ "${model_revision}" == "${MODEL_REVISION}" ]] \
    || fail_check "KV_EVICTION_MODEL_REPO is at revision ${model_revision}, not the pinned ${MODEL_REVISION} (KV_EVICTION_MODEL_REVISION)"
else
  fail_check "${model_check##*$'\n'}"
fi
[[ ! -e "${RUN_ROOT}" ]] || fail_check "run directory already exists: ${RUN_ROOT}"
docker_ok=0
if docker info >/dev/null 2>&1; then
  docker_ok=1
  docker_root="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null)" || docker_root=""
  docker_free="$(df -P -B1 -- "${docker_root:-/var/lib/docker}" 2>/dev/null | awk 'NR == 2 {print $4}')" || docker_free=""
  if [[ ! "${docker_free}" =~ ^[0-9]+$ ]]; then
    fail_check "cannot read the free space of the Docker root ${docker_root:-<unknown>}"
  elif (( docker_free < MIN_DOCKER_FREE_BYTES )); then
    fail_check "Docker root ${docker_root} has $(( docker_free >> 30 )) GiB free; the Judge scratch needs 12 GiB"
  fi
else
  fail_check "this shell cannot reach the Docker daemon; launch from a docker-group session (sg docker -c ...)"
fi
# Containers that claim a GPU without a lock or a compute process yet (created,
# paused, or started by a launcher that does not take the locks).
gpu_pairs=()
if (( gpus_resolved )); then
  for i in "${!all_uuids[@]}"; do gpu_pairs+=("${all_indices[$i]}=${all_uuids[$i]}"); done
fi
if (( gpus_resolved && docker_ok )); then
  if claims="$(container_gpu_claims "${gpu_pairs[@]}")"; then
    while IFS=$'\t' read -r c_index c_uuid _ _ c_how; do
      [[ -n "${c_uuid}" ]] || continue
      gpu_state[${c_uuid}]+="; ${c_how%%:*}"
      fail_check "GPU ${c_index} (${c_uuid}): ${c_how}"
    done <<<"${claims}"
  else
    fail_check "cannot list the Docker containers to check their GPU claims"
  fi
fi
# Optional host reservations (GPU_RESERVATIONS_FILE + GPU_RESERVATIONS_HOST_ID).
reservations="${GPU_RESERVATIONS_FILE:-}"
if [[ -n "${reservations}" ]]; then
  if [[ -z "${GPU_RESERVATIONS_HOST_ID:-}" ]]; then
    fail_check "GPU_RESERVATIONS_FILE needs GPU_RESERVATIONS_HOST_ID (this host's id in that file)"
  elif [[ ! -f "${reservations}" || ! -r "${reservations}" ]]; then
    fail_check "cannot read GPU_RESERVATIONS_FILE ${reservations}"
  elif (( gpus_resolved )); then
    for i in "${!all_uuids[@]}"; do
      while IFS= read -r why; do
        [[ -n "${why}" ]] || continue
        gpu_state[${all_uuids[$i]}]+="; reserved"
        fail_check "GPU ${all_indices[$i]} (${all_uuids[$i]}): ${why} (${reservations})"
      done < <(gpu_reservations "${reservations}" "${GPU_RESERVATIONS_HOST_ID}" "${all_indices[$i]}")
    done
  fi
fi
# The GPU-using children start through guarded_exec.
setpriv --pdeathsig TERM -- env --default-signal=INT --ignore-signal=HUP true >/dev/null 2>&1 \
  || fail_check "needs setpriv --pdeathsig (util-linux) and env --default-signal/--ignore-signal (coreutils 8.31+)"
image_state="current"
if ! image_check="$( (assert_built_images_current) 2>&1 )"; then
  image_state="NOT current: ${image_check##*$'\n'}"
  fail_check "images are not current: ${image_check##*$'\n'} (scripts/build_images.sh)"
fi

# --- CPU pinning ------------------------------------------------------------------
work_cpuset_policy="${SFORGE_WORK_CPUSET:-${SFORGE_CPUSET:-off}}"
judge_cpuset="(unresolved)"
if (( gpus_resolved )); then
  if judge_cpuset="$(
    cd -- "${REPOSITORY_ROOT}" && "${SFORGE_PYTHON}" -B - "${SCRIPT_DIR}" "${judge_uuids[@]}" <<'PY' 2>&1
import sys
sys.path.insert(0, sys.argv[1])
import sforge_entry
cpus = [int(cpu) for cpu in sforge_entry.judge_cpuset(sys.argv[2:]).split(",") if cpu]
ranges = []
for cpu in cpus:
    if ranges and cpu == ranges[-1][1] + 1:
        ranges[-1][1] = cpu
    else:
        ranges.append([cpu, cpu])
print(",".join(f"{a}-{b}" if a != b else str(a) for a, b in ranges) or "(none: no sysfs GPU locality)")
PY
  )"; then
    judge_cpuset="${judge_cpuset##*$'\n'}"
  else
    judge_cpuset="(unavailable: ${judge_cpuset##*$'\n'})"
  fi
fi

if (( DRY_RUN )); then
  echo "kvpress v2 agent run plan (dry run: no lock, container or GPU work)"
  echo "  run id:        ${RUN_ID}"
  echo "  run root:      ${RUN_ROOT}"
  if (( EVAL_MODE )); then
    echo "  mode:          eval-only ${EVAL_ARCHIVE}$( (( PUBLIC_SMOKE )) && echo ' + public smoke (task oracle, quick suite, Work GPU)')"
  else
    echo "  mode:          agent run"
    echo "  agent:         ${AGENT_NAME} model=${MODEL} effort=${REASONING_EFFORT}"
    echo "  credential:    ${credential}"
    echo "  runtime:       ${AGENT_RUNTIME_DIR:-<unset>} (${runtime_agent:-?} ${runtime_version:-?})"
    echo "  budget:        ${TASK_TIMEOUT} s, submission cooldown ${SUBMISSION_COOLDOWN} s, auto-eval off"
  fi
  echo "  work GPU:      ${work_index} ${work_uuid} [${gpu_state[${work_uuid}]:-state unresolved}]"
  for i in "${!judge_uuids[@]}"; do
    echo "  judge GPU:     ${judge_indices[$i]} ${judge_uuids[$i]} [${gpu_state[${judge_uuids[$i]}]:-state unresolved}]"
  done
  echo "  lock names:    /tmp/sforge-gpu-<uuid>.lock and /tmp/mlsysforge-gpu-<uuid>.lock, every GPU, whole run"
  echo "  reservations:  ${reservations:-none (GPU_RESERVATIONS_FILE unset)}${reservations:+ host ${GPU_RESERVATIONS_HOST_ID:-?}}"
  echo "  work limits:   ${effective_work_cpu} CPU, ${effective_work_mem} (${size_origin})"
  echo "  judge plan:    ${n_judge} GPU(s), ${effective_judge_cpu} CPU, ${effective_judge_mem} (${size_origin})"
  echo "  timeouts:      candidate ${t_candidate} s, total ${t_total} s, Judge ${t_judge} s, eval ${t_eval} s (lib.sh judge_timeouts ${n_judge}; task JSON eval_timeout ${task_eval_timeout} s)"
  echo "  judge env:     SFORGE_JUDGE_EXTRA_ENV=${judge_extra_env}"
  if (( ! EVAL_MODE )); then
    echo "  judge port:    ${JUDGE_PORT} (Docker bridge gateway only)"
    echo "  network:       internal Work network + restricted sidecar, public egress 80/443${model_api_host:+, model relay ${MODEL_FORWARD_PORT} -> ${model_api_host}:${model_api_port}}"
  fi
  echo "  model:         ${KV_EVICTION_MODEL_REPO} rev ${model_revision:-?} -> ${MODEL_MOUNT} (read-only, Work and Judge)"
  echo "  review:        ${review_summary:-<not ready>}"
  echo "  images:        ${image_state}"
  (( EVAL_MODE )) || echo "  version store: ${SFORGE_VERSION_STORE_ROOT:-<unset>}"
  if (( EVAL_MODE )) || [[ "${FINAL_EVAL}" == 1 ]]; then
    echo "  final Judge:   on the same Judge GPUs under the held locks -> ${RUN_ROOT}/final-eval"
  else
    echo "  final Judge:   off"
  fi
  echo "  cpuset:        work=${work_cpuset_policy} judge=${judge_cpuset} (task adapter, GPU-local blocks)"
  if (( ${#check_failures[@]} )); then
    printf '  CHECK FAILED:  %s\n' "${check_failures[@]}"
    exit 1
  fi
  echo "  checks:        PASS"
  exit 0
fi
if (( ${#check_failures[@]} )); then
  printf 'ERROR: %s\n' "${check_failures[@]}" >&2
  exit 1
fi

# --- Live run ---------------------------------------------------------------------
for command in curl docker df env flock jq python3 setpriv sha256sum ss stat tee; do
  require_command "${command}"
done
(( docker_ok )) || die "this shell cannot reach the Docker daemon; launch from a docker-group session"
require_sforge
assert_built_images_current

for uuid in "${work_uuid}" "${judge_uuids[@]}"; do
  hold_gpu_locks "${uuid}" "$([[ "${uuid}" == "${work_uuid}" ]] && echo work || echo judge)"
done
assert_gpu_idle "${work_uuid}" work
assert_no_compute_process "${work_uuid}" work
for uuid in "${judge_uuids[@]}"; do
  assert_gpu_idle "${uuid}" judge
  assert_no_compute_process "${uuid}" judge
done
assert_identical_gpus judge "${judge_uuids[@]}"
# Again under the locks: a container may have claimed a GPU since the check.
claims="$(container_gpu_claims "${gpu_pairs[@]}")" \
  || die "cannot list the Docker containers to check their GPU claims"
if [[ -n "${claims}" ]]; then
  while IFS=$'\t' read -r c_index c_uuid _ _ c_how; do
    echo "ERROR: GPU ${c_index} (${c_uuid}): ${c_how}" >&2
  done <<<"${claims}"
  die "a container claims a GPU of this run"
fi

# Private Judge outputs refuse a group/world-writable parent, so the run root
# must not inherit a permissive umask (e.g. 0002).
mkdir -p "${LOGS_DIR}/runs"
mkdir "${RUN_ROOT}"
chmod 0755 "${RUN_ROOT}"
# tee ignores the hangup and the stop signals: tmux kill-session or a lost pty
# hang up the whole process group, and the cleanup below must still reach
# launcher.log.  It exits at EOF, once this shell and its children are gone.
exec > >(close_gpu_lock_fds; trap '' HUP INT TERM; exec tee -a "${RUN_ROOT}/launcher.log") 2>&1
echo "[$(date -u +%FT%TZ)] ${RUN_ID}: GPUs locked (work ${work_index}, judge ${judge_indices[*]})"

network_name="${RUN_ID}-internal"
proxy_name="${RUN_ID}-proxy"
smoke_name="${RUN_ID}-public-smoke"
# SForge's Work container of this run (sforge/harness/run_agent.py).
work_container="sforge.run.${TASK_ID}.${RUN_ID}"
mode="agent"
(( ! EVAL_MODE )) || mode="eval-only"
# The GPU users this shell started (each through guarded_exec) and the sidecar
# log tailer; empty once reaped.
judge_pid=""
run_pid=""
final_pid=""
proxy_log_pid=""
run_rc=""
final_rc=""
completion_written=0
# Containers and networks this run created and still owns.
owned_containers=()
owned_network=""

# Remove every container of this run: the named ones (SForge's Work container,
# the sidecar, the public smoke), any container whose DeviceRequest names one of
# this run's GPU UUIDs (this shell holds both locks of every one of them, so such
# a container is this run's: an interactive or final Judge, or a Work container
# SForge left behind), and anything still attached to the Work network.  Prints
# what it removed.
remove_run_containers() {
  local name claims id exact how
  local -A seen=()
  for name in "${work_container}" "${owned_containers[@]}"; do
    docker rm -f "${name}" >/dev/null 2>&1 && echo "removed container ${name}"
  done
  if claims="$(container_gpu_claims "${gpu_pairs[@]}")"; then
    while IFS=$'\t' read -r _ _ id exact how; do
      [[ "${exact}" == 1 && -z "${seen[${id}]:-}" ]] || continue
      seen[${id}]=1
      if docker rm -f "${id}" >/dev/null 2>&1; then
        echo "removed ${how%%:*}, which held a GPU of this run"
      else
        echo "WARNING: could not remove ${how%%:*}, which holds a GPU of this run"
      fi
    done <<<"${claims}"
  else
    echo "WARNING: cannot list the Docker containers to find this run's GPU claims"
  fi
  if [[ -n "${owned_network}" ]]; then
    for id in $(docker ps -aq --filter "network=${owned_network}" 2>/dev/null); do
      docker rm -f "${id}" >/dev/null 2>&1 && echo "removed container ${id:0:12} (on ${owned_network})"
    done
  fi
  return 0
}

# The given GPUs that still have a compute process after up to SECONDS (one
# UUID per line).  Returns 1 when nvidia-smi cannot tell.
gpus_still_busy() {
  local limit="$1" apps uuid waited=0
  local -a busy=()
  shift
  while :; do
    apps="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null)" || return 1
    busy=()
    for uuid in "$@"; do
      grep -Fxq "${uuid}" <<<"${apps}" && busy+=("${uuid}")
    done
    (( ${#busy[@]} )) || return 0
    (( waited < limit )) || break
    sleep 1
    waited=$(( waited + 1 ))
  done
  printf '%s\n' "${busy[@]}"
}

# --- Final Judge and completion record ----------------------------------------------
# The scripts/eval_archive.sh invocation without that script's own locking:
# this shell already holds every lock, and re-locking would conflict with it.
# The eval runs as a child of this shell (guarded_exec), so a signal reaches the
# cleanup at once and the eval cannot outlive the launcher.  Whatever the eval
# does (a crash or a timeout can leave its Judge container behind), every
# container of this run is removed and the release barrier runs on every Judge
# GPU before this returns.  Sets final_rc.
run_final_judge() {
  local archive="$1" log="${RUN_ROOT}/final-eval.log" rc=0 uuid
  {
    echo "Judge plan: ${n_judge} GPU(s), ${effective_judge_cpu} CPU, ${effective_judge_mem} (${size_origin})"
    echo "Judge deadlines: ${judge_extra_env}; eval timeout ${t_eval} s"
  } >"${log}"
  set +e
  (
    set -e
    assert_gpu_released "${work_uuid}"
    for uuid in "${judge_uuids[@]}"; do
      assert_gpu_released "${uuid}"
      assert_gpu_idle "${uuid}" "final Judge"
      assert_no_compute_process "${uuid}" "final Judge"
    done
  ) >>"${log}" 2>&1
  rc=$?
  set -e
  if (( rc == 0 )); then
    (
      unset SFORGE_HTTP_PROXY SFORGE_HTTPS_PROXY SFORGE_NO_PROXY SFORGE_AGENT_EXTRA_ENV \
        SFORGE_ISOLATED_DOCKER_NETWORK
      export SFORGE_LOG_DIR="${RUN_ROOT}/final-eval"
      guarded_exec INT "${SFORGE_PYTHON}" "${TASK_ENTRY}" eval \
        --backend docker \
        --task "${TASK_ID}" \
        --archive "${archive}" \
        --run-id "${FINAL_RUN_ID}" \
        --timeout "${t_eval}" \
        --judge-cpu-limit "${judge_cpu}" \
        --judge-mem-limit "${judge_mem}" \
        --json
    ) >>"${log}" 2>&1 &
    final_pid=$!
    wait "${final_pid}" || rc=$?
    final_pid=""
    echo "final Judge eval exited rc=${rc}" >>"${log}"
    remove_run_containers >>"${log}" 2>&1
    for uuid in "${judge_uuids[@]}"; do
      ( assert_gpu_released "${uuid}" ) >>"${log}" 2>&1 || { (( rc != 0 )) || rc=1; }
    done
    (( rc != 0 )) \
      || echo "report: ${RUN_ROOT}/final-eval/runs/${FINAL_RUN_ID}/${TASK_ID}/submissions/manual-1/report.json" >>"${log}"
  fi
  final_rc="${rc}"
  printf '%s\n' "${final_rc}" >"${RUN_ROOT}/final-eval.exit"
  echo "[$(date -u +%FT%TZ)] final Judge finished rc=${final_rc} (${log})"
}

# completion.json (read by scripts/status.sh): STATUS FAILURE_STAGE ARCHIVE RUN_RC
write_completion() {
  local status="$1" stage="$2" archive="$3" run_rc="$4"
  local submission="${RUN_ROOT}/final-eval/runs/${FINAL_RUN_ID}/${TASK_ID}/submissions/manual-1"
  local report="${submission}/report.json" archive_sha="" evaluated_sha="" report_sha=""
  [[ -z "${archive}" || ! -s "${archive}" ]] || archive_sha="$(sha256sum "${archive}" | awk '{print $1}')"
  [[ ! -s "${submission}/submission.tar.gz" ]] \
    || evaluated_sha="$(sha256sum "${submission}/submission.tar.gz" | awk '{print $1}')"
  if [[ -s "${report}" ]]; then
    report_sha="$(sha256sum "${report}" | awk '{print $1}')"
  else
    report=""
  fi
  jq -n \
    --arg status "${status}" --arg stage "${stage}" \
    --arg finished_at "$(date -u +%FT%TZ)" --arg run_id "${RUN_ID}" --arg task_id "${TASK_ID}" \
    --arg mode "${mode}" --arg work_uuid "${work_uuid}" --arg judge_uuids "${judge_uuids[*]}" \
    --arg archive "${archive}" --arg archive_sha "${archive_sha}" --arg evaluated_sha "${evaluated_sha}" \
    --arg report "${report}" --arg report_sha "${report_sha}" \
    --arg run_rc "${run_rc}" --arg final_rc "${final_rc}" \
    --slurpfile scored <(if [[ -n "${report}" ]]; then jq -c '{score, valid}' "${report}"; else echo '{}'; fi) \
    '{schema_version: 1, status: $status,
      failure_stage: (if $stage == "" then null else $stage end),
      finished_at: $finished_at, run_id: $run_id, task_id: $task_id, mode: $mode,
      work_gpu_uuid: $work_uuid, judge_gpu_uuid: $judge_uuids,
      sforge_run_exit: (if $run_rc == "" then null else ($run_rc | tonumber) end),
      final_eval_exit: (if $final_rc == "" then null else ($final_rc | tonumber) end),
      final_archive: (if $archive == "" then null else $archive end),
      final_archive_sha256: (if $archive_sha == "" then null else $archive_sha end),
      evaluated_archive_sha256: (if $evaluated_sha == "" then null else $evaluated_sha end),
      final_report: (if $report == "" then null else $report end),
      final_report_sha256: (if $report_sha == "" then null else $report_sha end),
      score: $scored[0].score, valid: $scored[0].valid}' \
    >"${RUN_ROOT}/completion.json"
  completion_written=1
}

# Status of a finished final Judge: "" when the report exists and evaluated
# exactly the frozen archive, else the failing stage.
final_stage() {
  local archive="$1"
  local submission="${RUN_ROOT}/final-eval/runs/${FINAL_RUN_ID}/${TASK_ID}/submissions/manual-1"
  [[ "${final_rc}" == 0 ]] || { echo final_judge; return; }
  [[ -s "${submission}/report.json" ]] || { echo final_report_missing; return; }
  [[ -s "${submission}/submission.tar.gz" \
     && "$(sha256sum "${submission}/submission.tar.gz" | awk '{print $1}')" \
        == "$(sha256sum "${archive}" | awk '{print $1}')" ]] \
    || { echo evaluated_archive_mismatch; return; }
  echo ""
}

# Runs while this shell still holds every lock, whatever ended the run: stop the
# GPU users this shell started (SForge's graceful signal, bounded), remove every
# container of this run, check that no GPU of this run is still in use, and
# record the outcome in completion.json and the launch exit file.
cleanup() {
  local rc=$? release=PASS busy="" removed="" left="" foreign="" tmp
  set +e
  # A second signal must not cut the cleanup short, nor a dead log pipe kill it.
  trap '' HUP INT TERM PIPE
  if [[ -n "${run_pid}${final_pid}${judge_pid}" ]]; then
    echo "[$(date -u +%FT%TZ)] cleanup (exit ${rc}): stopping the managed run, final Judge and Judge server"
  fi
  [[ -z "${run_pid}" ]] || kill -s INT "${run_pid}" 2>/dev/null
  [[ -z "${final_pid}" ]] || kill -s INT "${final_pid}" 2>/dev/null
  [[ -z "${judge_pid}" ]] || kill -s TERM "${judge_pid}" 2>/dev/null
  stop_child "${run_pid}" - 180 "sforge run"
  stop_child "${final_pid}" - 120 "the final Judge"
  stop_child "${judge_pid}" - 60 "the Judge server"
  run_pid=""; final_pid=""; judge_pid=""
  [[ -z "${proxy_log_pid}" ]] || kill "${proxy_log_pid}" 2>/dev/null
  proxy_log_pid=""
  removed="$(remove_run_containers)"
  [[ -z "${removed}" ]] || printf '[%s] cleanup: %s\n' "$(date -u +%FT%TZ)" "${removed//$'\n'/; }"
  owned_containers=()
  if [[ -n "${owned_network}" ]]; then
    docker network rm "${owned_network}" >/dev/null 2>&1 \
      || echo "WARNING: could not remove the Docker network ${owned_network}"
  fi
  # No GPU of this run may be in use when the locks go.
  if busy="$(gpus_still_busy 60 "${work_uuid}" "${judge_uuids[@]}")"; then
    [[ -z "${busy}" ]] || release=FAILED
  else
    release=UNKNOWN
  fi
  while IFS=$'\t' read -r _ _ _ exact how; do
    [[ -n "${how}" ]] || continue
    if [[ "${exact}" == 1 ]]; then left+="${left:+; }${how}"; else foreign+="${foreign:+; }${how}"; fi
  done < <(container_gpu_claims "${gpu_pairs[@]}")
  [[ -z "${left}" ]] || release=FAILED
  [[ -z "${foreign}" ]] || echo "WARNING: containers outside this run claim its GPUs: ${foreign}"
  if [[ "${release}" != PASS ]]; then
    echo "ERROR: GPU release at exit ${release}: compute processes on ${busy:-none}; containers left: ${left:-none}"
    (( rc != 0 )) || rc=1
  else
    echo "[$(date -u +%FT%TZ)] GPU release at exit: PASS (no compute process or run container on any GPU)"
  fi
  # Ended before its completion record (a signal or an error): record that.
  (( completion_written )) || write_completion failed aborted "" "${run_rc}"
  if [[ -s "${RUN_ROOT}/completion.json" ]]; then
    tmp="${RUN_ROOT}/.completion.json.tmp"
    jq --argjson rc "${rc}" --arg release "${release}" --arg busy "${busy}" --arg removed "${removed}" \
      --arg left "${left}" \
      '.launcher_exit = $rc
       | .gpu_release = {status: $release,
           busy_gpu_uuids: ($busy | split("\n") | map(select(. != ""))),
           containers_left: (if $left == "" then [] else ($left | split("; ")) end),
           cleanup: ($removed | split("\n") | map(select(. != "")))}' \
      "${RUN_ROOT}/completion.json" >"${tmp}" && mv -f "${tmp}" "${RUN_ROOT}/completion.json"
  fi
  printf '%s\n' "${rc}" >"${LOGS_DIR}/launch-${RUN_ID}.exit"
  exit "${rc}"
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

# Every Judge (interactive and final) takes every listed GPU and the deadlines
# for their count; the Work adapter refuses a Work GPU that is also a Judge GPU.
export KV_EVICTION_MODEL_REPO
export KV_EVICTION_JUDGE_GPU_UUIDS="${judge_uuids[*]}"
export SFORGE_JUDGE_EXTRA_ENV="KV_EVICTION_CANDIDATE_TIMEOUT=${t_candidate},KV_EVICTION_TOTAL_TIMEOUT=${t_total},KV_EVICTION_JUDGE_TIMEOUT=${t_judge}"
export KV_EVICTION_JUDGE_EVAL_TIMEOUT="${t_eval}"
export MLSYSFORGE_WORK_INTERNET=true
unset SFORGE_SINGLE_GPU_HANDOFF

image_manifest="${LOGS_DIR}/build/image-manifest.json"
work_image_id="$(docker image inspect --format '{{.Id}}' "${WORK_IMAGE}")"
judge_image_id="$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}")"
source_digest="$(jq -er '.source_digest' "${image_manifest}")"
final_manifest_sha256="$(jq -er '.final_suite.manifest_sha256' "${image_manifest}")"
final_items_sha256="$(jq -er '.final_suite.items_sha256' "${image_manifest}")"
proxy_sha256="$(sha256sum "${PROXY_SOURCE}" | awk '{print $1}')"
agent_entry_sha256="$(sha256sum "${AGENT_ENTRY}" | awk '{print $1}')"
eval_archive_sha256=""
(( ! EVAL_MODE )) || eval_archive_sha256="$(sha256sum "${EVAL_ARCHIVE}" | awk '{print $1}')"
manifest_port="${JUDGE_PORT}"
(( ! EVAL_MODE )) || manifest_port=0

jq -n \
  --arg run_id "${RUN_ID}" --arg task_id "${TASK_ID}" --arg mode "${mode}" \
  --arg agent "${AGENT_NAME}" --arg model "${MODEL}" --arg effort "${REASONING_EFFORT}" \
  --arg started_at "$(date -u +%FT%TZ)" \
  --arg work_index "${work_index}" --arg work_uuid "${work_uuid}" \
  --arg judge_indices "${judge_indices[*]}" --arg judge_uuids "${judge_uuids[*]}" \
  --arg work_mem "${effective_work_mem}" --arg judge_mem "${effective_judge_mem}" \
  --arg passed_work_mem "${work_mem}" --arg passed_judge_mem "${judge_mem}" \
  --arg profile_mode "${profile_mode}" --arg profile_source "${profile_source}" \
  --arg profile_sha256 "${profile_sha256}" \
  --arg judge_extra_env "${judge_extra_env}" \
  --arg work_cpuset "${work_cpuset_policy}" --arg judge_cpuset "${judge_cpuset}" \
  --arg runtime_dir "${AGENT_RUNTIME_DIR}" --arg runtime_agent "${runtime_agent}" \
  --arg runtime_version "${runtime_version}" --arg runtime_tree_sha256 "${runtime_tree_sha256}" \
  --arg work_image_id "${work_image_id}" --arg judge_image_id "${judge_image_id}" \
  --arg source_digest "${source_digest}" \
  --arg final_manifest_sha256 "${final_manifest_sha256}" --arg final_items_sha256 "${final_items_sha256}" \
  --arg model_repo "${KV_EVICTION_MODEL_REPO}" --arg model_revision "${model_revision}" \
  --arg model_mount "${MODEL_MOUNT}" \
  --arg proxy_sha256 "${proxy_sha256}" --arg agent_entry_sha256 "${agent_entry_sha256}" \
  --arg review "${review_summary}" \
  --arg model_api "${MODEL_API_BASE_URL}" \
  --arg eval_archive "${EVAL_ARCHIVE}" --arg eval_archive_sha256 "${eval_archive_sha256}" \
  --argjson public_smoke "${PUBLIC_SMOKE}" \
  --argjson timeout "${TASK_TIMEOUT}" --argjson cooldown "${SUBMISSION_COOLDOWN}" \
  --argjson work_cpu "${effective_work_cpu}" --argjson judge_cpu "${effective_judge_cpu}" \
  --argjson passed_work_cpu "${work_cpu}" --argjson passed_judge_cpu "${judge_cpu}" \
  --argjson t_candidate "${t_candidate}" --argjson t_total "${t_total}" \
  --argjson t_judge "${t_judge}" --argjson t_eval "${t_eval}" \
  --argjson judge_port "${manifest_port}" \
  '{schema_version: 1, run_id: $run_id, task_id: $task_id, mode: $mode, started_at: $started_at,
    agent: (if $mode == "agent" then {name: $agent, model: $model, reasoning_effort: $effort,
      model_api_base_url: (if $model_api == "" then null else $model_api end),
      runtime: {dir: $runtime_dir, agent: $runtime_agent, version: $runtime_version,
        tree_sha256: $runtime_tree_sha256},
      timeout_seconds: $timeout, submission_cooldown_seconds: $cooldown,
      scheduled_auto_evaluation: false, judge_port: $judge_port} else null end),
    eval_archive: (if $mode == "agent" then null else
      {path: $eval_archive, sha256: $eval_archive_sha256, public_smoke: ($public_smoke == 1)} end),
    internet: true,
    work: {gpu: {index: $work_index, uuid: $work_uuid}, cpu_limit: $work_cpu, mem_limit: $work_mem,
      cpuset_policy: $work_cpuset},
    judge: {gpu_indices: ($judge_indices | split(" ")), gpu_uuids: ($judge_uuids | split(" ")),
      cpu_limit: $judge_cpu, mem_limit: $judge_mem, cpuset: $judge_cpuset,
      timeouts: {candidate: $t_candidate, total: $t_total, judge: $t_judge, eval: $t_eval},
      extra_env: $judge_extra_env},
    resources: {mode: $profile_mode,
      profile: (if $profile_source == "" then null else {source: $profile_source, sha256: $profile_sha256} end),
      passed: {work: {cpu: $passed_work_cpu, memory: $passed_work_mem},
               judge: {cpu: $passed_judge_cpu, memory: $passed_judge_mem}}},
    gpu_locks: ["/tmp/sforge-gpu-<uuid>.lock", "/tmp/mlsysforge-gpu-<uuid>.lock"],
    integrity_review: $review,
    images: {work: $work_image_id, judge: $judge_image_id, source_digest: $source_digest,
      final_manifest_sha256: $final_manifest_sha256, final_items_sha256: $final_items_sha256},
    model: {repo: $model_repo, revision: $model_revision, mount: $model_mount},
    restricted_proxy_sha256: $proxy_sha256, agent_entry_sha256: $agent_entry_sha256,
    final_evaluation: "independent full frozen suite on the same Judge GPUs under the held locks"}' \
  >"${RUN_ROOT}/launch-manifest.json"

if (( EVAL_MODE )); then
  echo "[$(date -u +%FT%TZ)] ${RUN_ID}: eval-only ${EVAL_ARCHIVE} (sha256 ${eval_archive_sha256})"
  if (( PUBLIC_SMOKE )); then
    echo "[$(date -u +%FT%TZ)] public smoke on Work GPU ${work_index} -> ${RUN_ROOT}/public-smoke.log"
    owned_containers=("${smoke_name}")
    # The public runner admits only an archive's own root entries, so the smoke
    # runs on the extracted evaluation archive, not on the task's oracle/ tree
    # (which also holds references/).
    smoke_lab="${RUN_ROOT}/public-smoke-lab"
    set +e
    (
      set -e
      rm -rf "${smoke_lab}"
      install -d -m 0755 "${smoke_lab}"
      tar -xzf "${EVAL_ARCHIVE}" -C "${smoke_lab}" --no-same-owner
      chmod -R a+rX "${smoke_lab}"
      assert_container_accelerator "${WORK_IMAGE}" "${work_uuid}" work
      docker run --rm --name "${smoke_name}" \
        --gpus "device=${work_uuid}" -e CUDA_VISIBLE_DEVICES=0 \
        --network none --security-opt no-new-privileges:true \
        --cpus "${effective_work_cpu}" --memory "${effective_work_mem}" \
        -e "KV_EVICTION_MODEL_DIR=${MODEL_MOUNT}/snapshots/${model_revision}" \
        -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
        -v "${KV_EVICTION_MODEL_REPO}:${MODEL_MOUNT}:ro" \
        -v "${smoke_lab}:/oracle:ro" \
        --entrypoint python \
        "${WORK_IMAGE}" \
        /workspace/benchmark/public_runner.py --lab-root /oracle --suite quick \
        --candidate-version "$((10#$(tr -d '[:space:]' < "${smoke_lab}/FINAL_VERSION")))"
    ) >"${RUN_ROOT}/public-smoke.log" 2>&1
    smoke_rc=$?
    set -e
    owned_containers=()
    printf '%s\n' "${smoke_rc}" >"${RUN_ROOT}/public-smoke.exit"
    echo "[$(date -u +%FT%TZ)] public smoke finished rc=${smoke_rc}"
    if (( smoke_rc != 0 )); then
      write_completion failed public_smoke "${EVAL_ARCHIVE}" ""
      exit "${smoke_rc}"
    fi
  fi
  run_final_judge "${EVAL_ARCHIVE}"
  stage="$(final_stage "${EVAL_ARCHIVE}")"
  completion_status=completed
  [[ -z "${stage}" ]] || completion_status=failed
  write_completion "${completion_status}" "${stage}" "${EVAL_ARCHIVE}" ""
  [[ -z "${stage}" ]] || exit 1
  exit 0
fi

# --- Agent run --------------------------------------------------------------------
MODEL_API_KEY=""
if [[ -n "${MODEL_API_BASE_URL}" ]]; then
  MODEL_API_KEY="$(tr -d '[:space:]' <"${MODEL_API_KEY_FILE}")"
  [[ -n "${MODEL_API_KEY}" ]] || die "MODEL_API_KEY_FILE is empty"
fi

export SFORGE_ADMIN_SECRET
SFORGE_ADMIN_SECRET="$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')"

bridge_gateway="$(docker network inspect bridge | jq -er '.[0].IPAM.Config[0].Gateway')"
[[ "${bridge_gateway}" =~ ^[0-9]+(\.[0-9]+){3}$ ]] || die "could not resolve the Docker bridge gateway"
if ss -H -ltn "sport = :${JUDGE_PORT}" 2>/dev/null | grep -q .; then
  die "JUDGE_PORT ${JUDGE_PORT} is already bound on this host"
fi
guarded_exec TERM "${SFORGE_PYTHON}" "${AGENT_ENTRY}" serve --host "${bridge_gateway}" --port "${JUDGE_PORT}" \
  >"${RUN_ROOT}/judge-server.log" 2>&1 &
judge_pid=$!
for _ in $(seq 1 90); do
  curl --noproxy '*' -fsS "http://${bridge_gateway}:${JUDGE_PORT}/openapi.json" >/dev/null 2>&1 && break
  kill -0 "${judge_pid}" 2>/dev/null || die "Judge server exited; see ${RUN_ROOT}/judge-server.log"
  sleep 1
done
curl --noproxy '*' -fsS "http://${bridge_gateway}:${JUDGE_PORT}/openapi.json" >/dev/null \
  || die "Judge server did not become ready"

owned_network="${network_name}"
docker network create --driver bridge --internal \
  --label "mlsysforge.run_id=${RUN_ID}" "${network_name}" >/dev/null
owned_containers=("${proxy_name}")
docker run --detach --rm --name "${proxy_name}" --network bridge \
  --label "mlsysforge.run_id=${RUN_ID}" \
  --add-host host.docker.internal:host-gateway --env NVIDIA_VISIBLE_DEVICES=void \
  --volume "${PROXY_SOURCE}:/opt/mlsysforge/restricted_proxy.py:ro" \
  --entrypoint python3 "${WORK_IMAGE}" /opt/mlsysforge/restricted_proxy.py \
  --judge-host "${bridge_gateway}" \
  --judge-port "${JUDGE_PORT}" \
  --judge-forward-port "${EVALUATION_FORWARD_PORT}" \
  --allow-public-egress \
  ${model_api_host:+--model-forward "${MODEL_FORWARD_PORT}:${model_api_host}:${model_api_port}"} >/dev/null
( close_gpu_lock_fds; exec docker logs -f "${proxy_name}" ) >"${RUN_ROOT}/sidecar.log" 2>&1 &
proxy_log_pid=$!
docker network connect --alias runtime-proxy "${network_name}" "${proxy_name}"
proxy_ip="$(docker inspect "${proxy_name}" | jq -er --arg n "${network_name}" '.[0].NetworkSettings.Networks[$n].IPAddress | select(length > 0)')"

in_work_network() {
  docker run --rm --network "${network_name}" --env NVIDIA_VISIBLE_DEVICES=void \
    --entrypoint curl "${WORK_IMAGE}" --noproxy '*' --connect-timeout 2 "$@"
}
for _ in $(seq 1 30); do
  in_work_network --max-time 3 -fsS "http://${proxy_ip}:3128/health" >/dev/null 2>&1 && break
  sleep 1
done
in_work_network --max-time 5 -fsS "http://${proxy_ip}:3128/health" >/dev/null \
  || die "restricted sidecar did not become ready"
in_work_network --max-time 5 -fsS "http://${proxy_ip}:${EVALUATION_FORWARD_PORT}/openapi.json" >/dev/null \
  || die "Judge is not reachable from the internal Work network"
if in_work_network --max-time 4 -fsS https://example.com >/dev/null 2>&1; then
  die "internal Work network unexpectedly has direct egress"
fi
if [[ -n "${model_api_host}" ]]; then
  in_work_network --max-time 10 -sS -o /dev/null \
    "http://${proxy_ip}:${MODEL_FORWARD_PORT}${model_api_path}/models" \
    || die "the model API relay ${MODEL_API_BASE_URL} is unreachable through the sidecar"
fi
echo "[$(date -u +%FT%TZ)] restricted network: PASS (${network_name}, sidecar ${proxy_ip})"

# Host-side registration and polling address the Judge directly.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export NO_PROXY="${bridge_gateway},${proxy_ip},127.0.0.1,localhost"
export no_proxy="${NO_PROXY}"
unset SFORGE_AGENT_API_KEY SFORGE_AGENT_API_BASE_URL SFORGE_CODEX_AUTH_JSON_PATH \
  CODEX_API_KEY OPENAI_API_KEY OPENAI_BASE_URL
if [[ -n "${MODEL_API_BASE_URL}" ]]; then
  export SFORGE_AGENT_API_BASE_URL="http://${proxy_ip}:${MODEL_FORWARD_PORT}${model_api_path}"
  export SFORGE_AGENT_API_KEY="${MODEL_API_KEY}"
elif [[ "${AGENT_NAME}" == claude-code ]]; then
  export MLSYSFORGE_CLAUDE_OAUTH_TOKEN_FILE="${CLAUDE_TOKEN_FILE}"
else
  export SFORGE_CODEX_AUTH_JSON_PATH="${AUTH_JSON}"
fi
export SFORGE_CODEX_REASONING_EFFORT="${REASONING_EFFORT}"
if [[ -n "${CODEX_CLI_VERSION}" ]]; then
  export SFORGE_CODEX_CLI_VERSION="${CODEX_CLI_VERSION}"
else
  unset SFORGE_CODEX_CLI_VERSION
fi
export MLSYSFORGE_AGENT="${AGENT_NAME}" MLSYSFORGE_AGENT_MODEL="${MODEL}"
export MLSYSFORGE_AGENT_EFFORT="${REASONING_EFFORT}" MLSYSFORGE_AGENT_BUDGET_SECONDS="${TASK_TIMEOUT}"
export MLSYSFORGE_RUN_LOG_ROOT="${RUN_ROOT}"
export SFORGE_AGENT_RUNTIME_DIR="${AGENT_RUNTIME_DIR}"
export SFORGE_ISOLATED_DOCKER_NETWORK="${network_name}"
export SFORGE_HTTP_PROXY="http://${proxy_ip}:3128"
export SFORGE_HTTPS_PROXY="http://${proxy_ip}:3128"
export SFORGE_NO_PROXY="${proxy_ip},127.0.0.1,localhost"
export SFORGE_AGENT_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=${work_uuid},CUDA_VISIBLE_DEVICES=0"

echo "[$(date -u +%FT%TZ)] starting ${RUN_ID}: ${AGENT_NAME} ${MODEL}/${REASONING_EFFORT}, ${TASK_TIMEOUT} s, cooldown ${SUBMISSION_COOLDOWN} s"
echo "  work GPU ${work_index} (${effective_work_cpu} CPU, ${effective_work_mem}); judge GPUs ${judge_indices[*]} (${effective_judge_cpu} CPU, ${effective_judge_mem}, timeout ${t_eval} s)"
# sforge run is a child of this shell (guarded_exec), waited for in the
# background so that a signal reaches the cleanup at once: the cleanup then
# stops it with SIGINT, SForge's graceful path, which retires the Work container.
guarded_exec INT "${SFORGE_PYTHON}" "${AGENT_ENTRY}" run \
  --backend docker \
  --task "${TASK_ID}" \
  --agent "${AGENT_NAME}" \
  --model "${MODEL}" \
  --timeout "${TASK_TIMEOUT}" \
  --disable-auto-eval \
  --submission-cooldown "${SUBMISSION_COOLDOWN}" \
  --judge-url "http://${proxy_ip}:${EVALUATION_FORWARD_PORT}" \
  --enable-internet \
  --work-cpu-limit "${work_cpu}" \
  --work-mem-limit "${work_mem}" \
  --judge-cpu-limit "${judge_cpu}" \
  --judge-mem-limit "${judge_mem}" \
  --run-id "${RUN_ID}" &
run_pid=$!
run_rc=0
wait "${run_pid}" || run_rc=$?
run_pid=""
printf '%s\n' "${run_rc}" >"${RUN_ROOT}/launcher.exit"
echo "[$(date -u +%FT%TZ)] managed run finished rc=${run_rc}"

# The interactive Judge server, the sidecar and the Work network are done (the
# final runs its own in-process Judge).  Anything sforge run left on a GPU of
# this run (its Work container after a crash, a Judge whose drain timed out) is
# removed now, while every lock is held, so that the final Judge can start.
stop_child "${judge_pid}" TERM 60 "the Judge server"
judge_pid=""
[[ -z "${proxy_log_pid}" ]] || kill "${proxy_log_pid}" 2>/dev/null || true
proxy_log_pid=""
leftover="$(remove_run_containers)"
owned_containers=()
[[ -z "${leftover}" ]] || echo "[$(date -u +%FT%TZ)] after the managed run: ${leftover//$'\n'/; }"
docker network rm "${network_name}" >/dev/null 2>&1 \
  || echo "WARNING: could not remove the Docker network ${network_name}"
owned_network=""

# The authoritative score is an independent Judge of the final archive, run
# here on the same Judge GPUs while this shell still holds every lock.
final_archive="${RUN_ROOT}/${TASK_ID}/final_archive.tar.gz"
stage=""
if [[ "${FINAL_EVAL}" == 1 && -s "${final_archive}" ]]; then
  run_final_judge "${final_archive}"
  stage="$(final_stage "${final_archive}")"
elif [[ "${FINAL_EVAL}" == 1 ]]; then
  echo "[$(date -u +%FT%TZ)] no final archive at ${final_archive}; final Judge skipped"
  stage="final_archive_missing"
else
  # A run without its independent final has no authoritative score.
  echo "[$(date -u +%FT%TZ)] FINAL_EVAL=0: final Judge disabled; the run has no authoritative score"
  stage="final_disabled"
fi
(( run_rc == 0 )) || stage="${stage:-sforge_run}"
completion_archive=""
[[ ! -s "${final_archive}" ]] || completion_archive="${final_archive}"
completion_status=completed
[[ -z "${stage}" ]] || completion_status=failed
write_completion "${completion_status}" "${stage}" "${completion_archive}" "${run_rc}"
(( run_rc == 0 )) || exit "${run_rc}"
[[ -z "${stage}" ]] || exit 1
exit 0
