#!/usr/bin/env bash
set -euo pipefail
# A die inside $(...) must stop the caller too: without this, a failing
# helper in a command substitution would leave an empty value behind.
shopt -s inherit_errexit

TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MIGRATION_ROOT="${TASK_ROOT}"

find_repository_root() {
  local candidate="${TASK_ROOT}"
  while [[ "${candidate}" != / ]]; do
    if [[ -f "${candidate}/pyproject.toml" && -d "${candidate}/sforge" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
    candidate="$(dirname -- "${candidate}")"
  done
  return 1
}

REPOSITORY_ROOT="${REPOSITORY_ROOT:-$(find_repository_root)}"
SFORGE_ROOT="${SFORGE_ROOT:-${REPOSITORY_ROOT}}"
# Retain the historical variable as a compatibility alias, but never default
# back to the 0715 checkout.
MLSYSFORGE_ROOT="${SFORGE_ROOT}"
SFORGE_BIN="${SFORGE_BIN:-${SFORGE_ROOT}/.venv/bin/sforge}"
SFORGE_PYTHON="${SFORGE_PYTHON:-$(dirname -- "${SFORGE_BIN}")/python}"
TASKS_DIR="${TASKS_DIR:-${TASK_ROOT}/tasks}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPOSITORY_ROOT}/outputs}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-quant-videogen-cross-model-kv-codec-h100-v4-scratch}"
LOGS_DIR="${LOGS_DIR:-${OUTPUT_ROOT}/${OUTPUT_SUBDIR}}"
# This task's Infrathon assets (assets/fetch.sh).  QVG_ARTIFACT_ROOT defaults to the tree
# assets/fetch.sh assembles from them (scripts/assemble_artifact_root.py).
ASSET_TASK_ROOT="${INFRATHON_ASSETS_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/infrathon/assets}/$(basename -- "${TASK_ROOT}")"
ARTIFACT_ROOT="${QVG_ARTIFACT_ROOT:-${ASSET_TASK_ROOT}/artifact-root}"
TASK_ID=quant_videogen_cross_model_kv_codec_h100_scratch_v4
BENCHMARK_NAME=quant_videogen_h100_scratch_v4
WORK_IMAGE="${QVG_WORK_IMAGE:-${BENCHMARK_NAME}.work.${TASK_ID}:local}"
JUDGE_IMAGE="${QVG_JUDGE_IMAGE:-${BENCHMARK_NAME}.judge.${TASK_ID}:local}"
# The Judge above is this public Judge (images/judge/Dockerfile, publishable) plus
# the holdout layer images/judge/Dockerfile.private, which is only ever built on
# the host.
JUDGE_PUBLIC_IMAGE="${QVG_JUDGE_PUBLIC_IMAGE:-${BENCHMARK_NAME}.judge-public.${TASK_ID}:local}"
# Built by scripts/build_images.sh from images/work/base/Dockerfile, which
# needs nothing but this repository and a digest-pinned public base image.
DEFAULT_BASE_WORK_IMAGE="${BENCHMARK_NAME}.base.${TASK_ID}:local"
# The frozen release: case table, Judge slot plan and per-model budgets.  It is
# Judge-only: the in-repository copy when present, else the asset-root copy
# written by assets/fetch.sh (<INFRATHON_ASSETS_ROOT>/<task directory>/judge/...).
RELEASE_PATH="${TASK_ROOT}/configs/h100_scratch_v4_release.json"
if [[ ! -f "${RELEASE_PATH}" ]]; then
  RELEASE_PATH="${ASSET_TASK_ROOT}/judge/h100_scratch_v4_release.json"
fi
# This device class's quality file: the zero point of the quality term and the
# oracle's score band, measured per hardware type (the release is shared).  The
# in-repository copy when present, else the asset-root copy
# <asset root>/<task directory>/hw/<type>/judge/h100_scratch_v4_quality.json for the
# type INFRATHON_HARDWARE names or scripts/infrathon/assets.py detects from GPU 0.
quality_path() {
  local hardware
  if [[ -f "${TASK_ROOT}/configs/h100_scratch_v4_quality.json" ]]; then
    printf '%s\n' "${TASK_ROOT}/configs/h100_scratch_v4_quality.json"
    return
  fi
  hardware="$(python3 -I "${REPOSITORY_ROOT}/scripts/infrathon/assets.py" hardware)" \
    || die "cannot tell this host's hardware type; set INFRATHON_HARDWARE"
  printf '%s\n' "${ASSET_TASK_ROOT}/hw/${hardware}/judge/h100_scratch_v4_quality.json"
}
RUNTIME_ENTRY="${TASK_ROOT}/scripts/sforge_runtime_entry.py"
IMAGE_MANIFEST="${LOGS_DIR}/build/image-manifest.json"

export SFORGE_TASKS_DIR="${TASKS_DIR}"
export SFORGE_LOG_DIR="${LOGS_DIR}"
export QVG_ARTIFACT_ROOT="${ARTIFACT_ROOT}"

die() { echo "ERROR: $*" >&2; exit 1; }

require_command() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }

require_sforge() {
  [[ -x "${SFORGE_BIN}" ]] || die "missing SForge executable: ${SFORGE_BIN}"
  [[ -x "${SFORGE_PYTHON}" ]] || die "missing SForge Python: ${SFORGE_PYTHON}"
}

require_artifact_root() {
  [[ -d "${ARTIFACT_ROOT}" ]] \
    || die "missing Quant-VideoGen artifact root: ${ARTIFACT_ROOT} (run assets/fetch.sh, or set QVG_ARTIFACT_ROOT)"
}

canonical_gpu_index() {
  local raw="$1"
  [[ "${raw}" =~ ^[0-9]+$ ]] || die "GPU index is not numeric: ${raw}"
  echo "$((10#${raw}))"
}

gpu_uuid() {
  nvidia-smi --id="$1" --query-gpu=uuid --format=csv,noheader | xargs
}

gpu_index() {
  local selector="$1" index
  if [[ "${selector}" =~ ^[0-9]+$ ]]; then
    canonical_gpu_index "${selector}"
    return
  fi
  [[ "${selector}" =~ ^GPU-[A-Za-z0-9-]+$ ]] \
    || die "GPU selector must be a physical index or UUID: ${selector}"
  index="$(
    nvidia-smi --id="${selector}" --query-gpu=index --format=csv,noheader \
      | xargs
  )"
  canonical_gpu_index "${index}"
}

gpu_name() {
  nvidia-smi --id="$1" --query-gpu=name --format=csv,noheader | xargs
}

gpu_lock_path() {
  local uuid="$1"
  [[ "${uuid}" =~ ^GPU-[A-Za-z0-9-]+$ ]] || die "unsafe GPU UUID: ${uuid}"
  echo "${GPU_LOCK_DIR}/sforge-gpu-${uuid}.lock"
}

# The host's GPU lease convention: a card is free only while neither
# /tmp/sforge-gpu-<uuid>.lock nor /tmp/mlsysforge-gpu-<uuid>.lock is held
# (launch/gpu-scheduler.sh checks both).  Deliberately not read from the
# environment, so no caller can move a lease out of the shared namespace.
GPU_LOCK_DIR=/tmp

# Hold both lock names for one GPU until the calling shell exits.  A path that
# is a symlink, not a regular file, or not owned by the launching user is
# refused; the file is opened for append, so even a link planted between the
# check and the open can never truncate its target, and the opened descriptor
# must be the very inode the unfollowed path names.
lock_gpu() {
  local uuid="$1" role="$2" path fd opened linked
  for path in "$(gpu_lock_path "${uuid}")" "${GPU_LOCK_DIR}/mlsysforge-gpu-${uuid}.lock"; do
    [[ ! -L "${path}" ]] || die "${role} GPU lock is a symlink: ${path}"
    exec {fd}>>"${path}" || die "cannot open ${role} GPU lock: ${path}"
    opened="$(stat -L -c '%d:%i:%u:%F' "/proc/self/fd/${fd}")"
    linked="$(stat -c '%d:%i:%u:%F' "${path}")"
    [[ "${opened}" == "${linked}" && "${opened}" == *":$(id -u):regular"* ]] \
      || die "${role} GPU lock is not a regular file owned by this user: ${path}"
    flock -n "${fd}" || die "${role} GPU ${uuid}: ${path} is held by another process"
  done
}

# Capability envelope per Judge GPU, mirroring the controller's floor: the
# release's judge.min_device_memory_bytes (79 GiB, 84,825,604,096 bytes) in whole
# MiB.  The official codec peaks at about 75.4e9 bytes per card (LongCat), so
# 80 GB-class cards (an H100 SXM reports 81,559 MiB) and 94 GB-class ones (an
# H100 NVL reports 95,830 MiB) both qualify; the model name is provenance, never a gate.
GPU_MIN_COMPUTE_CAPABILITY="${GPU_MIN_COMPUTE_CAPABILITY:-8.0}"
GPU_MIN_TOTAL_MIB="${GPU_MIN_TOTAL_MIB:-80896}"

# Admit one GPU (physical index or UUID) into the envelope and require it idle.
assert_gpu_idle() {
  local selector="$1" role="$2" label name capability total used util
  local -a required=() observed=()
  if [[ "${selector}" =~ ^GPU-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$ ]]; then
    label="${selector}"
  else
    selector="$(canonical_gpu_index "${selector}")"
    label="physical ${selector}"
  fi
  IFS=, read -r name capability total used util < <(
    nvidia-smi --id="${selector}" \
      --query-gpu=name,compute_cap,memory.total,memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )
  IFS=. read -r -a required <<<"${GPU_MIN_COMPUTE_CAPABILITY}"
  IFS=. read -r -a observed <<<"${capability// /}"
  [[ "${observed[0]:-}" =~ ^[0-9]+$ && "${observed[1]:-}" =~ ^[0-9]+$ ]] \
    || die "${role} GPU ${label} did not report a compute capability: ${capability}"
  (( 10#${observed[0]} * 100 + 10#${observed[1]} \
     >= 10#${required[0]} * 100 + 10#${required[1]} )) \
    || die "${role} GPU ${label} is below compute capability ${GPU_MIN_COMPUTE_CAPABILITY}: ${capability} (${name})"
  (( 10#${total// /} >= GPU_MIN_TOTAL_MIB )) \
    || die "${role} GPU ${label} has ${total} MiB, below the ${GPU_MIN_TOTAL_MIB} MiB floor (${name})"
  (( 10#${used// /} < 1024 && 10#${util// /} < 10 )) \
    || die "${role} GPU ${label} is busy (${used} MiB, ${util}%)"
  echo "${role} GPU: ${label}, ${name}, cc ${capability}, ${total} MiB total"
}

# Fail closed when the process list is unavailable: an unreadable list is not
# evidence of an idle card.
assert_no_compute_process() {
  local uuid="$1" role="$2" apps
  apps="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader)" \
    || die "cannot list the compute processes on the ${role} GPUs"
  ! grep -Fxq "${uuid}" <<<"${apps}" \
    || die "${role} GPU ${uuid} already has a compute process"
}

# The Judge accepts only interchangeable cards; refuse a mixed set before the
# container starts rather than after an hour of staging.
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

assert_gpu_released() {
  local uuid="$1" role="${2:-assigned}" attempt
  for attempt in $(seq 1 60); do
    if ! nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null \
        | grep -Fxq "${uuid}"; then
      echo "GPU release barrier: PASS (${uuid})"
      return 0
    fi
    sleep 1
  done
  die "${role} GPU still has a compute process after the release barrier: ${uuid}"
}

# Judge GPUs: JUDGE_GPUS is a comma-separated list of host indices or UUIDs
# (e.g. JUDGE_GPUS=0,1,2,3,4,5,6,7).  Prints one canonical whole-GPU UUID per
# line in the given order; duplicates are refused.  Callers capture the output
# with $(...) so a refusal stops them (a process substitution would not).
judge_gpu_uuids() {
  local raw="${1:-${JUDGE_GPUS:-}}" selector index uuid
  local -a selectors=()
  local -A seen=()
  [[ -n "${raw}" ]] \
    || die "JUDGE_GPUS must list the Judge GPUs, e.g. JUDGE_GPUS=0,1,2,3,4,5,6,7"
  IFS=, read -r -a selectors <<<"${raw}"
  (( ${#selectors[@]} >= 1 && ${#selectors[@]} <= 8 )) \
    || die "JUDGE_GPUS must name 1 to 8 GPUs"
  for selector in "${selectors[@]}"; do
    index="$(gpu_index "${selector// /}")"
    uuid="$(gpu_uuid "${index}")"
    [[ "${uuid}" =~ ^GPU-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$ ]] \
      || die "GPU ${index} has no whole-device UUID"
    [[ -z "${seen[${uuid}]:-}" ]] || die "JUDGE_GPUS names GPU ${index} twice"
    seen[${uuid}]=1
    echo "${uuid}"
  done
}

# Judge container plan for N GPUs, printed as
# "CPU_LIMIT MEM_LIMIT CONTROLLER_SECONDS EVAL_SECONDS".
#
# Limits: the task JSON's judge.cpu_limit / judge.mem_limit size the Judge on
# the release's judge.max_gpu_count GPUs (threads_per_slot CPUs and an equal
# memory share per slot); N GPUs get N of those per-slot shares.
#
# Timeouts: the controller's own bound (judge/evaluate_hidden.py and
# judge/run_judge.py) is a list-scheduling simulation over N GPUs - each case
# costs its model's stage_seconds + max(fidelity, VBench) metric seconds +
# case_overhead_seconds, GPU i starts start_stagger_seconds * i late - plus
# controller_overhead_seconds.  The harness timeout must never undercut that
# bound, so this takes the largest makespan of the list schedules the
# controller could use (the heaviest-first static plan, with the start offsets
# or with the largest offset added to its longest GPU, and the in-order greedy
# queue) and adds a fixed margin for container start-up and private-artifact
# collection.
JUDGE_HARNESS_MARGIN_SECONDS=300

judge_plan() {
  local n="$1"
  [[ "${n}" =~ ^[1-9][0-9]*$ ]] || die "Judge GPU count must be a positive integer: ${n}"
  python3 -I - "${RELEASE_PATH}" "${TASKS_DIR}/${TASK_ID}.json" "${n}" \
    "${JUDGE_HARNESS_MARGIN_SECONDS}" <<'PY'
import json
import math
import re
import sys

release_path, task_path, raw_count, raw_margin = sys.argv[1:5]
count, margin = int(raw_count), int(raw_margin)


def fail(message):
    raise SystemExit(f"ERROR: {message}")


def integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        fail(f"{name} must be an integer >= {minimum}")
    return value


with open(release_path, encoding="utf-8") as handle:
    release = json.load(handle)
with open(task_path, encoding="utf-8") as handle:
    task = json.load(handle)
judge = release.get("judge")
if not isinstance(judge, dict):
    fail("the release has no judge block")
low = integer(judge.get("min_gpu_count"), "judge.min_gpu_count", 1)
high = integer(judge.get("max_gpu_count"), "judge.max_gpu_count", low)
if not low <= count <= high:
    fail(f"the release admits {low}-{high} Judge GPUs, not {count}")
stagger = integer(judge.get("start_stagger_seconds"), "judge.start_stagger_seconds", 0)
case_overhead = integer(judge.get("case_overhead_seconds"), "judge.case_overhead_seconds", 0)
controller_overhead = integer(
    judge.get("controller_overhead_seconds"), "judge.controller_overhead_seconds", 1
)
units = []
for case in release.get("cases") or fail("the release has no cases"):
    model = release["models"][case["model"]]
    budgets = {
        key: integer(model.get(key), f"models.{case['model']}.{key}", 1)
        for key in ("stage_seconds", "fidelity_metric_seconds", "vbench_metric_seconds")
    }
    units.append(
        budgets["stage_seconds"]
        + max(budgets["fidelity_metric_seconds"], budgets["vbench_metric_seconds"])
        + case_overhead
    )


def makespan(order, offsets_in_assignment):
    loads = [index * stagger if offsets_in_assignment else 0 for index in range(count)]
    used = [False] * count
    for unit in order:
        target = min(range(count), key=lambda index: (loads[index], index))
        loads[target] += unit
        used[target] = True
    longest = max(load for load, busy in zip(loads, used) if busy)
    # A plan made without the offsets may start its longest GPU last.
    return longest if offsets_in_assignment else longest + (count - 1) * stagger


heaviest_first = sorted(units, reverse=True)
controller = controller_overhead + max(
    makespan(heaviest_first, False),
    makespan(heaviest_first, True),
    makespan(units, True),
)

limits = task.get("judge") or fail("the task JSON has no judge block")
cpu_total = integer(limits.get("cpu_limit"), "judge.cpu_limit", 1)
memory = re.fullmatch(r"([1-9][0-9]*)g", str(limits.get("mem_limit")))
if memory is None:
    fail("judge.mem_limit must be whole GiB such as '448g'")
cpu = math.ceil(cpu_total * count / high)
memory_gib = math.ceil(int(memory.group(1)) * count / high)
print(cpu, f"{memory_gib}g", controller, controller + margin)
PY
}

# Every file the Work and Judge images are built from: the .dockerignore
# allow-list, minus the Judge-private inputs that are mounted, never baked.
# The Judge bakes the release, so a recalibrated release needs a rebuild too.
# images/published.json is written by publishing, after the images are built
# from these sources.
image_source_digest() {
  (
    cd -- "${TASK_ROOT}"
    find .dockerignore assets configs images judge metrics policy runtime upstream \
      -type f ! -path 'judge/private/*' ! -path '*/__pycache__/*' \
      ! -path assets/manifest.json ! -path assets/fetch.sh ! -path images/published.json \
      ! -path '*/.pytest_cache/*' ! -name '*.pyc' -print0 \
      | LC_ALL=C sort -z \
      | xargs -0 sha256sum
  ) | sha256sum | awk '{print $1}'
}

assert_built_images_current() {
  local recorded_source recorded_work recorded_judge
  local current_source current_work current_judge
  [[ -s "${IMAGE_MANIFEST}" && ! -L "${IMAGE_MANIFEST}" ]] \
    || die "missing build manifest; run scripts/build_images.sh"
  recorded_source="$(jq -er '.source_digest' "${IMAGE_MANIFEST}")" \
    || die "build manifest has no source digest"
  recorded_work="$(jq -er '.work.id' "${IMAGE_MANIFEST}")" \
    || die "build manifest has no Work image ID"
  recorded_judge="$(jq -er '.judge.id' "${IMAGE_MANIFEST}")" \
    || die "build manifest has no Judge image ID"
  current_source="$(image_source_digest)"
  current_work="$(docker image inspect --format '{{.Id}}' "${WORK_IMAGE}" 2>/dev/null)" \
    || die "missing Work image; run scripts/build_images.sh"
  current_judge="$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}" 2>/dev/null)" \
    || die "missing Judge image; run scripts/build_images.sh"
  [[ "${current_source}" == "${recorded_source}" ]] \
    || die "image-driving source changed after the recorded build; run scripts/build_images.sh"
  [[ "${current_work}" == "${recorded_work}" ]] \
    || die "Work image tag moved after the recorded build"
  [[ "${current_judge}" == "${recorded_judge}" ]] \
    || die "Judge image tag moved after the recorded build"
  echo "Image/source freshness: PASS (${current_source:0:12})"
}

assert_private_auth_file() {
  local path="$1" mode
  [[ -f "${path}" && ! -L "${path}" && -s "${path}" ]] \
    || die "missing nonempty regular auth file: ${path}"
  mode="$(stat -c '%a' "${path}")"
  [[ "${mode}" == 600 ]] || die "auth file must have mode 600, got ${mode}"
  [[ "$(stat -c '%u' "${path}")" == "$(id -u)" ]] \
    || die "auth file must be owned by the launching user"
}

user_in_docker_group() {
  local launch_user="$1"
  getent group docker | awk -F: -v wanted="${launch_user}" '
    $1 == "docker" { n=split($4,a,","); for(i=1;i<=n;i++) if(a[i]==wanted) found=1 }
    END { exit(found ? 0 : 1) }'
}

ensure_docker_group() {
  if docker info >/dev/null 2>&1; then return; fi
  [[ "${SFORGE_DOCKER_GROUP_REEXEC:-0}" != 1 ]] || die "Docker remains unavailable after group re-exec"
  user_in_docker_group "$(id -un)" || die "launch user is not listed in docker group"
  local entry="${SFORGE_ENTRY_SCRIPT:-$0}" quoted name
  local -a command=(
    env
    SFORGE_DOCKER_GROUP_REEXEC=1
    REPOSITORY_ROOT="${REPOSITORY_ROOT}"
    SFORGE_ROOT="${SFORGE_ROOT}"
    MLSYSFORGE_ROOT="${MLSYSFORGE_ROOT}"
    SFORGE_BIN="${SFORGE_BIN}"
    SFORGE_PYTHON="${SFORGE_PYTHON}"
    TASKS_DIR="${TASKS_DIR}"
    OUTPUT_ROOT="${OUTPUT_ROOT}"
    OUTPUT_SUBDIR="${OUTPUT_SUBDIR}"
    LOGS_DIR="${LOGS_DIR}"
    QVG_ARTIFACT_ROOT="${ARTIFACT_ROOT}"
  )
  # sg may start a fresh shell. Preserve every task launcher control that can
  # legitimately be set before this helper runs, including the proxy bridge
  # variables the Work network preflight uses.
  for name in \
    CODEX_AUTH_JSON_PATH CODEX_CLI_VERSION \
    BASE_WORK_IMAGE \
    JUDGE_GPUS JUDGE_CPU_LIMIT JUDGE_MEM_LIMIT FINAL_EVAL_TIMEOUT ORACLE_TIMEOUT \
    RUN_STAMP RUN_ID \
    GPU_MIN_COMPUTE_CAPABILITY GPU_MIN_TOTAL_MIB \
    QVG_WORK_IMAGE QVG_JUDGE_IMAGE \
    QVG_HOST_PROXY_HOST QVG_HOST_PROXY_PORT QVG_DOCKER_PROXY_PORT \
    SFORGE_AGENT_RUNTIME_DIR \
    SFORGE_VERSION_STORE_ROOT \
    SFORGE_CPUSET SFORGE_JUDGE_CPUSET SFORGE_LLM_JUDGE_MODE \
    HTTP_PROXY HTTPS_PROXY ALL_PROXY NO_PROXY \
    http_proxy https_proxy all_proxy no_proxy; do
    if [[ -v "${name}" ]]; then
      command+=("${name}=${!name}")
    fi
  done
  command+=(bash "${entry}" "$@")
  printf -v quoted '%q ' "${command[@]}"
  exec sg docker -c "${quoted}"
}
