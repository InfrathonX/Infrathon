#!/usr/bin/env bash

set -euo pipefail

TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIGRATION_ROOT="${TASK_ROOT}"

find_repository_root() {
  local candidate="${TASK_ROOT}"
  while [[ "${candidate}" != "/" ]]; do
    if [[ -f "${candidate}/pyproject.toml" && -d "${candidate}/sforge" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
    candidate="$(dirname "${candidate}")"
  done
  (cd "${TASK_ROOT}/.." && pwd)
}

REPOSITORY_ROOT="${REPOSITORY_ROOT:-$(find_repository_root)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPOSITORY_ROOT}/outputs}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-$(basename "${TASK_ROOT}")}"
TASK_ID="qwen3_4b_to_1p7b_gkd_distillation_optimizer_dual_h100_v1"
BENCHMARK="qwen3_gkd_dual_h100"
WORK_IMAGE="${BENCHMARK}.work.${TASK_ID}:local"
JUDGE_IMAGE="${BENCHMARK}.judge.${TASK_ID}:local"
# The Judge above is this public Judge (images/judge/Dockerfile, publishable) plus
# the holdout layer images/judge/Dockerfile.private, which is only ever built on
# the host.
JUDGE_PUBLIC_IMAGE="${BENCHMARK}.judge-public.${TASK_ID}:local"
LOGS_DIR="${LOGS_DIR:-${OUTPUT_ROOT}/${OUTPUT_SUBDIR}}"
STUDENT_MODEL_REVISION="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
TEACHER_MODEL_REVISION="1cfa9a7208912126459214e8b04321603b3df60c"

die() {
  printf '%s\n' "ERROR: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command is missing: $1"
}

user_in_docker_group() {
  getent group docker | awk -F: -v user="$1" '
    $1 == "docker" { n=split($4,m,","); for(i=1;i<=n;i++) if(m[i]==user) ok=1 }
    END { exit(ok ? 0 : 1) }
  '
}

# Re-execute the calling script once under the docker group when the Docker
# socket is unreachable.
#
# A user who was added to the `docker` group has no docker-group credential in
# an already-running login session, so every entry point that shells out to
# `docker` has to refresh its supplementary groups itself.
# scripts/eval_archive.sh and scripts/replay_candidates.sh carry their own
# copies of this, placed after they resolve the GPU pair; scripts that need it
# before doing any work call this one.
ensure_docker_group() {
  docker info >/dev/null 2>&1 && return 0
  [[ "${GKD_DOCKER_GROUP_REEXEC:-0}" != "1" ]] \
    || die "Docker is still unavailable after docker-group re-exec"
  local launch_user
  launch_user="$(id -un)"
  user_in_docker_group "${launch_user}" \
    || die "Docker is unavailable and ${launch_user} is not in the docker group"
  local reexec
  printf -v reexec '%q ' env \
    GKD_DOCKER_GROUP_REEXEC=1 \
    REPOSITORY_ROOT="${REPOSITORY_ROOT}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    OUTPUT_SUBDIR="${OUTPUT_SUBDIR}" \
    SFORGE_PYTHON="${SFORGE_PYTHON:-}" \
    GKD_GPU_INDICES="${GKD_GPU_INDICES:-}" \
    GKD_GPU_UUIDS="${GKD_GPU_UUIDS:-}" \
    GKD_STUDENT_MODEL_HOST_PATH="${GKD_STUDENT_MODEL_HOST_PATH:-}" \
    GKD_TEACHER_MODEL_HOST_PATH="${GKD_TEACHER_MODEL_HOST_PATH:-}" \
    GKD_DATA_HOST_ROOT="${GKD_DATA_HOST_ROOT:-}" \
    GKD_PRIVATE_DATA_HOST_ROOT="${GKD_PRIVATE_DATA_HOST_ROOT:-}" \
    GKD_MANAGED_SMOKE_PORT="${GKD_MANAGED_SMOKE_PORT:-}" \
    GKD_MANAGED_SMOKE_RUN_ID="${GKD_MANAGED_SMOKE_RUN_ID:-}" \
    GKD_MANAGED_SMOKE_KIND="${GKD_MANAGED_SMOKE_KIND:-}" \
    GKD_MANAGED_SMOKE_DEADLINE_SECONDS="${GKD_MANAGED_SMOKE_DEADLINE_SECONDS:-}" \
    SFORGE_AGENT_RUNTIME_DIR="${SFORGE_AGENT_RUNTIME_DIR:-}" \
    SFORGE_VERSION_STORE_ROOT="${SFORGE_VERSION_STORE_ROOT:-}" \
    INFRATHON_ASSETS_ROOT="${INFRATHON_ASSETS_ROOT:-}" INFRATHON_HARDWARE="${INFRATHON_HARDWARE:-}" INFRATHON_PULL="${INFRATHON_PULL:-}" \
    http_proxy="${http_proxy:-}" https_proxy="${https_proxy:-}" no_proxy="${no_proxy:-}" \
    HTTP_PROXY="${HTTP_PROXY:-}" HTTPS_PROXY="${HTTPS_PROXY:-}" NO_PROXY="${NO_PROXY:-}" \
    bash "$0" "$@"
  # The umask is not optional: build contexts and smoke artifacts are staged
  # into the task tree and must stay readable.
  exec sg docker -c "umask 022; ${reexec}"
}

# Resolve the operator-named physical GPU pair to UUIDs.
#
# This task pins no GPU pair of its own: GKD_GPU_INDICES names two physical
# indices for the run, and every downstream consumer — per-GPU locks, the idle
# probe, preflight, and the container device request — works from the resolved
# UUIDs, so index renumbering cannot move a live run onto different silicon.
# Containers always see the selected pair as CUDA devices 0,1.
#
# Publishes GPU_PAIR_INDICES, GPU_PAIR_A_UUID, GPU_PAIR_B_UUID, GPU_PAIR_UUIDS.
resolve_gpu_pair() {
  local indices="${1:-${GKD_GPU_INDICES:-}}"
  [[ -n "${indices}" ]] \
    || die "set GKD_GPU_INDICES to the two physical GPU indices for this run (e.g. 4,5)"
  [[ "${indices}" =~ ^([0-9]+),([0-9]+)$ ]] \
    || die "GKD_GPU_INDICES must be two comma-separated physical GPU indices"
  local first="${BASH_REMATCH[1]}" second="${BASH_REMATCH[2]}"
  [[ "${first}" != "${second}" ]] || die "GKD_GPU_INDICES must name two distinct GPUs"
  require_command nvidia-smi
  local listing
  listing="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits)" \
    || die "nvidia-smi could not enumerate this host's GPUs"
  GPU_PAIR_A_UUID="$(awk -F', *' -v want="${first}" '$1 == want { print $2 }' <<<"${listing}")"
  GPU_PAIR_B_UUID="$(awk -F', *' -v want="${second}" '$1 == want { print $2 }' <<<"${listing}")"
  [[ -n "${GPU_PAIR_A_UUID}" ]] || die "this host does not expose GPU index ${first}"
  [[ -n "${GPU_PAIR_B_UUID}" ]] || die "this host does not expose GPU index ${second}"
  GPU_PAIR_INDICES="${indices}"
  GPU_PAIR_UUIDS="${GPU_PAIR_A_UUID},${GPU_PAIR_B_UUID}"
}

# The Work image is agent-free: SForge bind-mounts a host-provided runtime
# read-only at /opt/agent-runtime when SFORGE_AGENT_RUNTIME_DIR is set.
# require_codex_agent_runtime EXPECTED_VERSION fails closed unless a marked
# codex runtime matching that pinned CLI version is configured, re-exports
# SFORGE_AGENT_RUNTIME_DIR, and publishes the runtime identity as
# AGENT_RUNTIME_DIR / AGENT_RUNTIME_AGENT / AGENT_RUNTIME_VERSION /
# AGENT_RUNTIME_TREE_SHA256 so launchers can record it in run manifests.
require_codex_agent_runtime() {
  local expected_version="${1:-}"
  [[ -n "${expected_version}" ]] \
    || die "require_codex_agent_runtime needs the pinned codex CLI version"
  require_command jq
  AGENT_RUNTIME_DIR="${SFORGE_AGENT_RUNTIME_DIR:-}"
  [[ -n "${AGENT_RUNTIME_DIR}" ]] \
    || die "set SFORGE_AGENT_RUNTIME_DIR (build one with launch/build-agent-runtime.sh)"
  local runtime_marker="${AGENT_RUNTIME_DIR}/.sforge-agent-runtime.json"
  [[ -d "${AGENT_RUNTIME_DIR}" && ! -L "${AGENT_RUNTIME_DIR}" && -f "${runtime_marker}" ]] \
    || die "agent runtime directory is missing or unmarked: ${AGENT_RUNTIME_DIR}"
  local runtime_kind
  runtime_kind="$(jq -er '.kind' "${runtime_marker}")"
  AGENT_RUNTIME_AGENT="$(jq -er '.agent' "${runtime_marker}")"
  AGENT_RUNTIME_VERSION="$(jq -er '.version' "${runtime_marker}")"
  AGENT_RUNTIME_TREE_SHA256="$(jq -er '.tree_sha256' "${runtime_marker}")"
  [[ "${runtime_kind}" == "sforge-agent-runtime" ]] \
    || die "invalid agent runtime marker: ${runtime_marker}"
  [[ "${AGENT_RUNTIME_AGENT}" == "codex" ]] \
    || die "formal run requires a codex runtime, found: ${AGENT_RUNTIME_AGENT}"
  [[ "${AGENT_RUNTIME_VERSION}" == "${expected_version}" ]] \
    || die "agent runtime version ${AGENT_RUNTIME_VERSION} differs from the pinned ${expected_version}"
  export SFORGE_AGENT_RUNTIME_DIR="${AGENT_RUNTIME_DIR}"
}

# What the Work and Judge images are built from (Infrathon publication
# records it; scripts/infrathon/publish_images.sh refuses images that are not current).
image_source_digest() {
  (
    cd "${TASK_ROOT}"
    find images/work images/judge -type f ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 \
      | LC_ALL=C sort -z \
      | xargs -0 sha256sum
  ) | sha256sum | awk '{print $1}'
}

assert_built_images_current() {
  local manifest="${LOGS_DIR}/build/image-manifest.json"
  [[ -s "${manifest}" ]] || die "missing build manifest; run scripts/build_images.sh"
  [[ "$(jq -er '.source_digest' "${manifest}")" == "$(image_source_digest)" ]] \
    || die "image build sources changed; rebuild images"
  [[ "$(docker image inspect --format '{{.Id}}' "${WORK_IMAGE}" 2>/dev/null)" == "$(jq -er '.work.id' "${manifest}")" ]] \
    || die "work image tag no longer matches the recorded build"
  [[ "$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}" 2>/dev/null)" == "$(jq -er '.judge.id' "${manifest}")" ]] \
    || die "judge image tag no longer matches the recorded build"
  echo "Image/source freshness: PASS ($(image_source_digest | cut -c1-12))"
}
