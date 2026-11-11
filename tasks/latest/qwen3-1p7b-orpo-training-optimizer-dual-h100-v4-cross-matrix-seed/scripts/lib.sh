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
TASK_ID="qwen3_1p7b_orpo_training_optimizer_dual_h100_v4"
BENCHMARK="qwen3_1p7b_orpo_dual_h100"
WORK_IMAGE="${BENCHMARK}.work.${TASK_ID}:local"
JUDGE_IMAGE="${BENCHMARK}.judge.${TASK_ID}:local"
# The Judge above is this public Judge (images/judge/Dockerfile, publishable) plus
# the holdout layer images/judge/Dockerfile.private, which is only ever built on
# the host.
JUDGE_PUBLIC_IMAGE="${BENCHMARK}.judge-public.${TASK_ID}:local"
LOGS_DIR="${LOGS_DIR:-${OUTPUT_ROOT}/${OUTPUT_SUBDIR}}"
MODEL_REVISION="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

die() {
  printf '%s\n' "ERROR: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command is missing: $1"
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
