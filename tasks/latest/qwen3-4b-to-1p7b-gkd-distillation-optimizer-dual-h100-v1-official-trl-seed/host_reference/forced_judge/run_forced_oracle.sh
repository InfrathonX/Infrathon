#!/usr/bin/env bash
# Forced-oracle Judge run for a host-only treatment arm (LOCAL ONLY).
#
# Same pipeline as scripts/eval_archive.sh -- same preflight, same
# sforge_dual_h100_entry.py `eval` adapter (mounts, private data partition,
# network none, two attested GPU UUIDs, 16 CPUs / 96 GB, timeout 12000 s), same
# run_judge.py / verify_gkd.py / formal E2E controller -- with exactly two
# substitutions:
#   * the Judge image is the derived `:forced-oracle` tag built from
#     host_reference/forced_judge/Dockerfile (pinned Liger-Kernel + admission
#     override); selected through a temporary copy of tasks/ whose judge
#     image_tag is FORCED_TAG (the committed task JSON is untouched);
#   * the host-side LLM integrity review is off (SFORGE_LLM_JUDGE_MODE=off).
#
# Usage:
#   GKD_GPU_INDICES=2,3 [SFORGE_CPUSET=8-15,40-47] \
#   [LIGER_SRC=/path/to/Liger-Kernel@780e76b FORCED_BUILD=1] \
#   host_reference/forced_judge/run_forced_oracle.sh [ARCHIVE.tar.gz|ARM_DIR] [RUN_ID]
#
# ARM_DIR (default host_reference/full_liger_gkd) is packed exactly like a
# Work capture (solutions/, FINAL_VERSION, notes.md).  The report lands in
# ${OUTPUT_ROOT:-<repo>/outputs}/<task dir>/runs/RUN_ID/<task_id>/submissions/manual-1/.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/../../scripts/lib.sh"

FORCED_TAG="${FORCED_TAG:-forced-oracle}"
FORCED_JUDGE_IMAGE="${JUDGE_IMAGE%:*}:${FORCED_TAG}"
[[ "${FORCED_TAG}" != "local" ]] || die "FORCED_TAG must not be the published :local tag"
LIGER_COMMIT="780e76b8115bfd7b78592569853b245a33cc7bf0"

target="${1:-${TASK_ROOT}/host_reference/full_liger_gkd}"
run_id="${2:-forced-gkd-$(date -u +%Y%m%d-%H%M%S)}"
[[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "unsafe RUN_ID: ${run_id}"

work="$(mktemp -d "${TMPDIR:-/tmp}/forced-oracle.XXXXXX")"
trap 'rm -rf "${work}"' EXIT

if [[ -d "${target}" ]]; then
  archive="${work}/submission.tar.gz"
  tar -C "${target}" --owner=0 --group=0 --sort=name \
    --exclude=__pycache__ -czf "${archive}" solutions FINAL_VERSION notes.md
else
  [[ -f "${target}" && -s "${target}" ]] || die "missing archive or arm directory: ${target}"
  archive="$(cd "$(dirname "${target}")" && pwd)/$(basename "${target}")"
fi

if [[ "${FORCED_BUILD:-0}" == "1" ]]; then
  [[ -d "${LIGER_SRC:-}/src/liger_kernel" ]] || die "set LIGER_SRC to a Liger-Kernel checkout"
  head="$(git -C "${LIGER_SRC}" rev-parse HEAD)"
  [[ "${head}" == "${LIGER_COMMIT}" ]] || die "LIGER_SRC is at ${head}, expected ${LIGER_COMMIT}"
  docker build --build-context "liger=${LIGER_SRC}" \
    --build-arg "JUDGE_IMAGE=${JUDGE_IMAGE}" \
    --build-arg "LIGER_COMMIT=${LIGER_COMMIT}" \
    -t "${FORCED_JUDGE_IMAGE}" "${HERE}"
fi
docker image inspect "${WORK_IMAGE}" "${FORCED_JUDGE_IMAGE}" >/dev/null \
  || die "missing ${WORK_IMAGE} or ${FORCED_JUDGE_IMAGE} (FORCED_BUILD=1 LIGER_SRC=...)"

find_repository_root() {
  local candidate="${TASK_ROOT}"
  while [[ "${candidate}" != "/" ]]; do
    if [[ -f "${candidate}/pyproject.toml" && -d "${candidate}/sforge" ]]; then
      printf '%s\n' "${candidate}"; return 0
    fi
    candidate="$(dirname "${candidate}")"
  done
  die "cannot locate the SForge repository root"
}
REPOSITORY_ROOT="${REPOSITORY_ROOT:-$(find_repository_root)}"
SFORGE_PYTHON="${SFORGE_PYTHON:-${REPOSITORY_ROOT}/.venv/bin/python}"
[[ -x "${SFORGE_PYTHON}" ]] || die "missing SForge Python: ${SFORGE_PYTHON}"

resolve_gpu_pair "${GKD_GPU_INDICES:-}"
export GKD_GPU_INDICES="${GPU_PAIR_INDICES}" GKD_GPU_UUIDS="${GKD_GPU_UUIDS:-${GPU_PAIR_UUIDS}}"

hub="${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}"
model_repo="${GKD_STUDENT_MODEL_HOST_PATH:-${hub}/models--Qwen--Qwen3-1.7B}"
teacher_model_repo="${GKD_TEACHER_MODEL_HOST_PATH:-${hub}/models--Qwen--Qwen3-4B}"
[[ -n "${INFRATHON_ASSETS_ROOT:-}" || ! -d "${HOME}/.cache/infrathon/assets" ]] \
  || INFRATHON_ASSETS_ROOT="${HOME}/.cache/infrathon/assets"
if [[ -n "${INFRATHON_ASSETS_ROOT:-}" ]]; then
  default_data_root="${INFRATHON_ASSETS_ROOT}/$(basename "${TASK_ROOT}")/judge/tldr_gkd_v1"
else
  default_data_root="${HOME}/mlsysforge-artifacts/qwen3-gkd/tldr_gkd_v1"
fi
data_root="${GKD_DATA_HOST_ROOT:-${default_data_root}}"
private_data_root="${GKD_PRIVATE_DATA_HOST_ROOT:-${default_data_root}}"
python3 "${TASK_ROOT}/scripts/preflight.py" \
  --gpu-indices "${GKD_GPU_INDICES}" \
  --expected-gpu-uuids "${GKD_GPU_UUIDS}" \
  --model-path "${model_repo}/snapshots/${STUDENT_MODEL_REVISION}" \
  --teacher-model-path "${teacher_model_repo}/snapshots/${TEACHER_MODEL_REVISION}" \
  --data-root "${data_root}" \
  --private-data-root "${private_data_root}" >/dev/null

# Temporary task spec: identical except judge.image_tag.
mkdir -p "${work}/tasks"
cp "${TASK_ROOT}/tasks/BENCHMARK.yaml" "${work}/tasks/"
jq --arg tag "${FORCED_TAG}" '.judge.image_tag = $tag' \
  "${TASK_ROOT}/tasks/${TASK_ID}.json" > "${work}/tasks/${TASK_ID}.json"

mkdir -p "${LOGS_DIR}"
export SFORGE_TASKS_DIR="${work}/tasks"
export SFORGE_LOG_DIR="${LOGS_DIR}"
export SFORGE_LLM_JUDGE_MODE=off
export GKD_STUDENT_MODEL_HOST_PATH="${model_repo}"
export GKD_TEACHER_MODEL_HOST_PATH="${teacher_model_repo}"
export GKD_DATA_HOST_ROOT="${data_root}"
export GKD_PRIVATE_DATA_HOST_ROOT="${private_data_root}"
echo "forced-oracle: image=${FORCED_JUDGE_IMAGE} archive=${archive} run_id=${run_id} gpus=${GKD_GPU_INDICES}" >&2

"${SFORGE_PYTHON}" "${TASK_ROOT}/scripts/sforge_dual_h100_entry.py" eval \
  --backend docker \
  --task "${TASK_ID}" \
  --archive "${archive}" \
  --run-id "${run_id}" \
  --timeout 12000 \
  --judge-cpu-limit 16 \
  --judge-mem-limit 96g \
  --json
