#!/usr/bin/env bash
# Forced-oracle Judge run for the Flux MoE AG+scatter dual-H100 task.
#
# Replicates scripts/eval_archive.sh (same dual-H100 SForge entry and the same
# `sforge eval ... --timeout 3600 --judge-cpu-limit 16 --judge-mem-limit 96g`;
# both GPUs of the NVLink pair, network none, CUDA_VISIBLE_DEVICES=0,1) and
# substitutes only the Judge image with the LOCAL-ONLY :forced-oracle tag (via
# a temporary copy of the task spec whose judge image_tag is "forced-oracle").
# Adds a --cpuset-cpus pin for the pair's CPU slices.
#
# usage: FLUXMOE_GPU_INDICES=A,B [JUDGE_CPUSET=..] run_forced_oracle.sh ARCHIVE RUN_ID
# env:   MLSYSFORGE_ROOT  SForge checkout (required)
#        TASK_DEV_DIR     task dir whose scripts/tasks match the :local images
#        FORCED_TAG       derived Judge tag (default forced-oracle; `local` = normal Judge control)
#        FORCED_SKIP_PREFLIGHT=1  skip scripts/preflight.py (needs a CUDA torch on the host)
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_name="flux-moe-ag-scatter-grouped-gemm-dual-h100-v1-seed"
archive="$(realpath -- "${1:?usage: run_forced_oracle.sh ARCHIVE RUN_ID}")"
run_id="${2:?usage: run_forced_oracle.sh ARCHIVE RUN_ID}"
MLSYSFORGE_ROOT="${MLSYSFORGE_ROOT:?set MLSYSFORGE_ROOT to an SForge checkout}"
TASK_DEV_DIR="${TASK_DEV_DIR:-${MLSYSFORGE_ROOT}/tasks/latest/${task_name}}"
FORCED_TAG="${FORCED_TAG:-forced-oracle}"
source "${TASK_DEV_DIR}/scripts/lib.sh"
SFORGE_PYTHON="${SFORGE_PYTHON:-${REPOSITORY_ROOT}/.venv/bin/python}"
resolve_gpu_pair "${FLUXMOE_GPU_INDICES:-}"
export FLUXMOE_GPU_INDICES="${GPU_PAIR_INDICES}" FLUXMOE_GPU_UUIDS="${GPU_PAIR_UUIDS}"
forced_image="${BENCHMARK_NAME}.judge.${TASK_ID}:${FORCED_TAG}"
# FORCED_TAG=local (control runs of an admissible archive through the normal,
# admission-enforcing Judge in the same pipeline) is the only unlabeled tag allowed.
[[ "${FORCED_TAG}" == local || "$(docker image inspect --format '{{index .Config.Labels "org.infrathon.forced-oracle"}}' "${forced_image}")" == 1 ]] \
  || die "${forced_image} is not a forced-oracle image"
declare -A slice=([0]="0-3,32-35" [1]="4-7,36-39" [2]="8-11,40-43" [3]="12-15,44-47"
                  [4]="16-19,48-51" [5]="20-23,52-55" [6]="24-27,56-59" [7]="28-31,60-63")
IFS=, read -r gpu_a gpu_b <<<"${GPU_PAIR_INDICES}"
export FORCED_JUDGE_CPUSET="${JUDGE_CPUSET:-${slice[${gpu_a}]},${slice[${gpu_b}]}}"
# scripts/preflight.py needs a CUDA torch on the host; when the host has none
# (FORCED_SKIP_PREFLIGHT=1) only the NVLink link of the pair is checked here.
if [[ "${FORCED_SKIP_PREFLIGHT:-0}" == 1 ]]; then
  nvidia-smi topo -m | awk -v a="GPU${gpu_a}" -v b="${gpu_b}" '$1==a {print $(2+b)}' | grep -q '^NV' \
    || die "GPUs ${GPU_PAIR_INDICES} are not an NVLink pair"
else
  "${SFORGE_PYTHON}" "${TASK_DEV_DIR}/scripts/preflight.py" --gpu-indices "${GPU_PAIR_INDICES}" \
    --expected-gpu-uuids "${GPU_PAIR_UUIDS}" >/dev/null
fi

forced_root="${LOGS_DIR}/forced/${run_id}"
[[ ! -e "${forced_root}" ]] || die "forced run directory exists: ${forced_root}"
mkdir -p "${forced_root}/tasks"
cp -a "${TASK_ROOT}/tasks/." "${forced_root}/tasks/"
jq --arg tag "${FORCED_TAG}" '.judge.image_tag = $tag' \
  "${TASK_ROOT}/tasks/${TASK_ID}.json" > "${forced_root}/tasks/${TASK_ID}.json"
jq -n --arg run_id "${run_id}" --arg archive_sha256 "$(sha256sum "${archive}" | cut -d' ' -f1)" \
  --arg judge_image "${forced_image}" \
  --arg judge_image_id "$(docker image inspect --format '{{.Id}}' "${forced_image}")" \
  --arg base_judge_image_id "$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}")" \
  --arg gpu_uuids "${GPU_PAIR_UUIDS}" --arg cpuset "${FORCED_JUDGE_CPUSET}" \
  '{schema_version:1, forced_oracle:true, run_id:$run_id, archive_sha256:$archive_sha256,
    judge_image:$judge_image, judge_image_id:$judge_image_id, base_judge_image_id:$base_judge_image_id,
    gpu_uuids:$gpu_uuids, judge_cpuset:$cpuset}' > "${forced_root}/forced-evaluation-context.json"
cp -- "${archive}" "${forced_root}/submission.tar.gz"

mkdir -p "${LOGS_DIR}"
export SFORGE_TASKS_DIR="${forced_root}/tasks" SFORGE_LOG_DIR="${LOGS_DIR}"
export SFORGE_LLM_JUDGE_MODE=off TRUSTED_CONTROL=1
set +e
"${SFORGE_PYTHON}" "${here}/forced_entry.py" "${TASK_DEV_DIR}/scripts/sforge_dual_h100_entry.py" \
  eval --backend docker --task "${TASK_ID}" --archive "${forced_root}/submission.tar.gz" \
  --run-id "${run_id}" --timeout 3600 --judge-cpu-limit 16 --judge-mem-limit 96g --json \
  > "${forced_root}/eval.log" 2>&1
rc=$?
set -e
report="$(find "${LOGS_DIR}/runs/${run_id}" -name report.json 2>/dev/null | head -1)"
echo "rc=${rc} report=${report:-<none>} log=${forced_root}/eval.log"
[[ -n "${report}" ]] && jq '{valid, score, summary}' "${report}"
exit "${rc}"
