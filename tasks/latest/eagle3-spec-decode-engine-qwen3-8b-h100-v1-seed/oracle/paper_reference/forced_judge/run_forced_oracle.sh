#!/usr/bin/env bash
# Forced-oracle Judge run for the EAGLE-3 task.
#
# Replicates scripts/eval_archive.sh (same SForge entry, same `sforge eval`
# arguments, same Judge env: NVIDIA_VISIBLE_DEVICES=<uuid>, CUDA_VISIBLE_DEVICES=0,
# EAGLE3_GPU_ROLE=judge, timeouts; model mounts read-only; network none;
# 8 CPUs / 64 GiB) and substitutes only the Judge image with the LOCAL-ONLY
# :forced-oracle tag (via a temporary copy of the task spec whose judge
# image_tag is "forced-oracle"). Adds a --cpuset-cpus pin for the GPU's slice.
#
# usage: JUDGE_GPU=N [JUDGE_CPUSET=..] run_forced_oracle.sh ARCHIVE RUN_ID
# env:   MLSYSFORGE_ROOT  SForge checkout (required)
#        TASK_DEV_DIR     task dir whose scripts/tasks match the :local images
#                         (default: $MLSYSFORGE_ROOT/tasks/latest/<this task>)
#        EAGLE3_TARGET_REPO / EAGLE3_HEAD_DIR  host model paths
#        FORCED_TAG       derived Judge tag (default forced-oracle; `local` = normal Judge control)
set -Eeuo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_name="eagle3-spec-decode-engine-qwen3-8b-h100-v1-seed"
archive="$(realpath -- "${1:?usage: run_forced_oracle.sh ARCHIVE RUN_ID}")"
run_id="${2:?usage: run_forced_oracle.sh ARCHIVE RUN_ID}"
MLSYSFORGE_ROOT="${MLSYSFORGE_ROOT:?set MLSYSFORGE_ROOT to an SForge checkout}"
TASK_DEV_DIR="${TASK_DEV_DIR:-${MLSYSFORGE_ROOT}/tasks/latest/${task_name}}"
FORCED_TAG="${FORCED_TAG:-forced-oracle}"
timeout_s="${FINAL_EVAL_TIMEOUT:-5400}"
source "${TASK_DEV_DIR}/scripts/lib.sh"
forced_image="${BENCHMARK}.judge.${TASK_ID}:${FORCED_TAG}"
judge_gpu="$(canonical_gpu_index "${JUDGE_GPU:?set JUDGE_GPU}")"
judge_uuid="$(gpu_uuid "${judge_gpu}")"
declare -A slice=([0]="0-3,32-35" [1]="4-7,36-39" [2]="8-11,40-43" [3]="12-15,44-47"
                  [4]="16-19,48-51" [5]="20-23,52-55" [6]="24-27,56-59" [7]="28-31,60-63")
export FORCED_JUDGE_CPUSET="${JUDGE_CPUSET:-${slice[${judge_gpu}]:-}}"
work_image_id="$(docker image inspect --format '{{.Id}}' "${WORK_IMAGE}")"
judge_image_id="$(docker image inspect --format '{{.Id}}' "${forced_image}")"
# FORCED_TAG=local (control run of an admissible archive through the normal,
# admission-enforcing Judge in the same pipeline) is the only unlabeled tag allowed.
[[ "${FORCED_TAG}" == local || "$(docker image inspect --format '{{index .Config.Labels "org.infrathon.forced-oracle"}}' "${forced_image}")" == 1 ]] \
  || die "${forced_image} is not a forced-oracle image"

run_root="${LOGS_DIR}/runs/${run_id}"
task_run_root="${run_root}/${TASK_ID}"
[[ ! -e "${run_root}" ]] || die "run directory exists: ${run_root}"
mkdir -p "${task_run_root}"
chmod 700 "${run_root}" "${task_run_root}"
cp -- "${archive}" "${task_run_root}/submission.tar.gz"
chmod 400 "${task_run_root}/submission.tar.gz"
forced_tasks="${task_run_root}/forced-tasks"
mkdir -p "${forced_tasks}"
cp -a "${TASKS_DIR}/." "${forced_tasks}/"
jq --arg tag "${FORCED_TAG}" '.judge.image_tag = $tag' \
  "${TASKS_DIR}/${TASK_ID}.json" > "${forced_tasks}/${TASK_ID}.json"
jq -n --arg run_id "${run_id}" --arg archive_sha256 "$(sha256sum "${archive}" | cut -d' ' -f1)" \
  --arg judge_image "${forced_image}" --arg judge_image_id "${judge_image_id}" \
  --arg base_judge_image_id "$(docker image inspect --format '{{.Id}}' "${JUDGE_IMAGE}")" \
  --arg work_image_id "${work_image_id}" --arg cpuset "${FORCED_JUDGE_CPUSET}" \
  --arg gpu_uuid "${judge_uuid}" \
  '{schema_version:1, forced_oracle:true, run_id:$run_id, archive_sha256:$archive_sha256,
    judge_image:$judge_image, judge_image_id:$judge_image_id, base_judge_image_id:$base_judge_image_id,
    work_image_id:$work_image_id, judge_cpuset:$cpuset, judge_gpu_uuid:$gpu_uuid}' \
  > "${task_run_root}/forced-evaluation-context.json"

export SFORGE_TASKS_DIR="${forced_tasks}"
export EAGLE3_JUDGE_GPU_UUID="${judge_uuid}"
export EAGLE3_RUNTIME_RUN_ID="${run_id}"
export SFORGE_EXPECTED_WORK_IMAGE_ID="${work_image_id}"
export SFORGE_EXPECTED_JUDGE_IMAGE_ID="${judge_image_id}"
export SFORGE_LLM_JUDGE_MODE=off TRUSTED_CONTROL=1
export SFORGE_JUDGE_EXTRA_ENV="NVIDIA_VISIBLE_DEVICES=${judge_uuid},CUDA_VISIBLE_DEVICES=0,EAGLE3_GPU_ROLE=judge,EAGLE3_INTERACTIVE_JUDGE_TIMEOUT=${EAGLE3_INTERACTIVE_JUDGE_TIMEOUT:-3600},EAGLE3_FINAL_JUDGE_TIMEOUT=${timeout_s}"
set +e
"${PYTHON_BIN}" "${here}/forced_entry.py" "${SCRIPT_DIR}/sforge_h100_entry.py" \
  --output-dir "${run_root}" --tasks-dir "${forced_tasks}" \
  eval --backend docker --task "${TASK_ID}" --archive "${task_run_root}/submission.tar.gz" \
  --run-id "${run_id}" --timeout "${timeout_s}" \
  --judge-cpu-limit 8 --judge-mem-limit 64g --json > "${task_run_root}/eval.log" 2>&1
rc=$?
set -e
report="$(find "${run_root}" -path '*/submissions/*/report.json' | head -1)"
echo "rc=${rc} report=${report:-<none>}"
[[ -n "${report}" ]] && jq '{valid, score, summary}' "${report}"
exit "${rc}"
