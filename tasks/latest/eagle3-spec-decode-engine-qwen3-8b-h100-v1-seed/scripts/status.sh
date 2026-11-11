#!/usr/bin/env bash
# Summarize build, policy, active-run, and persisted-result state.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"

main() {
require_command jq
json_mode=0
if [[ "${1:-}" == "--json" ]]; then
  json_mode=1
  shift
fi
(( $# <= 1 )) || die "usage: status.sh [--json] [RUN_ID]"
requested_run="${1:-}"
[[ -z "${requested_run}" || "${requested_run}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$ ]] \
  || die "unsafe run ID: ${requested_run}"

policy="${EAGLE3_PRIVATE_DATA}/policy.json"
policy_status="$(jq -r '.status // "INVALID"' "${policy}" 2>/dev/null || printf 'INVALID')"
tier_count="$(jq -r '.tiers | length' "${policy}" 2>/dev/null || printf '0')"

build_manifest="${BUILD_DIR}/image-manifest.json"
if [[ -f "${build_manifest}" && ! -L "${build_manifest}" ]]; then
  build_status="present"
  work_image_id="$(jq -r '.work.id // "missing"' "${build_manifest}")"
  judge_image_id="$(jq -r '.judge.id // "missing"' "${build_manifest}")"
  recorded_source="$(jq -r '.source_digest // "missing"' "${build_manifest}")"
  current_source="$(image_source_digest)"
  if [[ "${recorded_source}" == "${current_source}" ]]; then
    source_status="current"
  else
    source_status="stale"
  fi
else
  build_status="missing"
  work_image_id="missing"
  judge_image_id="missing"
  source_status="unbuilt"
fi

selected_run_root=""
if [[ -n "${requested_run}" ]]; then
  selected_run_root="${LOGS_DIR}/runs/${requested_run}"
  [[ -d "${selected_run_root}" && ! -L "${selected_run_root}" ]] \
    || die "run does not exist: ${requested_run}"
elif [[ -d "${LOGS_DIR}/runs" ]]; then
  selected_run_root="$(
    find "${LOGS_DIR}/runs" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
      | sort -nr \
      | head -n 1 \
      | cut -d' ' -f2-
  )"
fi

latest_completion=""
if [[ -n "${selected_run_root}" ]]; then
  latest_completion="$(
    find "${selected_run_root}" -type f -name completion.json -print0 \
      | xargs -0 -r stat -c '%Y %n' \
      | sort -nr \
      | head -n 1 \
      | cut -d' ' -f2-
)"
fi
if [[ -n "${selected_run_root}" ]]; then
  latest_run_id="$(basename "${selected_run_root}")"
  if [[ -n "${latest_completion}" ]]; then
    latest_run_status="$(jq -r '.status // "unknown"' "${latest_completion}")"
  else
    latest_run_status="incomplete"
  fi
else
  latest_run_id="none"
  latest_run_status="none"
fi
if (( json_mode )); then
  jq -n \
    --arg task_id "${TASK_ID}" \
    --arg policy_status "${policy_status}" \
    --argjson tier_count "${tier_count}" \
    --arg build_status "${build_status}" \
    --arg source_status "${source_status}" \
    --arg work_image_id "${work_image_id}" \
    --arg judge_image_id "${judge_image_id}" \
    --arg latest_run_id "${latest_run_id}" \
    --arg latest_run_status "${latest_run_status}" \
    '{task_id:$task_id,policy:{status:$policy_status,tier_count:$tier_count},
      build:{status:$build_status,source:$source_status,
        work_image_id:$work_image_id,judge_image_id:$judge_image_id},
      latest_run:{run_id:$latest_run_id,status:$latest_run_status}}'
  exit 0
fi

printf 'Task: %s\n' "${TASK_ID}"
printf 'Policy: %s (%s frozen tiers)\n' "${policy_status}" "${tier_count}"
printf 'Images: %s; source %s\n' "${build_status}" "${source_status}"
printf 'Work image ID: %s\n' "${work_image_id}"
printf 'Judge image ID: %s\n' "${judge_image_id}"
printf 'Selected/latest run: %s (%s)\n' "${latest_run_id}" "${latest_run_status}"

if [[ -n "${selected_run_root}" ]]; then
  experiment_manifest="${selected_run_root}/experiment-manifest.json"
  if [[ -f "${experiment_manifest}" && ! -L "${experiment_manifest}" ]]; then
    printf '\nExperiment manifest:\n'
    jq . "${experiment_manifest}"
    printf '\nRole GPU snapshot:\n'
    while IFS= read -r expected_uuid; do
      [[ -n "${expected_uuid}" ]] || continue
      nvidia-smi \
        --query-gpu=index,name,uuid,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader \
        | grep -F -- "${expected_uuid}" \
        || printf 'GPU %s is not currently visible\n' "${expected_uuid}"
    done < <(jq -r '.work_gpu_uuid, .judge_gpu_uuid' "${experiment_manifest}")
  fi
  printf '\nPersisted final reports:\n'
  find "${selected_run_root}" -type f -path '*/submissions/*/report.json' \
    -print 2>/dev/null \
    | sort
  printf '\nRecent runner output:\n'
  tail -n 30 "${selected_run_root}/sforge-run.log" 2>/dev/null || true
fi

if [[ -d "${LOGS_DIR}" ]]; then
  find "${LOGS_DIR}" \
    -maxdepth 6 \
    -type f \
    \( -name 'reward.json' -o -name '*private.json' -o -name 'completion.json' \) \
    -printf '%TY-%Tm-%TdT%TH:%TM:%TSZ %p\n' \
    2>/dev/null \
    | sort
fi
}

main "$@"
