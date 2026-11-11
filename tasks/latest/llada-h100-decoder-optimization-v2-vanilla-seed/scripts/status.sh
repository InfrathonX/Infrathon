#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
require_command jq
require_command nvidia-smi

run_id="${1:-}"
if [[ -z "${run_id}" ]]; then
  latest="$(find "${LOGS_DIR}/runs" -mindepth 1 -maxdepth 1 -type d \
    -name 'llada-decoder-v2-h100-*' ! -name '*-final-*' \
    -printf '%T@ %f\n' 2>/dev/null | sort -nr | head -1 || true)"
  run_id="${latest#* }"
fi
[[ -n "${run_id}" ]] || die "no formal run found"

run_root="${LOGS_DIR}/runs/${run_id}"
task_root="${run_root}/${TASK_ID}"
launcher_exit="${LOGS_DIR}/launch-${run_id}.exit"
completion="${run_root}/completion.json"

echo "Run: ${run_id}"
if [[ -f "${run_root}/experiment-manifest.json" ]]; then
  jq . "${run_root}/experiment-manifest.json"
fi

echo
echo "Terminal status:"
launcher_rc=""
if [[ -f "${launcher_exit}" ]]; then
  launcher_rc="$(sed -n '1p' "${launcher_exit}")"
  if [[ "${launcher_rc}" =~ ^[0-9]+$ ]]; then
    echo "launcher exit: ${launcher_rc} (${launcher_exit})"
  else
    echo "launcher exit: malformed (${launcher_exit})"
  fi
else
  echo "launcher exit: pending or not recorded"
fi
completion_status=""
if [[ -f "${completion}" ]]; then
  completion_status="$(jq -r '.status // empty' "${completion}")"
  jq '{
    schema_version,
    status,
    finished_at,
    sforge_run_exit,
    final_eval_exit: (.final_eval_exit // .final_120m_eval_exit),
    scheduling,
    lifecycle,
    work_gpu,
    judge_gpu,
    final_archive,
    final_report,
    curve
  }' "${completion}"
else
  echo "completion: pending"
fi
if [[ "${launcher_rc}" == "0" && "${completion_status}" == "passed" ]]; then
  echo "terminal verdict: PASS (validated and cleanup-complete)"
elif [[ -n "${launcher_rc}" && "${launcher_rc}" != "0" ]]; then
  echo "terminal verdict: FAIL (launcher exit ${launcher_rc})"
else
  echo "terminal verdict: not yet proven"
fi

echo
echo "Role GPU snapshot:"
if [[ -f "${run_root}/experiment-manifest.json" ]]; then
  while IFS=$'\t' read -r role index expected_uuid; do
    [[ "${index}" =~ ^[0-9]+$ ]] || continue
    echo "${role}: expected index=${index}, uuid=${expected_uuid}"
    nvidia-smi --id="${index}" \
      --query-gpu=index,name,uuid,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader || true
  done < <(
    jq -r '
      ["Work", (.work_gpu.index | tostring), .work_gpu.uuid],
      ["Judge", (.judge_gpu.index | tostring), .judge_gpu.uuid]
      | @tsv
    ' "${run_root}/experiment-manifest.json"
  )
else
  nvidia-smi --query-gpu=index,name,uuid,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader | sed -n '1,8p'
fi

echo
echo "Final-only reports:"
find "${task_root}/submissions" -mindepth 2 -maxdepth 2 -name report.json \
  -printf '%h/report.json\n' 2>/dev/null | sort || true
find "${task_root}" -mindepth 1 -maxdepth 1 -type f \
  -name 'final-*m-report.json' -print 2>/dev/null | sort

if [[ -f "${task_root}/curve/curve.md" ]]; then
  echo
  cat "${task_root}/curve/curve.md"
fi

echo
echo "Recent runner output:"
tail -n 30 "${run_root}/sforge-run.log" 2>/dev/null || true
