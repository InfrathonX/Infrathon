#!/usr/bin/env bash
# Start an OPCD run once another run's supervisor session has exited (it holds its GPUs, the Work pair
# and with a separate Judge pair the Judge pair too, and the Judge port until its final evaluation is done).
#   queue-after.sh [--rebuild] <tmux session to wait for> <config> <run id>
# OPCD_TASK_VERSION picks the task version (default v3). --rebuild rebuilds its images first, so task changes made while the earlier run was
# still using the images take effect only for the new run.
set -euo pipefail
rebuild=0
[[ "${1:-}" == "--rebuild" ]] && { rebuild=1; shift; }
wait_for=$1 config=$2 run_id=$3
here=$(cd "$(dirname "$0")" && pwd)
echo "$(date -u +%FT%TZ) waiting for tmux session ${wait_for}"
while tmux has-session -t "=${wait_for}" 2>/dev/null; do sleep 60; done
sleep 60
if (( rebuild )); then
  echo "$(date -u +%FT%TZ) ${wait_for} exited; rebuilding task images"
  bash "${here}/../../tasks/latest/miles-qwen3-opcd-2xh100-${OPCD_TASK_VERSION:-v3}-seed/scripts/build_images.sh"
fi
echo "$(date -u +%FT%TZ) launching ${run_id}"
exec bash "${here}/run.sh" --task "${OPCD_TASK_VERSION:-v3}" --config "${config}" --run-id "${run_id}"
