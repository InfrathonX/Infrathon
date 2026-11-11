#!/usr/bin/env bash
# Start (or dry-run) an agent run of the Miles OPD-from-scratch task. The agent harness,
# Codex version, model, API endpoint, effort, budget and GPU pairs all come from --config.
#   run.sh [--task v2] --config CONFIG [--judge-gpus C,D | --shared-gpus] [--run-id ID] [--dry-run] [--foreground]
# Work runs on the config's GPU_INDICES and the Judge on its own pair, JUDGE_GPU_INDICES or --judge-gpus C,D (four
# GPUs; Work is never paused). --shared-gpus (JUDGE_GPU_INDICES=shared) runs the Judge on the Work pair and pauses
# Work for each Judge round. With neither, run_task.py refuses to start.
# v2 (default) trains and scores on DAPO-Math-17k (the data of Miles' own OPD example). v1
# (MedMCQA) is not part of this release.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
version=v2
if [[ "${1:-}" == "--task" ]]; then version="$2"; shift 2; fi
case "${version}" in
  v2) ;;
  v1) echo "OPD v1 is not part of this release; use v2" >&2; exit 2 ;;
  *) echo "unknown task version '${version}' (use v2)" >&2; exit 2 ;;
esac
exec python3 "${ROOT}/tasks/latest/miles-qwen3-opd-scratch-2xh100-${version}-seed/scripts/run_task.py" "$@"
