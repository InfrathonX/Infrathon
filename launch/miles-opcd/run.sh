#!/usr/bin/env bash
# Start (or dry-run) an agent run of the Miles OPCD task. The agent harness, Codex
# version, model, API endpoint, effort, budget and GPU pairs all come from --config.
#   run.sh [--task v3] --config CONFIG [--judge-gpus C,D | --shared-gpus] [--run-id ID] [--dry-run] [--foreground]
# Work runs on the config's GPU_INDICES and the Judge on its own pair, JUDGE_GPU_INDICES or --judge-gpus C,D (four
# GPUs; Work is never paused). --shared-gpus (JUDGE_GPU_INDICES=shared) runs the Judge on the Work pair and pauses
# Work for each Judge round. With neither, run_task.py refuses to start.
# v3 (default) uses the OPCD paper's same-size frozen teacher (the initial Qwen3-1.7B with the
# context). v1 and v2 (Qwen3-4B teacher) are not part of this release.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
version=v3
if [[ "${1:-}" == "--task" ]]; then version="$2"; shift 2; fi
case "${version}" in
  v3) ;;
  v1|v2) echo "OPCD ${version} is not part of this release; use v3" >&2; exit 2 ;;
  *) echo "unknown task version '${version}' (use v3)" >&2; exit 2 ;;
esac
exec python3 "${ROOT}/tasks/latest/miles-qwen3-opcd-2xh100-${version}-seed/scripts/run_task.py" "$@"
