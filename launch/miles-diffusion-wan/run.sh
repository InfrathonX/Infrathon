#!/usr/bin/env bash
# Start (or dry-run) an agent run of the miles_diffusion Wan2.2-TI2V-5B DanceGRPO task. The agent harness, Codex version,
# model, API endpoint, effort, budget, GPU pairs and the public / private data roots all come from --config.
#   run.sh [--task v2] --config CONFIG [--judge-gpus C,D | --shared-gpus] [--run-id ID] [--dry-run] [--foreground]
# Work runs on the config's GPU_INDICES and the Judge on its own pair, JUDGE_GPU_INDICES or --judge-gpus C,D (four
# GPUs; Work is never paused). --shared-gpus (JUDGE_GPU_INDICES=shared) runs the Judge on the Work pair and pauses
# Work for each Judge round. With neither, run_task.py refuses to start.
# v2 (default) gates on the held-out HPSv3 of 50-step videos (verl-omni's validation). v1 (10-step gate) is not part
# of this release.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
version=v2
if [[ "${1:-}" == "--task" ]]; then version="$2"; shift 2; fi
case "${version}" in
  v2) ;;
  v1) echo "Wan2.2 DanceGRPO v1 is not part of this release; use v2" >&2; exit 2 ;;
  *) echo "unknown task version '${version}' (use v2)" >&2; exit 2 ;;
esac
exec python3 "${ROOT}/tasks/latest/miles-diffusion-wan22-dancegrpo-2xh100-v2-seed/scripts/run_task.py" "$@"
