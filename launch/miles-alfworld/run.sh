#!/usr/bin/env bash
# Start (or dry-run) an agent run of the Miles ALFWorld task. The agent harness, Codex
# version, model, API endpoint, effort, budget and GPU pairs come from --config.
#   run.sh --config CONFIG [--judge-gpus C,D | --shared-gpus] [--run-id ID] [--dry-run] [--foreground]
# Work runs on the config's GPU_INDICES and the Judge on its own pair, JUDGE_GPU_INDICES or --judge-gpus C,D (four
# GPUs; Work is never paused). --shared-gpus (JUDGE_GPU_INDICES=shared) runs the Judge on the Work pair and pauses
# Work for each Judge round. With neither, run_task.py refuses to start.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python3 "${ROOT}/tasks/latest/miles-qwen3-alfworld-agentic-rl-2xh100-v1-seed/scripts/run_task.py" "$@"
