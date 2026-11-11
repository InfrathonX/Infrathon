#!/usr/bin/env bash
set -euo pipefail

# LLaDA H100 v2 currently supports candidate SAVING only. Its Judge requires
# live host artifact fingerprints (cohort secret, runtime file-set hashes)
# assembled per invocation, so a batch replay launcher is deferred.
#
# To evaluate one stored candidate today:
#   1. List saved candidates:
#        .venv/bin/sforge versions --task llada_h100_decoder_optimization_v2 \
#          --run-id <RUN_ID> --candidates --json
#   (Plan-only enumeration works today:
#        .venv/bin/sforge replay-candidates --task llada_h100_decoder_optimization_v2 \
#          --run-id <RUN_ID> --measurement-set <NAME> --dry-run)
#   2. Feed one archive_path to the task's hardened single evaluation:
#        JUDGE_GPU=<idx> tasks/latest/llada-h100-decoder-optimization-v2-vanilla-seed/scripts/eval_archive.sh \
#          <archive_path> <replay-run-id>

echo "ERROR: LLaDA H100 v2 has no batch replay launcher yet." >&2
echo "See the comments in this script for the per-candidate procedure." >&2
exit 2
