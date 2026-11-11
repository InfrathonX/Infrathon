#!/usr/bin/env bash
set -euo pipefail

timeout 1500s python3 /workspace/public_tests/public_test.py \
  --root /app/workdir/flash_kmeans_lab "$@"
