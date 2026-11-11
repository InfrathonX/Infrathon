#!/usr/bin/env bash
set -euo pipefail
workspace="${SDV2_WORKSPACE:-/app/workdir/sdv2_pipeline_lab}"
exec python3 /workspace/benchmark/public_runner.py --workspace "${workspace}" "$@"
