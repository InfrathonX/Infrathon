#!/usr/bin/env bash
set -euo pipefail

exec python3 /workspace/protocol_tests/protocol_smoke.py "$@"
