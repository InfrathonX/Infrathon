#!/usr/bin/env bash
# Regenerate images/judge/protected_hashes.txt from the current trusted files.
set -euo pipefail
cd "$(dirname "$0")/.."
{
  echo "# SHA-256 records for the trusted Judge adapter, verifier, and shared runtime."
  for f in candidate_worker.py public_cases.json run_judge.py source_admission.py verify_kv_transfer.py; do
    echo "$(sha256sum "images/judge/${f}" | awk '{print $1}') judge:${f}"
  done
  echo "$(sha256sum runtime/kv_transfer_runtime.py | awk '{print $1}') worker:kv_transfer_runtime.py"
  echo "$(sha256sum runtime/transfer_abi.h | awk '{print $1}') worker:transfer_abi.h"
} > images/judge/protected_hashes.txt
echo "wrote images/judge/protected_hashes.txt"
