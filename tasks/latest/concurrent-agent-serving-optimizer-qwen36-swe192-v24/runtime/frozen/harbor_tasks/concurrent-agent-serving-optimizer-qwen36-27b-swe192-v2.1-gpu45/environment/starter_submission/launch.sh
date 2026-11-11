#!/usr/bin/env bash
set -euo pipefail

: "${BACKEND_URL:?BACKEND_URL is required}"
: "${PORT:=9000}"

starter_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BACKEND_URL PORT PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
exec python3 "${starter_dir}/gateway.py"
