#!/usr/bin/env bash
set -euo pipefail

: "${BACKEND_URL:?BACKEND_URL is required}"
: "${PORT:=9000}"

export PYTHONDONTWRITEBYTECODE=1
exec python3 "$(dirname -- "$0")/gateway.py"
