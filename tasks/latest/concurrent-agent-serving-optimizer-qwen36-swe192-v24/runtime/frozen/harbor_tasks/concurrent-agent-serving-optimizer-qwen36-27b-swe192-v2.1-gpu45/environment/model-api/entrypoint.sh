#!/usr/bin/env bash
set -euo pipefail

: "${UPSTREAM_BASE_URL:?UPSTREAM_BASE_URL must name the OpenAI-compatible server origin}"
PYTHONPATH=/opt/model-api python3 - <<'PY'
from transport import configured_base_url

configured_base_url()
PY

: "${ADMIN_TOKEN_FILE:=/run/kvbench-admin/token}"
install -d -m 0700 "$(dirname "$ADMIN_TOKEN_FILE")"
if [[ ! -f "$ADMIN_TOKEN_FILE" ]]; then
  umask 077
  python3 - <<'PY' > "$ADMIN_TOKEN_FILE"
import secrets
print(secrets.token_hex(32))
PY
fi
chown root:root "$ADMIN_TOKEN_FILE"
chmod 0400 "$ADMIN_TOKEN_FILE"

exec python3 /opt/model-api/server.py
