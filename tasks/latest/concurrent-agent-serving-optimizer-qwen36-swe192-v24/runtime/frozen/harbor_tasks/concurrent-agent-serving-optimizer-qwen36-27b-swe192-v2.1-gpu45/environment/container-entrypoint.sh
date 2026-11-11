#!/usr/bin/env bash
set -euo pipefail

# Harbor bind-mounts the verifier output directory before this entrypoint runs.
# Seal it before the untrusted agent is ever started so UID 1000 cannot plant a
# symlink or FIFO at test-stdout.txt for the later root verifier redirection.
# The original bind-root metadata is restored by tests/test.sh after every
# candidate process has been destroyed, allowing the Harbor host to read the
# public reward files again.
verifier_log_root=/logs/verifier
verifier_log_guard=/run/kvbench-log-guard
verifier_log_metadata="$verifier_log_guard/verifier-root.meta"
install -d -o root -g root -m 0755 /logs
if [[ -e "$verifier_log_root" && ( -L "$verifier_log_root" || ! -d "$verifier_log_root" ) ]]; then
  echo "kv-task-entrypoint: verifier log root is not a real directory" >&2
  exit 70
fi
if [[ ! -e "$verifier_log_root" ]]; then
  install -d -o root -g root -m 0700 "$verifier_log_root"
fi
read -r verifier_log_uid verifier_log_gid verifier_log_mode < <(
  stat -c '%u %g %a' "$verifier_log_root"
)
[[ "$verifier_log_uid" =~ ^[0-9]+$ \
  && "$verifier_log_gid" =~ ^[0-9]+$ \
  && "$verifier_log_mode" =~ ^[0-7]{3,4}$ ]] || {
  echo "kv-task-entrypoint: invalid verifier log root metadata" >&2
  exit 70
}
install -d -o root -g root -m 0700 "$verifier_log_guard"
original_umask="$(umask)"
umask 077
printf '%s %s %s\n' \
  "$verifier_log_uid" "$verifier_log_gid" "$verifier_log_mode" \
  > "$verifier_log_metadata.tmp"
chown root:root "$verifier_log_metadata.tmp"
chmod 0400 "$verifier_log_metadata.tmp"
mv -f -- "$verifier_log_metadata.tmp" "$verifier_log_metadata"
chown root:root "$verifier_log_root"
chmod 0700 "$verifier_log_root"
umask "$original_umask"

install -d -o candidate -g candidate /app/submission
install -d -o candidate -g candidate -m 0755 /app/public_evals

# Derive a separate data-plane credential without ever placing the root-only
# admin token (or the derived value) in the candidate environment or command
# line.  The root-owned directory is traversable only by the bridge group; its
# immutable-at-runtime credential is root:bridge 0440.
: "${ADMIN_TOKEN_FILE:=/run/kvbench-admin/token}"
: "${BRIDGE_CREDENTIAL_FILE:=/run/kvbench-bridge/transport.credential}"
for _ in $(seq 1 300); do
  if [[ -r "$ADMIN_TOKEN_FILE" ]]; then
    break
  fi
  sleep 0.1
done
if [[ ! -r "$ADMIN_TOKEN_FILE" ]]; then
  echo "kv-task-entrypoint: trusted admin token was not provisioned" >&2
  exit 70
fi
bridge_uid="$(id -u bridge)"
bridge_gid="$(id -g bridge)"
install -d -o root -g bridge -m 0750 "$(dirname "$BRIDGE_CREDENTIAL_FILE")"
python3 /opt/kvbench/loopback_bridge.py provision \
  --admin-token-file "$ADMIN_TOKEN_FILE" \
  --credential-file "$BRIDGE_CREDENTIAL_FILE" \
  --owner-uid 0 \
  --owner-gid "$bridge_gid"

# The trusted API is a Compose service, while the public contract deliberately
# exposes it only on main's loopback interface.  The UID 900 bridge adds the
# credential and streams bytes with backpressure at the frozen concurrency.
setpriv \
  --reuid "$bridge_uid" \
  --regid "$bridge_gid" \
  --init-groups \
  --no-new-privs \
  python3 /opt/kvbench/loopback_bridge.py serve \
    --credential-file "$BRIDGE_CREDENTIAL_FILE" \
    --upstream-origin http://model-api:8100 \
    --host 127.0.0.1 \
    --port 8100 &
bridge_pid=$!

cleanup() {
  kill "$bridge_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Harbor launches the agent itself as task.toml's `candidate` user. Keep the
# service's long-lived command at UID 0 so the shared verifier can terminate
# every candidate process without stopping the container or its bridge.
exec "$@"
