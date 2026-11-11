#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SFORGE_ENTRY_SCRIPT="${SCRIPT_DIR}/network_preflight.sh"
source "${SCRIPT_DIR}/lib.sh"

require_command docker

# Keep the smoke controls intact if lib.sh has to re-enter through the docker
# group.  These arguments are private to this script; user-facing operation is
# environment-only.
run_codex="${RUN_CODEX:-0}"
host_proxy_host="${QVG_HOST_PROXY_HOST:-127.0.0.1}"
host_proxy_port="${QVG_HOST_PROXY_PORT:-18080}"
docker_proxy_port="${QVG_DOCKER_PROXY_PORT:-19080}"
auth_json="${CODEX_AUTH_JSON_PATH:-${HOME}/.codex/auth.json}"
codex_timeout="${CODEX_SMOKE_TIMEOUT:-300}"
codex_cli_version="${CODEX_CLI_VERSION:-0.148.0}"
codex_model="${MODEL:-}"
codex_reasoning_effort="ultra"
run_id="${PREFLIGHT_RUN_ID:-sv4-$(date -u +%Y%m%d-%H%M%S)-$$}"

while (( $# > 0 )); do
  case "$1" in
    --_qvg-run-codex=*) run_codex="${1#*=}" ;;
    --_qvg-host-proxy-host=*) host_proxy_host="${1#*=}" ;;
    --_qvg-host-proxy-port=*) host_proxy_port="${1#*=}" ;;
    --_qvg-docker-proxy-port=*) docker_proxy_port="${1#*=}" ;;
    --_qvg-auth-json=*) auth_json="${1#*=}" ;;
    --_qvg-codex-timeout=*) codex_timeout="${1#*=}" ;;
    --_qvg-run-id=*) run_id="${1#*=}" ;;
    --_qvg-model=*) codex_model="${1#*=}" ;;
    *) die "unsupported argument: $1" ;;
  esac
  shift
done

ensure_docker_group \
  "--_qvg-run-codex=${run_codex}" \
  "--_qvg-host-proxy-host=${host_proxy_host}" \
  "--_qvg-host-proxy-port=${host_proxy_port}" \
  "--_qvg-docker-proxy-port=${docker_proxy_port}" \
  "--_qvg-auth-json=${auth_json}" \
  "--_qvg-codex-timeout=${codex_timeout}" \
  "--_qvg-run-id=${run_id}" \
  "--_qvg-model=${codex_model}"

for command in curl grep jq python3 ss timeout; do
  require_command "${command}"
done

[[ "${run_codex}" == 0 || "${run_codex}" == 1 ]] \
  || die "RUN_CODEX must be 0 or 1"
if [[ "${run_codex}" == 1 ]]; then
  [[ -n "${codex_model}" ]] || die "RUN_CODEX=1 needs MODEL (the Codex model id to smoke)"
  [[ "${codex_model}" =~ ^[A-Za-z0-9][A-Za-z0-9_./:-]*$ ]] || die "unsafe MODEL: ${codex_model}"
fi
[[ "${host_proxy_host}" == 127.0.0.1 ]] \
  || die "host_proxy_bridge.py fixes the upstream host at 127.0.0.1"
[[ "${host_proxy_port}" == 18080 ]] \
  || die "host_proxy_bridge.py fixes the upstream port at 18080"
[[ "${docker_proxy_port}" =~ ^[0-9]+$ ]] \
  || die "QVG_DOCKER_PROXY_PORT must be numeric"
docker_proxy_port="$((10#${docker_proxy_port}))"
(( docker_proxy_port >= 1 && docker_proxy_port <= 65535 )) \
  || die "QVG_DOCKER_PROXY_PORT must be between 1 and 65535"
[[ "${codex_timeout}" =~ ^[0-9]+$ ]] \
  || die "CODEX_SMOKE_TIMEOUT must be numeric"
codex_timeout="$((10#${codex_timeout}))"
(( codex_timeout >= 30 && codex_timeout <= 1800 )) \
  || die "CODEX_SMOKE_TIMEOUT must be between 30 and 1800 seconds"
[[ "${codex_cli_version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] \
  || die "CODEX_CLI_VERSION must be an exact semantic version"
# The Codex version is chosen by the launcher (launch/common.sh pin or an
# explicit CODEX_CLI_VERSION) and must match the mounted runtime below; the
# agent-free Work image does not constrain it.

# The Work image is agent-free; a host-provided runtime supplies the CLI.
AGENT_RUNTIME_DIR="${SFORGE_AGENT_RUNTIME_DIR:-}"
[[ -n "${AGENT_RUNTIME_DIR}" ]] \
  || die "set SFORGE_AGENT_RUNTIME_DIR (build one with launch/build-agent-runtime.sh)"
runtime_marker="${AGENT_RUNTIME_DIR}/.sforge-agent-runtime.json"
[[ -d "${AGENT_RUNTIME_DIR}" && ! -L "${AGENT_RUNTIME_DIR}" && -f "${runtime_marker}" ]] \
  || die "agent runtime directory is missing or unmarked: ${AGENT_RUNTIME_DIR}"
runtime_kind="$(jq -er '.kind' "${runtime_marker}")"
runtime_agent="$(jq -er '.agent' "${runtime_marker}")"
runtime_version="$(jq -er '.version' "${runtime_marker}")"
[[ "${runtime_kind}" == "sforge-agent-runtime" ]] \
  || die "invalid agent runtime marker: ${runtime_marker}"
[[ "${runtime_agent}" == "codex" ]] \
  || die "preflight requires a codex runtime, found: ${runtime_agent}"
[[ "${runtime_version}" == "${codex_cli_version}" ]] \
  || die "agent runtime version ${runtime_version} differs from the pinned ${codex_cli_version}"
export SFORGE_AGENT_RUNTIME_DIR="${AGENT_RUNTIME_DIR}"

[[ "${run_id}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,35}$ ]] \
  || die "PREFLIGHT_RUN_ID must be 1-36 safe filename characters"

host_bridge_source="${SCRIPT_DIR}/host_proxy_bridge.py"
restricted_proxy_source="${SCRIPT_DIR}/restricted_connect_proxy.py"
[[ -f "${host_bridge_source}" && -f "${restricted_proxy_source}" ]] \
  || die "missing task-owned proxy source"
docker image inspect "${WORK_IMAGE}" >/dev/null \
  || die "missing H100 scratch v4 Work image: ${WORK_IMAGE}"
# The agent-free image resolves codex through the read-only host runtime
# mount, exactly as SForge injects it (BASH_ENV PATH injection).
actual_codex_version="$(
  docker run --rm --network none \
    --volume "${AGENT_RUNTIME_DIR}:/opt/agent-runtime:ro" \
    --env BASH_ENV=/opt/agent-runtime/env.sh \
    --entrypoint bash "${WORK_IMAGE}" -c 'codex --version'
)"
[[ "${actual_codex_version}" == "codex-cli ${codex_cli_version}" ]] \
  || die "mounted runtime Codex mismatch: ${actual_codex_version}; expected ${codex_cli_version}"

umask 077
preflight_parent="${LOGS_DIR}/network-preflight"
run_root="${preflight_parent}/${run_id}"
mkdir -p -- "${preflight_parent}"
mkdir -- "${run_root}" || die "preflight output already exists: ${run_root}"
preflight_log="${run_root}/preflight.log"
result_path="${run_root}/result.json"
bridge_log="${run_root}/host-bridge.log"
restricted_proxy_log="${run_root}/restricted-proxy.log"

network_name="qvg-sv4-pf-net-${run_id}"
sidecar_name="qvg-sv4-pf-proxy-${run_id}"
bridge_probe_name="qvg-sv4-pf-bridge-${run_id}"
isolated_probe_name="qvg-sv4-pf-work-${run_id}"
auth_stage_name="qvg-sv4-pf-auth-stage-${run_id}"
auth_check_name="qvg-sv4-pf-auth-check-${run_id}"
codex_name="qvg-sv4-pf-codex-${run_id}"
auth_volume="qvg-sv4-pf-auth-${run_id}"

started_at="$(date -u +%FT%TZ)"
stage="initialization"
preflight_status="failed"
host_proxy_http_code=""
bridge_http_code=""
full_chain_http_code=""
github_connect_status=""
host_proxy_reachable=false
bridge_reachable=false
sidecar_healthy=false
direct_egress_blocked=false
github_blocked=false
full_chain_reachable=false
codex_passed=false
bridge_gateway=""
proxy_ip=""
bridge_pid=""
network_created=0
sidecar_created=0
bridge_probe_created=0
isolated_probe_created=0
auth_stage_created=0
auth_check_created=0
codex_created=0
auth_volume_created=0

log() {
  printf '[%s] %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "${preflight_log}"
}

write_result() {
  local exit_code="$1" completed_at
  completed_at="$(date -u +%FT%TZ)"
  jq -n \
    --arg status "${preflight_status}" \
    --argjson exit_code "${exit_code}" \
    --arg stage "${stage}" \
    --arg run_id "${run_id}" \
    --arg started_at "${started_at}" \
    --arg completed_at "${completed_at}" \
    --arg host_proxy_host "${host_proxy_host}" \
    --argjson host_proxy_port "${host_proxy_port}" \
    --arg host_proxy_http_code "${host_proxy_http_code}" \
    --argjson host_proxy_reachable "${host_proxy_reachable}" \
    --arg bridge_gateway "${bridge_gateway}" \
    --argjson docker_proxy_port "${docker_proxy_port}" \
    --arg bridge_http_code "${bridge_http_code}" \
    --argjson bridge_reachable "${bridge_reachable}" \
    --arg proxy_ip "${proxy_ip}" \
    --argjson sidecar_healthy "${sidecar_healthy}" \
    --argjson direct_egress_blocked "${direct_egress_blocked}" \
    --arg github_connect_status "${github_connect_status}" \
    --argjson github_blocked "${github_blocked}" \
    --arg full_chain_http_code "${full_chain_http_code}" \
    --argjson full_chain_reachable "${full_chain_reachable}" \
    --argjson codex_requested "$([[ "${run_codex}" == 1 ]] && printf true || printf false)" \
    --argjson codex_passed "${codex_passed}" \
    --arg codex_cli_version "${codex_cli_version}" \
    --arg codex_model "${codex_model}" \
    '{status:$status,exit_code:$exit_code,stage:$stage,run_id:$run_id,
      started_at:$started_at,completed_at:$completed_at,
      host_proxy:{host:$host_proxy_host,port:$host_proxy_port,
        reachable:$host_proxy_reachable,http_code:$host_proxy_http_code},
      docker_bridge:{gateway:$bridge_gateway,port:$docker_proxy_port,
        reachable:$bridge_reachable,http_code:$bridge_http_code},
      restricted_sidecar:{ip:$proxy_ip,healthy:$sidecar_healthy,
        github_connect_status:$github_connect_status,
        github_connect_blocked:$github_blocked},
      direct_external_egress_blocked:$direct_egress_blocked,
      api_openai_full_chain:{reachable:$full_chain_reachable,
        http_code:$full_chain_http_code},
      codex:{requested:$codex_requested,
        cli_version:(if $codex_requested then $codex_cli_version else null end),
        model:(if $codex_requested then $codex_model else null end),
        reasoning_effort:(if $codex_requested then "ultra" else null end),
        mode:(if $codex_requested then "exec_non_interactive" else null end),
        passed:(if $codex_requested then $codex_passed else null end)}}' \
    >"${result_path}"
}

cleanup() {
  local exit_code="$?"
  trap - EXIT INT TERM
  set +e

  if (( codex_created )); then
    docker container rm -f "${codex_name}" >/dev/null 2>&1 || true
  fi
  if (( auth_check_created )); then
    docker container rm -f "${auth_check_name}" >/dev/null 2>&1 || true
  fi
  if (( auth_stage_created )); then
    docker container rm -f "${auth_stage_name}" >/dev/null 2>&1 || true
  fi
  if (( isolated_probe_created )); then
    docker container rm -f "${isolated_probe_name}" >/dev/null 2>&1 || true
  fi
  if (( bridge_probe_created )); then
    docker container rm -f "${bridge_probe_name}" >/dev/null 2>&1 || true
  fi
  if (( sidecar_created )); then
    docker logs "${sidecar_name}" >"${restricted_proxy_log}" 2>&1 || true
    docker container rm -f "${sidecar_name}" >/dev/null 2>&1 || true
  fi
  if (( network_created )); then
    docker network rm "${network_name}" >/dev/null 2>&1 || true
  fi
  if (( auth_volume_created )); then
    docker volume rm "${auth_volume}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${bridge_pid}" ]] && kill -0 "${bridge_pid}" 2>/dev/null; then
    kill "${bridge_pid}" 2>/dev/null || true
    wait "${bridge_pid}" 2>/dev/null || true
  fi

  if [[ "${preflight_status}" != passed ]]; then
    exit_code="$((exit_code == 0 ? 1 : exit_code))"
  fi
  write_result "${exit_code}" || true
  if [[ "${preflight_status}" == passed ]]; then
    log "PASS: network preflight completed; result=${result_path}"
  else
    log "FAIL: network preflight stopped at ${stage}; result=${result_path}"
  fi
  return "${exit_code}"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

resource_must_be_absent() {
  local kind="$1" name="$2"
  if docker "${kind}" inspect "${name}" >/dev/null 2>&1; then
    die "refusing to replace existing Docker ${kind}: ${name}"
  fi
}

valid_http_code() {
  [[ "$1" =~ ^[0-9]{3}$ && "$1" != 000 ]]
}

probe_api_via_proxy() {
  local proxy_url="$1"
  curl --silent --show-error --noproxy '' \
    --connect-timeout 5 --max-time 20 \
    --proxy "${proxy_url}" --output /dev/null \
    --write-out '%{http_code}' \
    https://api.openai.com/v1/models
}

# The loopback HTTP-to-SOCKS service occasionally leaves one CONNECT attempt
# waiting even though the service itself remains healthy.  Treat that as a
# transient tunnel failure, not as proof that the fixed chain is unavailable.
# Three independent, short attempts remain fail-closed, bound every individual
# wait, and make a transient tunnel stall observable in the diagnostics log.
probe_host_api_with_retries() {
  local proxy_url="$1" diagnostics="$2" attempt http_code=""
  for attempt in 1 2 3; do
    if http_code="$(probe_api_via_proxy "${proxy_url}" 2>>"${diagnostics}")" \
        && valid_http_code "${http_code}"; then
      printf '%s' "${http_code}"
      return 0
    fi
    printf 'attempt=%s http_code=%s\n' \
      "${attempt}" "${http_code:-none}" >>"${diagnostics}"
    (( attempt == 3 )) || sleep 1
  done
  printf '%s' "${http_code}"
  return 1
}

probe_container_api_with_retries() {
  local container="$1" proxy_url="$2" diagnostics="$3"
  local attempt http_code=""
  for attempt in 1 2 3; do
    if http_code="$(
      docker exec "${container}" curl --silent --show-error --noproxy '' \
        --connect-timeout 5 --max-time 20 \
        --proxy "${proxy_url}" --output /dev/null \
        --write-out '%{http_code}' \
        https://api.openai.com/v1/models \
        2>>"${diagnostics}"
    )" && valid_http_code "${http_code}"; then
      printf '%s' "${http_code}"
      return 0
    fi
    printf 'attempt=%s http_code=%s\n' \
      "${attempt}" "${http_code:-none}" >>"${diagnostics}"
    (( attempt == 3 )) || sleep 1
  done
  printf '%s' "${http_code}"
  return 1
}

stage="Docker resource collision checks"
resource_must_be_absent network "${network_name}"
for name in \
  "${sidecar_name}" "${bridge_probe_name}" "${isolated_probe_name}" \
  "${auth_stage_name}" "${auth_check_name}" "${codex_name}"; do
  resource_must_be_absent container "${name}"
done
if [[ "${run_codex}" == 1 ]]; then
  resource_must_be_absent volume "${auth_volume}"
fi

stage="Docker bridge discovery"
bridge_gateway="$(
  docker network inspect bridge \
    --format '{{with index .IPAM.Config 0}}{{.Gateway}}{{end}}'
)"
python3 -c \
  'import ipaddress,sys; a=ipaddress.ip_address(sys.argv[1]); assert a.version == 4 and not a.is_unspecified' \
  "${bridge_gateway}" \
  || die "Docker bridge has no usable IPv4 gateway"

stage="host port collision check"
if [[ -n "$(ss -H -ltn "sport = :${docker_proxy_port}")" ]]; then
  die "QVG_DOCKER_PROXY_PORT is already listening: ${docker_proxy_port}"
fi

stage="host loopback proxy probe"
log "Checking fixed host proxy ${host_proxy_host}:${host_proxy_port}"
if ! host_proxy_http_code="$(
  probe_host_api_with_retries \
    "http://${host_proxy_host}:${host_proxy_port}" \
    "${run_root}/host-proxy-probe.log"
)" || ! valid_http_code "${host_proxy_http_code}"; then
  die "host proxy did not return an HTTP response from api.openai.com"
fi
host_proxy_reachable=true

stage="host bridge startup"
log "Starting task bridge ${bridge_gateway}:${docker_proxy_port} -> ${host_proxy_host}:${host_proxy_port}"
python3 "${host_bridge_source}" \
  --listen-host "${bridge_gateway}" --listen-port "${docker_proxy_port}" \
  >"${bridge_log}" 2>&1 &
bridge_pid="$!"
bridge_ready=0
for _ in {1..30}; do
  if ! kill -0 "${bridge_pid}" 2>/dev/null; then
    break
  fi
  if python3 -c \
      'import socket,sys; s=socket.create_connection((sys.argv[1],int(sys.argv[2])),1); s.close()' \
      "${bridge_gateway}" "${docker_proxy_port}" >/dev/null 2>&1; then
    bridge_ready=1
    break
  fi
  sleep 0.2
done
(( bridge_ready == 1 )) || die "task host bridge did not become reachable"

stage="internal network startup"
docker network create --driver bridge --internal \
  --label "mlsysforge.task=${TASK_ID}" \
  --label "mlsysforge.preflight=${run_id}" \
  "${network_name}" >/dev/null
network_created=1

stage="restricted sidecar startup"
sidecar_created=1
docker run --detach --rm --name "${sidecar_name}" --network bridge \
  --label "mlsysforge.task=${TASK_ID}" \
  --label "mlsysforge.preflight=${run_id}" \
  --cap-drop ALL --security-opt no-new-privileges:true \
  --env NVIDIA_VISIBLE_DEVICES=void \
  --mount "type=bind,src=${restricted_proxy_source},dst=/opt/mlsysforge/restricted_proxy.py,readonly" \
  --entrypoint python3 "${WORK_IMAGE}" \
  /opt/mlsysforge/restricted_proxy.py \
  --judge-port 1 --judge-forward-port 18384 \
  --upstream-proxy-host "${bridge_gateway}" \
  --upstream-proxy-port "${docker_proxy_port}" >/dev/null
docker network connect --alias runtime-proxy "${network_name}" "${sidecar_name}"
proxy_ip="$(
  docker inspect "${sidecar_name}" \
    | jq -er --arg network "${network_name}" \
      '.[0].NetworkSettings.Networks[$network].IPAddress'
)"

stage="host bridge container probe"
bridge_probe_created=1
docker run --detach --rm --name "${bridge_probe_name}" --network bridge \
  --user agent --cap-drop ALL --security-opt no-new-privileges:true \
  --env NVIDIA_VISIBLE_DEVICES=void --entrypoint sleep \
  "${WORK_IMAGE}" infinity >/dev/null
if ! bridge_http_code="$(
  probe_container_api_with_retries \
    "${bridge_probe_name}" \
    "http://${bridge_gateway}:${docker_proxy_port}" \
    "${run_root}/docker-bridge-probe.log"
)" || ! valid_http_code "${bridge_http_code}"; then
  die "Docker container could not reach the task host bridge after bounded retries"
fi
bridge_reachable=true
docker container rm -f "${bridge_probe_name}" >/dev/null
bridge_probe_created=0

stage="isolated Work probe startup"
isolated_probe_created=1
docker run --detach --rm --name "${isolated_probe_name}" \
  --network "${network_name}" --user agent \
  --cap-drop ALL --security-opt no-new-privileges:true \
  --env NVIDIA_VISIBLE_DEVICES=void --entrypoint sleep \
  "${WORK_IMAGE}" infinity >/dev/null

stage="restricted sidecar health probe"
health_body=""
for _ in {1..30}; do
  if health_body="$(
    docker exec "${isolated_probe_name}" curl --silent --show-error \
      --connect-timeout 1 --max-time 2 "http://${proxy_ip}:3128/health" \
      2>>"${run_root}/sidecar-health.log"
  )" && [[ "${health_body}" == ok ]]; then
    sidecar_healthy=true
    break
  fi
  sleep 0.2
done
[[ "${sidecar_healthy}" == true ]] || die "restricted sidecar /health failed"

stage="direct external egress rejection"
if docker exec "${isolated_probe_name}" env \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    curl --silent --show-error --noproxy '*' \
      --connect-timeout 3 --max-time 8 --output /dev/null \
      https://example.com \
      >"${run_root}/direct-egress.log" 2>&1; then
  die "isolated Work container unexpectedly has direct public egress"
fi
direct_egress_blocked=true

stage="GitHub CONNECT rejection"
github_connect_status="$(
  docker exec "${isolated_probe_name}" python3 -c '
import socket
import sys

s = socket.create_connection((sys.argv[1], int(sys.argv[2])), 5)
s.settimeout(5)
s.sendall(
    b"CONNECT github.com:443 HTTP/1.1\r\n"
    b"Host: github.com:443\r\n\r\n"
)
data = b""
while b"\r\n" not in data and len(data) < 4096:
    chunk = s.recv(4096)
    if not chunk:
        break
    data += chunk
s.close()
print(data.split(b"\r\n", 1)[0].decode("ascii", "strict"))
' "${proxy_ip}" 3128
)"
printf '%s\n' "${github_connect_status}" >"${run_root}/github-connect.log"
[[ "${github_connect_status}" == "HTTP/1.1 403 Forbidden" ]] \
  || die "restricted sidecar did not reject github.com CONNECT with 403"
github_blocked=true

stage="full proxy-chain API probe"
sidecar_proxy_url="http://${proxy_ip}:3128"
if ! full_chain_http_code="$(
  probe_container_api_with_retries \
    "${isolated_probe_name}" "${sidecar_proxy_url}" \
    "${run_root}/full-chain-api.log"
)" || ! valid_http_code "${full_chain_http_code}"; then
  die "api.openai.com returned no HTTP response through the full proxy chain after bounded retries"
fi
full_chain_reachable=true
log "Full chain reached api.openai.com (HTTP ${full_chain_http_code}; 401 is acceptable)"

if [[ "${run_codex}" == 1 ]]; then
  stage="Codex auth validation"
  assert_private_auth_file "${auth_json}"

  docker volume create \
    --label "mlsysforge.task=${TASK_ID}" \
    --label "mlsysforge.preflight=${run_id}" \
    "${auth_volume}" >/dev/null
  auth_volume_created=1

  # The host user and image agent need not share a numeric UID, and Docker may
  # use user-namespace remapping.  Let the launching shell read the validated
  # 0600 file into container stdin; no credential bytes enter argv or logs.
  # The final Codex container mounts only the staged auth.json subpath read-only.
  auth_stage_created=1
  docker run --rm --interactive --name "${auth_stage_name}" \
    --network none --user root \
    --cap-drop ALL --cap-add CHOWN \
    --security-opt no-new-privileges:true \
    --mount "type=volume,src=${auth_volume},dst=/run/codex-auth" \
    --entrypoint sh "${WORK_IMAGE}" -c '
      umask 077
      cat > /run/codex-auth/auth.json
      chmod 600 /run/codex-auth/auth.json
      chown agent:agent /run/codex-auth/auth.json
      test "$(stat -c %a /run/codex-auth/auth.json)" = 600
    ' <"${auth_json}"
  auth_stage_created=0

  auth_check_created=1
  docker run --rm --name "${auth_check_name}" --network none --user agent \
    --cap-drop ALL --security-opt no-new-privileges:true \
    --mount "type=volume,src=${auth_volume},dst=/home/agent/.codex/auth.json,volume-subpath=auth.json,readonly" \
    --entrypoint sh "${WORK_IMAGE}" -c '
      test -r /home/agent/.codex/auth.json
      test ! -w /home/agent/.codex/auth.json
      test "$(stat -c %a /home/agent/.codex/auth.json)" = 600
    '
  auth_check_created=0

  stage="non-interactive Codex smoke"
  log "Running image-pinned Codex non-interactively as agent"
  codex_created=1
  if ! timeout --foreground "${codex_timeout}" \
      docker run --rm --name "${codex_name}" \
        --network "${network_name}" --user agent \
        --cap-drop ALL --security-opt no-new-privileges:true \
        --env NVIDIA_VISIBLE_DEVICES=void \
        --env "HTTP_PROXY=${sidecar_proxy_url}" \
        --env "HTTPS_PROXY=${sidecar_proxy_url}" \
        --env "http_proxy=${sidecar_proxy_url}" \
        --env "https_proxy=${sidecar_proxy_url}" \
        --env "BUNDLE_HTTP_PROXY=${sidecar_proxy_url}" \
        --env "BUNDLE_HTTPS_PROXY=${sidecar_proxy_url}" \
        --env ALL_PROXY= --env all_proxy= \
        --env NO_PROXY=127.0.0.1,localhost \
        --env no_proxy=127.0.0.1,localhost \
        --mount "type=volume,src=${auth_volume},dst=/home/agent/.codex/auth.json,volume-subpath=auth.json,readonly" \
        --volume "${AGENT_RUNTIME_DIR}:/opt/agent-runtime:ro" \
        --env BASH_ENV=/opt/agent-runtime/env.sh \
        --workdir /workspace --entrypoint bash "${WORK_IMAGE}" \
        -c 'exec codex "$@"' codex \
        --search exec --model "${codex_model}" \
        --config "model_reasoning_effort=\"${codex_reasoning_effort}\"" \
        --skip-git-repo-check \
        --dangerously-bypass-approvals-and-sandbox '只回答 OK。' \
        >"${run_root}/codex-exec.log" 2>&1; then
    die "non-interactive Codex smoke failed"
  fi
  codex_created=0
  grep -Fxq 'OK' "${run_root}/codex-exec.log" \
    || die "Codex smoke completed without the requested OK response"
  codex_passed=true
fi

stage="complete"
preflight_status=passed
exit 0
