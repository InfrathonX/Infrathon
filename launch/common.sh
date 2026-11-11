# Shared configuration and helpers for the unified launch/ entrypoints.
# This file is sourced, not executed. It is the single place to bump the
# Codex CLI pin; per-task hardening
# stays in each task's own scripts/ directory, which these entrypoints
# delegate to without bypassing.

# Default Codex CLI version for every launch. Task images are agent-free: the
# CLI comes from the host runtime ../agent_runtimes/codex-<pin> mounted at
# /opt/agent-runtime, so bumping this needs no image rebuild and no task edit
# (launch/bump-codex-version.sh). A config file or the environment may pick a
# different CODEX_CLI_VERSION + SFORGE_AGENT_RUNTIME_DIR pair per run.
CODEX_CLI_VERSION_PIN="0.159.3"
# Claude Code runtime for launchers that accept --agent claude-code
# (../agent_runtimes/claude-code-<pin>, launch/build-agent-runtime.sh).
CLAUDE_CODE_VERSION_PIN="2.1.287"

launch_die() {
  echo "ERROR: $*" >&2
  exit 1
}

# The sharded-Judge tasks use the paper's fixed Judge GPU count. Callers may
# choose which GPUs (indices or UUIDs), never how many. Checks JUDGE_GPUS
# (and WORK_GPU when set) as strings; the task scripts then resolve, lock and
# re-check the physical devices.
launch_require_judge_gpus() {
  local want="$1" raw="${JUDGE_GPUS:-}" sel
  local -a sels=()
  local -A seen=()
  [[ -n "${raw}" ]] \
    || launch_die "JUDGE_GPUS must name exactly ${want} Judge GPUs (v1 uses exactly ${want} Judge GPUs, the paper's setup)"
  [[ "${raw}" =~ ^([0-9]+|GPU-[A-Za-z0-9-]+)(,([0-9]+|GPU-[A-Za-z0-9-]+))*$ ]] \
    || launch_die "Judge GPUs must be comma-separated GPU indices or UUIDs: ${raw}"
  IFS=, read -r -a sels <<<"${raw}"
  (( ${#sels[@]} == want )) \
    || launch_die "v1 uses exactly ${want} Judge GPUs (the paper's setup); got ${#sels[@]}: ${raw}"
  for sel in "${sels[@]}"; do
    [[ "${sel}" =~ ^[0-9]+$ ]] && sel="$((10#${sel}))"
    [[ -z "${seen[${sel}]:-}" ]] || launch_die "Judge GPUs name ${sel} twice: ${raw}"
    seen[${sel}]=1
  done
  if [[ -n "${WORK_GPU:-}" ]]; then
    sel="${WORK_GPU}"
    [[ "${sel}" =~ ^[0-9]+$ ]] && sel="$((10#${sel}))"
    [[ -z "${seen[${sel}]:-}" ]] || launch_die "the Work GPU ${WORK_GPU} is also a Judge GPU"
  fi
}

# True when a task entry script exports its own container-side proxy, which
# makes the ambient http_proxy irrelevant for that run. Matches only a real
# uncommented assignment, so a mention in a comment does not disarm the guard.
task_entry_sets_own_proxy() {
  local entry="$1"
  [[ -f "${entry}" ]] || return 1
  grep -Eq '^[[:space:]]*export[[:space:]]+SFORGE_HTTP(S)?_PROXY=' "${entry}"
}

launch_repo_root() {
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd
}

# Every host-specific setting these entrypoints need can come from a config
# file instead of being exported by hand, so a fresh machine is configured in
# one place rather than in each caller's shell history.
#
# Discovery order (first hit wins):
#   1. --config FILE          (per-invocation, highest precedence)
#   2. $MLSYSFORGE_CONFIG
#   3. launch/mlsysforge.env   next to this file
#   4. launch/mlsysforge.json  next to this file
#
# A .env file is `KEY=value` lines with `#` comments; a .json file is a flat
# object of the same keys. Either way the keys are the real variable names
# (SFORGE_VERSION_STORE_ROOT, CODEX_AUTH_JSON_PATH, ...), and a value is
# applied only when that variable is not already set. Precedence is therefore
# command-line flag > environment > config file > built-in default, so a
# config file never silently overrides what the caller asked for.
# The SFORGE_LLM_JUDGE_* keys configure the opt-in LLM integrity review that
# the Judge server runs before the GPU Judge (see docs/LLM_JUDGE.md).
# The SFORGE_*CPUSET* keys are the opt-in CPU pinning policy (see
# export_cpuset_policy below and docs/CPU_AFFINITY.md).
# INFRATHON_ASSETS_ROOT, INFRATHON_HARDWARE and HF_HUB_CACHE locate the Infrathon assets
# (scripts/infrathon/README.md) and say which hardware type this host is.
# The keys from HARNESS_ROOT on are the Miles task family's host locations;
# scripts/miles_common/host_config.py reads them from the same file.
LAUNCH_CONFIG_KEYS=(
  SFORGE_VERSION_STORE_ROOT
  SFORGE_VERSION_STORE_OPTIONAL
  SFORGE_VERSION_STORE_SEARCH_PATH
  SFORGE_AGENT_RUNTIME_DIR
  SFORGE_HTTP_PROXY
  SFORGE_HTTPS_PROXY
  SFORGE_NO_PROXY
  CODEX_AUTH_JSON_PATH
  CLAUDE_OAUTH_TOKEN_FILE
  CODEX_CLI_VERSION
  WORK_GPU
  JUDGE_GPU
  JUDGE_PORT
  MODEL
  MODEL_API_BASE_URL
  MODEL_API_KEY_FILE
  REASONING_EFFORT
  TASK_TIMEOUT
  KVBENCH_VLLM_BASE_URL
  KVBENCH_HIDDEN_BASELINE_PATH
  KVBENCH_PUBLIC_BASELINE_PATH
  SFORGE_KVBENCH_NONCE_SOURCE
  LLADA_MODEL_REPO
  QVG_ARTIFACT_ROOT
  GKD_GPU_INDICES
  GKD_JUDGE_GPU_INDICES
  ORPO_GPU_INDICES
  ORPO_GPU_UUIDS
  ORPO_JUDGE_GPU_INDICES
  ORPO_JUDGE_GPU_UUIDS
  ORPO_MODEL_HOST_PATH
  ORPO_DATA_HOST_ROOT
  ORPO_PRIVATE_DATA_HOST_ROOT
  SDV2_ASSETS_ROOT
  SDV2_GPU_INDICES
  SDV2_JUDGE_GPU_INDICES
  FLUXMOE_GPU_INDICES
  FLUXMOE_JUDGE_GPU_INDICES
  FLUXMOE_PAUSE_WORK_FOR_JUDGE
  SFORGE_CPUSET
  SFORGE_WORK_CPUSET
  SFORGE_JUDGE_CPUSET
  SFORGE_CPUSET_MEMS
  SFORGE_CPUSET_RESERVE
  SFORGE_CPUSET_UNDERSIZE
  SFORGE_RESOURCE_PROFILE
  INFRATHON_ASSETS_ROOT
  INFRATHON_HARDWARE
  HF_HUB_CACHE
  SFORGE_LLM_JUDGE_MODE
  SFORGE_LLM_JUDGE_ENGINE
  SFORGE_LLM_JUDGE_AGENT
  SFORGE_LLM_JUDGE_AGENT_RUNTIME_DIR
  SFORGE_LLM_JUDGE_CODEX_AUTH_JSON
  SFORGE_LLM_JUDGE_MODEL
  SFORGE_LLM_JUDGE_REASONING_EFFORT
  SFORGE_LLM_JUDGE_API_BASE_URL
  SFORGE_LLM_JUDGE_API_KEY_FILE
  SFORGE_LLM_JUDGE_IMAGE
  SFORGE_LLM_JUDGE_NETWORK
  SFORGE_LLM_JUDGE_EGRESS_ALLOW
  SFORGE_LLM_JUDGE_ON_ERROR
  SFORGE_LLM_JUDGE_VOTES
  SFORGE_LLM_JUDGE_MIN_CONFIDENCE
  SFORGE_LLM_JUDGE_REVIEW_TIMEOUT
  SFORGE_LLM_JUDGE_CONCURRENCY
  SFORGE_LLM_JUDGE_CPU_LIMIT
  SFORGE_LLM_JUDGE_MEM_LIMIT
  SFORGE_LLM_JUDGE_ANNOUNCE
  HARNESS_ROOT
  MILES_UPSTREAM
  MD_UPSTREAM
  MILES_DATA_ROOT
  MILES_SCRATCH_ROOT
  MILES_HF_CACHE
  OPCD_ASSETS
  OPCD_MEGATRON_INIT
  OPCD_HF_CACHE
  OPCD_ASSETS_V2
  OPCD_V3_ROOT
  OPCD_ASSETS_V3
  OPD_ASSETS_V2
  OPD_ASSETS
  OPD_MEGATRON_INIT
  OPD_HF_CACHE
  OPD_CALIBRATION_DATA
  OPCD_DATA
  ALFW_ASSETS
  ALFW_MEGATRON_INIT
  ALFW_HF_CACHE
  MDOPD_ASSETS
  MDOPD_PUBLIC_DATA
  MDOPD_PRIVATE_DATA
  MDOPD_V2_ROOT
  MDOPD_V2_SCORERS
  MDOPD_V2_PUBLIC_DATA
  MDOPD_V2_PRIVATE_DATA
  MDOPD_CPU_CORES
  MDWAN_ASSETS
  MDWAN_PUBLIC_DATA
  MDWAN_PRIVATE_DATA
  MDWAN_CPU_CORES
  MILES_OLMO3_PACKAGE
  MILES_OLMO3_WORK_SCRATCH
  MILES_BASE_IMAGE_RECORDS
)

launch_config_path() {
  local explicit="${1:-}"
  if [[ -n "${explicit}" ]]; then
    [[ -f "${explicit}" ]] || launch_die "config file does not exist: ${explicit}"
    printf '%s\n' "${explicit}"; return 0
  fi
  if [[ -n "${MLSYSFORGE_CONFIG:-}" ]]; then
    [[ -f "${MLSYSFORGE_CONFIG}" ]] \
      || launch_die "MLSYSFORGE_CONFIG does not exist: ${MLSYSFORGE_CONFIG}"
    printf '%s\n' "${MLSYSFORGE_CONFIG}"; return 0
  fi
  local dir
  dir="$(launch_repo_root)/launch"
  local candidate
  for candidate in "${dir}/mlsysforge.env" "${dir}/mlsysforge.json"; do
    [[ -f "${candidate}" ]] || continue
    printf '%s\n' "${candidate}"; return 0
  done
  return 1
}

# launch_load_config [FILE]
# Applies config values for any LAUNCH_CONFIG_KEYS variable that is unset or
# empty. Unknown keys are rejected rather than ignored, so a typo surfaces at
# launch instead of behaving as if the setting had never been written.
launch_load_config() {
  local path
  path="$(launch_config_path "${1:-}")" || return 0
  # Command substitution, not process substitution: a process substitution's
  # exit status is not propagated, so a rejected config would look like an
  # empty one and the run would continue with nothing applied.
  local raw
  raw="$(
    python3 - "${path}" "${LAUNCH_CONFIG_KEYS[@]}" <<'PY'
import json, sys
path, allowed = sys.argv[1], set(sys.argv[2:])
data = {}
if path.endswith(".json"):
    with open(path) as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        sys.exit("launch config JSON must be a flat object")
    data = loaded
else:
    for number, raw in enumerate(open(path), start=1):
        text = raw.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("export "):
            text = text[len("export "):].lstrip()
        key, sep, value = text.partition("=")
        if not sep:
            sys.exit(f"{path}:{number}: expected KEY=value")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        data[key.strip()] = value
# Every integrity-review setting (docs/LLM_JUDGE.md) is accepted, not only the listed ones.
unknown = sorted(k for k in set(data) - allowed if not k.startswith("SFORGE_LLM_JUDGE_"))
if unknown:
    sys.exit(f"{path}: unknown config key(s): {', '.join(unknown)}")
for key, value in data.items():
    if value is None:
        continue
    text = str(value)
    if "\n" in text:
        sys.exit(f"{path}: value for {key} must be a single line")
    print(f"{key}={text}")
PY
  )" || launch_die "failed to read launch config: ${path}"

  local line name value
  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    name="${line%%=*}"
    value="${line#*=}"
    [[ -n "${!name:-}" ]] && continue
    export "${name}=${value}"
  done <<<"${raw}"
  export MLSYSFORGE_CONFIG_LOADED="${path}"
}

# Check the --model value: a full model ID (gpt-*, claude-*) passes through
# unchanged; there are no short aliases.
resolve_model_id() {
  case "$1" in
    gpt-*|claude-*) echo "$1" ;;
    *)
      # An OpenAI-compatible endpoint (MODEL_API_BASE_URL) serves whatever
      # ids it likes; the task script checks the same variable.
      if [[ -n "${MODEL_API_BASE_URL:-}" ]]; then echo "$1"
      else launch_die "unknown model '$1' (use a full gpt-*/claude-* model ID, or set MODEL_API_BASE_URL for a custom endpoint)"
      fi ;;
  esac
}

# One physical GPU index -> its canonical whole-GPU UUID. Single-GPU mode names
# its GPU by UUID; --gpu lets a caller name it by index instead.
gpu_uuid_for_index() {
  local index="$1" uuid
  [[ "${index}" =~ ^[0-9]+$ ]] || launch_die "a GPU index is a non-negative integer, got '${index}'"
  uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${index}" 2>/dev/null)" || uuid=""
  uuid="${uuid//[[:space:]]/}"
  [[ "${uuid}" =~ ^GPU-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$ ]] \
    || launch_die "nvidia-smi has no whole-GPU UUID for GPU ${index}"
  printf '%s\n' "${uuid}"
}

# Comma-separated GPU indices -> their UUIDs, comma separated, in the same order.
gpu_uuids_for_indices() {
  local index uuid uuids=()
  local -a indices
  IFS=, read -r -a indices <<< "$1"
  for index in "${indices[@]}"; do
    uuid="$(gpu_uuid_for_index "${index}")" || exit 1
    uuids+=("${uuid}")
  done
  (IFS=,; printf '%s\n' "${uuids[*]}")
}

validate_reasoning_effort() {
  case "$1" in
    max|ultra|xhigh|high|medium|low) ;;
    *) launch_die "unknown reasoning effort '$1' (use max|ultra|xhigh|high|medium|low)" ;;
  esac
}

# Docker cannot reach a proxy bound to the host's loopback, and SForge falls
# back to the ambient http_proxy/HTTP_PROXY when SFORGE_HTTP_PROXY is unset —
# which hands the container a 127.0.0.1 proxy pointing at itself, so every
# outbound request fails with "connection refused". Refuse to start rather
# than let a long run burn its budget retrying.
#
# Fix by running a bridge that republishes the loopback proxy on the Docker
# bridge IP, then exporting SFORGE_HTTP_PROXY at that address:
#
#   python3 tasks/latest/quant-videogen-cross-model-kv-codec-h100-v4-scratch/scripts/host_proxy_bridge.py \
#     --listen-host "$(ip -4 addr show docker0 | grep -oP 'inet \K[0-9.]+')" \
#     --listen-port 19081 &
#   export SFORGE_HTTP_PROXY=http://<docker0-ip>:19081
#   export SFORGE_HTTPS_PROXY="$SFORGE_HTTP_PROXY"
#   export SFORGE_NO_PROXY=host.docker.internal,localhost,127.0.0.1,<docker0-ip>
#
# Pick a port no other run holds; concurrent-serving's own bridge commonly
# owns 19080 and restricts itself to a single peer.
#
# Most tasks here do not need any of that: their entry script stands up a
# restricted CONNECT sidecar on an internal Docker network and exports
# SFORGE_HTTP_PROXY at the sidecar's container IP, overriding whatever the
# ambient environment holds. Pass that entry script with --entry so the guard
# can see it sets its own proxy and step aside; without the flag a
# loopback-only ambient proxy blocks those runs for no reason. The check reads
# the entry script itself rather than a hand-maintained list, so it cannot
# drift out of sync with what the scripts actually do.
assert_container_reachable_proxy() {
  local entry=""
  while (( $# )); do
    case "$1" in
      --entry)
        [[ $# -ge 2 ]] || launch_die "--entry requires a value"
        entry="$2"
        shift 2
        ;;
      *) launch_die "assert_container_reachable_proxy: unknown argument: $1" ;;
    esac
  done
  if [[ -n "${entry}" ]] && task_entry_sets_own_proxy "${entry}"; then
    return 0
  fi

  local proxy="${SFORGE_HTTP_PROXY:-${SFORGE_HTTPS_PROXY:-}}"
  if [[ -z "${proxy}" ]]; then
    # No SForge-specific proxy: SForge will inherit the ambient one.
    proxy="${HTTP_PROXY:-${http_proxy:-${HTTPS_PROXY:-${https_proxy:-}}}}"
    [[ -n "${proxy}" ]] || return 0
    case "${proxy}" in
      *//127.*|*//localhost*|*//::1*|*//[[]::1[]]*)
        launch_die "the ambient proxy ${proxy} is host-loopback only, so the
Work container would resolve it to itself and every request would fail.
Export SFORGE_HTTP_PROXY/SFORGE_HTTPS_PROXY at a Docker-bridge address (see
launch/common.sh), or unset http_proxy/HTTP_PROXY to run without a proxy."
        ;;
    esac
    return 0
  fi
  case "${proxy}" in
    *//127.*|*//localhost*|*//::1*|*//[[]::1[]]*)
      launch_die "SFORGE_HTTP_PROXY=${proxy} is host-loopback only; the Work
container would resolve it to itself. Publish the proxy on the Docker bridge
IP instead (see launch/common.sh)."
      ;;
  esac
}

# Default agent runtime: the marked sibling store's codex runtime at the
# shared pin, unless the caller already selected one.  Agent-free task images
# refuse to start without a runtime, so surface the miss loudly.
export_agent_runtime_dir() {
  if [[ -n "${SFORGE_AGENT_RUNTIME_DIR:-}" ]]; then
    return 0
  fi
  local candidate
  candidate="$(cd -- "$(launch_repo_root)/.." && pwd)/agent_runtimes/codex-${CODEX_CLI_VERSION_PIN}"
  if [[ -f "${candidate}/.sforge-agent-runtime.json" ]]; then
    export SFORGE_AGENT_RUNTIME_DIR="${candidate}"
    return 0
  fi
  echo "WARNING: no agent runtime at ${candidate};" \
    "build one with launch/build-agent-runtime.sh" >&2
}

# The Claude Code counterpart of export_agent_runtime_dir.
export_claude_runtime_dir() {
  [[ -z "${SFORGE_AGENT_RUNTIME_DIR:-}" ]] || return 0
  local candidate
  candidate="$(cd -- "$(launch_repo_root)/.." && pwd)/agent_runtimes/claude-code-${CLAUDE_CODE_VERSION_PIN}"
  [[ -f "${candidate}/.sforge-agent-runtime.json" ]] \
    || launch_die "no Claude Code runtime at ${candidate}; build one with launch/build-agent-runtime.sh --agent claude-code --version ${CLAUDE_CODE_VERSION_PIN}"
  export SFORGE_AGENT_RUNTIME_DIR="${candidate}"
}

# Opt-in CPU pinning of the Work and Judge containers (docs/CPU_AFFINITY.md).
# SForge itself pins every container it starts, between create and start,
# using the CPUs local to that container's own GPUs; the launcher only
# carries the policy there through the environment and never reads a GPU.
# Pinning is off unless one of these is set (flag, environment or config):
#
#   SFORGE_CPUSET            off | auto | CPULIST[@NODELIST], e.g. 8-11,40-43@0
#   SFORGE_WORK_CPUSET       same grammar; overrides SFORGE_CPUSET for Work
#   SFORGE_JUDGE_CPUSET      same grammar; overrides SFORGE_CPUSET for Judge
#   SFORGE_CPUSET_MEMS       bind (default) | none
#   SFORGE_CPUSET_RESERVE    CPULIST kept out of the automatic slices
#   SFORGE_CPUSET_UNDERSIZE  refuse (default) | allow a CPU budget above the slice
CPUSET_LIST_RE='[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*'

# True for a kernel cpulist with ascending ranges ("8-11,40-43").
cpuset_list_valid() {
  local value="$1" part
  [[ "${value}" =~ ^${CPUSET_LIST_RE}$ ]] || return 1
  local IFS=,
  for part in ${value}; do
    (( 10#${part%-*} <= 10#${part#*-} )) || return 1
  done
}

cpuset_policy_valid() {
  case "$1" in
    off|auto) return 0 ;;
    *@*) cpuset_list_valid "${1%%@*}" && cpuset_list_valid "${1#*@}" ;;
    *) cpuset_list_valid "$1" ;;
  esac
}

cpuset_policy_explicit() {
  [[ -n "$1" && "$1" != "off" && "$1" != "auto" ]]
}

# True when two valid cpulists share a CPU.
cpuset_lists_overlap() {
  local -A seen=()
  local part cpu IFS=,
  for part in $1; do
    for (( cpu = 10#${part%-*}; cpu <= 10#${part#*-}; cpu++ )); do
      seen[${cpu}]=1
    done
  done
  for part in $2; do
    for (( cpu = 10#${part%-*}; cpu <= 10#${part#*-}; cpu++ )); do
      [[ -z "${seen[${cpu}]:-}" ]] || return 0
    done
  done
  return 1
}

# cpuset_role_policy work|judge — the policy SForge applies to that role.
cpuset_role_policy() {
  local role_var="SFORGE_${1^^}_CPUSET"
  printf '%s\n' "${!role_var:-${SFORGE_CPUSET:-}}"
}

# export_cpuset_policy [FLAG_VALUE]
# Call right after launch_load_config with the --cpuset value (empty when the
# flag was not given). The flag sets SFORGE_CPUSET only; a per-role value from
# the environment or launch config still overrides it for that role (a
# warning names it). `--cpuset off` also turns off both per-role values, so it
# disables pinning outright.
# Every pinning variable that ends up set is grammar-checked here, so a typo
# fails at launch rather than at the first container start. When a policy is
# active the per-GPU slice table (sysfs only) is printed to the launching
# terminal (with --tmux it is not repeated in the tmux run log).
export_cpuset_policy() {
  local flag="${1:-}" name
  if [[ -n "${flag}" ]]; then
    cpuset_policy_valid "${flag}" \
      || launch_die "--cpuset takes off, auto or CPULIST[@NODELIST] (e.g. 8-11,40-43@0), not '${flag}'"
    export SFORGE_CPUSET="${flag}"
    if [[ "${flag}" == "off" ]]; then
      export SFORGE_WORK_CPUSET=off SFORGE_JUDGE_CPUSET=off
    fi
    for name in SFORGE_WORK_CPUSET SFORGE_JUDGE_CPUSET; do
      [[ -z "${!name:-}" || "${!name}" == "${flag}" ]] \
        || echo "WARNING: ${name}=${!name} (environment or launch config) overrides" \
          "--cpuset ${flag} for that role; use --cpuset off or unset it" >&2
    done
  fi
  for name in SFORGE_CPUSET SFORGE_WORK_CPUSET SFORGE_JUDGE_CPUSET; do
    [[ -z "${!name:-}" ]] || cpuset_policy_valid "${!name}" \
      || launch_die "${name} takes off, auto or CPULIST[@NODELIST] (e.g. 8-11,40-43@0), not '${!name}'"
  done
  case "${SFORGE_CPUSET_MEMS:-}" in
    ""|bind|none) ;;
    *) launch_die "SFORGE_CPUSET_MEMS takes bind or none, not '${SFORGE_CPUSET_MEMS}'" ;;
  esac
  case "${SFORGE_CPUSET_UNDERSIZE:-}" in
    ""|refuse|allow) ;;
    *) launch_die "SFORGE_CPUSET_UNDERSIZE takes refuse or allow, not '${SFORGE_CPUSET_UNDERSIZE}'" ;;
  esac
  [[ -z "${SFORGE_CPUSET_RESERVE:-}" ]] || cpuset_list_valid "${SFORGE_CPUSET_RESERVE}" \
    || launch_die "SFORGE_CPUSET_RESERVE takes a CPULIST (e.g. 0,32), not '${SFORGE_CPUSET_RESERVE}'"
  if [[ "${SFORGE_CPUSET_MEMS:-}" == "none" ]]; then
    for name in SFORGE_CPUSET SFORGE_WORK_CPUSET SFORGE_JUDGE_CPUSET; do
      [[ "${!name:-}" != *@* ]] \
        || launch_die "${name}=${!name} names memory nodes, but SFORGE_CPUSET_MEMS=none binds none; drop @NODELIST or use bind"
    done
  fi

  local work judge
  work="$(cpuset_role_policy work)"
  judge="$(cpuset_role_policy judge)"
  [[ "${work:-off}" != "off" || "${judge:-off}" != "off" ]] || return 0
  echo "CPU pinning: work=${work:-off} judge=${judge:-off}" \
    "mems=${SFORGE_CPUSET_MEMS:-bind} reserve=${SFORGE_CPUSET_RESERVE:-none}" \
    "undersize=${SFORGE_CPUSET_UNDERSIZE:-refuse}"
  # Audit only: SForge derives and verifies each container's slice itself,
  # so a missing or failing table never blocks the launch. Run from this
  # checkout, since `python3 -m` puts the working directory ahead of
  # PYTHONPATH and another checkout's sforge/ must not answer.
  local -a table_args=(table)
  [[ -z "${SFORGE_CPUSET_RESERVE:-}" ]] \
    || table_args+=(--reserve "${SFORGE_CPUSET_RESERVE}")
  local repo table
  repo="$(launch_repo_root)"
  if table="$(cd -- "${repo}" && PYTHONPATH="${repo}" \
      python3 -B -m sforge.harness.host_topology "${table_args[@]}" 2>&1)"; then
    printf '%s\n' "${table}"
  else
    echo "WARNING: per-GPU CPU slice table unavailable (${table##*$'\n'});" \
      "SForge still derives and verifies every container's slice at start" >&2
  fi
}

# Separate-GPU runs (the default) run Work and Judge at the same time on two GPUs, so one
# explicit CPU list for both would make them share CPUs. `auto` is safe there
# (each container gets its own GPU's slice); explicit lists must be per role
# and disjoint. The launcher never reads a GPU, so an explicit list for one
# role is not compared with the other role's `auto` slice.
assert_cpuset_legacy_safe() {
  local work="${SFORGE_WORK_CPUSET:-}" judge="${SFORGE_JUDGE_CPUSET:-}"
  if cpuset_policy_explicit "${SFORGE_CPUSET:-}" && [[ -z "${work}" || -z "${judge}" ]]; then
    launch_die "Work and Judge run concurrently on separate GPUs, so the explicit
SFORGE_CPUSET=${SFORGE_CPUSET} would put both on the same CPUs. Use --cpuset auto,
or set SFORGE_WORK_CPUSET and SFORGE_JUDGE_CPUSET to separate CPU lists."
  fi
  if cpuset_policy_explicit "${work}" && cpuset_policy_explicit "${judge}" \
      && cpuset_lists_overlap "${work%%@*}" "${judge%%@*}"; then
    launch_die "Work and Judge run concurrently on separate GPUs, but
SFORGE_WORK_CPUSET=${work} and SFORGE_JUDGE_CPUSET=${judge} share CPUs."
  fi
}

# Read the single task_id a task directory declares. Every task under
# tasks/latest/ ships exactly one tasks/*.json; anything else is a layout
# error rather than something to guess at.
task_id_for_task_root() {
  local task_root="$1"
  local -a specs=()
  local spec
  while IFS= read -r -d '' spec; do specs+=("${spec}"); done < <(
    find "${task_root}/tasks" -maxdepth 1 -type f -name '*.json' -print0 2>/dev/null
  )
  (( ${#specs[@]} == 1 )) || return 1
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["task_id"])' \
    "${specs[0]}" 2>/dev/null
}

# True when STORE registers TASK_ID (i.e. saving into it will actually work).
version_store_registers_task() {
  local store="$1" task_id="$2"
  python3 - "${store}" "${task_id}" <<'PY' 2>/dev/null
import glob, json, os, sys
store, task_id = sys.argv[1], sys.argv[2]
for path in glob.glob(os.path.join(store, "tasks", "*", "task.json")):
    try:
        if json.load(open(path)).get("task_id") == task_id:
            raise SystemExit(0)
    except (OSError, ValueError):
        continue
raise SystemExit(1)
PY
}

# Resolve the version store a run saves into.
#
# Getting this wrong is silent and expensive: SForge prints one line at
# startup and then keeps no replayable candidates for the whole run. The old
# implementation searched a fixed two-name list next to the repo and took the
# first hit, which on a host with more than one store always selected the
# same one regardless of whether it registered the task being launched.
#
# Resolution order: an explicit SFORGE_VERSION_STORE_ROOT (or the config
# file's value, applied earlier by launch_load_config), then a search of the
# known store locations filtered to those that actually register this task.
# Ambiguity and absence both fail closed; set SFORGE_VERSION_STORE_OPTIONAL=1
# to downgrade absence to a warning for a run that deliberately keeps nothing.
#
# Usage: export_version_store_root [--task-root DIR]
export_version_store_root() {
  local task_root="" task_id=""
  while (( $# )); do
    case "$1" in
      --task-root)
        [[ $# -ge 2 ]] || launch_die "--task-root requires a value"
        task_root="$2"; shift 2 ;;
      *) launch_die "export_version_store_root: unknown argument: $1" ;;
    esac
  done
  [[ -z "${task_root}" ]] || task_id="$(task_id_for_task_root "${task_root}" || true)"

  if [[ -n "${SFORGE_VERSION_STORE_ROOT:-}" ]]; then
    [[ -f "${SFORGE_VERSION_STORE_ROOT}/.sforge-version-store.json" ]] \
      || launch_die "SFORGE_VERSION_STORE_ROOT=${SFORGE_VERSION_STORE_ROOT} is not a
version store (no .sforge-version-store.json)."
    if [[ -n "${task_id}" ]] \
       && ! version_store_registers_task "${SFORGE_VERSION_STORE_ROOT}" "${task_id}"; then
      launch_die "version store ${SFORGE_VERSION_STORE_ROOT} does not register
task '${task_id}', so this run would silently keep no replayable candidates.
Register it by creating <store>/tasks/<task-directory>/task.json containing
{\"schema_version\": 1, \"task_directory\": \"...\", \"task_id\": \"${task_id}\"}."
    fi
    return 0
  fi

  local parent candidate
  local -a found=() eligible=() search=()
  parent="$(cd -- "$(launch_repo_root)/.." && pwd)"
  if [[ -n "${SFORGE_VERSION_STORE_SEARCH_PATH:-}" ]]; then
    # Colon-separated absolute paths, for a host that keeps its store
    # somewhere other than beside the checkout.
    local IFS=:
    read -r -a search <<<"${SFORGE_VERSION_STORE_SEARCH_PATH}"
  else
    for candidate in \
      agent-task-version-submit \
      agent_task_version_submit \
      worktrees/agent-task-version-submit \
      worktrees/agent_task_version_submit
    do
      search+=("${parent}/${candidate}")
    done
  fi
  for candidate in "${search[@]}"; do
    [[ -f "${candidate}/.sforge-version-store.json" ]] || continue
    found+=("${candidate}")
    if [[ -z "${task_id}" ]] \
       || version_store_registers_task "${candidate}" "${task_id}"; then
      eligible+=("${candidate}")
    fi
  done

  if (( ${#eligible[@]} == 1 )); then
    export SFORGE_VERSION_STORE_ROOT="${eligible[0]}"
    return 0
  fi
  if (( ${#eligible[@]} > 1 )); then
    launch_die "more than one version store registers task '${task_id}':
$(printf '  %s\n' "${eligible[@]}")
Refusing to guess. Set SFORGE_VERSION_STORE_ROOT (or put it in the launch
config file) to name the one this run should save into."
  fi
  local detail="no version store found next to ${parent}"
  (( ${#found[@]} == 0 )) || detail="none of these stores registers task '${task_id}':
$(printf '  %s\n' "${found[@]}")"
  if [[ "${SFORGE_VERSION_STORE_OPTIONAL:-}" == "1" ]]; then
    echo "WARNING: ${detail}; candidate saving will be disabled" >&2
    return 0
  fi
  launch_die "${detail}
This run would keep no replayable candidates. Set SFORGE_VERSION_STORE_ROOT
to a store that registers the task, register the task in one of the stores
above, or set SFORGE_VERSION_STORE_OPTIONAL=1 to run without saving."
}

# run_in_tmux SESSION LOG_FILE COMMAND [ARGS...]
#
# A tmux session inherits the tmux *server's* environment, not this shell's,
# so the command is wrapped in `env` carrying our complete environment. That
# keeps detached runs byte-identical to the direct `exec` path: task scripts
# read dozens of task-specific variables (QVG_HOST_PROXY_PORT,
# SFORGE_HTTP_PROXY, LLADA_MODEL_REPO, KVBENCH_VLLM_BASE_URL, ...) and an
# allowlist here would silently drop whichever one was added last.
run_in_tmux() {
  local session="$1" log_file="$2"
  shift 2
  command -v tmux >/dev/null 2>&1 || launch_die "tmux is not installed"
  tmux has-session -t "${session}" 2>/dev/null \
    && launch_die "tmux session already exists: ${session}"
  mkdir -p "$(dirname -- "${log_file}")"

  local -a env_args=()
  local entry name
  while IFS= read -r -d '' entry; do
    name="${entry%%=*}"
    # TMUX/TMUX_PANE would make the new session look nested; everything else
    # is forwarded verbatim.
    [[ "${name}" == "TMUX" || "${name}" == "TMUX_PANE" ]] && continue
    # Skip anything bash exported that is not a valid plain variable name
    # (e.g. exported functions), which `env` cannot re-create.
    [[ "${name}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    env_args+=("${entry}")
  done < <(env -0)

  tmux new-session -d -s "${session}" \
    bash -c "exec env $(printf '%q ' "${env_args[@]}")$(printf '%q ' "$@")>> $(printf '%q' "${log_file}") 2>&1"
  echo "started tmux session '${session}'"
  echo "log:    ${log_file}"
  echo "attach: tmux attach -t ${session}"
}
