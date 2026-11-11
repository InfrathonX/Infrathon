#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER_LIB="${SCRIPT_DIR}/lib.sh"
[[ -f "${LAUNCHER_LIB}" && ! -L "${LAUNCHER_LIB}" ]] || {
  echo "ERROR: launcher lib is missing or unsafe: ${LAUNCHER_LIB}" >&2
  exit 1
}
command -v sha256sum >/dev/null 2>&1 || {
  echo "ERROR: required command is missing: sha256sum" >&2
  exit 1
}
launcher_lib_sha256_before="$(sha256sum -- "${LAUNCHER_LIB}")"
launcher_lib_sha256_before="${launcher_lib_sha256_before%% *}"
source "${SCRIPT_DIR}/lib.sh"
ORIGINAL_ARGS=("$@")

usage() {
  cat <<'EOF'
Usage:
  replay_candidates.sh --origin-run RUN_ID --measurement-set NAME \
    --version-store-root DIR [options]

Required:
  --origin-run, --source-run-id RUN_ID  Agent run containing saved versions
  --measurement-set NAME                Stable evaluator series name
  --version-store-root DIR              Persistent SForge version store

Output/runtime:
  --replay-run-id RUN_ID                Replay output run ID (optional)
  --output-dir DIR                      SForge output root
  --tasks-dir DIR                       Task-spec directory
  --timeout SECONDS                     Per-candidate Judge timeout (default: 3000)

Selection/control:
  --candidate vN|CANDIDATE_ID            Repeatable exact candidate selector
  --from-v N / --to-v N                 Inclusive candidate range
  --every N                             Sample every Nth candidate (last always kept)
  --repeat N                            Extra measurements per candidate (median + spread)
  --resume                              Continue the same replay plan
  --dry-run                             Preview candidates with placeholder runtime identity
  --json                                Print the candidate curve as JSON
  --silent                              Suppress detailed Judge logs

A full replay requires SFORGE_KVBENCH_NONCE_SOURCE. It runs the formal v2.4
preflight, holds the formal-run lease for the serial batch, and enters SForge
through scripts/sforge_runtime_entry.py so trusted Judge mounts remain active.
Dry-run does not reproduce or validate an existing full replay fingerprint.
EOF
}

die_usage() {
  echo "ERROR: $*" >&2
  echo "Run $0 --help for usage." >&2
  exit 2
}

require_value() {
  [[ $# -ge 2 && -n "$2" ]] || die_usage "$1 requires a value"
}

safe_id() {
  [[ "$2" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$ ]] \
    || die_usage "unsafe $1: $2"
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

canonical_existing_dir() {
  local label="$1" path="$2"
  [[ -d "${path}" && ! -L "${path}" ]] \
    || die_usage "${label} is missing or unsafe: ${path}"
  (cd -- "${path}" && pwd -P)
}

canonical_existing_file() {
  local label="$1" path="$2" parent
  [[ -f "${path}" && ! -L "${path}" && -s "${path}" ]] \
    || die "${label} must be a nonempty regular file: ${path}"
  parent="$(cd -- "$(dirname -- "${path}")" && pwd -P)"
  printf '%s/%s\n' "${parent}" "$(basename -- "${path}")"
}

cleanup_private_evidence() {
  if [[ -n "${private_evidence_dir:-}" \
     && "${private_evidence_dir}" == /tmp/sforge-kv-replay-preflight.* \
     && -d "${private_evidence_dir}" && ! -L "${private_evidence_dir}" ]]; then
    [[ -z "${preflight_path:-}" || -L "${preflight_path}" ]] \
      || rm -f -- "${preflight_path}"
    rmdir -- "${private_evidence_dir}" 2>/dev/null || true
  fi
}

origin_run=""
measurement_set=""
replay_run_id=""
version_store_root="${SFORGE_VERSION_STORE_ROOT:-}"
output_dir="${LOGS_DIR}"
tasks_dir="${TASKS_DIR}"
timeout="${EVAL_TIMEOUT:-3000}"
dry_run=0
json_output=0
silent=0
selection_args=()

while (( $# > 0 )); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --origin-run|--source-run-id)
      require_value "$@"
      origin_run="$2"
      shift 2
      ;;
    --measurement-set)
      require_value "$@"
      measurement_set="$2"
      shift 2
      ;;
    --replay-run-id)
      require_value "$@"
      replay_run_id="$2"
      shift 2
      ;;
    --version-store-root)
      require_value "$@"
      version_store_root="$2"
      shift 2
      ;;
    --output-dir|--log-dir)
      require_value "$@"
      output_dir="$2"
      shift 2
      ;;
    --tasks-dir)
      require_value "$@"
      tasks_dir="$2"
      shift 2
      ;;
    --timeout)
      require_value "$@"
      timeout="$2"
      shift 2
      ;;
    --candidate)
      require_value "$@"
      selection_args+=(--candidate "$2")
      shift 2
      ;;
    --from-v|--to-v|--every|--repeat)
      require_value "$@"
      selection_args+=("$1" "$2")
      shift 2
      ;;
    --resume|--dry-run|--json|--silent)
      case "$1" in
        --dry-run) dry_run=1 ;;
        --json) json_output=1 ;;
        --silent) silent=1 ;;
        *) selection_args+=("$1") ;;
      esac
      shift
      ;;
    --)
      shift
      (( $# == 0 )) || die_usage "positional arguments are not supported"
      ;;
    *)
      die_usage "unknown option: $1"
      ;;
  esac
done

[[ -n "${origin_run}" ]] || die_usage "--origin-run is required"
[[ -n "${measurement_set}" ]] || die_usage "--measurement-set is required"
[[ -n "${version_store_root}" ]] || die_usage "--version-store-root is required"
safe_id "origin run ID" "${origin_run}"
safe_id "measurement set" "${measurement_set}"
[[ -z "${replay_run_id}" ]] || safe_id "replay run ID" "${replay_run_id}"
[[ "${timeout}" =~ ^[0-9]+$ && "${timeout}" -ge 2400 ]] \
  || die_usage "--timeout must be an integer of at least 2400 seconds"
[[ -d "${version_store_root}" && ! -L "${version_store_root}" ]] \
  || die_usage "version store is missing or unsafe: ${version_store_root}"
[[ -d "${tasks_dir}" && ! -L "${tasks_dir}" ]] \
  || die_usage "tasks directory is missing or unsafe: ${tasks_dir}"
require_command readlink
version_store_root="$(canonical_existing_dir "version store" "${version_store_root}")"
tasks_dir="$(canonical_existing_dir "tasks directory" "${tasks_dir}")"
if [[ -e "${output_dir}" || -L "${output_dir}" ]]; then
  [[ -d "${output_dir}" && ! -L "${output_dir}" ]] \
    || die_usage "output directory is unsafe: ${output_dir}"
fi
output_dir="$(readlink -m -- "${output_dir}")"
[[ -x "${PYTHON_BIN}" ]] || die "missing MLSys Forge Python: ${PYTHON_BIN}"
require_command sha256sum
launcher_lib_sha256="$(sha256_file "${LAUNCHER_LIB}")"
[[ "${launcher_lib_sha256}" == "${launcher_lib_sha256_before}" ]] \
  || die "launcher lib changed while it was loaded"

RUNTIME_ENTRY="${MIGRATION_ROOT}/scripts/sforge_runtime_entry.py"
PREFLIGHT_HELPER="${MIGRATION_ROOT}/scripts/preflight.py"
LOCK_HELPER="${MIGRATION_ROOT}/scripts/secure_flock_exec.py"
TASK_SPEC="${tasks_dir}/${TASK_ID}.json"
[[ -f "${RUNTIME_ENTRY}" && ! -L "${RUNTIME_ENTRY}" \
   && -f "${PREFLIGHT_HELPER}" && ! -L "${PREFLIGHT_HELPER}" \
   && -f "${LOCK_HELPER}" && ! -L "${LOCK_HELPER}" \
   && -f "${LAUNCHER_LIB}" && ! -L "${LAUNCHER_LIB}" \
   && -f "${TASK_SPEC}" && ! -L "${TASK_SPEC}" ]] \
  || die "task runtime, preflight, or task spec is missing or unsafe"

contexts=(
  "task_runtime_adapter_sha256=$(sha256_file "${RUNTIME_ENTRY}")"
  "replay_launcher_sha256=$(sha256_file "${SCRIPT_DIR}/replay_candidates.sh")"
  "launcher_lib_sha256=${launcher_lib_sha256}"
  "preflight_sha256=$(sha256_file "${PREFLIGHT_HELPER}")"
  "lock_helper_sha256=$(sha256_file "${LOCK_HELPER}")"
  "task_spec_sha256=$(sha256_file "${TASK_SPEC}")"
  "nonce_commitment_sha256=c5bd60bb88e42c0a014bda5466690af92a672b96d2426c2964aa3f8dc8e8ca9c"
  "public_nonce_commitment_sha256=110a9f3a7003e38656aaf99ad87b42aac992da3d0645df9aeb51c42a9bc29e24"
  "oracle_tree_sha256=b8f92b46bcd25529c19d20354be94d620b632e2bce209909f8ae666a71de0e75"
)
unset SFORGE_KVBENCH_REPLAY_PREFLIGHT \
  SFORGE_KVBENCH_REPLAY_PREFLIGHT_SHA256 \
  SFORGE_KVBENCH_REPLAY_PREFLIGHT_HELPER_SHA256

if (( dry_run == 0 )); then
  ensure_docker_group "${ORIGINAL_ARGS[@]}"
  require_sforge
  for command_name in jq mktemp python3; do
    require_command "${command_name}"
  done
  assert_built_images_current
  # Judge-only default (in-tree or asset root, lib.sh); SFORGE_KVBENCH_NONCE_SOURCE overrides it.
  nonce_source="${SFORGE_KVBENCH_NONCE_SOURCE:-$(default_nonce_source)}"
  [[ -n "${nonce_source}" ]] \
    || die "SFORGE_KVBENCH_NONCE_SOURCE must be set"
  nonce_source="$(canonical_existing_file "SFORGE_KVBENCH_NONCE_SOURCE" "${nonce_source}")"
  export SFORGE_KVBENCH_NONCE_SOURCE="${nonce_source}"

  # This batch lease excludes another formal launcher.
  # Each Judge attempt acquires the nested measurement lock itself; holding
  # that lock here would make the runner fail closed against its own parent.
  formal_lock="/tmp/harbor-kv-qwen36-27b-1002/mlsysforge-v24-formal-run.lock"
  if [[ "${KV_REPLAY_LOCKED:-0}" == 1 \
     || -n "${SFORGE_SECURE_LOCK_FD:-}" ]]; then
    [[ "${KV_REPLAY_LOCKED:-0}" == 1 \
       && "${SFORGE_SECURE_LOCK_FD:-}" == 8 ]] \
      || die "inconsistent inherited KV replay lock state"
    python3 "${LOCK_HELPER}" \
      --lock "${formal_lock}" --fd 8 --marker KV_REPLAY_LOCKED \
      --verify-held
  else
    exec python3 "${LOCK_HELPER}" \
      --lock "${formal_lock}" \
      --fd 8 --marker KV_REPLAY_LOCKED -- \
      bash "${SCRIPT_DIR}/replay_candidates.sh" "${ORIGINAL_ARGS[@]}"
  fi

  private_evidence_dir="$(mktemp -d -t sforge-kv-replay-preflight.XXXXXXXX)"
  chmod 0700 "${private_evidence_dir}"
  preflight_path="${private_evidence_dir}/preflight.json"
  trap cleanup_private_evidence EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP
  python3 "${MIGRATION_ROOT}/scripts/preflight.py" \
    --nonce-source "${nonce_source}" \
    --mlsysforge-root "${MLSYSFORGE_ROOT}" \
    --output "${preflight_path}" >/dev/null
  preflight_report_sha="$(sha256_file "${preflight_path}")"
  export SFORGE_KVBENCH_REPLAY_PREFLIGHT="${preflight_path}"
  export SFORGE_KVBENCH_REPLAY_PREFLIGHT_SHA256="${preflight_report_sha}"
  export SFORGE_KVBENCH_REPLAY_PREFLIGHT_HELPER_SHA256="$(sha256_file "${PREFLIGHT_HELPER}")"

  build_manifest="${LOGS_DIR}/build/image-manifest.json"
  contexts+=(
    "preflight_report_sha256=${preflight_report_sha}"
    "controller_image_id=$(jq -er '.public_controller.id' "${build_manifest}")"
    "build_source_digest=$(jq -er '.source_digest' "${build_manifest}")"
    "runtime_boot_identity_sha256=$(jq -er '.runtime_boot_identity_sha256' "${preflight_path}")"
    "hidden_baseline_sha256=$(jq -er '.baseline_file_sha256' "${preflight_path}")"
    "public_baseline_sha256=$(jq -er '.public_baseline_file_sha256' "${preflight_path}")"
    "corpus_manifest_sha256=$(jq -er '.corpus_manifest_sha256' "${preflight_path}")"
    "runtime_sources_tree_sha256=$(jq -er '.trusted_runtime_sources.tree_sha256' "${preflight_path}")"
    "execution_sources_tree_sha256=$(jq -er '.trusted_execution_sources.tree_sha256' "${preflight_path}")"
    "mlsysforge_harness_tree_sha256=$(jq -er '.mlsysforge_harness_sources.tree_sha256' "${preflight_path}")"
    "mlsysforge_root=$(jq -er '.mlsysforge_root' "${preflight_path}")"
    "nested_image_ids_sha256=$(jq -Sc '.nested_image_ids' "${preflight_path}" | sha256sum | awk '{print $1}')"
  )
else
  contexts+=(
    "preflight_report_sha256=unverified-dry-run"
    "controller_image_id=unverified-dry-run"
    "build_source_digest=unverified-dry-run"
    "runtime_boot_identity_sha256=unverified-dry-run"
    "hidden_baseline_sha256=unverified-dry-run"
    "public_baseline_sha256=unverified-dry-run"
    "corpus_manifest_sha256=unverified-dry-run"
    "runtime_sources_tree_sha256=unverified-dry-run"
    "execution_sources_tree_sha256=unverified-dry-run"
    "mlsysforge_harness_tree_sha256=unverified-dry-run"
    "mlsysforge_root=unverified-dry-run"
    "nested_image_ids_sha256=unverified-dry-run"
  )
fi

export SFORGE_TASKS_DIR="${tasks_dir}"
export SFORGE_LOG_DIR="${output_dir}"
export SFORGE_KVBENCH_MIGRATION_ROOT="${MIGRATION_ROOT}"
export SFORGE_KVBENCH_EVAL_ROOT="${SFORGE_KVBENCH_EVAL_ROOT:-${HOME}/.cache/mlsysforge-kvbench-qwen36-swe192-evals}"

command=(
  "${PYTHON_BIN}" "${RUNTIME_ENTRY}"
  --output-dir "${output_dir}"
  --tasks-dir "${tasks_dir}"
)
(( silent == 0 )) || command+=(--silent)
command+=(
  replay-candidates
  --backend docker
  --task "${TASK_ID}"
  --run-id "${origin_run}"
  --measurement-set "${measurement_set}"
  --version-store-root "${version_store_root}"
  --timeout "${timeout}"
  --judge-cpu-limit "${JUDGE_CPU_LIMIT:-4}"
  --judge-mem-limit "${JUDGE_MEM_LIMIT:-8g}"
)
[[ -z "${replay_run_id}" ]] || command+=(--replay-run-id "${replay_run_id}")
for context in "${contexts[@]}"; do
  command+=(--evaluator-context "${context}")
done
command+=("${selection_args[@]}")
(( dry_run == 0 )) || command+=(--dry-run)
(( json_output == 0 )) || command+=(--json)

set +e
"${command[@]}"
command_status=$?
set -e
cleanup_private_evidence
private_evidence_dir=""
trap - EXIT INT TERM HUP
exit "${command_status}"
