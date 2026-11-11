#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

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
  --timeout SECONDS                     Per-candidate Judge timeout (default: 12000)

Selection/control:
  --candidate vN|CANDIDATE_ID            Repeatable exact candidate selector
  --from-v N / --to-v N                 Inclusive candidate range
  --every N                             Sample every Nth candidate (last always kept)
  --repeat N                            Extra measurements per candidate (median + spread)
  --resume                              Continue the same replay plan
  --dry-run                             Preview candidates with placeholder runtime identity
  --json                                Print the candidate curve as JSON
  --silent                              Suppress detailed Judge logs

Full replay uses the task's fixed dual-H100 adapter, frozen model/data
preflight, and one managed Judge GPU lease held for the complete serial batch.
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

cleanup_private_evidence() {
  if [[ -n "${private_evidence_dir:-}" \
     && "${private_evidence_dir}" == /tmp/sforge-orpo-replay-preflight.* \
     && -d "${private_evidence_dir}" && ! -L "${private_evidence_dir}" ]]; then
    local evidence
    for evidence in "${preflight_path:-}" "${runtime_identity_path:-}"; do
      [[ -z "${evidence}" || -L "${evidence}" ]] || rm -f -- "${evidence}"
    done
    rmdir -- "${private_evidence_dir}" 2>/dev/null || true
  fi
}

origin_run=""
measurement_set=""
replay_run_id=""
version_store_root="${SFORGE_VERSION_STORE_ROOT:-}"
output_dir="${LOGS_DIR}"
tasks_dir="${TASK_ROOT}/tasks"
timeout="${EVAL_TIMEOUT:-12000}"
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
[[ "${timeout}" =~ ^[1-9][0-9]*$ ]] || die_usage "--timeout must be positive"
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
SFORGE_PYTHON="${SFORGE_PYTHON:-${REPOSITORY_ROOT}/.venv/bin/python}"
[[ -x "${SFORGE_PYTHON}" ]] || die "missing SForge Python: ${SFORGE_PYTHON}"
require_command sha256sum
launcher_lib_sha256="$(sha256_file "${LAUNCHER_LIB}")"
[[ "${launcher_lib_sha256}" == "${launcher_lib_sha256_before}" ]] \
  || die "launcher lib changed while it was loaded"

RUNTIME_ENTRY="${TASK_ROOT}/scripts/sforge_dual_h100_entry.py"
PREFLIGHT="${TASK_ROOT}/scripts/preflight.py"
LOCK_HELPER="${TASK_ROOT}/scripts/secure_flock_exec.py"
MODEL_IDENTITY_HELPER="${TASK_ROOT}/scripts/model_artifact_identity.py"
RUNTIME_IDENTITY_HELPER="${TASK_ROOT}/scripts/runtime_identity.py"
TASK_SPEC="${tasks_dir}/${TASK_ID}.json"
[[ -f "${RUNTIME_ENTRY}" && ! -L "${RUNTIME_ENTRY}" \
   && -f "${PREFLIGHT}" && ! -L "${PREFLIGHT}" \
   && -f "${LOCK_HELPER}" && ! -L "${LOCK_HELPER}" \
   && -f "${MODEL_IDENTITY_HELPER}" && ! -L "${MODEL_IDENTITY_HELPER}" \
   && -f "${RUNTIME_IDENTITY_HELPER}" && ! -L "${RUNTIME_IDENTITY_HELPER}" \
   && -f "${LAUNCHER_LIB}" && ! -L "${LAUNCHER_LIB}" \
   && -f "${TASK_SPEC}" && ! -L "${TASK_SPEC}" ]] \
  || die "task runtime, preflight, or task spec is missing or unsafe"

gpu_uuids="${ORPO_GPU_UUIDS:?set ORPO_GPU_UUIDS to the two physical GPU UUIDs of the NVLink pair this run owns, comma separated (nvidia-smi --query-gpu=index,uuid --format=csv,noheader)}"
gpu_indices="${ORPO_GPU_INDICES:?set ORPO_GPU_INDICES to the two physical GPU indices of the NVLink pair this run owns, comma separated (nvidia-smi --query-gpu=index,uuid --format=csv,noheader)}"
contexts=(
  "task_runtime_adapter_sha256=$(sha256_file "${RUNTIME_ENTRY}")"
  "replay_launcher_sha256=$(sha256_file "${SCRIPT_DIR}/replay_candidates.sh")"
  "launcher_lib_sha256=${launcher_lib_sha256}"
  "preflight_sha256=$(sha256_file "${PREFLIGHT}")"
  "lock_helper_sha256=$(sha256_file "${LOCK_HELPER}")"
  "model_identity_helper_sha256=$(sha256_file "${MODEL_IDENTITY_HELPER}")"
  "runtime_identity_helper_sha256=$(sha256_file "${RUNTIME_IDENTITY_HELPER}")"
  "task_spec_sha256=$(sha256_file "${TASK_SPEC}")"
  "model_revision=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
  "model_config_sha256=1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197"
  "public_dataset_root_manifest_sha256=a968283e5f7a78e1010ec59c47aeb5002eff87962be05e38a0c20df41e24a900"
  "public_manifest_sha256=ba9816131647aefe0d813fc7c7d3706894988943bc58e92c472667f722528c20"
  "public_tensors_sha256=e1218b35260ea9e7afd048962b53c25257d7f298e2a787628d8f5b640cfc7e79"
  "private_dataset_root_manifest_sha256=ac53d03376b7a874c404a1bd0f5e5f84a8b96db7fc1a2187554c2c286a91c175"
  "private_cross_512_manifest_sha256=554bf1261f72a848de3ac93d21e5c24a6581a52e0e802adea2454fac6d5d1369"
  "private_cross_512_tensors_sha256=2968c2b87f03b9fb2ee57b69d5c8a702ed994cdb16889bb4bbfe40e9d104315b"
  "private_cross_1024_manifest_sha256=b731a9642512720ac116b13ff88ccc8fa8a4751f94c7a7204d792e7e65fb6672"
  "private_cross_1024_tensors_sha256=1d347aa8f5047de37ca7f257439585a7caaff44179bc44c521190328f8883368"
  "private_cross_2048_manifest_sha256=fea2455400c3461f9c5797d4f8288728b1cdfec16d2258a431215fe47b5f429d"
  "private_cross_2048_tensors_sha256=05cc2316e4db9af9bec8f13695a07760ab30a77b4062b99b6baf020c0277a0e8"
  "gpu_uuids=${gpu_uuids}"
)
unset ORPO_REPLAY_EXPECTED_MODEL_MOUNT_SHA256 \
  ORPO_REPLAY_EXPECTED_PRIVATE_DATA_MOUNT_SHA256 \
  ORPO_REPLAY_MODEL_IDENTITY_HELPER_SHA256 \
  ORPO_REPLAY_RUNTIME_IDENTITY_PATH \
  ORPO_REPLAY_RUNTIME_IDENTITY_SHA256 \
  ORPO_REPLAY_RUNTIME_IDENTITY_HELPER_SHA256 \
  ORPO_REPLAY_GPU_INDICES

if (( dry_run == 0 )); then
  require_command docker
  require_command flock
  require_command mktemp
  require_command nvidia-smi
  require_command python3
  if ! docker info >/dev/null 2>&1; then
    [[ "${ORPO_DOCKER_GROUP_REEXEC:-0}" != 1 ]] \
      || die "Docker is still unavailable after docker-group re-exec"
    getent group docker | awk -F: -v user="$(id -un)" '
      $1 == "docker" { n=split($4,m,","); for(i=1;i<=n;i++) if(m[i]==user) ok=1 }
      END { exit(ok ? 0 : 1) }
    ' || die "Docker is unavailable and the current user is not in the docker group"
    printf -v reexec '%q ' env \
      ORPO_DOCKER_GROUP_REEXEC=1 \
      REPOSITORY_ROOT="${REPOSITORY_ROOT}" \
      SFORGE_PYTHON="${SFORGE_PYTHON}" \
      ORPO_GPU_UUIDS="${gpu_uuids}" \
      ORPO_GPU_INDICES="${gpu_indices}" \
      ORPO_MODEL_HOST_PATH="${ORPO_MODEL_HOST_PATH:-}" \
      ORPO_DATA_HOST_ROOT="${ORPO_DATA_HOST_ROOT:-}" \
      ORPO_PRIVATE_DATA_HOST_ROOT="${ORPO_PRIVATE_DATA_HOST_ROOT:-}" \
      SFORGE_AGENT_RUNTIME_DIR="${SFORGE_AGENT_RUNTIME_DIR:-}" \
      SFORGE_VERSION_STORE_ROOT="${SFORGE_VERSION_STORE_ROOT:-}" \
      bash "${SCRIPT_DIR}/replay_candidates.sh" "${ORIGINAL_ARGS[@]}"
    exec sg docker -c "${reexec}"
  fi

  docker image inspect "${WORK_IMAGE}" "${JUDGE_IMAGE}" >/dev/null \
    || die "final Work/Judge images are missing; run scripts/build_images.sh"
  model_repo="${ORPO_MODEL_HOST_PATH:-${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}/models--Qwen--Qwen3-1.7B}"
  # Data defaults: the task's Infrathon assets (assets/fetch.sh) when
  # INFRATHON_ASSETS_ROOT is set, else the host artifact directory.
  # Unset: the default root of assets/fetch.sh, when it exists.
  [[ -n "${INFRATHON_ASSETS_ROOT:-}" || ! -d "${HOME}/.cache/infrathon/assets" ]] \
    || INFRATHON_ASSETS_ROOT="${HOME}/.cache/infrathon/assets"
  if [[ -n "${INFRATHON_ASSETS_ROOT:-}" ]]; then
    default_data_root="${INFRATHON_ASSETS_ROOT}/qwen3-1p7b-orpo-training-optimizer-dual-h100-v4-cross-matrix-seed/public/tldr_preference_v1"
    default_private_data_root="${INFRATHON_ASSETS_ROOT}/qwen3-1p7b-orpo-training-optimizer-dual-h100-v4-cross-matrix-seed/judge/tldr_preference_cross_v4"
  else
    default_data_root="${HOME}/mlsysforge-artifacts/qwen3-orpo/tldr_preference_v1"
    default_private_data_root="${HOME}/mlsysforge-artifacts/qwen3-orpo/tldr_preference_cross_v4"
  fi
  data_root="${ORPO_DATA_HOST_ROOT:-${default_data_root}}"
  private_data_root="${ORPO_PRIVATE_DATA_HOST_ROOT:-${default_private_data_root}}"
  model_repo="$(canonical_existing_dir "model repository" "${model_repo}")"
  data_root="$(canonical_existing_dir "public data root" "${data_root}")"
  private_data_root="$(canonical_existing_dir "private data root" "${private_data_root}")"

  # replay-candidates uses the CLI's import-bound evaluator, not the adapter's
  # managed Work-pause wrapper. This inherited FD is therefore the single
  # lease spanning preflight and the complete serial replay.
  # Same per-pair lease name as sforge_dual_h100_entry._managed_judge_lock_path.
  replay_pair="$(tr ',' '\n' <<<"${gpu_uuids}" | tr '[:upper:]' '[:lower:]' | sort | paste -sd-)"
  replay_lock="/tmp/qwen3-orpo-dual-h100-managed-judge-${replay_pair}.lock"
  if [[ "${ORPO_REPLAY_LOCKED:-0}" == 1 \
     || -n "${SFORGE_SECURE_LOCK_FD:-}" ]]; then
    [[ "${ORPO_REPLAY_LOCKED:-0}" == 1 \
       && "${SFORGE_SECURE_LOCK_FD:-}" == 9 ]] \
      || die "inconsistent inherited ORPO replay lock state"
    python3 "${LOCK_HELPER}" \
      --lock "${replay_lock}" --fd 9 --marker ORPO_REPLAY_LOCKED \
      --verify-held
  else
    exec python3 "${LOCK_HELPER}" \
      --lock "${replay_lock}" \
      --fd 9 --marker ORPO_REPLAY_LOCKED -- \
      bash "${SCRIPT_DIR}/replay_candidates.sh" "${ORIGINAL_ARGS[@]}"
  fi

  private_evidence_dir="$(mktemp -d -t sforge-orpo-replay-preflight.XXXXXXXX)"
  chmod 0700 "${private_evidence_dir}"
  preflight_path="${private_evidence_dir}/preflight.json"
  trap cleanup_private_evidence EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP
  python3 "${TASK_ROOT}/scripts/preflight.py" \
    --gpu-indices "${gpu_indices}" \
    --expected-gpu-uuids "${gpu_uuids}" \
    --model-path "${model_repo}/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e" \
    --data-root "${data_root}" \
    --private-data-root "${private_data_root}" \
    >"${preflight_path}"

  model_identity_json="$(
    python3 "${MODEL_IDENTITY_HELPER}" \
      --model-repo "${model_repo}" \
      --private-data-root "${private_data_root}"
  )"
  model_files_content_sha256="$(
    python3 -c 'import json,sys; print(json.load(sys.stdin)["model_files_content_sha256"])' \
      <<<"${model_identity_json}"
  )"
  model_resolved_blobs_sha256="$(
    python3 -c 'import json,sys; print(json.load(sys.stdin)["model_resolved_blobs_sha256"])' \
      <<<"${model_identity_json}"
  )"
  model_mount_tree_sha256="$(
    python3 -c 'import json,sys; print(json.load(sys.stdin)["model_mount_tree_sha256"])' \
      <<<"${model_identity_json}"
  )"
  private_data_mount_tree_sha256="$(
    python3 -c 'import json,sys; print(json.load(sys.stdin)["private_data_mount_tree_sha256"])' \
      <<<"${model_identity_json}"
  )"

  runtime_identity_path="${private_evidence_dir}/runtime-identity.json"
  IFS=, read -r -a runtime_gpu_indices <<<"${gpu_indices}"
  IFS=, read -r -a runtime_gpu_uuids <<<"${gpu_uuids}"
  runtime_identity_command=(
    python3 "${RUNTIME_IDENTITY_HELPER}"
    --include-topology
    --output "${runtime_identity_path}"
  )
  for index in "${runtime_gpu_indices[@]}"; do
    index="${index//[[:space:]]/}"
    runtime_identity_command+=(--gpu-index "${index}")
  done
  for uuid in "${runtime_gpu_uuids[@]}"; do
    uuid="${uuid//[[:space:]]/}"
    runtime_identity_command+=(--expected-gpu-uuid "${uuid}")
  done
  "${runtime_identity_command[@]}"
  runtime_identity_file_sha256="$(sha256_file "${runtime_identity_path}")"
  host_runtime_identity_sha256="$(
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["identity_sha256"])' \
      "${runtime_identity_path}"
  )"

  export ORPO_MODEL_HOST_PATH="${model_repo}"
  export ORPO_DATA_HOST_ROOT="${data_root}"
  export ORPO_PRIVATE_DATA_HOST_ROOT="${private_data_root}"
  export ORPO_REPLAY_EXPECTED_MODEL_MOUNT_SHA256="${model_mount_tree_sha256}"
  export ORPO_REPLAY_EXPECTED_PRIVATE_DATA_MOUNT_SHA256="${private_data_mount_tree_sha256}"
  export ORPO_REPLAY_MODEL_IDENTITY_HELPER_SHA256="$(sha256_file "${MODEL_IDENTITY_HELPER}")"
  export ORPO_REPLAY_RUNTIME_IDENTITY_PATH="${runtime_identity_path}"
  export ORPO_REPLAY_RUNTIME_IDENTITY_SHA256="${runtime_identity_file_sha256}"
  export ORPO_REPLAY_RUNTIME_IDENTITY_HELPER_SHA256="$(sha256_file "${RUNTIME_IDENTITY_HELPER}")"
  export ORPO_REPLAY_GPU_INDICES="${gpu_indices}"
  contexts+=(
    "model_host_path=${model_repo}"
    "public_data_root=${data_root}"
    "private_data_root=${private_data_root}"
    "model_files_content_sha256=${model_files_content_sha256}"
    "model_resolved_blobs_sha256=${model_resolved_blobs_sha256}"
    "model_mount_tree_sha256=${model_mount_tree_sha256}"
    "private_data_mount_tree_sha256=${private_data_mount_tree_sha256}"
    "host_runtime_identity_sha256=${host_runtime_identity_sha256}"
  )
else
  contexts+=(
    "model_host_path=unverified-dry-run"
    "public_data_root=unverified-dry-run"
    "private_data_root=unverified-dry-run"
    "model_files_content_sha256=unverified-dry-run"
    "model_resolved_blobs_sha256=unverified-dry-run"
    "model_mount_tree_sha256=unverified-dry-run"
    "private_data_mount_tree_sha256=unverified-dry-run"
    "host_runtime_identity_sha256=unverified-dry-run"
  )
fi

export SFORGE_TASKS_DIR="${tasks_dir}"
export SFORGE_LOG_DIR="${output_dir}"
export ORPO_GPU_UUIDS="${gpu_uuids}"

command=(
  "${SFORGE_PYTHON}" "${RUNTIME_ENTRY}"
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
  --judge-cpu-limit 16
  --judge-mem-limit 96g
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
