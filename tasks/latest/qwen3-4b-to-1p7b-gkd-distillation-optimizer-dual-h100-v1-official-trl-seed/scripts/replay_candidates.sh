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
     && "${private_evidence_dir}" == /tmp/sforge-gkd-replay-preflight.* \
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

# The batch pins whichever pair the operator names; no pair belongs to the task.
# An explicit GKD_GPU_UUIDS pin is honored as-is, otherwise the UUIDs are
# resolved from the named indices.
gpu_indices="${GKD_GPU_INDICES:-}"
gpu_uuids="${GKD_GPU_UUIDS:-}"
if [[ -z "${gpu_uuids}" ]]; then
  resolve_gpu_pair "${gpu_indices}"
  gpu_indices="${GPU_PAIR_INDICES}"
  gpu_uuids="${GPU_PAIR_UUIDS}"
fi
[[ -n "${gpu_indices}" ]] \
  || die "set GKD_GPU_INDICES to the two physical GPU indices for this replay (e.g. 4,5)"
contexts=(
  "task_runtime_adapter_sha256=$(sha256_file "${RUNTIME_ENTRY}")"
  "replay_launcher_sha256=$(sha256_file "${SCRIPT_DIR}/replay_candidates.sh")"
  "launcher_lib_sha256=${launcher_lib_sha256}"
  "preflight_sha256=$(sha256_file "${PREFLIGHT}")"
  "lock_helper_sha256=$(sha256_file "${LOCK_HELPER}")"
  "model_identity_helper_sha256=$(sha256_file "${MODEL_IDENTITY_HELPER}")"
  "runtime_identity_helper_sha256=$(sha256_file "${RUNTIME_IDENTITY_HELPER}")"
  "task_spec_sha256=$(sha256_file "${TASK_SPEC}")"
  "student_model_revision=70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
  "student_model_config_sha256=1ddb5b89ebc90dcb417a45c213d818577e65976454d29385c8f6140771d95197"
  "teacher_model_revision=1cfa9a7208912126459214e8b04321603b3df60c"
  "teacher_model_config_sha256=8ba006f74fecfaaeb392872a60f4a480e7ec9860153d2e1b769ec81f9a147f8a"
  "dataset_root_manifest_sha256=7caea1c7a2166a230c5d1ec67f26e8df1470f37b68600aa10cde4d0a6dcdeb95"
  "public_manifest_sha256=e653d0a59a07d7341197263a28424f85940ef633cce1e3a93891095cf2e04956"
  "public_tensors_sha256=9a92460dad2938181bfc51f9cebefc81b6f9b59ac21c2144f4cba66739a953b6"
  "private_gkd_512_manifest_sha256=5fdb86a9ed8d1ce90db3b5354f2fdfd9a11dad0078d679555512ab8ea01cb715"
  "private_gkd_512_tensors_sha256=2acb96c7a24151b1e38d000654568910b68eb1e6071697bf14f034e7c5461b26"
  "private_gkd_1024_manifest_sha256=896d4a1424483d090c9dc5679ce5ad562c8058201cf7e783454d1cf16a887d8f"
  "private_gkd_1024_tensors_sha256=8f2003d6ab8129687207b5cdf86242576abfb42470eb661032bad843c5d31d67"
  "private_gkd_2048_manifest_sha256=a119c7062e3280d58eea03c44f553cbf917465b2ddae303e5ab1eb05bcd8ee5c"
  "private_gkd_2048_tensors_sha256=abf918fbc2e6d3263a0bfed70decc863f1e17f7671f937b79cf37d9a9afdc56a"
  "gpu_uuids=${gpu_uuids}"
)
unset GKD_REPLAY_EXPECTED_MODEL_MOUNT_SHA256 \
  GKD_REPLAY_EXPECTED_PRIVATE_DATA_MOUNT_SHA256 \
  GKD_REPLAY_MODEL_IDENTITY_HELPER_SHA256 \
  GKD_REPLAY_RUNTIME_IDENTITY_PATH \
  GKD_REPLAY_RUNTIME_IDENTITY_SHA256 \
  GKD_REPLAY_RUNTIME_IDENTITY_HELPER_SHA256 \
  GKD_REPLAY_GPU_INDICES

if (( dry_run == 0 )); then
  require_command docker
  require_command flock
  require_command mktemp
  require_command nvidia-smi
  require_command python3
  if ! docker info >/dev/null 2>&1; then
    [[ "${GKD_DOCKER_GROUP_REEXEC:-0}" != 1 ]] \
      || die "Docker is still unavailable after docker-group re-exec"
    getent group docker | awk -F: -v user="$(id -un)" '
      $1 == "docker" { n=split($4,m,","); for(i=1;i<=n;i++) if(m[i]==user) ok=1 }
      END { exit(ok ? 0 : 1) }
    ' || die "Docker is unavailable and the current user is not in the docker group"
    printf -v reexec '%q ' env \
      GKD_DOCKER_GROUP_REEXEC=1 \
      REPOSITORY_ROOT="${REPOSITORY_ROOT}" \
      SFORGE_PYTHON="${SFORGE_PYTHON}" \
      GKD_GPU_UUIDS="${gpu_uuids}" \
      GKD_GPU_INDICES="${gpu_indices}" \
      GKD_STUDENT_MODEL_HOST_PATH="${GKD_STUDENT_MODEL_HOST_PATH:-}" \
      GKD_TEACHER_MODEL_HOST_PATH="${GKD_TEACHER_MODEL_HOST_PATH:-}" \
      GKD_DATA_HOST_ROOT="${GKD_DATA_HOST_ROOT:-}" \
      GKD_PRIVATE_DATA_HOST_ROOT="${GKD_PRIVATE_DATA_HOST_ROOT:-}" \
      SFORGE_AGENT_RUNTIME_DIR="${SFORGE_AGENT_RUNTIME_DIR:-}" \
      SFORGE_VERSION_STORE_ROOT="${SFORGE_VERSION_STORE_ROOT:-}" \
      bash "${SCRIPT_DIR}/replay_candidates.sh" "${ORIGINAL_ARGS[@]}"
    exec sg docker -c "${reexec}"
  fi

  docker image inspect "${WORK_IMAGE}" "${JUDGE_IMAGE}" >/dev/null \
    || die "final Work/Judge images are missing; run scripts/build_images.sh"
  model_repo="${GKD_STUDENT_MODEL_HOST_PATH:-${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}/models--Qwen--Qwen3-1.7B}"
  teacher_model_repo="${GKD_TEACHER_MODEL_HOST_PATH:-${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}/models--Qwen--Qwen3-4B}"
  # Data defaults: the task's Infrathon assets (assets/fetch.sh) when
  # INFRATHON_ASSETS_ROOT is set, else the host artifact directory.
  # Unset: the default root of assets/fetch.sh, when it exists.
  [[ -n "${INFRATHON_ASSETS_ROOT:-}" || ! -d "${HOME}/.cache/infrathon/assets" ]] \
    || INFRATHON_ASSETS_ROOT="${HOME}/.cache/infrathon/assets"
  if [[ -n "${INFRATHON_ASSETS_ROOT:-}" ]]; then
    default_data_root="${INFRATHON_ASSETS_ROOT}/qwen3-4b-to-1p7b-gkd-distillation-optimizer-dual-h100-v1-official-trl-seed/judge/tldr_gkd_v1"
    default_private_data_root="${INFRATHON_ASSETS_ROOT}/qwen3-4b-to-1p7b-gkd-distillation-optimizer-dual-h100-v1-official-trl-seed/judge/tldr_gkd_v1"
  else
    default_data_root="${HOME}/mlsysforge-artifacts/qwen3-gkd/tldr_gkd_v1"
    default_private_data_root="${HOME}/mlsysforge-artifacts/qwen3-gkd/tldr_gkd_v1"
  fi
  data_root="${GKD_DATA_HOST_ROOT:-${default_data_root}}"
  private_data_root="${GKD_PRIVATE_DATA_HOST_ROOT:-${default_private_data_root}}"
  model_repo="$(canonical_existing_dir "model repository" "${model_repo}")"
  teacher_model_repo="$(canonical_existing_dir "teacher model repository" "${teacher_model_repo}")"
  data_root="$(canonical_existing_dir "public data root" "${data_root}")"
  private_data_root="$(canonical_existing_dir "private data root" "${private_data_root}")"

  # replay-candidates uses the CLI's import-bound evaluator, not the adapter's
  # managed Work-pause wrapper. This inherited FD is therefore the single
  # lease spanning preflight and the complete serial replay.
  replay_lock="/tmp/qwen3-gkd-dual-h100-managed-judge.lock"
  if [[ "${GKD_REPLAY_LOCKED:-0}" == 1 \
     || -n "${SFORGE_SECURE_LOCK_FD:-}" ]]; then
    [[ "${GKD_REPLAY_LOCKED:-0}" == 1 \
       && "${SFORGE_SECURE_LOCK_FD:-}" == 9 ]] \
      || die "inconsistent inherited GKD replay lock state"
    python3 "${LOCK_HELPER}" \
      --lock "${replay_lock}" --fd 9 --marker GKD_REPLAY_LOCKED \
      --verify-held
  else
    exec python3 "${LOCK_HELPER}" \
      --lock "${replay_lock}" \
      --fd 9 --marker GKD_REPLAY_LOCKED -- \
      bash "${SCRIPT_DIR}/replay_candidates.sh" "${ORIGINAL_ARGS[@]}"
  fi

  private_evidence_dir="$(mktemp -d -t sforge-gkd-replay-preflight.XXXXXXXX)"
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
    --teacher-model-path "${teacher_model_repo}/snapshots/1cfa9a7208912126459214e8b04321603b3df60c" \
    --data-root "${data_root}" \
    --private-data-root "${private_data_root}" \
    >"${preflight_path}"

  model_identity_json="$(
    python3 "${MODEL_IDENTITY_HELPER}" \
      --model-repo "${model_repo}" \
      --teacher-model-repo "${teacher_model_repo}" \
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

  export GKD_STUDENT_MODEL_HOST_PATH="${model_repo}"
  export GKD_TEACHER_MODEL_HOST_PATH="${teacher_model_repo}"
  export GKD_DATA_HOST_ROOT="${data_root}"
  export GKD_PRIVATE_DATA_HOST_ROOT="${private_data_root}"
  export GKD_REPLAY_EXPECTED_MODEL_MOUNT_SHA256="${model_mount_tree_sha256}"
  export GKD_REPLAY_EXPECTED_PRIVATE_DATA_MOUNT_SHA256="${private_data_mount_tree_sha256}"
  export GKD_REPLAY_MODEL_IDENTITY_HELPER_SHA256="$(sha256_file "${MODEL_IDENTITY_HELPER}")"
  export GKD_REPLAY_RUNTIME_IDENTITY_PATH="${runtime_identity_path}"
  export GKD_REPLAY_RUNTIME_IDENTITY_SHA256="${runtime_identity_file_sha256}"
  export GKD_REPLAY_RUNTIME_IDENTITY_HELPER_SHA256="$(sha256_file "${RUNTIME_IDENTITY_HELPER}")"
  export GKD_REPLAY_GPU_INDICES="${gpu_indices}"
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
export GKD_GPU_UUIDS="${gpu_uuids}"

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
