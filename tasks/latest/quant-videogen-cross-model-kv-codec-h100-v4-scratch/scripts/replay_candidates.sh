#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
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
  QVG_ARTIFACT_ROOT=/path JUDGE_GPUS=0,1,2,3,4,5,6,7 \
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
  --timeout SECONDS                     Per-candidate Judge timeout (default: the
                                        controller bound for the GPU count + margin)

Selection/control:
  --candidate vN|CANDIDATE_ID            Repeatable exact candidate selector
  --from-v N / --to-v N                 Inclusive candidate range
  --every N                             Sample every Nth candidate (last always kept)
  --repeat N                            Extra measurements per candidate (median + spread)
  --resume                              Continue the same replay plan
  --dry-run                             Preview candidates with placeholder runtime identity
  --json                                Print the candidate curve as JSON
  --silent                              Suppress detailed Judge logs

Full replay follows eval_archive.sh: the explicit JUDGE_GPUS (1-8 indices or
UUIDs, every candidate judged on the complete hidden suite across them), both
host lock names per GPU held for the whole serial batch, image freshness,
model/release verification, and the task's mount-aware
scripts/sforge_runtime_entry.py.
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

judge_gpu_count_from_selection() {
  local -a selectors=()
  IFS=, read -r -a selectors <<<"$1"
  echo "${#selectors[@]}"
}

cleanup_private_evidence() {
  if [[ -n "${private_evidence_dir:-}" \
     && "${private_evidence_dir}" == /tmp/sforge-qvg-replay-preflight.* \
     && -d "${private_evidence_dir}" && ! -L "${private_evidence_dir}" ]]; then
    local evidence
    for evidence in "${model_lock:-}" "${runtime_identity_path:-}"; do
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
tasks_dir="${TASKS_DIR}"
timeout="${ORACLE_TIMEOUT:-}"
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
[[ -z "${timeout}" || "${timeout}" =~ ^[1-9][0-9]*$ ]] \
  || die_usage "--timeout must be positive"
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
require_command sha256sum
require_command python3
launcher_lib_sha256="$(sha256_file "${LAUNCHER_LIB}")"
[[ "${launcher_lib_sha256}" == "${launcher_lib_sha256_before}" ]] \
  || die "launcher lib changed while it was loaded"

[[ -x "${SFORGE_PYTHON}" ]] || die "missing SForge Python: ${SFORGE_PYTHON}"
VERIFY_MODELS="${SCRIPT_DIR}/verify_models.py"
RUNTIME_IDENTITY_HELPER="${SCRIPT_DIR}/runtime_identity.py"
TASK_SPEC="${tasks_dir}/${TASK_ID}.json"
MODEL_LOCK_REFERENCE="${TASK_ROOT}/locks/model-lock.json"
# judge_plan sizes the Judge from the task JSON the replay actually uses.
TASKS_DIR="${tasks_dir}"
for required_file in \
  "${RUNTIME_ENTRY}" "${VERIFY_MODELS}" "${RELEASE_PATH}" \
  "${RUNTIME_IDENTITY_HELPER}" "${LAUNCHER_LIB}" \
  "${TASK_SPEC}" "${MODEL_LOCK_REFERENCE}"; do
  [[ -f "${required_file}" && ! -L "${required_file}" ]] \
    || die "missing or unsafe replay dependency: ${required_file}"
done

release_sha="$(sha256_file "${RELEASE_PATH}")"
protocol_sha="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["paper_protocol_sha256"])' "${RELEASE_PATH}")"
fidelity_sha="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["metric_implementation_sha256"]["fidelity"])' "${RELEASE_PATH}")"
vbench_sha="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["metric_implementation_sha256"]["vbench"])' "${RELEASE_PATH}")"
[[ "$(sha256_file "${TASK_ROOT}/configs/paper_protocol.json")" == "${protocol_sha}" ]] \
  || die "task protocol does not match the release"
[[ "$(sha256_file "${TASK_ROOT}/metrics/compare_frames.py")" == "${fidelity_sha}" ]] \
  || die "fidelity implementation does not match the release"
[[ "$(sha256_file "${TASK_ROOT}/metrics/run_vbench.py")" == "${vbench_sha}" ]] \
  || die "VBench implementation does not match the release"

contexts=(
  "task_runtime_adapter_sha256=$(sha256_file "${RUNTIME_ENTRY}")"
  "replay_launcher_sha256=$(sha256_file "${SCRIPT_DIR}/replay_candidates.sh")"
  "launcher_lib_sha256=${launcher_lib_sha256}"
  "verify_models_sha256=$(sha256_file "${VERIFY_MODELS}")"
  "runtime_identity_helper_sha256=$(sha256_file "${RUNTIME_IDENTITY_HELPER}")"
  "task_spec_sha256=$(sha256_file "${TASK_SPEC}")"
  "release_sha256=${release_sha}"
  "paper_protocol_sha256=${protocol_sha}"
  "fidelity_metric_sha256=${fidelity_sha}"
  "vbench_metric_sha256=${vbench_sha}"
)
unset QVG_REPLAY_MODEL_LOCK_PATH \
  QVG_REPLAY_MODEL_LOCK_SHA256 \
  QVG_REPLAY_VERIFY_MODELS_SHA256 \
  QVG_REPLAY_RUNTIME_IDENTITY_PATH \
  QVG_REPLAY_RUNTIME_IDENTITY_SHA256 \
  QVG_REPLAY_RUNTIME_IDENTITY_HELPER_SHA256 \
  QVG_REPLAY_GPU_INDICES \
  QVG_REPLAY_GPU_UUIDS \
  QVG_JUDGE_GPU_UUIDS

judge_limit_args=()
if (( dry_run == 0 )); then
  # ensure_docker_group preserves JUDGE_GPUS across an sg re-exec.
  ensure_docker_group "${ORIGINAL_ARGS[@]}"
  for command_name in docker flock jq mktemp nvidia-smi python3 stat; do
    require_command "${command_name}"
  done
  require_sforge
  require_artifact_root
  ARTIFACT_ROOT="$(canonical_existing_dir "Quant-VideoGen artifact root" "${ARTIFACT_ROOT}")"
  export QVG_ARTIFACT_ROOT="${ARTIFACT_ROOT}"

  assert_built_images_current
  judge_uuid_lines="$(judge_gpu_uuids "${JUDGE_GPUS:-}")"
  mapfile -t judge_uuids <<<"${judge_uuid_lines}"
  # This shell holds both host lock names of every Judge GPU until it exits,
  # so one lease covers every candidate in the serial batch.
  for uuid in "${judge_uuids[@]}"; do
    lock_gpu "${uuid}" judge
  done
  judge_indices=()
  for uuid in "${judge_uuids[@]}"; do
    assert_gpu_idle "${uuid}" judge
    assert_no_compute_process "${uuid}" judge
    judge_indices+=("$(gpu_index "${uuid}")")
  done
  assert_identical_gpus judge "${judge_uuids[@]}"
  plan="$(judge_plan "${#judge_uuids[@]}")"
  read -r cpu_limit mem_limit controller_seconds eval_seconds <<<"${plan}"
  timeout="${timeout:-${eval_seconds}}"
  (( timeout > controller_seconds )) \
    || die "--timeout must exceed the controller bound of ${controller_seconds} s for ${#judge_uuids[@]} GPU(s)"
  judge_limit_args=(
    --judge-cpu-limit "${JUDGE_CPU_LIMIT:-${cpu_limit}}"
    --judge-mem-limit "${JUDGE_MEM_LIMIT:-${mem_limit}}"
  )

  private_evidence_dir="$(mktemp -d -t sforge-qvg-replay-preflight.XXXXXXXX)"
  chmod 0700 "${private_evidence_dir}"
  model_lock="${private_evidence_dir}/model-lock.json"
  trap cleanup_private_evidence EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP
  python3 "${TASK_ROOT}/scripts/verify_models.py" \
    --artifact-root "${ARTIFACT_ROOT}" \
    --output "${model_lock}" >/dev/null
  model_lock_sha="$(sha256_file "${model_lock}")"
  runtime_identity_path="${private_evidence_dir}/runtime-identity.json"
  identity_args=()
  for position in "${!judge_uuids[@]}"; do
    identity_args+=(
      --gpu-index "${judge_indices[${position}]}"
      --expected-gpu-uuid "${judge_uuids[${position}]}"
    )
  done
  python3 "${RUNTIME_IDENTITY_HELPER}" \
    "${identity_args[@]}" \
    --output "${runtime_identity_path}"
  runtime_identity_file_sha256="$(sha256_file "${runtime_identity_path}")"
  host_runtime_identity_sha256="$(
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["identity_sha256"])' \
      "${runtime_identity_path}"
  )"
  export QVG_REPLAY_MODEL_LOCK_PATH="${model_lock}"
  export QVG_REPLAY_MODEL_LOCK_SHA256="${model_lock_sha}"
  export QVG_REPLAY_VERIFY_MODELS_SHA256="$(sha256_file "${VERIFY_MODELS}")"
  export QVG_REPLAY_RUNTIME_IDENTITY_PATH="${runtime_identity_path}"
  export QVG_REPLAY_RUNTIME_IDENTITY_SHA256="${runtime_identity_file_sha256}"
  export QVG_REPLAY_RUNTIME_IDENTITY_HELPER_SHA256="$(sha256_file "${RUNTIME_IDENTITY_HELPER}")"
  export QVG_REPLAY_GPU_INDICES="${judge_indices[*]}"
  export QVG_REPLAY_GPU_UUIDS="${judge_uuids[*]}"
  export QVG_JUDGE_GPU_UUIDS="${judge_uuids[*]}"

  contexts+=(
    "model_lock_sha256=${model_lock_sha}"
    "artifact_root=${ARTIFACT_ROOT}"
    "judge_gpu_uuids=$(IFS=,; echo "${judge_uuids[*]}")"
    "host_runtime_identity_sha256=${host_runtime_identity_sha256}"
  )
  # The controller reads its budget from the baked release and needs no extra
  # environment; a CUDA_VISIBLE_DEVICES mask would hide all but one Judge GPU.
  unset SFORGE_JUDGE_EXTRA_ENV
else
  # A preview only sizes the Judge when JUDGE_GPUS names the intended count;
  # it never probes or locks a GPU.
  if [[ -n "${JUDGE_GPUS:-}" ]]; then
    plan="$(judge_plan "$(judge_gpu_count_from_selection "${JUDGE_GPUS}")")"
    read -r cpu_limit mem_limit controller_seconds eval_seconds <<<"${plan}"
    timeout="${timeout:-${eval_seconds}}"
    judge_limit_args=(
      --judge-cpu-limit "${JUDGE_CPU_LIMIT:-${cpu_limit}}"
      --judge-mem-limit "${JUDGE_MEM_LIMIT:-${mem_limit}}"
    )
  fi
  contexts+=(
    "model_lock_sha256=$(sha256_file "${MODEL_LOCK_REFERENCE}")"
    "artifact_root=unverified-dry-run"
    "judge_gpu_uuids=unverified-dry-run"
    "host_runtime_identity_sha256=unverified-dry-run"
  )
fi

export SFORGE_TASKS_DIR="${tasks_dir}"
export SFORGE_LOG_DIR="${output_dir}"
if (( dry_run == 1 )) && [[ -z "${ARTIFACT_ROOT}" ]]; then
  # The adapter validates the actual mount tree only when a backend is made.
  # A nonempty local sentinel lets its import-only dry-run path stay useful.
  export QVG_ARTIFACT_ROOT="${TASK_ROOT}"
else
  export QVG_ARTIFACT_ROOT="${ARTIFACT_ROOT}"
fi

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
)
[[ -z "${timeout}" ]] || command+=(--timeout "${timeout}")
command+=(${judge_limit_args[@]+"${judge_limit_args[@]}"})
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
