#!/bin/bash
# Run the task's Judge image on a submission archive the way SForge does (no network, read-only
# model / initial checkpoint, host scratch at /scratch), outside a launcher run (operator rescoring).
#   judge_archive.sh ARCHIVE NAME [GPU_PAIR]
# GPU_PAIR is two GPU UUIDs or two nvidia-smi indices, comma-separated; without it ALFW_GPU_UUIDS,
# then GPU_INDICES, names the pair. The caller must hold that pair (scripts/miles_v2/hold_gpu_pair.py).
# Host locations come from scripts/miles_common/host_config.py: ALFW_HF_CACHE (Qwen3-4B),
# ALFW_MEGATRON_INIT, and the work directory $MILES_SCRATCH_ROOT/<task>/judge-archive/NAME.
# With SForge's opt-in CPU pinning policy exported (SFORGE_CPUSET..., docs/CPU_AFFINITY.md) the container is pinned
# to the pair's CPU slice like a harness Judge (record: NAME/cpu_affinity.json); unset, nothing changes.
set -euo pipefail
die() { echo "judge_archive.sh: $1" >&2; exit "${2:-2}"; }
[[ $# -ge 2 ]] || die "usage: judge_archive.sh ARCHIVE NAME [GPU_PAIR]"
AR="$(readlink -f "$1")"; NAME="$2"
[[ -f "${AR}" ]] || die "no archive $1"
[[ "${NAME}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || die "unsafe NAME ${NAME}"
PAIR="${3:-${ALFW_GPU_UUIDS:-${GPU_INDICES:-}}}"
[[ -n "${PAIR}" ]] || die "no GPU pair: pass GPU_PAIR, or set ALFW_GPU_UUIDS (two UUIDs) or GPU_INDICES (two indices)"
IFS=, read -r -a GPUS <<< "${PAIR// /}"
[[ ${#GPUS[@]} -eq 2 ]] || die "GPU_PAIR must name two GPUs, got '${PAIR}'"
UUIDS=()
for g in "${GPUS[@]}"; do
  if [[ "${g}" =~ ^[0-9]+$ ]]; then
    u="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "${g}")" || die "GPU index ${g} does not exist"
  else
    u="${g}"
  fi
  [[ "${u}" =~ ^GPU-[0-9a-fA-F-]{36}$ ]] || die "not a GPU UUID: ${u}"
  UUIDS+=("${u}")
done

TASK="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd -- "${TASK}/../../.." && pwd)"
TASK_ID=miles_qwen3_alfworld_agentic_rl_2xh100_v1
HOST_CONFIG=(python3 "${REPO}/scripts/miles_common/host_config.py")
# Assigned before use so that `set -e` stops on host_config.py's exit 78 (not configured).
host_settings="$("${HOST_CONFIG[@]}" shell ALFW_HF_CACHE ALFW_MEGATRON_INIT)"
eval "${host_settings}"
MODEL="${ALFW_HF_CACHE}/models--Qwen--Qwen3-4B"
[[ -d "${MODEL}" ]] || die "ALFW_HF_CACHE=${ALFW_HF_CACHE} has no models--Qwen--Qwen3-4B" 78
[[ -d "${ALFW_MEGATRON_INIT}" ]] || die "ALFW_MEGATRON_INIT=${ALFW_MEGATRON_INIT} is not a directory" 78
SCRATCH="$("${HOST_CONFIG[@]}" scratch "$(basename "${TASK}")" 200)"
W="${SCRATCH}/judge-archive/${NAME}"; mkdir -p "${W}/app" "${W}/scratch" "${W}/logs"; chmod 777 "${W}/scratch"
tar -xzf "${AR}" -C "${W}/app"
# The harness Judge's 24-CPU quota (runtime/sforge_entry.py nano_cpus): with the policy set, the slice is planned for that
# budget (24 CPUs on an H100 SXM pair; the whole 16-CPU slice on an NVL pair), as SForge plans it for a formal run.
PIN="$(python3 "${REPO}/scripts/miles_common/cpu_pinning.py" docker-args --gpu "${UUIDS[0]}" --gpu "${UUIDS[1]}" \
  --budget 24 --role judge --record "${W}/cpu_affinity.json")" || die "the CPU pinning policy cannot be applied to this pair" 78
[[ -z "${PIN}" ]] || PIN="${PIN} --cpus 24"
sg docker -c "docker run --rm --name alfworld-dev-judge-${NAME} --gpus '\"device=${UUIDS[0]},${UUIDS[1]}\"' ${PIN} \
  --network none --hostname alfw-judge --shm-size 64g --ulimit memlock=-1 --ulimit nofile=1048576:1048576 \
  -e CUDA_VISIBLE_DEVICES=0,1 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v ${MODEL}:/models/qwen3-4b:ro -v ${ALFW_MEGATRON_INIT}:/data/megatron:ro \
  -v ${W}/scratch:/scratch -v ${W}/logs:/logs/verifier -v ${W}/app/repo:/app/workdir/miles_lab/repo:ro \
  -v ${W}/app/task_entry.py:/app/workdir/miles_lab/task_entry.py:ro -v ${W}/app/config:/app/workdir/miles_lab/config:ro \
  miles_alfworld.judge.${TASK_ID}:local python3 /opt/alfworld_judge/run_judge.py --app /app/workdir/miles_lab --log-dir /logs/verifier" > "${W}/judge.log" 2>&1
sed -n '/Start Structured Result/,/End Structured Result/p' "${W}/judge.log"
