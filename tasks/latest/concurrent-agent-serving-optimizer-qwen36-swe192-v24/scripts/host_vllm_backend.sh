#!/usr/bin/env bash
# Start / stop the external vLLM backend from the recipe-bound runtime image.
#
# Every serving parameter comes from the frozen model profile; the image is
# images/backend/Dockerfile (contract v3).  The backend is operator-owned and
# sits outside the harness: this helper only makes the documented boot
# reproducible.  It prints the origin to export as KVBENCH_VLLM_BASE_URL.
#
#   KVBENCH_MODEL_PATH   the Qwen/Qwen3.6-27B snapshot directory at revision
#                        6a9e13bd... (hf download Qwen/Qwen3.6-27B --revision ...)
#   KVBENCH_BACKEND_GPUS two device selectors for --gpus, e.g. "0,1" or two UUIDs
#   KVBENCH_BACKEND_PORT TCP port on the Docker bridge gateway (default 18000)
set -euo pipefail

SFORGE_ENTRY_SCRIPT="$0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
ensure_docker_group "$@"
require_command docker
require_command python3

PROFILE="${NESTED_ENVIRONMENT_ROOT}/vllm-runtime/model-profile-qwen3.6-27b.json"
CONTAINER="${KVBENCH_BACKEND_CONTAINER:-kvbench-qwen36-27b-backend}"
PORT="${KVBENCH_BACKEND_PORT:-18000}"
action="${1:-}"

usage() {
  printf 'Usage: %s {build|start|status|logs|stop}\n' "$0"
}

bridge_gateway() {
  docker network inspect bridge --format '{{(index .IPAM.Config 0).Gateway}}'
}

build_image() {
  docker build --tag "${BACKEND_IMAGE}" --file "${BACKEND_DOCKERFILE}" "${MIGRATION_ROOT}/images/backend"
}

# The profile's runtime section, rendered as vllm CLI flags.  Keys that are
# identity or provenance rather than flags are skipped explicitly.
profile_args() {
  python3 - "${PROFILE}" <<'PY'
import json, sys
runtime = json.load(open(sys.argv[1], encoding="utf-8"))["runtime"]
skip = {
    "runtime_image_ref", "runtime_image_id", "runtime_image_recipe",
    "vllm_package_version", "vllm_engine_version", "server_dev_mode",
    "preserve_checkpoint_quantization", "checkpoint_quantization_method",
    "fail_on_environ_validation", "tool_call_parser",
}
args = []
for key, value in runtime.items():
    if key in skip or value is None:
        continue
    flag = "--" + key.replace("_", "-")
    if value is True:
        args.append(flag)
    elif value is False:
        continue
    else:
        args += [flag, str(value)]
print("\n".join(args))
PY
}

case "${action}" in
  build)
    build_image
    ;;
  start)
    model_path="${KVBENCH_MODEL_PATH:?set KVBENCH_MODEL_PATH to the Qwen/Qwen3.6-27B snapshot directory}"
    gpus="${KVBENCH_BACKEND_GPUS:?set KVBENCH_BACKEND_GPUS to the two device selectors this backend owns (nvidia-smi -L)}"
    [[ -f "${model_path}/config.json" ]] || die "KVBENCH_MODEL_PATH has no config.json: ${model_path}"
    docker image inspect "${BACKEND_IMAGE}" >/dev/null 2>&1 || build_image
    gateway="$(bridge_gateway)"
    mapfile -t args < <(profile_args)
    docker run -d --name "${CONTAINER}" \
      --gpus "\"device=${gpus}\"" --ipc host --network host \
      -e VLLM_SERVER_DEV_MODE=1 \
      -v "${model_path}:/models/qwen3.6-27b:ro" \
      "${BACKEND_IMAGE}" \
      /models/qwen3.6-27b --host "${gateway}" --port "${PORT}" "${args[@]}" >/dev/null
    echo "export KVBENCH_VLLM_BASE_URL=http://${gateway}:${PORT}"
    echo "health: curl --noproxy '*' http://${gateway}:${PORT}/health"
    ;;
  status)
    docker ps --filter "name=^${CONTAINER}$" --format '{{.Names}} {{.Status}} {{.Image}}'
    docker image inspect --format 'image {{.Id}}' "${BACKEND_IMAGE}" 2>/dev/null || echo "image not built"
    ;;
  logs)
    docker logs --tail 200 "${CONTAINER}"
    ;;
  stop)
    docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
