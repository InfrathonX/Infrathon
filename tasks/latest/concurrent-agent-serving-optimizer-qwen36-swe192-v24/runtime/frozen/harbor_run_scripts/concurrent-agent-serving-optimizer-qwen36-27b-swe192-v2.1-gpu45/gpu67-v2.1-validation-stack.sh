#!/usr/bin/env bash
set -Eeuo pipefail

runner_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
bench_root="$(cd -- "$runner_dir/../.." && pwd)"
task_dir="$bench_root/harbor_tasks/concurrent-agent-serving-optimizer-qwen36-27b-swe192-v2.1-gpu45"
overlay="$runner_dir/gpu67-v2.1-validation.compose.yaml"

project="kvbench-qwen36-swe192-v2-1-gpu67-validation"
# Both roots are bind-mounted into the Judge at their host paths by
# scripts/sforge_runtime_entry.py, which also exports them; the defaults are
# the same locations scripts/preflight.py and the entry script derive.
state_root="${HARBOR_KVBENCH_STATE_ROOT:-$HOME/.cache/harbor-kv-qwen36-swe192-v2-1-gpu67-validation}"
corpus_root="${HARBOR_SWEBENCH_CORPUS_SOURCE:-${KVBENCH_CORPUS_ROOT:-$HOME/.cache/harbor-kv-scheduler-v2/swebench-lite-heterogeneous-192-gpu45}}"
candidate_dir="${KV_AUTHOR_REFERENCE_GATEWAY_DIR:?set an absolute frozen candidate directory}"
owner_token="${KV_GPU67_V21_OWNER_TOKEN:?set an explicit Docker ownership label}"
vllm_base_url="${KVBENCH_VLLM_BASE_URL:?set a container-reachable vLLM HTTP(S) origin}"
main_image="harbor-kv/main:qwen36-27b-swe192-v2-1-gpu45-v1"
model_api_image="harbor-kv/model-api:qwen36-swe192-v24-url-v1"
frozen_controller_image="harbor-kv/bench-controller:qwen36-swe192-v24-url-v1"
controller_image="${KV_GPU67_CONTROLLER_IMAGE:-$frozen_controller_image}"
sandbox_worker_image="harbor-kv/sandbox-worker:qwen36-27b-tp2-public30-r2"
action="${1:-}"

usage() {
  printf '%s\n' "Usage: $0 {check|start|stop-oracle|restart-oracle|status|logs|stop}"
}

[[ -n "$action" ]] || { usage >&2; exit 2; }
[[ "$candidate_dir" == /* && -d "$candidate_dir" && ! -L "$candidate_dir" ]] || {
  echo "KV_AUTHOR_REFERENCE_GATEWAY_DIR must be an absolute real directory" >&2
  exit 2
}
[[ -x "$candidate_dir/launch.sh" && ! -L "$candidate_dir/launch.sh" ]] || {
  echo "candidate launch.sh is missing, non-executable, or a symlink" >&2
  exit 2
}
[[ "$owner_token" =~ ^[A-Za-z0-9_.-]{1,128}$ ]] || {
  echo "KV_GPU67_V21_OWNER_TOKEN has an invalid label value" >&2
  exit 2
}
[[ "$controller_image" == "$frozen_controller_image" ]] || {
  echo "unapproved validation controller image: $controller_image" >&2
  exit 4
}

if ! docker info >/dev/null 2>&1; then
  [[ "${KV_GPU67_V21_DOCKER_REEXEC:-0}" != 1 ]] || {
    echo "Docker remains unavailable after docker-group re-exec" >&2
    exit 2
  }
  command=(
    env
    KV_GPU67_V21_DOCKER_REEXEC=1
    "KV_AUTHOR_REFERENCE_GATEWAY_DIR=$candidate_dir"
    "KV_GPU67_V21_OWNER_TOKEN=$owner_token"
    "KVBENCH_VLLM_BASE_URL=$vllm_base_url"
    "$0"
    "$action"
  )
  command_string=""
  for argument in "${command[@]}"; do
    printf -v quoted '%q' "$argument"
    command_string+="${command_string:+ }$quoted"
  done
  exec sg docker -c "$command_string"
fi

export HARBOR_KVBENCH_STATE_ROOT="$state_root"
export HARBOR_SWEBENCH_CORPUS_SOURCE="$corpus_root"
export KVBENCH_VLLM_BASE_URL="$vllm_base_url"
export KV_AUTHOR_REFERENCE_GATEWAY_DIR="$candidate_dir"
export KV_GPU67_V21_OWNER_TOKEN="$owner_token"
export KV_GPU67_MODEL_API_IMAGE="$model_api_image"
export KV_GPU67_CONTROLLER_IMAGE="$controller_image"

compose=(
  docker compose
  --project-name "$project"
  --project-directory "$task_dir/environment"
  -f "$task_dir/scripts/compose-main-base.yaml"
  -f "$task_dir/environment/docker-compose.yaml"
  -f "$task_dir/scripts/compose-calibration.yaml"
  -f "$overlay"
)

check() {
  [[ -f "$corpus_root/manifest.json" && ! -L "$corpus_root/manifest.json" ]] || {
    echo "the frozen distinct-192 corpus is unavailable" >&2
    return 4
  }
  jq -e '
    .corpus_mode == "heterogeneous-192-distinct"
    and .selected_instance_count == 192
    and (.instances | length) == 192
  ' "$corpus_root/manifest.json" >/dev/null
  # Contract v3 (2026-09-18): main and sandbox-worker are built from the frozen
  # environment by scripts/build_images.sh; their image IDs are deployment
  # provenance (recorded in the build manifest and the preflight report), not
  # pinned identities, so only presence is asserted here.
  docker image inspect "$main_image" >/dev/null 2>&1 || {
    echo "the v2.1 main image is unavailable (run scripts/build_images.sh)" >&2
    return 4
  }
  docker image inspect "$model_api_image" >/dev/null 2>&1 || {
    echo "the URL-backed model-api image is unavailable" >&2
    return 4
  }
  docker image inspect "$controller_image" >/dev/null 2>&1 || {
    echo "the URL-backed controller image is unavailable" >&2
    return 4
  }
  docker image inspect "$sandbox_worker_image" >/dev/null 2>&1 || {
    echo "the v2.1 sandbox-worker image is unavailable (run scripts/build_images.sh)" >&2
    return 4
  }
  install -d -m 0755 "$state_root/workspaces"
  "${compose[@]}" config --quiet
}

wait_main() {
  local main_container
  main_container="$("${compose[@]}" ps --quiet main)"
  [[ -n "$main_container" ]] || return 4
  for _ in $(seq 1 90); do
    if docker exec "$main_container" python3 -c \
      'import urllib.request; urllib.request.urlopen("http://127.0.0.1:9000/health", timeout=2).read()' \
      >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  echo "candidate gateway did not become healthy" >&2
  return 4
}

case "$action" in
  check)
    check
    echo "v2.1 URL-backed validation contract passed"
    ;;
  start)
    check
    "${compose[@]}" up --detach --no-build --wait --wait-timeout 300
    "${compose[@]}" ps
    ;;
  stop-oracle)
    "${compose[@]}" stop main
    ;;
  restart-oracle)
    check
    "${compose[@]}" up --detach --no-deps --no-build --force-recreate main
    wait_main
    ;;
  status)
    check
    "${compose[@]}" ps
    ;;
  logs)
    "${compose[@]}" logs --tail 160
    ;;
  stop)
    "${compose[@]}" down --timeout 300 --remove-orphans
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
