#!/usr/bin/env bash
# Build the three Miles base images from public pins, on any host:
#
#   scripts/miles_base_images/build.sh model-flat   # miles-model-dependencies-flat:local  (OLMo3 v2, OPCD v2, OPD)
#   scripts/miles_base_images/build.sh alfworld     # miles-alfworld-base:local           (ALFWorld; needs model-flat)
#   scripts/miles_base_images/build.sh diffusion    # miles-diffusion-base:sgl859a278e-md29fde6ee (SD3.5-M MOPD, Wan2.2)
#   scripts/miles_base_images/build.sh all
#
# Each build writes outputs/miles-base-images/<name>.json with the resolved parent digest, the
# recipe digests and the resulting image ID. The per-task build scripts compare their base against
# the image ID recorded there (see README.md). Needs docker access; re-executes through `sg docker`
# when the shell is not in the docker group yet.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${HERE}/../.." && pwd)"
OUT="${MILES_BASE_IMAGE_RECORDS:-${REPO}/outputs/miles-base-images}"

MILES_PARENT="radixark/miles@sha256:45ae7833f97a01369dbd8f336784830a87d94587bc010c7b8065291cc15cd7ca"
SGLANG_PARENT="lmsysorg/sglang@sha256:9e02c8e1fe2790a1c445bd5f6814305fe43639a4adb01c8ad1e8e21e750bf581"
FLAT_TAG="miles-model-dependencies-flat:local"
ALFWORLD_TAG="miles-alfworld-base:local"
DIFFUSION_TAG="miles-diffusion-base:sgl859a278e-md29fde6ee"

die() { echo "error: $*" >&2; exit 1; }

if ! docker info >/dev/null 2>&1; then
  [[ "${MILES_BASE_IMAGES_REEXEC:-0}" == 1 ]] && die "docker is unreachable after the docker-group re-exec"
  getent group docker | grep -Eq "(^|[:,])${USER:-$(id -un)}(,|$)" \
    || die "docker is unreachable and ${USER:-$(id -un)} is not in the docker group"
  exec sg docker -c "$(printf '%q ' env MILES_BASE_IMAGES_REEXEC=1 \
    ${MILES_BASE_IMAGE_RECORDS:+MILES_BASE_IMAGE_RECORDS="${MILES_BASE_IMAGE_RECORDS}"} bash "$0" "$@")"
fi

sha() { sha256sum "$1" | cut -d' ' -f1; }
LOGS="${OUT}/logs"; mkdir -p "${LOGS}"
# dbuild LOG-NAME docker-build-args...: build with the full log kept under ${LOGS}.
dbuild() { local log="${LOGS}/$1.log"; shift; docker build --progress=plain "$@" > "${log}" 2>&1 || die "docker build failed, see ${log}"; }
image_id() { docker image inspect --format '{{.Id}}' "$1"; }
layers() { docker image inspect --format '{{len .RootFS.Layers}}' "$1"; }

# flatten SRC_TAG DEST_TAG [WORKDIR]: the merged rootfs of SRC_TAG, keeping its ENV (minus device
# selection and anything that looks like a credential), exported and imported as one layer, then
# cut into parts of at most 4 GiB (scripts/infrathon/split_layers.py: same files and config; a single
# 50-65 GB layer is fragile to push and pull).
flatten() {
  local src="$1" dest="$2" name
  name="miles-base-flatten-$(date +%s%N)"
  docker create --name "${name}" --env NVIDIA_VISIBLE_DEVICES=void "${src}" true >/dev/null
  local -a changes=(--change 'USER root' --change 'ENTRYPOINT []' --change 'CMD ["bash"]')
  local kv key
  while IFS= read -r kv; do
    [[ -n "${kv}" ]] || continue
    key="${kv%%=*}"
    case "${key^^}" in
      NVIDIA_VISIBLE_DEVICES|CUDA_VISIBLE_DEVICES|*TOKEN*|*SECRET*|*PASSWORD*|*API_KEY*) continue ;;
    esac
    changes+=(--change "ENV ${kv}")
  done < <(docker image inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "${src}")
  if [[ -n "${3:-}" ]]; then changes+=(--change "WORKDIR $3"); fi
  docker export "${name}" | docker import "${changes[@]}" - "${dest}-one-layer" >/dev/null
  docker rm -f "${name}" >/dev/null
  [[ "$(layers "${dest}-one-layer")" == 1 ]] || die "flattening ${dest} did not produce one layer"
  local work="${OUT}/split-work"
  rm -rf "${work}"
  python3 "${REPO}/scripts/infrathon/split_layers.py" "${dest}-one-layer" "${dest}" --workdir "${work}" \
    > "${LOGS}/split-$(basename "${dest%%:*}").log" 2>&1 || die "splitting ${dest} failed, see ${LOGS}"
  rm -rf "${work}"
  docker rmi "${dest}-one-layer" >/dev/null
}

record() { # name json-object
  mkdir -p "${OUT}"
  printf '%s\n' "$2" > "${OUT}/$1.json"
  echo "recorded ${OUT}/$1.json"; cat "${OUT}/$1.json"
}

build_model_flat() {
  local empty; empty="$(mktemp -d)"
  docker pull -q "${MILES_PARENT}" >/dev/null
  dbuild numpy-scipy -t miles-validation-parent:local -f "${HERE}/model-flat/numpy-scipy.Dockerfile" "${empty}"
  dbuild public-runtime -t miles-model-public-runtime:local --build-arg BASE_IMAGE=miles-validation-parent:local \
    -f "${HERE}/model-flat/public_runtime.Dockerfile" "${empty}"
  dbuild clean-stage -t miles-model-dependencies-clean-stage:local --build-arg PARENT=miles-model-public-runtime:local \
    -f "${HERE}/model-flat/clean.Dockerfile" "${empty}"
  rmdir "${empty}"
  flatten miles-model-dependencies-clean-stage:local "${FLAT_TAG}"
  docker run --rm --network none --entrypoint python "${FLAT_TAG}" -c "$(cat <<'PY'
import importlib.util as u
missing = [m for m in ('torch', 'sglang', 'megatron', 'transformer_engine', 'ray', 'mbridge') if u.find_spec(m) is None]
assert not missing, missing
assert u.find_spec('miles') is None
print('flat ok')
PY
)"
  record model-flat "$(printf '{"tag": "%s", "image_id": "%s", "parent": "%s", "parent_image_id": "%s", "numpy_scipy_image_id": "%s", "public_runtime_image_id": "%s", "clean_stage_image_id": "%s", "recipes": {"numpy-scipy.Dockerfile": "%s", "public_runtime.Dockerfile": "%s", "clean.Dockerfile": "%s"}, "builder_sha256": "%s", "method": "docker export of the sanitized merged rootfs, docker import, split into <=4 GiB layers"}' \
    "${FLAT_TAG}" "$(image_id "${FLAT_TAG}")" "${MILES_PARENT}" "$(image_id "${MILES_PARENT}")" \
    "$(image_id miles-validation-parent:local)" "$(image_id miles-model-public-runtime:local)" \
    "$(image_id miles-model-dependencies-clean-stage:local)" \
    "$(sha "${HERE}/model-flat/numpy-scipy.Dockerfile")" "$(sha "${HERE}/model-flat/public_runtime.Dockerfile")" \
    "$(sha "${HERE}/model-flat/clean.Dockerfile")" "$(sha "$0")")"
}

build_alfworld() {
  docker image inspect "${FLAT_TAG}" >/dev/null 2>&1 || die "${FLAT_TAG} is missing: run '$0 model-flat' first"
  dbuild alfworld -t "${ALFWORLD_TAG}" --build-arg "BASE_IMAGE=${FLAT_TAG}" "${HERE}/alfworld"
  record alfworld "$(printf '{"tag": "%s", "image_id": "%s", "base": "%s", "base_image_id": "%s", "recipes": {"Dockerfile": "%s", "requirements.txt": "%s"}, "builder_sha256": "%s"}' \
    "${ALFWORLD_TAG}" "$(image_id "${ALFWORLD_TAG}")" "${FLAT_TAG}" "$(image_id "${FLAT_TAG}")" \
    "$(sha "${HERE}/alfworld/Dockerfile")" "$(sha "${HERE}/alfworld/requirements.txt")" "$(sha "$0")")"
}

build_diffusion() {
  docker pull -q "${SGLANG_PARENT}" >/dev/null
  dbuild diffusion -t miles-diffusion-base-build:local --build-arg "SGLANG_BASE=${SGLANG_PARENT}" "${HERE}/diffusion"
  flatten miles-diffusion-base-build:local "${DIFFUSION_TAG}" /root
  local probe
  probe="$(docker run --rm --network none --entrypoint bash "${DIFFUSION_TAG}" -c \
    'cat /opt/miles-diffusion-base/sglang.commit /opt/miles-diffusion-base/torch.txt')"
  [[ "${probe}" == 859a278eb882dddbc4ccab1d0a3697da04a66f28* ]] || die "unexpected sglang commit in ${DIFFUSION_TAG}: ${probe}"
  record diffusion "$(printf '{"tag": "%s", "image_id": "%s", "parent": "%s", "parent_image_id": "%s", "build_image_id": "%s", "sglang_commit": "859a278eb882dddbc4ccab1d0a3697da04a66f28", "miles_diffusion_requirements_commit": "29fde6ee", "recipes": {"Dockerfile": "%s", "requirements.txt": "%s"}, "builder_sha256": "%s", "method": "docker export of the merged rootfs, docker import, split into <=4 GiB layers"}' \
    "${DIFFUSION_TAG}" "$(image_id "${DIFFUSION_TAG}")" "${SGLANG_PARENT}" "$(image_id "${SGLANG_PARENT}")" \
    "$(image_id miles-diffusion-base-build:local)" \
    "$(sha "${HERE}/diffusion/Dockerfile")" "$(sha "${HERE}/diffusion/requirements.txt")" "$(sha "$0")")"
}

case "${1:-}" in
  model-flat) build_model_flat ;;
  alfworld) build_alfworld ;;
  diffusion) build_diffusion ;;
  all) build_model_flat; build_alfworld; build_diffusion ;;
  *) die "usage: $0 model-flat|alfworld|diffusion|all" ;;
esac
