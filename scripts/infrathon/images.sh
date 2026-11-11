# Infrathon image helpers, sourced by a task's scripts/build_images.sh
# after its scripts/lib.sh (which provides die).
#
# A task publishes two images per hardware type: Work and the public Judge
# (images/judge/Dockerfile, no holdout).  The Judge a run uses is always built
# on the host: the public Judge plus images/judge/Dockerfile.private, which
# copies the Judge-only inputs from the task's assets.
#
# images/published.json records what was published, independent of the repository:
#   {"source_digest": "<the task's image_source_digest at publication>",
#    "hardware": {"h100-94gb-400w": {"work": "sha256:...", "judge_public": "sha256:...",
#                                    "work_tag": "...", "judge_public_tag": "..."}}}
# The digests are manifest digests, which a copy of the images into another repository
# keeps; infrathon_registry_repository names the repository to use.

infrathon_published_file() {
  printf '%s/images/published.json\n' "$1"
}

INFRATHON_TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# infrathon_registry_repository: print the image repository ([registry/]namespace/name) to
# pull from and publish to: INFRATHON_REGISTRY_REPOSITORY, or scripts/infrathon/registry.json.
infrathon_registry_repository() {
  local repository="${INFRATHON_REGISTRY_REPOSITORY:-}"
  if [[ -z "${repository}" ]]; then
    repository="$(jq -er '.repository' "${INFRATHON_TOOLS_DIR}/registry.json")" \
      || die "${INFRATHON_TOOLS_DIR}/registry.json has no repository"
  fi
  [[ "${repository}" =~ ^([A-Za-z0-9.-]+(:[0-9]+)?/)?[a-z0-9]+([._-][a-z0-9]+)*(/[a-z0-9]+([._-][a-z0-9]+)*)+$ ]] \
    || die "image repository must look like [registry/]namespace/name: ${repository}"
  printf '%s\n' "${repository}"
}

# infrathon_hardware: print this host's hardware type (INFRATHON_HARDWARE, or detected from
# GPU 0 by scripts/infrathon/assets.py hardware); dies when it cannot tell.
infrathon_hardware() {
  python3 "${INFRATHON_TOOLS_DIR}/assets.py" hardware \
    || die "set INFRATHON_HARDWARE to h100-94gb-400w or h100-80gb-700w"
}

# infrathon_pull_published_images TASK_ROOT WORK_IMAGE JUDGE_PUBLIC_IMAGE SOURCE_DIGEST
#
# With INFRATHON_PULL=1, pulls the Work and public Judge images published for this
# host's hardware (infrathon_hardware; exported as INFRATHON_HARDWARE) by digest from
# infrathon_registry_repository and tags
# them WORK_IMAGE and JUDGE_PUBLIC_IMAGE.  Refuses unless they were published
# from this exact image-driving source.  Without INFRATHON_PULL=1 it returns 1 and the
# caller builds both images from source.
infrathon_pull_published_images() {
  local task_root="$1" work_image="$2" judge_public_image="$3" source_digest="$4"
  [[ "${INFRATHON_PULL:-0}" == "1" ]] || return 1
  local hardware
  hardware="$(infrathon_hardware)" || exit 1
  export INFRATHON_HARDWARE="${hardware}"
  local published repository recorded work_digest judge_digest
  published="$(infrathon_published_file "${task_root}")"
  [[ -f "${published}" && ! -L "${published}" ]] \
    || die "no published images for this task: ${published}"
  recorded="$(jq -er '.source_digest' "${published}")" \
    || die "${published} has no source_digest"
  [[ "${recorded}" == "${source_digest}" ]] \
    || die "published images were built from source ${recorded:0:12}; this checkout is ${source_digest:0:12}"
  repository="$(infrathon_registry_repository)" || exit 1
  work_digest="$(jq -er --arg hw "${hardware}" '.hardware[$hw].work' "${published}")" \
    || die "no ${hardware} Work image published for this task"
  judge_digest="$(jq -er --arg hw "${hardware}" '.hardware[$hw].judge_public' "${published}")" \
    || die "no ${hardware} public Judge image published for this task"
  local digest
  for digest in "${work_digest}" "${judge_digest}"; do
    [[ "${digest}" =~ ^sha256:[0-9a-f]{64}$ ]] || die "malformed published digest: ${digest}"
  done
  # Explicit checks: callers run this function in an `||` list, where set -e is off.
  docker pull "${repository}@${work_digest}" >&2 || die "docker pull ${repository}@${work_digest} failed"
  docker pull "${repository}@${judge_digest}" >&2 || die "docker pull ${repository}@${judge_digest} failed"
  docker tag "${repository}@${work_digest}" "${work_image}" || die "docker tag ${work_image} failed"
  docker tag "${repository}@${judge_digest}" "${judge_public_image}" || die "docker tag ${judge_public_image} failed"
}
