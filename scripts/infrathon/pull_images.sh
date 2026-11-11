#!/usr/bin/env bash
# Pull a task's published images instead of building them, for a task whose own
# scripts/build_images.sh has no INFRATHON_PULL mode (it declares images/release.json:
# the Miles tasks, MInference v3).
#
#   INFRATHON_ASSETS_ROOT=<asset root> scripts/infrathon/pull_images.sh <task directory>
#
# Pulls the Work and public Judge published for this host's hardware type
# (INFRATHON_HARDWARE, or detected) by digest, refusing images published from other
# sources; tags them with the task's local names; builds the holdout layer from
# the task's Judge-only assets when the task has one (images/judge/Dockerfile.private);
# and writes the build record (scripts/infrathon/image_record.py) the task's checks read.
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
die() { echo "ERROR: $*" >&2; exit 1; }
(( $# == 1 )) || die "usage: pull_images.sh <task directory>"
task="$1"
task_root="${repository_root}/tasks/latest/${task}"
[[ -d "${task_root}" ]] || die "no such task: ${task_root}"
docker info >/dev/null 2>&1 || die "Docker is unreachable (run inside the docker group: sg docker -c ...)"
record=(python3 "${repository_root}/scripts/infrathon/image_record.py")
# shellcheck source=images.sh
source "${repository_root}/scripts/infrathon/images.sh"

spec="$("${record[@]}" spec "${task}")"
digest="$("${record[@]}" digest "${task}")"
work="$(jq -er '.local_tags.work' <<<"${spec}")"
judge="$(jq -er '.local_tags.judge' <<<"${spec}")"
judge_public="$(jq -er '.local_tags.judge_public // .local_tags.judge' <<<"${spec}")"

INFRATHON_PULL=1 infrathon_pull_published_images "${task_root}" "${work}" "${judge_public}" "${digest}" \
  || die "nothing was pulled"
echo "Pulled published ${INFRATHON_HARDWARE} images as ${work} and ${judge_public}"

if jq -e '.holdout' <<<"${spec}" >/dev/null; then
  : "${INFRATHON_ASSETS_ROOT:?set INFRATHON_ASSETS_ROOT to the asset root holding the Judge-only assets of this task (assets/fetch.sh)}"
  prefix="$(jq -er '.holdout.prefix' <<<"${spec}")"
  context_name="$(jq -er '.holdout.context' <<<"${spec}")"
  unlisted=()
  [[ "$(jq -r '.holdout.allow_unlisted // false' <<<"${spec}")" == true ]] && unlisted=(--allow-unlisted)
  context="$(mktemp -d)"; trap 'rm -rf "${context}"' EXIT
  python3 "${repository_root}/scripts/infrathon/assets.py" context "${task}" "${prefix}" \
    "${INFRATHON_ASSETS_ROOT}/${task}/${prefix}" "${context}/${context_name}" "${unlisted[@]}"
  cp "${task_root}/images/judge/Dockerfile.private" "${context}/Dockerfile"
  # The holdout layer is built here in every mode and never published.
  docker build --build-arg "JUDGE_PUBLIC_IMAGE=${judge_public}" -t "${judge}" "${context}"
fi
"${record[@]}" write "${task}"
# Owner-root assets (the Miles diffusion private roots) that a host user with the
# container agent's uid fetched: hand them to root through the Judge just pulled (no
# sudo); assets/fetch.sh does the same when the images are already here.
if [[ -n "${INFRATHON_ASSETS_ROOT:-}" ]] \
    && jq -e '[.assets[] | select(.owner == "root")] | length > 0' "${task_root}/assets/manifest.json" >/dev/null 2>&1; then
  python3 "${repository_root}/scripts/infrathon/assets.py" own "${task}" --image "${judge_public}" --agent-owned \
    --hardware "${INFRATHON_HARDWARE}"
fi
