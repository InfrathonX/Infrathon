#!/usr/bin/env bash
# Publish a task's Work and public Judge images for one or more hardware types.
#
#   scripts/infrathon/publish_images.sh <task directory> <hardware>... [--dry-run] [--allow-untested]
#
# Pushes the images the task's last scripts/build_images.sh recorded
# (outputs/<task directory>/build/image-manifest.json, checked current first) as
#
#   ${INFRATHON_REGISTRY_REPOSITORY}:<hardware>-<task directory>-work
#   ${INFRATHON_REGISTRY_REPOSITORY}:<hardware>-<task directory>-judge
#
# (the -judge image is the public Judge: it holds no held-out data).  INFRATHON_RELEASE=<label>
# appends -<label> to both tags, to publish a second set next to an existing one.
#
# and records their digests in tasks/latest/<task directory>/images/published.json.
# INFRATHON_REGISTRY_REPOSITORY defaults to scripts/infrathon/registry.json; the record
# does not name the repository, so publishing to another one with the same digests
# (crane copy) needs no new record.
# Only a hardware type scripts/infrathon/tasks.json marks qualified is published
# (untested too with --allow-untested, after a smoke run on that hardware).
# One build serves every hardware type named here: the tags differ, the digests
# do not.  A build that records its hardware type (image-manifest.json
# "hardware") is published for that type only.  The host-built holdout Judge is
# never pushed.
#
# INFRATHON_PUSH_TOOL=crane uploads through crane (go-containerregistry; INFRATHON_CRANE names the binary)
# from a `docker save` of each image instead of `docker push`.  docker push tries to mount
# layers from the local images' implied docker.io/library/<name> origins, which a token scoped
# to the target repository is refused; crane uploads every missing blob itself.  The saved
# layouts go to a temporary directory under INFRATHON_PUSH_TMP (default TMPDIR or /tmp).
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
die() { echo "ERROR: $*" >&2; exit 1; }
# shellcheck source=images.sh
source "${repository_root}/scripts/infrathon/images.sh"
registry_repository="$(infrathon_registry_repository)" || exit 1
release="${INFRATHON_RELEASE:-}"

dry_run=0
allow_untested=()
positional=()
for argument in "$@"; do
  case "${argument}" in
    --dry-run) dry_run=1 ;;
    --allow-untested) allow_untested=(--allow-untested) ;;
    -*) die "unknown option: ${argument}" ;;
    *) positional+=("${argument}") ;;
  esac
done
(( ${#positional[@]} >= 2 )) || die "usage: publish_images.sh <task directory> <hardware>... [--dry-run]"
task="${positional[0]}"
hardware=("${positional[@]:1}")
task_root="${repository_root}/tasks/latest/${task}"
[[ -d "${task_root}" ]] || die "no such task: ${task_root}"
for hw in "${hardware[@]}"; do
  [[ "${hw}" =~ ^h100-[0-9]+gb-[0-9]+w$ ]] || die "hardware must look like h100-94gb-400w: ${hw}"
  python3 "${repository_root}/scripts/infrathon/tasks.py" check "${task}" "${hw}" "${allow_untested[@]}" \
    || die "not publishing ${task} for ${hw}"
done
[[ -z "${release}" || "${release}" =~ ^[a-z0-9][a-z0-9.]*$ ]] || die "INFRATHON_RELEASE must be a short lower-case label: ${release}"

# The task's own freshness check: the recorded images are the ones built from
# this checkout's image-driving source.  A task that keeps no source digest
# (images/release.json) is checked through scripts/infrathon/image_record.py.
if [[ -f "${task_root}/images/release.json" ]]; then
  python3 "${repository_root}/scripts/infrathon/image_record.py" check "${task}"
  manifest="${repository_root}/outputs/${task}/build/image-manifest.json"
else
  # shellcheck source=/dev/null
  source "${task_root}/scripts/lib.sh"
  assert_built_images_current
  manifest="${BUILD_DIR:-${LOGS_DIR}/build}/image-manifest.json"
fi
source_digest="$(jq -er '.source_digest' "${manifest}")"
work_image="$(jq -er '.work.name // .work.tag' "${manifest}")"
judge_public_image="$(jq -er '.judge_public.name // .judge_public.tag' "${manifest}")" \
  || die "${manifest} has no judge_public image; this task has no public Judge split yet"
# A task whose published images depend on the card records the hardware type it
# built for ("hardware"); publish each type from its own build.
built_for="$(jq -r '.hardware // empty' "${manifest}")"
if [[ -n "${built_for}" ]]; then
  for hw in "${hardware[@]}"; do
    [[ "${hw}" == "${built_for}" ]] || die "these images were built for ${built_for}, not ${hw}"
  done
fi

push_tool="${INFRATHON_PUSH_TOOL:-docker}"
crane="${INFRATHON_CRANE:-crane}"
case "${push_tool}" in
  docker) ;;
  crane) command -v "${crane}" >/dev/null || die "INFRATHON_PUSH_TOOL=crane: ${crane} not found" ;;
  *) die "INFRATHON_PUSH_TOOL must be docker or crane, not ${push_tool}" ;;
esac

pushed_digest() {
  # The registry digest of a pushed tag (repository@sha256:...).
  if [[ "${push_tool}" == crane ]]; then
    "${crane}" digest "$1"
    return
  fi
  docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' "$1" \
    | grep -m1 "^${registry_repository}@sha256:" | cut -d@ -f2
}

work_layout=""
judge_layout=""
if [[ "${push_tool}" == crane ]] && (( ! dry_run )); then
  work_layout="$(mktemp -d "${INFRATHON_PUSH_TMP:-${TMPDIR:-/tmp}}/infrathon-push.XXXXXX")"
  judge_layout="$(mktemp -d "${INFRATHON_PUSH_TMP:-${TMPDIR:-/tmp}}/infrathon-push.XXXXXX")"
  trap 'rm -rf -- "${work_layout}" "${judge_layout}"' EXIT
  docker save "${work_image}" | tar -x -C "${work_layout}"
  docker save "${judge_public_image}" | tar -x -C "${judge_layout}"
fi

push() {
  # push LAYOUT TAG: upload one image (crane: its saved layout) under TAG.
  if [[ "${push_tool}" == crane ]]; then
    "${crane}" push "$1" "$2" >/dev/null
  else
    docker push "$2"
  fi
}

published="${task_root}/images/published.json"
document='{}'
if [[ -f "${published}" ]]; then
  document="$(cat "${published}")"
  [[ "$(jq -r '.source_digest // empty' <<<"${document}")" == "${source_digest}" ]] \
    || document='{}'
fi
for hw in "${hardware[@]}"; do
  work_tag="${registry_repository}:${hw}-${task}-work${release:+-${release}}"
  judge_tag="${registry_repository}:${hw}-${task}-judge${release:+-${release}}"
  (( ${#work_tag} - ${#registry_repository} - 1 <= 128 )) || die "tag too long: ${work_tag}"
  (( ${#judge_tag} - ${#registry_repository} - 1 <= 128 )) || die "tag too long: ${judge_tag}"
  echo "${work_image} -> ${work_tag}"
  echo "${judge_public_image} -> ${judge_tag}"
  (( dry_run )) && continue
  docker tag "${work_image}" "${work_tag}"
  docker tag "${judge_public_image}" "${judge_tag}"
  push "${work_layout}" "${work_tag}"
  push "${judge_layout}" "${judge_tag}"
  work_digest="$(pushed_digest "${work_tag}")"
  judge_digest="$(pushed_digest "${judge_tag}")"
  [[ "${work_digest}" =~ ^sha256:[0-9a-f]{64}$ && "${judge_digest}" =~ ^sha256:[0-9a-f]{64}$ ]] \
    || die "could not read the pushed digests for ${hw}"
  document="$(jq \
    --arg source_digest "${source_digest}" \
    --arg hw "${hw}" --arg release "${release}" \
    --arg work "${work_digest}" --arg work_tag "${work_tag##*:}" \
    --arg judge "${judge_digest}" --arg judge_tag "${judge_tag##*:}" \
    'del(.repository) | .source_digest = $source_digest
     | .hardware[$hw] = ({work: $work, work_tag: $work_tag, judge_public: $judge, judge_public_tag: $judge_tag}
                         + (if $release == "" then {} else {release: $release} end))' \
    <<<"${document}")"
done
(( dry_run )) && { echo "dry run: nothing pushed"; exit 0; }
printf '%s\n' "${document}" | jq -S . >"${published}.tmp.$$"
mv -f -- "${published}.tmp.$$" "${published}"
echo "recorded ${published}"
