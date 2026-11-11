#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib.sh"
SFORGE_ENTRY_SCRIPT="$0"; export SFORGE_ENTRY_SCRIPT
ensure_docker_group "$@"
require_command docker

source "${REPOSITORY_ROOT}/scripts/infrathon/images.sh"

assets="${SDV2_ASSETS_ROOT:-${MIGRATION_ROOT}/build/assets}"   # produced by scripts/stage_assets.py
core="${SDV2_CORE_ROOT:-${MIGRATION_ROOT}/runtime}"
# The Judge's goldens are measured on the card, so they and the hash manifest that binds
# them (images/judge/protected_hashes/<hardware>.txt) are per hardware type: build/
# holds this host's (scripts/generate_goldens.py).  With INFRATHON_ASSETS_ROOT set, the
# goldens and the Judge-only inputs come from <root>/<task directory>/ (hw/<hardware>/
# and judge/), checked against assets/manifest.json first.  They enter only the holdout
# layer: the Work and public Judge images are the same for every hardware type.
hardware="$(infrathon_hardware)" || exit 1
goldens_judge="${MIGRATION_ROOT}/build/goldens_judge"
final_cases="${MIGRATION_ROOT}/build/private"
judge_assets="${assets}"
# Unset: the default root of assets/fetch.sh, when it exists.
[[ -n "${INFRATHON_ASSETS_ROOT:-}" || ! -d "${HOME}/.cache/infrathon/assets" ]] \
  || INFRATHON_ASSETS_ROOT="${HOME}/.cache/infrathon/assets"
if [[ -n "${INFRATHON_ASSETS_ROOT:-}" ]]; then
  task_assets="${INFRATHON_ASSETS_ROOT}/$(basename "${MIGRATION_ROOT}")"
  python3 "${REPOSITORY_ROOT}/scripts/infrathon/assets.py" check "$(basename "${MIGRATION_ROOT}")" --hardware "${hardware}" \
    || die "fetch the task's assets first: assets/fetch.sh"
  goldens_judge="${task_assets}/hw/${hardware}/judge/goldens_judge"
  final_cases="${task_assets}/judge"
  judge_assets="${task_assets}/judge"
fi
protected_hashes="${MIGRATION_ROOT}/images/judge/protected_hashes/${hardware}.txt"
for path in "${protected_hashes}" "${judge_assets}/embeddings/hidden" "${judge_assets}/clips" "${goldens_judge}/final/manifest.json" "${final_cases}/final_cases.json"; do [[ -e "${path}" ]] || die "missing build context input: ${path}"; done
# Refuse another card's goldens before building; the Judge re-checks every line.
grep ' private:goldens/' "${protected_hashes}" | while read -r digest label; do printf '%s  %s\n' "${digest}" "${goldens_judge}/${label#private:goldens/}"; done \
  | sha256sum --check --quiet --strict \
  || die "${goldens_judge} does not hold the ${hardware} goldens that ${protected_hashes#"${MIGRATION_ROOT}/"} binds"
mkdir -p "${LOGS_DIR}/build"
mkdir -p "${LOGS_DIR}/build/final-cases-context" "${LOGS_DIR}/build/protected-hashes-context"
cp -f "${final_cases}/final_cases.json" "${LOGS_DIR}/build/final-cases-context/final_cases.json"
cp -f "${protected_hashes}" "${LOGS_DIR}/build/protected-hashes-context/protected_hashes.txt"
source_digest="$(image_source_digest)"
started="$(date +%s)"
if infrathon_pull_published_images "${MIGRATION_ROOT}" "${WORK_IMAGE}" "${JUDGE_PUBLIC_IMAGE}" "${source_digest}"; then
  echo "Pulled published ${INFRATHON_HARDWARE} images as ${WORK_IMAGE} and ${JUDGE_PUBLIC_IMAGE}"
else
  for path in "${assets}/weights/checkpoint_manifest.sha256" "${assets}/wheels" "${assets}/work-context/embeddings/public" "${core}/sdv2_core" "${MIGRATION_ROOT}/oracle/candidates/runtime_v001/runtime.py"; do [[ -e "${path}" ]] || die "missing build context input: ${path}"; done
  common=(--build-context "weights=${assets}/weights" --build-context "flash_wheel=${assets}/wheels" --build-context "upstream_smoke=${assets}/upstream/StreamDiffusionV2")
  # The Work image only ever sees the public subset that stage_assets.py hard-links
  # into work-context/; hidden embeddings and hidden clips are Judge-only inputs.
  [[ -d "${assets}/work-context/embeddings/public" ]] || die "missing build context input: ${assets}/work-context (run scripts/stage_assets.py)"
  # Build a sanitized Work context so final definitions never enter an image layer.
  work_core="${MIGRATION_ROOT}/build/work_runtime"
  python3 "${MIGRATION_ROOT}/scripts/prepare_work_runtime.py" "${core}" "${work_core}"
  docker build --progress=plain "${common[@]}" --build-context "assets=${assets}/work-context" --build-context "task_runtime=${work_core}" --build-context "seed_runtime=${MIGRATION_ROOT}/oracle/candidates" -t "${WORK_IMAGE}" "${MIGRATION_ROOT}/images/work"
  docker build --progress=plain "${common[@]}" --build-context "assets=${assets}" --build-context "task_runtime=${core}" --build-context "seed_runtime=${MIGRATION_ROOT}/oracle/candidates" -t "${JUDGE_PUBLIC_IMAGE}" "${MIGRATION_ROOT}/images/judge"
fi
# The holdout layer is built here in every mode and never published.
docker build --progress=plain --build-arg "JUDGE_PUBLIC_IMAGE=${JUDGE_PUBLIC_IMAGE}" --build-context "assets=${judge_assets}" --build-context "goldens_judge=${goldens_judge}" --build-context "final_cases=${final_cases}" --build-context "protected_hashes=${LOGS_DIR}/build/protected-hashes-context" -f "${MIGRATION_ROOT}/images/judge/Dockerfile.private" -t "${JUDGE_IMAGE}" "${MIGRATION_ROOT}/images/judge"
ended="$(date +%s)"
[[ "$(image_source_digest)" == "${source_digest}" ]] || die "image sources changed during the build; discard and rebuild"
final_sha="$(sha256sum "${LOGS_DIR}/build/final-cases-context/final_cases.json" | awk '{print $1}')"
# Only the holdout layer (the Judge's goldens) is for this hardware type; the Work and public
# Judge images are published for both.
python3 - "${LOGS_DIR}/build/image-manifest.json" "${WORK_IMAGE}" "${JUDGE_IMAGE}" "${JUDGE_PUBLIC_IMAGE}" "$((ended-started))" "${source_digest}" "${final_sha}" "${hardware}" <<'PY'
import json, subprocess, sys
out, work, judge, judge_public, wall, source_digest, final_sha, hardware = sys.argv[1:]
def inspect(image):
    raw=subprocess.check_output(["docker","image","inspect",image], text=True)
    row=json.loads(raw)[0]
    return {"tag":image,"id":row["Id"],"size_bytes":row["Size"]}
open(out,"w").write(json.dumps({"schema_version":1,"wall_seconds":int(wall),"source_digest":source_digest,"holdout_hardware":hardware,"final_cases":{"sha256":final_sha},"work":inspect(work),"judge":inspect(judge),"judge_public":inspect(judge_public)},indent=2,sort_keys=True)+"\n")
PY
cat "${LOGS_DIR}/build/image-manifest.json"
