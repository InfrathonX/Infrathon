#!/usr/bin/env bash
# Build the Work and Judge images of the Miles ALFWorld task from a clean staging area.
set -euo pipefail
TASK="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd -- "${TASK}/../../.." && pwd)"
TASK_ID=miles_qwen3_alfworld_agentic_rl_2xh100_v1
BENCH=miles_alfworld
BASE_IMAGE="${BASE_IMAGE:-miles-alfworld-base:local}"
MILES_BASE=78527b9102b3a5f15891128cbec3ac6c2dc4089e
# Host locations come from scripts/miles_common/host_config.py (environment, launch/mlsysforge.env,
# or derived from MILES_DATA_ROOT); it exits 78 naming whatever is not configured. Each value is
# assigned before use so that `set -e` stops the build on that exit.
HOST_CONFIG=(python3 "${REPO}/scripts/miles_common/host_config.py")
# The qualified build pinned the ID in tasks/BENCHMARK.yaml; a host that rebuilt the base with
# scripts/miles_base_images/build.sh accepts the image ID recorded there instead.
PINNED_BASE_ID="$(sed -n 's/.*official_image: //p' "${TASK}/tasks/BENCHMARK.yaml")"
BASE_ID_EXPECTED="$("${HOST_CONFIG[@]}" base-id alfworld "${PINNED_BASE_ID}")"
MILES_UPSTREAM="$("${HOST_CONFIG[@]}" upstream MILES_UPSTREAM "${MILES_BASE}")"
# Unset: the default root of assets/fetch.sh, when it exists.
[[ -n "${INFRATHON_ASSETS_ROOT:-}" || ! -d "${HOME}/.cache/infrathon/assets" ]] \
  || INFRATHON_ASSETS_ROOT="${HOME}/.cache/infrathon/assets"
if [[ -n "${INFRATHON_ASSETS_ROOT:-}" ]]; then
  # Infrathon assets (assets/fetch.sh): the public task data and the Judge-only data.
  public_source="${INFRATHON_ASSETS_ROOT}/$(basename "${TASK}")/public/task-data"
  work_source="${INFRATHON_ASSETS_ROOT}/$(basename "${TASK}")/public/work-task-data"
  private_source="${INFRATHON_ASSETS_ROOT}/$(basename "${TASK}")/judge/private"
else
  host_settings="$("${HOST_CONFIG[@]}" shell ALFW_ASSETS)"; eval "${host_settings}"
  ASSETS="${ALFW_ASSETS}"
  public_source="${ASSETS}/public"
  work_source="${ASSETS}/work-public"
  private_source="${ASSETS}/private"
fi
CHECKS="${REPO}/scripts/miles_alfworld"
docker_() { if docker info >/dev/null 2>&1; then docker "$@"; else sg docker -c "$(printf '%q ' docker "$@")"; fi; }

[[ "$(docker_ image inspect --format '{{.Id}}' "${BASE_IMAGE}")" == "${BASE_ID_EXPECTED}" ]] \
  || { echo "base image ${BASE_IMAGE} is not ${BASE_ID_EXPECTED}" >&2; exit 1; }
for f in "${public_source}/probe_batch.json" "${private_source}/probe_batch.json" "${public_source}/sample_run/contract.json" \
         "${work_source}/probe_batch.json" "${work_source}/sample_run/contract.json"; do
  [[ -f "${f}" ]] || { echo "missing asset ${f}" >&2; exit 1; }
done
STAGE="$(mktemp -d)"; trap 'rm -rf "${STAGE}"' EXIT
for role in work judge; do mkdir -p "${STAGE}/${role}"/{init,workdir,hook,public-data}; done
git -C "${MILES_UPSTREAM}" archive --format=tar "${MILES_BASE}" | tar -x -C "${STAGE}/work/init"
find "${STAGE}/work/init" -type l -print -delete
cp -a "${STAGE}/work/init/." "${STAGE}/judge/init/"
for role in work judge; do
  cp -a "${TASK}/images/work/workdir/." "${STAGE}/${role}/workdir/"
  cp "${CHECKS}"/audit/alfworld_{audit,provenance}_hook.{py,pth} "${STAGE}/${role}/hook/"
done
# Work holds its small public data; the Judge's public data stays as it was.
cp -a "${work_source}/." "${STAGE}/work/public-data/"
cp -a "${public_source}/." "${STAGE}/judge/public-data/"
public() {  # $1 = destination
  mkdir -p "$1/audit"
  cp "${CHECKS}"/{alfworld_checks.py,alfworld_pipeline.py,alfworld_oracle.py,alfworld_probe_check.py,eval_alfworld.py,run_public.py} "$1/"
  cp "${CHECKS}"/audit/{alfworld_audit_check.py,replay_check.py,alfworld_provenance_check.py,alfworld_provenance_hook.py} "$1/audit/"
  cp "${TASK}/runtime/contract.json" "$1/"
}
public "${STAGE}/work/public"
# Work gets the small public contract; the scored contract stays with the Judge.
cp "${TASK}/runtime/public_contract.json" "${STAGE}/work/public/contract.json"
cp "${TASK}/images/work/AGENT_TASK.md" "${STAGE}/work/AGENT_TASK.md"
cp "${TASK}/docs/INTERFACE.md" "${STAGE}/work/INTERFACE.md"
public "${STAGE}/judge/judge/public"
cp "${CHECKS}/run_judge.py" "${STAGE}/judge/judge/run_judge.py"
cp "${TASK}/runtime/contract.json" "${TASK}/runtime/qualification.json" "${STAGE}/judge/judge/"
# Judge-only inputs enter only the host-built holdout layer (images/judge/Dockerfile.private),
# exactly as assets/manifest.json lists them (from the host asset directory, or from
# <INFRATHON_ASSETS_ROOT>/<task directory>/judge/private).
mkdir -p "${STAGE}/judge-private"
python3 "${REPO}/scripts/infrathon/assets.py" context "$(basename "${TASK}")" judge/private \
  "${private_source}" "${STAGE}/judge-private/private" --allow-unlisted
cp "${TASK}/images/judge/Dockerfile.private" "${STAGE}/judge-private/Dockerfile"
cp "${TASK}/images/work/Dockerfile" "${STAGE}/work/Dockerfile"
cp "${TASK}/images/judge/Dockerfile" "${STAGE}/judge/Dockerfile"
find "${STAGE}" -name __pycache__ -prune -exec rm -rf {} +
docker_ build --build-arg "BASE_IMAGE=${BASE_IMAGE}" -t "${BENCH}.work.${TASK_ID}:local" "${STAGE}/work"
# The public Judge has no private/ and may be published; the holdout layer is built here only.
docker_ build --build-arg "BASE_IMAGE=${BASE_IMAGE}" -t "${BENCH}.judge-public.${TASK_ID}:local" "${STAGE}/judge"
docker_ build --build-arg "JUDGE_PUBLIC_IMAGE=${BENCH}.judge-public.${TASK_ID}:local" \
  -t "${BENCH}.judge.${TASK_ID}:local" "${STAGE}/judge-private"
for role in work judge-public judge; do
  printf '%s %s\n' "${BENCH}.${role}.${TASK_ID}:local" "$(docker_ image inspect --format '{{.Id}}' "${BENCH}.${role}.${TASK_ID}:local")"
done
# The build record (images/release.json, scripts/infrathon/image_record.py): what publishing and
# scripts/infrathon/pull_images.sh compare against.
if docker info >/dev/null 2>&1; then python3 "${REPO}/scripts/infrathon/image_record.py" write "$(basename "${TASK}")";
else sg docker -c "$(printf '%q ' python3 "${REPO}/scripts/infrathon/image_record.py" write "$(basename "${TASK}")")"; fi
