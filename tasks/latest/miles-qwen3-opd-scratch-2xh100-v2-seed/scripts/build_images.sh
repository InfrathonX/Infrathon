#!/usr/bin/env bash
# Build the Work and Judge images of the Miles OPD-from-scratch v2 (math) task from a clean staging area.
# Work repo/ = Miles 78527b9 + the starter patch (on-policy distillation removed); the Judge
# also gets the reference tree = Miles 78527b9 + the reference patch + reference/task_entry.py. The patches and the
# reference are v1's, unchanged; the checks, hooks and the math eval come from scripts/miles_opd/v2/.
set -euo pipefail
TASK="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd -- "${TASK}/../../.." && pwd)"
TASK_ID=miles_qwen3_opd_scratch_2xh100_v2
BENCH=miles_opd
BASE_IMAGE="${BASE_IMAGE:-miles-model-dependencies-flat:local}"
MILES_BASE=78527b9102b3a5f15891128cbec3ac6c2dc4089e
# Host locations come from scripts/miles_common/host_config.py (environment, launch/mlsysforge.env,
# or derived from MILES_DATA_ROOT); it exits 78 naming whatever is not configured. Each value is
# assigned before use so that `set -e` stops the build on that exit.
HOST_CONFIG=(python3 "${REPO}/scripts/miles_common/host_config.py")
# The qualified build pinned sha256:1d61c60a...; a host that rebuilt the base with
# scripts/miles_base_images/build.sh accepts the image ID recorded there instead.
BASE_ID_EXPECTED="$("${HOST_CONFIG[@]}" base-id model-flat sha256:1d61c60ae09f7b2a100b0ba4c1032a0a6e098a326ec15a28c35deb725bbf172d)"
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
  host_settings="$("${HOST_CONFIG[@]}" shell OPD_ASSETS_V2)"; eval "${host_settings}"
  ASSETS="${OPD_ASSETS_V2}"
  [[ -d "${ASSETS}/public" && -d "${ASSETS}/private" ]] || { echo "OPD_ASSETS_V2=${ASSETS} has no public/ and private/" >&2; exit 78; }
  public_source="${ASSETS}/public"
  work_source="${ASSETS}/work-public"
  private_source="${ASSETS}/private"
fi
PATCHES="${REPO}/scripts/miles_opd"
CHECKS="${REPO}/scripts/miles_opd/v2"
STARTER_PATCH="${PATCHES}/starter/opd-starter-78527b910.patch"
REFERENCE_PATCH="${PATCHES}/reference/opd-reference-78527b910.patch"
docker_() { if docker info >/dev/null 2>&1; then docker "$@"; else sg docker -c "$(printf '%q ' docker "$@")"; fi; }

[[ "$(docker_ image inspect --format '{{.Id}}' "${BASE_IMAGE}")" == "${BASE_ID_EXPECTED}" ]] \
  || { echo "base image ${BASE_IMAGE} is not ${BASE_ID_EXPECTED}" >&2; exit 1; }
STAGE="$(mktemp -d)"; trap 'rm -rf "${STAGE}"' EXIT
miles_tree() {  # $1 = destination, $2 = patch: Miles base + patch, without links
  mkdir -p "$1"
  git -C "${MILES_UPSTREAM}" archive --format=tar "${MILES_BASE}" | tar -x -C "$1"
  # plain patch application: never let git discover an enclosing repository
  (cd "$1" && GIT_CEILING_DIRECTORIES="$(dirname "$1")" git apply --whitespace=nowarn "$2")
  # SForge rejects links in submissions; the base has one (.agents/skills -> ../.claude/skills, editor docs).
  find "$1" -type l -print -delete
  [[ -z "$(find "$1" -type l)" ]] || { echo "symlinks remain in $1" >&2; exit 1; }
}
for role in work judge; do mkdir -p "${STAGE}/${role}"/{workdir,hook,public-data}; done
miles_tree "${STAGE}/work/init" "${STARTER_PATCH}"
! grep -rqiE "opd|on_policy_distill|teacher_log" "${STAGE}/work/init/miles" \
  || { echo "the starter still mentions on-policy distillation" >&2; exit 1; }
cp -a "${STAGE}/work/init" "${STAGE}/judge/init"
for role in work judge; do
  cp -a "${TASK}/images/work/workdir/." "${STAGE}/${role}/workdir/"
  cp "${CHECKS}/audit/opd_audit_hook.py" "${CHECKS}/audit/opd_audit_hook.pth" \
     "${CHECKS}/audit/opd_provenance_hook.py" "${CHECKS}/audit/opd_provenance_hook.pth" "${STAGE}/${role}/hook/"
done
# Work holds its small public data; the scored rows and the split record stay in the Judge's /data/public.
cp "${work_source}/"* "${STAGE}/work/public-data/"
cp "${public_source}/"* "${STAGE}/judge/public-data/"
public() {  # $1 = destination
  mkdir -p "$1/audit" "$1/math_grader"
  cp "${CHECKS}"/{run_public.py,opd_checks.py,opd_probe_check.py,opd_step_oracle.py,eval_math.py} "$1/"
  cp "${CHECKS}"/math_grader/{__init__.py,grade.py,math_utils.py,math_dapo_utils.py,README.md} "$1/math_grader/"
  cp "${CHECKS}"/audit/{opd_audit_check.py,opd_provenance_check.py,opd_provenance_hook.py} "$1/audit/"
  cp "${TASK}/runtime/contract.json" "${TASK}/runtime/published_reference.json" "$1/"
}
public "${STAGE}/work/public"
# Work gets the small public contract; the scored contract stays with the Judge.
cp "${TASK}/runtime/public_contract.json" "${STAGE}/work/public/contract.json"
mv "${STAGE}/work/public/run_public.py" "${STAGE}/work/public/run_public_checks.py"
# Work only: the agent's run_public.py wraps the public runner with disk hygiene (scripts/miles_common/public_cleanup.py:
# a suite's bulky outputs are removed when it ends unless --keep-artifacts; only the last two outputs stay complete)
cp "${REPO}/scripts/miles_common/public_cleanup.py" "${STAGE}/work/public/"
printf 'import sys\nsys.path.insert(0, "/opt/miles_opd_public")\nfrom public_cleanup import run\nrun("/opt/miles_opd_public/run_public_checks.py", "--output", keep_flags=("--keep-snapshots",))\n' \
  > "${STAGE}/work/public/run_public.py"
cp "${TASK}/images/work/AGENT_TASK.md" "${STAGE}/work/AGENT_TASK.md"
cp "${TASK}/docs/INTERFACE.md" "${STAGE}/work/INTERFACE.md"
public "${STAGE}/judge/judge/public"
cp "${CHECKS}/run_judge.py" "${TASK}/runtime/contract.json" "${TASK}/runtime/qualification.json" "${STAGE}/judge/judge/"
# Judge-only inputs enter only the host-built holdout layer (images/judge/Dockerfile.private),
# exactly as assets/manifest.json lists them (from the host asset directory, or from
# <INFRATHON_ASSETS_ROOT>/<task directory>/judge/private).
mkdir -p "${STAGE}/judge-private"
python3 "${REPO}/scripts/infrathon/assets.py" context "$(basename "${TASK}")" judge/private \
  "${private_source}" "${STAGE}/judge-private/private"
cp "${TASK}/images/judge/Dockerfile.private" "${STAGE}/judge-private/Dockerfile"
miles_tree "${STAGE}/judge/judge/reference/repo" "${REFERENCE_PATCH}"
cp "${TASK}/reference/task_entry.py" "${STAGE}/judge/judge/reference/task_entry.py"
cp "${TASK}/images/work/Dockerfile" "${STAGE}/work/Dockerfile"
cp "${TASK}/images/judge/Dockerfile" "${STAGE}/judge/Dockerfile"
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
