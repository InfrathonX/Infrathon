#!/usr/bin/env bash
# Build the Work and Judge images of the Miles OPCD task (v3) from a clean staging area.
# v3 = v2 with the teacher replaced by the frozen initial Qwen3-1.7B snapshot (contract teacher_hf), the data
# rebuilt from MedMCQA (scripts/prepare_assets.py; the same questions as v2), the checks that read candidate
# records with torch.load(weights_only=True), and the report-only paper-protocol diagnostic. The v3 checks,
# public runner and Judge come from scripts/miles_opcd/v3/; the instrumentation hooks are v2's, unchanged.
set -euo pipefail
TASK="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd -- "${TASK}/../../.." && pwd)"
TASK_ID=miles_qwen3_opcd_2xh100_v3
BENCH=miles_opcd
BASE_IMAGE="${BASE_IMAGE:-miles-model-dependencies-flat:local}"
MILES_BASE=78527b9102b3a5f15891128cbec3ac6c2dc4089e
# Host locations come from scripts/miles_common/host_config.py (environment, launch/mlsysforge.env,
# or derived from MILES_DATA_ROOT); it exits 78 naming whatever is not configured.
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
  host_settings="$("${HOST_CONFIG[@]}" shell OPCD_ASSETS_V3)"; eval "${host_settings}"
  ASSETS="${OPCD_ASSETS_V3}"
  public_source="${ASSETS}/public"
  work_source="${ASSETS}/work-public"
  private_source="${ASSETS}/private"
fi
HOOKS="${REPO}/scripts/miles_opcd/audit"
CHECKS="${REPO}/scripts/miles_opcd/v3"
docker_() { if docker info >/dev/null 2>&1; then docker "$@"; else sg docker -c "$(printf '%q ' docker "$@")"; fi; }

[[ "$(docker_ image inspect --format '{{.Id}}' "${BASE_IMAGE}")" == "${BASE_ID_EXPECTED}" ]] \
  || { echo "base image ${BASE_IMAGE} is not ${BASE_ID_EXPECTED}" >&2; exit 1; }
STAGE="$(mktemp -d)"; trap 'rm -rf "${STAGE}"' EXIT
for role in work judge; do mkdir -p "${STAGE}/${role}"/{init,workdir,hook,public-data}; done
git -C "${MILES_UPSTREAM}" archive --format=tar "${MILES_BASE}" | tar -x -C "${STAGE}/work/init"
# SForge rejects links in submissions; the base has one (.agents/skills -> ../.claude/skills, editor docs).
find "${STAGE}/work/init" -type l -print -delete
[[ -z "$(find "${STAGE}/work/init" -type l)" ]] || { echo "symlinks remain in the base export" >&2; exit 1; }
cp -a "${STAGE}/work/init/." "${STAGE}/judge/init/"
for role in work judge; do
  cp -a "${TASK}/images/work/workdir/." "${STAGE}/${role}/workdir/"
  cp "${HOOKS}/opcd_audit_hook.py" "${HOOKS}/opcd_audit_hook.pth" \
     "${HOOKS}/opcd_provenance_hook.py" "${HOOKS}/opcd_provenance_hook.pth" "${STAGE}/${role}/hook/"
done
# Work holds its small public data; the scored rows stay in the Judge's /data/public.
cp "${work_source}/"* "${STAGE}/work/public-data/"
cp "${public_source}/"* "${STAGE}/judge/public-data/"
public() {  # $1 = destination
  mkdir -p "$1/audit"
  cp "${CHECKS}"/{opcd_checks.py,opcd_oracle.py,opcd_probe_check.py,eval_medmcqa.py} "$1/"
  cp "${CHECKS}/run_public_v3.py" "$1/run_public.py"
  cp "${CHECKS}"/audit/{opcd_audit_check.py,opcd_provenance_check.py} "${HOOKS}/opcd_provenance_hook.py" "$1/audit/"
  cp "${TASK}/runtime/contract.json" "${TASK}/runtime/published_reference.json" "$1/"
}
public "${STAGE}/work/public"
# Work gets the small public contract; the scored contract stays with the Judge.
cp "${TASK}/runtime/public_contract.json" "${STAGE}/work/public/contract.json"
mv "${STAGE}/work/public/run_public.py" "${STAGE}/work/public/run_public_checks.py"
# Work only: the agent's run_public.py wraps the public runner with disk hygiene (scripts/miles_common/public_cleanup.py:
# a suite's bulky outputs are removed when it ends unless --keep-artifacts; only the last two outputs stay complete)
cp "${REPO}/scripts/miles_common/public_cleanup.py" "${STAGE}/work/public/"
printf 'import sys\nsys.path.insert(0, "/opt/miles_opcd_public")\nfrom public_cleanup import run\nrun("/opt/miles_opcd_public/run_public_checks.py", "--output", keep_flags=("--keep-snapshots",))\n' \
  > "${STAGE}/work/public/run_public.py"
cp "${TASK}/images/work/AGENT_TASK.md" "${STAGE}/work/AGENT_TASK.md"
cp "${TASK}/docs/INTERFACE.md" "${STAGE}/work/INTERFACE.md"
public "${STAGE}/judge/judge/public"
cp "${CHECKS}/run_judge_v3.py" "${STAGE}/judge/judge/run_judge.py"
cp "${TASK}/runtime/contract.json" "${STAGE}/judge/judge/"
# The Judge reads only the scoring part of the qualification record (run_judge.py: metric, timeouts, grad_limits).
# status and the qualification evidence stay on the host (scripts/run_task.py reads them there), so recording a
# reference Judge or clearing a PENDING status does not change the Judge image that was validated.
python3 - "${TASK}/runtime/qualification.json" "${STAGE}/judge/judge/qualification.json" <<'PY'
import json, sys
record = json.load(open(sys.argv[1]))
judge = {k: record[k] for k in ("schema_version", "task_id", "metric", "timeouts", "grad_limits")}
open(sys.argv[2], "w").write(json.dumps(judge, indent=2) + "\n")
PY
# Judge-only inputs enter only the host-built holdout layer (images/judge/Dockerfile.private),
# exactly as assets/manifest.json lists them (from the host asset directory, or from
# <INFRATHON_ASSETS_ROOT>/<task directory>/judge/private).
mkdir -p "${STAGE}/judge-private"
python3 "${REPO}/scripts/infrathon/assets.py" context "$(basename "${TASK}")" judge/private \
  "${private_source}" "${STAGE}/judge-private/private"
cp "${TASK}/images/judge/Dockerfile.private" "${STAGE}/judge-private/Dockerfile"
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
