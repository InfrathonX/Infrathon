#!/usr/bin/env bash
# Build the Work and Judge images of the SD3.5-M DiffusionOPD task v2 from a clean staging area.
#   MDOPD_V2_PUBLIC_DATA  Work's public root (scripts/miles_diffusion_opd/trim_public.py); only its small files are baked in (rows.jsonl, student_init.safetensors and
#                heldout.jsonl, the public held-out prompts from the training split; the public frozen values are in
#                runtime/published_reference.json). The probe/ and replay/ recordings (~3 GB), the task-metric
#                scorers (MDOPD_V2_SCORERS) and the Judge's private root are bind-mounted read-only at launch
#                (runtime/sforge_entry.py) rather than baked into image layers.
# The checkers are scripts/miles_diffusion_opd/judge_v2 (v1 keeps scripts/miles_diffusion_opd/judge).
# Host locations come from scripts/miles_common/host_config.py (environment, launch/mlsysforge.env, or derived from
# MILES_DATA_ROOT); it exits 78 naming whatever is not configured. The staging area lives under
# $MILES_SCRATCH_ROOT/<task>/build.
set -euo pipefail
TASK="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd -- "${TASK}/../../.." && pwd)"
TASK_ID=miles_diffusion_sd35m_mopd_2xh100_v2
BENCH=miles_diffusion_opd
BASE_IMAGE="${BASE_IMAGE:-miles-diffusion-base:sgl859a278e-md29fde6ee}"
MD_BASE=29fde6ee
HOST_CONFIG=(python3 "${REPO}/scripts/miles_common/host_config.py")
# Each value is assigned before use so that `set -e` stops the build on host_config.py's exit 78.
# The qualified build pinned sha256:5ba5a974...; a host that rebuilt the base with
# scripts/miles_base_images/build.sh accepts the image ID recorded there instead.
BASE_ID_EXPECTED="$("${HOST_CONFIG[@]}" base-id diffusion sha256:5ba5a9748fb816907e045d616f4d44f7fecfe5db24881e94fd285859857df84a)"
MD_UPSTREAM="$("${HOST_CONFIG[@]}" upstream MD_UPSTREAM "${MD_BASE}")"
host_settings="$("${HOST_CONFIG[@]}" shell MDOPD_V2_PUBLIC_DATA)"; eval "${host_settings}"
PUBLIC_DATA="${MDOPD_V2_PUBLIC_DATA}"
for f in rows.jsonl student_init.safetensors heldout.jsonl; do
  [[ -f "${PUBLIC_DATA}/${f}" ]] || { echo "MDOPD_V2_PUBLIC_DATA=${PUBLIC_DATA} lacks ${f}" >&2; exit 78; }
done
BUILD_SCRATCH="$("${HOST_CONFIG[@]}" scratch "$(basename "${TASK}")" 50)/build"
OPD="${REPO}/scripts/miles_diffusion_opd"
CHECKS=(run_judge.py replay_server.py probe_check.py adamw_check.py audit_check.py framework_check.py publication_check.py
        quality_eval.py task_quality_eval.py)
docker_() { if docker info >/dev/null 2>&1; then docker "$@"; else sg docker -c "$(printf '%q ' docker "$@")"; fi; }

[[ "$(docker_ image inspect --format '{{.Id}}' "${BASE_IMAGE}")" == "${BASE_ID_EXPECTED}" ]] \
  || { echo "base image ${BASE_IMAGE} is not ${BASE_ID_EXPECTED}" >&2; exit 1; }
mkdir -p "${BUILD_SCRATCH}"
STAGE="$(mktemp -d -p "${BUILD_SCRATCH}")"; trap 'rm -rf "${STAGE}"' EXIT
for role in work judge; do mkdir -p "${STAGE}/${role}"/{hook,md_pristine}; done
mkdir -p "${STAGE}/work"/{init,workdir,public/judge,public/oracle,public-data}
git -C "${MD_UPSTREAM}" archive --format=tar "${MD_BASE}" | tar -x -C "${STAGE}/work/init"
find "${STAGE}/work/init" -type l -print -delete
git -C "${MD_UPSTREAM}" archive --format=tar "${MD_BASE}" | tar -x -C "${STAGE}/work/md_pristine"
cp -a "${STAGE}/work/md_pristine/." "${STAGE}/judge/md_pristine/"
cp -a "${TASK}/images/work/workdir/." "${STAGE}/work/workdir/"
cp "${TASK}/images/work/AGENT_TASK.md" "${TASK}/docs/INTERFACE.md" "${STAGE}/work/"
for role in work judge; do cp "${OPD}/judge/audit/opd_audit_hook.py" "${OPD}/judge/audit/opd_audit_hook.pth" "${STAGE}/${role}/hook/"; done
# public tools: the Judge's checkers without run_judge.py, public thresholds and the published public values
for f in run_public.py public_common.py "${CHECKS[@]}"; do
  [[ "${f}" == run_judge.py ]] || cp "${OPD}/judge_v2/${f}" "${STAGE}/work/public/judge/"
done
cp "${OPD}/oracle/sd3_oracle.py" "${STAGE}/work/public/oracle/"
# Work gets the small public contract; the scored contract stays with the Judge.
cp "${TASK}/runtime/public_contract.json" "${STAGE}/work/public/judge/contract.json"
cp "${TASK}/runtime/public_qualification.json" "${TASK}/runtime/published_reference.json" "${STAGE}/work/public/judge/"
mv "${STAGE}/work/public/judge/public_qualification.json" "${STAGE}/work/public/judge/qualification.json"
# Work only: the agent's run_public.py wraps the public runner with disk hygiene (scripts/miles_common/public_cleanup.py:
# a suite's bulky outputs are removed when it ends unless --keep-artifacts; only the last two outputs stay complete)
cp "${REPO}/scripts/miles_common/public_cleanup.py" "${STAGE}/work/public/"
printf 'import sys\nsys.path.insert(0, "/opt/opd_public")\nfrom public_cleanup import run\nrun("/opt/opd_public/judge/run_public.py", "--output")\n' \
  > "${STAGE}/work/public/run_public.py"
cp "${PUBLIC_DATA}"/{rows.jsonl,student_init.safetensors,heldout.jsonl} "${STAGE}/work/public-data/"
cp "${TASK}/images/work/Dockerfile" "${STAGE}/work/Dockerfile"
# judge: checkers + oracle + contract + qualification (private root mounted at /judge_private)
mkdir -p "${STAGE}/judge/judge/judge" "${STAGE}/judge/judge/oracle"
for f in "${CHECKS[@]}"; do cp "${OPD}/judge_v2/${f}" "${STAGE}/judge/judge/judge/"; done
cp "${OPD}/oracle/sd3_oracle.py" "${STAGE}/judge/judge/oracle/"
cp "${TASK}/runtime/contract.json" "${TASK}/runtime/qualification.json" "${STAGE}/judge/judge/judge/"
cp "${TASK}/images/judge/Dockerfile" "${STAGE}/judge/Dockerfile"
for role in work judge; do
  docker_ build --build-arg "BASE_IMAGE=${BASE_IMAGE}" -t "${BENCH}.${role}.${TASK_ID}:local" "${STAGE}/${role}"
done
for role in work judge; do
  printf '%s %s\n' "${BENCH}.${role}.${TASK_ID}:local" "$(docker_ image inspect --format '{{.Id}}' "${BENCH}.${role}.${TASK_ID}:local")"
done
# The build record (images/release.json, scripts/infrathon/image_record.py): what publishing and
# scripts/infrathon/pull_images.sh compare against.
if docker info >/dev/null 2>&1; then python3 "${REPO}/scripts/infrathon/image_record.py" write "$(basename "${TASK}")";
else sg docker -c "$(printf '%q ' python3 "${REPO}/scripts/infrathon/image_record.py" write "$(basename "${TASK}")")"; fi
