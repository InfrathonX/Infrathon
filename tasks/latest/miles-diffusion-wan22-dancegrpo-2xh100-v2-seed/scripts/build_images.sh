#!/usr/bin/env bash
# Build the Work and Judge images of the Wan2.2-TI2V-5B DanceGRPO task from a clean staging area. Models, the public
# data root and the Judge's private root are bind-mounted read-only at launch (runtime/sforge_entry.py), never baked
# into image layers. Host locations come from scripts/miles_common/host_config.py (environment, launch/mlsysforge.env);
# it exits 78 naming whatever is not configured. The staging area lives under $MILES_SCRATCH_ROOT/<task>/build.
set -euo pipefail
TASK="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd -- "${TASK}/../../.." && pwd)"
TASK_ID=miles_diffusion_wan22_dancegrpo_2xh100_v2
BENCH=miles_diffusion_wan
BASE_IMAGE="${BASE_IMAGE:-miles-diffusion-base:sgl859a278e-md29fde6ee}"
MD_BASE=29fde6ee
HOST_CONFIG=(python3 "${REPO}/scripts/miles_common/host_config.py")
# Each value is assigned before use so that `set -e` stops the build on host_config.py's exit 78.
# The qualified build pinned sha256:5ba5a974...; a host that rebuilt the base with
# scripts/miles_base_images/build.sh accepts the image ID recorded there instead.
BASE_ID_EXPECTED="$("${HOST_CONFIG[@]}" base-id diffusion sha256:5ba5a9748fb816907e045d616f4d44f7fecfe5db24881e94fd285859857df84a)"
MD_UPSTREAM="$("${HOST_CONFIG[@]}" upstream MD_UPSTREAM "${MD_BASE}")"
BUILD_SCRATCH="$("${HOST_CONFIG[@]}" scratch "$(basename "${TASK}")" 50)/build"
WAN="${REPO}/scripts/miles_diffusion_wan"
QUAL="${TASK}/runtime/qualification.json"
# metric v2: the Judge's quality eval reads the held-out prompts of the frozen base values (public Hao7 test.txt, used by
# no training input) from its image; host_config.py resolves where the assets are
ASSETS_ENV="$("${HOST_CONFIG[@]}" shell MDWAN_ASSETS)"
eval "${ASSETS_ENV}"
metric_field() { python3 -c 'import json, sys; m = json.load(open(sys.argv[1]))["metric"]
for k in sys.argv[2:]: m = m[k]
print(m if isinstance(m, str) else json.dumps(m, indent=1))' "${QUAL}" "$@"; }
# The Work image's copy holds the scoring fields only (formula, scale, gate, frozen base values, eval protocol): the
# record's evidence (other runs' T and scores, the gate's calibration, rationale) is for reviewers, not the agent
public_metric() { python3 -c 'import json, sys; m = json.load(open(sys.argv[1]))["metric"]
out = {k: m[k] for k in ("version", "formula", "score_scale_seconds", "quality_repeats", "quality_unit",
                         "quality_gate_min", "base_quality_per_repeat", "base_quality")}
out["quality_protocol"] = {k: m["quality_protocol"][k] for k in ("eval_version", "prompt_set", "prompts_file",
                           "prompts_sha256", "rows", "namespace", "seed_rule", "num_inference_steps", "sampler", "scorer", "repeat")}
print(json.dumps(out, indent=1))' "${QUAL}"; }
PROMPTS_FILE="$(metric_field quality_protocol prompts_file)"
PROMPTS_SHA256="$(metric_field quality_protocol prompts_sha256)"
docker_() { if docker info >/dev/null 2>&1; then docker "$@"; else sg docker -c "$(printf '%q ' docker "$@")"; fi; }

[[ "$(docker_ image inspect --format '{{.Id}}' "${BASE_IMAGE}")" == "${BASE_ID_EXPECTED}" ]] \
  || { echo "base image ${BASE_IMAGE} is not ${BASE_ID_EXPECTED}" >&2; exit 1; }
mkdir -p "${BUILD_SCRATCH}"
STAGE="$(mktemp -d -p "${BUILD_SCRATCH}")"; trap 'rm -rf "${STAGE}"' EXIT
for role in work judge; do mkdir -p "${STAGE}/${role}/hook"; done
mkdir -p "${STAGE}/work"/{init,workdir,public}
git -C "${MD_UPSTREAM}" archive --format=tar "${MD_BASE}" | tar -x -C "${STAGE}/work/init"
find "${STAGE}/work/init" -type l -print -delete
cp -a "${TASK}/images/work/starter/." "${STAGE}/work/init/"          # task_entry.py stub, engine_overlay/README.md
cp -a "${TASK}/images/work/workdir/." "${STAGE}/work/workdir/"
cp "${TASK}/images/work/AGENT_TASK.md" "${TASK}/docs/INTERFACE.md" "${STAGE}/work/"
for role in work judge; do cp "${WAN}/audit/wan_audit_hook.py" "${WAN}/audit/wan_audit_hook.pth" "${STAGE}/${role}/hook/"; done
# checks shared by Judge and public runner: judge/*.py + contract/oracle helpers + the J0 legacy test, and metric.json
# (formula, frozen base quality, quality gate): the whole metric block of runtime/qualification.json for the Judge, its
# scoring fields for the public runner and the agent
for dst in "${STAGE}/work/public" "${STAGE}/judge/judge"; do
  mkdir -p "${dst}/judge" "${dst}/tests" "${dst}/audit"
  cp "${WAN}"/judge_v2/{run_judge.py,run_public.py,checks_probe.py,checks_replay.py,checks_run.py,checks_score.py,traj_delta.py,quality_eval.py} "${dst}/judge/"
  cp "${WAN}"/{wan_dance_contract.py,oracle_dance_fp32.py} "${dst}/"
  cp "${WAN}/tests/test_engine_overlay.py" "${dst}/tests/"
  cp "${WAN}/audit/wan_audit_hook.py" "${WAN}/audit/wan_audit_hook.pth" "${dst}/audit/"
done
metric_field > "${STAGE}/judge/judge/metric.json"
public_metric > "${STAGE}/work/public/metric.json"
# Judge only: the quality eval's HPSv3 scorer (a complete HPSv3 port: the agent writes its own, J2) and its prompts
cp "${WAN}/hpsv3_scorer.py" "${STAGE}/judge/judge/"
mkdir -p "$(dirname "${STAGE}/judge/judge/${PROMPTS_FILE}")"
cp "${MDWAN_ASSETS}/video_prompts/test.txt" "${STAGE}/judge/judge/${PROMPTS_FILE}"
[[ "$(sha256sum < "${STAGE}/judge/judge/${PROMPTS_FILE}" | cut -d' ' -f1)" == "${PROMPTS_SHA256}" ]] \
  || { echo "${MDWAN_ASSETS}/video_prompts/test.txt is not the eval prompt set ${PROMPTS_SHA256}" >&2; exit 1; }
# Work only: the agent's run_public.py wraps the public runner with disk hygiene (scripts/miles_common/public_cleanup.py:
# a suite's bulky outputs are removed when it ends unless --keep-artifacts; only the last two outputs stay complete)
cp "${REPO}/scripts/miles_common/public_cleanup.py" "${STAGE}/work/public/"
printf 'import sys\nsys.path.insert(0, "/opt/wan_public")\nfrom public_cleanup import run, suite_dir\nrun("/opt/wan_public/judge/run_public.py", "--out", default_out=suite_dir)\n' \
  > "${STAGE}/work/public/run_public.py"
cp "${TASK}/images/work/Dockerfile" "${STAGE}/work/Dockerfile"
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
