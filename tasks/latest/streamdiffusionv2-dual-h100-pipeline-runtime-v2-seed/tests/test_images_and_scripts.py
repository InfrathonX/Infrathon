from __future__ import annotations
import re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def prefix(path):
    lines=path.read_text().splitlines(keepends=True)
    return "".join(lines[:next(i for i,x in enumerate(lines) if x.startswith("# ---- divergence marker:"))])
def test_docker_prefix_and_pins():
    work=ROOT/"images/work/Dockerfile"; judge=ROOT/"images/judge/Dockerfile"
    assert prefix(work)==prefix(judge)
    shared=prefix(work)
    assert "@sha256:0cf3402e946b7c384ba943ee05c90b4c5a4a05227923921f2b0918c011cfaf56" in shared
    assert "2.7.4.post1" in shared and "flash_attn_interface" in shared
    assert "sha256sum --check --strict" in shared
def test_private_assets_are_not_in_work_image():
    work=(ROOT/"images/work/Dockerfile").read_text(); judge=(ROOT/"images/judge/Dockerfile").read_text()
    assert "embeddings/hidden" not in work and "final_cases" not in work and "/opt/sdv2_judge/private" not in work
    # The public Judge holds no private input; the host-built holdout layer adds them root-only.
    holdout=(ROOT/"images/judge/Dockerfile.private").read_text()
    assert "embeddings/hidden" not in judge and "final_cases" not in judge and "/opt/sdv2_judge/private" not in judge
    assert "FROM ${JUDGE_PUBLIC_IMAGE}" in holdout
    assert "embeddings/hidden" in holdout and "final_cases" in holdout and "chmod 0500" in holdout

def test_task_runtime_and_goldens_are_image_contexts():
    work=(ROOT/"images/work/Dockerfile").read_text(); judge=(ROOT/"images/judge/Dockerfile").read_text()
    assert "COPY --from=task_runtime . /workspace/benchmark/runtime/" in work
    assert "COPY --from=task_runtime . /opt/sdv2_judge/worker_runtime/" in judge
    holdout=(ROOT/"images/judge/Dockerfile.private").read_text()
    assert "goldens" not in work and "goldens_judge" in holdout and "goldens_judge" not in judge

def test_pristine_seed_ships_in_both_images_for_the_reference_pass():
    work=(ROOT/"images/work/Dockerfile").read_text(); judge=(ROOT/"images/judge/Dockerfile").read_text()
    build=(ROOT/"scripts/build_images.sh").read_text()
    assert "COPY --from=seed_runtime runtime_v001/ /opt/sdv2_judge/seed_runtime/runtime_v001/" in judge
    assert "COPY --from=seed_runtime runtime_v001/ /workspace/benchmark/seed_runtime/runtime_v001/" in work
    assert build.count('--build-context "seed_runtime=${MIGRATION_ROOT}/oracle/candidates"')==2

def test_controller_budget_covers_two_measured_passes():
    import json
    judge=(ROOT/"images/judge/Dockerfile").read_text()
    spec=json.loads((ROOT/"tasks/streamdiffusionv2_dual_h100_pipeline_runtime_seed_v2.json").read_text())
    worker=int(re.search(r"SDV2_WORKER_TIMEOUT=(\d+)",judge).group(1))
    total=int(re.search(r"SDV2_TOTAL_TIMEOUT=(\d+)",judge).group(1))
    assert total>=2*worker and spec["judge"]["eval_timeout"]==total
    assert f'"{total}"' in (ROOT/"judge/run_judge.py").read_text()
def test_lib_exports_repository_contract():
    text=(ROOT/"scripts/lib.sh").read_text()
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-${REPOSITORY_ROOT}/outputs}"' in text
    assert 'LOGS_DIR="${LOGS_DIR:-${OUTPUT_ROOT}/${OUTPUT_SUBDIR}}"' in text
    assert "export REPOSITORY_ROOT MIGRATION_ROOT LOGS_DIR" in text


def test_work_image_ships_curl_for_the_harness_wrappers():
    """sforge-submit and sforge-save-version are curl-based host wrappers.

    The Work image built from the pytorch devel base has no curl; on the
    2026-09-11/12 formal runs the agent could only submit by writing its own
    python `curl` shim. The shared apt block must install curl in both images
    (the layout validator requires the Dockerfile prefixes to stay identical).
    """
    for name in ("images/work/Dockerfile", "images/judge/Dockerfile"):
        text = (ROOT / name).read_text()
        apt = text[text.index("apt-get install"):text.index("rm -rf /var/lib/apt/lists/*")]
        assert " curl " in apt.replace("\\\n", " ").replace("\n", " ") + " ", name
