"""The local Judge-only path: scripts/eval_archive.sh, run_oracle.sh and lib.sh.

Agent runs happen on PyroMind; locally an archive is judged on 1-8 GPUs named
by JUDGE_GPUS.  These CPU-only tests drive the real scripts against fake
``nvidia-smi``/``docker`` executables and a fake SForge Python that records
what it was asked to run, so no GPU or Docker daemon is touched.
"""

from __future__ import annotations

import fcntl
import importlib.util
import json
import math
import os
import secrets
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from conftest import release_file


ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "scripts/lib.sh"
EVAL_ARCHIVE = ROOT / "scripts/eval_archive.sh"
RUN_ORACLE = ROOT / "scripts/run_oracle.sh"
TASK_ID = "quant_videogen_cross_model_kv_codec_h100_scratch_v4"
LONGCAT_CASES = ("lc-012", "lc-185", "lc-320", "lc-593", "lc-893", "lc-894")
HY_CASES = (*(f"hy-{n:02d}" for n in range(1, 11)), "hy-qvg")
TASK_FIXTURE = {"task_id": TASK_ID, "judge": {"cpu_limit": 32, "mem_limit": "448g"}}

FAKE_NVIDIA_SMI = r"""#!/usr/bin/env bash
set -euo pipefail
selector=""; query=""; apps=0
for argument in "$@"; do
  case "$argument" in
    --id=*) selector="${argument#--id=}" ;;
    --query-gpu=*) query="${argument#--query-gpu=}" ;;
    --query-compute-apps=*) apps=1 ;;
  esac
done
if (( apps )); then printf '%s' "${FAKE_COMPUTE_APPS:-}"; exit 0; fi
if [[ "$selector" == GPU-* ]]; then index=$((16#${selector##*-})); else index="$selector"; fi
uuid="$(printf 'GPU-%08x-%s-4000-8000-%012x' "$index" "$FAKE_GPU_SALT" "$index")"
name="NVIDIA H100 NVL"
[[ "$index" != "${FAKE_ODD_INDEX:-none}" ]] || name="NVIDIA H100 80GB HBM3"
case "$query" in
  index) echo "$index" ;;
  uuid) echo "$uuid" ;;
  name,compute_cap,memory.total,memory.used,utilization.gpu) echo "$name, 9.0, 95830, 0, 0" ;;
  name,memory.total,compute_cap) echo "$name, 95830, 9.0" ;;
  *) echo "unsupported fake query: $query" >&2; exit 2 ;;
esac
"""

FAKE_DOCKER = r"""#!/usr/bin/env bash
set -euo pipefail
case "$1" in
  info) exit 0 ;;
  image)
    name="${!#}"
    case "$name" in
      *.work.*) echo sha256:work ;;
      *.judge.*) echo sha256:judge ;;
      *) exit 1 ;;
    esac ;;
  *) echo "unexpected docker call: $*" >&2; exit 1 ;;
esac
"""

# Records the SForge invocation and, when asked, writes the report and the
# private artifact `sforge eval` would leave behind.
FAKE_SFORGE_PYTHON = r"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" >"$CAPTURE_DIR/argv"
{
  echo "QVG_JUDGE_GPU_UUIDS=${QVG_JUDGE_GPU_UUIDS-unset}"
  echo "SFORGE_JUDGE_EXTRA_ENV=${SFORGE_JUDGE_EXTRA_ENV-unset}"
  echo "FINAL_EVAL_TIMEOUT=${FINAL_EVAL_TIMEOUT-unset}"
} >"$CAPTURE_DIR/env"
cp -- "$(sed -n '/^--archive$/{n;p;}' "$CAPTURE_DIR/argv")" "$CAPTURE_DIR/archive.tar.gz"
if [[ -n "${FAKE_REPORT:-}" ]]; then
  run_id="$(sed -n '/^--run-id$/{n;p;}' "$CAPTURE_DIR/argv")"
  out="$SFORGE_LOG_DIR/runs/$run_id/quant_videogen_cross_model_kv_codec_h100_scratch_v4/submissions/manual-1"
  mkdir -p "$out"
  printf '%s' "$FAKE_REPORT" >"$out/report.json"
  if [[ -n "${FAKE_PRIVATE:-}" ]]; then
    printf '%s' "$FAKE_PRIVATE" >"$CAPTURE_DIR/qvg-private.json"
    tar -C "$CAPTURE_DIR" -czf "$out/private_judge_artifacts.tar.gz" qvg-private.json
  fi
fi
"""


def release_fixture(status: str = "calibrating", **judge: object) -> dict[str, object]:
    block: dict[str, object] = {
        "max_gpu_count": 8,
        "min_gpu_count": 1,
        "slot_uids": list(range(23001, 23009)),
        "master_port_base": 29600,
        "threads_per_slot": 4,
        "start_stagger_seconds": 20,
        "case_overhead_seconds": 120,
        "controller_overhead_seconds": 300,
        "min_device_memory_bytes": 99_000_000_000,
    }
    block.update(judge)
    return {
        "status": status,
        "judge": block,
        "models": {
            "longcat": {"stage_seconds": 3000, "fidelity_metric_seconds": 600, "vbench_metric_seconds": 1200},
            "hy_worldplay_8b": {"stage_seconds": 900, "fidelity_metric_seconds": 600, "vbench_metric_seconds": 900},
        },
        "cases": [{"case_id": case, "model": "longcat"} for case in LONGCAT_CASES]
        + [{"case_id": case, "model": "hy_worldplay_8b"} for case in HY_CASES],
    }


def _bash(command: str, *, env: dict[str, str] | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", command],
        text=True,
        capture_output=True,
        check=False,
        cwd=cwd,
        env={**os.environ, "REPOSITORY_ROOT": str(ROOT.parents[2]), **(env or {})},
    )


def _judge_plan(tmp_path: Path, count: int, release: dict[str, object] | None = None) -> subprocess.CompletedProcess[str]:
    release_path = tmp_path / "release.json"
    release_path.write_text(json.dumps(release or release_fixture()), encoding="utf-8")
    tasks = tmp_path / "tasks"
    tasks.mkdir(exist_ok=True)
    (tasks / f"{TASK_ID}.json").write_text(json.dumps(TASK_FIXTURE), encoding="utf-8")
    return _bash(
        f'source "{LIB}"; RELEASE_PATH="{release_path}"; TASKS_DIR="{tasks}"; judge_plan {count}'
    )


# --------------------------------------------------------------------------
# lib.sh helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        # 6 LongCat units of 3000+1200+120 and 11 HY units of 900+900+120: on
        # eight GPUs the heaviest-first plan's longest GPU carries 6240 s, plus
        # the last GPU's 140 s start offset and the 300 s controller overhead.
        (8, "32 448g 6680 6980"),
        # One GPU runs every case in turn: 6*4320 + 11*1920 + 300.
        (1, "4 56g 47340 47640"),
    ],
)
def test_judge_plan_sizes_the_judge_per_gpu_and_never_undercuts_the_controller(
    tmp_path: Path, count: int, expected: str
) -> None:
    result = _judge_plan(tmp_path, count)
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == expected.split()


def test_judge_plan_bound_covers_every_list_schedule_of_the_full_budgets(tmp_path: Path) -> None:
    release = release_fixture()
    units = [4320] * 6 + [1920] * 11
    for count in range(1, 9):
        result = _judge_plan(tmp_path, count, release)
        assert result.returncode == 0, result.stderr
        controller = int(result.stdout.split()[2])
        # Greedy in suite order and heaviest-first, GPU i starting 20 s * i late.
        for order in (units, sorted(units, reverse=True)):
            free = [index * 20 for index in range(count)]
            for unit in order:
                slot = min(range(count), key=lambda index: (free[index], index))
                free[slot] += unit
            assert max(free) + 300 <= controller


def test_judge_plan_never_undercuts_the_controllers_own_bound(tmp_path: Path) -> None:
    release = json.loads(release_file().read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("qvg_v4_run_judge_bound", ROOT / "judge/run_judge.py")
    assert spec is not None and spec.loader is not None
    run_judge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_judge)
    controller = run_judge._controller_module()
    for count in range(release["judge"]["min_gpu_count"], release["judge"]["max_gpu_count"] + 1):
        result = _judge_plan(tmp_path, count, release)
        assert result.returncode == 0, result.stderr
        _, _, bound, timeout = result.stdout.split()
        expected = controller.controller_budget_seconds(release, count)
        assert int(bound) >= expected
        assert int(timeout) > int(bound)


@pytest.mark.parametrize("count", [0, 9])
def test_judge_plan_refuses_a_gpu_count_outside_the_release(tmp_path: Path, count: int) -> None:
    result = _judge_plan(tmp_path, count)
    assert result.returncode != 0
    assert "Judge GPU" in result.stderr


def test_judge_plan_fails_closed_without_a_judge_block(tmp_path: Path) -> None:
    release = release_fixture()
    del release["judge"]
    result = _judge_plan(tmp_path, 8, release)
    assert result.returncode != 0
    assert "no judge block" in result.stderr


def test_launcher_floor_matches_the_release_device_memory_floor() -> None:
    release = json.loads(release_file().read_text(encoding="utf-8"))
    result = _bash(f'source "{LIB}"; printf "%s" "$GPU_MIN_TOTAL_MIB"')
    assert result.returncode == 0, result.stderr
    assert int(result.stdout) == math.ceil(release["judge"]["min_device_memory_bytes"] / 2**20)


def _fake_bin(tmp_path: Path, salt: str) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in (("nvidia-smi", FAKE_NVIDIA_SMI), ("docker", FAKE_DOCKER)):
        path = bin_dir / name
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)
    return {"PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_GPU_SALT": salt}


def _fake_uuid(index: int, salt: str) -> str:
    return f"GPU-{index:08x}-{salt}-4000-8000-{index:012x}"


def test_judge_gpu_uuids_resolves_indices_and_uuids_in_order(tmp_path: Path) -> None:
    salt = secrets.token_hex(2)
    env = _fake_bin(tmp_path, salt)
    result = _bash(
        f'source "{LIB}"; judge_gpu_uuids "3, 1,{_fake_uuid(5, salt)}"', env=env
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [_fake_uuid(3, salt), _fake_uuid(1, salt), _fake_uuid(5, salt)]


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        ("", "JUDGE_GPUS must list"),
        ("0,0", "twice"),
        ("0,1,2,3,4,5,6,7,8", "1 to 8"),
        ("zero", "GPU selector"),
    ],
)
def test_judge_gpu_uuids_refuses_bad_selections(
    tmp_path: Path, selection: str, message: str
) -> None:
    env = _fake_bin(tmp_path, secrets.token_hex(2))
    env["JUDGE_GPUS"] = ""
    result = _bash(f'source "{LIB}"; out="$(judge_gpu_uuids "{selection}")"; echo reached', env=env)
    assert result.returncode != 0
    assert message in result.stderr
    assert "reached" not in result.stdout


def test_lock_gpu_holds_both_host_lock_names(tmp_path: Path) -> None:
    uuid = _fake_uuid(1, "abcd")
    result = _bash(
        f'source "{LIB}"; GPU_LOCK_DIR="{tmp_path}"; lock_gpu {uuid} judge; '
        f'for name in sforge mlsysforge; do '
        f'  ( exec 3>>"{tmp_path}/$name-gpu-{uuid}.lock"; flock -n 3 ) && echo "$name free" || echo "$name held"; '
        f'done'
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split("\n")[:2] == ["sforge held", "mlsysforge held"]
    assert (tmp_path / f"sforge-gpu-{uuid}.lock").is_file()
    assert (tmp_path / f"mlsysforge-gpu-{uuid}.lock").is_file()


def test_lock_gpu_refuses_a_lock_another_process_holds(tmp_path: Path) -> None:
    uuid = _fake_uuid(2, "abcd")
    held = tmp_path / f"mlsysforge-gpu-{uuid}.lock"
    with held.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = _bash(f'source "{LIB}"; GPU_LOCK_DIR="{tmp_path}"; lock_gpu {uuid} judge; echo reached')
    assert result.returncode != 0
    assert "held by another process" in result.stderr
    assert "reached" not in result.stdout


def test_lock_gpu_refuses_a_symlink_without_touching_its_target(tmp_path: Path) -> None:
    uuid = _fake_uuid(3, "abcd")
    target = tmp_path / "target"
    target.write_text("keep", encoding="utf-8")
    (tmp_path / f"sforge-gpu-{uuid}.lock").symlink_to(target)
    result = _bash(f'source "{LIB}"; GPU_LOCK_DIR="{tmp_path}"; lock_gpu {uuid} judge; echo reached')
    assert result.returncode != 0
    assert "symlink" in result.stderr
    assert target.read_text(encoding="utf-8") == "keep"


def test_lock_directory_is_not_taken_from_the_environment(tmp_path: Path) -> None:
    result = _bash(f'source "{LIB}"; gpu_lock_path GPU-1234', env={"GPU_LOCK_DIR": str(tmp_path)})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "/tmp/sforge-gpu-GPU-1234.lock"


def test_idle_and_compute_process_checks(tmp_path: Path) -> None:
    salt = secrets.token_hex(2)
    env = _fake_bin(tmp_path, salt)
    uuid = _fake_uuid(4, salt)
    idle = _bash(f'source "{LIB}"; assert_gpu_idle {uuid} judge; assert_no_compute_process {uuid} judge', env=env)
    assert idle.returncode == 0, idle.stderr
    assert uuid in idle.stdout and "NVIDIA H100 NVL" in idle.stdout
    busy = _bash(
        f'source "{LIB}"; assert_no_compute_process {uuid} judge',
        env={**env, "FAKE_COMPUTE_APPS": f"{_fake_uuid(1, salt)}\n{uuid}\n"},
    )
    assert busy.returncode != 0 and "already has a compute process" in busy.stderr
    mixed = _bash(
        f'source "{LIB}"; assert_identical_gpus judge {_fake_uuid(1, salt)} {_fake_uuid(2, salt)}',
        env={**env, "FAKE_ODD_INDEX": "2"},
    )
    assert mixed.returncode != 0 and "not identical" in mixed.stderr


def test_image_source_digest_tracks_baked_sources_only(tmp_path: Path) -> None:
    root = tmp_path / "task"
    for directory in ("assets", "configs", "images", "judge/private", "metrics", "policy", "runtime", "upstream"):
        (root / directory).mkdir(parents=True)
    (root / ".dockerignore").write_text("*\n", encoding="utf-8")
    (root / "judge/evaluate_hidden.py").write_text("one\n", encoding="utf-8")

    def digest() -> str:
        result = _bash(f'source "{LIB}"; TASK_ROOT="{root}"; image_source_digest')
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    first = digest()
    (root / "judge/private/prompts.txt").write_text("mounted, never baked\n", encoding="utf-8")
    (root / "judge/__pycache__").mkdir()
    (root / "judge/__pycache__/x.pyc").write_bytes(b"\0")
    (root / "images/published.json").write_text("{}\n", encoding="utf-8")
    assert digest() == first
    (root / "configs/release.json").write_text("{}\n", encoding="utf-8")
    assert digest() != first


# --------------------------------------------------------------------------
# eval_archive.sh and run_oracle.sh end to end, against fakes
# --------------------------------------------------------------------------


class Harness:
    """A copy of the task scripts with fake GPU, Docker and SForge executables."""

    def __init__(self, tmp_path: Path, release: dict[str, object]) -> None:
        self.salt = secrets.token_hex(2)
        self.root = tmp_path / "task"
        (self.root / "scripts").mkdir(parents=True)
        for name in ("lib.sh", "eval_archive.sh", "run_oracle.sh", "sforge_runtime_entry.py"):
            shutil.copy2(ROOT / "scripts" / name, self.root / "scripts" / name)
        for directory in ("assets", "configs", "images", "judge", "metrics", "policy", "runtime", "upstream", "tasks", "oracle/codec"):
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        (self.root / ".dockerignore").write_text("*\n", encoding="utf-8")
        (self.root / "oracle/mlsysforge_codec.py").write_text("ORACLE = True\n", encoding="utf-8")
        (self.root / "oracle/codec/__init__.py").write_text("", encoding="utf-8")
        self.write_release(release)
        (self.root / f"tasks/{TASK_ID}.json").write_text(json.dumps(TASK_FIXTURE), encoding="utf-8")
        self.capture = tmp_path / "capture"
        self.capture.mkdir()
        self.logs = tmp_path / "logs"
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        sforge_python = tmp_path / "sforge-python"
        sforge_python.write_text(FAKE_SFORGE_PYTHON, encoding="utf-8")
        sforge_python.chmod(0o755)
        self.env = {
            **_fake_bin(tmp_path, self.salt),
            "LOGS_DIR": str(self.logs),
            "QVG_ARTIFACT_ROOT": str(artifacts),
            "SFORGE_BIN": str(sforge_python),
            "SFORGE_PYTHON": str(sforge_python),
            "CAPTURE_DIR": str(self.capture),
            "SFORGE_JUDGE_EXTRA_ENV": "CUDA_VISIBLE_DEVICES=0",
        }
        for name in ("JUDGE_CPU_LIMIT", "JUDGE_MEM_LIMIT", "FINAL_EVAL_TIMEOUT", "ORACLE_TIMEOUT", "RUN_ID", "RUN_STAMP"):
            self.env[name] = ""
        self.archive = tmp_path / "submission.tar.gz"
        self.archive.write_bytes(b"archive")
        self.record_build()

    def write_release(self, release: dict[str, object]) -> None:
        (self.root / "configs/h100_scratch_v4_release.json").write_text(json.dumps(release), encoding="utf-8")

    def write_quality(self, quality: dict[str, object]) -> None:
        (self.root / "configs/h100_scratch_v4_quality.json").write_text(json.dumps(quality), encoding="utf-8")

    def record_build(self) -> None:
        digest = _bash(f'source "{self.root}/scripts/lib.sh"; image_source_digest', env=self.env)
        assert digest.returncode == 0, digest.stderr
        manifest = self.logs / "build/image-manifest.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps({
            "source_digest": digest.stdout.strip(),
            "work": {"id": "sha256:work"},
            "judge": {"id": "sha256:judge"},
        }), encoding="utf-8")

    def uuid(self, index: int) -> str:
        return _fake_uuid(index, self.salt)

    def run(self, script: str, *arguments: str, **env: str) -> subprocess.CompletedProcess[str]:
        environment = {**os.environ, "REPOSITORY_ROOT": str(ROOT.parents[2]), **self.env, **env}
        environment = {key: value for key, value in environment.items() if value != ""}
        return subprocess.run(
            ["bash", str(self.root / "scripts" / script), *arguments],
            text=True, capture_output=True, check=False, env=environment,
        )

    def plan(self, count: int) -> list[str]:
        result = _bash(f'source "{self.root}/scripts/lib.sh"; judge_plan {count}', env=self.env)
        assert result.returncode == 0, result.stderr
        return result.stdout.split()

    def argv(self) -> list[str]:
        return (self.capture / "argv").read_text(encoding="utf-8").splitlines()

    def child_env(self) -> dict[str, str]:
        lines = (self.capture / "env").read_text(encoding="utf-8").splitlines()
        return dict(line.split("=", 1) for line in lines)

    def cleanup_locks(self) -> None:
        for index in range(16):
            for prefix in ("sforge", "mlsysforge"):
                path = Path(f"/tmp/{prefix}-gpu-{self.uuid(index)}.lock")
                if path.is_file() and not path.is_symlink():
                    path.unlink()


@pytest.fixture
def harness(tmp_path: Path):
    instance = Harness(tmp_path, release_fixture())
    try:
        yield instance
    finally:
        instance.cleanup_locks()


def test_eval_archive_hands_every_judge_gpu_and_the_derived_plan_to_sforge(harness: Harness) -> None:
    result = harness.run("eval_archive.sh", str(harness.archive), "qvg-eval-test", JUDGE_GPUS="2,0")
    assert result.returncode == 0, result.stderr
    cpu, memory, controller, timeout = harness.plan(2)
    assert harness.argv() == [
        str(harness.root / "scripts/sforge_runtime_entry.py"), "eval",
        "--backend", "docker", "--task", TASK_ID,
        "--archive", str(harness.archive), "--run-id", "qvg-eval-test",
        "--timeout", timeout, "--judge-cpu-limit", cpu, "--judge-mem-limit", memory,
        "--json",
    ]
    assert int(timeout) > int(controller)
    child = harness.child_env()
    assert child["QVG_JUDGE_GPU_UUIDS"] == f"{harness.uuid(2)} {harness.uuid(0)}"
    # No CUDA_VISIBLE_DEVICES mask (or anything else) reaches the Judge.
    assert child["SFORGE_JUDGE_EXTRA_ENV"] == "unset"
    assert "GPU release barrier: PASS" in result.stdout


def test_eval_archive_forwards_only_the_explicit_review_skip_and_overrides(harness: Harness) -> None:
    result = harness.run(
        "eval_archive.sh", "--skip-integrity-review", str(harness.archive), "qvg-eval-skip",
        JUDGE_GPUS="1", JUDGE_CPU_LIMIT="6", JUDGE_MEM_LIMIT="80g", FINAL_EVAL_TIMEOUT="99999",
    )
    assert result.returncode == 0, result.stderr
    argv = harness.argv()
    assert argv[-1] == "--skip-integrity-review"
    assert argv[argv.index("--timeout") + 1] == "99999"
    assert argv[argv.index("--judge-cpu-limit") + 1] == "6"
    assert argv[argv.index("--judge-mem-limit") + 1] == "80g"


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"JUDGE_GPUS": "0,0"}, "twice"),
        ({"JUDGE_GPUS": "0,1,2,3,4,5,6,7,8"}, "1 to 8"),
        ({"JUDGE_GPUS": ""}, "JUDGE_GPUS must list"),
        ({"JUDGE_GPUS": "0,1", "FAKE_ODD_INDEX": "1"}, "not identical"),
        ({"JUDGE_GPUS": "0", "FINAL_EVAL_TIMEOUT": "60"}, "must exceed the controller bound"),
    ],
)
def test_eval_archive_refuses_before_sforge_runs(
    harness: Harness, env: dict[str, str], message: str
) -> None:
    result = harness.run("eval_archive.sh", str(harness.archive), "qvg-eval-refused", **env)
    assert result.returncode != 0
    assert message in result.stderr
    assert not (harness.capture / "argv").exists()


def test_eval_archive_refuses_a_busy_or_locked_gpu(harness: Harness) -> None:
    busy = harness.run(
        "eval_archive.sh", str(harness.archive), "qvg-eval-busy",
        JUDGE_GPUS="0,1", FAKE_COMPUTE_APPS=f"{harness.uuid(1)}\n",
    )
    assert busy.returncode != 0 and "already has a compute process" in busy.stderr
    lock = Path(f"/tmp/mlsysforge-gpu-{harness.uuid(3)}.lock")
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = harness.run("eval_archive.sh", str(harness.archive), "qvg-eval-locked", JUDGE_GPUS="3")
    assert locked.returncode != 0 and "held by another process" in locked.stderr
    assert not (harness.capture / "argv").exists()


def test_eval_archive_refuses_images_built_from_older_sources(harness: Harness) -> None:
    (harness.root / "judge/evaluate_hidden.py").write_text("changed\n", encoding="utf-8")
    result = harness.run("eval_archive.sh", str(harness.archive), "qvg-eval-stale", JUDGE_GPUS="0")
    assert result.returncode != 0
    assert "image-driving source changed" in result.stderr
    assert not (harness.capture / "argv").exists()


def test_eval_archive_rejects_bad_arguments_before_any_runtime(harness: Harness) -> None:
    for arguments in ((), (str(harness.archive), "bad id"), ("/nonexistent.tar.gz",)):
        result = harness.run("eval_archive.sh", *arguments, JUDGE_GPUS="0")
        assert result.returncode != 0
        assert "usage" in result.stderr or "unsafe RUN_ID" in result.stderr


def _report(*, valid: bool, score: float, status: str, timed_out: bool = False) -> str:
    """The public report run_judge.py's payload becomes (``status`` as it names it)."""
    return json.dumps({
        "valid": valid,
        "score": score,
        "timed_out": timed_out,
        "metrics": {
            "candidate_invalid": status == "candidate_error",
            # A calibration run is withheld as an infrastructure-classified non-verdict.
            "infrastructure_invalid": status in {"infrastructure_error", "uncalibrated"},
            "evaluation_status": status,
        },
    })


UNCALIBRATED_PRIVATE = json.dumps(
    {"outcome": "evaluated", "reason": "uncalibrated_release", "valid": False}
)


def test_run_oracle_judges_the_trusted_archive_without_review(harness: Harness) -> None:
    result = harness.run(
        "run_oracle.sh", JUDGE_GPUS="0,1", RUN_ID="qvg-oracle-calibrating",
        FAKE_REPORT=_report(valid=False, score=0.0, status="uncalibrated"),
        FAKE_PRIVATE=UNCALIBRATED_PRIVATE,
    )
    assert result.returncode == 0, result.stderr
    assert "calibration run complete" in result.stdout
    argv = harness.argv()
    assert argv[-1] == "--skip-integrity-review"
    assert Path(argv[argv.index("--archive") + 1]).name == "oracle.tar.gz"
    with tarfile.open(harness.capture / "archive.tar.gz") as archive:
        assert "submission/mlsysforge_codec.py" in archive.getnames()


@pytest.mark.parametrize(
    ("report", "private"),
    [
        # A calibration run must reach the uncalibrated verdict, i.e. run every case.
        (_report(valid=False, score=0.0, status="candidate_error"), json.dumps({"outcome": "candidate_failure"})),
        (_report(valid=False, score=0.0, status="infrastructure_error"), json.dumps({"outcome": "infrastructure_failure"})),
        (_report(valid=False, score=0.0, status="uncalibrated", timed_out=True), UNCALIBRATED_PRIVATE),
        (_report(valid=False, score=0.0, status="uncalibrated"), json.dumps({"outcome": "evaluated"})),
    ],
)
def test_run_oracle_calibration_run_fails_unless_it_completes(
    harness: Harness, report: str, private: str
) -> None:
    result = harness.run(
        "run_oracle.sh", JUDGE_GPUS="0", RUN_ID=f"qvg-oracle-{secrets.token_hex(3)}",
        FAKE_REPORT=report, FAKE_PRIVATE=private,
    )
    assert result.returncode != 0


@pytest.mark.parametrize(
    ("band", "score", "passes"),
    [
        (None, 7.5, True),
        ([7.0, 8.0], 7.5, True),
        ([7.0, 8.0], 9.0, False),
    ],
)
def test_run_oracle_on_a_calibrated_release_requires_validity_and_the_band(
    harness: Harness, band: list[float] | None, score: float, passes: bool
) -> None:
    harness.write_release(release_fixture(status="calibrated"))
    if band is not None:
        # The band is per device class, in the quality file beside the release.
        harness.write_quality({"schema_version": 1, "hardware": "h100-94gb-400w", "models": {},
                               "oracle_calibration": {"score_band": band}})
    harness.record_build()
    result = harness.run(
        "run_oracle.sh", JUDGE_GPUS="0", RUN_ID=f"qvg-oracle-{secrets.token_hex(3)}",
        FAKE_REPORT=_report(valid=True, score=score, status="completed"), INFRATHON_HARDWARE="h100-94gb-400w",
    )
    assert (result.returncode == 0) is passes, result.stderr
    if passes:
        assert "Trusted oracle PASS" in result.stdout


def test_run_oracle_rejects_an_invalid_oracle_on_a_calibrated_release(harness: Harness) -> None:
    harness.write_release(release_fixture(status="calibrated"))
    harness.record_build()
    result = harness.run(
        "run_oracle.sh", JUDGE_GPUS="0", RUN_ID="qvg-oracle-invalid",
        FAKE_REPORT=_report(valid=False, score=0.0, status="candidate_error"),
    )
    assert result.returncode != 0
    assert "not valid on the calibrated release" in result.stderr


def test_judge_only_scripts_never_mask_the_judge_gpus() -> None:
    for name in ("eval_archive.sh", "run_oracle.sh", "replay_candidates.sh", "lib.sh"):
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "CUDA_VISIBLE_DEVICES=0" not in source, name
        assert "QVG_JUDGE_GPU=" not in source, name
    oracle = RUN_ORACLE.read_text(encoding="utf-8")
    assert "eval_archive.sh" in oracle and "--skip-integrity-review" in oracle
    assert "7.64" not in oracle and "7.68" not in oracle
    archive = EVAL_ARCHIVE.read_text(encoding="utf-8")
    lock = archive.index('lock_gpu "${uuid}" judge')
    assert lock < archive.index('assert_gpu_idle "${uuid}" judge') < archive.index('"${SFORGE_PYTHON}" "${RUNTIME_ENTRY}" eval')


@pytest.mark.parametrize("name", ["lib.sh", "eval_archive.sh", "run_oracle.sh", "replay_candidates.sh", "status.sh", "build_images.sh", "network_preflight.sh", "verify_models.sh"])
def test_task_scripts_pass_shell_syntax(name: str) -> None:
    result = subprocess.run(["bash", "-n", str(ROOT / "scripts" / name)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
