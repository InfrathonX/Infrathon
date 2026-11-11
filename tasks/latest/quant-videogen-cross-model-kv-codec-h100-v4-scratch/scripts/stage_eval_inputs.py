#!/usr/bin/env python3
"""Produce the Judge-only hidden tree of the v4 release, plus the public references.

Everything under ``$QVG_ARTIFACT_ROOT/staging`` is public and pinned
(``portable/REBUILD_PROMPT.md`` §2, ``scripts/verify_models.sh``).  The release
(``configs/h100_scratch_v4_release.json``) names 17 cases whose inputs and
goldens live in one Judge-only tree, mounted read-only at ``/eval/hidden``.
This script derives that tree (default ``<root>/hidden``) from the staging
area, following the release's own paths:

* ``prompts/MovieGenVideoBench_extended.txt`` — copied from the in-tree
  Judge-only path ``judge/private/prompts/`` (1003 Movie Gen Video Bench
  prompts; the release names records 12, 185, 320, 593, 893 and 894).
* ``hy/<nn|qvg>/image.<png|jpeg>`` — copied from the pinned official
  HY-WorldPlay checkout (``staging/hy-worldplay-official``, the release's
  ``image_source``).
* ``longcat/<idx3>/init.mp4`` — generated with ``run_longcat.py --mode bf16
  --make-init`` for every case whose release ``init_video_sha256`` is null.  A
  release-bound init (lc-012 carries v3's) is never regenerated: supply it
  with ``--reuse longcat/012/init.mp4=<v3 root>/hidden/goldens/longcat/init.mp4``.
* ``longcat/<idx3>/bf16-10seg.frames.pt`` — generated: ``--mode bf16`` continued
  from the case init for the case's ``segments`` with
  ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` (without it the
  10-segment BF16 run fragments and OOMs at segment 2).
* ``hy/<nn|qvg>/bf16-285.frames.pt`` — generated: ``--mode bf16`` with the
  case image, prompt, seed, ``video_length`` and ``pose``.  hy-qvg is the v3
  case, and the release pins its golden to v3's a8ee4657…: a regeneration
  with the v4 runner must reproduce it bit for bit (else it is refused, as
  for any release-bound file), or the v3 file is carried over with
  ``--reuse hy/qvg/bf16-285.frames.pt=<v3 golden>``.
* ``MANIFEST.json`` — every produced file with its SHA-256, the release pin it
  was checked against, the device it came from and the BF16 run profile, plus
  ``release_values``: the digests to write into a calibrating release.

``public-eval/self-forcing/{bf16,qvg-reference}`` (the public Self-Forcing
diagnostics' references) are generated as in v3 by the Work image's own
``run_self_forcing_paper.py``.

Generation runs the Judge's own runner adapters (the checked-out
``judge/run_*.py`` and ``upstream/experiments``, bind-mounted over the copies
in the Judge image) with the Judge's read-only model mounts, one container per
job and one GPU per container: each container sees exactly one device, so the
runners get ``--device-index 0`` and a unique ``--master-port``.  Jobs are
spread over ``--gpus`` (heaviest first: the LongCat cases, then HY).  Every
GPU is locked for the whole run through both ``/tmp/sforge-gpu-<uuid>.lock``
and ``/tmp/mlsysforge-gpu-<uuid>.lock`` and must be idle at start.

Release-bound files (a non-null SHA-256 in the release) are never overwritten:
an existing one must match its pin, and a regenerated one that differs is left
in the scratch directory -- this device is a new measurement profile.  Only
``--adopt`` installs such a file (recorded with ``matches_release: false``);
the Judge then refuses it until the release is deliberately re-bound from
``MANIFEST.json``.  Unpinned files that already exist are reused, so an
interrupted run resumes; delete one to regenerate it.  ``--dry-run`` prints
the plan without touching Docker or a GPU.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TASK_ROOT = Path(__file__).resolve().parents[1]
def release_path(task_root: Path) -> Path:
    """The Judge-only release config: the in-repository copy when present, else the
    asset-root copy ``<INFRATHON_ASSETS_ROOT>/<task directory>/judge/h100_scratch_v4_release.json``
    written by ``assets/fetch.sh`` (default root ``~/.cache/infrathon/assets``)."""
    local = task_root / "configs/h100_scratch_v4_release.json"
    if local.is_file():
        return local
    assets = os.environ.get("INFRATHON_ASSETS_ROOT") or str(
        Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "infrathon" / "assets")
    return Path(assets).expanduser() / task_root.name / "judge/h100_scratch_v4_release.json"


RELEASE = release_path(TASK_ROOT)
IN_TREE_PROMPTS = TASK_ROOT / "judge/private/prompts/MovieGenVideoBench_extended.txt"
PROMPT_NAME = "MovieGenVideoBench_extended.txt"
BENCHMARK_NAME = "quant_videogen_h100_scratch_v4"
TASK_ID = "quant_videogen_cross_model_kv_codec_h100_scratch_v4"
HIDDEN_MOUNT = "/eval/hidden"
HY_OFFICIAL = "staging/hy-worldplay-official"
HY_SOURCE_PREFIX = "HY-WorldPlay "
LOCK_PREFIXES = ("sforge", "mlsysforge")
NVIDIA_SMI_TIMEOUT = 60
IDLE_MEMORY_MIB = 2048
# Ordering estimates only (measured on H100 NVL, 2026-10-06); nothing waits on them.
JOB_COST_SECONDS = {"longcat-init": 300, "longcat-golden": 2300, "hy-golden": 500}


class StageError(RuntimeError):
    """A job or a release binding failed; the message says what to do."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def load_release() -> dict[str, Any]:
    release = json.loads(RELEASE.read_text(encoding="utf-8"))
    if release.get("protocol_schema_version") != 2:
        raise SystemExit("the v4 staging recipe needs a protocol_schema_version 2 release")
    if release.get("prompt_file") != f"{HIDDEN_MOUNT}/prompts/{PROMPT_NAME}":
        raise SystemExit("release names an unexpected prompt file")
    return release


def hidden_relative(guest: str) -> str:
    """Map a release path under /eval/hidden to its path inside the hidden tree."""
    prefix = HIDDEN_MOUNT + "/"
    if not guest.startswith(prefix) or ".." in Path(guest).parts:
        raise SystemExit(f"release path is outside {HIDDEN_MOUNT}: {guest}")
    return guest[len(prefix):]


# --------------------------------------------------------------------------
# Plan (CPU only)
# --------------------------------------------------------------------------


@dataclass
class Output:
    """One file a container run produces and where it is installed."""

    produced: str  # path relative to the job's scratch directory
    target: str  # path relative to the hidden tree
    pin: str | None  # the release SHA-256, or None while unpinned
    field: str  # release field the digest belongs to
    profile: str | None = None  # BF16 run profile beside the output, if any


@dataclass
class ContainerRun:
    step: str  # longcat-init | longcat-golden | hy-golden
    command: list[str]
    env: dict[str, str]
    master_port: int
    output: Output


@dataclass
class Job:
    case_id: str
    model: str
    scratch: str  # job directory under the scratch root; unique per invocation
    runs: list[ContainerRun] = field(default_factory=list)

    @property
    def cost(self) -> int:
        return sum(JOB_COST_SECONDS[run.step] for run in self.runs)


def _thread_env(release: dict[str, Any]) -> dict[str, str]:
    threads = str(int(release["judge"]["threads_per_slot"]))
    return {"OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


def plan_jobs(release: dict[str, Any], hidden: Path, *, steps: set[str], cases: set[str] | None,
              reused: frozenset[str] = frozenset(), token: str = "plan") -> list[Job]:
    """Container runs still needed, heaviest job first.  Never touches a GPU.

    Container outputs land in a directory unique to this invocation (``token``):
    a failed run can never leave an earlier run's file where this one expects
    its own.
    """

    def present(relative: str) -> bool:
        return relative in reused or (hidden / relative).exists()

    port = int(release["judge"]["master_port_base"])
    jobs: list[Job] = []
    prompt_file = release["prompt_file"]
    for case in release["cases"]:
        if cases is not None and case["case_id"] not in cases:
            continue
        job = Job(case["case_id"], case["model"], f"{case['case_id']}/{token}")
        out = f"/out/{job.scratch}"
        seed = str(case["seed"])
        common = ["--seed", seed]
        if case["model"] == "longcat":
            index = str(case["prompt_index"])
            common = ["--prompt-file", prompt_file, "--prompt-index", index, "--seed", seed]
            env = _thread_env(release) | {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
            init = hidden_relative(case["init_video"])
            if "inits" in steps and not present(init):
                if case["init_video_sha256"] is not None:
                    raise SystemExit(
                        f"{case['case_id']}: {init} is release-bound ({case['init_video_sha256'][:12]}…) and is "
                        f"never regenerated; supply it with --reuse {init}=<path>")
                job.runs.append(ContainerRun(
                    "longcat-init",
                    ["python3", "/opt/qvg-runners/run_longcat.py", "--mode", "bf16", "--make-init", *common,
                     "--output", f"{out}/init", "--master-port", str(port), "--device-index", "0"],
                    env, port, Output(f"init/{index}-{seed}.mp4", init, None, "init_video_sha256")))
                port += 1
            golden = hidden_relative(case["bf16_frames"])
            if "goldens" in steps and not present(golden):
                if not job.runs and not present(init):
                    raise SystemExit(f"{case['case_id']}: the golden continues from {init}, which is missing; "
                                     "stage the init first (--only inits or --reuse)")
                job.runs.append(ContainerRun(
                    "longcat-golden",
                    ["python3", "/opt/qvg-runners/run_longcat.py", "--mode", "bf16", *common,
                     "--output", f"{out}/bf16", "--init-video", case["init_video"],
                     "--segments", str(case["segments"]), "--master-port", str(port), "--device-index", "0"],
                    env, port, Output(f"bf16/{index}-{seed}/final.frames.pt", golden, case["bf16_frames_sha256"],
                                      "bf16_frames_sha256", profile=f"bf16/{index}-{seed}/profile.json")))
                port += 1
        elif case["model"] == "hy_worldplay_8b":
            golden = hidden_relative(case["bf16_frames"])
            if "goldens" in steps and not present(golden):
                # No allocator override here: hyvideo/generate.py itself defaults
                # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, exactly as for
                # the v3 golden and for every candidate run.
                job.runs.append(ContainerRun(
                    "hy-golden",
                    ["python3", "/opt/qvg-runners/run_hy_worldplay_8b.py", "--mode", "bf16", *common,
                     "--video-length", str(case["video_length"]), "--image-path", case["image"],
                     "--prompt", case["prompt"], "--pose", case["pose"], "--output", f"{out}/bf16",
                     "--master-port", str(port), "--device-index", "0"],
                    _thread_env(release), port,
                    Output("bf16/gen.frames.pt", golden, case["bf16_frames_sha256"], "bf16_frames_sha256",
                           profile="bf16/profile.json")))
                port += 1
        else:
            raise SystemExit(f"{case['case_id']}: unknown model {case['model']!r}")
        if job.runs:
            jobs.append(job)
    # Stable sort: release order among equally heavy jobs.
    return sorted(jobs, key=lambda job: -job.cost)


# --------------------------------------------------------------------------
# Release-bound placement
# --------------------------------------------------------------------------


class Manifest:
    """``<hidden>/MANIFEST.json``, merged across runs and written atomically."""

    def __init__(self, hidden: Path, release: dict[str, Any]) -> None:
        self.path = hidden / "MANIFEST.json"
        self.lock = threading.Lock()
        previous = json.loads(self.path.read_text(encoding="utf-8")) if self.path.is_file() else {}
        self.files: dict[str, Any] = previous.get("files", {}) if previous.get("schema_version") == 2 else {}
        self.release = release
        self.header: dict[str, Any] = {}

    def record(self, target: str, entry: dict[str, Any]) -> None:
        with self.lock:
            self.files[target] = entry
            self._write()

    def write(self) -> None:
        with self.lock:
            self._write()

    def _write(self) -> None:
        values: dict[str, dict[str, str]] = {}
        for case in self.release["cases"]:
            for field_name, guest in (("init_video_sha256", case.get("init_video")),
                                      ("image_sha256", case.get("image")),
                                      ("bf16_frames_sha256", case["bf16_frames"])):
                if guest and hidden_relative(guest) in self.files:
                    values.setdefault(case["case_id"], {})[field_name] = self.files[hidden_relative(guest)]["sha256"]
        document = {
            "schema_version": 2,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "release_config": RELEASE.name,
            "release_sha256": sha256_file(RELEASE),
            "release_status": self.release["status"],
            **self.header,
            "files": dict(sorted(self.files.items())),
            "release_values": values,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.path)


def install(source: Path, hidden: Path, target: str, pin: str | None, *, adopt: bool = False) -> str:
    """Copy ``source`` to a missing ``target`` as a read-only file; enforce the pin."""
    destination = hidden / target
    observed = sha256_file(source)
    if pin is not None and observed != pin and not adopt:
        raise StageError(
            f"{target}: produced {observed[:12]}… but the release pins {pin[:12]}…; left at {source}. "
            "This device is a new measurement profile: re-run with --adopt, then re-bind the release deliberately")
    if destination.exists() or destination.is_symlink():
        raise StageError(f"{target}: already present; refusing to overwrite it")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    # A run killed between the chmod and the rename leaves a read-only
    # .partial that a plain copy cannot reopen; its owner can always unlink it.
    temporary.unlink(missing_ok=True)
    try:
        shutil.copyfile(source, temporary)
        temporary.chmod(0o444)
        if sha256_file(temporary) != observed:
            raise StageError(f"{target}: copy changed while installing")
        os.rename(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return observed


def audit_existing(release: dict[str, Any], hidden: Path) -> list[str]:
    """Release-bound files already on disk must match their pins."""
    errors = []
    for case in release["cases"]:
        for guest_key, pin_key in (("init_video", "init_video_sha256"), ("image", "image_sha256"),
                                   ("bf16_frames", "bf16_frames_sha256")):
            if guest_key not in case or case[pin_key] is None:
                continue
            path = hidden / hidden_relative(case[guest_key])
            if path.exists() and sha256_file(path) != case[pin_key]:
                errors.append(f"{case['case_id']}: {path} differs from the release pin {case[pin_key][:12]}…")
    prompts = hidden / "prompts" / PROMPT_NAME
    if prompts.exists() and sha256_file(prompts) != release["prompt_file_sha256"]:
        errors.append(f"{prompts} differs from the release prompt_file_sha256")
    return errors


def stage_prompts(release: dict[str, Any], hidden: Path, manifest: Manifest) -> None:
    target = f"prompts/{PROMPT_NAME}"
    if sha256_file(IN_TREE_PROMPTS) != release["prompt_file_sha256"]:
        raise SystemExit("judge/private/prompts disagrees with the release prompt_file_sha256")
    if not (hidden / target).exists():
        install(IN_TREE_PROMPTS, hidden, target, release["prompt_file_sha256"])
        manifest.record(target, {"sha256": release["prompt_file_sha256"],
                                 "release_sha256": release["prompt_file_sha256"],
                                 "matches_release": True, "source": "judge/private/prompts"})


def stage_images(release: dict[str, Any], root: Path, hidden: Path, manifest: Manifest, cases: set[str] | None) -> None:
    for case in release["cases"]:
        if case["model"] != "hy_worldplay_8b" or (cases is not None and case["case_id"] not in cases):
            continue
        target = hidden_relative(case["image"])
        if (hidden / target).exists():
            continue
        source_spec = case["image_source"]
        if not source_spec.startswith(HY_SOURCE_PREFIX):
            raise SystemExit(f"{case['case_id']}: unexpected image_source {source_spec!r}")
        relative = source_spec[len(HY_SOURCE_PREFIX):].split(" @", 1)[0]
        if not relative.startswith("assets/img/") or ".." in Path(relative).parts:
            raise SystemExit(f"{case['case_id']}: unexpected image_source {source_spec!r}")
        source = root / HY_OFFICIAL / relative
        if not source.is_file():
            raise SystemExit(f"missing pinned HY-WorldPlay checkout file: {source} (see portable/REBUILD_PROMPT.md §2)")
        observed = install(source, hidden, target, case["image_sha256"])
        manifest.record(target, {"sha256": observed, "release_sha256": case["image_sha256"], "matches_release": True,
                                 "case_id": case["case_id"], "source": f"{HY_OFFICIAL}/{relative}"})


def stage_reuse(release: dict[str, Any], hidden: Path, manifest: Manifest, reuse: dict[str, Path]) -> None:
    """Seed hidden-tree files from existing copies (e.g. the v3 init and HY golden)."""
    pins: dict[str, tuple[str, str | None]] = {}
    for case in release["cases"]:
        for guest_key, pin_key in (("init_video", "init_video_sha256"), ("image", "image_sha256"),
                                   ("bf16_frames", "bf16_frames_sha256")):
            if guest_key in case:
                pins[hidden_relative(case[guest_key])] = (case["case_id"], case[pin_key])
    for target, source in reuse.items():
        if target not in pins:
            raise SystemExit(f"--reuse names {target}, which is not a file of the release")
        if (hidden / target).exists():
            continue
        if not source.is_file() or source.is_symlink():
            raise SystemExit(f"--reuse source is not a regular file: {source}")
        case_id, pin = pins[target]
        observed = install(source, hidden, target, pin)
        manifest.record(target, {"sha256": observed, "release_sha256": pin,
                                 "matches_release": None if pin is None else True,
                                 "case_id": case_id, "source": f"reused: {source}"})


# --------------------------------------------------------------------------
# GPUs: resolve, lock, check idle (all through nvidia-smi with a timeout)
# --------------------------------------------------------------------------


def _nvidia_smi(*arguments: str) -> list[list[str]]:
    try:
        out = subprocess.run(["nvidia-smi", *arguments, "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, check=True, timeout=NVIDIA_SMI_TIMEOUT).stdout
    except subprocess.TimeoutExpired as exc:
        raise SystemExit("nvidia-smi did not answer: the driver may be wedged; stop all GPU work") from exc
    return [[part.strip() for part in line.split(",")] for line in out.splitlines() if line.strip()]


def resolve_gpus(requested: list[str]) -> list[dict[str, str]]:
    inventory = _nvidia_smi("--query-gpu=index,uuid,name,driver_version,memory.total,memory.used")
    by_key: dict[str, dict[str, str]] = {}
    for index, uuid, name, driver, total, used in inventory:
        record = {"index": index, "uuid": uuid, "name": name, "driver_version": driver,
                  "memory_total_mib": total, "memory_used_mib": used}
        by_key[index] = by_key[uuid] = record
    resolved = []
    for key in requested:
        if key not in by_key:
            raise SystemExit(f"unknown GPU {key!r}")
        resolved.append(by_key[key])
    if len({gpu["uuid"] for gpu in resolved}) != len(resolved):
        raise SystemExit("--gpus names one device twice")
    return resolved


def lock_gpus(gpus: list[dict[str, str]]) -> list[Any]:
    """Hold both host lock names of every GPU for the life of this process."""
    handles = []
    for gpu in gpus:
        for prefix in LOCK_PREFIXES:
            path = Path("/tmp") / f"{prefix}-gpu-{gpu['uuid']}.lock"
            handle = path.open("a+")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit(f"{path} is held by another process; GPU {gpu['index']} is not free") from None
            handles.append(handle)
    return handles


def assert_idle(gpus: list[dict[str, str]]) -> None:
    busy = {uuid for uuid, *_ in _nvidia_smi("--query-compute-apps=gpu_uuid,pid")}
    for gpu in gpus:
        if gpu["uuid"] in busy or int(gpu["memory_used_mib"]) > IDLE_MEMORY_MIB:
            raise SystemExit(f"GPU {gpu['index']} ({gpu['uuid']}) is not idle")


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


def judge_mounts(root: Path) -> list[str]:
    hy = root / "staging/hy-worldplay-8b"
    pairs = [
        (root / "staging/longcat/LongCat-Video", "/models/hidden/longcat"),
        (hy / "HunyuanVideo-1.5/transformer", "/models/hidden/hy-worldplay/base/transformer"),
        (hy / "HunyuanVideo-1.5/vae", "/models/hidden/hy-worldplay/base/vae"),
        (hy / "HunyuanVideo-1.5/scheduler", "/models/hidden/hy-worldplay/base/scheduler"),
        (hy / "HY-WorldPlay/ar_distilled_action_model", "/models/hidden/hy-worldplay/action"),
        (hy / "Qwen2.5-VL-7B-Instruct", "/models/hidden/hy-worldplay/base/text_encoder/llm"),
        (hy / "byt5-small", "/models/hidden/hy-worldplay/base/text_encoder/byt5-small"),
        (hy / "Glyph-SDXL-v2", "/models/hidden/hy-worldplay/base/text_encoder/Glyph-SDXL-v2"),
        (hy / "FLUX.1-Redux-dev", "/models/hidden/hy-worldplay/base/vision_encoder/siglip"),
        (root / "staging/vbench", "/opt/vbench"),
        (root / "staging/vbench-cache", "/opt/vbench-cache"),
    ]
    args: list[str] = []
    for host, guest in pairs:
        if not host.is_dir():
            raise SystemExit(f"missing public staging tree: {host} (see portable/REBUILD_PROMPT.md §2)")
        args += ["--volume", f"{host}:{guest}:ro"]
    return args


def recipe_mounts() -> list[str]:
    """The recipe is the checked-out tree, not whichever copy the image was built with."""
    return [
        "--volume", f"{TASK_ROOT / 'judge/run_longcat.py'}:/opt/qvg-runners/run_longcat.py:ro",
        "--volume", f"{TASK_ROOT / 'judge/run_hy_worldplay_8b.py'}:/opt/qvg-runners/run_hy_worldplay_8b.py:ro",
        "--volume", f"{TASK_ROOT / 'upstream/experiments/LongCat'}:/opt/qvg-src/experiments/LongCat:ro",
        "--volume", f"{TASK_ROOT / 'upstream/experiments/HY-WorldPlay'}:/opt/qvg-src/experiments/HY-WorldPlay:ro",
        # The pipeline imports the task ABI module even with quantisation off;
        # the in-tree reference codec (oracle/) provides it.
        "--volume", f"{TASK_ROOT / 'oracle'}:/opt/qvg-golden-codec:ro",
    ]


def docker_argv(image: str, gpu_uuid: str, name: str, mounts: list[str], env: dict[str, str],
                command: list[str]) -> list[str]:
    argv = ["docker", "run", "--rm", "--name", name, "--network", "none", "--gpus", f"device={gpu_uuid}",
            "--security-opt", "no-new-privileges:true", "--ipc", "host"]
    for key, value in sorted(env.items()):
        argv += ["--env", f"{key}={value}"]
    return argv + mounts + [image] + command


class Runner:
    """One worker thread per GPU; a job's container runs stay on one GPU, in order."""

    def __init__(self, *, root: Path, hidden: Path, scratch: Path, image: str, gpus: list[dict[str, str]],
                 manifest: Manifest, adopt: bool) -> None:
        self.hidden, self.scratch, self.image, self.gpus, self.manifest = hidden, scratch, image, gpus, manifest
        self.adopt = adopt
        self.mounts = judge_mounts(root) + recipe_mounts() + [
            "--volume", f"{hidden}:{HIDDEN_MOUNT}:ro", "--volume", f"{scratch}:/out:rw"]
        self.pending: queue.Queue[Job] = queue.Queue()
        self.stop = threading.Event()
        self.errors: list[str] = []
        self.running: dict[str, str] = {}
        self.lock = threading.Lock()

    def run(self, jobs: list[Job]) -> list[str]:
        for job in jobs:
            self.pending.put(job)
        workers = [threading.Thread(target=self._worker, args=(gpu,), daemon=True) for gpu in self.gpus]
        for worker in workers:
            worker.start()
        try:
            for worker in workers:
                while worker.is_alive():
                    worker.join(timeout=1.0)
        except KeyboardInterrupt:
            self.stop.set()
            with self.lock:
                names = list(self.running)
            if names:
                subprocess.run(["docker", "kill", *names], capture_output=True, check=False)
            raise
        return self.errors

    def _worker(self, gpu: dict[str, str]) -> None:
        while not self.stop.is_set():
            try:
                job = self.pending.get_nowait()
            except queue.Empty:
                return
            try:
                for run in job.runs:
                    self._run_one(job, run, gpu)
            except (StageError, subprocess.CalledProcessError, OSError) as exc:
                with self.lock:
                    self.errors.append(f"{job.case_id}: {exc}")
                # Running jobs finish; no new job starts after a failure.
                self.stop.set()

    def _run_one(self, job: Job, run: ContainerRun, gpu: dict[str, str]) -> None:
        name = f"qvg-v4-stage-{job.case_id}-{run.step}-{os.getpid()}"
        job_dir = self.scratch / job.scratch
        job_dir.mkdir(parents=True, exist_ok=True)
        job_dir.chmod(0o777)
        log = self.scratch / "logs" / f"{job.case_id}-{run.step}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        argv = docker_argv(self.image, gpu["uuid"], name, self.mounts, run.env, run.command)
        print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu['index']}: {job.case_id} {run.step} -> {log}", flush=True)
        started = time.monotonic()
        with self.lock:
            self.running[name] = job.case_id
        try:
            with log.open("w", encoding="utf-8") as stream:
                stream.write("+ " + " ".join(argv) + "\n")
                stream.flush()
                subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, check=True)
        finally:
            with self.lock:
                self.running.pop(name, None)
        produced = job_dir / run.output.produced
        if not produced.is_file():
            raise StageError(f"{run.step} produced no {run.output.produced} (see {log})")
        observed = install(produced, self.hidden, run.output.target, run.output.pin, adopt=self.adopt)
        profile_path = job_dir / run.output.profile if run.output.profile else None
        profile = (json.loads(profile_path.read_text(encoding="utf-8"))
                   if profile_path is not None and profile_path.is_file() else None)
        self.manifest.record(run.output.target, {
            "sha256": observed, "release_sha256": run.output.pin,
            "matches_release": None if run.output.pin is None else observed == run.output.pin,
            "case_id": job.case_id, "release_field": run.output.field, "source": "generated", "step": run.step,
            "command": run.command, "env": run.env,
            "device": {key: gpu[key] for key in ("index", "uuid", "name", "driver_version", "memory_total_mib")},
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "container_seconds": round(time.monotonic() - started, 1), "profile": profile,
        })
        print(f"[{time.strftime('%H:%M:%S')}] GPU {gpu['index']}: {job.case_id} {run.step} installed "
              f"{run.output.target} {observed[:12]}", flush=True)


def stage_public_eval(root: Path, gpu: dict[str, str], work_image: str, scratch: Path) -> None:
    scratch.mkdir(parents=True, exist_ok=True)
    scratch.chmod(0o777)
    target = root / "public-eval/self-forcing"
    mounts = [
        "--volume", f"{root / 'staging/self-forcing'}:/opt/qvg-src/ckpts/Self-Forcing:ro",
        "--volume", f"{root / 'staging/vbench'}:/opt/vbench:ro",
        "--volume", f"{root / 'staging/vbench-cache'}:/opt/vbench-cache:ro",
        "--volume", f"{scratch}:/workspace/public-results:rw",
    ]
    name = f"qvg-v4-stage-public-{os.getpid()}"
    # bf16 first: the candidate pass compares against /eval/public/self-forcing/bf16.
    subprocess.run(docker_argv(work_image, gpu["uuid"], name, mounts, {},
                               ["python3", "/workspace/public_tests/run_self_forcing_paper.py", "--mode", "bf16"]),
                   check=True)
    bf16 = scratch / "self-forcing/bf16"
    staged_bf16 = scratch / "eval-public/self-forcing/bf16"
    if staged_bf16.exists():
        shutil.rmtree(staged_bf16)
    shutil.copytree(bf16, staged_bf16)
    # The candidate pass imports the task ABI module (mlsysforge_codec) from
    # /workspace/submission; the pristine reference codec is the in-tree
    # oracle/, mounted read-only exactly as recipe_mounts() mounts it for the
    # Judge runners.
    candidate_mounts = mounts + [
        "--volume", f"{scratch / 'eval-public'}:/eval/public:ro",
        "--volume", f"{TASK_ROOT / 'oracle'}:/workspace/submission:ro",
    ]
    subprocess.run(docker_argv(work_image, gpu["uuid"], name, candidate_mounts, {},
                               ["python3", "/workspace/public_tests/run_self_forcing_paper.py",
                                "--mode", "candidate", "--vbench"]), check=True)
    for mode, source in (("bf16", bf16), ("qvg-reference", scratch / "self-forcing/candidate")):
        destination = target / mode
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    frames = {relative.as_posix(): sha256_file(path)
              for path in sorted(target.rglob("*.frames.pt"))
              for relative in [path.relative_to(target)]}
    (target / "MANIFEST.json").write_text(json.dumps(
        {"schema_version": 1, "generated_at": datetime.now(timezone.utc).isoformat(),
         "work_image": work_image, "device": gpu, "frames_sha256": frames},
        indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_reuse(values: list[str]) -> dict[str, Path]:
    reuse: dict[str, Path] = {}
    for value in values:
        target, separator, source = value.partition("=")
        if not separator or not target or not source:
            raise SystemExit(f"--reuse expects HIDDEN_RELATIVE_PATH=SOURCE, got {value!r}")
        reuse[target.strip("/")] = Path(source).resolve()
    return reuse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("QVG_ARTIFACT_ROOT", "")))
    parser.add_argument("--hidden", type=Path,
                        help="hidden tree to produce (default <root>/hidden, mounted at /eval/hidden)")
    parser.add_argument("--gpus", default=os.environ.get("QVG_STAGE_GPUS", ""),
                        help="comma-separated idle devices (indices or UUIDs); one container per device at a time")
    parser.add_argument("--only", action="append", default=[],
                        choices=["prompts", "images", "inits", "goldens", "public-eval"])
    parser.add_argument("--cases", help="comma-separated release case_ids (default: all 17)")
    parser.add_argument("--reuse", action="append", default=[], metavar="RELATIVE=SOURCE",
                        help="install an existing file at a hidden-tree path (checked against its release pin)")
    parser.add_argument("--judge-image", default=f"{BENCHMARK_NAME}.judge.{TASK_ID}:local")
    parser.add_argument("--work-image", default=f"{BENCHMARK_NAME}.work.{TASK_ID}:local")
    parser.add_argument("--scratch", type=Path, help="container output directory (default <root>/.stage-v4)")
    parser.add_argument("--adopt", action="store_true",
                        help="install regenerated files that differ from their release pin (new measurement "
                             "profile: re-bind the release from MANIFEST.json afterwards)")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; touch no GPU, no Docker, no file")
    args = parser.parse_args()
    if not str(args.root):
        raise SystemExit("set QVG_ARTIFACT_ROOT or pass --root")
    root = args.root.resolve()
    hidden = (args.hidden or root / "hidden").resolve()
    scratch = (args.scratch or root / ".stage-v4").resolve()
    steps = set(args.only) or {"prompts", "images", "inits", "goldens", "public-eval"}
    release = load_release()
    known = {case["case_id"] for case in release["cases"]}
    cases = None if not args.cases else {name.strip() for name in args.cases.split(",") if name.strip()}
    if cases is not None and not cases <= known:
        raise SystemExit(f"unknown case ids: {sorted(cases - known)}")
    reuse = parse_reuse(args.reuse)
    errors = audit_existing(release, hidden)
    if errors and not args.adopt:
        raise SystemExit("release-bound files differ from their pins (never overwritten; fix by hand):\n  "
                         + "\n  ".join(errors))
    for error in errors:
        print(f"adopted earlier, release not yet re-bound: {error}", file=sys.stderr)
    if args.dry_run:
        jobs = plan_jobs(release, hidden, steps=steps, cases=cases, reused=frozenset(reuse))
        print(json.dumps({"hidden": str(hidden), "steps": sorted(steps), "reuse": {k: str(v) for k, v in reuse.items()},
                          "jobs": [{"case_id": job.case_id, "cost_seconds": job.cost,
                                    "runs": [{"step": run.step, "command": run.command, "env": run.env,
                                              "target": run.output.target, "pin": run.output.pin}
                                             for run in job.runs]} for job in jobs]}, indent=2))
        return 0
    gpu_keys = [key.strip() for key in args.gpus.split(",") if key.strip()]
    needs_gpu = bool(steps & {"inits", "goldens", "public-eval"})
    if needs_gpu and not gpu_keys:
        raise SystemExit("set QVG_STAGE_GPUS or pass --gpus for the GPU stages")
    gpus = resolve_gpus(gpu_keys) if needs_gpu else []
    _locks = lock_gpus(gpus)  # noqa: F841 - held until exit
    if gpus:
        assert_idle(gpus)
    manifest = Manifest(hidden, release)
    if gpus:
        image_id = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", args.judge_image],
                                  capture_output=True, text=True, check=True).stdout.strip()
        manifest.header = {"judge_image": args.judge_image, "judge_image_id": image_id}
    if "prompts" in steps or "inits" in steps or "goldens" in steps:
        stage_prompts(release, hidden, manifest)
    if "images" in steps or "goldens" in steps:
        stage_images(release, root, hidden, manifest, cases)
    stage_reuse(release, hidden, manifest, reuse)
    failures: list[str] = []
    if steps & {"inits", "goldens"}:
        token = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
        jobs = plan_jobs(release, hidden, steps=steps, cases=cases, token=token)
        print(f"{len(jobs)} job(s) over {len(gpus)} GPU(s): " + ", ".join(job.case_id for job in jobs), flush=True)
        scratch.mkdir(parents=True, exist_ok=True)
        scratch.chmod(0o777)
        failures = Runner(root=root, hidden=hidden, scratch=scratch, image=args.judge_image, gpus=gpus,
                          manifest=manifest, adopt=args.adopt).run(jobs)
    if "public-eval" in steps and not failures:
        stage_public_eval(root, gpus[0], args.work_image, scratch / "public-eval")
        print("public-eval staged", root / "public-eval/self-forcing")
    manifest.write()
    if failures:
        print("staging failed:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 2
    print(f"hidden tree ready: {hidden} (see {manifest.path}; copy release_values into a calibrating release)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
