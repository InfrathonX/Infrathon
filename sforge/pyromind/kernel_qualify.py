"""Qualify the real private kernel Judge with a trusted oracle, never a Worker.

The oracle and private evaluation evidence stay on the controller and in the
single disposable Judge. This command does not create a shared artifact mount.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import tarfile
import time
import uuid

from sforge.pyromind.cli import write_json


def oracle_archive(task_root: Path) -> bytes:
    source = task_root / "oracle"
    for name in ("candidates", "FINAL_VERSION", "notes.md"):
        if not (source / name).exists():
            raise ValueError(f"Missing qualification oracle component: {name}")
    selected = (source / "FINAL_VERSION").read_text().strip()
    if not selected.isdecimal() or not 1 <= int(selected) <= 999:
        raise ValueError("Invalid oracle version selector")
    if not (source / f"candidates/kernel_v{int(selected):03d}/submission.py").is_file():
        raise ValueError("Missing selected oracle implementation")
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as bundle:
        for name in ("candidates", "FINAL_VERSION", "notes.md"):
            bundle.add(source / name, arcname=name)
    from sforge.harness.run_evaluation import _validate_submission_archive
    payload = output.getvalue()
    _validate_submission_archive(payload)
    return payload


def _judge_resources():
    """The FP8 kernel deployment's Judge size (deploy/pyromind/RESOURCES.md)."""
    from sforge.pyromind.deployment import Deployment

    repo = Path(__file__).resolve().parents[2]
    deployment = Deployment.load(repo / "deploy/pyromind/kernel/deployment.json", repo)
    return deployment.resources_for_role("judge")


def qualify_kernel(args):
    from sforge.pyromind.deployment import Deployment

    # The FP8 v2 profile is archived; its blockers refuse this legacy path too.
    repo = Path(__file__).resolve().parents[2]
    Deployment.load(repo / "deploy/pyromind/kernel/deployment.json", repo).assert_launch_ready(preflight=True)
    from sforge.harness.backend.pyromind_backend import PyroMindBackend, PyroMindHandle
    from sforge.harness.benchmark import load_benchmark
    from sforge.harness.config import SForgeConfig
    from sforge.harness.run_evaluation import judge_submission
    from sforge.harness.task_spec import load_all_tasks
    from sforge.pyromind.kernel_run import _private_installer, private_runtime_diagnostics
    from sforge.pyromind.gpu_runtime import prepare_hami_runtime, prepare_private_tmp

    task_root = args.task_dir.resolve()
    tasks_dir = task_root / "tasks"
    task = load_all_tasks(tasks_dir, load_benchmark(tasks_dir))[0]
    image = json.loads(args.images_file.read_text())["judge"]
    if "@sha256:" not in image:
        raise ValueError("Qualification requires a published immutable Judge digest")
    run_id = "qualify-" + time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
    output = args.artifacts.resolve() / run_id
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    output.chmod(0o700)
    events = output / "sandbox-events.jsonl"
    backend = PyroMindBackend(event_path=events, gpu_count=1, gpu_card="H100")
    backend.ready_timeout = 900
    install_private = _private_installer(Path(__file__).resolve().parents[2])

    def started(handle):
        with private_runtime_diagnostics(output,handle.id):
            write_json(output / 'gpu-runtime.json', prepare_hami_runtime(backend, handle))
            write_json(output / 'private-temporary-permissions.json', prepare_private_tmp(backend, handle))
        gpu = backend._checked(handle, ["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version",
                                         "--format=csv,noheader"]).strip()
        if len(gpu.splitlines()) != 1 or "H100" not in gpu:
            raise RuntimeError("Qualification needs exactly one H100")
        write_json(output / "gpu.json", {"sandbox_id": handle.id, "gpu": gpu})
        print(f"Judge {handle.id}: {gpu}", flush=True)
        write_json(output / "private-payload-metadata.json", install_private(backend, handle, task_root))
        # Qualification-only root diagnostics: preserve child stderr before the
        # unchanged verifier removes its temporary directory. Never shared.
        watcher = '''import glob, pathlib, time
root=pathlib.Path('/logs/private-diagnostics');root.mkdir(mode=0o700,exist_ok=True)
opened={}
while True:
    for name in glob.glob('/tmp/standalone-grouped-fp8-v2-*/candidate_worker.log'):
        if name not in opened:
            try: opened[name]=open(name,'rb')
            except FileNotFoundError: pass
    for name,stream in opened.items():
        data=stream.read()
        if data:
            with (root/(pathlib.Path(name).parent.name+'.log')).open('ab') as out: out.write(data)
    time.sleep(.02)
'''
        backend.write_to_container(handle, watcher, PurePosixPath('/root/qualification-diagnostics.py'))
        backend._checked(handle, 'nohup python3 /root/qualification-diagnostics.py >/logs/qualification-diagnostics.log 2>&1 </dev/null &')

    backend.on_started = started
    cleanup = backend.cleanup_container
    def cleanup_with_diagnostics(handle, logger=None):
        try:
            diagnostics = backend.copy_from_container(handle, PurePosixPath('/logs/private-diagnostics'))
            (output / 'private-diagnostics.tar').write_bytes(diagnostics)
        except Exception:
            pass
        finally:
            cleanup(handle, logger)
    backend.cleanup_container = cleanup_with_diagnostics
    archive = oracle_archive(task_root)
    resources = _judge_resources()
    config = SForgeConfig(backend="pyromind", tasks_dir=tasks_dir, log_dir=output / "logs",
                          judge_cpu_limit=resources["cpu"], judge_mem_limit=resources["memory"])
    state = {"run_id": run_id, "kind": "trusted-oracle-qualification",
             "task_id": task.task_id, "judge_image": image,
             "oracle_sha256": hashlib.sha256(archive).hexdigest(), "status": "running",
             "worker_created": False, "shared_artifacts": False}
    write_json(output / "run.json", state)
    print(f"QUALIFICATION_DIRECTORY={output}", flush=True)
    try:
        report = judge_submission(task, archive, config, backend, submission_id=run_id,
                                  log_dir=output / "evaluation", run_id=run_id,
                                  round_id="manual-1", judge_image_ref=image)
        write_json(output / "report.json", report.to_dict())
        if not report.valid or report.score is None or report.score <= 0:
            raise RuntimeError(f"Trusted oracle qualification failed; inspect {output / 'report.json'}")
        state.update(status="passed", score=report.score, valid=report.valid,
                     evaluation_seconds=report.runtime_seconds)
    except BaseException as error:
        state.update(status="failed", error=str(error).replace(os.environ.get("PYROMIND_API_KEY", "\0"), "[REDACTED]"))
        raise
    finally:
        cleanup_errors = []
        if events.exists():
            entries = [json.loads(line) for line in events.read_text().splitlines()]
            created = {e["id"]: e["name"] for e in entries if e["event"] == "created"}
            deleted = {e["id"] for e in entries if e["event"] == "deleted"}
            for identifier in created.keys() - deleted:
                try:
                    backend.cleanup_container(PyroMindHandle(identifier, created[identifier]))
                except Exception:
                    cleanup_errors.append(identifier)
        state["cleanup_errors"] = cleanup_errors
        write_json(output / "run.json", state)
    if cleanup_errors:
        raise RuntimeError("Qualification Judge cleanup is incomplete")
    print(f"PASS: trusted oracle final holdout score={state['score']}; Judge deleted", flush=True)
