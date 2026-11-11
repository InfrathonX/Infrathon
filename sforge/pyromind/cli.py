"""Run remote Worker and Judge Sandboxes with shared persistent artifacts."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import socket
import sys
import tarfile
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from sforge.pyromind.bootstrap import DEMO, initialize_store


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def safe_extract(payload, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError(f"Unsafe artifact entry: {member.name}")
        archive.extractall(destination, filter="data")


def checked(backend, handle, cmd, *, timeout=120, **kwargs):
    result = backend.exec_run_with_timeout(handle, cmd, timeout, **kwargs)
    if result.exit_code:
        raise RuntimeError(f"Sandbox {handle.id} command failed ({result.exit_code}): {result.output[-4000:]}")
    return result.output


def package_source(destination):
    package = Path(__file__).resolve().parents[1]
    repo = package.parent
    if not (repo / "pyproject.toml").is_file():
        raise RuntimeError("Injection mode requires an editable source checkout; use --prebuilt with installed wheels")
    with tarfile.open(destination, "w:gz") as archive:
        for name in ("pyproject.toml", "README.md"):
            archive.add(repo / name, arcname=name)
        for path in sorted(package.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                archive.add(path, arcname=str(path.relative_to(repo)))


def install_worker(backend, handle, *, prebuilt):
    checked(backend, handle, "mkdir -p /workspace/project")
    if not prebuilt:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.tar.gz"
            package_source(source)
            backend.copy_to_container(handle, source, PurePosixPath("/tmp/sforge-source.tar.gz"))
        checked(backend, handle, "mkdir -p /opt/sforge && tar xzf /tmp/sforge-source.tar.gz -C /opt/sforge")
        checked(backend, handle,
                "python -m pip install --disable-pip-version-check --no-cache-dir /opt/sforge",
                timeout=500)
        checked(backend, handle,
                "apt-get update -qq && apt-get install -y -qq --no-install-recommends curl jq ca-certificates",
                timeout=500)
    checked(backend, handle, "python -m sforge.pyromind.bootstrap")


def verify_sandbox_network(backend, worker, judge, output):
    """Verify both directions using isolated temporary HTTP directories."""
    if (output / "sandbox-connectivity.json").exists():
        return
    nonce = uuid.uuid4().hex
    root = f"/tmp/sforge-net-{nonce}"
    api_ips = {"worker": backend.get_container_ip(worker), "judge": backend.get_container_ip(judge)}
    ips = {role: backend._checked(handle, ["python", "-c",
           "import socket; print(socket.gethostbyname(socket.gethostname()))"]).strip()
           for role, handle in (("worker", worker), ("judge", judge))}
    started = []
    try:
        for handle in (worker, judge):
            backend._checked(handle, ["mkdir", "-p", root])
            backend.write_to_container(handle, nonce, PurePosixPath(root + "/probe"))
            backend._checked(handle,
                f"nohup python -m http.server 18081 --bind 0.0.0.0 --directory {root} "
                f">{root}/http.log 2>&1 </dev/null & echo $! >{root}/pid")
            started.append(handle)
        for sender, target in ((worker, "judge"), (judge, "worker")):
            code = (
                "import time,urllib.request\n"
                "opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))\n"
                "for attempt in range(10):\n"
                " try:\n"
                f"  data=opener.open('http://{ips[target]}:18081/probe',timeout=3).read().decode()\n"
                f"  assert data == {nonce!r}\n"
                "  break\n"
                " except OSError:\n"
                "  if attempt == 9: raise\n"
                "  time.sleep(0.5)\n"
            )
            checked(backend, sender, ["python", "-c", code], timeout=45)
        write_json(output / "sandbox-connectivity.json", {
            "worker_ip": ips["worker"], "judge_ip": ips["judge"],
            "sdk_internal_ips": api_ips,
            "worker_to_judge": True, "judge_to_worker": True,
            "method": "HTTP over actual Pod IP, port 18081; no port mapping",
        })
        print("Worker <-> Judge Pod-IP HTTP connectivity verified in both directions", flush=True)
    finally:
        for handle in started:
            backend._checked(handle, f"kill $(cat {root}/pid)")


def smoke(args):
    # Must happen before importing SForge constants: this secret never goes to Worker.
    os.environ["SFORGE_ADMIN_SECRET"] = secrets.token_urlsafe(40)
    from sforge.harness.backend.pyromind_backend import PyroMindBackend, PyroMindHandle
    from sforge.harness.config import SForgeConfig
    from sforge.harness.judge_server import create_app
    from sforge.harness.version_store import VersionStore
    import requests
    import uvicorn

    run_id = "pyro-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
    output = args.artifacts.resolve() / run_id
    output.mkdir(parents=True)
    shutil.copytree(DEMO, output / "tasks", ignore=shutil.ignore_patterns("__pycache__"))
    initialize_store(output / "version-store", run_id)
    os.environ["SFORGE_LOG_DIR"] = str(output / "logs")
    os.environ["SFORGE_PYROMIND_EVENTS"] = str(output / "sandbox-events.jsonl")
    judge_entry = args.judge_image if args.prebuilt else {
        "image": args.judge_image,
        "setup_commands": ["mkdir -p /workspace/project /logs/verifier /opt/sforge-demo"],
        "files": {str(output / "tasks/judge.py"): "/opt/sforge-demo/judge.py"},
    }
    os.environ["SFORGE_PYROMIND_IMAGE_MAP"] = json.dumps({
        "pyromind-demo.judge.addition:demo": judge_entry,
    })
    backend = PyroMindBackend()
    config = SForgeConfig(backend="pyromind", tasks_dir=output / "tasks",
                          version_store_root=output / "version-store", log_dir=output / "logs",
                          judge_cpu_limit=4, judge_mem_limit="8Gi")
    state = {"run_id": run_id, "status": "starting", "worker_id": None,
             "judge_url": args.judge_url, "worker_image": args.worker_image,
             "judge_image": args.judge_image, "controller": "local",
             "artifact_store": str(output / "version-store")}
    state_file = output / "run.json"
    write_json(state_file, state)
    print(f"Run {run_id}; persistent artifacts: {output}", flush=True)
    worker = None
    server = thread = listener = app = None
    cleanup_errors = []
    session = requests.Session()
    session.trust_env = False
    try:
        listener = socket.socket()
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.listen_host, args.listen_port))
        app = create_app(config)
        server = uvicorn.Server(uvicorn.Config(app, access_log=False, log_level="warning"))
        thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise RuntimeError("Local Judge Server did not start")
            time.sleep(0.1)
        response = session.post(f"http://127.0.0.1:{args.listen_port}/api/v1/register", json={
            "task_id": "addition", "run_id": run_id,
            "admin_secret": os.environ["SFORGE_ADMIN_SECRET"],
            "max_agent_submissions": 4, "submission_cooldown": 0,
            "require_version_saving": True, "required_version_store_id": run_id,
        }, timeout=10)
        response.raise_for_status()
        registration = response.json()
        if not registration["version_saving_enabled"]:
            raise RuntimeError("Server did not enable Version Store")
        worker = backend.create_container(args.worker_image, run_id + "-worker", environment={
            "SFORGE_JUDGE_URL": args.judge_url.rstrip("/"), "SFORGE_TOKEN": registration["token"],
            "SFORGE_PATCH_DIR": "/workspace/project", "SFORGE_SUBMIT_PATHS": "solution.py",
        })
        state["worker_id"] = worker.id
        write_json(state_file, state)
        print(f"Worker {worker.id}: waiting for image", flush=True)
        backend.start_container(worker)
        # Verify reachability with Python already in the base image, before installing anything.
        probe = (
            "import os,urllib.request; "
            "o=urllib.request.build_opener(urllib.request.ProxyHandler({})); "
            "print(o.open(os.environ['SFORGE_JUDGE_URL']+'/openapi.json',timeout=10).status)"
        )
        checked(backend, worker, ["python", "-c", probe], timeout=15)
        print("Worker -> LOCAL Judge Server HTTP connectivity verified", flush=True)
        state["network_verified"] = True
        write_json(state_file, state)
        install_worker(backend, worker, prebuilt=args.prebuilt)
        cwd_probe = backend.exec_run(worker, ["python", "-c", "import os; print(os.getcwd())"],
                                     workdir="/workspace/project")
        if cwd_probe.exit_code or cwd_probe.output.strip() != "/workspace/project":
            raise RuntimeError("SDK working-directory/argv preflight failed")
        app.state.judge_state.backend.on_started = lambda judge: verify_sandbox_network(
            app.state.judge_state.backend, worker, judge, output
        )
        # Confirm the platform key/admin secret is absent, without printing any environment values.
        checked(backend, worker, ["python", "-c",
                "import os; assert 'PYROMIND_API_KEY' not in os.environ; assert 'SFORGE_ADMIN_SECRET' not in os.environ"])
        for label, content in [
            ("v1-wrong", "def add(a, b):\n    return a - b\n"),
            ("v2-fixed", "def add(a, b):\n    return a + b\n"),
        ]:
            backend.write_to_container(worker, content, PurePosixPath("/workspace/project/solution.py"))
            saved = checked(backend, worker, ["sforge-save-version", "--label", label])
            (output / f"{label}-checkpoint.txt").write_text(saved)
            # Saving returns only after the existing Version Store has persisted the archive.
            if not any(v.label == label for v in VersionStore.open(output / "version-store").list_versions()):
                raise RuntimeError("Checkpoint acknowledged without a durable local version")
            print(f"{label}: checkpoint persisted locally; triggering sforge-submit", flush=True)
            result = checked(backend, worker, ["sforge-submit", "--details"], timeout=570)
            (output / f"{label}-submit.txt").write_text(result)
            print(result, flush=True)
            current = session.get(f"http://127.0.0.1:{args.listen_port}/api/v1/history",
                                  params={"token": registration["token"]}, timeout=10)
            current.raise_for_status()
            latest = current.json()["entries"][-1]
            expected = 0.25 if label == "v1-wrong" else 1.0
            if latest.get("status") != "completed" or latest.get("valid") is not True or latest.get("score") != expected:
                raise RuntimeError(f"{label} failed evaluation; inspect {output / 'logs'} before creating another Judge")
        # Replay v1 through the same Worker command, reusing the normal Judge lifecycle.
        store = VersionStore.open(output / "version-store")
        first = next(v for v in store.list_versions() if v.label == "v1-wrong")
        with tarfile.open(first.archive_path, "r:gz") as bundle:
            member = bundle.extractfile("solution.py")
            if member is None:
                raise RuntimeError("Checkpoint has no solution.py")
            restored = member.read().decode()
        backend.write_to_container(worker, restored, PurePosixPath("/workspace/project/solution.py"))
        replay = checked(backend, worker, ["sforge-submit", "--details"], timeout=570)
        (output / "v1-replay-submit.txt").write_text(replay)
        history_response = session.get(f"http://127.0.0.1:{args.listen_port}/api/v1/history",
                                       params={"token": registration["token"]}, timeout=10)
        history_response.raise_for_status()
        history = history_response.json()
        write_json(output / "history.json", history)
        entries = [e for e in history["entries"] if e.get("type") == "submission"]
        scores = [e.get("score") for e in entries]
        if scores != [0.25, 1.0, 0.25] or any(e.get("status") != "completed" for e in entries):
            raise RuntimeError(f"Unexpected evaluation history: {scores}")
        events = [json.loads(line) for line in (output / "sandbox-events.jsonl").read_text().splitlines()]
        judges = {e["id"] for e in events if e["event"] == "created" and e["id"] != worker.id}
        deleted = {e["id"] for e in events if e["event"] == "deleted"}
        if len(judges) != 3 or not judges.issubset(deleted):
            raise RuntimeError("Expected three distinct, cleaned-up ephemeral Judges")
        state.update(status="passed", scores=scores, judge_ids=sorted(judges),
                     replay="v1 restored from local Version Store and submitted to a new Judge")
        write_json(output / "verification.json", {
            "scores": scores, "worker_id": worker.id, "ephemeral_judges": sorted(judges),
            "version_records": len(store.list_versions()),
            "platform_credentials_in_worker": False,
            "checkpoint_archive_sha256": hashlib.sha256(first.archive_path.read_bytes()).hexdigest(),
        })
    except BaseException as error:
        state["status"] = "failed"
        message = str(error)
        for key in ("PYROMIND_API_KEY", "SFORGE_ADMIN_SECRET"):
            if os.environ.get(key):
                message = message.replace(os.environ[key], "[REDACTED]")
        state["error"] = message
        raise
    finally:
        # Drain server work before cleaning IDs; interrupted/failed runs retain all local artifacts.
        if server:
            server.should_exit = True
        if thread:
            thread.join(timeout=20)
        if app:
            # Judge work runs in its own thread, beyond uvicorn request lifetime.
            deadline = time.monotonic() + backend.ready_timeout + 60
            judge_state = app.state.judge_state
            while any(item["status"] in {"queued", "running"}
                      for item in judge_state.submissions.values()):
                if time.monotonic() >= deadline:
                    break
                time.sleep(1)
            judge_state._reaper_stop.set()
        if listener:
            listener.close()
        session.close()
        events_file = output / "sandbox-events.jsonl"
        if events_file.exists():
            events = [json.loads(line) for line in events_file.read_text().splitlines()]
            created = {e["id"]: e["name"] for e in events if e["event"] == "created"}
            deleted = {e["id"] for e in events if e["event"] == "deleted"}
            for identifier in created.keys() - deleted:
                try:
                    backend.cleanup_container(PyroMindHandle(identifier, created[identifier]))
                except Exception:
                    cleanup_errors.append(identifier)
        state["cleanup_errors"] = cleanup_errors
        write_json(state_file, state)
    if cleanup_errors:
        raise RuntimeError(f"Cleanup incomplete; see {state_file}")
    persisted = VersionStore.open(output / "version-store").list_versions()
    if len(persisted) != 5 or any(
        hashlib.sha256(version.archive_path.read_bytes()).hexdigest() != version.archive_sha256
        for version in persisted
    ):
        state["status"] = "failed"
        state["error"] = "Post-cleanup artifact integrity verification failed"
        write_json(state_file, state)
        raise RuntimeError(state["error"])
    remaining = {sb.id for sb in backend.client.list()} & set(created)
    if remaining:
        state["cleanup_errors"] = sorted(remaining)
        write_json(state_file, state)
        raise RuntimeError(f"Sandbox deletion still pending: {sorted(remaining)}")
    verification = json.loads((output / "verification.json").read_text())
    verification.update(all_run_sandboxes_deleted=True, archives_verified_after_cleanup=len(persisted))
    write_json(output / "verification.json", verification)
    print(f"PASS: scores {state['scores']}; Worker and all Judges deleted; versions remain in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    repo = Path(__file__).resolve().parents[2]
    inventory = sub.add_parser('inventory', help='Read local task and asset bindings; no credentials, uploads or Sandbox creation')
    inventory.add_argument('--artifact-root', type=Path, default=repo.parent / 'artifacts')
    inventory.add_argument('--output', type=Path, help='Write a rebuildable local JSON index; otherwise print it')
    runs = sub.add_parser('runs', help='Classify Storage run metadata; no Sandbox creation')
    runs.add_argument('--output-dir', type=Path, default=repo.parent / 'artifacts/run-catalog',
                      help='Local directory for readable and machine-readable run indexes')
    runs.add_argument('--publish-index', action='store_true',
                      help='Publish only RUNS.md, run-index.json and _index pointers; preserve original runs')
    images = sub.add_parser('build-images', help='Build audited task images locally; publication is explicit')
    images.add_argument('--deployment', type=Path, required=True)
    image_action = images.add_mutually_exclusive_group()
    image_action.add_argument('--stage-only', action='store_true', help='Inspect allowlisted build contexts without Docker build')
    image_action.add_argument('--publish-audited', action='store_true', help='Publish already built and audited image IDs, without rebuilding')
    assets = sub.add_parser('prepare-assets', help='Verify and upload immutable shared model/data assets')
    assets.add_argument('--spec', type=Path, required=True, help='Local asset source/manifest/mount specification')
    assets.add_argument('--output', type=Path, required=True, help='Write bindings consumed by run --assets-file')
    kernel = sub.add_parser("run", aliases=["kernel-run"], help="Run a deployment profile with persistent Worker, shared assets and on-demand Judges")
    kernel.add_argument("--deployment", type=Path, default=repo / "deploy/pyromind/kernel/deployment.json")
    kernel.add_argument("--assets-file", type=Path, help="Prepared read-only asset bindings with SHA-256 manifests")
    kernel.add_argument("--judge-assets-file", type=Path, help="Private immutable assets mounted only in Judges")
    kernel.add_argument("--hours", type=float, default=6)
    kernel.add_argument("--resume-from-run", type=Path,
                        help="Recover a failed run's saved candidate into a new Worker; retain its original budget and deadline")
    kernel.add_argument("--resume-version",
                        help="Saved version ID in the parent run; defaults to its latest checkpoint")
    kernel.add_argument("--task-dir", type=Path, help="Override task path; task ID must match the profile")
    kernel.add_argument("--images-file", type=Path, help="Override the profile's pinned image manifest")
    kernel.add_argument("--model-config", type=Path, default=repo.parent.parent / "mlsys_forge/launch/mlsysforge-deepseek.env")
    kernel.add_argument("--judge-url", help="Worker-visible lightweight coordinator URL; otherwise autodetect this host")
    kernel.add_argument("--listen-host", default="0.0.0.0")
    kernel.add_argument("--listen-port", type=int, default=19097)
    kernel.add_argument("--artifacts", type=Path, default=Path("artifacts/pyromind-kernel"))
    kernel.add_argument("--preflight", action="store_true", help="Qualify native Worker/save/submit/Judge flow without Agent runtime or model credentials")
    kernel.add_argument("--review-model-config", type=Path,
                        help="Model/runtime/credentials (agent.env format) for the mandatory CPU integrity reviewer; "
                             "defaults to --model-config (ChatGPT credentials then run the calibrated reviewer model)")
    kernel.add_argument("--llm-judge-mode", choices=["enforce", "shadow", "off"], default=None,
                        help="Integrity review mode (default enforce; shadow/off are for debugging only)")
    qualify = sub.add_parser("kernel-qualify", help="Qualify the private GPU Judge with a trusted oracle; no Worker")
    qualify.add_argument("--task-dir", type=Path, default=repo / "tasks/archive/standalone-dynamic-grouped-fp8-h100-v2-scratch")
    qualify.add_argument("--images-file", type=Path, default=repo / "deploy/pyromind/kernel/images.json")
    qualify.add_argument("--artifacts", type=Path, default=Path("artifacts/pyromind-qualification"))
    review = sub.add_parser('review', help='Integrity-review archives in CPU-only review Sandboxes; no Worker, no GPU')
    review.add_argument('--deployment', type=Path, required=True)
    review.add_argument('--archive', type=Path, action='append', required=True, help='Repeatable')
    review.add_argument('--review-model-config', type=Path, required=True,
                        help='Reviewer model/runtime/credentials in agent.env format')
    review.add_argument('--baseline-dir', type=Path, help="The agent's starting files (task cwd layout)")
    review.add_argument('--documents-root', type=Path, help='Host directory mirroring the briefs\' Work-image paths')
    review.add_argument('--task-dir', type=Path)
    review.add_argument('--images-file', type=Path)
    review.add_argument('--artifacts', type=Path, default=Path('artifacts/integrity-reviews'))
    task_qualify = sub.add_parser('qualify', help='Run original interactive/final GPU Judges with a controller-only positive control')
    task_qualify.add_argument('--deployment', type=Path, required=True)
    task_qualify.add_argument('--candidate-archive', type=Path, required=True)
    task_qualify.add_argument('--assets-file', type=Path)
    task_qualify.add_argument('--judge-assets-file', type=Path)
    task_qualify.add_argument('--rounds', nargs='+', choices=['interactive','final'], default=['interactive','final'])
    task_qualify.add_argument('--native-requalification', action='store_true', help='Explicit task-owned non-release qualification protocol; does not authorize ordinary runs')
    task_qualify.add_argument('--model-validation', type=Path, help='Original full model-validation receipt required by native requalification')
    task_qualify.add_argument('--artifacts', type=Path, default=Path('artifacts/gpu-validation'))
    for command in ("launch", "smoke"):
        pair = sub.add_parser(command, help=("Start and keep two remote Sandboxes" if command == "launch"
            else "Test two remote Sandboxes, shared checkpoints, replay and persistence; clean up"))
        pair.add_argument("--images-file", type=Path, default=DEMO.parent / "images.json")
        pair.add_argument("--worker-image")
        pair.add_argument("--judge-image")
        pair.add_argument("--artifacts", type=Path, default=Path("artifacts/pyromind"))
    execute = sub.add_parser("exec", help="Run a command in an existing Worker with SForge configured")
    execute.add_argument("--run-dir", type=Path, required=True)
    execute.add_argument("worker_command", nargs=argparse.REMAINDER)
    launch = sub.add_parser("ephemeral-smoke", help="Legacy local control plane with ephemeral remote Judges")
    launch.add_argument("--judge-url", help="Worker-visible URL; defaults to this host's outbound IPv4 and listen port")
    launch.add_argument("--listen-host", default="0.0.0.0")
    launch.add_argument("--listen-port", type=int, default=8080)
    launch.add_argument("--worker-image")
    launch.add_argument("--judge-image")
    launch.add_argument("--images-file", type=Path, help="JSON with immutable worker and judge image references")
    launch.add_argument("--prebuilt", action="store_true", help="Use images from deploy/pyromind instead of runtime installation")
    launch.add_argument("--artifacts", type=Path, default=Path("artifacts/pyromind"))
    clean = sub.add_parser("cleanup", help="Clean up only the exact Sandbox IDs recorded for one run")
    clean.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'inventory':
        # Local metadata inspection must also work without a platform .env.
        from sforge.pyromind.inventory import build_inventory
        try:
            result = build_inventory(repo, args.artifact_root.resolve())
            if args.output:
                os.umask(0o077)
                write_json(args.output, result)
                print(f'Local inventory written: {args.output.resolve()}')
            else:
                print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        except Exception as error:
            print(f'Inventory failed: {type(error).__name__}', file=sys.stderr)
            return 1
    if args.command in {"launch", "smoke"}:
        images = json.loads(args.images_file.read_text())
        args.worker_image = args.worker_image or images["worker"]
        args.judge_image = args.judge_image or images["judge"]
    if args.command == "ephemeral-smoke":
        images = json.loads(args.images_file.read_text()) if args.images_file else {}
        args.worker_image = args.worker_image or images.get("worker", "python:3.11-slim-bookworm")
        args.judge_image = args.judge_image or images.get("judge", "python:3.11-slim-bookworm")
        if args.prebuilt and not (args.images_file or (args.worker_image != "python:3.11-slim-bookworm" and args.judge_image != "python:3.11-slim-bookworm")):
            parser.error("--prebuilt requires --images-file or explicit built Worker/Judge images")
    if args.command == "ephemeral-smoke" and not args.judge_url:
        # UDP connect selects the source interface without sending any packet.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
            route.connect(("8.8.8.8", 53))
            address = route.getsockname()[0]
        args.judge_url = f"http://{address}:{args.listen_port}"
    if args.command == "ephemeral-smoke":
        parsed = urlparse(args.judge_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            parser.error("--judge-url must be an HTTP(S) URL without embedded credentials")
    if args.env_file:
        from dotenv import load_dotenv
        if not args.env_file.is_file():
            parser.error("env file does not exist")
        load_dotenv(args.env_file, override=False)
    os.environ.setdefault("PYROMIND_CLUSTER", "us-west-1")
    os.environ.setdefault("PYROMIND_BASE_URL", "https://api.pyromind.ai/api/v1")
    if args.command != 'build-images' and not os.environ.get("PYROMIND_API_KEY"):
        parser.error("Set PYROMIND_API_KEY or pass --env-file")
    os.umask(0o077)
    try:
        if args.command == 'runs':
            from collections import Counter
            from sforge.pyromind.run_catalog import (
                build_run_catalog, render_catalog_files, publish_catalog_files,
            )
            from sforge.pyromind.storage import storage_client
            client, bucket = storage_client()
            catalog = build_run_catalog(client, bucket)
            files = render_catalog_files(catalog)
            output = args.output_dir.resolve()
            for relative, content in files.items():
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            if args.publish_index:
                publish_catalog_files(client, bucket, files)
            print(json.dumps({'output': str(output), 'run_count': len(catalog['runs']),
                              'kinds': dict(Counter(row['run_kind'] for row in catalog['runs'])),
                              'issue_count': len(catalog['issues']),
                              'index_published': args.publish_index}, ensure_ascii=False, indent=2))
            return 0
        if args.command == 'build-images':
            from sforge.pyromind.image_commands import build_images
            build_images(args)
        elif args.command == 'prepare-assets':
            from sforge.pyromind.asset_commands import prepare_assets
            prepare_assets(args)
        elif args.command == "cleanup":
            from sforge.harness.backend.pyromind_backend import PyroMindBackend, PyroMindHandle
            from pyromind_sdk.client.base import PyroMindAPIError
            events_file = args.run_dir.resolve() / "sandbox-events.jsonl"
            events = [json.loads(line) for line in events_file.read_text().splitlines()]
            backend = PyroMindBackend(event_path=events_file)
            created = {e["id"]: e["name"] for e in events if e["event"] == "created"}
            for identifier, name in created.items():
                if not re.fullmatch(r"sb-[a-f0-9]{12}", identifier):
                    raise ValueError("Invalid recorded Sandbox ID")
                try:
                    current = backend.client.get_sandbox(identifier)
                except PyroMindAPIError as error:
                    if error.status_code == 404:
                        continue
                    raise
                if current.name != name:
                    raise RuntimeError(f"Sandbox identity mismatch: {identifier}")
                backend.cleanup_container(PyroMindHandle(identifier, name))
            print(f"Cleanup completed for {args.run_dir}")
        elif args.command == "kernel-qualify":
            from sforge.pyromind.kernel_qualify import qualify_kernel
            qualify_kernel(args)
        elif args.command == 'review':
            from sforge.pyromind.integrity_review import run_offline_review
            return run_offline_review(args)
        elif args.command == 'qualify':
            from sforge.pyromind.qualification import qualify_task
            qualify_task(args)
        elif args.command in {"run", "kernel-run"}:
            if not 0 < args.hours <= 24:
                raise ValueError("--hours must be positive and at most 24")
            from sforge.pyromind.runner import run_task
            run_task(args)
        elif args.command in {"launch", "smoke"}:
            from sforge.pyromind.two_sandbox import launch_pair
            launch_pair(args)
        elif args.command == "exec":
            from sforge.pyromind.two_sandbox import exec_worker
            exec_worker(args)
        else:
            smoke(args)
    except Exception as error:
        message = str(error).replace(os.environ["PYROMIND_API_KEY"], "[REDACTED]")
        print(f"ERROR: {message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
