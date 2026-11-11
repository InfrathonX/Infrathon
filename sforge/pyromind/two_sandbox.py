"""Launch exactly two remote sandboxes; all judging runs on the remote Judge."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import secrets
import shlex
import time
from types import SimpleNamespace
import uuid

from sforge.harness.backend.pyromind_backend import PyroMindBackend
from sforge.harness.version_store import VersionStore


def verify_object_storage(prefix, versions, local_root):
    """Verify persisted bytes after deletion using controller-only credentials."""
    from pyromind_sdk import PyroMindAPIClient, StorageClient
    info = PyroMindAPIClient(timeout=30, max_retries=0).profile.get_storage_info()
    storage = StorageClient(endpoint=info.url, access_key=info.access_key,
                            secret_key=info.secret_key, region="us-east-1")
    bucket = os.getenv("PYROMIND_STORAGE_BUCKET")
    if not bucket:
        buckets = storage.client.list_buckets()
        if len(buckets) != 1:
            raise ValueError("Set PYROMIND_STORAGE_BUCKET to select the artifact bucket")
        bucket = buckets[0].name
    for version in versions:
        name = prefix + "/" + version.archive_path.relative_to(local_root).as_posix()
        response = storage.client.get_object(bucket, name)
        try:
            digest = hashlib.sha256(response.read()).hexdigest()
        finally:
            response.close()
            response.release_conn()
        if digest != version.archive_sha256:
            raise RuntimeError("Persisted object archive digest mismatch")
    return {"bucket": bucket, "prefix": prefix, "verified_archives": len(versions)}


def launch_pair(args):
    from pyromind_sdk.client.models import VolumeMount
    from sforge.pyromind.cli import checked, safe_extract, write_json

    run_id = "pair-" + time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
    output = args.artifacts.resolve() / run_id
    output.mkdir(parents=True, exist_ok=False)
    prefix = f"sforge-pyromind/{run_id}"
    shared_path = "/workspace/" + prefix
    backend = PyroMindBackend(event_path=output / "sandbox-events.jsonl")
    state = {"run_id": run_id, "mode": "two-remote-sandboxes", "status": "starting",
             "worker_image": args.worker_image, "judge_image": args.judge_image,
             "shared_host_path": shared_path, "shared_mount": "/shared",
             "object_prefix": prefix, "local_judge_server": False}
    write_json(output / "run.json", state)
    handles = []
    judge = worker = None
    passed = False
    print(f"Run {run_id}; shared storage: {shared_path}", flush=True)
    try:
        for role, image in (("judge", args.judge_image), ("worker", args.worker_image)):
            backend.volume_mounts = [VolumeMount(host_path=shared_path, mount_path="/shared",
                                                read_only=(role == "worker"))]
            handle = backend.create_container(image, f"{run_id}-{role}")
            handles.append(handle)
            state[role + "_id"] = handle.id
            state[role + "_name"] = handle.name
            write_json(output / "run.json", state)
            print(f"Created remote {role}: {handle.id}", flush=True)
            if role == "judge":
                judge = handle
            else:
                worker = handle
        backend.start_container(judge)
        print("Judge Sandbox running; starting its SForge HTTP service", flush=True)
        admin_secret = secrets.token_urlsafe(32)
        backend.write_to_container(judge, json.dumps({"secret": admin_secret}),
                                    PurePosixPath("/run/sforge/admin.json"))
        backend._checked(judge, "chmod 700 /run/sforge && chmod 600 /run/sforge/admin.json")
        backend._checked(judge,
            f"umask 077; nohup python -m sforge.pyromind.resident_judge --run-id {run_id} "
            ">/run/sforge/service.log 2>&1 </dev/null & echo $! >/run/sforge/service.pid")
        ready = """import time, urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for i in range(30):
    try:
        assert opener.open('http://127.0.0.1:8080/openapi.json', timeout=2).status == 200
        break
    except Exception:
        time.sleep(1)
else:
    raise RuntimeError('Remote Judge service did not become ready')
"""
        checked(backend, judge, ["python", "-c", ready], timeout=40)
        address = backend._checked(judge, ["python", "-c",
            "import socket; print(socket.gethostbyname(socket.gethostname()))"]).strip()
        state["judge_url"] = f"http://{address}:8080"
        # Use the launcher's newly generated SForge registration secret normally.
        # Do not use SDK exec to read credential files from the remote Sandbox.
        # This is not the PyroMind key. Only the task-scoped token goes to Worker.
        register = """import json, urllib.request
body = dict(task_id='addition', run_id=RUN_ID,
    admin_secret=ADMIN_SECRET,
    max_agent_submissions=100, submission_cooldown=0,
    require_version_saving=True, required_version_store_id=RUN_ID)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
req = urllib.request.Request('http://127.0.0.1:8080/api/v1/register',
    data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
print(opener.open(req, timeout=10).read().decode())
""".replace("RUN_ID", repr(run_id)).replace("ADMIN_SECRET", repr(admin_secret))
        registration = json.loads(backend._checked(judge, ["python", "-c", register]))
        worker.environment.update({"SFORGE_JUDGE_URL": state["judge_url"],
            "SFORGE_TOKEN": registration["token"], "SFORGE_PATCH_DIR": "/workspace/project",
            "SFORGE_SUBMIT_PATHS": "solution.py"})
        backend.start_container(worker)
        environment = "\n".join("export " + k + "=" + shlex.quote(v)
                                for k, v in worker.environment.items()) + "\n"
        backend.write_to_container(worker, environment, PurePosixPath("/etc/sforge/worker.env"))
        backend._checked(worker, "chmod 700 /etc/sforge && chmod 600 /etc/sforge/worker.env")
        backend.write_to_container(worker, ". /etc/sforge/worker.env\n",
                                    PurePosixPath("/etc/profile.d/sforge.sh"))
        # A unique sentinel proves shared bytes; Worker must not be able to modify them.
        nonce = uuid.uuid4().hex
        backend.write_to_container(judge, nonce, PurePosixPath("/shared/connectivity.txt"))
        probe = """import errno, os, urllib.request
from pathlib import Path
assert Path('/shared/connectivity.txt').read_text() == NONCE
try:
    Path('/shared/worker-write-must-fail').write_text('unexpected')
except OSError as e:
    assert e.errno in (errno.EROFS, errno.EACCES), e
else:
    raise RuntimeError('Worker artifact mount must be read-only')
assert 'PYROMIND_API_KEY' not in os.environ
assert 'SFORGE_ADMIN_SECRET' not in os.environ
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
assert opener.open(os.environ['SFORGE_JUDGE_URL']+'/openapi.json',timeout=10).status == 200
print('Worker -> remote Judge HTTP; shared files visible; Worker mount read-only')
""".replace("NONCE", repr(nonce))
        print(backend._checked(worker, ["python", "-c", probe]).strip(), flush=True)
        backend._checked(judge, ["python", "-c",
            "import os; assert 'PYROMIND_API_KEY' not in os.environ"])
        mounts = backend._checked(judge, ["python", "-c",
            "from pathlib import Path; print(''.join(l+'\\n' for l in Path('/proc/mounts').read_text().splitlines() if ' /shared ' in l))"])
        state.update(status="ready", shared_mount_info=mounts.strip())
        write_json(output / "run.json", state)
        backend.write_to_container(judge, json.dumps(state, indent=2), PurePosixPath("/shared/metadata.json"))
        exec_worker(SimpleNamespace(run_dir=output, worker_command=["python", "-c",
            "import os; assert os.environ.get('SFORGE_TOKEN'); print('Worker shell configured')"]))
        if args.command == "launch":
            passed = True
            print(f"READY: Worker {worker.id} -> Judge {judge.id} ({state['judge_url']})", flush=True)
            print(f"Run directory: {output}\nUse sforge-pyromind exec --run-dir {output} -- sforge-save-version --label v1", flush=True)
            return

        def history():
            code = """import json,os,urllib.request,urllib.parse
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
url=os.environ['SFORGE_JUDGE_URL']+'/api/v1/history?'+urllib.parse.urlencode({'token':os.environ['SFORGE_TOKEN']})
print(opener.open(url,timeout=10).read().decode())
"""
            return json.loads(backend._checked(worker, ["python", "-c", code]))

        for label, operator, expected in (("v1-wrong", "-", 0.25), ("v2-fixed", "+", 1.0)):
            backend.write_to_container(worker, f"def add(a, b):\n    return a {operator} b\n",
                                        PurePosixPath("/workspace/project/solution.py"))
            saved = checked(backend, worker, ["sforge-save-version", "--label", label])
            (output / f"{label}-checkpoint.txt").write_text(saved)
            visible = """from pathlib import Path
from sforge.harness.version_store import VersionStore
versions=VersionStore.open(Path('/shared/version-store')).list_versions()
assert any(v.label == LABEL for v in versions)
print('Checkpoint visible in Worker /shared/version-store')
""".replace("LABEL", repr(label))
            backend._checked(worker, ["python", "-c", visible])
            result = checked(backend, worker, ["sforge-submit", "--details"], timeout=120)
            (output / f"{label}-submit.txt").write_text(result)
            entry = history()["entries"][-1]
            if entry.get("status") != "completed" or entry.get("score") != expected:
                raise RuntimeError(f"Unexpected evaluation for {label}: {entry}")
            print(f"{label}: shared checkpoint saved; same remote Judge returned {expected}", flush=True)
        replay = """from pathlib import Path
import tarfile
from sforge.harness.version_store import VersionStore
v=next(v for v in VersionStore.open(Path('/shared/version-store')).list_versions() if v.label=='v1-wrong')
with tarfile.open(v.archive_path) as bundle:
    Path('/workspace/project/solution.py').write_bytes(bundle.extractfile('solution.py').read())
"""
        backend._checked(worker, ["python", "-c", replay])
        (output / "v1-replay-submit.txt").write_text(
            checked(backend, worker, ["sforge-submit", "--details"], timeout=120))
        report = history()
        write_json(output / "history.json", report)
        scores = [e.get("score") for e in report["entries"] if e.get("type") == "submission"]
        if scores != [0.25, 1.0, 0.25]:
            raise RuntimeError(f"Wrong replay scores: {scores}")
        safe_extract(backend.copy_from_container(judge, PurePosixPath("/shared")), output)
        versions = VersionStore.open(output / "shared/version-store").list_versions()
        if len(versions) != 5:
            raise RuntimeError("Expected two checkpoints and three submitted archives")
        state.update(status="passed", scores=scores, version_records=len(versions))
        passed = True
    except BaseException as error:
        state.update(status="failed", error=str(error).replace(os.environ.get("PYROMIND_API_KEY", "\0"), "[REDACTED]"))
        raise
    finally:
        if judge:
            try:
                (output / "judge-service.log").write_bytes(
                    backend.client.read_file(judge.id, "/run/sforge/service.log"))
            except Exception:
                pass
        if args.command != "launch" or not passed:
            errors = []
            for handle in reversed(handles):
                try:
                    backend.cleanup_container(handle)
                except Exception:
                    errors.append(handle.id)
            state["cleanup_errors"] = errors
        write_json(output / "run.json", state)
    if state.get("cleanup_errors"):
        raise RuntimeError(f"Cleanup incomplete; see {output}")
    remaining = {s.id for s in backend.client.list()} & {h.id for h in handles}
    if remaining:
        raise RuntimeError(f"Sandboxes still present: {sorted(remaining)}")
    # This read goes straight to PyroMind Storage after both Sandboxes are gone.
    try:
        persisted = verify_object_storage(prefix, versions, output / "shared")
    except Exception:
        state["status"] = "persistence-verification-failed"
        write_json(output / "run.json", state)
        raise
    verification = {**state, "created_sandboxes": len(handles),
                   "same_judge_for_all_submissions": True, "all_run_sandboxes_deleted": True,
                   "platform_key_on_launcher_only": True, "worker_shared_mount_read_only": True,
                   "storage_after_deletion": persisted}
    write_json(output / "verification.json", verification)
    print(f"PASS: exactly two remote Sandboxes; scores {scores}; deleted both; "
          f"verified {len(versions)} archives in PyroMind Storage. Evidence: {output}", flush=True)


def exec_worker(args):
    """Execute in an existing Worker with the native SForge session configured."""
    from sforge.harness.backend.pyromind_backend import PyroMindHandle
    state = json.loads((args.run_dir / "run.json").read_text())
    backend = PyroMindBackend()
    current = backend.client.get_sandbox(state["worker_id"])
    if current.name != state["worker_name"]:
        raise RuntimeError("Worker identity mismatch")
    command = args.worker_command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ValueError("Provide a command after --")
    result = backend.exec_run_with_timeout(PyroMindHandle(current.id, current.name),
        ". /etc/sforge/worker.env && cd /workspace/project && exec " + shlex.join(command), timeout=570)
    print(result.output, end="")
    if result.exit_code:
        raise RuntimeError(f"Worker command exited {result.exit_code}")
