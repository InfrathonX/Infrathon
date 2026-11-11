"""SForge HTTP/version service inside one persistent PyroMind Judge Sandbox.

The resident evaluator is deliberately limited to the public addition pilot.
Fresh work directories and an unprivileged process are not container isolation.
Real private benchmarks must define a stronger per-submission execution policy.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile
import time

from sforge.pyromind.bootstrap import DEMO, initialize_store


def evaluate_addition(*, task_spec, archive, submission_id, log_dir, timeout=None,
                      **kwargs):
    # Harness imports load auth constants. In server mode they must happen only
    # after main has installed the configured SForge admin credential.
    from sforge.harness.grading import grade_output
    from sforge.harness.run_evaluation import (
        _canonical_submission_member_name, _filter_archive_files,
        _validate_submission_archive,
    )
    if task_spec.task_id != "addition" or kwargs.get("eval_cmd"):
        raise ValueError("Resident pilot supports only the public addition evaluator")
    names = _filter_archive_files(_validate_submission_archive(archive),
                                  task_spec.submit_paths, task_spec.submit_exclude)
    if names != ["solution.py"]:
        raise ValueError("The addition pilot requires exactly solution.py")
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "submission.tar.gz").write_bytes(archive)
    started = time.monotonic()
    timed_out = False
    with tempfile.TemporaryDirectory(prefix="sforge-candidate-") as temporary:
        work = Path(temporary)
        work.chmod(0o755)
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            member = next(m for m in bundle if
                          _canonical_submission_member_name(m.name) == "solution.py")
            (work / "solution.py").write_bytes(bundle.extractfile(member).read())
        (work / "solution.py").chmod(0o444)
        with tempfile.TemporaryFile() as output:
            process = subprocess.Popen(
                [sys.executable, "-B", str(DEMO / "judge.py"), str(work / "solution.py")],
                cwd=work, stdout=output, stderr=subprocess.STDOUT,
                env={"PATH": os.defpath, "HOME": str(work), "LANG": "C.UTF-8"},
                start_new_session=True,
                **({"user": 65534, "group": 65534, "extra_groups": []}
                   if os.geteuid() == 0 else {}),
            )
            try:
                process.wait(timeout=timeout or task_spec.judge.eval_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                # Reap descendants too, including after a normally exiting parent.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            output.seek(0)
            raw = output.read(1024 * 1024).decode(errors="replace")
    (log_dir / "test_output.txt").write_text(raw)
    report = grade_output(task_spec, raw, submission_id, timed_out,
                          time.monotonic() - started)
    if process.returncode != 0 and report.valid:
        raise RuntimeError("Evaluator reported success but exited unsuccessfully")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--shared-dir", type=Path, default=Path("/shared"))
    parser.add_argument("--admin-file", type=Path, default=Path("/run/sforge/admin.json"))
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    if os.getenv("PYROMIND_API_KEY"):
        raise RuntimeError("Platform credentials must stay on the launcher")
    # Candidate processes get a fresh environment; credentials never enter /shared.
    os.environ["SFORGE_ADMIN_SECRET"] = json.loads(args.admin_file.read_text())["secret"]
    from sforge.harness.config import SForgeConfig
    from sforge.harness.judge_server import create_app
    import uvicorn

    store = args.shared_dir / "version-store"
    if not store.exists():
        initialize_store(store, args.run_id)
    config = SForgeConfig(tasks_dir=DEMO, version_store_root=store,
                          log_dir=args.shared_dir / "logs")
    app = create_app(config, submission_evaluator=evaluate_addition)
    try:
        uvicorn.run(app, host="0.0.0.0", port=args.port, access_log=False)
    finally:
        app.state.judge_state._reaper_stop.set()


if __name__ == "__main__":
    main()
