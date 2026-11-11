"""Author-only orchestration API; no automatic Docker/GPU scheduling.

Caller supplies a root-owned admitted snapshot, trusted root model preparer,
actual source/image/GPU lease attestations and a private prompt partition.
One root model remains loaded, workers are separate permanently dropped
processes, and only one <=8 GiB five-file bundle exists at a time. This API has
CPU tests only. It must not be called a qualified Judge before actual GPU and
source/asset/privacy integration tests.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import selectors
import time
import uuid
from single_prompt_archive_v3 import (canonical_bytes, canonical_sha, open_bundle,
    stable_stat, aggregate_prompt_reports, directory_identity)
from prepare_single_prompt_v3 import write_private_json


def release_bundle(root, bundle, identities):
    """Only release this freshly generated exact five-file directory."""
    root = Path(root)
    if directory_identity(root.lstat()) != bundle.get("scratch_directory_identity"):
        raise ValueError("scratch directory identity changed before release")
    expected = {"bundle.json", *(f["file"] for f in bundle["fixtures"].values())}
    if set(p.name for p in root.iterdir()) != expected:
        raise ValueError("scratch membership changed before release")
    for fid, f in bundle["fixtures"].items():
        if stable_stat((root / f["file"]).lstat()) != identities[fid]:
            raise ValueError("scratch file identity changed before release")
    if (root / "bundle.json").read_bytes() != canonical_bytes(bundle):
        raise ValueError("scratch bundle metadata changed before release")
    # All checks complete before the first unlink; no other tree is selected.
    for f in bundle["fixtures"].values(): (root / f["file"]).unlink()
    (root / "bundle.json").unlink()
    root.rmdir()


def run_stream(preparer, prompts, *, plan, candidate, candidate_sha256, policy,
               policy_sha256, archive, trusted_root, scratch_parent, output,
               worker_timeout=600, purpose="private-judge", nonce=None):
    from single_prompt_archive_v3 import validate_purpose
    validate_purpose(purpose, plan["partition"], os.geteuid())
    required_uid = 0 if purpose == "private-judge" else 1000
    candidate, trusted_root = Path(candidate), Path(trusted_root)
    scratch_parent, output = Path(scratch_parent), Path(output)
    if not scratch_parent.is_absolute() or not output.is_absolute() or output.exists():
        raise ValueError("explicit absolute fresh output and scratch parent required")
    if hashlib.sha256(candidate.read_bytes()).hexdigest() != candidate_sha256:
        raise ValueError("admitted snapshot identity differs")
    info = candidate.lstat()
    if info.st_uid != required_uid or info.st_nlink != 1 or info.st_mode & 0o222:
        raise ValueError("candidate must be a root-owned read-only single-link snapshot")
    output.mkdir(mode=0o700)
    scratch_parent.mkdir(mode=0o700, exist_ok=True)
    if scratch_parent.is_symlink() or scratch_parent.stat().st_mode & 0o077 or scratch_parent.stat().st_uid != required_uid:
        raise ValueError("scratch parent must be private")
    if any(scratch_parent.iterdir()):
        raise ValueError("scratch parent has retained files; resolve that run before preparing another bundle")
    expected_ids = {f["prompt_id"] for f in plan["fixtures"].values()}
    if len(prompts) != 36 or {p["id"] for p in prompts} != expected_ids:
        raise ValueError("full fixed performance prompt set differs")
    reports = []
    nonce = (secrets.randbits(128) or 1) if nonce is None else nonce
    if type(nonce) is not int or not 0 < nonce < 2**128:
        raise ValueError("invalid evaluation row nonce")
    write_private_json(output / "evaluation_nonce.json", {"nonce": nonce,
        "protocol": "private-query-row-nonce-v3", "plan_sha256": canonical_sha(plan),
        "candidate_sha256": candidate_sha256})
    for index, prompt in enumerate(prompts):
        scratch = scratch_parent / ("prompt-" + uuid.uuid4().hex)
        bundle = preparer.capture(prompt, scratch)
        handles, identities = open_bundle(scratch, bundle, plan, archive)
        process = None
        try:
            transport = {"plan": plan, "bundle": bundle, "policy": policy,
                "policy_sha256": policy_sha256, "candidate_sha256": candidate_sha256,
                "nonce": nonce, "fds": {fid: {"fd": h.fd, "header": h.header,
                "payload_offset": h.payload_offset, "identity": identities[fid]}
                for fid,h in handles.items()}}
            env = dict(os.environ)
            # Reuse only shape/code compilation artifacts across child workers.
            env.pop("PYTHONPATH", None)
            process = subprocess.Popen([sys.executable, "-I",
                str(Path(__file__).with_name("run_single_prompt_worker_v3.py")),
                "--trusted-root", str(trusted_root), "--candidate", str(candidate),
                "--purpose", purpose],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                pass_fds=tuple(h.fd for h in handles.values()), start_new_session=True, env=env)
            stdout, stderr = communicate_bounded(process, canonical_bytes(transport), worker_timeout)
            if process.returncode != 0:
                raise ValueError("single-prompt candidate worker failed closed")
            if len(stdout) > 1024**2:
                raise ValueError("worker report exceeded bound")
            report = archive.strict_json(stdout)
            if (report.get("bundle_sha256") != canonical_sha(bundle)
                    or report.get("execution_uid") != 1000
                    or report.get("purpose") != purpose
                    or report.get("candidate_privilege_drop") is not (purpose == "private-judge")):
                raise ValueError("worker report bundle differs")
            reports.append(report)
            write_private_json(output / f"prompt-{index:02d}-bundle.json", bundle)
            write_private_json(output / f"prompt-{index:02d}-report.json", report)
        finally:
            if process is not None:
                try: os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
            for handle in handles.values(): handle.close()
        # Release only after exact completed worker, authenticated result, and
        # parent FD closure. Failures retain the current bounded bundle.
        release_bundle(scratch, bundle, identities)
        if any(trial.get("numeric_passed") is not True for trial in report.get("trials", [])):
            raise ValueError("single-prompt numerical fidelity gate failed")
    result = aggregate_prompt_reports(plan, reports, candidate_sha256=candidate_sha256,
                                      numerical_policy_sha256=policy_sha256, policy=policy)
    result.update(purpose=purpose, execution_uid=required_uid,
                  candidate_privilege_drop=purpose == "private-judge")
    write_private_json(output / "kernel-stream-result.json", result)
    return result


def communicate_bounded(process, payload, timeout, *, stdout_limit=1024**2, stderr_limit=65536):
    """Drain trusted-owned child pipes with limits enforced while reading."""
    if timeout <= 0 or len(payload) > 1024**2:
        raise ValueError("invalid child deadline/transport bound")
    selector = selectors.DefaultSelector()
    output = {"stdout": bytearray(), "stderr": bytearray()}
    offset = 0; deadline = time.monotonic() + timeout
    try:
        for name in ("stdin", "stdout", "stderr"):
            stream = getattr(process, name); os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_WRITE if name == "stdin" else selectors.EVENT_READ, name)
        while selector.get_map():
            if time.monotonic() >= deadline:
                raise ValueError("single-prompt worker timed out")
            for key, _ in selector.select(min(.1, max(0., deadline-time.monotonic()))):
                name, stream = key.data, key.fileobj
                if name == "stdin":
                    try: written = os.write(stream.fileno(), payload[offset:offset+65536])
                    except BrokenPipeError: written = 0; offset = len(payload)
                    offset += written
                    if offset >= len(payload): selector.unregister(stream); stream.close()
                else:
                    data = os.read(stream.fileno(), 65536)
                    if not data: selector.unregister(stream); stream.close(); continue
                    limit = stdout_limit if name == "stdout" else stderr_limit
                    if len(output[name]) + len(data) > limit:
                        raise ValueError("worker output exceeded its live byte bound")
                    output[name].extend(data)
        process.wait(timeout=max(.001, deadline-time.monotonic()))
        return bytes(output["stdout"]), bytes(output["stderr"])
    except BaseException:
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        process.wait(timeout=5)
        raise
    finally:
        selector.close()
        for name in ("stdin", "stdout", "stderr"):
            stream = getattr(process, name)
            if not stream.closed: stream.close()
