#!/usr/bin/env python3
"""Private verifier for the frozen two-H100 Qwen3 GKD distillation workload.

The verifier first qualifies the isolated objective contract, then delegates
formal end-to-end correctness, memory, and throughput evaluation to the
Judge-owned paired controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pwd
import re
import secrets
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


JUDGE_ROOT = Path("/opt/gkd_judge")
LOCAL_ROOT = Path(__file__).resolve().parents[1]
HIDDEN_PATH = JUDGE_ROOT / "private_data/hidden_cases.json"
WORKER_PATH = Path("/opt/gkd_candidate_worker/candidate_worker.py")
PROTECTED_MANIFEST_PATH = JUDGE_ROOT / "protected_hashes.txt"
PROTECTED_HEADER = "# mlsysforge-protected-files-v1"
PROTECTED_COLUMNS = "# sha256  relative-path"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
WORKER_CASE_FIELDS = (
    "batch",
    "sequence_length",
    "student_hidden_size",
    "teacher_hidden_size",
    "vocab_size",
    "beta",
    "temperature",
    "ignore_index",
    "bias",
    "mask_stride",
)
# Forwarded only when the case defines it. Required fields stay strict above so
# a malformed case is still a verifier failure rather than a silent default.
OPTIONAL_WORKER_CASE_FIELDS = ("mask_ignored_count",)


@dataclass(frozen=True)
class CandidateIdentity:
    """The non-root account used for untrusted candidate execution."""

    name: str
    uid: int
    gid: int


class ProtectedIntegrityError(RuntimeError):
    """Raised when a trusted Judge/runtime file does not match its manifest."""


def _protected_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_protected_manifest(manifest_path: Path) -> list[tuple[str, str]]:
    """Parse the tiny canonical manifest without accepting loose syntax."""

    try:
        info = manifest_path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ProtectedIntegrityError("protected manifest is not a regular file")
        lines = manifest_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ProtectedIntegrityError("protected manifest is unreadable") from exc
    if len(lines) < 3 or lines[0] != PROTECTED_HEADER or lines[1] != PROTECTED_COLUMNS:
        raise ProtectedIntegrityError("protected manifest is not canonical")
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in lines[2:]:
        fields = line.split("  ")
        if len(fields) != 2 or not _SHA256_RE.fullmatch(fields[0]):
            raise ProtectedIntegrityError("protected manifest has an invalid entry")
        digest, relative = fields
        path = Path(relative)
        if (
            not relative
            or path.is_absolute()
            or "/" not in relative
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.as_posix() != relative
        ):
            raise ProtectedIntegrityError("protected manifest has an unsafe path")
        if relative in seen:
            raise ProtectedIntegrityError("protected manifest has a duplicate path")
        seen.add(relative)
        entries.append((digest, relative))
    if not entries:
        raise ProtectedIntegrityError("protected manifest has no entries")
    return entries


def verify_protected_files(
    manifest_path: Path = PROTECTED_MANIFEST_PATH, *, container_root: Path = Path("/")
) -> None:
    """Fail closed before candidate admission/import if trusted files changed.

    Paths are manifest-relative to the container root.  ``lstat`` is used so
    an attacker cannot replace a protected target with a symlink to a matching
    or outside-tree file.  The manifest itself is generated/checked from the
    full task tree before build and is copied root-owned into the Judge image.
    """

    for expected, relative in _parse_protected_manifest(manifest_path):
        target = container_root / relative
        try:
            info = target.lstat()
        except OSError as exc:
            raise ProtectedIntegrityError("protected target is unavailable") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise ProtectedIntegrityError("protected target is not a regular file")
        try:
            actual = _protected_digest(target)
        except OSError as exc:
            raise ProtectedIntegrityError("protected target cannot be hashed") from exc
        if not secrets.compare_digest(actual, expected):
            raise ProtectedIntegrityError("protected target digest mismatch")


def resolve_candidate_identity() -> CandidateIdentity:
    """Resolve a dedicated non-root identity, preferring the Work user."""

    for name in ("agent", "nobody"):
        try:
            record = pwd.getpwnam(name)
        except KeyError:
            continue
        if record.pw_uid > 0 and record.pw_gid > 0:
            return CandidateIdentity(name=name, uid=record.pw_uid, gid=record.pw_gid)
    raise RuntimeError("no non-root candidate account is available")


def candidate_launch_kwargs(
    identity: CandidateIdentity, *, effective_uid: int | None = None
) -> dict[str, Any]:
    """Return fail-closed Popen options for a privilege-dropping launch."""

    caller_uid = os.geteuid() if effective_uid is None else effective_uid
    if caller_uid != 0:
        raise PermissionError("the private verifier must run as root before dropping privileges")
    if identity.uid <= 0 or identity.gid <= 0:
        raise ValueError("candidate UID and GID must both be non-root")
    return {
        "user": identity.uid,
        "group": identity.gid,
        "extra_groups": (),
        "umask": 0o077,
        "start_new_session": True,
        "close_fds": True,
    }


def minimal_worker_environment(*, expose_cuda: bool = True) -> dict[str, str]:
    """Build an explicit environment without verifier or host secrets."""

    python_bin = str(Path(sys.executable).resolve().parent)
    environment = {
        "PATH": f"{python_bin}:/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "TRITON_CACHE_DIR": "/tmp/gkd-candidate-triton-cache",
    }
    # These are execution plumbing rather than credentials.  They are copied
    # only when present because CUDA-enabled Judge images may require them.
    for name in ("LD_LIBRARY_PATH",):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    if expose_cuda:
        for name in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
            value = os.environ.get(name)
            if value:
                environment[name] = value
    else:
        # Unit contract checks must remain CPU-only even when the Judge
        # container itself is attached to the formal GPU pair.
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["NVIDIA_VISIBLE_DEVICES"] = "none"
    return environment


def sanitized_worker_case(hidden_case: dict[str, Any], ordinal: int) -> dict[str, Any]:
    """Copy only fields required by the ABI and replace private identifiers."""

    case = {name: hidden_case[name] for name in WORKER_CASE_FIELDS}
    case.update(
        {
            name: hidden_case[name]
            for name in OPTIONAL_WORKER_CASE_FIELDS
            if name in hidden_case
        }
    )
    case["case_id"] = f"request_{ordinal}"
    case["seed"] = secrets.randbits(63)
    return case


def _permission_bits(path_stat: os.stat_result, identity: CandidateIdentity) -> int:
    if path_stat.st_uid == identity.uid:
        return (path_stat.st_mode >> 6) & 0o7
    if path_stat.st_gid == identity.gid:
        return (path_stat.st_mode >> 3) & 0o7
    return path_stat.st_mode & 0o7


def assert_candidate_access(
    path: Path,
    identity: CandidateIdentity,
    *,
    read: bool = False,
    execute: bool = False,
    regular_file: bool = False,
) -> None:
    """Check access using the identity's primary group and no supplements."""

    path = path.absolute()
    current = Path(path.anchor)
    for component in path.parts[1:-1]:
        current /= component
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise PermissionError("candidate path has a linked or non-directory ancestor")
        if not (_permission_bits(info, identity) & 0o1):
            raise PermissionError("candidate cannot traverse a required path")
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode):
        raise PermissionError("candidate-required path must not be a symlink")
    if regular_file and not stat.S_ISREG(info.st_mode):
        raise PermissionError("candidate-required path must be a regular file")
    required = (0o4 if read else 0) | (0o1 if execute else 0)
    if _permission_bits(info, identity) & required != required:
        raise PermissionError("candidate account cannot access a required file")


def freeze_submission(lab_root: Path) -> None:
    """Make the already-admitted submission immutable to the worker account."""

    if os.geteuid() != 0:
        raise PermissionError("only root may freeze the admitted submission")
    targets = [lab_root / "FINAL_VERSION", lab_root / "notes.md"]
    targets.extend((lab_root / "solutions").rglob("*"))
    for path in sorted(targets, key=lambda item: len(item.parts), reverse=True):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise PermissionError("submission changed after admission")
        os.chown(path, 0, 0, follow_symlinks=False)
        os.chmod(path, 0o555 if stat.S_ISDIR(info.st_mode) else 0o444)
    os.chown(lab_root / "solutions", 0, 0, follow_symlinks=False)
    os.chmod(lab_root / "solutions", 0o555)
    # A low-privilege cohort must not rename the protected runtime/solutions
    # entries or plant top-level Python startup hooks between paired launches.
    root_info = lab_root.lstat()
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise PermissionError("lab root must be a regular directory")
    os.chown(lab_root, 0, 0, follow_symlinks=False)
    os.chmod(lab_root, 0o555)


def validate_candidate_files(
    lab_root: Path, worker_path: Path, version: int, identity: CandidateIdentity
) -> None:
    """Fail before launch unless all and only required code is readable."""

    assert_candidate_access(worker_path, identity, read=True, execute=True, regular_file=True)
    for relative in (
        "runtime/__init__.py",
        "runtime/_source_policy_impl.py",
        "runtime/api.py",
        "runtime/distributed_probe.py",
        "runtime/hf_reference.py",
        "runtime/loader.py",
        "runtime/source_policy.py",
        "runtime/train_step.py",
        f"solutions/objective_v{version:03d}/__init__.py",
        f"solutions/objective_v{version:03d}/candidate.py",
    ):
        assert_candidate_access(lab_root / relative, identity, read=True, regular_file=True)


def secure_log_dir(log_dir: Path) -> None:
    """Keep private diagnostics inaccessible to the candidate UID."""

    if os.geteuid() != 0:
        raise PermissionError("private verifier logs require root ownership")
    if log_dir.is_symlink():
        raise PermissionError("private verifier log directory must not be a symlink")
    log_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    info = log_dir.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise PermissionError("private verifier log path is not a directory")
    os.chown(log_dir, 0, 0, follow_symlinks=False)
    os.chmod(log_dir, 0o700)


def locate_lab(app_dir: Path) -> Path:
    for candidate in (
        app_dir / "workdir/gkd_lab",
        app_dir / "environment/workdir/gkd_lab",
        app_dir / "app/workdir/gkd_lab",
        app_dir,
    ):
        if (candidate / "solutions").is_dir() and (candidate / "FINAL_VERSION").is_file():
            return candidate.resolve()
    raise ValueError(f"cannot locate gkd_lab below {app_dir}")


def make_inputs(case: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(int(case["seed"]))
    batch = int(case["batch"])
    sequence = int(case["sequence_length"])
    student_hidden_size = int(case["student_hidden_size"])
    teacher_hidden_size = int(case["teacher_hidden_size"])
    vocab = int(case["vocab_size"])
    student_weight = (
        torch.randn(vocab, student_hidden_size, generator=generator, dtype=torch.float32) * 0.07
    )
    student_hidden = (
        torch.randn(batch, sequence, student_hidden_size, generator=generator, dtype=torch.float32)
        * 0.11
    )
    teacher_weight = (
        torch.randn(vocab, teacher_hidden_size, generator=generator, dtype=torch.float32) * 0.07
    )
    teacher_hidden = (
        torch.randn(batch, sequence, teacher_hidden_size, generator=generator, dtype=torch.float32)
        * 0.11
    )
    labels = torch.randint(vocab, (batch, sequence), generator=generator)
    ignored_count = case.get("mask_ignored_count")
    if ignored_count is None:
        labels.reshape(-1)[int(case["mask_stride"]) - 1 :: int(case["mask_stride"])] = int(case["ignore_index"])
    else:
        # A fixed stride is a pattern a candidate can special-case. Real
        # completion-only labels mask whatever the prompt happens to cover, so
        # this case ignores an irregular set of positions instead. Drawn from
        # the same seeded generator, so the case stays exactly reproducible.
        shuffled = torch.randperm(batch * sequence, generator=generator, device=labels.device)
        labels.reshape(-1)[shuffled[: int(ignored_count)]] = int(case["ignore_index"])
    labels[:, 0] = torch.arange(batch) % vocab
    student_bias = torch.randn(vocab, generator=generator) * 0.03 if case["bias"] else None
    teacher_bias = torch.randn(vocab, generator=generator) * 0.03 if case["bias"] else None
    return (
        student_weight,
        student_hidden,
        teacher_weight,
        teacher_hidden,
        labels,
        student_bias,
        teacher_bias,
    )


def execute_reference(case: dict[str, Any], lab_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(lab_root))
    from runtime.api import ObjectiveContract
    from runtime.hf_reference import reference_gkd

    contract = ObjectiveContract(
        beta=float(case["beta"]),
        temperature=float(case["temperature"]),
        ignore_index=int(case["ignore_index"]),
    )
    (
        student_weight,
        student_hidden,
        teacher_weight,
        teacher_hidden,
        labels,
        student_bias,
        teacher_bias,
    ) = make_inputs(case)
    student_weight.requires_grad_(True)
    student_hidden.requires_grad_(True)
    if student_bias is not None:
        student_bias.requires_grad_(True)
    result = reference_gkd(
        student_weight,
        student_hidden,
        teacher_weight,
        teacher_hidden,
        labels,
        student_bias,
        teacher_bias,
        contract=contract,
    )
    result.loss.backward()
    return {
        "loss": result.loss.detach(),
        "aux": [item.detach() for item in result.aux],
        "student_weight_grad": student_weight.grad.detach(),
        "student_hidden_grad": student_hidden.grad.detach(),
        "student_bias_grad": None if student_bias is None else student_bias.grad.detach(),
    }


def parse_worker(stdout: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.startswith("EDGE_RESULT ")]
    if len(lines) != 1:
        raise ValueError("candidate worker did not return exactly one result")
    payload = json.loads(lines[0][len("EDGE_RESULT ") :])
    if not isinstance(payload, dict):
        raise TypeError("candidate worker result must be an object")
    return payload


def compare(actual: Any, expected: torch.Tensor | None, label: str) -> None:
    if expected is None:
        if actual is not None:
            raise AssertionError(f"{label} should be absent")
        return
    observed = torch.tensor(actual, dtype=expected.dtype)
    torch.testing.assert_close(observed, expected.cpu(), rtol=5e-4, atol=1e-5, msg=label)


def build_reward(detail: dict[str, Any]) -> dict[str, float]:
    """Translate private controller state into the fail-closed reward ABI."""

    formal = detail.get("e2e_formal")
    if not isinstance(formal, dict):
        formal = {}
    gates = {
        "source_ok": float(detail.get("source_ok") is True),
        "correctness_ok": float(detail.get("correctness_ok") is True),
        "protocol_ok": float(detail.get("protocol_ok") is True),
        "e2e_schema_ok": float(formal.get("schema_ok") is True),
        "e2e_execution_ok": float(formal.get("execution_ok") is True),
        "loss_gate_ok": float(formal.get("loss_gate_ok") is True),
        "memory_gate_ok": float(formal.get("memory_gate_ok") is True),
        "correctness_evidence_ok": float(
            formal.get("correctness_evidence_ok") is True
        ),
        "formal_ready": float(formal.get("formal_ready") is True),
        "formal_score_enabled": float(formal.get("formal_score_enabled") is True),
    }
    formal_score = formal.get("score")
    score_is_valid = (
        not isinstance(formal_score, bool)
        and isinstance(formal_score, (int, float))
        and math.isfinite(float(formal_score))
        and float(formal_score) > 0.0
        and formal.get("status") == "PASS"
        and all(value == 1.0 for value in gates.values())
    )
    return {"reward": float(formal_score) if score_is_valid else 0.0, **gates}


def write_results(log_dir: Path, detail: dict[str, Any]) -> None:
    secure_log_dir(log_dir)
    private_path = log_dir / "gkd_private.json"
    private_path.write_text(
        json.dumps(detail, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.chown(private_path, 0, 0, follow_symlinks=False)
    os.chmod(private_path, 0o600)
    reward = build_reward(detail)
    reward_path = log_dir / "reward.json"
    reward_path.write_text(
        json.dumps(reward, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.chown(reward_path, 0, 0, follow_symlinks=False)
    os.chmod(reward_path, 0o600)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app-dir", type=Path, default=Path("/app"))
    parser.add_argument("--log-dir", type=Path, default=Path("/logs/verifier"))
    parser.add_argument(
        "--verify-protected-files",
        action="store_true",
        help="verify the immutable Judge/runtime file manifest and exit",
    )
    parser.add_argument("--protected-manifest", type=Path, default=PROTECTED_MANIFEST_PATH)
    parser.add_argument("--container-root", type=Path, default=Path("/"))
    args = parser.parse_args()
    if args.verify_protected_files:
        try:
            verify_protected_files(args.protected_manifest, container_root=args.container_root)
        except ProtectedIntegrityError:
            return 1
        return 0
    detail: dict[str, Any] = {
        "schema_version": 1,
        "source_ok": False,
        "correctness_ok": False,
        "protocol_ok": False,
        "formal_ready": False,
        "formal_score_enabled": False,
        "stage": "formal_dual_h100_evaluation",
    }
    try:
        secure_log_dir(args.log_dir)
        # This precedes locating/admitting the submitted lab and any candidate
        # import.  A stale, missing, substituted, or malformed trusted file is
        # a verifier failure, never a candidate-visible fallback.
        verify_protected_files(args.protected_manifest, container_root=args.container_root)
        identity = resolve_candidate_identity()
        launch_kwargs = candidate_launch_kwargs(identity)
        lab_root = locate_lab(args.app_dir)
        sys.path.insert(0, str(lab_root))
        from runtime.source_policy import admit

        policy = admit(lab_root)
        freeze_submission(lab_root)
        # Re-admit after the ownership/mode transition so the executed source
        # is exactly the source that passed policy.
        policy = admit(lab_root)
        detail["source_ok"] = True
        hidden_path = HIDDEN_PATH if HIDDEN_PATH.is_file() else LOCAL_ROOT / "private_data/hidden_cases.json"
        worker_path = WORKER_PATH if WORKER_PATH.is_file() else LOCAL_ROOT / "verifier/candidate_worker.py"
        validate_candidate_files(
            lab_root,
            worker_path,
            int(policy["final_version"]),
            identity,
        )
        manifest = json.loads(hidden_path.read_text(encoding="utf-8"))
        detail["formal_ready"] = manifest.get("formal_ready") is True
        detail["formal_score_enabled"] = (
            manifest.get("formal_score_enabled") is True
        )
        rows = []
        for ordinal, hidden_case in enumerate(manifest["cases"]):
            # The worker needs the objective shape/contract to construct its
            # input tensors.  Do not forward hidden case IDs, frozen seeds,
            # manifest paths, or readiness metadata.
            case = sanitized_worker_case(hidden_case, ordinal)
            expected = execute_reference(case, lab_root)
            process = subprocess.run(
                [sys.executable, str(worker_path), "--lab-root", str(lab_root)],
                input=json.dumps({"version": policy["final_version"], "case": case}) + "\n",
                text=True,
                capture_output=True,
                timeout=45,
                check=False,
                env=minimal_worker_environment(expose_cuda=False),
                cwd=lab_root,
                **launch_kwargs,
            )
            if process.returncode != 0:
                raise RuntimeError("candidate worker failed")
            actual = parse_worker(process.stdout)
            if actual.get("case_id") != case["case_id"]:
                raise ValueError("candidate worker returned a mismatched request identifier")
            compare(actual["loss"], expected["loss"], "loss")
            if len(actual["aux"]) != len(expected["aux"]):
                raise AssertionError("auxiliary output count differs")
            for index, (got, want) in enumerate(zip(actual["aux"], expected["aux"], strict=True)):
                compare(got, want, f"aux[{index}]")
            for name in ("student_weight_grad", "student_hidden_grad", "student_bias_grad"):
                compare(actual[name], expected[name], name)
            rows.append({"case_id": hidden_case["case_id"], "status": "PASS"})
        detail.update(
            correctness_ok=True,
            protocol_ok=True,
            rows=rows,
            final_version=policy["final_version"],
            readiness_reason=manifest["reason"],
        )
        verifier_dir = LOCAL_ROOT / "verifier"
        if str(verifier_dir) not in sys.path:
            sys.path.insert(0, str(verifier_dir))
        from e2e_formal_controller import run_e2e_controller

        worker_environment = minimal_worker_environment()
        formal_report = run_e2e_controller(
            manifest,
            lab_root=lab_root,
            model_path=Path(
                os.environ.get("GKD_STUDENT_MODEL_PATH", "/models/Qwen3-1.7B")
            ),
            teacher_model_path=Path(
                os.environ.get("GKD_TEACHER_MODEL_PATH", "/models/Qwen3-4B")
            ),
            data_root=Path(
                os.environ.get("GKD_DATA_PATH", "/datasets/tldr-gkd")
            ),
            log_dir=args.log_dir / "formal_e2e",
            candidate_version=int(policy["final_version"]),
            candidate_environment=worker_environment,
            reference_environment=worker_environment,
            candidate_launch_options=launch_kwargs,
        )
        detail["e2e_formal"] = formal_report
        if formal_report.get("status") != "PASS":
            detail["error"] = "formal e2e controller failed closed"
            write_results(args.log_dir, detail)
            return 1
        write_results(args.log_dir, detail)
        return 0
    except Exception as exc:
        # Keep exception messages (which can contain paths or input fragments)
        # out of all structured results, including root-only diagnostics.
        detail["error"] = "verification failed"
        detail["error_type"] = type(exc).__name__
        try:
            write_results(args.log_dir, detail)
        except Exception:
            # If secure logging itself is unavailable, fail without printing
            # verifier paths, hidden metadata, or candidate stderr.
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
