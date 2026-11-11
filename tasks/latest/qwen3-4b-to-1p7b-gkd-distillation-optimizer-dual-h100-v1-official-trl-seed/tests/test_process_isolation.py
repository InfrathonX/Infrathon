from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from conftest import TASK_ROOT, hidden_cases_path


@pytest.fixture()
def verifier_module():
    path = TASK_ROOT / "images/judge/verifier/verify_gkd.py"
    name = "mlsysforge_test_verify_gkd"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


def test_candidate_worker_has_no_reference_import():
    source = (TASK_ROOT / "images/judge/verifier/candidate_worker.py").read_text()
    assert "hf_reference" not in source
    assert "reference_gkd" not in source


def test_controller_uses_subprocess_worker():
    source = (TASK_ROOT / "images/judge/verifier/verify_gkd.py").read_text()
    assert "subprocess.run" in source
    assert "candidate_worker.py" in source
    assert "start_new_session" in source
    assert '"extra_groups": ()' in source


def test_identity_resolution_rejects_root_and_falls_back(monkeypatch, verifier_module):
    records = {
        "agent": SimpleNamespace(pw_uid=0, pw_gid=0),
        "nobody": SimpleNamespace(pw_uid=65534, pw_gid=65534),
    }
    monkeypatch.setattr(verifier_module.pwd, "getpwnam", records.__getitem__)
    identity = verifier_module.resolve_candidate_identity()
    assert (identity.name, identity.uid, identity.gid) == ("nobody", 65534, 65534)


def test_identity_resolution_fails_without_nonroot_account(monkeypatch, verifier_module):
    monkeypatch.setattr(
        verifier_module.pwd,
        "getpwnam",
        lambda _name: SimpleNamespace(pw_uid=0, pw_gid=0),
    )
    with pytest.raises(RuntimeError, match="no non-root"):
        verifier_module.resolve_candidate_identity()


def test_launch_configuration_fails_closed_and_clears_groups(verifier_module):
    identity = verifier_module.CandidateIdentity("agent", 1000, 1000)
    with pytest.raises(PermissionError, match="must run as root"):
        verifier_module.candidate_launch_kwargs(identity, effective_uid=1002)

    options = verifier_module.candidate_launch_kwargs(identity, effective_uid=0)
    assert options == {
        "user": 1000,
        "group": 1000,
        "extra_groups": (),
        "umask": 0o077,
        "start_new_session": True,
        "close_fds": True,
    }


def test_worker_environment_is_explicit_and_drops_secrets(monkeypatch, verifier_module):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-pass")
    monkeypatch.setenv("PYTHONPATH", "/opt/gkd_judge/private_data")
    monkeypatch.setenv("HIDDEN_CASE_PATH", "/opt/gkd_judge/private_data/hidden_cases.json")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    environment = verifier_module.minimal_worker_environment()
    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert "AWS_SECRET_ACCESS_KEY" not in environment
    assert "PYTHONPATH" not in environment
    assert "HIDDEN_CASE_PATH" not in environment
    assert all("gkd_judge" not in value for value in environment.values())

    cpu_environment = verifier_module.minimal_worker_environment(expose_cuda=False)
    assert cpu_environment["CUDA_VISIBLE_DEVICES"] == ""
    assert cpu_environment["NVIDIA_VISIBLE_DEVICES"] == "none"


def test_worker_request_allowlists_fields_and_replaces_hidden_identifiers(
    monkeypatch, verifier_module
):
    hidden = {
        name: index
        for index, name in enumerate(verifier_module.WORKER_CASE_FIELDS, start=1)
    }
    hidden.update(case_id="hidden_name", seed=104729, private_note="do not forward")
    monkeypatch.setattr(verifier_module.secrets, "randbits", lambda _bits: 777)
    request = verifier_module.sanitized_worker_case(hidden, 3)
    assert request["case_id"] == "request_3"
    assert request["seed"] == 777
    assert "private_note" not in request
    assert "hidden_name" not in request.values()
    assert 104729 not in request.values()


def test_optional_worker_fields_do_not_break_cases_that_omit_them(
    monkeypatch, verifier_module
):
    """An optional field must not make the sanitizer strict about it.

    The allowlist indexes required fields directly, so listing an optional one
    there raises KeyError for every case that predates it -- which fails the
    whole verification at the correctness stage, before any GPU work.
    """

    monkeypatch.setattr(verifier_module.secrets, "randbits", lambda _bits: 11)
    required = {
        name: index
        for index, name in enumerate(verifier_module.WORKER_CASE_FIELDS, start=1)
    }
    without = verifier_module.sanitized_worker_case(dict(required), 0)
    for name in verifier_module.OPTIONAL_WORKER_CASE_FIELDS:
        assert name not in without

    for name in verifier_module.OPTIONAL_WORKER_CASE_FIELDS:
        with_optional = verifier_module.sanitized_worker_case(
            {**required, name: 17}, 1
        )
        assert with_optional[name] == 17


def test_reward_requires_every_formal_e2e_gate(verifier_module):
    detail = {
        "source_ok": True,
        "correctness_ok": True,
        "protocol_ok": True,
        "e2e_formal": {
            "status": "DRY_RUN",
            "schema_ok": True,
            "execution_ok": False,
            "loss_gate_ok": False,
            "memory_gate_ok": False,
            "correctness_evidence_ok": False,
            "formal_ready": False,
            "formal_score_enabled": False,
            "score": 123.0,
        },
    }
    dry_reward = verifier_module.build_reward(detail)
    assert dry_reward["reward"] == 0.0
    assert dry_reward["e2e_schema_ok"] == 1.0
    assert dry_reward["e2e_execution_ok"] == 0.0

    detail["e2e_formal"].update(
        status="PASS",
        execution_ok=True,
        loss_gate_ok=True,
        memory_gate_ok=True,
        formal_ready=True,
        formal_score_enabled=True,
    )
    assert verifier_module.build_reward(detail)["reward"] == 0.0

    detail["e2e_formal"]["correctness_evidence_ok"] = True
    formal_reward = verifier_module.build_reward(detail)
    assert formal_reward["reward"] == 123.0
    assert all(value == 1.0 for key, value in formal_reward.items() if key != "reward")

    detail["e2e_formal"]["status"] = "GATE_FAILED"
    assert verifier_module.build_reward(detail)["reward"] == 0.0


def test_access_check_has_a_real_denial_path(tmp_path, verifier_module):
    protected = tmp_path / "protected.json"
    protected.write_text('{"hidden": true}\n', encoding="utf-8")
    protected.chmod(0o000)
    identity = verifier_module.CandidateIdentity("current", os.getuid(), os.getgid())
    with pytest.raises(PermissionError, match="cannot access"):
        verifier_module.assert_candidate_access(
            protected, identity, read=True, regular_file=True
        )

    protected.chmod(0o400)
    verifier_module.assert_candidate_access(protected, identity, read=True, regular_file=True)


@pytest.mark.skipif(os.geteuid() != 0, reason="actual UID/GID transition requires root")
def test_real_launch_drops_uid_gid_groups_and_starts_session(verifier_module):
    identity = verifier_module.resolve_candidate_identity()
    code = (
        "import json,os;"
        "print(json.dumps({'uid':os.geteuid(),'gid':os.getegid(),"
        "'groups':os.getgroups(),'new_session':os.getsid(0)==os.getpid()}))"
    )
    process = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        check=True,
        cwd=Path("/tmp"),
        env=verifier_module.minimal_worker_environment(),
        **verifier_module.candidate_launch_kwargs(identity),
    )
    result = json.loads(process.stdout)
    assert result == {
        "uid": identity.uid,
        "gid": identity.gid,
        "groups": [],
        "new_session": True,
    }


def test_judge_image_separates_worker_from_root_only_material():
    dockerfile = (TASK_ROOT / "images/judge/Dockerfile").read_text()
    assert "/opt/gkd_candidate_worker/candidate_worker.py" in dockerfile
    assert "/opt/gkd_candidate_worker/e2e_rank_bootstrap.py" in dockerfile
    assert "chmod 700 /opt/gkd_judge" in dockerfile
    assert "chmod 700 /opt/gkd_judge /opt/gkd_judge/verifier" in dockerfile
    # private_data/ and its root-only modes come from the host-built holdout layer.
    assert "COPY private_data/" not in dockerfile
    holdout_layer = (TASK_ROOT / "images/judge/Dockerfile.private").read_text()
    assert "chmod 700 /opt/gkd_judge/private_data" in holdout_layer
    assert "verify_gkd.py --verify-protected-files" in holdout_layer
    assert "install -d -o root -g root -m 0700 /logs/verifier" in dockerfile
    assert "find /app/workdir/gkd_lab/runtime -type d -exec chmod 555" in dockerfile


def test_reference_and_worker_build_identical_inputs_for_every_hidden_case(
    verifier_module,
):
    """The three input builders must agree tensor-for-tensor.

    `public_eval.py`, `candidate_worker.py`, and `verify_gkd.py` each construct
    the case inputs independently. If they drift, the reference and the
    candidate are scored on different data and every downstream comparison is
    meaningless -- so compare the tensors, not the source text.
    """

    import importlib.util
    import json

    import torch

    from conftest import TASK_ROOT, hidden_cases_path

    def _load(relative: str, name: str):
        spec = importlib.util.spec_from_file_location(name, TASK_ROOT / relative)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    worker = _load("images/judge/verifier/candidate_worker.py", "gkd_worker_inputs")
    public = _load("images/work/public_tests/public_eval.py", "gkd_public_inputs")

    hidden = json.loads(
        hidden_cases_path().read_text(encoding="utf-8")
    )
    visible = json.loads(
        (TASK_ROOT / "images/work/public_tests/public_cases.json").read_text(
            encoding="utf-8"
        )
    )
    cases = hidden["cases"] + visible["correctness"]
    assert any("mask_ignored_count" in case for case in cases)

    for case in cases:
        expected = verifier_module.make_inputs(case)
        from_worker = worker.make_inputs(case)
        from_public = public.make_inputs(case, torch.device("cpu"), torch.float32)
        assert len(expected) == len(from_worker) == len(from_public)
        for want, got_worker, got_public in zip(
            expected, from_worker, from_public, strict=True
        ):
            if want is None:
                assert got_worker is None and got_public is None
                continue
            assert torch.equal(want, got_worker), case["case_id"]
            assert torch.equal(want, got_public), case["case_id"]
