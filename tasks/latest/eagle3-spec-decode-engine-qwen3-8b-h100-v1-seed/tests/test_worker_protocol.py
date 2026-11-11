"""Candidate worker privilege and protocol ordering tests."""

from __future__ import annotations

from conftest import ROOT


def test_worker_binds_runtime_then_drops_then_imports() -> None:
    text = (ROOT / "images/judge/verifier/candidate_worker.py").read_text()
    binding = text.index("bound_validate_prompts")
    dropping = text.index("drop_privileges()", binding)
    importing = text.index("importlib.import_module", dropping)
    assert binding < dropping < importing


def test_privilege_drop_clears_supplementary_groups() -> None:
    text = (ROOT / "images/judge/verifier/candidate_worker.py").read_text()
    assert "os.setgroups([])" in text
    assert text.index("os.setgroups([])") < text.index("os.setgid") < text.index("os.setuid")


def test_worker_input_and_output_are_bounded_and_atomic() -> None:
    text = (ROOT / "images/judge/verifier/candidate_worker.py").read_text()
    assert "MAX_REQUEST_BYTES + 1" in text
    assert "os.replace(temporary, path)" in text


def test_worker_checks_candidate_threads_at_the_device_idle_boundary() -> None:
    """A living candidate helper thread must make the idle gate fail."""
    text = (ROOT / "images/judge/verifier/candidate_worker.py").read_text()
    assert "trusted_thread_ids" in text
    assert "candidate_threads_idle()" in text
    assert '"device_idle": threads_idle' in text
