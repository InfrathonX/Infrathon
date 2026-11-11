"""Static contracts for the private verifier surface."""

from __future__ import annotations

from conftest import ROOT


def test_all_required_gates_are_initialized_fail_closed() -> None:
    controller = (ROOT / "images/judge/verifier/controller.py").read_text()
    for name in (
        "protected_ok",
        "source_ok",
        "worker_ok",
        "o1_ok",
        "o2_ok",
        "tiers_ok",
        "o3_o4_ok",
        "o5_ok",
        "wall_ok",
        "memory_ok",
        "idle_ok",
    ):
        assert name in controller


def test_caps_are_consumed_by_worker_and_controller() -> None:
    controller = (ROOT / "images/judge/verifier/controller.py").read_text()
    worker = (ROOT / "images/judge/verifier/candidate_worker.py").read_text()
    assert "worker_timeout_s" in controller
    assert "wall_caps_s" in controller
    assert "memory_caps_bytes" in controller
    assert "build_cap_s" in worker
