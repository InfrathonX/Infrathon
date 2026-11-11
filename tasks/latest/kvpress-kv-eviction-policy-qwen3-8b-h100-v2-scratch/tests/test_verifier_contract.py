from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from conftest import TASK_ROOT, load_module

verifier = load_module("kv_eviction_v2_verifier", "images/judge/verify_kv_eviction.py")
adapter = load_module("kv_eviction_v2_adapter_for_contract", "images/judge/run_judge.py")
hashes = load_module("kv_eviction_v2_update_hashes", "scripts/update_protected_hashes.py")


@pytest.mark.parametrize(("round_id", "override", "expected"), (
    ("agent-3", None, ("final", "final-holdout-v2", None)),
    ("auto-1", None, ("final", "final-holdout-v2", None)),
    ("manual-1", None, ("final", "final-holdout-v2", None)),
    ("candidate-2", 7, ("final", "final-holdout-v2", 7)),
))
def test_every_scored_round_is_the_full_final_suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, round_id, override, expected) -> None:
    context = tmp_path / "ctx.json"; context.write_text(json.dumps({"round_id": round_id, "candidate_version": override}))
    monkeypatch.setattr(verifier, "EVALUATION_CONTEXT", context)
    monkeypatch.setattr(verifier, "_strict_json", lambda path, label, **_: json.loads(Path(path).read_text()))
    assert verifier.evaluation_profile() == expected


def test_interactive_rounds_cannot_override_the_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = tmp_path / "ctx.json"; context.write_text(json.dumps({"round_id": "agent-1", "candidate_version": 2}))
    monkeypatch.setattr(verifier, "EVALUATION_CONTEXT", context)
    monkeypatch.setattr(verifier, "_strict_json", lambda path, label, **_: json.loads(Path(path).read_text()))
    with pytest.raises(RuntimeError, match="override forbidden"):
        verifier.evaluation_profile()


def test_gate_lists_agree() -> None:
    assert tuple(verifier.GATE_KEYS) == tuple(name for name, _label in adapter.GATES)
    assert set(verifier.RATIO_KEYS.values()) == set(adapter.RATIO_KEYS)


def test_protected_hash_list_matches_the_image() -> None:
    labels = {label for label, _path in hashes.FILES}
    assert labels == set(verifier.PROTECTED)
    recorded = {line.split()[1] for line in (TASK_ROOT / "images/judge/protected_hashes.txt").read_text().splitlines() if line.strip()}
    assert recorded == labels
    dockerignore = (TASK_ROOT / "images/judge/.dockerignore").read_text()
    for label in labels:
        if label.startswith("judge:"):
            assert f"!{label.split(':', 1)[1]}" in dockerignore


def test_private_rows_are_compact() -> None:
    assert "layer_budgets" not in verifier.PRIVATE_ROW_KEYS
    task = json.loads((TASK_ROOT / "tasks/kvpress_kv_eviction_policy_qwen3_8b_h100_scratch_v2.json").read_text())
    assert task["judge"]["private_artifact_path"] == f"/logs/verifier/{verifier.PRIVATE_NAME}"


def test_failure_record_is_well_formed() -> None:
    record = verifier.failure("boom", "final")
    assert record["schema_version"] == 2 and record["profile_code"] == 2.0 and record["score_fidelity_auc"] == 0.0
    assert all(record[key] == 0.0 for key in verifier.GATE_KEYS)


def _timed_rows(item_id: str = "f00001", select: float = 0.5, prefill: float = 0.6) -> list[dict]:
    return [{"item_id": item_id, "ratio": float(r), "select_time_sum": select, "select_time_max_layer": select / 2, "select_time_own": select / 3,
             "memory_delta": 10, "prefill_s": prefill, "masks_sha256": f"{i:064x}"} for i, r in enumerate(verifier.RATIOS)]


def _fresh(rows: list[dict], select: float, **change) -> dict:
    fresh = {}
    for row in rows:
        t = {"item_id": row["item_id"], "ratio": row["ratio"], "select_time_sum": select, "select_time_max_layer": select / 2, "select_time_own": select / 3,
             "memory_delta": 20, "masks_sha256": row["masks_sha256"], "indices_ok": True, "immutable_ok": True, "rng_ok": True, "budgets_ok": True}
        t.update(change); fresh[float(row["ratio"])] = t
    return fresh


def test_over_cap_items_follow_the_per_item_cap() -> None:
    rows = _timed_rows("f1", select=0.37, prefill=0.6) + _timed_rows("f2", select=0.35, prefill=0.6) + _timed_rows("f3", select=0.1, prefill=0.0)
    assert verifier.over_cap_items(rows) == ["f1", "f3"]  # 0.37 / 0.6 > 0.60; a zero prefill never passes


def test_retime_adopts_only_a_faster_bit_identical_selection() -> None:
    rows = _timed_rows(select=0.5)
    assert verifier.apply_retime(rows, "f00001", _fresh(rows, 0.1))
    assert all(r["retimed"] is True and r["select_time_sum"] == 0.1 and r["select_time_sum_first"] == 0.5 and r["memory_delta"] == 20 for r in rows)
    rows = _timed_rows(select=0.5)
    assert not verifier.apply_retime(rows, "f00001", _fresh(rows, 0.1, masks_sha256="f" * 64))  # different selection
    assert not verifier.apply_retime(rows, "f00001", _fresh(rows, 0.6))  # not faster
    assert not verifier.apply_retime(rows, "f00001", _fresh(rows, 0.1, rng_ok=False))
    assert not verifier.apply_retime(rows, "f00001", {k: v for k, v in list(_fresh(rows, 0.1).items())[:2]})  # a ratio missing
    assert all("retimed" not in r and r["select_time_sum"] == 0.5 for r in rows)


def test_retime_is_skipped_for_a_genuinely_slow_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [row for i in range(verifier.RETIME_MAX_ITEMS + 1) for row in _timed_rows(f"f{i:05d}", select=0.9, prefill=1.0)]
    pool = type("NoPool", (), {"submit": lambda *a, **k: pytest.fail("re-timed a slow policy")})()
    info = verifier.retime_over_cap(pool, rows, [], [], [], {}, Path("x"), None, Path("x"), {}, None)
    assert info == {"over_cap_items": float(verifier.RETIME_MAX_ITEMS + 1), "retimed_items": 0.0, "retime_wall_s": 0.0}
