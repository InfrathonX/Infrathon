from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from conftest import TASK_ROOT, load_module

verifier = load_module("kv_scoring_v2_verifier", "images/judge/verify_kv_scoring.py")
adapter = load_module("kv_scoring_v2_adapter_for_contract", "images/judge/run_judge.py")
hashes = load_module("kv_scoring_v2_update_hashes", "scripts/update_protected_hashes.py")


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
    recorded = dict(reversed(line.split()) for line in (TASK_ROOT / "images/judge/protected_hashes.txt").read_text().splitlines() if line.strip())
    assert recorded == {label: hashlib.sha256(path.read_bytes()).hexdigest() for label, path in hashes.FILES}
    dockerignore = (TASK_ROOT / "images/judge/.dockerignore").read_text()
    for label in labels:
        if label.startswith("judge:"):
            assert f"!{label.split(':', 1)[1]}" in dockerignore


def test_private_artifact_path() -> None:
    task = json.loads((TASK_ROOT / "tasks/kvzip_query_agnostic_kv_scoring_qwen3_8b_h100_scratch_v2.json").read_text())
    assert task["judge"]["private_artifact_path"] == f"/logs/verifier/{verifier.PRIVATE_NAME}"


def test_failure_record_is_well_formed() -> None:
    record = verifier.failure("boom", "final")
    assert record["schema_version"] == 2 and record["profile_code"] == 2.0 and record["score_rel_auc"] == 0.0
    assert all(record[key] == 0.0 for key in verifier.GATE_KEYS)


def _row(item_id: str = "r00001", score: float = 0.5, prefill: float = 0.6) -> dict:
    return {"item_id": item_id, "score_time_sum": score, "score_time_max_layer": score / 2, "memory_delta": 10, "prefill_s": prefill, "scores_sha256": "a" * 64}


def _fresh(row: dict, score: float, **change) -> dict:
    t = {"item_id": row["item_id"], "score_time_sum": score, "score_time_max_layer": score / 2, "memory_delta": 20, "scores_sha256": row["scores_sha256"],
         "scores_ok": True, "immutable_ok": True, "rng_ok": True}
    t.update(change); return t


def test_over_cap_rows_follow_the_cap_and_device_floor() -> None:
    rows = [_row("r1", 0.40, 0.6), _row("r2", 0.29, 0.5), _row("r3", 0.24, 0.025)]  # floor = 10 x 0.025 = 0.25 s
    assert verifier.over_cap_items(rows) == ["r1", "r2"]  # over max(50% of prefill, floor); r3 stays under the floor


def test_retime_adopts_only_faster_bit_identical_scores() -> None:
    rows = [_row()]
    assert verifier.apply_retime(rows, "r00001", _fresh(rows[0], 0.1))
    assert rows[0]["retimed"] is True and rows[0]["score_time_sum"] == 0.1 and rows[0]["score_time_sum_first"] == 0.5 and rows[0]["memory_delta"] == 20
    rows = [_row()]
    assert not verifier.apply_retime(rows, "r00001", _fresh(rows[0], 0.1, scores_sha256="b" * 64))
    assert not verifier.apply_retime(rows, "r00001", _fresh(rows[0], 0.7))
    assert not verifier.apply_retime(rows, "r00001", _fresh(rows[0], 0.1, immutable_ok=False))
    assert not verifier.apply_retime(rows, "r00001", _fresh(rows[0], 0.1, item_id="r00002"))
    assert "retimed" not in rows[0] and rows[0]["score_time_sum"] == 0.5


def test_retime_is_skipped_for_a_genuinely_slow_scorer() -> None:
    rows = [_row(f"r{i:05d}", 0.9, 1.0) for i in range(verifier.RETIME_MAX_ITEMS + 1)] + [_row("short", 0.0, 0.05)]  # floor 0.5 s
    pool = type("NoPool", (), {"submit": lambda *a, **k: pytest.fail("re-timed a slow scorer")})()
    info = verifier.retime_over_cap(pool, rows, [], [], [], {}, Path("x"), None, Path("x"), {}, None)
    assert info == {"over_cap_items": float(verifier.RETIME_MAX_ITEMS + 1), "retimed_items": 0.0, "retime_wall_s": 0.0}
