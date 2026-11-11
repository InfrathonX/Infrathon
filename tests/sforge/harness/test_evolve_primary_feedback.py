from __future__ import annotations

from sforge.harness.evolve_scripts import generate_submit_script


def test_submit_feedback_emphasizes_opt_in_primary_score() -> None:
    script = generate_submit_script()

    assert "PRIMARY LEADERBOARD OBJECTIVE" in script
    assert ".metrics.primary_score_name" in script
    assert ".metrics.primary_score_value" in script
    assert ".metrics.primary_score_direction" in script
    assert "Manual delta:" in script
    assert "Best manual:" in script
    assert "previous manual best" in script
    assert "Best visible:" not in script
    assert "\n                echo \"  Delta:" not in script
    assert "PREVIOUS_BEST_SCORE" in script
    assert "Secondary diagnostics below do not replace" in script


def test_display_cache_uses_authoritative_judge_best() -> None:
    script = generate_submit_script()

    assert "HISTORY_AFTER=" in script
    assert ".best_score = $best_score" in script
    assert ".best_round = $best_round" in script
    assert "if $rate > (.best_pass_rate // 0)" not in script


def test_submit_waits_for_authoritative_history_publication() -> None:
    script = generate_submit_script()

    assert "HISTORY_CANDIDATE=" in script
    assert '.round == $round' in script
    assert '.status == "completed"' in script
    assert "for _ in $(seq 1 10)" in script


def test_submit_preserves_explicit_invalid_and_valid_history_json() -> None:
    script = generate_submit_script()

    assert 'if has("valid") then .valid else true end' in script
    assert '${HISTORY_BEFORE:-{}}' not in script
    assert '${HISTORY_CANDIDATE:-{}}' not in script
    assert '${HISTORY_AFTER:-{}}' not in script


def test_list_mode_describes_manual_submission_history() -> None:
    script = generate_submit_script()

    assert "Manual Submission History" in script
    assert "Total manual submissions:" in script
    assert "Best manual pass rate:" in script
    assert "Best manual score:" in script


def test_submit_script_canonicalizes_dot_prefixed_tar_members() -> None:
    script = generate_submit_script()

    assert "--transform='s,^\\.$,__sforge_archive_root__,'" in script
    assert "--transform='s,^\\./,,'" in script


def test_submit_script_streams_the_archive_with_authenticated_headers() -> None:
    script = generate_submit_script()

    assert '--data-binary "@$ARCHIVE_FILE"' in script
    assert '"X-SForge-Token: $TOKEN"' in script
    assert '"X-SForge-Submission-Kind: agent"' in script
    assert '"Content-Type: application/gzip"' in script
    assert '-F "archive=@$ARCHIVE_FILE"' not in script


def test_submit_keeps_internal_candidate_identity_out_of_agent_output() -> None:
    script = generate_submit_script()

    assert "CANDIDATE_ID=$(echo \"$SUBMIT_RESP\"" not in script
    assert "VERSION_EVENT_ID=$(echo \"$SUBMIT_RESP\"" not in script
    assert 'echo "  Candidate:' not in script
    assert 'echo "  Version event:' not in script
