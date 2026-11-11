"""Host-side per-version replay may score a version FINAL_VERSION never named.

The agent keeps every complete experiment in kernels/flash_kmeans_vNNN.py and
tests unselected ones publicly with --candidate-version, so most retained
implementations never carry a hidden score. Replay closes that gap by asking
the Judge for one specific version, through SForge's root-only evaluation
context. These tests pin the trust boundary: the request is accepted only on
host-side rounds the agent never sees.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from test_hidden_benchmark_protocol import _load_verifier


class CandidateVersionOverrideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.verifier = _load_verifier()

    def _context(self, payload: object, *, mode: int = 0o600) -> Path:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / ".sforge-evaluation-context.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        path.chmod(mode)
        return path

    def _load(self, payload: object, *, mode: int = 0o600) -> int | None:
        return self.verifier.load_candidate_version_override(
            self._context(payload, mode=mode), expected_uid=os.getuid()
        )

    def test_host_side_round_may_request_a_specific_version(self) -> None:
        for round_id in ("candidate-impl-v007-abc", "manual-1"):
            with self.subTest(round_id=round_id):
                self.assertEqual(
                    self._load({"round_id": round_id, "candidate_version": 7}), 7
                )

    def test_absent_override_scores_the_submission_selector(self) -> None:
        self.assertIsNone(self._load({"round_id": "manual-1"}))
        self.assertIsNone(
            self._load({"round_id": "manual-1", "candidate_version": None})
        )

    def test_missing_context_predates_per_version_replay(self) -> None:
        with TemporaryDirectory() as directory:
            self.assertIsNone(
                self.verifier.load_candidate_version_override(
                    Path(directory) / "absent.json", expected_uid=os.getuid()
                )
            )

    def test_interactive_rounds_refuse_the_override(self) -> None:
        """An agent-visible score must describe the agent's own selection."""

        for round_id in ("agent-3", "auto-3"):
            with self.subTest(round_id=round_id):
                with self.assertRaisesRegex(RuntimeError, "host-side only"):
                    self._load({"round_id": round_id, "candidate_version": 7})

    def test_invalid_versions_are_rejected(self) -> None:
        for value in (0, 1000, -1, True, "3", 3.0, [3]):
            with self.subTest(value=value):
                with self.assertRaisesRegex(RuntimeError, "candidate_version"):
                    self._load({"round_id": "manual-1", "candidate_version": value})

    def test_a_context_the_agent_could_write_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "permissions"):
            self._load(
                {"round_id": "manual-1", "candidate_version": 7}, mode=0o666
            )

    def test_a_context_owned_by_another_account_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "owner"):
            self.verifier.load_candidate_version_override(
                self._context({"round_id": "manual-1", "candidate_version": 7}),
                expected_uid=os.getuid() + 1,
            )

    def test_an_override_without_a_round_id_is_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "round_id"):
            self._load({"candidate_version": 7})


if __name__ == "__main__":
    unittest.main()
