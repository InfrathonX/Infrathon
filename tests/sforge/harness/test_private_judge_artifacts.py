from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path, PurePosixPath

from sforge.harness.run_evaluation import (
    PRIVATE_JUDGE_ARTIFACT_NAME,
    _persist_private_judge_artifact,
)


def _tar_bytes(name: str, payload: bytes, *, mode: int = tarfile.REGTYPE) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        info = tarfile.TarInfo(name)
        info.type = mode
        info.size = len(payload) if mode == tarfile.REGTYPE else 0
        archive.addfile(
            info,
            io.BytesIO(payload) if mode == tarfile.REGTYPE else None,
        )
    return output.getvalue()


class _FakeBackend:
    def __init__(self, transport: bytes):
        self.transport = transport
        self.requested: PurePosixPath | None = None

    def copy_from_container(self, _handle, src: PurePosixPath) -> bytes:
        self.requested = src
        return self.transport


class PrivateJudgeArtifactTests(unittest.TestCase):
    def test_persists_valid_tarball_host_only(self) -> None:
        payload = json.dumps(
            {
                "cases": [
                    {
                        "name": "PRIVATE_CASE",
                        "shape": [1, 2, 3, 4],
                        "candidate_ms": 12.5,
                    }
                ]
            }
        ).encode()
        transport = _tar_bytes("flash_kmeans_results.json", payload)
        backend = _FakeBackend(transport)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = _persist_private_judge_artifact(
                backend,
                object(),
                "/logs/verifier/flash_kmeans_results.json",
                root,
            )
            self.assertEqual(
                backend.requested,
                PurePosixPath("/logs/verifier/flash_kmeans_results.json"),
            )
            self.assertEqual(destination.name, PRIVATE_JUDGE_ARTIFACT_NAME)
            self.assertEqual(destination.stat().st_mode & 0o777, 0o600)
            with tarfile.open(destination, mode="r:gz") as archive:
                self.assertEqual(
                    archive.getnames(),
                    ["flash_kmeans_results.json"],
                )
                member = archive.getmember("flash_kmeans_results.json")
                self.assertEqual(member.mode, 0o600)
                self.assertEqual(member.uid, 0)
                self.assertEqual(member.gid, 0)
                self.assertEqual(member.mtime, 0)
                self.assertEqual(
                    archive.extractfile(member).read(),
                    payload,
                )

    def test_rejects_transport_with_extra_or_mismatched_members(self) -> None:
        payload = b'{"cases": []}\n'
        extra = io.BytesIO()
        with tarfile.open(fileobj=extra, mode="w") as archive:
            for name in ("flash_kmeans_results.json", "extra"):
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        for transport in (
            extra.getvalue(),
            _tar_bytes("wrong-name.json", payload),
        ):
            with self.subTest(size=len(transport)):
                with tempfile.TemporaryDirectory() as temporary:
                    with self.assertRaises(ValueError):
                        _persist_private_judge_artifact(
                            _FakeBackend(transport),
                            object(),
                            "/logs/verifier/flash_kmeans_results.json",
                            Path(temporary),
                        )

    def test_rejects_non_regular_transport_or_unsafe_configured_path(self) -> None:
        for name, member_type, configured_path in (
            (
                "flash_kmeans_results.json",
                tarfile.SYMTYPE,
                "/logs/verifier/flash_kmeans_results.json",
            ),
            (
                "flash_kmeans_results.json",
                tarfile.LNKTYPE,
                "/logs/verifier/flash_kmeans_results.json",
            ),
            (
                "flash_kmeans_results.json",
                tarfile.REGTYPE,
                "relative/flash_kmeans_results.json",
            ),
            (
                "flash_kmeans_results.json",
                tarfile.REGTYPE,
                "/logs/verifier/../flash_kmeans_results.json",
            ),
        ):
            transport = _tar_bytes(name, b"{}", mode=member_type)
            with self.subTest(path=configured_path, member_type=member_type):
                with tempfile.TemporaryDirectory() as temporary:
                    with self.assertRaises(ValueError):
                        _persist_private_judge_artifact(
                            _FakeBackend(transport),
                            object(),
                            configured_path,
                            Path(temporary),
                        )

    def test_accepts_kubernetes_style_transport_member_name(self) -> None:
        payload = b'{"cases": []}\n'
        transport = _tar_bytes(
            "logs/verifier/flash_kmeans_results.json",
            payload,
        )
        with tempfile.TemporaryDirectory() as temporary:
            destination = _persist_private_judge_artifact(
                _FakeBackend(transport),
                object(),
                "/logs/verifier/flash_kmeans_results.json",
                Path(temporary),
            )
            with tarfile.open(destination, mode="r:gz") as archive:
                self.assertEqual(
                    archive.extractfile(
                        "flash_kmeans_results.json"
                    ).read(),
                    payload,
                )


if __name__ == "__main__":
    unittest.main()
