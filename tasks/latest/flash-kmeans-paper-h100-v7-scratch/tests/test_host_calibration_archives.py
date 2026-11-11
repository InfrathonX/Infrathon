from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# The official-oracle archive is built from an upstream Flash-KMeans checkout
# that this repository does not vendor.  Point FLASH_KMEANS_UPSTREAM at a
# checkout pinned to 12ed49b2eddc207dfc5ed27dba244af2365bd9c6 to exercise it;
# without one, that single reproducibility case is skipped.
_UPSTREAM = os.environ.get("FLASH_KMEANS_UPSTREAM") or os.environ.get("VENDOR_ROOT")
VENDOR_ROOT = Path(_UPSTREAM).resolve() if _UPSTREAM else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HostCalibrationArchiveTests(unittest.TestCase):
    def _assert_gzip_is_reproducible(
        self,
        command_prefix: list[str],
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = []
            for sequence in (1, 2):
                archive = root / f"archive-{sequence}.tar.gz"
                source = root / f"source-{sequence}.py"
                subprocess.run(
                    command_prefix
                    + [
                        "--source-output",
                        str(source),
                        "--output",
                        str(archive),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                raw = archive.read_bytes()
                self.assertEqual(raw[:3], b"\x1f\x8b\x08")
                self.assertEqual(
                    raw[4:8],
                    b"\x00\x00\x00\x00",
                    "gzip header must not contain build time",
                )
                with tarfile.open(archive, "r:gz") as bundle:
                    self.assertEqual(
                        sorted(bundle.getnames()),
                        [
                            "FINAL_VERSION",
                            "kernels/flash_kmeans_v001.py",
                            "notes.md",
                        ],
                    )
                outputs.append((archive, source))
            self.assertEqual(_sha256(outputs[0][0]), _sha256(outputs[1][0]))
            self.assertEqual(_sha256(outputs[0][1]), _sha256(outputs[1][1]))

    def test_torch_fallback_archive_is_reproducible(self) -> None:
        self._assert_gzip_is_reproducible(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "build_torch_fallback_archive.py"),
                "--source",
                str(PROJECT_ROOT / "host_reference" / "baseline" / "torch_fallback.py"),
            ]
        )

    @unittest.skipIf(
        VENDOR_ROOT is None or not (
            VENDOR_ROOT / "flash_kmeans" / "assign_euclid_triton.py"
        ).is_file(),
        "set FLASH_KMEANS_UPSTREAM to an upstream flash-kmeans checkout",
    )
    def test_official_archive_is_reproducible(self) -> None:
        self._assert_gzip_is_reproducible(
            [
                sys.executable,
                str(
                    PROJECT_ROOT
                    / "scripts"
                    / "build_official_repo_head_archive.py"
                ),
                "--vendor-root",
                str(VENDOR_ROOT),
            ]
        )

    def test_standard_pytorch_archive_is_reproducible(self) -> None:
        self._assert_gzip_is_reproducible(
            [
                sys.executable,
                str(
                    PROJECT_ROOT
                    / "host_reference"
                    / "standard_pytorch_kmeans"
                    / "build_archive.py"
                ),
                "--source",
                str(
                    PROJECT_ROOT
                    / "host_reference"
                    / "standard_pytorch_kmeans"
                    / "standard_kmeans.py"
                ),
            ]
        )


if __name__ == "__main__":
    unittest.main()


class FastPytorchKmeansMarkerTests(unittest.TestCase):
    def test_upstream_commit_marker_matches_the_recorded_checkout(self) -> None:
        task = Path(__file__).resolve().parents[1]
        marker = task / "host_reference" / "fast_pytorch_kmeans" / "UPSTREAM_COMMIT"
        self.assertTrue(marker.is_file() and not marker.is_symlink())
        commit = marker.read_text(encoding="utf-8").strip()
        readme = (task / "host_reference" / "README.md").read_text(encoding="utf-8")
        self.assertRegex(commit, r"^[0-9a-f]{40}$")
        self.assertIn(commit, readme)
        builder = (task / "scripts" / "build_fast_pytorch_kmeans_archive.py").read_text(encoding="utf-8")
        self.assertIn('"UPSTREAM_COMMIT"', builder)

