"""Source admission positive and negative fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import ROOT
from source_admission import validate_submission


def workspace(tmp_path: Path, source: str, extra: tuple[str, bytes] | None = None):
    """Create one minimal submission workspace and admit it."""
    version = tmp_path / "solutions/engine_v001"
    version.mkdir(parents=True)
    (tmp_path / "FINAL_VERSION").write_text("1\n")
    (tmp_path / "notes.md").write_text("notes\n")
    (version / "candidate.py").write_text(source)
    if extra:
        (version / extra[0]).write_bytes(extra[1])
    return validate_submission(tmp_path)


@pytest.mark.parametrize(
    "source",
    [
        "import torch\ng = torch.cuda.CUDAGraph()\n",
        "from torch.utils.cpp_extension import load_inline\n",
        "import torch\ndef build(*args):\n    return torch.compile(lambda value: value)\n",
    ],
)
def test_accepted_surface(tmp_path, source) -> None:
    assert workspace(tmp_path, source)["ok"]


@pytest.mark.parametrize(
    "source,needle",
    [
        ("import socket\n", "socket"),
        ("import torch\ntorch.load('x')\n", "torch.load"),
        ("import runtime\nruntime.api = 1\n", "monkey"),
        ("import subprocess\n", "subprocess"),
        ("open('/etc/passwd').read()\n", "outside"),
    ],
)
def test_rejected_surface(tmp_path, source, needle) -> None:
    result = workspace(tmp_path, source)
    assert not result["ok"]
    assert needle in " ".join(result["errors"])


def test_rejects_1p1_mib_file(tmp_path) -> None:
    result = workspace(tmp_path, "x = 1\n", ("blob.txt", b"x" * (1150 * 1024)))
    assert not result["ok"]


def test_public_and_private_admission_policies_are_identical() -> None:
    """Public disclosure must not drift from the policy enforced by Judge."""
    private = ROOT / "images/judge/verifier/source_admission.py"
    public = ROOT / "images/work/public_tests/source_admission.py"
    assert public.read_bytes() == private.read_bytes()
