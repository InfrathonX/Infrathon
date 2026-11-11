"""Expanded model-engine source admission matrix."""

from __future__ import annotations

import pytest

from source_admission import scan_python


@pytest.mark.parametrize(
    "source,fragment",
    [
        ("import socket\n", "forbidden import"),
        ("import subprocess\n", "forbidden import"),
        ("import inspect\n", "forbidden import"),
        ("import torch\ntorch.load('x.pt')\n", "forbidden call"),
        ("import os\nos.getenv('ROUND_ID')\n", "environment-variable routing"),
        ("import os\nos.popen('id')\n", "forbidden call"),
        (
            "import os\nrunner = os.system\nrunner('id')\n",
            "forbidden call",
        ),
        ("import sys\nsys.modules['runtime.api'] = None\n", "control-plane access"),
    ],
)
def test_forbidden_source_routes(tmp_path, source, fragment) -> None:
    path = tmp_path / "candidate.py"
    path.write_text(source, encoding="utf-8")
    assert any(fragment in error for error in scan_python(path, "candidate.py"))


def test_allowed_compiler_surface(tmp_path) -> None:
    path = tmp_path / "candidate.py"
    path.write_text(
        "from torch.utils.cpp_extension import load\n"
        "def build(*args):\n"
        "    return load(name='x', sources=['kernel.cu'], "
        "build_directory='/tmp/torch_extensions/x')\n",
        encoding="utf-8",
    )
    assert scan_python(path, "candidate.py") == []
