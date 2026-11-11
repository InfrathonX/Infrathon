"""Work/Judge image content and secret-split tests."""

from __future__ import annotations

from conftest import ROOT


def test_work_image_has_toolchain_and_agent_user() -> None:
    dockerfile = (ROOT / "images/work/Dockerfile").read_text()
    assert "pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel" in dockerfile
    assert "transformers==4.53.1" in dockerfile
    assert "useradd" in dockerfile
    assert "nvcc" in dockerfile


def test_judge_image_copies_verifier_but_not_host_controls() -> None:
    dockerfile = (ROOT / "images/judge/Dockerfile").read_text()
    assert "COPY verify_eagle3.py" in dockerfile
    assert "COPY private_data/" not in dockerfile
    holdout_layer = (ROOT / "images/judge/Dockerfile.private").read_text()
    assert "COPY --from=private_data . /opt/eagle3_judge/private_data/" in holdout_layer
    assert "host_reference" not in dockerfile


def test_work_image_does_not_copy_private_data() -> None:
    dockerfile = (ROOT / "images/work/Dockerfile").read_text()
    assert "private_data" not in dockerfile
    assert "host_reference" not in dockerfile
