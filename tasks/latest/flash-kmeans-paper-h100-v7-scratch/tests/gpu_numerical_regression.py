#!/usr/bin/env python3
"""Pinned-image GPU regression for the assignment error envelope."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import torch


sys.path.insert(0, "/opt/flash_kmeans_benchmark")

from kernel_bench import (  # noqa: E402
    CaseSpec,
    _check_assignment_distance,
    check_output,
    make_inputs,
    reference_assign,
)


def load_oracle(path: Path):
    spec = importlib.util.spec_from_file_location(
        "_flash_kmeans_gpu_regression_oracle",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import oracle: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_high_energy_wrong_label_rejected() -> None:
    spec = CaseSpec("energy_clamp", "assign", "online", 1, 1, 2, 128, 91)
    x = torch.full((1, 1, 128), 4.0, device="cuda", dtype=torch.float16)
    centroids = x.expand(1, 2, 128).clone()
    centroids[0, 1, 0] += 0.03125
    data = {
        "x": x,
        "centroids": centroids,
        "x_sq": x.float().square().sum(dim=-1),
    }
    reference = torch.zeros((1, 1), device="cuda", dtype=torch.int64)
    wrong = torch.ones((1, 1), device="cuda", dtype=torch.int32)
    ok, message = _check_assignment_distance(spec, wrong, reference, data)
    if ok:
        raise AssertionError(
            "high-energy wrong label escaped the 2.5e-4 envelope clamp"
        )
    if "meaningfully farther" not in message:
        raise AssertionError(f"unexpected rejection reason: {message}")


def assert_shifted_random_labels_rejected() -> None:
    spec = CaseSpec("shifted_labels", "assign", "online", 2, 1024, 33, 64, 92)
    data = make_inputs(spec)
    reference = reference_assign(
        data["x"],
        data["centroids"],
        data["x_sq"],
    )
    wrong = ((reference + 1) % spec.k).to(torch.int32)
    ok, message = _check_assignment_distance(spec, wrong, reference, data)
    if ok:
        raise AssertionError("random labels shifted by +1 were accepted")
    if "meaningfully farther" not in message:
        raise AssertionError(f"unexpected rejection reason: {message}")


def assert_close_fresh_seeds_accepted(oracle) -> None:
    cases = (
        (
            CaseSpec(
                "public_close",
                "assign",
                "close_centroid",
                1,
                8192,
                128,
                128,
                1104,
            ),
            (1104, 90_001_104, 10_002_105),
        ),
        (
            CaseSpec(
                "hidden_close",
                "assign",
                "close_centroid",
                1,
                8192,
                128,
                256,
                81_102,
            ),
            (81_102, 90_081_102, 10_082_102),
        ),
    )
    for template, seeds in cases:
        for seed in seeds:
            spec = replace(template, seed=seed)
            data = make_inputs(spec)
            labels = oracle.assign(
                data["x"],
                data["centroids"],
                data["x_sq"],
            )
            torch.cuda.synchronize()
            ok, message = check_output(spec, labels, data)
            if not ok:
                raise AssertionError(
                    f"{template.name} seed {seed} was rejected: {message}"
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the numerical regression")
    assert_high_energy_wrong_label_rejected()
    assert_shifted_random_labels_rejected()
    assert_close_fresh_seeds_accepted(load_oracle(args.oracle))
    print("Pinned numerical envelope regression: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
