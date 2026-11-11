"""Benchmark metadata — loaded from BENCHMARK.yaml in the tasks directory."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

BENCHMARK_FILENAME = "BENCHMARK.yaml"


@dataclass
class BenchmarkMeta:
    name: str
    base_images: dict[str, dict] = field(default_factory=dict)


def load_benchmark(tasks_dir: Path) -> BenchmarkMeta:
    """Load benchmark metadata from tasks_dir/BENCHMARK.yaml."""
    path = tasks_dir / BENCHMARK_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark metadata not found: {path}\n"
            f"Expected a {BENCHMARK_FILENAME} file in the tasks directory."
        )
    with open(path) as f:
        data = yaml.safe_load(f)
    return BenchmarkMeta(
        name=data["name"],
        base_images=data.get("base_images", {}),
    )
