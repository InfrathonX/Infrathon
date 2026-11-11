#!/usr/bin/env python3
"""Atomically materialize pending or owner-approved private policy state."""

from __future__ import annotations

import argparse
import json
import os
import secrets
from pathlib import Path
from typing import Any

FROZEN_TIERS = (
    (0.0, 0.03284789994157488),
    (0.25, 0.001034126163391934),
    (0.5, 0.00031023784901758014),
    (0.75, 0.00020682523267838676),
    (1.0, 0.00020682523267838676),
    (1.5, 0.00015511892450879007),
    (2.0, 0.0001),
    (3.0, 0.0001),
)
# Qualification freezes budgets and image identities, never a device-measured
# wall or peak: the per-cell caps are derived by the Judge from a same-run seed
# measurement on whatever GPU it is running on, so the cap policy below is
# template-owned and image qualification cannot restate it.
REQUIRED_CAPS = {
    "build_cap_s",
    "worker_timeout_s",
}
RELATIVE_CAP_POLICY = {
    "resource_cap_mode": "seed_relative_same_run",
    "wall_cap_seed_multiple": 8.0,
    "memory_cap_seed_multiple": 1.1,
    "memory_cap_headroom_bytes": 8 * 1024**3,
    "oracle_rate_seed_multiple": 3.0,
}


def load_object(path: Path, label: str) -> dict[str, Any]:
    """Load a real JSON object."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is missing or unsafe: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def validate_frozen_oracle(data: dict[str, Any]) -> None:
    """Require exact A2 oracle values in both pending and qualified output."""
    if data.get("attn_implementation") != "eager":
        raise ValueError("oracle attention implementation differs")
    if data.get("eps_hard") != 24.0 or data.get("scope") != "global_run":
        raise ValueError("oracle hard cap or scope differs")
    if data.get("p_soft") != FROZEN_TIERS[0][1]:
        raise ValueError("oracle soft-rate cap differs")
    observed = tuple(
        (row.get("margin"), row.get("max_rate"))
        for row in data.get("tiers", [])
        if isinstance(row, dict)
    )
    if observed != FROZEN_TIERS:
        raise ValueError("oracle tier values differ from helper A2")
    if any(data.get(key) != value for key, value in RELATIVE_CAP_POLICY.items()):
        raise ValueError("relative resource-cap policy differs")
    if {"wall_caps_s", "memory_caps_bytes"} & set(data):
        raise ValueError("frozen per-device caps may not enter the policy")


def qualified_policy(template: dict[str, Any], freeze: dict[str, Any]) -> dict[str, Any]:
    """Merge only qualification-owned caps and immutable identities."""
    missing = REQUIRED_CAPS - set(freeze)
    if missing:
        raise ValueError(f"qualification input lacks {sorted(missing)}")
    data = dict(template)
    # Oracle thresholds and the relative cap policy are template-owned and may
    # not be overridden by image qualification. D supplies only the effort
    # budgets and the image/runtime bindings.
    for key in REQUIRED_CAPS | {
        "work_image_id",
        "judge_image_id",
        "source_digest",
        "runtime_provenance_sha256",
        "qualified_at",
    }:
        if key in freeze:
            data[key] = freeze[key]
    data["status"] = "QUALIFIED"
    validate_frozen_oracle(data)
    return data


def atomic_create(path: Path, data: dict[str, Any]) -> None:
    """Create a mode-0600 JSON file without replacing an existing manifest."""
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace an existing manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse manifest materialization arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).parents[1] / "images/judge/private_data/policy.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Write the requested manifest state."""
    args = parse_args(argv)
    template = load_object(args.template, "policy template")
    validate_frozen_oracle(template)
    if args.freeze is None:
        data = dict(template)
        data["status"] = "PENDING_QUALIFICATION"
    else:
        data = qualified_policy(template, load_object(args.freeze, "qualification input"))
    atomic_create(args.output, data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
