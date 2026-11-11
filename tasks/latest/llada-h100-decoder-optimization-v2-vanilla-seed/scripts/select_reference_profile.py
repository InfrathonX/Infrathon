#!/usr/bin/env python3
"""Select the quality reference that matches this host's reference profile.

A producer-relative quality reference is comparable only on the same GPU model
and compute capability under the same software ABI (CUDA runtime / torch /
transformers, pinned in the Judge verifier and checked there).  The host
driver version and the Work/Judge image ids are recorded as provenance but do
not select or refuse a reference: a driver update or an image rebuild with
unchanged software keeps the frozen reference.  This step runs on the host
before Work starts: it resolves the reference profile of the GPU the Judge will
use, looks it up in the per-profile cache next to the reference, and installs
the cached reference as the single-profile file the Judge image carries.  When
the cache has no entry the step fails closed and prints the exact regeneration
command instead of silently reusing another device's numbers.

``--check`` never writes.  It accepts a device the installed reference was
measured on, or one with its own cache entry: the Judge image carries the whole
cache and the verifier selects the entry for the GPU it observes before it
falls back to the installed file, so one Judge image serves every calibrated
GPU model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


REFERENCE_NAME = "reference_quality.json"
CACHE_DIRNAME = "reference_cache"
# The selection/refusal key.  driver_version is queried and printed as
# provenance only.
REFERENCE_PROFILE_FIELDS = ("compute_capability", "gpu_name")


def profile_key(profile: dict[str, str]) -> str:
    """Hash one profile exactly as the Judge does."""
    canonical = json.dumps(profile, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def reference_profile_key(profile: dict[str, str]) -> str:
    """Hash the GPU model + compute capability half of a profile."""
    return profile_key({field: str(profile[field]) for field in REFERENCE_PROFILE_FIELDS})


def installed_reference_profile_key(reference: dict[str, object]) -> str | None:
    """Derive the reference profile key a calibrated reference was measured under."""
    profile = reference.get("measurement_profile")
    if not isinstance(profile, dict):
        return None
    try:
        return reference_profile_key(profile)  # type: ignore[arg-type]
    except KeyError:
        return None


def query(index: int) -> dict[str, str]:
    """Read the device profile of one physical GPU from nvidia-smi."""
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=name,compute_cap,driver_version",
            "--format=csv,noheader",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"nvidia-smi failed for GPU {index}: {completed.stderr.strip()}")
    fields = [item.strip() for item in completed.stdout.strip().split(",")]
    if len(fields) != 3 or not all(fields):
        raise SystemExit(f"nvidia-smi did not describe GPU {index}: {completed.stdout!r}")
    name, capability, driver = fields
    return {
        "gpu_name": name,
        "compute_capability": capability,
        "driver_version": driver,
    }


def private_data_dir(root: Path) -> Path:
    """The Judge-only private_data/ directory (hidden cases, references, policies).

    The in-repository images/judge/private_data/ when it is present, else the
    task's dataset asset that assets/fetch.sh places at
    <INFRATHON_ASSETS_ROOT>/<task directory>/judge/private_data (default asset root:
    ~/.cache/infrathon/assets).  Mirrors scripts/h100_policy_contract.py.
    """
    in_repository = root / "images/judge/private_data"
    if in_repository.is_dir() and any(in_repository.iterdir()):
        return in_repository
    assets_root = os.environ.get("INFRATHON_ASSETS_ROOT") or str(
        Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
        / "infrathon/assets"
    )
    return Path(assets_root).expanduser() / root.name / "judge/private_data"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the host selection arguments."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judge-gpu", type=int, required=True)
    parser.add_argument(
        "--private-data",
        type=Path,
        default=private_data_dir(root),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the Judge has a reference for this device, never write",
    )
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="accept the empty placeholder when no profile has been calibrated yet",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Install the cached reference for this host's device profile."""
    args = parse_args(argv)
    private = args.private_data.resolve()
    reference = private / REFERENCE_NAME
    if reference.is_symlink() or not reference.is_file():
        raise SystemExit(f"quality reference is missing or unsafe: {reference}")
    device = query(args.judge_gpu)
    key = reference_profile_key(device)
    cached = private / CACHE_DIRNAME / f"{key}.json"
    installed = json.loads(reference.read_text(encoding="utf-8"))
    if installed_reference_profile_key(installed) == key:
        recorded = installed.get("measurement_profile") or {}
        print(
            f"reference already matches device profile {key} "
            f"({device['gpu_name']}, CC {device['compute_capability']}); "
            f"host driver {device['driver_version']}, reference measured under "
            f"driver {recorded.get('driver_version')!r} (provenance only)"
        )
        return 0
    if args.check:
        if cached.is_file() and not cached.is_symlink():
            print(
                f"reference cache entry {cached.name} matches device profile {key} "
                f"({device['gpu_name']}, CC {device['compute_capability']}); "
                "the Judge selects it on-device"
            )
            return 0
        if args.allow_pending and installed.get("calibrated") is False:
            print(f"no calibrated reference for device profile {key}; placeholder retained")
            return 0
        raise SystemExit(
            f"the built reference does not match device profile {key}; run "
            "scripts/select_reference_profile.py without --check and rebuild "
            "the Judge image"
        )
    if cached.is_file() and not cached.is_symlink():
        reference.write_bytes(cached.read_bytes())
        print(f"installed {cached} for device profile {key}")
        return 0
    if args.allow_pending and installed.get("calibrated") is False:
        print(f"no calibrated reference for device profile {key}; placeholder retained")
        return 0
    raise SystemExit(
        f"no quality reference is calibrated for device profile {key} "
        f"({device['gpu_name']}, CC {device['compute_capability']}, driver "
        f"{device['driver_version']}).\n"
        "Regenerate it on this device: run the producer calibration cycle "
        "(LLADA_CALIBRATION=1 over the corrected reference "
        "archive), then scripts/calibrate_reference.py, which writes "
        f"{cached}."
    )


if __name__ == "__main__":
    sys.exit(main())
