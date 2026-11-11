#!/usr/bin/env python3
"""GPU admission for the Miles tasks: the leased cards must meet the task's execution profile.

The profile is read from the task JSON
(portable.execution_profiles, default portable.defaults.execution_profile_id): every card of the
pair must have compute capability >= min_compute_capability and memory.total >= min_memory_bytes,
the pair must have `count` cards, and the two cards must be NVLink peers in `nvidia-smi topo -m`
(set MILES_ALLOW_NON_NVLINK=1 to accept a PCIe pair for diagnostics).

  gpu_profile.py TASK_JSON INDICES     # prints the admitted profile, UUIDs and card names as JSON
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path


class ProfileMismatch(RuntimeError):
    pass


def _cards() -> dict[int, dict]:
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,compute_cap,memory.total,memory.used",
         "--format=csv,noheader,nounits"], text=True)
    cards = {}
    for row in rows.splitlines():
        index, uuid, name, cap, total, used = [x.strip() for x in row.split(",")]
        cards[int(index)] = {"uuid": uuid, "name": name, "compute_capability": cap,
                             "memory_total_bytes": int(total) * 2**20, "memory_used_mib": int(used)}
    return cards


def _nvlink_peers(a: int, b: int) -> bool:
    text = re.sub(r"\x1b\[[0-9;]*m", "", subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True))
    links: dict[int, list[str]] = {}
    for line in text.splitlines():
        cells = line.split()
        if cells and re.fullmatch(r"GPU\d+", cells[0]):
            links[int(cells[0][3:])] = [c for c in cells[1:] if re.fullmatch(r"X|NV\d+|SYS|NODE|PHB|PXB|PIX", c)]
    try:
        return links[a][b].startswith("NV") and links[b][a].startswith("NV")
    except (KeyError, IndexError):
        return False


def _cap(value: str) -> tuple[int, int]:
    major, _, minor = str(value).partition(".")
    return int(major), int(minor or 0)


def admit(task_json: Path, indices: list[int], *, profile_id: str | None = None) -> dict:
    """The profile the pair satisfies, with UUIDs and card names; ProfileMismatch otherwise.

    Without ``profile_id``: the default profile (portable.defaults.execution_profile_id) if the pair satisfies it,
    else the first other profile, in declaration order, that it satisfies."""
    portable = json.loads(Path(task_json).read_text())["portable"]
    profiles = portable["execution_profiles"]
    if profile_id is not None:
        return _admit_profile(profiles, profile_id, indices)
    default = portable.get("defaults", {}).get("execution_profile_id") or next(iter(profiles))
    failures = []
    for candidate in [default] + [p for p in profiles if p != default]:
        try:
            return _admit_profile(profiles, candidate, indices)
        except ProfileMismatch as exc:
            failures.append(str(exc))
    raise ProfileMismatch(" | ".join(failures))


def _admit_profile(profiles: dict, profile_id: str, indices: list[int]) -> dict:
    if profile_id not in profiles:
        raise ProfileMismatch(f"unknown execution profile {profile_id!r}")
    profile = profiles[profile_id]
    if not indices or any(type(i) is not int or i < 0 for i in indices) or len(set(indices)) != len(indices):
        raise ProfileMismatch("GPU indices must be distinct non-negative integers")
    cards = _cards()
    problems = []
    if len(indices) != int(profile.get("count", 1)):
        problems.append(f"profile {profile_id} needs {profile.get('count')} GPUs, got {len(indices)}")
    for i in indices:
        card = cards.get(i)
        if card is None:
            problems.append(f"GPU {i} does not exist")
            continue
        if _cap(card["compute_capability"]) < _cap(profile["min_compute_capability"]):
            problems.append(f"GPU {i} ({card['name']}) has compute capability {card['compute_capability']} "
                            f"< {profile['min_compute_capability']}")
        if card["memory_total_bytes"] < int(profile["min_memory_bytes"]):
            problems.append(f"GPU {i} ({card['name']}) has {card['memory_total_bytes']} bytes "
                            f"< {profile['min_memory_bytes']}")
    required_pairs = profile.get("required_nvlink_pairs", [])
    if not isinstance(required_pairs, list) or any(
        not isinstance(pair, list) or len(pair) != 2 or pair[0] == pair[1]
        or any(type(i) is not int or not 0 <= i < int(profile.get("count", 1)) for i in pair)
        for pair in required_pairs
    ):
        raise ProfileMismatch(f"profile {profile_id} has invalid required_nvlink_pairs")
    if not problems:
        for a, b in required_pairs:
            if not _nvlink_peers(indices[a], indices[b]):
                problems.append(f"required trainer GPUs {indices[a]},{indices[b]} are not NVLink peers")
    if len(indices) == 2 and not required_pairs and not problems and not _nvlink_peers(*indices):
        if os.environ.get("MILES_ALLOW_NON_NVLINK") != "1":
            problems.append(f"GPUs {indices[0]},{indices[1]} are not NVLink peers (MILES_ALLOW_NON_NVLINK=1 accepts a PCIe pair)")
    if problems:
        raise ProfileMismatch(f"execution profile {profile_id}: " + "; ".join(problems))
    return {"execution_profile_id": profile_id, "gpu_indices": indices,
            "gpu_uuids": [cards[i]["uuid"] for i in indices], "gpu_names": [cards[i]["name"] for i in indices]}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    try:
        print(json.dumps(admit(Path(argv[0]), [int(x) for x in argv[1].split(",")]), indent=2))
    except ProfileMismatch as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 78
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
