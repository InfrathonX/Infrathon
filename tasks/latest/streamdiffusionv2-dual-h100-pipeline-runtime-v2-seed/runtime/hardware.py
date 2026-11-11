"""Capability envelope for the SDV2 pipeline task: what it needs, not what it was qualified on.

The GPU model name and UUID are recorded as measurement provenance and never
gate admission.  Keep the four constants below identical in the Judge verifier
(`images/judge/verify_sdv2_pipeline.py`); the capability-envelope test asserts
the pair.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Two ranks are the workload: the DiT block stack is split across them and the
# per-chunk payload crosses the pair on every chunk.
DEVICE_COUNT = 2
# BF16 + flash-attn 2 only; no Hopper-specific instruction is used.
MIN_COMPUTE_CAPABILITY = (8, 0)
# 26 GiB: the 832x480 / 4-step family peaks at 18,289 MiB of device memory and
# the resource gate admits seed + 8 GiB on top of it.
MIN_DEVICE_MEMORY_BYTES = 26 * 1024**3
# 64 GiB: the host-side resident set of one worker session is capped at
# HOST_RSS_GROWTH_CAP_MIB (scoring.py); observed peak is ~48 GiB.
MIN_HOST_MEMORY_BYTES = 64 * 1024**3

_CGROUP_LIMITS = ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes")


def host_memory_bytes() -> int:
    """Memory this container may actually use: physical RAM capped by the cgroup."""
    total = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    for path in _CGROUP_LIMITS:
        try:
            raw = Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if raw.isdigit():
            total = min(total, int(raw))
        break
    return total


def hardware_inventory(torch: Any) -> dict[str, Any]:
    """Enforce the capability envelope and record the device identity as provenance."""
    if not torch.cuda.is_available() or torch.cuda.device_count() != DEVICE_COUNT:
        raise RuntimeError(f"exactly {DEVICE_COUNT} visible CUDA GPUs required")
    devices = []
    for index in range(DEVICE_COUNT):
        capability = tuple(int(value) for value in torch.cuda.get_device_capability(index))
        name = str(torch.cuda.get_device_name(index))
        props = torch.cuda.get_device_properties(index)
        if capability < MIN_COMPUTE_CAPABILITY:
            raise RuntimeError(
                f"GPU {index} compute capability {capability} is below the task minimum "
                f"{MIN_COMPUTE_CAPABILITY}: {name}"
            )
        if int(props.total_memory) < MIN_DEVICE_MEMORY_BYTES:
            raise RuntimeError(
                f"GPU {index} memory {int(props.total_memory)} bytes is below the task minimum "
                f"{MIN_DEVICE_MEMORY_BYTES} bytes: {name}"
            )
        devices.append(
            {
                "index": index,
                "name": name,
                "uuid": str(getattr(props, "uuid", "")),
                "capability": list(capability),
                "total_memory_bytes": int(props.total_memory),
                # NCCL also runs without peer access; recorded, never required.
                "peer_access_to_other": bool(
                    torch.cuda.can_device_access_peer(index, DEVICE_COUNT - 1 - index)
                ),
            }
        )
    host_memory = host_memory_bytes()
    if host_memory < MIN_HOST_MEMORY_BYTES:
        raise RuntimeError(
            f"host memory {host_memory} bytes is below the task minimum "
            f"{MIN_HOST_MEMORY_BYTES} bytes"
        )
    return {
        "count": DEVICE_COUNT,
        "devices": devices,
        "host_memory_bytes": host_memory,
        "torch_version": str(getattr(torch, "__version__", "")),
        "cuda_version": str(getattr(getattr(torch, "version", None), "cuda", "")),
    }
