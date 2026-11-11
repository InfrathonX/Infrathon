#!/usr/bin/env python3
"""Fast public ABI, fidelity, and compression diagnostics."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch


PUBLIC_TESTS = Path(__file__).resolve().parent
SUBMISSION = Path("/workspace/submission")
sys.path.insert(0, str(PUBLIC_TESTS))
from codec_source_policy import PolicyError, validate_submission  # noqa: E402

try:
    SOURCE_POLICY = validate_submission(SUBMISSION)
except PolicyError as exc:
    raise SystemExit(f"submission source policy failed: {exc}") from exc

sys.path.insert(0, str(SUBMISSION))
from mlsysforge_codec import decode, encode  # noqa: E402
sys.path.insert(1, "/opt/qvg-src")
from quant_videogen.payload import payload_nbytes  # noqa: E402


def _assert_tensor(actual: torch.Tensor, expected: torch.Tensor, label: str) -> float:
    if not isinstance(actual, torch.Tensor):
        raise AssertionError(f"decoded {label} is not a tensor")
    if actual.shape != expected.shape:
        raise AssertionError(f"decoded {label} shape mismatch")
    if actual.dtype != torch.bfloat16 or actual.device != expected.device:
        raise AssertionError(f"decoded {label} must remain CUDA BF16 on the input device")
    error = float(
        torch.linalg.vector_norm(actual - expected)
        / torch.linalg.vector_norm(expected)
    )
    if not math.isfinite(error):
        raise AssertionError(f"decoded {label} has non-finite relative L2")
    return error


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.manual_seed(20260802)
    torch.cuda.manual_seed_all(20260802)
    shape = (1, 12, 24 * 1560, 128)
    k = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 100).contiguous()
    v = (torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 1000).contiguous()
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(k.device).clone()

    torch.cuda.synchronize(k.device)
    started = time.perf_counter()
    packed_k, packed_v = encode(k, v)
    k_hat, v_hat = decode(packed_k, packed_v)
    torch.cuda.synchronize(k.device)
    round_trip_seconds = time.perf_counter() - started

    if not torch.equal(cpu_rng, torch.random.get_rng_state()) or not torch.equal(
        cuda_rng, torch.cuda.get_rng_state(k.device)
    ):
        raise AssertionError("codec changed the global RNG stream")

    e_k = _assert_tensor(k_hat, k, "K")
    e_v = _assert_tensor(v_hat, v, "V")
    if max(e_k, e_v) > 0.75:
        raise AssertionError(
            f"public reconstruction diagnostic exceeded 0.75 relative L2: K={e_k:.4f}, V={e_v:.4f}"
        )

    # Each payload is stored and decoded independently by the cache runtime.
    k_self, _ = decode(packed_k, packed_k)
    e_k_self = _assert_tensor(k_self, k, "independent K payload")
    del k_self
    v_self, _ = decode(packed_v, packed_v)
    e_v_self = _assert_tensor(v_self, v, "independent V payload")
    del v_self
    if max(e_k_self, e_v_self) > 0.75:
        raise AssertionError("payloads do not preserve useful fidelity when decoded independently")

    raw_bytes = (k.numel() + v.numel()) * k.element_size()
    packed_bytes = payload_nbytes((packed_k, packed_v))
    if packed_bytes <= 0:
        raise AssertionError("codec payload contains no persistent CUDA storage")
    ratio = raw_bytes / packed_bytes
    if ratio < 1.5:
        raise AssertionError(f"public compression diagnostic requires at least 1.5x, got {ratio:.3f}x")

    print(json.dumps({
        "valid": True,
        "diagnostic_only": True,
        "shape": shape,
        # Compression only: the hidden score also scales it by a quality factor
        # (AGENT_TASK.md, "Score"), which this fast check cannot measure.
        "compression_ratio": ratio,
        "packed_bytes": packed_bytes,
        "k_relative_l2": e_k,
        "v_relative_l2": e_v,
        "round_trip_seconds": round_trip_seconds,
        "source_policy": SOURCE_POLICY,
    }, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
