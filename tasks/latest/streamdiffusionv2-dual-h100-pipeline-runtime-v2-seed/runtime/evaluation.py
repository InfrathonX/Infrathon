"""Shared exact-ledger persistence, timing, and golden comparison helpers."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np
import torch

from sdv2_core.sink import RunLedger


WARMUP_CHUNK = 12


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numpy_payload(value: torch.Tensor) -> np.ndarray:
    stable = value.detach().cpu().contiguous()
    if stable.dtype == torch.bfloat16:
        return stable.view(torch.uint16).numpy()
    return stable.numpy()


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8).tobytes()).hexdigest()


def frame_hash_rows(frames: Mapping[int, torch.Tensor]) -> list[dict[str, Any]]:
    rows = []
    for chunk_idx, value in sorted(frames.items()):
        array = numpy_payload(value)
        rows.append(
            {
                "chunk_idx": int(chunk_idx),
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "sha256": array_sha256(array),
            }
        )
    return rows


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute an empty quantile")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def ledger_timing(
    ledger: RunLedger,
    push_timestamps_ns: Mapping[int, int],
    *,
    start_ns: int,
    construction_seconds: float,
    execution_seconds: float,
    construction_to_exit_seconds: float,
    warmup_chunk: int = WARMUP_CHUNK,
) -> dict[str, Any]:
    """Timing from the controller's push stamps and the sink's emission stamps.

    ``push_timestamps_ns`` is recorded by the trusted controller immediately
    before it calls ``start`` (chunk 0) or ``push`` (every later chunk); the
    runtime never sees or sets it. The steady window runs from the push of the
    first steady chunk to the emission of the last chunk, so all steady-chunk
    work must finish inside it: holding outputs back cannot shorten it and
    taking input early cannot either.
    """
    frame_records = sorted(
        (record for record in ledger.records if record.kind == "frames"),
        key=lambda record: record.chunk_idx,
    )
    emitted = {record.chunk_idx: record.timestamp_ns for record in frame_records}
    expected = list(range(len(ledger.frames)))
    if sorted(emitted) != expected or sorted(push_timestamps_ns) != expected:
        raise ValueError("frame/push timestamp chunks are incomplete")
    if expected[-1] < warmup_chunk:
        raise ValueError("case is too short for the steady-state window")
    pushed = {int(index): int(value) for index, value in push_timestamps_ns.items()}
    if pushed[0] != start_ns:
        raise ValueError("chunk 0 push stamp must be the start() entry stamp")
    if any(pushed[index] <= pushed[index - 1] for index in expected[1:]):
        raise ValueError("push stamps are not strictly ordered")
    if any(emitted[index] <= emitted[index - 1] for index in expected[1:]):
        raise ValueError("frame emissions are not strictly ordered")
    if any(emitted[index] <= pushed[index] for index in expected):
        raise ValueError("a chunk was emitted before it was pushed")
    steady_frames = sum(int(ledger.frames[index].shape[0]) for index in expected[warmup_chunk:])
    steady_elapsed = (emitted[expected[-1]] - pushed[warmup_chunk]) / 1e9
    if steady_elapsed <= 0:
        raise ValueError("steady-state elapsed time is not positive")
    latencies = [(emitted[index] - pushed[index]) / 1e9 for index in expected[warmup_chunk:]]
    frames_total = sum(int(ledger.frames[index].shape[0]) for index in expected)
    return {
        "warmup_chunk": warmup_chunk,
        "steady_window": "push(warmup_chunk) -> emit(last chunk)",
        "steady_frames": steady_frames,
        "steady_elapsed_seconds": steady_elapsed,
        "steady_fps": steady_frames / steady_elapsed,
        "ttff_seconds": (emitted[0] - start_ns) / 1e9,
        "push_to_emit_p50_seconds": quantile(latencies, 0.50),
        "push_to_emit_p99_seconds": quantile(latencies, 0.99),
        "construction_seconds": construction_seconds,
        "execution_seconds": execution_seconds,
        "construction_to_exit_seconds": construction_to_exit_seconds,
        "frames_total": frames_total,
    }


def load_case_frames(case: Mapping[str, Any], clip_root: Path) -> torch.Tensor:
    item = case["input"]
    if item["kind"] == "asset":
        array = np.load(clip_root / f"{item['asset_id']}.npy", allow_pickle=False)
    elif item["kind"] == "procedural":
        from media_gen import generate

        array = generate(
            int(item["seed"]),
            int(item["num_frames"]),
            int(item["height"]),
            int(item["width"]),
            cut_at=item.get("cut_at"),
        )
    else:
        raise ValueError(f"unsupported input kind: {item.get('kind')}")
    expected = (int(case["frames"]), int(case["height"]), int(case["width"]), 3)
    if array.dtype != np.uint8 or array.shape != expected:
        raise ValueError(
            f"case {case['id']} input must be uint8 {expected}, got {array.dtype} {array.shape}"
        )
    return torch.from_numpy(array).permute(3, 0, 1, 2).float().mul_(2.0 / 255.0).sub_(1.0)


def load_prompt(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    prompt = payload.get("prompt_embeds") if isinstance(payload, dict) else payload
    if not isinstance(prompt, torch.Tensor):
        raise TypeError(f"prompt embedding is not a tensor: {path}")
    if prompt.dtype != torch.bfloat16 or tuple(prompt.shape) != (1, 512, 4096):
        raise ValueError(f"unexpected prompt embedding shape/dtype: {path}")
    return prompt


def save_golden(
    destination: Path,
    *,
    case: Mapping[str, Any],
    ledger: RunLedger,
    reference_timing: Mapping[str, Any],
    include_full_frames: bool,
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"golden destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        np.savez_compressed(
            temporary / "latents.npz",
            **{
                f"chunk_{index:05d}": numpy_payload(value)
                for index, value in sorted(ledger.latents.items())
            },
        )
        hashes = frame_hash_rows(ledger.frames)
        atomic_json(temporary / "frame_hashes.json", {"schema_version": 1, "chunks": hashes})
        if include_full_frames:
            np.savez_compressed(
                temporary / "frames.npz",
                **{
                    f"chunk_{index:05d}": numpy_payload(value)
                    for index, value in sorted(ledger.frames.items())
                },
            )
        files = {}
        for path in sorted(temporary.iterdir()):
            if path.name == "manifest.json" or not path.is_file():
                continue
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        manifest = {
            "schema_version": 1,
            "case": dict(case),
            "chunks": len(ledger.frames),
            "frames_storage": "lossless_npz_and_sha256" if include_full_frames else "sha256_only",
            "latent_storage": "bf16_bits_as_uint16_lossless_npz",
            "reference_timing": dict(reference_timing),
            "files": files,
        }
        atomic_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def digest_saved_ledger(source: Path, destination: Path, *, expected_chunks: int) -> dict[str, Any]:
    """Compact a trusted in-run reference run into a golden-shaped directory.

    The seed reference pass writes the same artifacts a candidate writes.  This
    turns one such directory into the frame-hash + lossless-latent form that
    `compare_saved_ledger` already consumes, so the in-run reference and the
    frozen goldens share one comparison path.  Frames are kept as SHA-256 only;
    a full frame suite would be tens of GiB for the holdout tier.
    """
    if destination.exists():
        raise FileExistsError(f"reference destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_keys = [f"chunk_{index:05d}" for index in range(expected_chunks)]
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        rows: list[dict[str, Any]] = []
        with np.load(source / "frames.npz", allow_pickle=False) as frames_file:
            if sorted(frames_file.files) != expected_keys:
                raise ValueError("reference frame chunks are incomplete")
            for index, key in enumerate(expected_keys):
                array = frames_file[key]
                if array.dtype != np.float32 or not bool(np.isfinite(array).all()):
                    raise ValueError(f"reference frames are not finite float32: {key}")
                rows.append(
                    {
                        "chunk_idx": index,
                        "dtype": str(array.dtype),
                        "shape": list(array.shape),
                        "sha256": array_sha256(array),
                    }
                )
        with np.load(source / "latents.npz", allow_pickle=False) as latent_file:
            if sorted(latent_file.files) != expected_keys:
                raise ValueError("reference latent chunks are incomplete")
            for key in expected_keys:
                if latent_file[key].dtype != np.uint16:
                    raise ValueError(f"reference latent payload is not bf16 bits: {key}")
        atomic_json(temporary / "frame_hashes.json", {"schema_version": 1, "chunks": rows})
        shutil.copyfile(source / "latents.npz", temporary / "latents.npz")
        files = {}
        for path in sorted(temporary.iterdir()):
            if path.name == "manifest.json" or not path.is_file():
                continue
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        manifest = {
            "schema_version": 1,
            "origin": "in_run_seed_reference",
            "chunks": expected_chunks,
            "frames_storage": "sha256_only",
            "latent_storage": "bf16_bits_as_uint16_lossless_npz",
            "files": files,
        }
        atomic_json(temporary / "manifest.json", manifest)
        os.replace(temporary, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _checked_npz(path: Path, expected_sha: str):
    if not path.is_file() or path.is_symlink() or sha256_file(path) != expected_sha:
        raise ValueError(f"golden payload hash mismatch: {path}")
    return np.load(path, allow_pickle=False)


def compare_saved_ledger(candidate_dir: Path, golden_dir: Path) -> dict[str, Any]:
    golden_manifest = json.loads((golden_dir / "manifest.json").read_text(encoding="utf-8"))
    for name, metadata in golden_manifest["files"].items():
        path = golden_dir / name
        if not path.is_file() or path.is_symlink() or sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"golden file failed its manifest: {name}")

    records = json.loads((candidate_dir / "ledger.json").read_text(encoding="utf-8"))
    pushes = json.loads((candidate_dir / "push_timestamps.json").read_text(encoding="utf-8"))
    expected_chunks = int(golden_manifest["chunks"])
    expected_keys = [f"chunk_{index:05d}" for index in range(expected_chunks)]
    frame_records = [row for row in records if row.get("kind") == "frames"]
    latent_records = [row for row in records if row.get("kind") == "latent"]
    frame_order = [int(row["chunk_idx"]) for row in sorted(frame_records, key=lambda row: row["timestamp_ns"])]
    latent_order = [int(row["chunk_idx"]) for row in sorted(latent_records, key=lambda row: row["timestamp_ns"])]
    expected_order = list(range(expected_chunks))
    frame_emitted = {int(row["chunk_idx"]): int(row["timestamp_ns"]) for row in frame_records}
    completeness_ok = (
        frame_order == expected_order
        and latent_order == expected_order
        and sorted(int(key) for key in pushes) == expected_order
        # Causality: no chunk can be ready before the controller pushed it.
        and all(frame_emitted.get(index, 0) > int(pushes.get(str(index), 2**63)) for index in expected_order)
    )

    frame_max_abs = 0.0
    latent_max_abs = 0.0
    finite_ok = True
    frame_exact = True
    latent_exact = True
    frame_hashes = json.loads((golden_dir / "frame_hashes.json").read_text(encoding="utf-8"))["chunks"]
    golden_hashes = {int(row["chunk_idx"]): row for row in frame_hashes}
    frame_file = np.load(candidate_dir / "frames.npz", allow_pickle=False)
    latent_file = np.load(candidate_dir / "latents.npz", allow_pickle=False)
    golden_latent_file = _checked_npz(
        golden_dir / "latents.npz", golden_manifest["files"]["latents.npz"]["sha256"]
    )
    golden_frame_path = golden_dir / "frames.npz"
    golden_frame_file = (
        _checked_npz(golden_frame_path, golden_manifest["files"]["frames.npz"]["sha256"])
        if golden_frame_path.exists()
        else None
    )
    try:
        if sorted(frame_file.files) != expected_keys or sorted(latent_file.files) != expected_keys:
            completeness_ok = False
        for index, key in enumerate(expected_keys):
            if key not in frame_file or key not in latent_file or key not in golden_latent_file:
                continue
            candidate_frame = frame_file[key]
            candidate_latent_bits = latent_file[key]
            golden_latent_bits = golden_latent_file[key]
            if candidate_frame.dtype != np.float32 or candidate_latent_bits.dtype != np.uint16:
                finite_ok = False
            finite_ok = finite_ok and bool(np.isfinite(candidate_frame).all())
            candidate_latent = torch.from_numpy(candidate_latent_bits.copy()).view(torch.bfloat16).float()
            finite_ok = finite_ok and bool(torch.isfinite(candidate_latent).all().item())
            if candidate_latent_bits.shape != golden_latent_bits.shape:
                latent_exact = False
                latent_max_abs = math.inf
            elif not np.array_equal(candidate_latent_bits, golden_latent_bits):
                latent_exact = False
                golden_latent = torch.from_numpy(golden_latent_bits.copy()).view(torch.bfloat16).float()
                latent_max_abs = max(
                    latent_max_abs,
                    float((candidate_latent - golden_latent).abs().max().item()),
                )
            frame_meta = golden_hashes[index]
            candidate_hash = array_sha256(candidate_frame)
            if list(candidate_frame.shape) != frame_meta["shape"] or str(candidate_frame.dtype) != frame_meta["dtype"]:
                frame_exact = False
                frame_max_abs = math.inf
            elif candidate_hash != frame_meta["sha256"]:
                frame_exact = False
                if golden_frame_file is not None:
                    golden_frame = golden_frame_file[key]
                    if golden_frame.shape == candidate_frame.shape:
                        frame_max_abs = max(
                            frame_max_abs,
                            float(np.max(np.abs(candidate_frame - golden_frame))),
                        )
                    else:
                        frame_max_abs = math.inf
                else:
                    frame_max_abs = math.inf
    finally:
        frame_file.close()
        latent_file.close()
        golden_latent_file.close()
        if golden_frame_file is not None:
            golden_frame_file.close()
    return {
        "completeness_ok": completeness_ok,
        "finite_ok": finite_ok,
        "parity_ok": completeness_ok and finite_ok and frame_exact and latent_exact,
        "frame_exact": frame_exact,
        "latent_exact": latent_exact,
        "frame_max_abs": frame_max_abs,
        "latent_max_abs": latent_max_abs,
    }

