# SPDX-License-Identifier: Apache-2.0
"""Non-blocking CUDA output collector owned by the frozen judge layer.

The sink records only when each output became ready. When a chunk was offered
to the runtime is recorded by the trusted controller, outside the candidate's
reach (see ``evaluation.OfferLog``), so a runtime cannot move the start of the
latency or throughput windows.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class OutputRecord:
    chunk_idx: int
    kind: str
    timestamp_ns: int
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class RunLedger:
    records: tuple[OutputRecord, ...]
    frames: dict[int, torch.Tensor]
    latents: dict[int, torch.Tensor]

    def ordered_chunks(self) -> tuple[int, ...]:
        return tuple(sorted(self.frames))


class OutputSink:
    """Collect frames and latents without synchronizing the producer stream."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reserved: set[tuple[str, int]] = set()
        self._records: dict[tuple[str, int], OutputRecord] = {}
        self._frames: dict[int, torch.Tensor] = {}
        self._latents: dict[int, torch.Tensor] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sdv2-output-sink")
        self._futures: list[Future[None]] = []
        self._copy_streams: dict[torch.device, torch.cuda.Stream] = {}

    def _reserve(self, kind: str, chunk_idx: int) -> None:
        if chunk_idx < 0:
            raise ValueError("chunk_idx must be non-negative")
        key = (kind, int(chunk_idx))
        with self._lock:
            if key in self._reserved:
                raise ValueError(f"duplicate {kind} emission for chunk {chunk_idx}")
            self._reserved.add(key)

    def _emit_cpu(self, kind: str, chunk_idx: int, tensor: torch.Tensor) -> None:
        host = tensor.detach().contiguous().clone().cpu()
        timestamp_ns = time.monotonic_ns()
        record = OutputRecord(chunk_idx, kind, timestamp_ns, tuple(host.shape), str(host.dtype))
        with self._lock:
            self._records[(kind, chunk_idx)] = record
            (self._frames if kind == "frames" else self._latents)[chunk_idx] = host

    def _emit_cuda(
        self,
        kind: str,
        chunk_idx: int,
        tensor: torch.Tensor,
        producer_stream: torch.cuda.Stream | None,
    ) -> None:
        device = tensor.device
        stream = producer_stream or torch.cuda.current_stream(device)
        ready = torch.cuda.Event(enable_timing=False)
        ready.record(stream)
        copy_stream = self._copy_streams.get(device)
        if copy_stream is None:
            copy_stream = torch.cuda.Stream(device=device)
            self._copy_streams[device] = copy_stream
        # Keep the source allocation live until the side-stream copy finishes.
        # Without this guard the producer can drop its last Python reference,
        # allowing the caching allocator to reuse the storage while the copy is
        # still pending.
        source = tensor.detach()
        host = torch.empty(tuple(tensor.shape), dtype=tensor.dtype, device="cpu", pin_memory=True)
        copied = torch.cuda.Event(enable_timing=False)
        with torch.cuda.stream(copy_stream):
            copy_stream.wait_event(ready)
            host.copy_(source, non_blocking=True)
            source.record_stream(copy_stream)
            copied.record(copy_stream)

        def finish(source_guard: torch.Tensor = source) -> None:
            ready.synchronize()
            timestamp_ns = time.monotonic_ns()
            copied.synchronize()
            # The default argument owns the CUDA allocation through the copy.
            del source_guard
            stable_host = host.clone()
            record = OutputRecord(
                chunk_idx,
                kind,
                timestamp_ns,
                tuple(stable_host.shape),
                str(stable_host.dtype),
            )
            with self._lock:
                self._records[(kind, chunk_idx)] = record
                (self._frames if kind == "frames" else self._latents)[chunk_idx] = stable_host

        self._futures.append(self._executor.submit(finish))

    def _emit(
        self,
        kind: str,
        chunk_idx: int,
        tensor: torch.Tensor,
        stream: torch.cuda.Stream | None,
    ) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("sink values must be torch tensors")
        self._reserve(kind, chunk_idx)
        if tensor.device.type == "cuda":
            self._emit_cuda(kind, chunk_idx, tensor, stream)
        else:
            if stream is not None:
                raise ValueError("a CUDA stream cannot accompany a CPU tensor")
            self._emit_cpu(kind, chunk_idx, tensor)

    def emit_frames(
        self,
        chunk_idx: int,
        frames_gpu: torch.Tensor,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        if frames_gpu.ndim != 4 or frames_gpu.shape[-1] != 3:
            raise ValueError(f"frames must have shape [T,H,W,3], got {tuple(frames_gpu.shape)}")
        if frames_gpu.dtype != torch.float32:
            raise TypeError(f"frames must be float32, got {frames_gpu.dtype}")
        self._emit("frames", int(chunk_idx), frames_gpu, stream)

    def emit_latent(self, chunk_idx: int, x0_gpu: torch.Tensor) -> None:
        if x0_gpu.ndim != 5:
            raise ValueError(f"latent must have shape [B,F,C,H,W], got {tuple(x0_gpu.shape)}")
        self._emit("latent", int(chunk_idx), x0_gpu, None)

    def _wait(self) -> None:
        for future in self._futures:
            future.result()
        self._futures.clear()

    def ledger(self) -> RunLedger:
        self._wait()
        with self._lock:
            records = tuple(
                sorted(self._records.values(), key=lambda item: (item.chunk_idx, item.kind))
            )
            return RunLedger(
                records,
                dict(self._frames),
                dict(self._latents),
            )

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        ledger = self.ledger()
        json_records = [
            {
                "chunk_idx": record.chunk_idx,
                "kind": record.kind,
                "timestamp_ns": record.timestamp_ns,
                "shape": list(record.shape),
                "dtype": record.dtype,
            }
            for record in ledger.records
        ]
        (destination / "ledger.json").write_text(
            json.dumps(json_records, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        np.savez(destination / "frames.npz", **{
            f"chunk_{index:05d}": self._numpy_payload(value)
            for index, value in sorted(ledger.frames.items())
        })
        np.savez(destination / "latents.npz", **{
            f"chunk_{index:05d}": self._numpy_payload(value)
            for index, value in sorted(ledger.latents.items())
        })

    @staticmethod
    def _numpy_payload(value: torch.Tensor) -> np.ndarray:
        # NumPy has no native bfloat16 scalar. Preserve its exact 16 bits.
        if value.dtype == torch.bfloat16:
            return value.contiguous().view(torch.uint16).numpy()
        return value.numpy()

    def merge_saved(self, path: str | Path) -> None:
        """Merge a child process's saved emission ledger while preserving timestamps."""
        source = Path(path)
        raw_records: list[dict[str, Any]] = json.loads((source / "ledger.json").read_text(encoding="utf-8"))
        with np.load(source / "frames.npz") as frames_file, np.load(source / "latents.npz") as latents_file:
            for raw in raw_records:
                kind = str(raw["kind"])
                chunk_idx = int(raw["chunk_idx"])
                self._reserve(kind, chunk_idx)
                key = f"chunk_{chunk_idx:05d}"
                array = frames_file[key] if kind == "frames" else latents_file[key]
                tensor = torch.from_numpy(array.copy())
                if str(raw["dtype"]) == "torch.bfloat16":
                    if tensor.dtype != torch.uint16:
                        raise TypeError("saved bfloat16 payload is not uint16")
                    tensor = tensor.view(torch.bfloat16)
                record = OutputRecord(
                    chunk_idx=chunk_idx,
                    kind=kind,
                    timestamp_ns=int(raw["timestamp_ns"]),
                    shape=tuple(int(value) for value in raw["shape"]),
                    dtype=str(raw["dtype"]),
                )
                with self._lock:
                    self._records[(kind, chunk_idx)] = record
                    (self._frames if kind == "frames" else self._latents)[chunk_idx] = tensor

    def close(self) -> None:
        self._wait()
        self._executor.shutdown(wait=True)

    def __enter__(self) -> "OutputSink":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
