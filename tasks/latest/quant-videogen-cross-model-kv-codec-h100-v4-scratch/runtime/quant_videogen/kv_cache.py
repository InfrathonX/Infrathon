"""Chunked K-or-V storage used by the task-owned model integrations."""

from __future__ import annotations

from enum import IntEnum
from typing import Optional

import torch

from .uncompress import uncompress_kv_cache


class ChunkState(IntEnum):
    EMPTY = 0
    BF16 = 1
    ENCODED = 2


class ChunkedKVCache:
    """Store BF16 chunks and independently decodable opaque codec spans."""

    def __init__(
        self,
        batch_size: int,
        frame_seq_length: int,
        num_heads: int,
        head_dim: int,
        max_num_chunks: int,
        dtype: torch.dtype,
        device: torch.device,
        layout: str = "BHSD",
    ) -> None:
        if layout not in {"BHSD", "BSHD"}:
            raise ValueError(f"layout must be BHSD or BSHD, got {layout}")
        self.batch_size = batch_size
        self.frame_seq_length = frame_seq_length
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_num_chunks = max_num_chunks
        self.dtype = dtype
        self.device = device
        self.layout = layout
        self.seq_dim = 2 if layout == "BHSD" else 1
        self.chunks: list[Optional[torch.Tensor]] = [None] * max_num_chunks
        self.chunk_state = [ChunkState.EMPTY] * max_num_chunks
        self.encoded_spans: list[dict[str, object]] = []

    @property
    def kv_cache_size(self) -> int:
        return self.max_num_chunks * self.frame_seq_length

    def _token_to_chunk(self, token_index: int) -> int:
        if token_index % self.frame_seq_length:
            raise ValueError(f"token index {token_index} is not frame-aligned")
        return token_index // self.frame_seq_length

    def _alloc_chunk(self, index: int) -> None:
        if self.chunks[index] is not None:
            return
        if self.layout == "BHSD":
            shape = (self.batch_size, self.num_heads, self.frame_seq_length, self.head_dim)
        else:
            shape = (self.batch_size, self.frame_seq_length, self.num_heads, self.head_dim)
        self.chunks[index] = torch.empty(shape, dtype=self.dtype, device=self.device)

    def _remove_overlaps(self, start: int, end: int) -> None:
        self.encoded_spans = [
            span
            for span in self.encoded_spans
            if int(span["end_chunk"]) <= start or int(span["start_chunk"]) >= end
        ]

    def _find_span(self, index: int) -> dict[str, object]:
        for span in self.encoded_spans:
            if int(span["start_chunk"]) <= index < int(span["end_chunk"]):
                return span
        raise ValueError(f"no encoded span covers chunk {index}")

    def write(self, start_index: int, end_index: int, data: torch.Tensor) -> None:
        if end_index - start_index != data.shape[self.seq_dim]:
            raise ValueError("write range does not match tensor sequence length")
        start = self._token_to_chunk(start_index)
        end = self._token_to_chunk(end_index)
        if end > self.max_num_chunks:
            raise ValueError("write exceeds cache capacity")
        for offset, index in enumerate(range(start, end)):
            if self.chunk_state[index] == ChunkState.ENCODED:
                self._remove_overlaps(index, index + 1)
            begin = offset * self.frame_seq_length
            finish = begin + self.frame_seq_length
            self._alloc_chunk(index)
            destination = self.chunks[index]
            assert destination is not None
            if self.seq_dim == 2:
                destination.copy_(data[:, :, begin:finish, :])
            else:
                destination.copy_(data[:, begin:finish, :, :])
            self.chunk_state[index] = ChunkState.BF16

    def read(self, start_index: int, end_index: int) -> torch.Tensor:
        start = self._token_to_chunk(start_index)
        end = self._token_to_chunk(end_index)
        parts: list[torch.Tensor] = []
        index = start
        while index < end:
            state = self.chunk_state[index]
            if state == ChunkState.BF16:
                chunk = self.chunks[index]
                assert chunk is not None
                parts.append(chunk)
                index += 1
                continue
            if state != ChunkState.ENCODED:
                raise ValueError(f"chunk {index} is empty")
            span = self._find_span(index)
            decoded, _ = uncompress_kv_cache(span["payload"], span["payload"])
            if self.layout == "BSHD":
                decoded = decoded.permute(0, 2, 1, 3).contiguous()
            span_start = int(span["start_chunk"])
            span_end = int(span["end_chunk"])
            offset = (index - span_start) * self.frame_seq_length
            length = (min(span_end, end) - index) * self.frame_seq_length
            if self.seq_dim == 2:
                parts.append(decoded[:, :, offset:offset + length, :])
            else:
                parts.append(decoded[:, offset:offset + length, :, :])
            index = min(span_end, end)
        return torch.cat(parts, dim=self.seq_dim)

    def store_quantized(self, start_index: int, end_index: int, payload: object) -> None:
        start = self._token_to_chunk(start_index)
        end = self._token_to_chunk(end_index)
        self._remove_overlaps(start, end)
        for index in range(start, end):
            self.chunks[index] = None
            self.chunk_state[index] = ChunkState.ENCODED
        self.encoded_spans.append({
            "start_chunk": start,
            "end_chunk": end,
            "payload": payload,
        })

    def evict_range(self, start_index: int, end_index: int) -> None:
        start = self._token_to_chunk(start_index)
        end = self._token_to_chunk(end_index)
        for index in range(start, end):
            if self.chunk_state[index] == ChunkState.BF16:
                self.chunks[index] = None
                self.chunk_state[index] = ChunkState.EMPTY
        retained = []
        for span in self.encoded_spans:
            span_start = int(span["start_chunk"])
            span_end = int(span["end_chunk"])
            if start <= span_start and span_end <= end:
                for index in range(span_start, span_end):
                    self.chunk_state[index] = ChunkState.EMPTY
            else:
                retained.append(span)
        self.encoded_spans = retained

    def clear(self) -> None:
        self.chunks = [None] * self.max_num_chunks
        self.chunk_state = [ChunkState.EMPTY] * self.max_num_chunks
        self.encoded_spans.clear()

    def offload(self) -> None:
        raise RuntimeError("codec payload offload is disabled")

    def onload(self, device: torch.device) -> None:
        if torch.device(device) != torch.device(self.device):
            raise RuntimeError("codec payload migration is disabled")


def offload_kv_cache_layer(layer: dict[str, object]) -> None:
    raise RuntimeError("KV-cache CPU offload is disabled for this task")


def onload_kv_cache_layer(layer: dict[str, object], device: torch.device) -> None:
    for value in layer.values():
        if isinstance(value, ChunkedKVCache):
            value.onload(device)
