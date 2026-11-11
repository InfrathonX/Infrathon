"""Validate and count the persistent CUDA storage in a codec payload tree."""

from __future__ import annotations

import math
from typing import Any

import torch


MAX_SCALAR_STRING_BYTES = 4096
MAX_TREE_NODES = 100_000


def payload_nbytes(value: Any) -> int:
    """Return the bytes a plain payload tree keeps: unique CUDA storage plus scalars.

    Container cycles and custom objects are rejected. Repeated containers are
    permitted when acyclic; aliased tensors count their full storage once.
    Scalars are not free metadata: an int or float costs 8 bytes, a string or a
    dictionary key its UTF-8 length, a bool 1 byte (otherwise KV values could
    travel as text or Python numbers at zero counted bytes).
    """
    storages: set[tuple[str, int, int]] = set()
    active_containers: set[int] = set()
    nodes = 0

    def visit(item: Any) -> int:
        nonlocal nodes
        nodes += 1
        if nodes > MAX_TREE_NODES:
            raise ValueError(f"payload exceeds {MAX_TREE_NODES} nodes")

        if isinstance(item, torch.Tensor):
            # A tensor subclass can override untyped_storage()/nbytes() to lie
            # about how much storage it keeps alive, so only a plain tensor --
            # exactly ``torch.Tensor``, never a subclass or Parameter -- is
            # counted. ``type(item) is`` cannot be fooled by a ``__class__``
            # property the way ``isinstance`` can.
            if type(item) is not torch.Tensor:
                raise ValueError(
                    "payload tensors must be exactly torch.Tensor, not a subclass: "
                    f"{type(item).__module__}.{type(item).__qualname__}"
                )
            # Reject any tensor carrying state the storage counter cannot see,
            # because autograd graphs, saved gradients, Python attributes and
            # hooks can all keep *other* tensors alive outside the payload tree.
            # ``_base`` is deliberately allowed: an aliased view's full backing
            # storage is counted once above, so sharing it costs nothing extra.
            if item.requires_grad:
                raise ValueError("payload tensors must not require grad")
            if item.grad_fn is not None:
                raise ValueError("payload tensors must not carry an autograd graph (grad_fn)")
            if item.grad is not None:
                raise ValueError("payload tensors must not carry a saved .grad tensor")
            extra_attributes = getattr(item, "__dict__", None)
            if extra_attributes:
                raise ValueError(
                    "payload tensors must not carry Python attributes that can "
                    f"reference other tensors: {sorted(extra_attributes)}"
                )
            if getattr(item, "_backward_hooks", None):
                raise ValueError("payload tensors must not carry backward hooks")
            if getattr(item, "_post_accumulate_grad_hooks", None):
                raise ValueError("payload tensors must not carry post-accumulate-grad hooks")
            # Only dense, unnamed, non-quantized strided tensors: a nested
            # tensor's component shapes, a quantized tensor's scales/zero
            # points and a named tensor's dimension names all live outside the
            # counted storage.
            if item.is_nested or item.layout != torch.strided:
                raise ValueError("payload tensors must be dense strided tensors (no nested/sparse layouts)")
            if item.is_quantized:
                raise ValueError("payload tensors must not be quantized tensors (scales live outside storage)")
            names = getattr(item, "names", None)  # named tensors were removed after torch 2.8
            if names and any(name is not None for name in names):
                raise ValueError("payload tensors must not carry dimension names")
            if not item.is_cuda:
                raise ValueError("payload tensors must remain on CUDA")
            try:
                storage = item.untyped_storage()
                size = int(storage.nbytes())
                pointer = int(storage.data_ptr())
            except (RuntimeError, TypeError) as exc:
                raise ValueError("payload tensor must expose ordinary CUDA storage") from exc
            key = (str(item.device), pointer, size)
            if key in storages:
                return 0
            storages.add(key)
            return size

        item_type = type(item)
        if item is None:
            return 0
        if item_type is bool:
            return 1
        if item_type is int:
            if item.bit_length() > 64:
                raise ValueError("payload integers must fit in 64 bits")
            return 8
        if item_type is float:
            if not math.isfinite(item):
                raise ValueError("payload floats must be finite")
            return 8
        if item_type is str:
            encoded = len(item.encode("utf-8"))
            if encoded > MAX_SCALAR_STRING_BYTES:
                raise ValueError("payload strings are too large")
            return encoded
        if item_type not in {dict, list, tuple}:
            raise TypeError(f"unsupported payload object: {item_type.__name__}")

        identity = id(item)
        if identity in active_containers:
            raise ValueError("payload container cycle detected")
        active_containers.add(identity)
        try:
            if item_type is dict:
                total = 0
                for key, child in item.items():
                    if type(key) is not str:
                        raise TypeError("payload dictionaries require string keys")
                    encoded = len(key.encode("utf-8"))
                    if encoded > MAX_SCALAR_STRING_BYTES:
                        raise ValueError("payload dictionary key is too large")
                    total += encoded + visit(child)
                return total
            return sum(visit(child) for child in item)
        finally:
            active_containers.remove(identity)

    return visit(value)
