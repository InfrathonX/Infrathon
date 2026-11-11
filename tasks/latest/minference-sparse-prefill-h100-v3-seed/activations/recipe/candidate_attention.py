"""Generic HF prefill dispatch through the public five-argument candidate ABI.

No seed internals, prompt padding or pattern estimator are invoked here.
The callable under test owns variable-N handling. Decode uses the original
Transformers attention implementation. This module imports Torch lazily.
"""
from __future__ import annotations

from contextlib import contextmanager
import math


class CandidateAttentionError(RuntimeError):
    pass


class CandidateAttentionAdapter:
    def __init__(self, candidate, original_attention, *, expected_heads=32,
                 expected_kv_heads=8, expected_head_dim=128,
                 require_bf16=True, check_input_integrity=True, on_prefill=None):
        self.candidate = candidate
        self.original_attention = original_attention
        self.expected_heads = expected_heads
        self.expected_kv_heads = expected_kv_heads
        self.expected_head_dim = expected_head_dim
        self.require_bf16 = require_bf16
        self.check_input_integrity = check_input_integrity
        self.on_prefill = on_prefill
        self.reset()

    def reset(self):
        self.prefill_layers = []
        self.decode_calls = 0

    def __call__(self, module, query, key, value, attention_mask,
                 dropout=0.0, scaling=None, **kwargs):
        import torch

        qn, kn = int(query.shape[2]), int(key.shape[2])
        if qn != kn or qn == 1:
            self.decode_calls += 1
            return self.original_attention(module, query, key, value, attention_mask,
                                           dropout=dropout, scaling=scaling, **kwargs)
        if attention_mask is not None or dropout != 0.0 or kwargs.get("is_causal") is False:
            raise CandidateAttentionError("candidate prefill requires an unpadded causal input and dropout=0")
        expected_q = (1, self.expected_heads, qn, self.expected_head_dim)
        expected_kv = (1, self.expected_kv_heads, qn, self.expected_head_dim)
        if tuple(query.shape) != expected_q or tuple(key.shape) != expected_kv or tuple(value.shape) != expected_kv:
            raise CandidateAttentionError("HF QKV shapes do not match the native-GQA candidate ABI")
        if self.require_bf16 and any(t.dtype != torch.bfloat16 for t in (query, key, value)):
            raise CandidateAttentionError("candidate ABI requires BF16 QKV")
        if any(t.device != query.device for t in (key, value)):
            raise CandidateAttentionError("candidate QKV devices differ")
        scale = self.expected_head_dim ** -0.5 if scaling is None else float(scaling)
        if not math.isfinite(scale) or scale <= 0:
            raise CandidateAttentionError("candidate attention scale is invalid")
        # Clone: a candidate must not alter HF's tensors or its cache through an
        # alias. Full equality below checks the actual arguments passed to it.
        tensors = tuple(t.contiguous().clone() for t in (query, key, value))
        before = tuple(t.clone() for t in tensors) if self.check_input_integrity else None
        out = torch.full_like(tensors[0], float("nan"))
        returned = self.candidate(*tensors, scale, out)
        if query.is_cuda:
            torch.cuda.synchronize(query.device)
        if returned is not None:
            raise CandidateAttentionError("candidate must write out and return None")
        if before is not None and any(not torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                                      for a, b in zip(tensors, before)):
            raise CandidateAttentionError("candidate modified a QKV input")
        if not bool(torch.isfinite(out).all().item()):
            raise CandidateAttentionError("candidate left unwritten/nonfinite output")
        layer = int(module.layer_idx)
        self.prefill_layers.append(layer)
        if self.on_prefill is not None:
            self.on_prefill(layer, tensors, scale, out)
        return out.transpose(1, 2).contiguous(), None

    def assert_complete_prefill(self, num_layers=32):
        if self.prefill_layers != list(range(num_layers)):
            raise CandidateAttentionError(
                f"candidate prefill layer dispatch differs: {self.prefill_layers}; expected 0..{num_layers - 1}")


@contextmanager
def candidate_attention_scope(registry, adapter):
    """Restore the actual HF registry entry even when candidate execution fails."""
    original = registry["sdpa"]
    registry.register("sdpa", adapter)
    try:
        yield adapter
    finally:
        registry.register("sdpa", original)
