"""FlashAttention-3 (hopper) forward on the paged-GQA ABI, writing into the Judge's `out`.

Host-only oracle entrypoint.  NOT a submission: source admission rejects
'import flash_attn_3' (FORBIDDEN_LITERAL_MARKERS contains the package name and
the import allowlist is math/typing/torch/triton).  It is timed by
fa3_runner.py with the task's own trusted runtime (run_suite: decoy pre-call,
fresh tensors, guard bands, max(host wall, CUDA event), FP32 + BF16 reference
gates), i.e. the Judge protocol minus source admission and the sanitizer pass.

Audit fix (2026-10-29): this calls the registered C++ op
torch.ops.flash_attn_3.fwd directly -- exactly the call that
flash_attn_interface._flash_attn_forward makes, with the same 34 positional
arguments at both 0251105a and 3.0.0b1 -- instead of going through
_flash_attn_forward.  At the pinned commit 0251105a, _flash_attn_forward is a
torch.library.custom_op with mutates_args=() (hopper/flash_attn_interface.py:59);
torch 2.8's custom-op wrapper (torch/_library/custom_ops.py:350,
_c_check_aliasing_constraint) raises "The output of this custom operator (1)
must not also be an input..." when `out_` is passed and returned, so the
previous version failed on its first call in the Dockerfile.pinned image.
Calling the op directly is what vLLM's FA3 backend does (output passed in),
and it skips no work: the wrapper only applies maybe_contiguous (a no-op for
the Judge's contiguous tensors) before the same fwd call.

Arguments are those of flash_attn_with_kvcache(q, k_cache, v_cache,
cache_seqlens=..., page_table=..., softmax_scale=..., causal=True):
num_splits=0 -> FA3's num_splits_heuristic, pack_gqa=None -> FA3's
get_pack_gqa / should_pack_gqa, sm_margin=0, no scheduler_metadata (FA3
computes its tile scheduler metadata inside the call, as a cold single call
must).
"""

import torch
import flash_attn_3._C  # noqa: F401  registers torch.ops.flash_attn_3

_fwd = torch.ops.flash_attn_3.fwd


def paged_gqa_decode_bf16(q, k_cache, v_cache, page_table, cache_seqlens, softmax_scale, out):
    _fwd(
        q,
        k_cache,
        v_cache,
        None,  # k_new
        None,  # v_new
        None,  # qv
        out,  # out_
        None,  # cu_seqlens_q
        None,  # cu_seqlens_k
        None,  # cu_seqlens_k_new
        None,  # seqused_q
        cache_seqlens,  # seqused_k
        None,  # max_seqlen_q
        None,  # max_seqlen_k
        page_table,
        None,  # kv_batch_idx
        None,  # leftpad_k
        None,  # rotary_cos
        None,  # rotary_sin
        None,  # seqlens_rotary
        None,  # q_descale
        None,  # k_descale
        None,  # v_descale
        softmax_scale,
        True,  # causal
        -1,  # window_size_left
        -1,  # window_size_right
        0,  # attention_chunk
        0.0,  # softcap
        True,  # rotary_interleaved
        None,  # scheduler_metadata
        0,  # num_splits (0 -> heuristic)
        None,  # pack_gqa (None -> heuristic)
        0,  # sm_margin
    )
    return None
