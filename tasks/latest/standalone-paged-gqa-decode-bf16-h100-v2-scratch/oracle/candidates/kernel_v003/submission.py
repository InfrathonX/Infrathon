"""FlashAttention-3 (hopper) paged decode on this ABI: the forced-Judge paper oracle.

Code identical to oracle/paper_reference/fa3_entry.py; only this docstring
differs.  Upstream Dao-AILab/flash-attention
0251105a2fb19d2957484b7f023cd8c115286ced (hopper/, flash_attn_3, built for
sm_90a), called through the registered op torch.ops.flash_attn_3.fwd with
out_=out -- the call flash_attn_with_kvcache makes -- with cache_seqlens as
seqused_k, page_table, causal=True, num_splits=0 (FA3's num_splits
heuristic) and pack_gqa=None (FA3's pack-GQA heuristic), no scheduler
metadata.  The op is called directly because, at this commit, the
torch.library.custom_op wrapper rejects out_ aliasing its return value; the
wrapper adds no work for the Judge's contiguous tensors.

Not admissible: `import flash_attn_3` (a compiled extension) is forbidden by
source admission.  It is scored only by the LOCAL forced-oracle Judge image
(oracle/paper_reference/forced_judge/).  Known Judge gaps: it fails the
private edge cases whose unused KV page slots hold NaN/Inf (masked slots
reach the output).
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
