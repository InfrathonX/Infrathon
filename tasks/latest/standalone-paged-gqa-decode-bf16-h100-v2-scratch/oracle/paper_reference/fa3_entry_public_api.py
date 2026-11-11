"""Same as fa3_entry.py but through the public flash_attn_with_kvcache API.

The public API allocates its own output, so the result is copied into the
Judge's `out` (one extra BF16 copy kernel inside the timed call).  Works at the
pinned commit (flash_attn_with_kvcache passes out_=None to the custom op, so the
custom-op aliasing check is not hit; hopper/flash_attn_interface.py:1068-1075).
"""

import flash_attn_interface as _fai


def paged_gqa_decode_bf16(q, k_cache, v_cache, page_table, cache_seqlens, softmax_scale, out):
    result = _fai.flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        page_table=page_table,
        softmax_scale=softmax_scale,
        causal=True,
    )
    out.copy_(result)
    return None
