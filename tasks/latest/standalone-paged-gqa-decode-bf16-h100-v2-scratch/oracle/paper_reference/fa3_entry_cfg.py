"""Sweep variant of fa3_entry.py (fairness check, 2026-10-29): FA3 knobs from env.
FA3_NUM_SPLITS (0 = heuristic), FA3_PACK_GQA (none|true|false), optional FA3_TUNED_JSON
{"B,S_q,H_q,H_kv,page": [num_splits, "none|true|false"]} per-shape override."""
import json
import os

import torch
import flash_attn_3._C  # noqa: F401  registers torch.ops.flash_attn_3

_fwd = torch.ops.flash_attn_3.fwd
_M = {"none": None, "true": True, "false": False}
_NS = int(os.environ.get("FA3_NUM_SPLITS", "0"))
_PG = _M[os.environ.get("FA3_PACK_GQA", "none")]
_TUNED = json.loads(open(os.environ["FA3_TUNED_JSON"]).read()) if os.environ.get("FA3_TUNED_JSON") else {}
_TUNED = {k: (v[0], _M[v[1]]) for k, v in _TUNED.items()}


def paged_gqa_decode_bf16(q, k_cache, v_cache, page_table, cache_seqlens, softmax_scale, out):
    ns, pg = _NS, _PG
    if _TUNED:
        key = f"{q.shape[0]},{q.shape[1]},{q.shape[2]},{k_cache.shape[2]},{k_cache.shape[1]}"
        ns, pg = _TUNED.get(key, (ns, pg))
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
        ns,  # num_splits
        pg,  # pack_gqa
        0,  # sm_margin
    )
    return None
