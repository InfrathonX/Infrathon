# Paper solution, unmodified upstream API (no glue patch needed): FlashMLA's public
# flash_mla_sparse_fwd allocates its own outputs, so the task ABI costs an extra
# out/max_logits/lse copy inside the timed call (~0.35 ms for 512 MiB out, ~0.18 ms for
# v4_flash).  Conservative lower bound for the paper kernel under the task ABI.
import torch
from flash_mla import flash_mla_sparse_fwd

D_V = 512


def sparse_mla_prefill(q, kv, indices, sm_scale, attn_sink, topk_length, out, max_logits, lse):
    o, ml, l = flash_mla_sparse_fwd(q, kv, indices, float(sm_scale), D_V, attn_sink, topk_length)
    out.copy_(o)
    max_logits.copy_(ml)
    lse.copy_(l)
    return None
