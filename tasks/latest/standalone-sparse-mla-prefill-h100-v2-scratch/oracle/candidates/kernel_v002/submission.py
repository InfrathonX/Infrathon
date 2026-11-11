# Paper solution: FlashMLA SM90 sparse-prefill kernel (deepseek-ai/FlashMLA @ ba89a34,
# csrc/kernels/sm90/prefill/sparse/phase1.cuh, dispatched by csrc/api/sparse_prefill.cpp).
# The kernel source, its tile plan (B_H = 64, B_TOPK = 64, 3 warpgroups: 2 math + 1 gather
# producer) and launch are upstream's, untouched.  The only change is ABI glue
# (../flashmla_out_abi.patch): an extra binding `sparse_prefill_fwd_out` that writes into the
# caller-owned out / max_logits / lse instead of allocating new tensors, so no extra copy
# is timed.  NOT admissible on the task Judge: `import flash_mla` is outside the import
# allowlist and FlashMLA is a compiled CUDA/CUTLASS extension (an existing attention route).
import torch
import flash_mla.cuda as _flash_mla_cuda

D_V = 512


def sparse_mla_prefill(q, kv, indices, sm_scale, attn_sink, topk_length, out, max_logits, lse):
    _flash_mla_cuda.sparse_prefill_fwd_out(
        q, kv, indices, float(sm_scale), D_V, attn_sink, topk_length, out, max_logits, lse
    )
    return None
