"""Upstream SageAttention2 sm90 (wgmma) CUDA path behind the task ABI -- FORCED ORACLE, NOT admissible.

Upstream: thu-ml/SageAttention @ d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5 (Apache-2.0),
sageattention/core.py::sageattn_qk_int8_pv_fp8_cuda_sm90 -- what upstream's `sageattn()` dispatches
to on sm90 -- with tensor_layout="NHD", qk_quant_gran="per_thread", pv_accum_dtype="fp32+fp32",
smooth_k=True, return_lse=False.  QK quantizer: upstream's default QK quantizer (Triton per_thread_int8).

The only change from upstream: the CUDA kernel writes straight into the caller's `out` instead of a
fresh `o = torch.empty(...)` (same dtype/shape; head_dim is 64 or 128 so no padding/slicing happens).
Everything upstream does on the host (k.mean, quantization, V zero-padding to a multiple of 128,
transpose/pad/permute + FP8 quant of V) stays inside the call and is timed by the Judge.

It imports the compiled `sageattention` extension, which the task's source admission forbids, so it
can only be scored in the local-only forced-oracle Judge image (oracle/paper_reference/forced_judge/).
"""

import torch
from sageattention import sm90_compile
from sageattention.core import SM90_ENABLED
from sageattention.quant import per_channel_fp8
from sageattention.quant import per_warp_int8 as per_warp_int8_cuda
from sageattention.triton.quant_per_thread import per_thread_int8 as per_thread_int8_triton

_QK_GRAN = "per_thread"


def quantized_attention_bf16(q, k, v, is_causal, softmax_scale, out):
    assert SM90_ENABLED, "sageattention._qattn_sm90 not built"
    tensor_layout = "NHD"
    _tensor_layout = 0
    _is_caual = 1 if is_causal else 0
    _qk_quant_gran = 3 if _QK_GRAN == "per_thread" else 2
    _return_lse = 0
    torch.cuda.set_device(v.device)
    head_dim_og = q.size(-1)
    assert head_dim_og in (64, 128)
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1
    sm_scale = float(softmax_scale)
    seq_dim = 1

    # smooth_k=True (upstream)
    km = k.mean(dim=seq_dim, keepdim=True)

    if _QK_GRAN == "per_warp":
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, km, tensor_layout=tensor_layout, BLKQ=64, WARPQ=16, BLKK=128)
    else:
        q_int8, q_scale, k_int8, k_scale = per_thread_int8_triton(q, k, km, tensor_layout=tensor_layout, BLKQ=64, WARPQ=16, BLKK=128, WARPK=128)

    # upstream: o = torch.empty(q.size(), dtype=dtype, device=q.device)  -> write into `out`
    o = out

    kv_len = k.size(seq_dim)
    v_pad_len = 128 - (kv_len % 128) if kv_len % 128 != 0 else 0
    if v_pad_len > 0:
        v = torch.cat([v, torch.zeros(v.size(0), v_pad_len, v.size(2), v.size(3), dtype=v.dtype, device=v.device)], dim=1)

    v_fp8, v_scale, _ = per_channel_fp8(v, tensor_layout=tensor_layout, smooth_v=False)

    sm90_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(
        q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale,
        _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    return None
