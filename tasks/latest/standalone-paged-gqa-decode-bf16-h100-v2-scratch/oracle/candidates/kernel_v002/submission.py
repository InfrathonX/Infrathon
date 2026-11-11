"""Upstream vLLM Triton paged attention (unified 2D / split-KV 3D) on the paged-GQA ABI.

Vendored from vllm-project/vllm at commit 3c39dda2a5da9e4b71b5eb61db686bf669374569:

  vllm/v1/attention/ops/triton_unified_attention.py
      kernel_unified_attention, reduce_segments, unified_attention (launcher)
  vllm/v1/attention/ops/triton_attention_helpers.py
      cdiv_fn, resolve_seq_and_query_len, find_seq_idx, init_softmax_M,
      compute_tile_loop_bounds, store_segm_reduce_scalars, compute_kv_seq_mask,
      softmax_step
  vllm/v1/attention/backends/triton_attn.py
      MIN_LAUNCH_GRID_SIZE_2D = 128, NUM_PAR_SOFTMAX_SEGMENTS = 16, seq_threshold_3D

This is the kernel vLLM's TritonAttentionBackend runs on H100 for paged decode
and chunked prefill.  Changes are glue only:

  1. Constexpr branches that are dead for this ABI (BF16 cache, causal, no
     sliding window / chunking / softcap / sinks / ALiBi / qq-bias / multimodal
     prefix / R-SWA / per-sequence causal, no KV or output quantization, no
     tensor descriptors) are deleted.  Triton eliminates them at compile time
     in upstream as well, so the compiled kernels are the same.
  2. The per-request query offsets (vLLM's query_start_loc) are b * S_q here:
     every load of query_start_len_ptr[i] becomes i * query_len.
  3. Unused optional arguments are removed (the submission rules forbid
     default argument values); everything upstream passes is passed.
  4. The page-table loads are masked to the tile's visible prefix (other=0),
     so a table exactly ceil(L / page_size) wide is never over-read when the
     KV tile is wider than a page.  Upstream reads the table unmasked.
  5. The 3D scratch is allocated per call ([tokens, H_q, 16, 128] FP32 and
     [tokens, H_q, 16] max / expsum) instead of being preallocated by the
     backend's metadata builder; layout and indexing are upstream's.

Launch heuristics are upstream's: BLOCK_M = 16 (or next_pow2(group)),
BLOCK_Q = BLOCK_M // group, 2D tile 32, 3D tile 16 with 16 segments, Triton
default num_warps=4 / num_stages=3, and the 3D (split-KV) path only when
S_q <= 8, causal, and num_seqs * cdiv(S_q, BLOCK_Q) <= 128 // H_kv with the
scratch holding every query token (no speculative decoding configured).
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def find_seq_idx(
    query_len,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = mid * query_len
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1


@triton.jit
def resolve_seq_and_query_len(
    query_len,
    seq_lens_ptr,
    q_block_global_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
):
    seq_idx = find_seq_idx(query_len, q_block_global_idx, num_seqs, BLOCK_Q, True)
    q_block_start_idx = (seq_idx * query_len) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx
    cur_start = seq_idx * query_len
    cur_stop = (seq_idx + 1) * query_len
    cur_batch_query_len = cur_stop - cur_start
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    return seq_idx, q_block_local_idx, cur_start, cur_batch_query_len, seq_len


@triton.jit
def init_softmax_M(BLOCK_M: tl.constexpr):
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    return M


@triton.jit
def compute_tile_loop_bounds(
    context_len,
    seq_len,
    q_block_local_idx,
    segm_idx_or_0,
    tiles_per_segment_or_0,
    TILE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    IS_3D: tl.constexpr,
):
    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    tile_start = 0
    tile_end = num_tiles

    if IS_3D:
        loop_lo = max(segm_idx_or_0 * tiles_per_segment_or_0, tile_start)
        loop_hi = min((segm_idx_or_0 + 1) * tiles_per_segment_or_0, tile_end)
    else:
        loop_lo = tile_start
        loop_hi = tile_end

    return loop_lo, loop_hi, max_seq_prefix_len


@triton.jit
def store_segm_reduce_scalars(
    segm_max_ptr,
    segm_expsum_ptr,
    query_offset_0,
    query_offset_1,
    segm_idx,
    M,
    L,
    query_mask_0,
    query_mask_1,
    num_query_heads: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
):
    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


@triton.jit
def compute_kv_seq_mask(query_abs_pos, seq_offset):
    # USE_CAUSAL branch of upstream compute_kv_seq_mask
    seq_mask = seq_offset[None, :] <= query_abs_pos
    return seq_mask


@triton.jit
def softmax_step(S, M, L):
    # compute running maximum
    # m_j : (BLOCK_M,)
    m_j = tl.maximum(M, tl.max(S, axis=1))
    # Keep an empty row's maximum at -inf for later tiles; use a finite shift
    # only for the exponent arithmetic below.
    safe_m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
    # P : (BLOCK_M, TILE_SIZE)
    P = tl.exp(S - safe_m_j[:, None])
    # l_j : (BLOCK_M,)
    l_j = tl.sum(P, axis=1)
    # alpha : (BLOCK_M, )
    alpha = tl.exp(M - safe_m_j)
    # update constants
    L_new = L * alpha + l_j
    return m_j, L_new, P, alpha


@triton.jit
def kernel_unified_attention(
    output_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    scale,
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride,  # int
    query_stride_0,  # int
    query_stride_1,  # int, should be equal to head_size
    output_stride_0,  # int
    output_stride_1,  # int, should be equal to head_size
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    stride_k_cache_0,  # int
    stride_k_cache_1,  # int
    stride_k_cache_2,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0,  # int
    stride_v_cache_1,  # int
    stride_v_cache_2,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_len,
    BLOCK_Q: tl.constexpr,
    num_seqs,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    IS_3D: tl.constexpr,
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
):
    # ABI change 3b: source admission forbids tl.int64 parameter annotations;
    # upstream's 64-bit stride arithmetic is kept by casting here.
    block_table_stride = tl.cast(block_table_stride, tl.int64)
    query_stride_0 = tl.cast(query_stride_0, tl.int64)
    query_stride_1 = tl.cast(query_stride_1, tl.int64)
    output_stride_0 = tl.cast(output_stride_0, tl.int64)
    output_stride_1 = tl.cast(output_stride_1, tl.int64)
    stride_k_cache_0 = tl.cast(stride_k_cache_0, tl.int64)
    stride_k_cache_1 = tl.cast(stride_k_cache_1, tl.int64)
    stride_k_cache_2 = tl.cast(stride_k_cache_2, tl.int64)
    stride_v_cache_0 = tl.cast(stride_v_cache_0, tl.int64)
    stride_v_cache_1 = tl.cast(stride_v_cache_1, tl.int64)
    stride_v_cache_2 = tl.cast(stride_v_cache_2, tl.int64)

    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2) if IS_3D else 0

    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_len, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    if IS_3D:
        tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
        if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
            return
    else:
        tiles_per_segment = 0

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    M = init_softmax_M(BLOCK_M)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    # acc : (BLOCK_M, HEAD_SIZE_PADDED)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)
    score_scale = scale

    context_len = seq_len - cur_batch_query_len

    loop_lo, loop_hi, max_seq_prefix_len = compute_tile_loop_bounds(
        context_len,
        seq_len,
        q_block_local_idx,
        segm_idx,
        tiles_per_segment,
        TILE_SIZE,
        BLOCK_M,
        BLOCK_Q,
        num_queries_per_kv,
        IS_3D,
    )

    # iterate through tiles
    for j in range(loop_lo, loop_hi):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        # ABI change 4: masked page-table load (upstream: unmasked)
        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE,
            mask=tile_mask,
            other=0,
        ).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )
        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )
        # K : (HEAD_SIZE, TILE_SIZE)
        K = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & tile_mask[None, :],
            other=0.0,
        )
        # V : (TILE_SIZE, HEAD_SIZE)
        V = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & tile_mask[:, None],
            other=0.0,
        )

        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = compute_kv_seq_mask(query_abs_pos, seq_offset)

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        S += score_scale * tl.dot(Q, K)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        M, L, P, alpha = softmax_step(S, M, L)
        acc = acc * alpha[:, None]

        # BF16 values: upstream's p_scale is 1.0 for non-FP8 V
        p_scale = 1.0
        acc += tl.dot((P * p_scale).to(V.dtype), V) / p_scale

    # ---- Epilogue ---------------------------------------------------------
    if IS_3D:
        # Store per-segment partials; finalized by ``reduce_segments``.
        segm_output_offset = (
            query_offset_0[:, None].to(tl.int64)
            * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
            + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
            + segm_idx * HEAD_SIZE_PADDED
            + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
        )
        tl.store(
            segm_output_ptr + segm_output_offset,
            acc,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        )
        store_segm_reduce_scalars(
            segm_max_ptr,
            segm_expsum_ptr,
            query_offset_0,
            query_offset_1,
            segm_idx,
            M,
            L,
            query_mask_0,
            query_mask_1,
            num_query_heads,
            NUM_SEGMENTS_PER_SEQ,
        )
    else:
        acc = acc / L[:, None]
        output_offset = (
            query_offset_0[:, None] * output_stride_0
            + query_offset_1[:, None] * output_stride_1
            + offs_d[None, :]
        )
        tl.store(
            output_ptr + output_offset,
            acc,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        )


@triton.jit
def reduce_segments(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    output_stride_0,  # int
    output_stride_1,  # int, should be equal to head_size
    block_table_stride,  # int
    TILE_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    query_len,
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
):
    output_stride_0 = tl.cast(output_stride_0, tl.int64)
    output_stride_1 = tl.cast(output_stride_1, tl.int64)

    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    # Graph padding must be rejected before sequence lookup or scratch access.
    if query_token_idx >= num_seqs * query_len:
        return

    seq_idx = find_seq_idx(query_len, query_token_idx, num_seqs, BLOCK_Q, False)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    if seq_len <= 0:
        return

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # load segment maxima
    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    # Empty segments must not set the maximum. Unvisited segments retain
    # the M=-inf,L=1 initial state.
    valid_segment = segm_mask & (segm_expsum > 0) & (segm_max != float("-inf"))
    segm_max = tl.where(valid_segment, segm_max, float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums
    segment_scale = tl.where(valid_segment, tl.exp(segm_max - overall_max), 0.0)
    segm_expsum = segm_expsum * segment_scale
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= segment_scale[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    # write result
    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


def paged_gqa_decode_bf16(q, k_cache, v_cache, page_table, cache_seqlens, softmax_scale, out):
    # unified_attention(q.view(B*S_q, H_q, D), k_cache, v_cache, out.view(...),
    #   cu_seqlens_q = [0, S_q, 2 S_q, ...], max_seqlen_q = S_q, seqused_k = cache_seqlens,
    #   causal = True, window_size = (-1, -1), block_table = page_table, softcap = 0)
    num_seqs = cache_seqlens.shape[0]
    query_len = q.shape[1]
    num_tokens = q.shape[0] * q.shape[1]
    num_query_heads = q.shape[2]
    head_size = q.shape[3]
    block_size = v_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    max_seqlen_q = query_len

    # q / out are [B, S_q, H_q, D] contiguous == [B*S_q, H_q, D] in vLLM's layout
    query_stride_0 = q.stride(1)
    query_stride_1 = q.stride(2)
    output_stride_0 = out.stride(1)
    output_stride_1 = out.stride(2)

    BLOCK_M = 16
    if num_queries_per_kv > 16:
        # triton.next_power_of_2(num_queries_per_kv)
        BLOCK_M = 1 << int(math.ceil(math.log2(num_queries_per_kv)))
    BLOCK_Q = BLOCK_M // num_queries_per_kv

    total_num_q_blocks = num_tokens // BLOCK_Q + num_seqs

    # _get_tile_size: head 128, no sliding window, BF16
    TILE_SIZE_PREFILL = 32
    TILE_SIZE_DECODE = 16

    # backends/triton_attn.py: MIN_LAUNCH_GRID_SIZE_2D // num_heads_kv, no CUDA graphs
    seq_threshold_3D = 128 // num_kv_heads
    # NUM_PAR_SOFTMAX_SEGMENTS (64 only on SM120)
    num_par_softmax_segments = 16
    # scratch rows = min(seq_threshold_3D, max_num_seqs) * max_query_len_3d (1 without spec decode)
    max_num_tokens_3d = seq_threshold_3D

    # MAX_3D_QUERY_LEN = 8
    use_3d = True
    if max_seqlen_q > 8:
        use_3d = False
    if num_seqs * triton.cdiv(max_seqlen_q, BLOCK_Q) > seq_threshold_3D:
        use_3d = False
    if max_num_tokens_3d < num_tokens:
        # "Split-K scratch rows < query tokens; using 2D."
        use_3d = False

    if use_3d:
        softmax_segm_output = torch.empty(
            (num_tokens, num_query_heads, num_par_softmax_segments, head_size),
            dtype=torch.float32,
            device=q.device,
        )
        softmax_segm_max = torch.empty(
            (num_tokens, num_query_heads, num_par_softmax_segments),
            dtype=torch.float32,
            device=q.device,
        )
        softmax_segm_expsum = torch.empty(
            (num_tokens, num_query_heads, num_par_softmax_segments),
            dtype=torch.float32,
            device=q.device,
        )
        kernel_unified_attention[(total_num_q_blocks, num_kv_heads, num_par_softmax_segments)](
            out,
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            softmax_scale,
            num_query_heads,
            num_queries_per_kv,
            page_table.stride(0),
            query_stride_0,
            query_stride_1,
            output_stride_0,
            output_stride_1,
            block_size,
            TILE_SIZE_DECODE,
            head_size,
            head_size,
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            v_cache.stride(3),
            query_len,
            BLOCK_Q,
            num_seqs,
            BLOCK_M,
            num_par_softmax_segments,
            True,
            softmax_segm_output,
            softmax_segm_max,
            softmax_segm_expsum,
            num_warps=4,
            num_stages=3,
        )
        reduce_segments[(num_tokens, num_query_heads)](
            out,
            softmax_segm_output,
            softmax_segm_max,
            softmax_segm_expsum,
            cache_seqlens,
            num_seqs,
            num_query_heads,
            output_stride_0,
            output_stride_1,
            page_table.stride(0),
            TILE_SIZE_DECODE,
            head_size,
            head_size,
            query_len,
            BLOCK_Q,
            num_par_softmax_segments,
            num_warps=4,
            num_stages=3,
        )
    else:
        kernel_unified_attention[(total_num_q_blocks, num_kv_heads)](
            out,
            q,
            k_cache,
            v_cache,
            page_table,
            cache_seqlens,
            softmax_scale,
            num_query_heads,
            num_queries_per_kv,
            page_table.stride(0),
            query_stride_0,
            query_stride_1,
            output_stride_0,
            output_stride_1,
            block_size,
            TILE_SIZE_PREFILL,
            head_size,
            head_size,
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            v_cache.stride(3),
            query_len,
            BLOCK_Q,
            num_seqs,
            BLOCK_M,
            1,
            False,
            None,
            None,
            None,
            num_warps=4,
            num_stages=3,
        )
    return None
