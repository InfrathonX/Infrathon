"""Seed candidate: MoE all-gather, scatter and grouped GEMM across W GPUs, self-contained Triton.

Structure (single stream, three launches per call):
  * `epoch_kernel`: bumps the call counter kept in this rank's workspace header.  Chunk
    flags are never cleared: a remote chunk is ready for this call when its flag equals
    the current epoch, so the workspace can be reused by consecutive calls.
  * `count_local_kernel`: per expert, how many of its rows precede this rank's own token
    range and how many fall inside it (rows of an expert are in token order).
  * `moe_ag_scatter_kernel`: programs 0..N_COMM-1 are communication producers that pull
    every peer's token shard into the local symmetric workspace chunk by chunk, peers in
    ring order starting at rank + 1, and release a per-chunk flag; the remaining programs
    are grouped-GEMM consumers.  Consumer tiles are scheduled in two stages: first every
    tile whose rows are all local (no wait), then the tiles that touch remote rows, which
    acquire the flags of every chunk they need before loading A.
Tiles: 64x128x32 with 4 warps and 4 stages.
Numerics: BF16 operands, FP32 accumulation, BF16 output.
"""
import torch
import triton
import triton.language as tl

BLOCK_M = 64
BLOCK_N = 128
BLOCK_K = 32
GROUP_M = 8
E_PAD = 512
CHUNK_TOKENS = 512
N_COMM = 8
COPY_BLOCK = 8192
MAX_WORLD = 8
FLAG_STRIDE = 1024                 # chunk flags per peer (up to 524288 tokens per rank)
EPOCH_INTS = 64                    # header ints [0, 64): call counter at [0]
FLAG_INTS = MAX_WORLD * FLAG_STRIDE
COUNT_INTS = 2 * E_PAD             # per-expert rows before / inside this rank's range
HEADER_BYTES = 65536
HEADER_INTS_USED = EPOCH_INTS + FLAG_INTS + COUNT_INTS


@triton.jit
def epoch_kernel(epoch):
    tl.store(epoch, tl.load(epoch) + 1)


@triton.jit
def count_local_kernel(sorted_token_ids, expert_offsets, token_offsets, before_counts,
                       local_counts, rank, BLOCK: tl.constexpr):
    e = tl.program_id(0)
    start = tl.load(expert_offsets + e)
    end = tl.load(expert_offsets + e + 1)
    lo = tl.load(token_offsets + rank)
    hi = tl.load(token_offsets + rank + 1)
    acc_before = tl.zeros([BLOCK], dtype=tl.int32)
    acc_local = tl.zeros([BLOCK], dtype=tl.int32)
    for base in range(start, end, BLOCK):
        rows = base + tl.arange(0, BLOCK)
        mask = rows < end
        tok = tl.load(sorted_token_ids + rows, mask=mask, other=-1)
        acc_before += (mask & (tok < lo)).to(tl.int32)
        acc_local += ((tok >= lo) & (tok < hi)).to(tl.int32)
    tl.store(before_counts + e, tl.sum(acc_before, axis=0))
    tl.store(local_counts + e, tl.sum(acc_local, axis=0))


@triton.jit
def moe_ag_scatter_kernel(
    x_local, xbuf, w, sorted_token_ids, expert_offsets, token_offsets, peer_ptrs,
    before_counts, local_counts, flags, epoch_ptr, out,
    N, K, E, rank, world,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr,
    E_PAD: tl.constexpr, CHUNK: tl.constexpr, N_COMM: tl.constexpr, COPY_BLOCK: tl.constexpr,
    FLAG_STRIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    epoch = tl.load(epoch_ptr)
    lo = tl.load(token_offsets + rank)
    hi = tl.load(token_offsets + rank + 1)
    if pid < N_COMM:
        # ---------------- communication producer: pull peer chunks in ring order, release flags
        for step in range(1, world):
            p = (rank + step) % world
            p_lo = tl.load(token_offsets + p)
            p_hi = tl.load(token_offsets + p + 1)
            total = (p_hi - p_lo).to(tl.int64) * K
            src = tl.load(peer_ptrs + p).to(tl.pointer_type(tl.bfloat16))
            dst = xbuf + p_lo.to(tl.int64) * K
            n_chunks = (p_hi - p_lo + CHUNK - 1) // CHUNK
            for c in range(pid, n_chunks, N_COMM):
                base = c.to(tl.int64) * CHUNK * K
                for i in range(0, CHUNK * K, COPY_BLOCK):
                    offs = base + i + tl.arange(0, COPY_BLOCK)
                    m = offs < total
                    v = tl.load(src + offs, mask=m)
                    tl.store(dst + offs, v, mask=m)
                tl.debug_barrier()
                tl.atomic_xchg(flags + p * FLAG_STRIDE + c, epoch, sem="release", scope="gpu")
        return
    # ---------------- grouped-GEMM consumer
    t = pid - N_COMM
    eidx = tl.arange(0, E_PAD)
    emask = eidx < E
    off_lo = tl.load(expert_offsets + eidx, mask=emask, other=0)
    off_hi = tl.load(expert_offsets + eidx + 1, mask=emask, other=0)
    cnt = off_hi - off_lo
    before = tl.load(before_counts + eidx, mask=emask, other=0)
    loc = tl.load(local_counts + eidx, mask=emask, other=0)
    tiles = (cnt + BLOCK_M - 1) // BLOCK_M
    # Rows inside an expert are in token order, so this rank's rows form one contiguous run
    # [before, before + loc).  Stage 0 = tiles entirely inside that run.
    first_local = (before + BLOCK_M - 1) // BLOCK_M
    end_local = (before + loc) // BLOCK_M
    tiles0 = tl.maximum(end_local - first_local, 0)
    tiles1 = tiles - tiles0
    cum0 = tl.cumsum(tiles0, axis=0)
    cum1 = tl.cumsum(tiles1, axis=0)
    total0 = tl.sum(tiles0, axis=0)
    total1 = tl.sum(tiles1, axis=0)
    num_block_m = total0 + total1
    num_block_n = tl.cdiv(N, BLOCK_N)
    if t >= num_block_m * num_block_n:
        return
    num_in_group = GROUP_M * num_block_n
    group_id = t // num_in_group
    first_m = group_id * GROUP_M
    group_size = tl.minimum(num_block_m - first_m, GROUP_M)
    pid_m = first_m + (t % num_in_group) % group_size
    pid_n = (t % num_in_group) // group_size
    # map the stage-ordered m-tile to (expert, tile-within-expert)
    if pid_m < total0:
        e = tl.sum((cum0 <= pid_m).to(tl.int32), axis=0)
        excl = tl.sum(tl.where(eidx == e, cum0 - tiles0, 0), axis=0)
        j = tl.sum(tl.where(eidx == e, first_local, 0), axis=0) + (pid_m - excl)
    else:
        q = pid_m - total0
        e = tl.sum((cum1 <= q).to(tl.int32), axis=0)
        excl = tl.sum(tl.where(eidx == e, cum1 - tiles1, 0), axis=0)
        i1 = q - excl
        fl = tl.sum(tl.where(eidx == e, first_local, 0), axis=0)
        t0 = tl.sum(tl.where(eidx == e, tiles0, 0), axis=0)
        j = tl.where(i1 < fl, i1, i1 + t0)
    row_lo = tl.load(expert_offsets + e)
    row_hi = tl.load(expert_offsets + e + 1)
    rows = row_lo + j * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < row_hi
    tok = tl.load(sorted_token_ids + rows, mask=row_mask, other=lo)
    is_remote = row_mask & ((tok < lo) | (tok >= hi))
    # wait for every remote chunk this tile needs
    if tl.max(is_remote.to(tl.int32), axis=0) > 0:
        for step in range(1, world):
            p = (rank + step) % world
            p_lo = tl.load(token_offsets + p)
            p_hi = tl.load(token_offsets + p + 1)
            sel = is_remote & (tok >= p_lo) & (tok < p_hi)
            c_max = tl.max(tl.where(sel, (tok - p_lo) // CHUNK, -1), axis=0)
            if c_max >= 0:
                c_min = tl.min(tl.where(sel, (tok - p_lo) // CHUNK, FLAG_STRIDE), axis=0)
                for c in range(c_min, c_max + 1):
                    while tl.atomic_add(flags + p * FLAG_STRIDE + c, 0, sem="acquire", scope="gpu") != epoch:
                        pass
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    a_loc = x_local + (tok - lo).to(tl.int64)[:, None] * K + offs_k[None, :]
    a_rem = xbuf + tok.to(tl.int64)[:, None] * K + offs_k[None, :]
    a_ptrs = tl.where(is_remote[:, None], a_rem, a_loc)
    b_ptrs = w + e.to(tl.int64) * N * K + offs_n[None, :].to(tl.int64) * K + offs_k[:, None]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k_mask = offs_k < K - k0
        a = tl.load(a_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K
    c_ptrs = out + rows.to(tl.int64)[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=row_mask[:, None] & n_mask[None, :])


def moe_ag_scatter_gemm_bf16(x_local, x_peers, w, sorted_token_ids, sorted_expert_ids,
                             expert_offsets, token_offsets, out, workspace, peer_workspaces,
                             peer_ptrs, workspace_ptrs, rank, world_size):
    K = x_local.shape[1]
    E = w.shape[0]
    N = w.shape[1]
    R = out.shape[0]
    assert E <= E_PAD and world_size <= MAX_WORLD
    header = workspace[:HEADER_BYTES].view(torch.int32)
    epoch = header[:1]
    flags = header[EPOCH_INTS:EPOCH_INTS + FLAG_INTS]
    before_counts = header[EPOCH_INTS + FLAG_INTS:EPOCH_INTS + FLAG_INTS + E_PAD]
    local_counts = header[EPOCH_INTS + FLAG_INTS + E_PAD:HEADER_INTS_USED]
    xbuf = workspace[HEADER_BYTES:].view(torch.bfloat16)
    epoch_kernel[(1,)](epoch)
    count_local_kernel[(E,)](sorted_token_ids, expert_offsets, token_offsets, before_counts,
                             local_counts, rank, BLOCK=1024)
    max_tiles = ((R + BLOCK_M - 1) // BLOCK_M + E) * ((N + BLOCK_N - 1) // BLOCK_N)
    moe_ag_scatter_kernel[(N_COMM + max_tiles,)](
        x_local, xbuf, w, sorted_token_ids, expert_offsets, token_offsets, peer_ptrs,
        before_counts, local_counts, flags, epoch, out,
        N, K, E, rank, world_size,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M, E_PAD=E_PAD,
        CHUNK=CHUNK_TOKENS, N_COMM=N_COMM, COPY_BLOCK=COPY_BLOCK, FLAG_STRIDE=FLAG_STRIDE,
        num_warps=4, num_stages=4,
    )
