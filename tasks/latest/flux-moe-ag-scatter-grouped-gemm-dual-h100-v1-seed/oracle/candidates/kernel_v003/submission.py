"""FORCED-ORACLE candidate: ByteDance Flux CUTLASS AG+scatter grouped GEMM (SM90).

NOT an admissible agent submission: it imports the compiled `flux` extension
(bytedance/flux @ 19831ca2d820e3e782ed1d15d8b52d0898b78b26, built for sm_90a
with NVSHMEM) and calls torch.distributed, which only the LOCAL-ONLY
forced-oracle Judge image provides (oracle/paper_reference/forced_judge/).

Paper path: flux.GemmGroupedV3AGScatter, the Hopper MoE layer-0 operator of
FLUX (arXiv:2406.06858) / Comet (arXiv:2502.19811): each rank copies its shard
into an NVSHMEM symmetric buffer, a copy stream pulls the peer shard while a
warp-specialised CUTLASS 3.x grouped GEMM with a fused gather (TMA A-fetch) and
scatter epilogue consumes local tiles first and remote tiles as their arrival
flags flip. Tile configuration: Flux's own autotuner. On the first (untimed
warm-up) call of every new shape both ranks run `op.profiling(...)` (Flux's
official tuning path, as in test/python/moe_ag_scatter/test_moe_ag.py --tune:
every registered GEMM config is timed, the max over ranks is reduced through
NVSHMEM so both ranks pick the same one) and `flux.load_tuning_record(...)`
installs the winner for that (meta, runtime-config) key; later calls use it.

Glue (the only non-Flux code):
  * one op per (K, N, E, topk, M_local) built lazily on the first (untimed
    warm-up) call of a case, both ranks in lock-step;
  * the task's expert-sorted gather list `sorted_token_ids` is turned into the
    [ntokens, topk] `scatter_index` Flux expects (row j of token t goes to slot
    k = its arrival order among t's rows; any bijection is equivalent because
    Flux only uses its inverse, the gather index), and `expert_offsets` into
    `splits`. Both are single small device kernels on the current stream.
  * the output is written in place into `out` (outputs_buf), on the current
    stream; `x_peer`, `workspace` and `peer_workspace` are not used because
    Flux performs its own all-gather through NVSHMEM.
"""
import os

os.environ.setdefault("NVSHMEM_SYMMETRIC_SIZE", str(8 << 30))
os.environ.setdefault("NVSHMEM_REMOTE_TRANSPORT", "none")
os.environ.setdefault("NVSHMEM_DISABLE_NCCL", "1")
os.environ.setdefault("NVSHMEM_BOOTSTRAP", "UID")

import torch
import torch.distributed as dist
import triton
import triton.language as tl

import flux

_STATE = {"shm": False, "ops": {}, "tuned": set()}


@triton.jit
def _scatter_index_kernel(sti_ptr, counter_ptr, scatter_ptr, R, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < R
    token = tl.load(sti_ptr + offs, mask=mask, other=0)
    slot = tl.atomic_add(counter_ptr + token, 1, mask=mask, sem="relaxed")
    tl.store(scatter_ptr + token * TOPK + slot, offs.to(tl.int32), mask=mask)


def _get_op(m_local, K, N, E, topk, dtype):
    group = dist.group.WORLD
    if not _STATE["shm"]:
        flux.init_flux_shm(group)
        torch.cuda.synchronize()
        _STATE["shm"] = True
    key = (m_local, K, N, E, topk, dtype)
    op = _STATE["ops"].get(key)
    if op is None:
        world = dist.get_world_size(group)
        tp_env = flux.DistEnvTPWithEP(tp_group=group, nnodes=1, ep_group=None)
        moe_args = flux.MoeArguments(
            max_ntokens=m_local * world,
            hidden=K,
            ffn_hidden=N * world,
            nexperts=E,
            topk=topk,
            input_dtype=dtype,
            output_dtype=dtype,
        )
        op = flux.GemmGroupedV3AGScatter(tp_env=tp_env, moe_args=moe_args)
        _STATE["ops"][key] = op
    return op


def moe_ag_scatter_gemm_bf16(x_local, x_peer, w, sorted_token_ids, sorted_expert_ids,
                             expert_offsets, out, workspace, peer_workspace, rank):
    m_local, K = int(x_local.shape[0]), int(x_local.shape[1])
    E, N = int(w.shape[0]), int(w.shape[1])
    R = int(out.shape[0])
    ntokens = 2 * m_local
    topk = R // ntokens
    op = _get_op(m_local, K, N, E, topk, x_local.dtype)
    counter = torch.zeros(ntokens, dtype=torch.int32, device=x_local.device)
    scatter_index = torch.empty((ntokens, topk), dtype=torch.int32, device=x_local.device)
    BLOCK = 1024
    _scatter_index_kernel[(triton.cdiv(R, BLOCK),)](
        sorted_token_ids, counter, scatter_index, R, TOPK=topk, BLOCK=BLOCK, num_warps=4
    )
    splits = (expert_offsets[1:] - expert_offsets[:-1]).to(torch.int32)
    key = (m_local, K, N, E, topk, x_local.dtype)
    if key not in _STATE["tuned"]:
        prof_ctx = flux.ProfilingContext("moe_ag_scatter_forced_oracle")
        op.profiling(
            inputs_shard=x_local,
            weights=[w],
            splits_gpu=splits,
            scatter_index=scatter_index,
            outputs_buf=[out],
            fast_accum=False,
            sm_margin=0,
            prof_ctx=prof_ctx,
        )
        flux.load_tuning_record(prof_ctx.get_latest_record())
        _STATE["tuned"].add(key)
    op.forward(
        inputs_shard=x_local,
        weights=w,
        splits_gpu=splits,
        scatter_index=scatter_index,
        outputs_buf=out,
        fast_accum=False,
        sm_margin=0,
    )
    return None
