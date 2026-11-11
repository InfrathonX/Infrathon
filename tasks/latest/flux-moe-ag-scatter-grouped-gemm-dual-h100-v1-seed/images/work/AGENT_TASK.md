# MoE all-gather, scatter and grouped GEMM on 2 to 8 GPUs

The seed in `candidates/kernel_v001/submission.py` is correct. Optimize one
self-contained Triton implementation of:

```python
moe_ag_scatter_gemm_bf16(x_local, x_peers, w, sorted_token_ids, sorted_expert_ids,
                         expert_offsets, token_offsets, out, workspace, peer_workspaces,
                         peer_ptrs, workspace_ptrs, rank, world_size) -> None
```

This is the input side of a mixture-of-experts layer under tensor and expert
parallelism (the layer-0 operator of FLUX/Comet): every rank all-gathers the
token shards of the other ranks, scatters the rows routed to its experts into
expert-sorted order, and runs a grouped GEMM with its slice of the expert weights.
Communication, scatter, synchronization and GEMM all count toward the time.

`world_size` ranks (2 to 8, one per GPU) call the function simultaneously; `rank`
is this call's rank. Tokens are numbered globally: rank `p` owns tokens
`[token_offsets[p], token_offsets[p+1])`, and ranks can own different numbers of
tokens. Tensors:

| argument | meaning |
| --- | --- |
| `x_local` | BF16 `[M_rank, K]`, this rank's tokens, in symmetric memory |
| `x_peers` | tuple of `world_size` BF16 views; `x_peers[p]` is rank `p`'s shard `[M_p, K]` (`x_peers[rank]` is `x_local`); remote views are readable through GPU peer access |
| `w` | BF16 `[E_rank, N, K]`, this rank's experts (its slice of the output channels) |
| `sorted_token_ids` | int32 `[R]`, the global token of each row |
| `sorted_expert_ids` | int32 `[R]`, the local expert (`0 <= e < E_rank`) of each row |
| `expert_offsets` | int32 `[E_rank + 1]`; rows of expert `e` are `[expert_offsets[e], expert_offsets[e+1])`, in token order |
| `token_offsets` | int32 `[world_size + 1]` on the device |
| `out` | BF16 `[R, N]`; write every element |
| `workspace` | 4 GiB of local symmetric `uint8` scratch; `peer_workspaces[p]` is rank `p`'s workspace (writable) |
| `peer_ptrs`, `workspace_ptrs` | int64 `[world_size]` device tables of the addresses of `x_peers[p]` and `peer_workspaces[p]`, for indexing peers inside a kernel |

Row `j` is `x_gathered[sorted_token_ids[j]] @ w[sorted_expert_ids[j]].T`, where
`x_gathered` concatenates all shards in rank order. Under expert parallelism a
rank only receives the rows routed to its own experts, so `R` differs between
ranks, can be zero, and a rank may need only some of the peers' tokens. Write
`out` on the current stream and return `None`. All inputs must stay byte-identical.

**Workspace.** The workspace is zeroed once when it is allocated and is never
cleared again: the same workspace serves every call of a run, including
consecutive calls (one per MoE layer) that are issued with no synchronization
between them. Leave it in a state the next call can use; do not assume it is zero.

**Synchronization.** Cross-GPU synchronization may use Triton atomics with
release/acquire semantics and GPU/system scope. Private CUDA streams, host tensor
arithmetic, distributed/NCCL calls, extensions, and dynamic imports are forbidden.
Triton accepts the `x_peers`/`peer_workspaces` tuples as kernel arguments
(index them with `tl.static_range`), and an address from `peer_ptrs` becomes a
pointer with `tl.load(peer_ptrs + p).to(tl.pointer_type(tl.bfloat16))`.

## Support domain

Every call satisfies:

- `world_size` is 2 to 8; ranks are not tied to particular GPUs or NUMA nodes;
- `K` is a multiple of 64 from 64 to 16384 and `N` a multiple of 8 from 8 to 16384;
- `E_rank` is from 1 to 512 and `topk` from 1 to 16; experts can be empty and a
  rank can receive no rows;
- every rank owns at least one token, and all `M` tokens times `K` times 2 bytes fit
  in the workspace with 1 MiB to spare;
- one rank's weight `[E_rank, N, K]` is at most 12 GiB.

## Public and hidden tests

`/workspace/benchmark/public_cases.json` lists the public cases: tensor, expert
and hybrid parallelism, decode, consecutive layers called back to back with no
synchronization (timed together), dense layers (`E_rank = 1`, `topk = 1`),
routing from uniform to heavily skewed, routing replayed from real MoE routers
(the tables in `/workspace/benchmark/routing/`), uneven tokens per rank, and
shuffled or cross-NUMA rank placements, plus correctness-only edge cases. They
are for development and profiling.

The Judge scores a larger hidden suite that differs from the public tests in
models, shapes, token counts, routing and placement. Nothing about it is
disclosed beyond the support domain above, so a good public number does not
guarantee a good Judge score; write your own benchmarks for performance work
across the support domain.

## Score and measurement

A case's work is the mean per-rank GEMM work `2 * M * topk * N * K / EP` (the sum
over ranks does not depend on the routing draw), and case TFLOPS is that work
divided by the slowest rank's time; a sequence case sums the work of its layers.
Time is the larger of the synchronized host wall time and the CUDA-event time,
the median over 7 timed trials. Before every timed trial an untimed call runs
with the same shapes and fresh values; one untimed warm-up per case absorbs
compilation. Every trial uses fresh values; part of each expert's weights is
redrawn before every call.

Cases are grouped into six families: `tp8`, `tp4`, `tp2`, `ep`, `decode` and
`sequence`. A family's score is a weighted geometric mean of its cases' TFLOPS,
and the primary score is the geometric mean of the six family scores; higher is
better. The public tools report the same aggregates over the public cases.

For some cases the candidate also runs once with every peer shard's values in
local tensors and the peer workspaces aliased to the local workspace, so no byte
crosses between GPUs; `hidden_comm =
clamp(1 - max(0, t_fused - t_twin) / t_comm, 0, 1)`, with `t_comm` the trusted
time to pull every peer shard in ring order. It is reported, not scored.

## Accuracy and edge gates

Every call (timed or not) is checked for G1 and G2, and every timed call also for
G7; the FP32 golden and the ordinary BF16 framework reference (G3-G6) may run on
any trial of every case.

| gate | requirement |
| --- | --- |
| G1 | every output finite; all NaN poison overwritten |
| G2 | all inputs byte-identical after the call |
| G3 RelL1 | `<= max(1.25 * reference, 1.0e-3)` |
| G4 CosSim | `>= 0.9999965` |
| G5 row relative-L2 p99.9 | `<= max(1.25 * reference, 1.0e-3)` |
| G6 row relative-L2 maximum | `<= max(1.25 * reference, 1.0e-3)` |
| G7 random projection | `out @ v` matches `x . (W^T v)` per row: relative L2 `<= 2e-2`, every row within `0.2` of the RMS |

## Commands

```bash
./tools/run_one.sh --version 1                                  # one public case
./tools/run_public_tests.sh --quick --candidate-version 1       # one case per family + 4 edges
./tools/run_public_tests.sh --candidate-version 1               # every public case
```

The full public suite runs the edge cases first and each scored case with three
measured trials (one and quick use one). The public tools use all eight GPUs.
Public tools never run or reveal hidden cases.

Before finishing, run the full public suite once on the selected version.

## Submission evaluation policy

Every `sforge-submit` runs the evaluation on the complete hidden suite. `sforge-submit --details` returns the gate states, the score and the six family scores. Hidden cases and per-case results stay on the Judge. Your final result is the best valid score you reach within the time budget, so keep `FINAL_VERSION` on your strongest candidate.

## Submitting

Wait for every public test and compiler process to finish, then run
`sforge-submit --details`. The Judge uses the same eight GPUs, so your container
is paused while it runs. A Judge run covers every hidden case and uses part of
your budget, so iterate on the public suites and submit when a candidate is worth
measuring.

## Source surface

Imports are limited to `math`, `typing`, `torch`, `triton`, and
`triton.language`. Tensor data paths must be reachable Triton kernels. Host code
may inspect metadata, slice/view scratch, allocate safe same-device scratch, and
launch kernels. No host loops, tensor computation/copying, filesystem/process/
network access, module mutation, callbacks outside the reviewed decorator routes,
or dead host helpers. The single-current-stream rule is part of the task. Cache
only compiled code and scheduling metadata; do not keep anything derived from
tensor contents across calls.
