# Paged-KV block transfer

Work in `/app/workdir/kv_transfer_lab` on the NVIDIA GPU allocated to you
(compute capability 8.0 or newer, at least 64 GiB of device memory). A correct
CUDA implementation is supplied as `candidates/kernel_v001/submission.cu`. Make
it faster without changing what it moves. The score is effective bandwidth in
GB/s, with no cap.

## Why this kernel matters

A serving engine keeps its KV cache in paged per-layer buffers. A KV cache
layer offloads that cache to CPU memory and loads it back when a later request
reuses the same prefix, so the engine skips recomputing it. This kernel is the
hop between the engine's pages and the cache layer's contiguous chunks. It
runs on every cache hit and every offload, alongside the engine's own compute,
for every model and parallel split the engine serves.

## What the kernel does

Chunk objects are contiguous, token-major buffers of logical shape
`[kv, NL, T, NH*HS]`. Your kernel moves whole 16-token blocks between the
engine's pages and chunk objects, for 1-4 chunk objects per call, with an
arbitrary (unique, in-range) block-id map and an optional skipped prefix:

- `store`: paged engine buffers -> chunk objects;
- `retrieve`: chunk objects -> paged engine buffers.

Two engine layouts exist:

- `split`: per-layer tensor `[2, NB, NH, BS, HS]`, `kv=2`;
- `fused`: per-layer tensor `[NB, NH, BS, 2*HS]`, `kv=1`.

`BS=16`. `params->hs` is the per-head width: the head size for `split`, twice
the head size for `fused`. Elements are 2 bytes (BF16 storage). In every
family except `device-large` and `device-small` (see Workloads), chunk objects
live in mapped pinned host memory: their pointers are host virtual addresses
that your kernel dereferences directly over PCIe, or that you copy with
`cudaMemcpy*Async`.

## Support domain

Every call satisfies:

- `NL` is any integer from 1 to 128, and `NH` any integer from 1 to 32;
- the head size is 64 or 128;
- `NB` is any integer from 1 to 65536;
- `T` is any positive multiple of 16, with 1-4 objects and
  `num_objects * T / 16 <= NB`;
- `skip` is any integer from 0 to the total block count;
- the block map is unique and in range, in any order;
- a chunk object can be larger than the scratch arena.

## Entry point

`submission.cu` must define exactly this, against the read-only header
`transfer_abi.h` (a copy sits in this directory; the build uses the canonical
one):

```c
extern "C" void lmcache_paged_kv_block_transfer(const TransferParams* params,
                                                cudaStream_t stream);
```

All work must be enqueued on `stream` and complete with it. You never see an
invalid descriptor. `params->device_scratch` is a 256 MiB device arena you may
use within a call; nothing survives between calls. Calls that run at the same
time on different streams never share a scratch arena.

## Correctness

- Every targeted destination byte must match the source bit for bit, including
  NaN patterns, infinities and signed zeros. There is no tolerance.
- Untargeted destination bytes (skipped prefix blocks, unselected pool blocks)
  must stay unchanged.
- Sources, the block-id table and the pointer array are read-only.

Every timed pass is verified in full: every stored chunk, every retrieved
block, and every other byte of the engine pool.

## Workloads

A case is a list of back-to-back calls ("waves") against one engine pool.
Cases model one GPU of a serving engine, in seven families:

| Family | Calls |
| --- | --- |
| `device-large` | chunks in device memory, at least 16 MiB per call |
| `device-small` | chunks in device memory, under 16 MiB per call |
| `host-large` | chunks in pinned host memory, at least 16 MiB per call |
| `host-small` | chunks in pinned host memory, under 16 MiB per call |
| `host-duplex` | host stores and host retrieves at the same time, on two streams, over disjoint blocks of one pool |
| `trace` | a window of a serving session: stores and retrieves of different sizes, mixed, against one pool |
| `overlap` | host calls while the GPU also runs other work on another stream |

Shapes come from real models split across 1-8 GPUs (layers, KV heads per GPU,
head size), in both layouts. The pool is
sized like a serving engine's: tens of gigabytes. Block maps come from a
serving engine's allocator, so they contain runs of adjacent blocks in either
direction mixed with scattered blocks across the whole pool.
Small calls are the last partial chunk of a request, a chunk stored during
decoding, or one chunk of a deep tensor-parallel split. Every trial moves the
whole map by one offset, so addresses change between trials.

**Public tests.** `/workspace/benchmark/public_cases.json` holds the public
cases; they cover every family and both layouts. They are for development and
profiling. A good public number does not guarantee a good Judge score; write
your own benchmarks for performance work across the support domain.

**Hidden tests.** Your score comes from hidden cases on the Judge. They use
other models, other tensor-parallel splits, other pool sizes and other serving
traces than the public tests, and there are many more of them. Keep every path
correct across the support domain and make optimizations general rather than
fitting particular shapes.

## Score and measurement

```text
case GB/s     = logical payload bytes moved / median trial seconds
                (host-duplex: bytes in both directions)
overlap case  = case GB/s * (1 - foreground loss)
shape score   = geometric mean of the shape's cases in the family
family score  = geometric mean over groups of models of the geometric mean
                of each group's shape scores
device        = geometric mean of device-large and device-small
host          = geometric mean of host-large, host-small and host-duplex
score         = device^0.35 * host^0.25 * trace^0.10 * overlap^0.30
```

A shape is one (layout, layers, KV heads per GPU, head size). Every shape
counts once in its family however many cases it has. The hidden tests split
their shapes into two groups of models that count equally; the public tests
have one group.

Logical bytes do not depend on your implementation: staging through scratch
earns credit only for the payload. A trial's time is the larger of the CUDA
event time and the host wall time from the first call to the end of the last.
Each case runs one untimed, verified pass first, then three timed, verified
trials; an `overlap` case also times its transfer three times on its own first.

In a `host-duplex` case the store calls go to one stream and the retrieve
calls to another, interleaved; a trial ends when both streams are done. The
host link carries both directions at once, so this family can exceed the
one-way host bandwidth.

In an `overlap` case a second stream runs other work (a large BF16 matrix
multiply, or a large device-to-device copy) repeated for about three times the
transfer's own time, and longer if needed, so it covers the whole transfer. The
foreground loss is the extra time that work takes because your transfer ran
beside it, divided by the transfer time, clamped to [0, 0.95]. A transfer that
occupies the GPU's SMs slows that work down and is charged for it.

## Allowed implementation surface

One CUDA C++ translation unit per candidate: `candidates/kernel_vNNN/submission.cu`.

- Includes: `transfer_abi.h`, `cuda_runtime.h`, `stdint.h`/`cstdint`,
  `stddef.h`/`cstddef`, `type_traits`.  Nothing else.
- CUDA runtime API allowlist: kernel launches (including cooperative),
  `cudaMemcpy*Async` / `cudaMemset*Async` **on the given stream**, error
  queries, occupancy/attribute introspection, `cudaFuncSetAttribute` (for
  large dynamic shared memory).  Everything else is rejected — in source
  scan **and** by an undefined-symbol audit of your compiled binary.
- Inline PTX `asm` is allowed in device code (the seed itself uses
  `ld.global.cs`/`st.global.cs`); host-ISA asm is rejected.
- Forbidden and enforced fail-closed: any allocation (`cudaMalloc*`, host
  allocation, `__managed__`), private streams or events, any
  synchronisation call, CUDA graphs and stream capture, the driver API,
  `cudaMemcpy`/`cudaMemset` (synchronous), peer access, host callbacks,
  threads, files, processes, loaders, environment access, persistence of
  results across calls, and reading `params` after your function returns.
- The compile command is frozen; you do not control flags, and the public runner
  uses the same command. Write architecture-portable CUDA C++.

## Commands

```bash
./tools/run_one.sh --version 1
./tools/run_public_tests.sh --quick --candidate-version 1
./tools/run_public_tests.sh --candidate-version 1
```

`run_one.sh` runs one case, quick runs one case per family, and the full suite
runs every public case. Each compiles with the frozen command, audits the
binary, and runs the public edge cases first. Public tools never run hidden cases.

Keep versions at `candidates/kernel_vNNN/submission.cu` with three-digit
directories, write the selected integer to `FINAL_VERSION`, and keep concise
measurements in `notes.md`. Before finishing, run the full public suite once on
the selected version.

## Submission evaluation policy

Every `sforge-submit` runs the evaluation on hidden test cases.
`sforge-submit --details` returns the gate states, the score, the four group
scores and the seven family scores. Hidden cases, their shapes and per-case
results stay on the Judge. Your final result is the best valid score you reach
within the time budget, so keep `FINAL_VERSION` on your strongest candidate.

## Submitting

Wait for every public test and compiler process to finish, then run
`sforge-submit --details`. A Judge run uses part of your budget.
