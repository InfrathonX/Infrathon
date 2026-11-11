# StreamDiffusionV2 dual-GPU pipeline runtime v2 — participant brief

Make one live video stream run as fast as possible on two CUDA GPUs while every
output stays bit-exact. The pristine runtime_v001 is a static 15/15 two-rank lag
ring. Keep every attempt as a new candidates/runtime_vNNN package and leave
FINAL_VERSION on the strongest version that passes the complete public suite.

The score counts one stream. Each case is a single continuous video; its chunks
depend on earlier chunks through the per-step rolling KV caches and the streaming
VAE caches. How you use the two GPUs is up to you: split blocks, split denoising
steps, replicate weights, move the VAE, or run everything on one GPU. Running two
independent copies of the stream does not help, because each case has only one
stream and an independent copy loses the other chunks' cache history (parity
fails). Using a single GPU is legal.

Requirements: two CUDA devices, compute capability >= 8.0, >= 26 GiB each,
>= 64 GiB usable host RAM, NCCL across the pair, cu124 / torch 2.6.0 /
flash-attn 2.7.4.post1.

## Frozen and candidate-owned surfaces

Root-owned sdv2_core and the Judge own weights and numerical math: patchification, all
30 Wan DiT blocks and attention projections, head, flow-to-x0/add-noise, streaming VAE,
input clips, prompt tensors, indexed noise, plan metadata, output sink, clock, goldens, and
gate thresholds. The protocol is 5 prepare frames followed by 4-frame chunks;
steps=[700,500,400,200][:S], cache length 6, three sink tokens, adaptive threshold 0.2,
position refresh at latent frame 50, and BF16 math.

You own placement, complete KV/policy migration, activation/state transport, streams and
events, VAE/DiT/transfer overlap, exact rolling-KV bookkeeping, lane scheduling, buffers,
and CUDA graphs. You may spawn descendants of the driver. Both logical devices are
cuda:0,cuda:1; every process must exit and release both GPUs.

## Required API

The file candidates/runtime_vNNN/runtime.py must export Runtime with this surface:

    Runtime(core_loader, devices, plan_meta, sink, *, master_port=None)
    start(prompt_embeds, first_chunk_frames) -> None
    push(chunk_idx, frames) -> None
    flush() -> None
    close() -> RunLedger

`plan_meta` is a `StreamPlanMeta`: chunk descriptors (noise scale, motion-derived
first step, cache positions), the indexed noise provider and cache geometry. It
carries no input video. Frames reach the runtime only through `start` (chunk 0)
and `push` (every later chunk).

The sink records only emissions (`emit_latent`, `emit_frames`). The controller,
not the runtime, records when each chunk is pushed: it takes the timestamp
immediately before calling `start` or `push`. Do not read, write or reinterpret
those records.

The controller passes master_port explicitly per case. The optional None default
selects a free loopback TCP port at construction. Accept this keyword in Runtime.

Construction (model loads, spawn, process-group setup) is timed separately and
gated (see below). start prepares and emits chunk 0. push may enqueue
asynchronously but must never drop work. The controller calls every push
back-to-back in throughput mode. flush drains every lane; close returns the merged
complete ledger only after all communication is matched, children are gone, and
both devices are idle.

## Per-chunk correctness protocol

For each real chunk:

1. It visits denoising_step_list[:steps] in order; lane 0 uses that chunk's
   motion-derived current_step.
2. Step s+1 receives add_noise(x0_s, noise_step(chunk,s), t[s+1]).
3. Every block's step-s cache receives chunk writes in chunk order, and chunk i sees
   exactly step-s writes from chunks <= i, with the frozen rolling/sink/refresh policy.
4. Refresh and token windows depend only on chunk index.

Interleavings and deeper pipelines are legal only when those invariants remain exact.
Migration must move K, V, both indices, cross-attention state, and every policy
counter/queue at a quiescent payload boundary.

## Metrics and gates

Every x0 latent and decoded float32 frame must be bit-exact (torch.equal semantics)
against the pristine seed measured on the same GPUs in the same evaluation.
All twelve gates are fail-closed: profile, hashes, admission, seed reference,
worker, completeness, parity, liveness, resources, TTFF, p99, and construction.

Your baseline is measured, not looked up. Each Judge run executes the pristine
seed over the suite first, on the same two devices, and every seed number below
comes from that pass. `tools/run_public_tests.sh` does the same locally and
caches the measurement per GPU pair under `build/seed_reference_v2/`; pass
`--seed-reference refresh` to re-measure it (do that after anything that could
change machine state).
Each public run writes its outputs under `build/public-<time>-<pid>/`; only the
two newest runs keep their saved frames (`artifacts/`), older ones keep just
their JSON results and logs, so copy anything you want to keep.

- push time: stamped by the controller immediately before it calls start/push.
- steady_fps: output frames in chunks 12..end divided by the interval from the
  push of chunk 12 to the emission of the final chunk. All of that work has to
  finish inside the interval, so holding outputs back or taking input early does
  not shorten it.
- latency p50/p99: emission minus push, over chunks >= 12 only.
- ttff: chunk-0 emission minus the start() push stamp; construction is excluded.
- case score: steady FPS multiplied by the fourth root of seed p99 / your p99.
- suite score: uncapped geometric mean of case scores, named latency_weighted_fps (fps).
- p99 limit: 3 * in-run seed p99.
- TTFF limit: max(2.5 * seed_ttff, seed_ttff + 4 seconds).
- construction limit: max(2 * seed construction, seed construction + 60 seconds).
- GPU peak limit, the same on both ranks: the larger of the in-run seed's two
  per-rank peaks + 8 GiB. A rank you never touch may read 0 MiB.
- host RSS limit: 65536 MiB per worker session.
- post-close GPU memory: at most 1024 MiB above what each rank held before the run.
- causality: no chunk may be emitted before it was pushed.

Results also show candidate/seed FPS, candidate/seed p99 latency, construction time,
per-rank NVML peak, host RSS growth, and exact max-absolute parity diagnostics.

## Commands

    tools/run_one.sh --version 1 --case public_bbb_480p_129_s2
    tools/run_public_tests.sh --candidate-version 1 --quick
    tools/run_public_tests.sh --candidate-version 1
    tools/run_public_tests.sh --candidate-version 1 --seed-reference refresh

The first run of a case also measures the seed on that case, so it takes roughly
twice as long; later runs reuse the cached on-device reference.

    cp -a candidates/runtime_v001 candidates/runtime_v002
    # edit and test runtime_v002
    printf '2\n' > FINAL_VERSION

Public case definitions are visible under /workspace/benchmark/runtime/manifests.
Public uses only 2- and 4-step cases at 832x480 and 512x512; hidden cases differ
from the public ones but stay inside the support domain: 1–4 steps, height and
width multiples of 16, 4n+1-frame streams. The full public
controller must pass before promotion. Fetched implementations or pretrained assets
may not enter a submission. Hidden prompts, final cases, and Judge goldens are never
shipped in Work.


## Formal submission and final evaluation

Set `FINAL_VERSION`, then call `sforge-submit --details`. Every formal submission
and the terminal evaluation run the same complete frozen private suite,
with the same correctness gates, latency-weighted FPS formula, and time budgets.
Public inputs are local diagnostics. `sforge-submit` always evaluates the version
named in `FINAL_VERSION`. The terminal evaluation uses the
sealed candidate and returns no feedback to the finished Agent.

Your final result is the best valid score you reach within the time budget, so keep `FINAL_VERSION` on your strongest candidate.
