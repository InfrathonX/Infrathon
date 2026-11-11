# Flux MoE All-Gather + Scatter + Grouped GEMM on Eight H100s

This task asks an agent to optimize the fused MoE layer-0 operator of FLUX and Comet in
self-contained Triton. W GPUs (2 to 8) of one host each hold a shard of the tokens; every
rank all-gathers the other ranks' shards through GPU peer access, scatters rows into
expert-sorted order and runs a grouped BF16 GEMM over the experts it owns, in tensor-,
expert- or hybrid-parallel deployments. It is a seed task: the starting point is a working
W-rank Triton implementation derived from ByteDance Flux (rotated ring-order remote-pull
producers, per-chunk release/acquire flags, local-first grouped-GEMM consumers), and the
agent improves its throughput while keeping it exact.

(The task id is historical; the task needs eight GPUs of one host and runs placement groups
of 2 to 8 of them.)

## The task

The agent works in `/app/workdir/moe_ag_lab` inside the Work container. The brief is
`AGENT_TASK.md` (also `README_AGENT.md` in the lab). Each candidate is a file
`candidates/kernel_vNNN/submission.py`; `FINAL_VERSION` names the one to evaluate. Only
`candidates/`, `FINAL_VERSION` and `notes.md` are submitted.

The submission must export this exact ABI and return `None`:

```python
moe_ag_scatter_gemm_bf16(x_local, x_peers, w, sorted_token_ids, sorted_expert_ids,
                         expert_offsets, token_offsets, out, workspace, peer_workspaces,
                         peer_ptrs, workspace_ptrs, rank, world_size) -> None
```

- All `world_size` ranks of a placement group call it at the same time. `x_local` is the
  rank's BF16 `[M_rank, K]` token shard and `x_peers[p]` a readable symmetric-memory view of
  rank `p`'s shard (`peer_ptrs` holds the same addresses as an int64 device table);
  `token_offsets[W+1]` maps global token ids to ranks. Weights are the rank's experts, BF16
  `[E_rank, N, K]`. Expert-sorted row `j` is
  `x_gathered[sorted_token_ids[j]] @ w[sorted_expert_ids[j]].T`; every element of `out[R, N]`
  must be written on the current stream.
- `workspace` is 4 GiB of local symmetric scratch, zeroed once when it is allocated and never
  again; `peer_workspaces[p]` (and the `workspace_ptrs` table) are the writable views of the
  other ranks' workspaces. Cross-GPU synchronization may use Triton atomics with
  release/acquire semantics.
- All inputs must stay byte-identical.

Rules enforced by the Judge (a fail-closed AST source admission plus a required LLM integrity
review, see `llm_judge/POLICY.md`):

- Imports are limited to `math`, `typing`, `torch`, `triton` and `triton.language`. All tensor
  data paths must be reachable Triton kernels. Host code may only inspect metadata, slice or view
  scratch, allocate same-device scratch, and launch kernels.
- Forbidden: host tensor arithmetic or copies, host loops, private CUDA streams,
  distributed/NCCL calls, compiled extensions, dynamic imports, filesystem/process/network
  access, module mutation, dead host helpers, and keeping anything derived from tensor
  contents across calls.
- Upstream implementations may not be copied into the submission. Skipping peer data, rows,
  experts or part of the reduction, replaying outputs, or detecting decoys or the no-transfer
  twin (below) is rejected.

The Work image has no Flux, NVSHMEM, DeepEP, DeepGEMM or similar packages installed. Work has
outbound web access through a restricted sidecar proxy (the task sets `internet: true`).
The launcher's default agent budget is 12 hours (`--hours` / `--timeout` to change it).

Public tools inside Work (they use all eight GPUs):

```bash
./tools/run_one.sh --version 1                                  # one public case
./tools/run_public_tests.sh --quick --candidate-version 1       # one case per family + 4 edges
./tools/run_public_tests.sh --candidate-version 1               # every public case
```

The public suite (`images/work/benchmark/public_cases.json`, 112 scored cases and 19
correctness-only edge cases) covers every family, tier and placement of the Judge suite with
other models and shapes; its real-routing cases replay the public fixtures in
`images/work/benchmark/routing/` (OLMoE-1B-7B and Qwen3-30B-A3B).

## Evaluation

The Judge runs a frozen, Judge-only suite of 986 cases (`judge/final_cases.json` under the asset root):
967 scored cases and 19 correctness-only edge cases, on placement groups of 8, 4 and 2 GPUs
(some with shuffled rank order or ranks spread across NUMA nodes; the edges also use 3 and 6).
`scripts/generate_final_manifest.py` draws it reproducibly from a seeded generator and records
every case's source. Scored cases come in three tiers:

| tier | cases | what |
| --- | --- | --- |
| `paper` | 41 | the reproducible single-node experiment points of Comet (arXiv:2502.19811 v3, Figs 10-14), FLUX (arXiv:2406.06858 v5) and bytedance/flux @ 19831ca2 (docs, examples, tuning and test defaults), at the papers' `N = I / TP` (no gate projection) |
| `model` | 864 | 33 real MoE and dense model configurations (Hugging Face `config.json`) in TP8/TP4/TP2, EP and TP x EP deployments at gated-MLP (SwiGLU gate and up fused) `N = 2 I / TP`, over training/prefill (up to 131072 tokens), decode (tens to a few thousand tokens) and multi-layer regimes, with synthetic routing from uniform to heavily skewed, hot experts, token locality and uneven tokens per rank |
| `real` | 62 | the same deployments of four models (gpt-oss-20b, Phi-3.5-MoE, Mixtral-8x7B, Granite-3.1-3B-A800M) replaying token windows of their own router decisions on mixed real text |

Cases form six families: `tp8`, `tp4`, `tp2`, `ep`, `decode` and `sequence` (several
consecutive MoE layers called back to back, timed together). The public and Judge suites share
no `(K, N, E_rank, topk)` identity, seed, model or routing fixture.

Every call (warm-up, decoy and timed) is checked for finite output with all NaN poison
overwritten (G1) and byte-identical inputs (G2); every timed call also gets a Freivalds random
projection `out @ v` against `x . (W^T v)` (G7, relative L2 `<= 2e-2`, every row within `0.2`
of the RMS); two hidden trials per case are compared with an
FP32 golden and an ordinary BF16 framework reference, which the Judge builds only after the
candidate has synchronized and the inputs have passed the immutability check:

| gate | requirement |
| --- | --- |
| G3 RelL1 | `<= max(1.25 * reference, 1e-3)` |
| G4 CosSim | `>= 0.9999965` |
| G5 row relative-L2 p99.9 | `<= max(1.25 * reference, 1e-3)` |
| G6 row relative-L2 max | `<= max(1.25 * reference, 1e-3)` |

Each measured trial uses fresh values after a shape-matched decoy, and part of every expert's
weights is redrawn before every call. A submission is valid only when all Judge gates pass:
profile, suite, source admission, worker execution, hardware envelope, correctness, edge
cases, fresh input and rank agreement.

Score (maximize, uncapped, in TFLOPS):

- a case's work is the mean per-rank GEMM work `2 * M * topk * N * K / EP` (independent of the
  routing draw; a sequence case sums its layers); case TFLOPS divides it by the median of seven
  measured trials (after one warm-up), each timed as the slowest rank's
  `max(host wall time, CUDA-event time)`;
- within a family, the evidence cases (`paper` and `real`) and the `model` cases each give a
  geometric mean; the family score is the geometric mean of the two (or of the one group a
  family has);
- the score is the geometric mean of the six family scores.

The Judge also reports, but does not score, `hidden_comm` on the paper tier: it reruns the
candidate with every peer shard in local tensors and the peer workspaces aliased locally (no
peer traffic), and computes `clamp(1 - max(0, t_fused - t_twin) / t_comm, 0, 1)` from the
fused time, this twin time and a trusted ring-order transfer time.

Interactive submissions (`sforge-submit`, unlimited, 300 s cooldown) and the final evaluation run
the same complete frozen suite with the same timing and thresholds. Scheduled automatic
evaluation is off. After the agent session ends the launcher runs a fresh Judge of the final
snapshot; the run's result is the best valid score across the run's submissions and this final evaluation, not an average. Only sanitized
aggregates are returned to the agent. Raw TFLOPS are comparable only within one GPU type.

A full Judge of the seed takes about 12 minutes on 8 x H100 SXM (683 s in the worker; two Judge
runs scored 48.20 and 48.16 TFLOPS with every gate passing; per-case
run-to-run difference median 0.2%, trial spread median 2%). The runtime's own peak is 16 GiB
per GPU plus about 10 GiB of context, NCCL and symmetric buffers, so candidates keep more than
50 GiB of each GPU for their own scratch. On the paper tier the seed's fused call is about
2 ms slower than its no-transfer twin while a trusted ring transfer of the same data takes
0.1-0.7 ms, so its `hidden_comm` is 0: communication is fully exposed, which leaves room for
better overlap.

On 8 x H100 NVL (through the full build and `smoke_test.sh --oracle`) a
full Judge of the seed takes about 21 minutes and scores 21.8 TFLOPS with every gate passing
(four other Judge runs: 21.77-21.82). Only the bridged pairs use NVLink there; the other peers are
reached over PCIe (6.7-15 GB/s per rank at W = 8), so communication-bound families score far
below H100 SXM. Scores compare only within one card type.

| family (seed) | tp8 | tp4 | tp2 | ep | decode | sequence | score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H100 SXM | 61.5 | 118.3 | 196.7 | 62.9 | 8.5 | 16.3 | 48.2 |
| H100 NVL | 12.3 | 71.8 | 172.6 | 14.4 | 6.7 | 7.3 | 21.8 |

## Hardware

Work and the Judge share all eight GPUs of one host; the Judge pauses Work and waits for the
GPUs to be idle before it measures, so an agent's 12 hours include its Judge rounds. Work and
Judge each get 32 CPUs and 128 GiB of RAM.

The capability envelope is eight CUDA GPUs with compute capability 8.0 or higher, at least
64 GiB each, and peer access between every pair; the launcher's preflight checks peer access
between every pair and an eight-rank symmetric-memory rendezvous, and records whether each
link is NVLink or PCIe.

| hardware type | status |
| --- | --- |
| `h100-80gb-700w` (H100 SXM 80 GB, NVSwitch: NVLink between every pair) | qualified |
| `h100-94gb-400w` (H100 NVL 94 GB, NVLink bridges in pairs, PCIe between them) | qualified |

Only one run per host: a run holds all eight GPUs for its whole life. Change `--judge-port`
(default 18611) only if that port is taken.

## Running

1. Assets. The task needs no external models or datasets at run time. Its Judge-only inputs
   are the final-case manifest and the four Judge routing fixtures its real-tier cases pin by
   sha256. They are not in the repository: `fetch.sh` downloads them from the dataset's
   `judge/` part, and the build reads them from `INFRATHON_ASSETS_ROOT`:

   ```bash
   tasks/latest/flux-moe-ag-scatter-grouped-gemm-dual-h100-v1-seed/assets/fetch.sh   # into INFRATHON_ASSETS_ROOT
   ```

2. Images. Both images are based on `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel` (Torch 2.8.0,
   Triton 3.4.0). Pull the published Work and public Judge images and build the holdout layer
   locally, or build everything from source:

   ```bash
   INFRATHON_PULL=1 tasks/latest/flux-moe-ag-scatter-grouped-gemm-dual-h100-v1-seed/scripts/build_images.sh
   # or: tasks/latest/flux-moe-ag-scatter-grouped-gemm-dual-h100-v1-seed/scripts/build_images.sh
   ```

   Optional checks: `scripts/make_oracle_archive.sh` packs `oracle/FINAL_VERSION` as an
   archive (`ORACLE_VERSION=N` packs another candidate), and `scripts/smoke_test.sh --oracle`
   runs the public suite and a full Judge on the admissible oracle candidate
   (`ORACLE_ADMISSIBLE_VERSION`, default 1).

3. Run an agent:

   ```bash
   launch/flux-moe-ag/run.sh --model MODEL [--gpus 0,1,2,3,4,5,6,7] --auth ~/.codex/auth.json
   ```

   Defaults: GPUs 0-7, 12 hours, effort `max`, Judge port 18611; `--gpus`, `--hours`,
   `--effort` and `--judge-port` change them. `--agent claude-code --claude-token FILE` runs
   Claude Code instead of Codex. The launcher waits for the eight GPUs to be idle and their
   locks to be free, runs the preflight, starts the Judge server and the restricted proxy, runs
   the agent, and then runs the final Judge. Logs go to
   `tasks/latest/<task>/logs/runs/<RUN_ID>/`.

   Evaluate an archive without an agent run:

   ```bash
   FLUXMOE_GPU_INDICES=0,1,2,3,4,5,6,7 tasks/latest/flux-moe-ag-scatter-grouped-gemm-dual-h100-v1-seed/scripts/eval_archive.sh ARCHIVE.tar.gz
   ```

   Replay saved candidates from a finished run: `launch/flux-moe-ag/replay.sh --origin-run RUN_ID
   [--candidate vN | --from-v N --to-v N] [--repeat N]` (`--dry-run` previews the selection).

Prerequisites the launcher checks: Docker with GPU access, `flock`, `jq`, `nvidia-smi`, the
repository's `.venv`, a CUDA build of `torch` in the `python3` first on `PATH` for the GPU
preflight (`docs/HOST_PREREQUISITES.md`, task-specific host extras), the pinned agent runtime
(`launch/build-agent-runtime.sh`), and model credentials (a Codex `auth.json`, a Claude Code
OAuth token file, or an OpenAI-compatible endpoint via `MODEL_API_BASE_URL` and a mode-0600
`MODEL_API_KEY_FILE`). The LLM integrity review is configured with the `SFORGE_LLM_JUDGE_*`
keys in the launch config (`--config`).

## Routing fixtures

`scripts/capture_routing.py` runs an open MoE model (BF16; gpt-oss-20b with its MXFP4 experts
dequantized) over about 164k tokens of text that interleaves, sequence by sequence,
encyclopedia prose (WikiText-103), chat (UltraChat 200k), source code (codeparrot-clean),
math word problems (GSM8K) and reading-comprehension QA (SQuAD), and records every MoE layer's
router top-k from the router's own weights on its input. `scripts/build_routing_fixtures.py`
keeps eight evenly spaced layers and the first 49152 tokens (all five sources) as uint8
`[layers, tokens, topk]` tables with a JSON provenance record (model revision, kept layers,
per-sequence sources, digests). Fixtures hold only expert ids, no text or weights. A
captured-routing case names a fixture, a layer, a token offset and the fixture's sha256; the
runtime hashes the file before parsing it, the Judge stages only the pinned fixtures, and
layer `l` of a multi-layer case reads the fixture layer after layer `l - 1`'s.

## Files

```text
images/work/       Work image: AGENT_TASK.md, the seed lab (moe_ag_lab/), public test controller,
                   public cases and routing fixtures
images/judge/      public Judge: verifier, candidate worker, source admission; Dockerfile.private adds the holdout
runtime/           trusted runtime: input generation, reference, gates, timing protocol
judge/             Judge entry; private/MANIFEST.json lists the Judge-only files that fetch.sh downloads
oracle/            reference candidates: the W-rank seed and the two-GPU Flux CUTLASS paper path (not admissible)
paper_reference/   local-only Judge image and runner that scores the paper path with source admission and the LLM review turned off
llm_judge/         integrity-review policy
scripts/           suite generator, routing capture, image build, preflight, launcher, final evaluation, replay, validators
tests/             CPU and static contract tests
tasks/             SForge task spec (Judge command, timeouts, resources, submit paths)
assets/            asset manifest and fetch script
```

## Reference solutions (oracle)

`oracle/FINAL_VERSION` = 1 selects the W-rank seed, which is also the admissible oracle
(48.2 TFLOPS on 8 x H100 SXM and 21.8 on 8 x H100 NVL, above). The
Flux CUTLASS paper candidates `kernel_v002`/`kernel_v003` are not admissible (compiled Flux
kernels, `torch.distributed`) and were written for the task's earlier two-GPU ABI; they are kept
for reference only and do not run on the current suite (their two-GPU records are in
`oracle/notes.md`).

## Upstream and license

- The operator design and seed derive from ByteDance Flux, commit
  `19831ca2d820e3e782ed1d15d8b52d0898b78b26` (Apache-2.0), in particular
  `python/flux_triton/kernels/moe_ag_scatter.py` and `python/flux/triton/moe_ag_scatter.py`.
- The communication/computation decomposition follows the Comet paper (arXiv 2502.19811);
  the dense fusion context follows FLUX (arXiv 2406.06858).
- Routing fixtures are derived from the router decisions of the models and texts named in
  each fixture's JSON record.
- The task's own code is under the repository's root `LICENSE` (PolyForm Noncommercial 1.0.0).
