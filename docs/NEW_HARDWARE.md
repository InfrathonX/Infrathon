# Bringing Infrathon to a new NVIDIA GPU

Infrathon tasks are released per **hardware type**. Today there are two:
`h100-94gb-400w` (H100 NVL) and `h100-80gb-700w` (H100 SXM 80 GB). A task runs on
another card only after it has been **qualified** there: its reference (oracle)
solution evaluated by the Judge on that card, with valid results and enough
margin. Code changes are usually small. Most of the work is measuring each task
again on the new card and checking that its timeouts and references still hold.

This page lists what has to be done, in order, and where each task depends on
the card. Scores from different hardware types are not comparable: compare
runs only within one type.

## 1. Check what the card can run

Measure on the real card, not from the datasheet:

- **Usable memory**: `python -c "import torch; print(torch.cuda.get_device_properties(0).total_memory)"`.
  A card sold as 80 GB reports 81,081 MiB (79.2 GiB), less than the 80 GiB floor
  that some tasks had.
- **Compute capability, name, power limit**: `nvidia-smi --query-gpu=name,compute_cap,memory.total,power.limit --format=csv`.
- **Interconnect**: `nvidia-smi topo -m`. The two-GPU tasks need an NVLink pair
  with peer access for Work and, by default, a second one for the Judge: the
  five Miles tasks, qwen3-orpo, qwen3-gkd and streamdiffusionv2 (which also
  needs torch symmetric memory). flux-moe-ag needs eight GPUs with peer access
  between every pair and torch symmetric memory across all eight; an NVSwitch
  node links every pair by NVLink, an H100 NVL host bridges pairs and uses PCIe
  for the rest (lower scores, compared only within the card type).

Compare these with each task's memory floor and GPU count (the task README and
the table in section 4). A task whose floor is above the card is `unsupported`
on it. The others start as `untested`.

**Architecture.** The published images are built for Hopper (sm_90). On another
Hopper card (for example H200 or H20) they normally run as they are. On a
different architecture, the compiled extensions, Triton/TileLang kernels and
base images must be rebuilt and checked, and some features are not portable:

- FP8 tasks need compute capability 8.9 or higher.
- FlashAttention 3 (Miles diffusion tasks, MInference's reference) and wgmma/TMA
  kernels (TileLang, optional `sm_90a` builds) are Hopper-specific.
- attn-qat's NVFP4 is emulated in Triton and needs no Blackwell hardware.

## 2. Register the hardware type

Pick a name `<model>-<memory>gb-<power>w`, e.g. `h20-96gb-400w`, then:

| Place | Change |
| --- | --- |
| `scripts/infrathon/assets.py` `HARDWARE_TYPES` | add the name and the exact GPU name `nvidia-smi` prints; `detect_hardware()` matches it exactly (`INFRATHON_HARDWARE` overrides detection) |
| `scripts/infrathon/tasks.py` `HARDWARE_TYPES` | add the name; `tasks.json` must then give **every** released task a status for it, or loading fails |
| `scripts/infrathon/tasks.json` | a status per task: `untested` or `unsupported` |
| `scripts/infrathon/publish_images.sh` | the hardware-name check `^h100-[0-9]+gb-[0-9]+w$` rejects other models; widen it |
| `sforge/harness/portable_spec.py` | the list of GPU model names that must not leak into task text; add the new model if missing |
| `sforge/harness/resource_profile.py` | assumes 8 logical CPUs per GPU; check the new host's CPU slice per GPU |
| `tests/test_infrathon_assets.py`, `tests/test_infrathon_registry.py` | they name the two existing types |
| README task tables | regenerate with `python3 scripts/infrathon/tasks.py table` |

**Capability is not a card identity.** Values chosen by compute capability alone
apply to every card with that capability. An H20 reports (9, 0) like an H100, so:

- the standalone kernels' peak-plausibility bounds, keyed by capability, would
  use the H100 values;
- `scripts/miles_common/gpu_profile.py` admits a pair by capability, memory and
  NVLink only, so a pair of new cards that meets the NVL profile's floors would
  be scored under the H100 NVL qualification record without any warning.

Add a GPU-name or memory check wherever the new card must be told apart.

## 3. Prepare a host

Follow [HOST_PREREQUISITES.md](HOST_PREREQUISITES.md). Then, per task,
`tasks/latest/<task>/assets/fetch.sh` with `INFRATHON_HARDWARE=<new type>` fetches
the shared assets. There are no `hw/<new type>/` assets yet; step 5 makes them.

## 4. Where each task depends on the card

Unless the table says otherwise, a task's Judge checks only a minimum compute
capability and memory size, and its score is absolute. "Timeouts" are the fixed
limits sized on H100 speed that a slower card may exceed. Check each one against
a measured run on the new card.

| Task | GPUs | Memory floor / measured peak | Per-card files or values | Timeouts sized on H100 |
| --- | --- | --- | --- | --- |
| attn-qat-nvfp4-training-attention | 1 | 40 GiB (the public tests need 16 GiB) | e7 fixture captured on H100 NVL, tolerances widened | candidate 3600 s, Judge 4200 s |
| concurrent-agent-serving-optimizer | 2 (vLLM TP2 backend) | the backend needs at least 80,000 MiB per card (the frozen serving contract still declares 95,000 MiB) | Direct baselines measured on the host | window 300 + 1800 s, `eval_timeout` 3000 s, p99 wait ≤ 720 s |
| eagle3-spec-decode-engine | 1 | 34 GiB / about 21 GiB | none (limits are relative to the seed run) | Judge 5400 s |
| flash-kmeans-paper | 1 | 79 GiB / about 53 GiB on H100 SXM 80 GB | none | 180 s per case, verifier 3300 s, Judge 3600 s |
| flux-moe-ag-scatter-grouped-gemm | 8 shared by Work and Judge (P2P between every pair, symmetric memory) | 64 GiB / runtime peak 16 GiB plus about 10 GiB of context, NCCL and symmetric buffers on H100 SXM 80 GB | none | candidate worker 5400 s, Judge 6000 s, `eval_timeout` 7200 s (seed Judge about 12 min on H100 SXM, 21 min on H100 NVL) |
| kvpress-kv-eviction-policy | 1 Work + 7 Judge | 56 GiB | none | timeout formula in `scripts/lib.sh`, `eval_timeout` 11000 s |
| kvzip-query-agnostic-kv-scoring | 1 Work + 7 Judge | 79 GiB, `expandable_segments` on | none | timeout formula in `scripts/lib.sh` |
| llada-h100-decoder-optimization | 1 | 34 GiB / 82 GiB on H100 NVL | **quality reference keyed by (compute capability, GPU name)**; an unknown card is refused (`scripts/select_reference_profile.py`); `memory_policy.json` | `workload_policy.json` 420 / 900 / 3600 / 3900 s (hash-protected) |
| lmcache-paged-kv-block-transfer | 1 | 64 GiB | bandwidth probed live on each card | 3600 / 4200 / 4800 s |
| miles-diffusion-sd35m-mopd | 2 (NVLink) | 80 GiB / 81.5 GiB | qualification profile and record per card | session 1500 / 2700 s, Judge 7500 s |
| miles-diffusion-wan22-dancegrpo | 2 (NVLink) | 90 GiB / 87.8 GiB | `hw/<type>/judge/private/` and its `qualification.json` | quality 2100 s, run 2400 s, Judge 5000 s |
| miles-qwen3-alfworld-agentic-rl | 2 (NVLink) | 90 GiB / 84.7 GiB | qualification profile and record per card | run 7300 s, Judge 14400 s |
| miles-qwen3-opcd | 2 (NVLink) | 80 GiB / 77.5 GiB | qualification profile and record per card; gradient and teacher tolerances | run 3600 s |
| miles-qwen3-opd-scratch | 2 (NVLink) | 80 GiB / 84.5 GiB | qualification profile and record per card; gradient cosine ≥ 0.93 | run 5400 s |
| minference-sparse-prefill | 2 (Work + Judge) | 80e9 B, compute capability ≥ 9.0 | none | worker 1800 s, evaluation 7200 s |
| quant-videogen-cross-model-kv-codec | 1 Work + 6 Judge | 79 GiB / 71,895 MiB on H100 SXM 80 GB (LongCat, official codec) | `hw/<type>/{public-eval,judge/hidden}` goldens (H100 NVL files, shared) and `hw/<type>/judge/h100_scratch_v4_quality.json` (quality reference and oracle band, two oracle runs per type); gates calibrated on H100 NVL | 4500 / 1500 s per case |
| qwen3-1p7b-orpo-training-optimizer | 2 (NVLink) | 60 GiB / 50.6 GiB | none (score is a ratio to a reference run on the same pair) | worker 600 s, Judge 12000 s |
| qwen3-4b-to-1p7b-gkd-distillation | 2 (NVLink) | 79 GiB / 75,454 MiB reserved per rank on H100 SXM 80 GB (up to 76,865 MiB) | none (reference run on the same pair); the candidate ceiling is card total minus 4 GiB, 76,985 MiB on H100 SXM 80 GB | worker 600 s, Judge 12000 s |
| sageattention-quantized-attention | 1 | 64 GiB | real-layer captures made on H100 NVL, tolerances widened | 3000 / 3300 s |
| standalone-dynamic-grouped-fp8 | 1 | 56 GiB, compute capability 8.9 | peak-plausibility bound by capability | candidate 2400 s, sanitizer 780 s, verifier 3300 s |
| standalone-k-grouped-fp8-wgrad | 1 | 48 GiB | peak bound by capability; tolerances calibrated on H100 | as above |
| standalone-paged-gqa-decode-bf16 | 1 | 32 GiB | peak bound by capability | as above |
| standalone-sparse-mla-prefill | 1 | 32 GiB | peak bound by capability | as above |
| streamdiffusionv2-dual-h100-pipeline | 2 (NVLink, symmetric memory) | 26 GiB / 18.3 GiB | **`hw/<type>/` goldens, `images/judge/protected_hashes/<type>.txt`, the `--hardware` choices in `scripts/generate_protected_hashes.py`** (the Work and public Judge images are the same for every card) | worker 1800 s, Judge 4200 s |
| tilelang-mamba2-ssd-forward | 1 | 73 GiB / 72.6 GiB | tolerance qualified on H100 | warm-up 300 s per case, verifier 4900 s, final 5400 s |

## 5. Make the card-specific references

On the new card, before qualifying:

- **llada**: run the reference producer once to add a quality reference for the
  card, then rebuild the Judge.
- **streamdiffusionv2**: generate the goldens and protected hashes for the new
  type (only its host-built holdout layer differs per card).
- **miles-diffusion-wan22**: generate `hw/<type>/judge/private/qualification.json`.
- **Miles tasks**: add an execution profile for the card and qualify it, which
  writes `runtime/qualification.<profile>.json`.
- **quant-videogen**: run the official oracle twice through the Judge and write
  `hw/<type>/judge/h100_scratch_v4_quality.json` from the case means (the
  quality reference) and the scores (the oracle band). The BF16 goldens and
  gates are shared; reuse the existing `hw/<type>/{public-eval,judge/hidden}`
  files unless the card's fidelity to them drifts, and only regenerate goldens
  on a card with room for the BF16 rollout (HY peaks at 97.4e9 bytes).

Card-specific dataset files go under `hw/<new type>/` in the task's
`assets/manifest.json` and in the dataset.

## 6. Qualify each task

For each task on the new card:

- [ ] `assets.py hardware` prints the new type; `fetch.sh` and `assets.py check --hardware <type>` pass.
- [ ] The images are pulled by digest, or built from source and pass the build freshness check.
- [ ] A full Judge of the reference/oracle solution is valid and passes every gate.
- [ ] The seed/base solution and at least one negative solution score as expected (the negative is rejected or scores 0).
- [ ] Repeat the oracle 2-3 times: the spread must sit well inside the gate margins. If the noise is the size of a gate, valid solutions fail at random.
- [ ] Record the peak memory against `total_memory` and keep headroom for candidates.
- [ ] Record one Judge's and one worker run's duration and compare them with the timeouts above. Decide before release whether a timeout changes for this card.
- [ ] Two-GPU tasks and Flux: the task's preflight passes NVLink, peer access and (sdv2, flux) symmetric memory checks (Flux across all eight GPUs).
- [ ] One short agent run (for example 1 h) completes start to finish, including the final evaluation.
- [ ] Every difference from the other hardware types is written in the task README as a deviation for this card.

Keep the scoring, gates and test sets unchanged. Only runtime settings change
(memory floor, allocator settings, GPU count, timeouts), each recorded as a
deviation. Start with the single-GPU, bandwidth-bound tasks (attn-qat,
sageattention, lmcache, paged GQA decode) to settle the host and the process,
then the rest.

## 7. Publish

- Set the task's status to `qualified` in `tasks.json`.
- Upload the new `hw/<type>/` files to the dataset.
- Images that did not change get the new hardware tag on the same digest
  (`crane tag`). Images built for the card, if any, go through
  `scripts/infrathon/publish_images.sh <task> <type>`. Either way,
  `images/published.json` gains `hardware.<type>`.
- Regenerate the README task tables.
- Compare results only with runs on the same hardware type, and measure the
  oracle line on that type.

## Example: H20

H20 is a Hopper card (sm_90). It has 96 GB of HBM3 at about 4.0 TB/s, which is
more than H100. Its dense BF16 tensor throughput is about 148 TFLOPS, roughly 15-18%
of an H100. The images should run unchanged. Expect the following:

- **Compute-bound tasks run several times slower**: training (Miles, ORPO, GKD,
  attn-qat) and GEMM/prefill kernels. Their timeouts and Miles' step budgets need
  checking first. llada, miles-wan22, miles-opd and quant-videogen need the most
  work: new references or goldens plus timeouts that will likely be exceeded.
- **Bandwidth-bound tasks run about as fast as on H100**: paged decode, KV-cache
  tasks and lmcache.
- **The best optimization can change.** With far less compute per byte of
  bandwidth, an oracle tuned on H100 may give a different speedup on H20, and
  case splits frozen at an H100 throughput threshold lose their meaning.
- **It reports the same compute capability as H100**, so add the name checks
  from section 2 before qualifying.
- Check that the H20 system has NVLink between GPU pairs. Without it the two-GPU tasks cannot run;
  Flux needs peer access between all eight GPUs.
