# Oracle candidates

**v2 (eight-GPU W-rank ABI, suite `final-holdout-v2`).** `FINAL_VERSION` = 1:
`kernel_v001` is the admitted W-rank seed (= Work seed, sha256 091ccd2b...): W-1
producer programs pull every peer shard in rotated ring order through
`peer_ptrs` into the local workspace (per-chunk release/acquire flags stamped
with a per-call epoch, so the never-cleared workspace needs no reset), and
consumer tiles run local rows first, then gathered rows as their chunks land
(BF16 grouped GEMM, FP32 accumulate). It is also `ORACLE_ADMISSIBLE_VERSION`
in `scripts/smoke_test.sh`.

`kernel_v002` / `kernel_v003` below are the **v1 (two-GPU ABI)** forced paper
oracles: they wrap Flux's `GemmGroupedV3AGScatter` for a fixed 2-rank pair and
do not run on the v2 suite until they are ported to W-rank placement groups (Flux
NVSHMEM teams per group). Every paper-tier case of the v2 suite runs on the full
8-GPU group `w8`, so a port can be limited to that group. Their v1 records are
kept as history.

## v1 history (two-GPU ABI)

| version | what | admissible | Judge |
| --- | --- | --- | --- |
| kernel_v001 (v1) | admitted Flux-derived seed (sha256 880cefbf...): upstream Flux Triton path (`python/flux_triton/kernels/moe_ag_scatter.py`), chunked remote-pull producers, release/acquire flags, local-first tiles, BF16/FP32 grouped GEMM | yes | normal `:local` |
| kernel_v002 | Flux CUTLASS `GemmGroupedV3AGScatter` (FLUX/Comet SM90 AG+scatter grouped GEMM, NVSHMEM all-gather), Flux runtime tile heuristic | no (imports `flux`, `torch.distributed`, `os`) | forced |
| kernel_v003 | kernel_v002 + Flux's own autotuner (`op.profiling` + `flux.load_tuning_record`) on the first, untimed call of every shape | no | forced |

Both forced candidates use bytedance/flux @ `19831ca2d820e3e782ed1d15d8b52d0898b78b26`
built for sm_90a with NVSHMEM; build and run recipe in
`../paper_reference/forced_judge/README.md`.

### Measured, v1 (final holdout `2762d961`, H100 NVL pair GPUs 6,7, Judge cpuset 24-31,56-63, 2026-10-29)

| candidate | Judge | runs (score TFLOPS) | mean | X1 / X2 / X3 / X4 / X5 (first run) | gates |
| --- | --- | --- | --- | --- | --- |
| kernel_v001 seed (best admissible) | normal `:local` | 171.65 `forced-flux-seedctl-gpu67-20261029-215756`, 172.43 `forced-flux-seedctl-gpu67-r-20261029-221504` | 172.04 | 125.2 / 161.0 / 183.5 / 230.2 / 211.5 | all pass |
| kernel_v002 Flux CUTLASS, heuristic | forced | 187.63 `forced-flux-v2-gpu67-20261029-215619`, 185.08 `forced-flux-v2-gpu67-r-20261029-220653`, 187.69 `forced-flux-v2-gpu67-r-20261029-221538` | 186.80 | 106.3 / 182.5 / 215.9 / 262.1 / 285.5 | all runtime gates pass; source admission bypassed |
| **kernel_v003 Flux CUTLASS, autotuned (v1 FINAL_VERSION)** | forced | 188.58 `forced-flux-v3-gpu67-20261029-215855`, 190.37 `forced-flux-v3-gpu67-r-20261029-220726`, 191.05 `forced-flux-v3-gpu67-r-20261029-221609` | 190.00 | 116.2 / 177.8 / 210.7 / 243.2 / 295.7 | all runtime gates pass; source admission bypassed |

Earlier seed records: NVL 166.20, 158.08, 156.58; SXM mean 163.28 (docs/RUN_RESULTS.md
in the dev tree). The paper path wins the moderate family (X3-X5, up to 1.4x on
X5), ties or slightly loses on X2, and loses on X1 (E=512, N=128: tiny per-expert
GEMMs), so the overall gain over the seed is +10% (190.0 vs 172.0 on the same
pair). It is far below the best agent runs (about 350 TFLOPS on SXM).

Gates in the forced runs: profile, suite, worker, hardware, correctness
(RelL1 / CosSim / row-L2 equal to the BF16 reference), all eight edge fixtures,
fresh input, fresh output and rank agreement all passed. Bypassed: source
admission (36 enforced errors for kernel_v003, 28 for kernel_v002: forbidden
imports `os`, `torch.distributed`, `flux`, module-level statements, host
tensor ops), and the LLM integrity review (off). `hidden_comm` (reported, not
scored) is not meaningful for the Flux path: Flux all-gathers through its own
NVSHMEM buffers and ignores `x_peer`/`workspace`, so the no-transfer twin
measures the same work.
