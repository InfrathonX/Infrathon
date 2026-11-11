# Paper-derived oracle candidate (host-only, staged outside `oracle/`)

This directory holds the strongest MInference paper solution that passes this task's Judge,
measured against the current oracle. Agents never see it: it is outside every Work build
context, and no image is built from it.

`candidates/kernel_v002` has been promoted to `oracle/candidates/kernel_v002` and is the selected
oracle (`oracle/FINAL_VERSION` = 2). `images/release.json` now lists only
`oracle/candidates/kernel_v001/` (the seed control, the one oracle file image staging reads) as an
image-driving source, and `images/published.json` records the recomputed source digest. The
published images themselves are unchanged: no oracle file enters an image. This directory keeps the
measurement record.

## Candidates

| candidate | contents | sha256 (submission.py) | H100 NVL Judge, 2026-10-29 |
|---|---|---|---|
| current oracle `oracle/candidates/kernel_v001` (seed) | upstream budgets x1.5 (768/3072) + seed probe guard at band NRMSE 0.10 | `01ef11ae...` | 196.52 / 197.05 local NVL; PyroMind 197.89 / 199.45 / 201.21 |
| `candidates/kernel_v002` (this directory) | MInference upstream default config + the seed's probe guard at band NRMSE 0.30 | `4d5e1508...` | **valid, 364.04364314548496**, 1666 s (run `oracleup-minference-guarded`). A rerun of the `make_archive.sh` archive (FINAL_VERSION 2) gave **valid, 363.5351770940296** (run `oracleup-minference-paperoracle-v002`). |
| `upstream_strict/submission.py` (reference only) | strict upstream port, no repair | `a895afa6...` | **invalid** (run `oracleup-minference-strict`). It fails the single-prompt fidelity gate on `final-niah_single_2-32768-r0-L1` (N = 32178), tail worst-head NRMSE 0.934 > 0.50. |
| strict + exact dense for N <= 8448 (not kept) | `upstream_strict` + small-N dense path | `03cab088...` | **invalid** (run `oracleup-minference-densesmall`), same case and same failure |

How the scores were measured:

- Every run used the published h100-94gb-400w Judge, `sha256:0656e30b...`
  (pulled with `scripts/infrathon/pull_images.sh`; source digest `04fc7cdc5edf`). Each was a
  single Judge GPU (physical 0) with the full final path (48 groups, 144 timed calls,
  11 boundaries), run with
  `launch/minference-v3/run.sh --judge-gpu 0 --eval-archive ARCHIVE --skip-integrity-review --run-id RUN_ID`.
- The archives measured were the paper-vs-agent archives `paper_guarded_submission.tar.gz` (sha256
  `769166c8...`, FINAL_VERSION 4), `paper_submission.tar.gz` (`6f03d10a...`, FINAL_VERSION 2) and
  `paper_densesmall_submission.tar.gz` (`17e66b0a...`, FINAL_VERSION 3). Each holds all four
  kernels.
- `candidates/kernel_v002/submission.py` is byte-identical to the measured `kernel_v004`.
- For comparison, the best agent (Claude Opus 5.5) final scored 3055.37 on PyroMind.

## Provenance

- Paper: MInference 1.0 (arXiv 2407.02490), `https://github.com/microsoft/MInference` at
  `a4eb395f949ea39e871f9bc586d683390692c6be` (the task's own pin).
- Estimator: `minference/modules/minference_forward.py` `vertical_and_slash_kernel`, using the
  last 64 queries, sinks `[:30]` and locals `[-100:]`.
- Budgets: `minference/configs/Llama_3.1_8B_Instruct_128k_kv_out_v32_fit_o_best_pattern_v2.json`.
  This is the `model2path` default for Llama-3.1-8B: all 1024 heads vertical_and_slash 512/2048,
  `minference_ratio` 1.
- Attention: upstream's Triton `_triton_mixed_sparse_attn_fwd_kernel`
  (`minference/ops/pit_sparse_flash_attention_v2.py`).
- Index converter: `csrc/vertical_slash_index.cu`, ported 1:1 to Triton, because CUDA extensions
  are not admissible.
- Not used: the faster upstream path (`sgl_kernel` / `vllm_flash_attn` `sparse_attn_func`). It is
  a vendor CUDA kernel, which the source admission rules forbid (allowed imports: `__future__`,
  `math`, `torch`, `triton`, `typing`).

Changes for the ABI:

- native GQA (`h // 4`);
- all heads in one launch;
- caller-owned `out`;
- zero padding for N % 64 != 0.

`kernel_v002` adds one piece that is not upstream: the task seed's per-call 192-row probe
and dense-head repair. Its band threshold is relaxed from 0.10 to 0.30 so that it only repairs
heads that would break the 0.50 worst-head gate. The strict upstream config is not valid here,
because individual heads reach an NRMSE near 1 on real RULER prompts (see the table).

## Pack and measure

```bash
paper_reference/make_archive.sh /tmp/minference-paper-oracle.tar.gz   # FINAL_VERSION 2, kernel_v001 seed + kernel_v002
launch/minference-v3/run.sh --judge-gpu G --eval-archive /tmp/minference-paper-oracle.tar.gz --skip-integrity-review
```

A Judge run takes about 28 minutes.
