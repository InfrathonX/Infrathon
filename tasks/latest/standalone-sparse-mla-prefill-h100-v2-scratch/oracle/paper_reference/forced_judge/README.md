# Forced-oracle Judge: FlashMLA through the full Judge pipeline (LOCAL ONLY)

The oracle does not have to follow the rules imposed on agent submissions
(source-admission import allowlist, no compiled extensions, LLM integrity
review).  This directory scores the strongest official paper solution,
upstream FlashMLA, with the task's **real** Judge pipeline by building a
local-only Judge image that admits it.  The result is a normal Judge
`report.json` (plus the private per-case artifact) for
`oracle/candidates/kernel_v002/submission.py`, which `oracle/FINAL_VERSION`
selects.

Never use this image or script for an agent archive, never push the image,
and never retag it `:local`.

## What differs from the normal Judge

| | normal Judge | forced-oracle Judge |
|---|---|---|
| image | `standalone_sparse_mla_prefill.judge.standalone_sparse_mla_prefill_h100_scratch_v2:local` (private: public Judge + frozen holdouts) | `...:forced-oracle`, `FROM` that image (`Dockerfile`) |
| FlashMLA | absent | upstream package copied in (`site-packages/flash_mla`, built for `sm_90a`) |
| source admission | `source_admission.py` (allowlist + bans) | `forced_source_admission.py` installed as `source_admission.py`; the original runs unchanged as `source_admission_enforcing.py` and makes every structural decision (layout, `FINAL_VERSION`, selected path, SHA-256), only its verdict is overridden; its errors are kept as warnings |
| `protected_hashes.txt` | original | the `judge:source_admission.py` line updated to the override's hash; every other record unchanged and still checked |
| LLM integrity review | runs for agent archives | skipped (`TRUSTED_CONTROL=1`, as for every trusted oracle smoke; `SFORGE_LLM_JUDGE_MODE=off`) |
| everything else | | identical: `verify_sparse_mla.py`, `run_judge.py`, `candidate_worker.py`, trusted runtime, frozen final holdout (13 cases) and 401 private edge cases, compute-sanitizer memcheck pass, 7 trials + 1 warm-up, guard bands, L2 flush, decoy pre-call, plausibility bound, aggregation |

The run itself is the task's own `scripts/eval_archive.sh` (`sforge eval
--backend docker`: GPU pinned by UUID, the per-UUID GPU lock, `--network none`,
8 CPUs / 32 GiB, optional `SFORGE_JUDGE_CPUSET`), with `TASKS_DIR` pointing at
a temporary copy of `tasks/standalone_sparse_mla_prefill_h100_scratch_v2.json`
whose only change is `judge.image_tag = forced-oracle`.

## Provenance

- Upstream: https://github.com/deepseek-ai/FlashMLA @
  `ba89a3466e9470ad08ab39738d4e7bb66989e1e7` (the commit the task was ported
  from); submodule `csrc/cutlass` @ `147f5673d0c1c3dcf66f78d677fd647e4a020219`.
- Kernel: `csrc/kernels/sm90/prefill/sparse/` unchanged.  Glue: only
  `../flashmla_out_abi.patch` (`csrc/api/sparse_prefill.cpp`: new binding
  `sparse_prefill_fwd_out` writing the caller-owned `out` / `max_logits` /
  `lse`).
- Build: `../Dockerfile` (FROM `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel`,
  the Judge's base; `TORCH_CUDA_ARCH_LIST=9.0a`, `FLASH_MLA_DISABLE_SM100=1`,
  `pip install --no-build-isolation .`) produces the paper image; `Dockerfile`
  here copies the installed package into the private Judge image.  The
  upstream tree is supplied as the `flashmla` named build context
  (`FLASHMLA_SOURCE`), never vendored.
- Candidate: `oracle/candidates/kernel_v002/submission.py`, sha256
  `e41e9303cf4011c5e98356074de952ee8f213dd4da3839dd3e8d018700474fec`,
  byte-identical to `../candidates/flashmla_out_v001/submission.py` and to the
  judged archive member.

## Build and run

```bash
git clone https://github.com/deepseek-ai/FlashMLA /path/to/FlashMLA
git -C /path/to/FlashMLA checkout ba89a3466e9470ad08ab39738d4e7bb66989e1e7
git -C /path/to/FlashMLA submodule update --init csrc/cutlass
scripts/build_images.sh        # the private Judge image this derives from

# from the task root; builds the paper image first if it is missing (30-60 min CPU)
FLASHMLA_SOURCE=/path/to/FlashMLA oracle/paper_reference/forced_judge/run_forced_oracle.sh --build-only
# one full Judge run on an idle H100 (about 2 min of Judge time)
PHYSICAL_GPU=<idx> SFORGE_JUDGE_CPUSET=<cpus of that GPU's slice> \
  oracle/paper_reference/forced_judge/run_forced_oracle.sh --skip-build <RUN_ID>
```

`EVAL_TASK_ROOT=<task tree>` runs the Judge from another checkout of this task
whose build manifest matches the local `:local` images (the measurements below
used a development checkout of the same task; the oracle archive is always
packed from this tree).  The report lands at
`<outputs>/runs/<RUN_ID>/standalone_sparse_mla_prefill_h100_scratch_v2/submissions/manual-1/report.json`.

## Measured (H100 NVL 94 GB, GPU 7, cpuset 28-31,60-63, 2026-10-29)

Score = geomean over v32 / v4_flash / v4_pro of the family geomean of case
median TFLOPS (higher is better).

| run id | candidate | Judge | valid | score | v32 | v4_flash | v4_pro | sanitizer | edge failures |
|---|---|---|---|---|---|---|---|---|---|
| `forced-sparse-mla-flashmla-r1-g7` | `kernel_v002` (FlashMLA) | forced-oracle | yes | **386.37** | 421.2 | 329.8 | 415.3 | ran, passed (14 cases) | 0 / 401 |
| `forced-sparse-mla-flashmla-r2-g7` | `kernel_v002` (FlashMLA) | forced-oracle | yes | 381.59 | 420.2 | 322.1 | 410.5 | ran, passed (14 cases) | 0 / 401 |
| `forced-sparse-mla-admissible-v001-r1-g7` | `kernel_v001` (Triton, `--admissible` archive) | normal | yes | 146.00 | 141.6 | 135.6 | 162.1 | ran, passed (14 cases) | 0 / 401 |

Forced image `...:forced-oracle` = `sha256:e7f81d223add7688a12fe4ac47b61ec180d517dc828d560adb8d78ff21f95f22`, built from
private Judge `...:local` = `sha256:6d1a65b96e393914c7389149d4b4614f0aed13ad09dab3b53d0eea27df48b1e2` and paper image
`standalone_sparse_mla_prefill.paper-oracle-flashmla:local` (`476178f2cf77`).
Gates bypassed: source admission verdict (bypassed error:
`candidates/kernel_v002/submission.py:10: forbidden import: 'flash_mla.cuda'`)
and the LLM integrity review.  No other gate was bypassed, and none failed.
The normal Judge rejects the same archive at `source_ok`
(`pvsa-sparse-mla-paper-admission-20261029-120907`).
