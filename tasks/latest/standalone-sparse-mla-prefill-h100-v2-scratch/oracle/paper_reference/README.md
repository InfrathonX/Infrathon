# Paper reference: FlashMLA SM90 sparse prefill (host-only)

This directory times **upstream FlashMLA**, the task's source implementation, on
this task's ABI with the Judge's own trusted runtime, so agent scores can be put
next to the strongest published implementation.

**Update (forced oracle):** `oracle/FINAL_VERSION` now selects
`oracle/candidates/kernel_v002`, a byte-identical copy of
`candidates/flashmla_out_v001/submission.py`, scored by the full Judge
pipeline in a local-only derived Judge image that admits it
(`forced_judge/`, README there): **386.37 / 381.59 TFLOPS, valid**, sanitizer
passed, 0/401 edge failures (`forced-sparse-mla-flashmla-r1-g7`, `-r2-g7`).
`oracle/ADMISSIBLE_VERSION` keeps the Triton candidate `kernel_v001` for the
normal Judge (`scripts/make_oracle_archive.sh --admissible`,
`scripts/smoke_test.sh --oracle`); step 3 of `run_flashmla_reference.sh`
times that admissible candidate.

FlashMLA cannot take the normal scored path. Source admission rejects it
(`candidates/kernel_v001/submission.py:10: forbidden import: 'flash_mla.cuda'`;
the allowlist is `__future__`, `collections`, `dataclasses`, `enum`,
`functools`, `itertools`, `math`, `torch`, `triton` and `typing`), and the brief
forbids compiled extensions. On the real Judge the paper archive is INVALID at
`source_ok` (run `pvsa-sparse-mla-paper-admission-20261029-120907`). No
admissible (Triton) port of FlashMLA's SM90 kernel exists upstream, so the
scored Oracle stays the in-task Triton control.

Isolation:

- no Work or Judge build context reaches `oracle/`, and
  `scripts/validate_layout.py` fails if an image build names it;
- nothing here feeds `image_source_digest()` (`images/work`, `images/judge`,
  `runtime/`, `judge/run_judge.py`);
- `scripts/make_oracle_archive.sh` packs only `FINAL_VERSION`, `notes.md` and
  the selected candidate (`--admissible`: the `ADMISSIBLE_VERSION` candidate);
- the FlashMLA tree is not vendored. `run_flashmla_reference.sh` builds from a
  checkout you supply through `FLASHMLA_SOURCE`.

## Provenance

- Repository: https://github.com/deepseek-ai/FlashMLA
- Commit: `ba89a3466e9470ad08ab39738d4e7bb66989e1e7`. The task's semantics,
  generator, tolerances and scored suite were ported from this commit.
  Submodule `csrc/cutlass` is at `147f5673d0c1c3dcf66f78d677fd647e4a020219`.
- The kernel is unchanged: `csrc/kernels/sm90/prefill/sparse/{config.h,phase1.cuh}`
  plus the `phase1_k512{,_topklen}.cu` and `phase1_k576{,_topklen}.cu`
  instantiations. It runs one CTA per (query token, 64-head block) with 384
  threads in three warpgroups: two math warpgroups and one cp.async gather
  producer. Q comes in and O goes out by TMA, and the softmax is an online FP32
  softmax.
- The glue is `flashmla_out_abi.patch`, which touches only
  `csrc/api/sparse_prefill.cpp`. It adds optional caller-owned `out` /
  `max_logits` / `lse` and a new binding, `sparse_prefill_fwd_out`, so the
  kernel writes the Judge's tensors directly and no copy is timed. Upstream's
  own `sparse_prefill_fwd` path and its dispatch are unchanged.
- There are two candidates:
  - `candidates/flashmla_out_v001` uses the glue binding. **This is the paper
    number.**
  - `candidates/flashmla_copy_v001` uses the unpatched public
    `flash_mla_sparse_fwd` plus `copy_` into the Judge's tensors. It is a
    conservative lower bound.
- The image is built FROM the Judge's base image
  `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel` (torch 2.8.0, CUDA 12.9,
  Triton 3.4.0; the build asserts these) with `TORCH_CUDA_ARCH_LIST=9.0a` and
  `FLASH_MLA_DISABLE_SM100=1`.
- `oracle_runner.py` drives `sparse_mla_prefill_runtime.run_suite`, with the
  runtime file mounted read-only from `runtime/`. It covers the 13 frozen scored
  cases and the 401 private edge cases (400 run natively, 1 sanitizer-only), with guard bands, the decoy pre-call, a
  256 MiB L2 flush, max(host wall, CUDA event) timing, and the median of 7
  trials after 1 warm-up. Source admission is not applied, and neither is the
  sanitizer pass. `--mode peaked` is a distribution probe (q scaled by alpha),
  not a task score.

## Build and run

```bash
git clone https://github.com/deepseek-ai/FlashMLA /path/to/FlashMLA
git -C /path/to/FlashMLA checkout ba89a3466e9470ad08ab39738d4e7bb66989e1e7
git -C /path/to/FlashMLA submodule update --init csrc/cutlass

# from the task root: CPU-only image build (nvcc sm_90a, 30-60 min), no GPU taken
FLASHMLA_SOURCE=/path/to/FlashMLA oracle/paper_reference/run_flashmla_reference.sh --build-only
# measure on one idle H100 (takes the Judge scripts' per-UUID GPU lock)
oracle/paper_reference/run_flashmla_reference.sh --gpu <idx> --skip-build [--peaked]
```

The script resolves the Judge-only manifests the way `scripts/lib.sh` does. You
can override them with `FINAL_CASES_PATH` and `EDGE_CASES_PATH`. It checks their
SHA-256 against the frozen suites. `PAPER_CPUSET` pins CPUs. Results land in
`oracle/paper_reference/results/`, which git ignores. The script also times the
admissible Oracle candidate in the same image.

## Measured (H100 NVL 94 GB, 2026-10-29, score = family-geomean TFLOPS, higher is better)

| implementation | harness GPU 5 (this script) | harness GPU 7 (earlier copy) | full Judge |
|---|---|---|---|
| FlashMLA, caller-owned outputs (`flashmla_out_v001` = `oracle/candidates/kernel_v002`) | 400.86, edge 400/400 pass | 392.50, 392.39, edge 400/400 pass | INVALID at `source_ok`; forced-oracle Judge (GPU 7): 386.37 / 381.59 valid, sanitizer passed, 0/401 edge failures |
| FlashMLA, public API + copy (`flashmla_copy_v001`) | 356.63 | 350.58 | INVALID at `source_ok` |
| Oracle `kernel_v001` (Triton, admissible) | 149.84 | 149.50 | 145.998 valid (GPU 7, `forced-sparse-mla-admissible-v001-r1-g7`); see `oracle/notes.md` |

The GPU 5 column was produced by
`run_flashmla_reference.sh --gpu 5 --skip-build`, using the image
`standalone_sparse_mla_prefill.paper-oracle-flashmla:local` (`476178f2cf77`).
That image was built from the earlier copy of this `Dockerfile`, which differs
only in the names of the candidate directories, together with the pinned
commits. At run time the script mounts the candidates, the runner and the
runtime from this tree.
