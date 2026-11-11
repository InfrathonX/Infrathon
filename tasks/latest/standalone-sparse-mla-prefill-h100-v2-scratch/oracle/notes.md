# Notes

Host-only Oracle control for the standalone BF16 sparse-MLA prefill task.

`kernel_v001` is one Triton program per `(query token, head block)` with the
full `d_v = 512` accumulator in registers, the QK product split into the
512-wide NoPE half and the optional 64-wide RoPE tail, a running maximum and
denominator, P rounded to BF16 for the PV product with FP32 accumulation, and
the attention sink folded into the output normaliser only.  It imports nothing
outside the source-admission allowlist (in particular, not `warnings`).

Properties the contract relies on (checked by reading the source; the GPU
check is the sanitizer-only s_q = 70000 case):

- s_q is grid axis 0, so s_q may exceed 65535 (grid axes 1 and 2 stop there);
- every row offset (`q`, `out`, `indices`, `topk_length`, `max_logits`,
  `lse`) is formed from the int64 program id, and key rows from an int64 cast
  of the index, so `q` and `out` may hold more than 2**31 elements;
- `topk` is walked in blocks of 64 and the contract makes it a multiple of
  128, so no index load leaves its row.

On GPUs with less shared memory the kernel falls back to a smaller tile plan;
it does not report the fallback through `warnings`.

## Oracle selection and the paper solution

Selectors: `FINAL_VERSION` = 2 (strongest paper solution, forced),
`ADMISSIBLE_VERSION` = 1 (strongest candidate the normal Judge admits).

- `kernel_v001`: the in-task Triton control above.  It is the strongest
  *admissible* solution known (no admissible Triton port of FlashMLA exists
  upstream).  `scripts/make_oracle_archive.sh --admissible` packs it, and
  `scripts/smoke_test.sh --oracle` runs it through the normal Judge.
- `kernel_v002`: the paper solution, upstream FlashMLA's SM90 sparse-prefill
  kernel (https://github.com/deepseek-ai/FlashMLA @
  `ba89a3466e9470ad08ab39738d4e7bb66989e1e7`, `csrc/cutlass` @
  `147f5673d0c1c3dcf66f78d677fd647e4a020219`), called through the ABI glue
  binding `sparse_prefill_fwd_out` (`paper_reference/flashmla_out_abi.patch`)
  so it writes the Judge's tensors directly.  Byte-identical to
  `paper_reference/candidates/flashmla_out_v001/submission.py`
  (sha256 `e41e9303...0474fec`).  Source admission rejects it
  (`forbidden import: 'flash_mla.cuda'`; INVALID at `source_ok` on the normal
  Judge, `pvsa-sparse-mla-paper-admission-20261029-120907`), so it is scored
  only by the local forced-oracle Judge in `paper_reference/forced_judge/`: the
  private Judge image plus FlashMLA plus an admission override; everything else
  (holdouts, compute-sanitizer pass, 401 edge cases, timing, aggregation) is
  the normal Judge, and the LLM integrity review is skipped.

Measured on H100 NVL 94 GB, 2026-10-29 (family-geomean TFLOPS, higher is
better):

| oracle | candidate | Judge | score | gates | run id |
|---|---|---|---|---|---|
| old / admissible oracle (Triton) | `kernel_v001` | normal | 145.998 (GPU 7); earlier 148.72 (GPU 7), 150.13 (GPU 5) | valid, sanitizer ran, 0/401 edge failures | `forced-sparse-mla-admissible-v001-r1-g7`; `pvsa-sparse-mla-oracle-20261029-120907`; `oracleup-sparse-mla-v001-r1-g5` |
| best admissible paper solution | none | - | - | FlashMLA is the only upstream implementation; it is not admissible | - |
| forced strongest paper solution (FlashMLA, caller-owned outputs) | `kernel_v002` | forced-oracle (admission bypassed, LLM review off) | **386.37**, 381.59 (GPU 7) | valid; all other gates enforced and passed: sanitizer ran and passed (14 cases), 0/401 edge failures | `forced-sparse-mla-flashmla-r1-g7`, `forced-sparse-mla-flashmla-r2-g7` |
| FlashMLA, normal Judge | `kernel_v002` | normal | 0 | INVALID at `source_ok` | `pvsa-sparse-mla-paper-admission-20261029-120907` |

The forced run's families: v32 421.2, v4_flash 329.8, v4_pro 415.3 (r1).  The
trusted-runtime harness in `paper_reference/` (no admission, no sanitizer)
gave 392.4-400.9 for the same candidate and 350.6-356.6 for the public API
plus copy (`flashmla_copy_v001`); see `paper_reference/README.md` and
`paper_reference/forced_judge/README.md` for provenance, build steps and the
exact commands.
