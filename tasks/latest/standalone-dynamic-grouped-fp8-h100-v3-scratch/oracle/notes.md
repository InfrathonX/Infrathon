# Oracle notes

`kernel_v001` is a self-contained Triton positive control for both entry
points.  One shared kernel implements the declared FP8 block-scale math; its
only layout-specific logic maps a group-local M tile either through the padded
masked base or the 128-aligned previous prefix-sum end.  It calls no existing
GEMM API and is never copied into either task image.

Selectors: `FINAL_VERSION` = 2 (strongest paper solution, forced),
`ADMISSIBLE_VERSION` = 1 (`kernel_v001`, the strongest candidate the normal
Judge admits; `scripts/make_oracle_archive.sh --admissible` and
`scripts/smoke_test.sh --oracle` use it).

`kernel_v002` is the paper solution, upstream DeepGEMM
(https://github.com/deepseek-ai/DeepGEMM @
`559d79fb6994a58b8a15b4b93bf13ccc16edf247`, v2.6.1): masked ->
`m_grouped_fp8_gemm_nt_masked`, psum -> `m_grouped_fp8_gemm_nt_contiguous(
use_psum_layout=True)`, upstream heuristics.  It equals
`paper_reference/candidates/deepgemm_v001/submission.py` plus one library
setting, `DG_USE_TEMP_CUBLASLT_WORKSPACE=1` (upstream's own compute-sanitizer
setting; without it DeepGEMM's exit-time destructor aborts the sanitizer
worker after all cases pass).  Source admission rejects `import deep_gemm`, so
it is scored only by the local forced-oracle Judge in
`paper_reference/forced_judge/` (admission bypassed, LLM review off, every
other gate enforced).

H100 NVL 94 GB, 2026-10-29, score = sqrt(masked x psum) of the per-family
geomean TFLOPS (higher is better):

| oracle | candidate | Judge | score | gates | run id |
|---|---|---|---|---|---|
| old / admissible oracle (Triton) | `kernel_v001` | normal | 126.88 (GPU 7); earlier 124.44 | valid, sanitizer passed, 158/158 edge | `forced-dyn-grouped-fp8-admissible-v001-r1-g7`; `pvsa-dyn-grouped-fp8-oracle-120736` |
| best admissible paper solution | none | - | - | DeepGEMM is the only upstream implementation; not admissible | - |
| forced strongest paper solution (DeepGEMM) | `kernel_v002` | forced-oracle | **166.76**, 164.94 (GPU 6; r2 masked 169.90, psum 163.68) | valid, sanitizer passed (6 cases), 158/158 edge | `forced-dyn-grouped-fp8-deepgemm-r2-g6`, `-r3-g6` |
| same, verbatim adapter (no workspace setting) | `deepgemm_v001` | forced-oracle | 0 (scored cases measured 166.12) | INVALID at `memory_safety_ok`: the sanitizer worker aborted at exit (0 memcheck errors); edge 158/158 passed | `forced-dyn-grouped-fp8-deepgemm-r1-g6` |
| DeepGEMM, normal Judge | `deepgemm_v001` | normal | 0 | INVALID at `source_ok` | `pvsa-dyn-grouped-fp8-paper-r1-120736` |

The SFA-transpose-hoisted variant (`paper_reference` `--mode hoisted`,
191.9) is not a valid glue: the ABI passes K-major FP32 `a_scale` per call
(fresh tensors every trial), so the MN-major transpose that SM90 DeepGEMM
needs belongs inside the call.  See `paper_reference/README.md` and
`paper_reference/forced_judge/README.md`.
