Oracle notes

The Oracle is host-only: no Work or Judge build context reaches this
directory, and agents never see it.  It exists so the host can check the
Judge with a known-good solution and compare agent scores against the
strongest admissible upstream implementation.

FINAL_VERSION selects kernel_v003, the strongest paper candidate
(FlashAttention-3).  It is NOT admissible and is scored only by the local
forced-oracle Judge (paper_reference/forced_judge/).  ADMISSIBLE_VERSION
selects kernel_v002 (vLLM Triton), the strongest admissible candidate.
Everything that must pass the normal Judge uses ADMISSIBLE_VERSION:
scripts/make_oracle_archive.sh, scripts/smoke_test.sh --oracle, and the
tests.

Summary (H100 NVL, TFLOPS, higher is better)

  role                    candidate    Judge                  gates                         score      run id
  old oracle / control    kernel_v001  normal                 valid, all pass               13.26      oracleup-paged-gqa-v001-r1-g3
  best admissible paper   kernel_v002  normal                 valid, all pass               24.03      oracleup-paged-gqa-v002-r1-g3
  forced strongest paper  kernel_v003  forced (admission      INVALID: edge_correctness_ok  53.58 (*)  forced-paged-gqa-fa3-v003-r2-g3-20261029-215854, -r3-g3-20261029-220005
                                       and LLM review         3/160 edge cases fail;
                                       bypassed)              sanitizer passed

  (*) Scored-grid score measured by the forced Judge after its own per-case
  validation and aggregation (forced_scored_score_tflops in the private
  result).  The Judge reports score 0, because the edge gate failed.

Scored suite v4 (final-holdout-v4, 2026-11-08)
---------------------------------------------
The v4 suite replaces the ten-case grid with 1114 frozen serving-engine decode
cases in four families (decode 440, spec 374, long 220, stream 80; plan
paged-gqa-plan-v2, Mooncake conversation and tool-agent traces); the table
above is the v2 grid and is not comparable.  Real Judge flow
(scripts/build_images.sh and scripts/eval_archive.sh with TRUSTED_CONTROL=1),
H100 SXM 80 GB, one GPU each:

  candidate                      verdict  score  decode  spec   long   stream  Judge wall
  kernel_v002 (oracle.tar.gz)    valid    3.63   3.82    4.36   2.25   4.64    1027 s
  r1 agent-20 kernel_v041        valid    2.77   4.85    8.40   3.79   0.38    2344 s

Both pass the sanitizer subset and all 168 private edge cases.  The r1 kernel
compiles per batch size and table width, so its sequence family collapses.
The candidate budget (7200 s) leaves about three times the slower run.
kernel_v003 (FA3, forced) was not measured on v4.

kernel_v003 (FINAL_VERSION, forced only): FlashAttention-3
----------------------------------------------------------
Upstream https://github.com/Dao-AILab/flash-attention at
0251105a2fb19d2957484b7f023cd8c115286ced (hopper/, flash_attn_3, sm_90a;
csrc/cutlass 7127592069c2fe01b041e174ba4345ef9b279671).  One call to
torch.ops.flash_attn_3.fwd with out_=out, seqused_k=cache_seqlens,
page_table, causal=True, num_splits=0 and pack_gqa=None, so FA3 applies its
own heuristics.  The code is identical to paper_reference/fa3_entry.py; only
the docstring differs (sha256
b501a7a41dbf73e4b5bc7389ad2f132d55b196c2b19144903becaa56ca29015f).

The Judge's own source admission rejects it (forbidden import
flash_attn_3._C, forbidden literal 'flash_attn', host call _fwd, no
@triton.jit launch).  In the forced Judge it passes the compute-sanitizer
subset and 156 of the 159 native edge cases.  It fails edge-061-lengths,
edge-098-heads and edge-099-heads: "out contains ... unwritten, NaN or
infinite elements", because NaN/Inf in unused KV page slots leaks into the
output.  The scored grid passes every scored-case check.

Forced Judge runs (image <judge>:forced-oracle built FROM the private
Judge image f5461236c1ce; GPU 3, cpuset 12-15,44-47;
SFORGE_LLM_JUDGE_MODE=off, --network none)

  run id                                           verdict        edge fail  sanitizer  scored grid (q1 / q16 / score)
  forced-paged-gqa-fa3-v003-r1-g3-20261029-214509  INVALID, edge  3/160      passed     15.61 / 185.25 / 53.78 (+)
  forced-paged-gqa-fa3-v003-r2-g3-20261029-215854  INVALID, edge  3/160      passed     15.52 / 184.97 / 53.58
  forced-paged-gqa-fa3-v003-r3-g3-20261029-220005  INVALID, edge  3/160      passed     15.51 / 185.15 / 53.58

  (+) r1 ran an earlier forced_verify.py.  Its sentinel lower bound counted
  2 checks for each failed edge case as well, so the forced_scored_*
  validation was rejected.  The value shown is the Judge's family geomean
  recomputed from the r1 per-case records.  r2 onwards use the committed
  forced_verify.py.

  r2 per case (TFLOPS): M1 8.64 / 112.24, M2 12.74 / 170.29,
  M3 17.68 / 276.52, M4 25.60 / 241.10, M5 18.06 / 169.89 (q1 / q16).

kernel_v002 (ADMISSIBLE_VERSION): upstream vLLM Triton paged attention
----------------------------------------------------------------------
The kernel vLLM's TritonAttentionBackend runs on H100 for paged decode and
chunked prefill: kernel_unified_attention (2D: one program per (q-block,
KV head); 3D: plus 16 KV segments) and reduce_segments, vendored from
https://github.com/vllm-project/vllm at commit
3c39dda2a5da9e4b71b5eb61db686bf669374569
(vllm/v1/attention/ops/triton_unified_attention.py,
vllm/v1/attention/ops/triton_attention_helpers.py), with the launch
heuristics of vllm/v1/attention/backends/triton_attn.py
(MIN_LAUNCH_GRID_SIZE_2D = 128, NUM_PAR_SOFTMAX_SEGMENTS = 16,
MAX_3D_QUERY_LEN = 8).

ABI glue only (the module docstring lists each change): dead constexpr
feature branches removed (quantized KV, sliding window, softcap, sinks,
ALiBi, qq-bias, multimodal prefix, R-SWA, tensor descriptors, FP8 output);
query_start_loc[i] replaced by i * S_q; optional/default arguments removed;
tl.int64 parameter annotations replaced by in-kernel casts (source
admission forbids annotations); page-table loads masked to the visible
prefix; split-KV scratch allocated per call.  Tile sizes, BLOCK_M /
BLOCK_Q, warps, stages, segment count and the 2D/3D switch are upstream's.
No H100-specific retune was applied.

kernel_v001 (kept): straightforward positive control
----------------------------------------------------
One Triton program per (request, query head), BLOCK_M=16 with a row mask
covering every S_q from 1 to 16, gather-style page translation (int64
addresses) with the row width read from page_table.shape[1], K/V loads
masked past the cache length so invisible slots never enter, online softmax
with a finite mask value applied by tl.where, and a bf16 P.V dot with fp32
accumulation.  It intentionally leaves the main optimisation levers unused:
every query head re-reads its KV head's cache (no GQA packing), there is no
split-KV, and page locality is ignored.

Judge scores (full Judge: source admission, compute-sanitizer subset, 160
private edge cases, 10 scored cases x 7 trials; H100 NVL, TRUSTED_CONTROL=1
which skips only the LLM integrity review; score = TFLOPS, higher is better)

  candidate    run id                                   valid  edge fail  score
  kernel_v001  pvsa-paged-gqa-oracle-r1-g4 (GPU 4)      yes    0/160      13.33
  kernel_v001  oracleup-paged-gqa-v001-r1-g3 (GPU 3)    yes    0/160      13.26
  kernel_v002  pvsa-paged-gqa-paper-vllm-r1-g4 (GPU 4)  yes    0/160      24.40
  kernel_v002  pvsa-paged-gqa-paper-vllm-r2-g4 (GPU 4)  yes    0/160      24.17
  kernel_v002  oracleup-paged-gqa-v002-r1-g3 (GPU 3)    yes    0/160      24.03
  (2026-10-29, dev-tree Judge image f5461236c1ce.  kernel_v002/submission.py
  has sha256 f654ce2edc7502ee43f5a0edb5d1567b1366a94c03e0371734d415f6c5322f82
  in every run; the oracleup-*-v002 run used the archive produced by
  scripts/make_oracle_archive.sh with FINAL_VERSION=2.)

Harness numbers for FA3 (paper_reference/)
------------------------------------------
paper_reference/run_fa3_reference.sh times FA3 with the Judge's runtime but
without admission, the sanitizer or the Judge container.  It gives 53.7-55.2
TFLOPS on the scored grid, and 54.1-55.6 with per-shape tuned FA3 knobs.
These agree with the forced Judge above.  The forced Judge is the reference
of record.  The tuned variant (fa3_entry_cfg.py) is a fairness check, not
upstream behaviour, and is not an Oracle candidate.  Neither kernel_v003 nor
paper_reference/ is part of the admissible Oracle archive.
