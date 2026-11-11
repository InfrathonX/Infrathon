# Paper reference: FlashAttention-3 (host-only)

This directory times **upstream FlashAttention-3**, the task's source paper, on
this task's ABI with the Judge's own trusted runtime, so agent scores can be put
next to the strongest published implementation. This is a timing harness
only. The real Judge result of the same FA3 call lives in `forced_judge/`:
`oracle/candidates/kernel_v003`, which `oracle/FINAL_VERSION` selects, is
scored by a local forced-oracle Judge image with source admission and the
LLM review bypassed. It gets 53.58 TFLOPS on the scored grid and is INVALID
on the edge gate. The admissible Oracle archive
(`scripts/make_oracle_archive.sh`) packs `oracle/ADMISSIBLE_VERSION`
(`kernel_v002`, vLLM Triton).

FA3 cannot take the scored path. It is compiled CUDA/CUTLASS reached through
`import flash_attn_3`, and source admission allows only
math/typing/torch/triton imports. It also fails the Judge's edge gate (see
below), so even without admission it would be INVALID on the Judge.

Isolation:

- no Work or Judge build context reaches `oracle/`; `scripts/build_images.sh`
  builds only `images/work` and `images/judge`, and
  `scripts/validate_layout.py` fails if either build names the oracle tree;
- nothing here feeds `image_source_digest()` (it hashes `images/work`,
  `images/judge`, `runtime/` and `judge/run_judge.py`);
- `scripts/make_oracle_archive.sh` packs only `FINAL_VERSION` (from
  `ADMISSIBLE_VERSION`), `notes.md` and that candidate, never this directory;
- the FlashAttention source tree is not vendored. `run_fa3_reference.sh`
  builds from a checkout you supply through `FA3_SOURCE`.

## Provenance

- Repository: https://github.com/Dao-AILab/flash-attention
- Commit: `0251105a2fb19d2957484b7f023cd8c115286ced` (the commit the task's
  semantics, tolerances and scored head configurations come from), with the
  submodule `csrc/cutlass` at `7127592069c2fe01b041e174ba4345ef9b279671`.
- Built: `hopper/` (flash_attn_3) for SM90 only. The variants this ABI never
  reaches are compiled out (backward, FP16, FP8, other head dims, SM80,
  softcap, local, append-KV); see `Dockerfile`.
- Glue (`fa3_entry.py`): one function with the task signature that calls the
  registered op `torch.ops.flash_attn_3.fwd` with `out_=out`, which is the call
  `flash_attn_with_kvcache` makes, with `cache_seqlens` as `seqused_k`,
  `page_table`, `causal=True`, `num_splits=0` and `pack_gqa=None` (FA3's own
  heuristics). The op is called directly because, at this commit, the
  `torch.library.custom_op` wrapper rejects `out_` aliasing its return value.
- `fa3_entry_public_api.py` goes through the public `flash_attn_with_kvcache`,
  which allocates its own output and then pays an `out.copy_`.
- `fa3_entry_cfg.py` with `fa3_tuned_pinned.json` uses per-shape
  `(num_splits, pack_gqa)` from a sweep of FA3's own knobs. This is a fairness
  check, not upstream behaviour.
- `fa3_preflight.py` is a CPU-only check that refuses FA3 builds with varlen or
  paged KV compiled out. The `flash_attn_3 3.0.0b1` wheel has both compiled
  out and cannot run this ABI at all.
- `fa3_runner.py` drives `paged_gqa_decode_runtime.run_suite`: the frozen
  scored manifest (10 cases, 1 warm-up and 7 trials, a decoy pre-call, and
  max(host wall, CUDA event)), and optionally the 160 private edge cases. The
  runtime file is mounted read-only from `runtime/`. Source admission and the
  compute-sanitizer pass are not applied.

## Build and run

```bash
git clone https://github.com/Dao-AILab/flash-attention /path/to/fa
git -C /path/to/fa checkout 0251105a2fb19d2957484b7f023cd8c115286ced
git -C /path/to/fa submodule update --init csrc/cutlass

# from the task root; the build is CPU-only (tens of minutes with MAX_JOBS=8)
FA3_SOURCE=/path/to/fa oracle/paper_reference/run_fa3_reference.sh --gpu <idx> --tuned
# later runs reuse the image
oracle/paper_reference/run_fa3_reference.sh --gpu <idx> --skip-build --tuned
```

The script resolves the Judge-only manifests the same way `scripts/lib.sh`
does (`judge/private/`, or the dataset assets that `assets/fetch.sh` places
under `${INFRATHON_ASSETS_ROOT}/<task directory>/judge/`). You can override them
with `FINAL_CASES_PATH` and `EDGE_CASES_PATH`. It checks their SHA-256 against
the frozen suites. It takes the per-UUID GPU lock that the Judge scripts use,
refuses a busy GPU, and accepts `FA3_CPUSET` for CPU pinning. Results land in
`oracle/paper_reference/results/`, which git ignores. The script also times
every `oracle/candidates/kernel_v*` in the same image, as a control for
torch/Triton differences against the Judge image.

## Measured (H100 NVL 94 GB, 2026-10-29, score = TFLOPS, higher is better)

| implementation | harness GPU 3 | harness GPU 4 | full Judge |
|---|---|---|---|
| FA3, own heuristics (`fa3_entry.py`) | 53.66 | 55.21 | INVALID: fails 3 of 159 edge cases (and source admission) |
| FA3, per-shape tuned knobs | 54.08 | 55.03 / 55.57 | not applicable |
| FA3, public API + `out.copy_` | 46.23 | 47.32 | not applicable |
| Oracle kernel_v002 (vLLM Triton, selected) | 24.13 | 24.38 | 24.40 / 24.17 (GPU 4), 24.03 (GPU 3), valid |
| Oracle kernel_v001 (straightforward control) | 13.25 | 13.41 | 13.33 (GPU 4), 13.26 (GPU 3), valid |

The GPU 3 harness column was produced by this exact script:
`run_fa3_reference.sh --gpu 3 --skip-build --tuned`, using the image
`paged-gqa-fa3-oracle:0251105a` (`d700e1078853`), which was built with the same
recipe as `Dockerfile`. The GPU 4 column comes from the earlier copy of the
same harness. For the Judge run ids, see `oracle/notes.md`.

FA3 edge failures: `edge-061-lengths`, `edge-098-heads` and `edge-099-heads`.
Each one passes when the unused KV page slots hold finite random values. Each
one fails when those slots hold NaN/Inf: the output then contains NaN, so FA3
lets invisible (masked) slots reach the result. The Judge's edge gate poisons
exactly those slots.

Regime notes (from the paper-vs-agent study): FA3 wins on compute-bound
S_q=16 cells and on skewed lengths. Triton kernels can match or beat it on
bandwidth-bound S_q=1 decode.
