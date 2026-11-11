# Oracle candidates

Until 2026-10-29 the task had no `oracle/`; the reference was the Work seed
`engine_v001`, a statement-level port of upstream EAGLE (SafeAILab/EAGLE @
`cb7e0841`, HF eager, dynamic tree 60/7/10).

- `FINAL_VERSION` = 2 selects the strongest official paper solution, a
  **forced** (non-admissible) candidate: `solutions/engine_v002` = EAGLE-3 in
  SGLang 0.5.20 behind the task ABI. It needs the LOCAL-ONLY forced-oracle
  Judge image (`paper_reference/forced_judge/`).
- `ADMISSIBLE_VERSION` = 1: the best admissible paper candidate is the seed
  itself (`images/work/workdir/spec_lab/solutions/engine_v001`, not duplicated
  here; `paper_reference/forced_judge/make_forced_oracle_archive.sh` packs it).
  Use it for anything that must pass the normal, admission-enforcing Judge.

## Measured (final holdout, H100 NVL GPU 7, Judge cpuset 28-31,60-63, 2026-10-29)

| candidate | Judge | score (tokens/s) | run id | gates |
| --- | --- | --- | --- | --- |
| engine_v001 seed = upstream EAGLE HF-eager (old reference, best admissible) | normal `:local` | 122.63 | `forced-eagle3-seedctl-gpu7-20261029-222346` | all 11 pass |
| **engine_v002 SGLang EAGLE-3 (FINAL_VERSION)** | forced | 414.16 | `forced-eagle3-v2-gpu7-20261029-213711` | all 11 pass; admission bypassed |
| engine_v002 repeat | forced | 415.10 | `forced-eagle3-v2-gpu7-r2-20261029-214543` | all 11 pass; admission bypassed |

Per cell (first forced run; seed control in brackets): chat_bs1 328.5 (118.0),
math_bs1 384.4 (136.9), code_bs1 357.9 (130.7), longctx_bs1 262.1 (104.9),
think_bs1 346.0 (125.8), mixed_bs8 1231.2 (122.1). Earlier seed records on NVL:
125.62 (qualification), 126.05 (release holdout), 115.98-126.12 (manual runs).
The forced Judge's own same-run seed_reference (warm-up cohort) measured 130.4
and 131.3 tokens/s. The forced result is 3.4x the seed and about half of the
best agent run (846 tokens/s, on SXM).

Oracle quality (forced run 1, 57,599 positions): soft disagreement rate 0.60%
(limit 3.28%; same-run seed 0.40%), max margin 0.75 (eps_hard 24), tier counts
>0.25: 11, >0.5: 1, >0.75: 0 (all within limits). O5 passed on all six cells.

Bypassed / caveats: source admission (7 enforced errors: `http.client`,
`socket`, `subprocess`, environment routing) and the LLM review (off). The
`memory_ok` gate only sees the worker's PyTorch allocator (17.16 GiB = the
handed-in target and draft); the SGLang server process holds its own weight
copy plus KV pool (`--mem-fraction-static 0.55`), so by device-level
accounting the engine would exceed the 27.7-30.0 GiB same-run cap. Build took
161-165 s of the 600 s cap.
