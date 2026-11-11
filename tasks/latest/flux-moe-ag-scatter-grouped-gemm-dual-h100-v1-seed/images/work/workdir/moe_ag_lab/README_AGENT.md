# MoE all-gather, scatter and grouped GEMM lab (2 to 8 GPUs)

## Submission evaluation policy

Every `sforge-submit` runs the evaluation on the complete hidden suite. `sforge-submit --details` returns the gate states, the score and the six family scores. Hidden cases and per-case results stay on the Judge. Your final result is the best valid score you reach within the time budget, so keep `FINAL_VERSION` on your strongest candidate.

Read `AGENT_TASK.md` first. Version 1 is a correct seed behind the fourteen-argument
ABI: a call-counter kernel, a per-expert counting kernel, and one kernel whose first
programs pull every peer shard (ring order, chunked, per-chunk flags) while the rest
run the grouped GEMM, local tiles first.

The public cases in `/workspace/benchmark/public_cases.json` are for development and
profiling. The hidden suite is larger and differs from them, so make your
optimizations hold up across the support domain in `AGENT_TASK.md` rather than
fitting these shapes.

The six family scores are `tp8`, `tp4`, `tp2`, `ep`, `decode` and `sequence`; the
score is their geometric mean.

```bash
./tools/run_one.sh --version 1
./tools/run_public_tests.sh --quick --candidate-version 1
./tools/run_public_tests.sh --candidate-version 1
```

Quick runs four edge cases and one case per family; full runs every public case.
Only `candidates/`, `FINAL_VERSION`, and `notes.md` are submitted.
