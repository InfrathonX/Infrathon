# KV Scoring Lab — Scratch

Read `AGENT_TASK.md` first. Version 1 is intentionally empty; write the whole
`KVScorer` and keep every attempt as `scorer_vNNN`.

```bash
./tools/run_one.sh --version N
./tools/run_public_tests.sh --quick --candidate-version N
./tools/run_public_tests.sh --candidate-version N
./tools/check_submission_size.sh
```

The public rows are in `/workspace/benchmark/corpus/items_public.jsonl`
(`--quick` takes the first two rows of every task, the full suite all 63).
Only `candidates/`, `FINAL_VERSION`, and `notes.md` are submitted.

Public tests are for getting your scorer working; your score comes from hidden
test cases. `sforge-submit` waits for the result, so iterate on the public suite
and submit deliberately. Your final result is the best valid score you reach
within the time budget. Wait for public-test and compiler processes to exit
before submitting. Scorers must not use token identity in any form, and each
score call must do its own work; see `AGENT_TASK.md`.
