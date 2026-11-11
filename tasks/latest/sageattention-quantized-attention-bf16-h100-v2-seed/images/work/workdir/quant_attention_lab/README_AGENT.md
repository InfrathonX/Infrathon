# Quantized Attention Lab

Every `sforge-submit` runs the evaluation on hidden test cases. `sforge-submit --details` returns the gate states, the score and the four family scores. Hidden cases and per-case results stay on the Judge. Your final result is the best valid score you reach within the time budget, so keep `FINAL_VERSION` on your strongest candidate.

Read `AGENT_TASK.md` first. Version 1 is a working Triton seed. Preserve a
known-correct version while you change it. The public tests only exist to get
your kernel working; your score comes from hidden test cases, which differ from
them and can be much larger.

```bash
./tools/run_one.sh --version 1
./tools/run_public_tests.sh --quick --candidate-version 1
./tools/run_public_tests.sh --candidate-version 1
./tools/run_public_tests.sh --case public-p3 --perf-only --candidate-version 1
```

`--case ID` runs one public case; `--perf-only` times without the golden comparison.

Only `candidates/`, `FINAL_VERSION`, and `notes.md` are submitted. The exact
ABI, gates, static host subset, workload, and scoring formula are in
`AGENT_TASK.md`.

Before each `sforge-submit --details`, wait for all public-test and compiler
workers to exit.
