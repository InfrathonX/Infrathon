# Attn-QAT Training Attention Lab

## Submission evaluation policy

Every `sforge-submit` runs the complete hidden suite. `sforge-submit --details` returns the gate states and the score. Hidden cases and per-case results stay on the Judge. Your final result is the best valid score you reach within the time budget, so keep `FINAL_VERSION` on your strongest candidate.

Read `AGENT_TASK.md` first. Version 1 is a correct starting point. Both the forward and the
backward are timed. Keep a known-correct version while you experiment.

```bash
./tools/run_one.sh --version 1
./tools/run_public_tests.sh --quick --candidate-version 1
./tools/run_public_tests.sh --candidate-version 1
```

Only `candidates/`, `FINAL_VERSION`, and `notes.md` are submitted. The exact ABI, gates,
static host subset, public workloads, and scoring formula are in `AGENT_TASK.md`.

Before each `sforge-submit --details`, wait for all public-test and compiler workers to exit.
