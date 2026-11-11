# KV Eviction Lab v2 — Scratch

Read `AGENT_TASK.md` first. Version 1 keeps the first four cached positions
plus the most recent ones and computes no score. Preserve this known-good
version while you explore.

```bash
./tools/run_one.sh --version 1
./tools/run_public_tests.sh --quick --candidate-version 1
./tools/run_public_tests.sh --candidate-version 1
```

The public items are in `/workspace/benchmark/corpus/items_public.jsonl`
(`--quick` takes the first item of every public task, the full suite all 85).
Only `candidates/`, `FINAL_VERSION`, and `notes.md` are submitted.

Every scored `sforge-submit` runs the complete private suite on separate Judge
GPUs and `sforge-submit` waits for the result (on the order of an hour), so
iterate on the public suite and submit deliberately. Wait for public-test and
compiler processes to exit before submitting. Policies must not use token
identity in any form; see "What a policy may use" in `AGENT_TASK.md`.
