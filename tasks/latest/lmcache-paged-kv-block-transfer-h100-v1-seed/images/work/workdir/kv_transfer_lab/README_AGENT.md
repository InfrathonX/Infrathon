# Working notes

Every `sforge-submit` runs the evaluation on hidden test cases. `sforge-submit --details` returns the gate states, the score, the four group scores and the seven family scores. Hidden cases, their shapes and per-case results stay on the Judge. Your final result is the best valid score you reach within the time budget, so keep `FINAL_VERSION` on your strongest candidate.

Read `AGENT_TASK.md` first: it holds the ABI, the correctness contract, the
support domain and how the score is computed. This file is only about the
mechanics of working here.

## Layout

```text
candidates/kernel_vNNN/submission.cu   your versions; NNN is three digits
FINAL_VERSION                          the integer version that will be judged
notes.md                               your measurements, one block per version
transfer_abi.h                         read-only copy of the contract header
tools/                                 local runners (not submitted)
build/                                 compile artifacts (not submitted)
```

Only `candidates/`, `FINAL_VERSION` and `notes.md` are submitted. Anything else
stays local and is never seen by the Judge, so use it freely.

`candidates/kernel_v001/submission.cu` is the supplied seed. Keep it as your
fallback; `FINAL_VERSION` can point back at it at any time.

## Loop

```bash
./tools/run_public_tests.sh --quick --candidate-version 3   # one case per family
./tools/run_public_tests.sh --candidate-version 3           # every public case
```

Both compile your file with the frozen nvcc command, audit the compiled
library's undefined symbols, and run the public edge cases before any timing.
Public tests differ from the hidden cases, which can be much larger.

Source admission runs before anything else and reports every violation at once
with file and line. If it rejects your file, nothing was compiled.

## Profiling

`nvcc` is available; compile private experiments however you like inside
`build/`. Only the submitted file must obey the contract.

## Submitting

Before each `sforge-submit --details`, wait for all public-test and compiler
workers to exit. A Judge run uses part of your budget.
