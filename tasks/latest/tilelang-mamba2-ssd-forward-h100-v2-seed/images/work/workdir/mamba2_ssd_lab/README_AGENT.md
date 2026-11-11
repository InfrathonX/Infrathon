# Working notes

Every `sforge-submit` runs the evaluation on hidden test cases. `sforge-submit --details` returns the gate states, the score and the four family scores. Hidden cases and per-case results stay on the Judge. Your final result is the best valid score you reach within the time budget, so keep `FINAL_VERSION` on your strongest candidate.

Read `AGENT_TASK.md` first: it holds the frozen tensor and math contract, the
allowed implementation surface, the public cases, and how the score is
computed.  This file is only about the mechanics of working here.

## Layout

```text
candidates/kernel_vNNN/submission.py   your versions; NNN is three digits
FINAL_VERSION                          the integer version that will be judged
notes.md                               your measurements, one block per version
tools/                                 local runners (not submitted)
build/                                 scratch and TileLang cache (not submitted)
```

Only `candidates/`, `FINAL_VERSION`, and `notes.md` are submitted.  Anything
else — scratch scripts, tuning harnesses, logs — stays local and is never seen
by the Judge, so use it freely.

`candidates/kernel_v001/submission.py` is the supplied seed.  Keep it: it is
your fallback if a later version regresses, and `FINAL_VERSION` can point back
at it at any time.

## Loop

```bash
./tools/run_one.sh --version 3                              # P1
./tools/run_public_tests.sh --quick --candidate-version 3   # P1 and P20
./tools/run_public_tests.sh --candidate-version 3           # edge fixtures, then all 24 cases
./tools/run_public_tests.sh --family long --candidate-version 3              # one family
./tools/run_public_tests.sh --case P13 --perf-only --candidate-version 3     # time one case
```

The full suite runs the 21 edge fixtures first.  A fixture failure stops the run
before any timing, which is deliberate: a fast wrong kernel is worth nothing
here.  The public tests cover the same four workload families as the hidden suite.  Your
score comes from hidden test cases, which have different shapes and are far more
numerous.
Make your kernel correct and fast across the whole support domain, not just on
the public shapes.

Source admission runs before anything else and reports every violation at once
with file and line.  If it rejects your file, nothing was executed.

## Torch inside a submission

Torch is for allocating your own buffers and making views of tensors -- nothing
else.  Every run records the torch operators each untimed call dispatches and fails the `tilelang_only_ok` gate on anything beyond
allocations and views: a GEMM, an elementwise or scan operator, a copy, a dtype
conversion, or `.item()`.  The public runner prints the operators it saw, so a
stray `x.float()` shows up locally, long before a submission.

## Autotuning

`tilelang.autotune` is rejected **inside a submission**.  It is allowed in your
own scratch scripts.  Every path must stay correct for every shape that can reach
it.

## Cost

Each case allows 300 s of warm-up, outside the timer.  The public runner reports
it in the `warm s` column.  The Judge compiles each hidden shape cold, inside one 4500 s evaluation limit: a kernel that
recompiles for every shape has to compile in well under 30 s per shape.

## Recording versions

Before the first scored public measurement of a new implementation, run
`sforge-save-version --label <short-hypothesis>` and wait for it to report
success.  Keep `notes.md` current: one short block per version saying what you
changed, what it measured, and whether you kept it.

Before each `sforge-submit --details`, wait for all public-test and compiler
workers to exit.
