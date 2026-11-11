# Experiment notes

`v001` is the thin factory for the pinned official Hugging Face TRL 0.12.1
baseline. `v001` is the recovery/reference version; v001 itself contains no
rewritten ORPO math.

For every retained version, record:

- the optimization hypothesis and files changed;
- public command and correctness/gradient result;
- warm-up and repetition count;
- complete-step scored-completion TPS and run-to-run variance;
- peak CUDA allocated and reserved memory on both ranks;
- compile/autotune cost separately from timed steady state;
- comparison with the immediately preceding valid version; and
- keep/rollback decision.

Keep `FINAL_VERSION` on the strongest complete version that passed the visible
tests. Do not replace a stable version based on a single noisy timing sample.
