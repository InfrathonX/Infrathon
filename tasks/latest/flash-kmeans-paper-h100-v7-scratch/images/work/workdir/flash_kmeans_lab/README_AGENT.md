# Workspace

Read `AGENT_TASK.md`.

This scratch workspace intentionally has no starting implementation, starter,
reference, baseline, or oracle source. Create
`kernels/flash_kmeans_v001.py`, write `1` to `FINAL_VERSION`, run
`bash /workspace/public_tests/run_public_tests.sh --quick`, and use
`sforge-submit --details` for authoritative Judge feedback.

Wait for public tests and compiler workers to finish before you submit.

Use the full public fresh-process scale ladders to study how the implementation
behaves when `N`, `K`, and `D` become very large. Treat their sampled
correctness, raw effective throughput, and relative scaling ratios as
development diagnostics. Do not hard-code public tuples; hidden tuples
and per-case timings are not disclosed.

Keep only complete versioned implementations in `kernels/`, keep the best
Judge-passing version selected, and record experiments in `notes.md`.
Your final result is the best valid score you reach within the time budget, so keep `FINAL_VERSION` on your strongest candidate.
