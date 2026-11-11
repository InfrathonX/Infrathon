# Workspace

Read `/app/workdir/decoder_lab/AGENT_TASK.md` before editing.

`solutions/decoder_v001/candidate.py` is the known deterministic starter. Keep
it unchanged and create complete new version directories for experiments.

Only `solutions/`, `FINAL_VERSION`, and `notes.md` are captured. Runtime,
visible-test, and instruction edits are not included.

Quick commands:

```bash
bash /workspace/public_tests/run_public_tests.sh --cpu-only
bash /workspace/public_tests/run_public_tests.sh --quick
bash /workspace/public_tests/run_public_tests.sh
```

By default Work and the Judge run on separate GPUs. Before each manual
submission wait for every public-test and compiler process to exit, then run
`sforge-submit --details`. In the shared-GPU (single-accelerator) profile Work
and Judge share the one GPU allocated to this run, and `sforge-submit`
automatically runs the root-owned `.sforge/prepare_submit.sh` drain hook first
and refuses to submit while your own processes still hold the GPU.

Every submitted candidate uses the complete scored evaluation: the same
requests, gates and limits as the final evaluation. The visible tests are
separate diagnostics and run their own, shorter generation lengths than the
scored evaluation.
