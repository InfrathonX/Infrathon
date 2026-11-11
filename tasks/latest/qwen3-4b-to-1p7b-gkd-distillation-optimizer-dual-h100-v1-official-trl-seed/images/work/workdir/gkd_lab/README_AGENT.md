# Workspace

Read `/app/workdir/gkd_lab/AGENT_TASK.md` before editing.

Keep `solutions/objective_v001/` unchanged. Create complete versioned
directories for experiments, and implement the interfaces defined in
`runtime.api`. The optional model interfaces (student and teacher patches) are declarative;
do not export `optimize_model`.

Only `solutions/`, `FINAL_VERSION`, and `notes.md` are captured.

Public commands:

```bash
bash /workspace/public_tests/run_public_tests.sh --cpu-only
bash /workspace/public_tests/run_public_tests.sh --quick
bash /workspace/public_tests/run_public_tests.sh
```
