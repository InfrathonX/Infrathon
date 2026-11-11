# Miles OPCD: Judge, public runner and audit hooks

Shared code of [`miles-qwen3-opcd-2xh100-v3-seed`](../../tasks/latest/miles-qwen3-opcd-2xh100-v3-seed/).
The task's `scripts/build_images.sh` copies it into the Work and Judge images.

- `v3/run_judge_v3.py` — the Judge: a one-step probe on hidden fixed trajectories, the full
  training contract under the audit and provenance hooks, and the accuracy evaluation of the
  final student.
- `v3/run_public_v3.py` — the public suite inside Work: small runs for getting a pipeline working, no score.
- `v3/opcd_checks.py` — the scoring arithmetic, the quality protocol and shared checks.
- `v3/opcd_oracle.py` — an fp32 Hugging Face recomputation of the top-k reverse-KL loss from a
  Miles debug dump, independent of Miles.
- `v3/opcd_probe_check.py` — the one-step probe checks.
- `v3/eval_medmcqa.py` — the SGLang accuracy evaluation (seeded sampled repeats).
- `v3/audit/` — the SGLang audit check and the Megatron/SGLang provenance check.
- `audit/` — the instrumentation hooks (`*.pth` + modules) installed in the images, the
  same checks, and negative controls for them.
- `tests/test_v2_provenance.py` — a GPU self-test of the provenance hook.

Miles upstream: [radixark/miles](https://github.com/radixark/miles) at `78527b910`, Apache-2.0.
