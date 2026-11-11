# Miles OPD: starter, reference, Judge and audit hooks

Shared code of [`miles-qwen3-opd-scratch-2xh100-v2-seed`](../../tasks/latest/miles-qwen3-opd-scratch-2xh100-v2-seed/).
The task's `scripts/build_images.sh` copies it into the Work and Judge images.

- `starter/opd-starter-78527b910.patch` — the starter: Miles upstream `78527b910` with all
  on-policy distillation removed. Apply with `git apply` on that commit; it becomes the Work
  image's `repo/`.
- `reference/opd-reference-78527b910.patch` — the reference: upstream on-policy distillation
  plus a per-step record dump. The Judge image holds it as a root-only tree and runs its probe
  next to every candidate's.
- `v2/run_judge.py` — the Judge: a one-step probe of the reference and the candidate on hidden
  fixed trajectories, the full training contract under the audit and provenance hooks, and the
  accuracy evaluation of the final student.
- `v2/run_public.py` — the public suite inside Work.
- `v2/opd_checks.py` — the scoring arithmetic, per-step self-consistency checks and shared helpers.
- `v2/opd_probe_check.py` — the one-step probe checks (P1–P5).
- `v2/opd_step_oracle.py` — an fp32 Hugging Face recomputation of one training step.
- `v2/eval_math.py` and `v2/math_grader/` — the math accuracy evaluation and its grader
  (vendored from Miles, see its README).
- `v2/make_split.py` — builds the task's dataset split (used by the task's `scripts/prepare_assets.py`).
- `v2/audit/` — the instrumentation hooks (`*.pth` + modules) and their checks.

Miles upstream: [radixark/miles](https://github.com/radixark/miles) at `78527b910`, Apache-2.0.
