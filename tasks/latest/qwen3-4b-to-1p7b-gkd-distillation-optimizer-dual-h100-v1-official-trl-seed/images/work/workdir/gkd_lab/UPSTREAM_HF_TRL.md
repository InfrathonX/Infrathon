# Upstream Hugging Face TRL baseline

The baseline in this workspace is backed by the official Hugging Face TRL
package installed in the Work image. It is not a task-local reimplementation of
the generalized knowledge-distillation (GKD) equations.

Frozen provenance:

| Field | Value |
|---|---|
| Distribution | `trl==0.12.1` |
| Official repository | `https://github.com/huggingface/trl` |
| Git tag | `v0.12.1` |
| Peeled tag commit | `c1e9ea8ecfdaf1486e2fc84bc55379b2338d5320` |
| PyPI wheel SHA-256 | `502b958ac95313bd3e1a31ce91bc852b14f90303bc9b973ede1ee959f40c5c5a` |
| Installed source | `trl/trainer/gkd_trainer.py` |
| Installed source SHA-256 | `09e6e8932297ca3793aa5c429852bb46e3f395f9f3aaa48479d3b7965a028f1a` |
| License | Apache-2.0 |

The image build and public tests verify the installed version and source hash.
The task-owned harness supplies fixed tensors, the frozen teacher forward,
distributed execution, timing, memory measurement, and validation. The
baseline objective performs both vocabulary projections and delegates the
generalized Jensen-Shannon divergence to the pinned upstream static method
`GKDTrainer.generalized_jsd_loss` (temperature scaling, the beta mixture, and
the batchmean reduction over completion tokens).

The training contract follows the upstream off-policy semantics:

- each row is one prompt-plus-chosen-completion sequence;
- completion-masked, next-token-shifted labels select the scored positions;
- the on-policy student-sampling path of the upstream trainer (`lmbda`,
  `seq_kd`) is data generation, not loss math, and is outside this benchmark.

`solutions/objective_v001/candidate.py` is only a factory selecting this pinned
official baseline. It intentionally contains no independent GKD loss math.
