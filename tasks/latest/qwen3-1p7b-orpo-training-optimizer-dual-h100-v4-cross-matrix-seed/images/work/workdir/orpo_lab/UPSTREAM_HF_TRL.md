# Upstream Hugging Face TRL baseline

The baseline in this workspace is backed by the official Hugging Face TRL
package installed in the Work image. It is not a task-local reimplementation of
the ORPO equations.

Frozen provenance:

| Field | Value |
|---|---|
| Distribution | `trl==0.12.1` |
| Official repository | `https://github.com/huggingface/trl` |
| Git tag | `v0.12.1` |
| Peeled tag commit | `c1e9ea8ecfdaf1486e2fc84bc55379b2338d5320` |
| PyPI wheel SHA-256 | `502b958ac95313bd3e1a31ce91bc852b14f90303bc9b973ede1ee959f40c5c5a` |
| Installed source | `trl/trainer/orpo_trainer.py` |
| Installed source SHA-256 | `ac04e1e4b530af2f8fc64e5d47202b4a123a74bf08f1ccc0b60ba250a670c0fe` |
| License | Apache-2.0 |

The image build and public tests verify the installed version and source hash.
The task-owned harness supplies fixed tensors, distributed execution, timing,
memory measurement, and validation. The baseline objective delegates its ORPO
log-probability and odds-ratio operations to the pinned upstream class.

The training contract follows the upstream decoder-only semantics:

- completion-masked labels are used for chosen/rejected OR log-probabilities;
- a separate target containing prompt plus completion, with padding masked, is
  used for the chosen NLL term.

`solutions/objective_v001/candidate.py` is only a factory selecting this pinned
official baseline. It intentionally contains no independent ORPO loss math.
