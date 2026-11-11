# Host-only references

These sources never enter the Work image or ordinary candidate submission.
They are packaged host-side as ordinary candidate archives and evaluated
independently through the same Judge lifecycle for calibration and controls.

- `vanilla/` is materialized from the visible v001 seed by the archive script.
- `corrected_decoder/` is the frozen calibration candidate. It is not a timing
  oracle and is not loaded alongside another candidate.
- `scripts/make_reference_archives.py` deterministically derives an independent
  0.85 threshold positive control, a negative control that performs the complete
  corrected model call before degrading only its returned tokens, and a
  near-cap memory fixture that allocates/touches an explicitly requested MiB
  buffer inside the measured generation window. OOM is never suppressed.

The Judge-only quality manifest is derived from the corrected reference's exact
520-generation stratified-final run. Independent precommitted positive and
degraded-negative controls must qualify any final hard quality gate. The
evaluator image is then rebuilt from frozen workload, memory, timeout, quality,
and provenance policy before ordinary release replay.
