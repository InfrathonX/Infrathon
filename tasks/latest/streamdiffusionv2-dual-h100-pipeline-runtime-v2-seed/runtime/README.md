# Frozen task runtime (v2)

`sdv2_core/` is the frozen numerical and I/O layer. Its v2 changes are confined to
measurement ownership:

- `plan.StreamPlanMeta` is the frames-free plan handed to a candidate runtime. It
  carries chunk descriptors, the indexed noise provider and cache geometry; the
  input video reaches the runtime only through `start` and `push`.
- `sink.OutputSink` records emissions only. Push times are recorded by the trusted
  controller (`images/judge/candidate_worker.py`, mirrored in the Work image)
  immediately before it calls `start`/`push`, and written to `push_timestamps.json`.
- `reference.run_case` (the single-GPU Stream-Batch reference used for goldens)
  takes an optional `push_log` so its timing uses the same definitions.

`cases.py` defines and validates the two disjoint tiers (public 6, final 12; the
final tier must cover the paper's 832x480 / 512x512 x 1-4 step grid);
`manifests/` contains the public canonical JSON. `evaluation.py` owns exact ledger
persistence/comparison and the push-anchored timing; `scoring.py` owns
latency-weighted FPS and the seed-relative gates (TTFF, p99, construction, and the
per-rank GPU memory cap shared by both ranks); `hardware.py` owns the capability
envelope (two CUDA devices, cc >= 8.0, >= 26 GiB each, >= 64 GiB host RAM) that the
trusted worker enforces and the Judge verifier re-checks; GPU names and UUIDs are
recorded as provenance only. `media_gen.py` is the deterministic task-owned
procedural input generator.

The Work build omits `cases.py` before constructing its image layers. Work sees only
the public manifest, and it measures its own seed reference on the local GPU pair
rather than reading a frozen baseline.

`evaluation_protocol.json` is protected by the Judge build hash manifest. It binds
every formal round to the same full private suite by SHA-256, case count, metric and
time budgets without including private case parameters. The public runtime copy can
expose this non-secret identity; private inputs remain Judge-only.
