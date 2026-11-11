# Build a CUDA BF16 KV codec from scratch

`/workspace/submission` starts empty. Design and implement a persistent KV-cache
codec that works across video-generation workloads. The goal is to use as few
persistent GPU bytes as possible while keeping generation quality high and
end-to-end runtime practical. The score rewards both: the persistent-KV
compression ratio (uncompressed BF16 KV bytes divided by the unique
CUDA-storage bytes reachable from the encoded payload), scaled by a
generation-quality factor; see "Score" below. Higher is better.

Only `/workspace/submission` is editable and submitted. Create:

```text
mlsysforge_codec.py          required
codec/**/*.py               optional implementation modules
notes.md                    optional
```

## API

`mlsysforge_codec.py` must export exactly the useful codec boundary:

```python
def encode(k, v) -> tuple[object, object]: ...
def decode(packed_k, packed_v) -> tuple[torch.Tensor, torch.Tensor]: ...
```

`k` and `v` are same-shaped, contiguous CUDA BF16 tensors in four-dimensional
BHSD layout. `decode(*encode(k, v))` must return same-shaped tensors on the
same CUDA device with BF16 dtype.

Each returned payload represents one tensor independently: `decode(p, p)`
must work for either the K or V payload. Payloads may contain only plain
`dict[str, ...]`, tuple/list, CUDA tensors, and small scalar metadata
(`None`, bool, int, finite float, or string). The runtime passes payloads
unchanged. All input-derived state needed by `decode` must be reachable from
them.

Persistent size is the full underlying CUDA storage of every payload tensor,
counted once across aliases, plus the scalar metadata: 8 bytes per int or
float, 1 per bool, and the UTF-8 length of every string and dictionary key.
Payload tensors must be plain dense `torch.Tensor`s (no subclass, nested,
sparse, quantized or named tensors, no autograd state or Python attributes).
Both sides of the ratio cover exactly the tensors
the runtime passes to `encode`; a workload may keep part of its cache (for
example its newest chunk) in BF16 outside the codec, and that part counts
toward neither side. Both sides are summed over every `encode` call of a
generation, not only the last: each compressed segment or chunk calls `encode`
once for every cached attention layer that has history to compress (one
`[B,H,S,D]` layer per call), and the
sums also run over every hidden case of a workload before the ratio is taken. Do not retain
original BF16 inputs or input-derived module/global state; do not use CPU/disk
offload, filesystem, process, network, serialization, raw pointers, or hidden
side channels. Do not change global CPU or CUDA RNG state. Allocate only on
the device of the tensors you are given: do not switch the current CUDA device
or reset CUDA memory statistics, and leave no memory on any other GPU (the
Judge rejects a case that does). Generalize across
compatible shapes; never identify or specialize to a caller, model, prompt, or
test case.

The participant-visible admission policy is
`/workspace/public_tests/codec_source_policy.py`.

## Score

There are two hidden workloads. Each one gets a compression ratio `R` and a
quality term `q`:

```text
R     = sum of BF16 KV bytes / sum of stored payload bytes
        (over every encode call of every hidden case of the workload)
score = sqrt( R_1 * (1 + q_1)  *  R_2 * (1 + q_2) )
q     = 0.5 * f + 0.5 * v
```

`q` compares generation quality with a fixed reference codec, which scores
`q = 0`. All readings are means over
the workload's hidden cases.

- `f` (fidelity): PSNR, SSIM and LPIPS of the generated frames against BF16
  frames, on the first chunk generated from compressed KV. Each metric scores
  the fraction of the way from the reference codec's reading to the ideal
  (PSNR 40 dB, SSIM 1, LPIPS 0). The fraction is capped at 1 and is negative
  when the metric is worse than the reference. `f` is the mean of the three.
- `v` (video quality): four VBench dimensions (aesthetic quality, background
  consistency, image quality, subject consistency) over the whole generated
  video. Each dimension has a margin,
  `1 - max(BF16 reading - reading, 0) / allowed drop`. The margin is 1 at or
  above the BF16 reading and 0 when the reading has dropped by the full
  allowed drop. `v` is the smallest of the four margins minus the reference
  codec's smallest margin. The dimension closest to its limit therefore
  decides `v`, and readings above BF16 earn nothing extra.

`1 + q` multiplies the compression ratio, so `q = 0.5` is worth as much as
50% more compression. A codec that is worse than the reference scores below
its compression ratio. Compression still
matters most, but bytes saved at the cost of fidelity or of blur later in the
video cost score. The reference readings, the BF16 readings and the
allowed drops are hidden.

The score is 0 unless every hidden gate passes: the case-mean and per-case
limits on the same fidelity and VBench metrics, and every case's time limit.

## Development loop

Start with the fast ABI diagnostic:

```bash
python /workspace/public_tests/codec_source_policy.py \
  --submission /workspace/submission --json
python /workspace/public_tests/run_public_tests.py
```

When a candidate is promising, run the larger public end-to-end diagnostic:

```bash
python /workspace/public_tests/run_self_forcing_paper.py \
  --mode candidate --prefixes 350 --vbench
```

It writes PSNR, SSIM and LPIPS against BF16 reference frames and, with
`--vbench`, the four VBench dimensions: the local view of both parts of `q`.
`run_public_tests.py` reports the compression ratio and relative L2, not `q`.

GPU-heavy checks from multiple delegated agents must run serially. Local tests
are diagnostics, not the evaluator: submit strong milestones with
`sforge-submit`, and submit the strongest complete implementation.

Every scored `sforge-submit` runs the complete frozen hidden suite: the same
cases, quality gates and time limits as the evaluation after the deadline.
Each case generates a full long video with your codec in the loop, so one
submission takes roughly 45-60 minutes and `sforge-submit` waits for it. Use
the public diagnostics for iteration and submit deliberately. Feedback is
aggregate only: validity, score and a generic status.

## Submitting

This lab has one GPU of its own; the Judge runs on separate GPUs. Wait for
every public test and compiler process to finish before
`sforge-submit --details`, so the archive is the version you intend. In-run
automatic hidden evaluation is disabled: only your manual submissions run the
Judge while you work, and the archive at the deadline is judged independently
after Work ends.

Your final result is the best valid score you reach within the time budget.
