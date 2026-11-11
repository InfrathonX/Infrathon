# Quant-VideoGen: cross-model KV-cache codec (H100, scratch)

Autoregressive video generators keep a large attention KV cache across long
generations. In this task the agent designs a persistent KV-cache codec for
CUDA BF16 tensors that works unchanged across two hidden video-generation
workloads, preserving generation quality and runtime while storing as few GPU
bytes as possible. It is a "scratch" task: the submission directory starts
empty, and the agent writes the codec from nothing.

## The task

The agent works in a Work container with one GPU of its own. Only
`/workspace/submission` is editable and submitted:

```text
mlsysforge_codec.py    required
codec/**/*.py          optional helper modules
notes.md               optional
```

`mlsysforge_codec.py` exports the codec ABI:

```python
def encode(k, v) -> tuple[object, object]: ...
def decode(packed_k, packed_v) -> tuple[torch.Tensor, torch.Tensor]: ...
```

`k` and `v` are same-shaped, contiguous CUDA BF16 tensors in BHSD layout;
`decode(*encode(k, v))` returns same-shaped BF16 tensors on the same device.
Each payload must represent one tensor on its own (`decode(p, p)` works for
either payload) and may contain only plain `dict`/tuple/list containers, plain
dense CUDA tensors and small scalar metadata. A read-only task runtime owns the
model plumbing and forwards cache compression calls to the submission; it has
no fallback codec.

Rules, enforced by the Judge (static source admission, runtime checks and an
LLM integrity review):

- Only regular Python files in the small submission closure are admitted. The
  participant-visible policy is `/workspace/public_tests/codec_source_policy.py`
  (`policy/codec_source_policy.py` in this directory). It rejects, among other
  things, filesystem, network and process access, native artifacts,
  serialization, CPU/disk offload, dynamic code, raw pointers, tensor or
  torch/Triton class subclassing, and input-derived global or module state.
- No retained BF16 inputs or hidden side channels; all state `decode` needs must
  be reachable from the payload.
- The global CPU and CUDA RNG state must not change.
- Allocate only on the device of the given tensors: no device switching, no
  resetting of CUDA memory statistics, and no memory left on any other GPU.
- The codec must generalize across compatible shapes and must not identify or
  specialize to a caller, model, prompt or case.

The full agent instructions are in `images/work/AGENT_TASK.md`. Public
diagnostics in the Work image:

- `python /workspace/public_tests/run_public_tests.py`: a fast check of the ABI,
  source boundary, independent K/V decoding, RNG behavior, basic fidelity and a
  minimum compression floor (it reports timing but does not gate on it);
- `python /workspace/public_tests/run_self_forcing_paper.py --mode candidate
  --prefixes 350 --vbench`: an end-to-end public Self-Forcing (Wan2.1 1.3B)
  generation with the candidate codec in the loop, compared against
  precomputed BF16 references.

The launcher's default agent budget is 24 hours (`--hours` / `--timeout`
override it). Scheduled automatic evaluation is off; the agent submits
manually with `sforge-submit`, with a 300 s cooldown between submissions.

## Evaluation

The score is quality-weighted compression: per hidden workload, the
persistent-KV compression ratio times a quality factor, then the geometric
mean over the two workloads:

```text
R_w   = sum over cases of w, sum over encode calls (BF16 KV bytes)
      / sum over cases of w, sum over encode calls (stored payload bytes)
q_w   = 0.5 * f_w + 0.5 * v_w
score = sqrt(R_1 * (1 + q_1) * R_2 * (1 + q_2))
```

Unit: `x` (the official Quant-VideoGen codec scores its plain compression,
sqrt(6.88 x 6.75) = 6.82); higher is better. Stored bytes
are the full underlying CUDA storage of every payload tensor, counted once
across aliases, plus scalar metadata (8 bytes per int or float, 1 per bool,
the UTF-8 length of every string and dictionary key). Both sides cover exactly
the tensors passed to `encode` and are summed over every compression call of a
generation (one call per cached attention layer per compressed segment or
chunk) and over every case of the workload. Byte counters are accumulated by
trusted, digest-pinned integration code; each case's BF16 byte count is frozen
in the release and must be reproduced exactly. There is no timing term.

The quality term `q_w` is 0 for the official Quant-VideoGen codec
(`svg-project/Quant-VideoGen` at `0601468f` with the `run_qvg.sh` settings:
INT2, 256 centroids, block 64). Its case means on the hidden suite are
measured on each hardware type (two full Judge runs, averaged) and frozen per
type in the Judge-only asset `hw/<type>/judge/h100_scratch_v4_quality.json`
(`models.<workload>.quality_reference`, provenance and the oracle's score band
in `oracle_calibration`). Hardware without such a file is refused
(`docs/NEW_HARDWARE.md` describes how to add one).
`q_w` is computed from the workload's case means:

- `f_w`, fidelity to BF16 on the first compressed chunk's frame window: for
  PSNR, SSIM and LPIPS, the fraction of the way from the reference codec's
  reading to the ideal (PSNR 40 dB, SSIM 1, LPIPS 0), capped at 1 and negative
  when worse than the reference; `f_w` is the mean of the three.
- `v_w`, video quality over every frame: for each of the four VBench
  dimensions the margin `1 - max(BF16 - reading, 0) / allowed drop` (allowed
  drop of the mean gate), so 1 at or above BF16 and 0 at the gate. `v_w` is
  the smallest margin minus the reference codec's smallest margin: the
  dimension closest to its gate decides, and
  readings above BF16 earn nothing.

`1 + q_w` is at most about 1.8, so compression still dominates; a codec worse
than the reference scores below its compression ratio. The private result also
records the plain compression geometric mean as `compression_score`. The
compression ratio plus the seven quality metrics are the eight columns of the QVG
paper's main table; the way
they are combined, the 40 dB ceiling and the equal weights are this task's
choices.

A submission is valid only when, for each workload:

- the case-mean fidelity gates pass: PSNR, SSIM and LPIPS of the generated
  frames against frozen BF16 goldens (on the first compressed chunk's frame
  window), and VBench (aesthetic quality, background consistency, image quality,
  subject consistency) as an allowed drop below each case's BF16 reading;
- every case passes looser per-case floors for the same metrics;
- every case finishes within the workload's fixed per-case wall-time budget;
- the runtime checks pass (expected number of compression calls, the assigned
  device only, no memory on other GPUs).

The gate thresholds are calibrated from repeated runs of a trusted reference
codec and live in the Judge-only release config. The gates alone decide
validity; the quality term never admits or rejects. An invalid result is
reported as `valid=false, score=0`. Ranking is valid-then-score.

The hidden suite is a fixed set of long-video generation cases (text-to-video
continuations over several segments on one workload, image-conditioned
camera-trajectory generations on the other). The agent never sees the models,
cases, prompts, goldens or thresholds; the Judge returns only validity, score
and a generic status.

Every interactive `sforge-submit` runs the complete hidden suite, the same one
used for the final evaluation (about an hour per submission: the reference
solution took 56 minutes on the six H100 NVL Judge GPUs and 44 minutes on six H100 SXM 80 GB). The final evaluation judges the submission archive captured when the agent budget
ends, independently and without feedback to the agent.

## Hardware

| hardware type | support |
|---|---|
| `h100-94gb-400w` (H100 NVL 94 GB) | qualified |
| `h100-80gb-700w` (H100 SXM 80 GB) | qualified |

This task uses 1 Work GPU and 6 Judge GPUs (7 per run), on separate cards of the
same host: the benchmark setup. By default Work uses GPU 0 and the Judge GPUs 1-6;
`--judge-gpus` picks other GPUs, and the launchers refuse any count other than six.
The Judge shards the hidden cases across its six GPUs (heaviest first).

Every GPU must be idle at admission, have compute capability 8.0 or higher and
at least 80,896 MiB of memory (the release floor is 79 GiB, 84,825,604,096
bytes, per Judge GPU), so 80 GB- and 94 GB-class H100 cards both qualify. Each
case runs on one card. With the official codec a LongCat case peaks at 71,895
MiB on its card (69.2e9 bytes allocated by torch) and an HY-WorldPlay case at
57,983 MiB, which on an 80 GB card (81,559 MiB) leaves about 9.4 GiB for a
candidate's extra memory: a codec that keeps much more persistent KV or needs
much more scratch than the official one can run out of memory on an 80 GB card
but not on a 94 GB one. Only the BF16 reference generation needs more (HY peaks
at 97.4e9 bytes); it produced the goldens on H100 NVL once, and every hardware
type uses those frozen files. Judge GPUs must be identical. Running six cases
at once (one per Judge GPU) needs up to roughly 300 GiB of host RAM (a LongCat
case peaks at about 37 GiB; the Judge container is sized at 32 CPUs and 448 GiB).

## Running

1. **Assets.** `tasks/latest/quant-videogen-cross-model-kv-codec-h100-v4-scratch/assets/fetch.sh`
   fetches everything the task reads, about 200 GiB under `INFRATHON_ASSETS_ROOT`
   (`assets/manifest.json` lists every item with its SHA-256):
   - eight models from their Hugging Face repositories at pinned revisions (about
     184 GiB), downloaded as `hf download --local-dir` does, so each tree keeps the
     revision records `scripts/verify_models.py` checks. `FLUX.1-Redux-dev` is gated:
     accept its license and run `hf auth login` (or export `HF_TOKEN`) first;
   - four git checkouts at pinned commits (VBench, DINO, the official HY-WorldPlay
     repository, Glyph-SDXL-v2 from ModelScope, which needs `git-lfs` on the host);
   - the eight VBench/LPIPS weights from their URLs;
   - from the Infrathon dataset: the Judge-only release config, the public evaluation
     inputs (`/eval/public`, 3.7 GiB) and the Judge-only hidden ones (`/eval/hidden`,
     5.3 GiB: the prompt suite, the six LongCat init videos, the HY case images and the
     17 BF16 goldens), both generated on H100 NVL.

   It then assembles the tree the launcher mounts from (`staging/`, `public-eval/`,
   `hidden/`) at `<asset root>/<task directory>/artifact-root`, as hard links into the
   asset root (`scripts/assemble_artifact_root.py`; no second copy, so it must be on the
   same file system). The scripts use that tree unless `QVG_ARTIFACT_ROOT` names another
   one. An interrupted download resumes on the next `fetch.sh`. `bash
   scripts/verify_models.sh` checks the tree; `portable/REBUILD_PROMPT.md` records the
   sources and how `scripts/stage_eval_inputs.py` generated the evaluation inputs.
2. **Images.** `INFRATHON_PULL=1 tasks/latest/quant-videogen-cross-model-kv-codec-h100-v4-scratch/scripts/build_images.sh`
   pulls the published Work and public Judge images for this host's hardware type and builds the
   Judge holdout layer on the host, which adds the release config from the asset root.
   Without `INFRATHON_PULL=1` it builds everything from source: the base (a digest-pinned
   `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` plus a fully pinned requirement lock),
   Work, public Judge and the holdout layer.
3. **Run.**

   ```bash
   launch/quant-videogen/run.sh --model MODEL \
     [--work-gpu 0 --judge-gpus 1,2,3,4,5,6] [--dry-run]
   ```

   Defaults: Work GPU 0, Judge GPUs 1-6 (the benchmark setup), 24 hours, effort `max`, Judge
   port 18623; `--work-gpu`, `--judge-gpus`, `--hours`, `--effort` and `--judge-port` change
   which GPUs are used (always exactly six Judge GPUs).
   `--agent codex` (default) or `--agent claude-code`; `--dry-run` resolves and
   checks the plan without touching GPUs or containers. The launcher requires
   the artifact tree from step 1, a host agent runtime (`SFORGE_AGENT_RUNTIME_DIR`), a version store
   (`SFORGE_VERSION_STORE_ROOT`, see `launch/mlsysforge.env.example`), model
   credentials (`--auth` for Codex, `--claude-token` for Claude Code, or an
   OpenAI-compatible `MODEL_API_BASE_URL` with a key file), and the
   `SFORGE_LLM_JUDGE_*` integrity review configured with
   `SFORGE_LLM_JUDGE_MODE=enforce`. Work runs on an internal network whose only
   egress is a restricted proxy.

   Evaluate one archive without an agent:

   ```bash
   JUDGE_GPUS=1,2,3,4,5,6 tasks/latest/quant-videogen-cross-model-kv-codec-h100-v4-scratch/scripts/eval_archive.sh ARCHIVE.tar.gz
   ```

   Use exactly six Judge GPUs here too (the benchmark setup). Replay saved submissions of an
   earlier run with `JUDGE_GPUS=1,2,3,4,5,6 launch/quant-videogen/replay.sh` (delegates to
   `scripts/replay_candidates.sh`; the launcher refuses another count).

## Files

```text
images/work/      Work image: AGENT_TASK.md, public diagnostics, base image
                  (images/work/base/: Dockerfile and pinned requirement lock)
images/judge/     public Judge image and the private holdout layer Dockerfile
judge/            Judge controller, case sharding and per-workload runners
metrics/          frame fidelity (PSNR/SSIM/LPIPS) and VBench scoring
policy/           source admission policy for submissions
runtime/          read-only codec compatibility runtime used in Work and Judge
upstream/         pinned Quant-VideoGen source and model integrations
configs/          paper protocol (the Judge-only release config comes from the assets)
locks/            model lock: pinned revisions and tree digests
portable/         task contract manifests and REBUILD_PROMPT.md (asset recipe)
scripts/          build, model verification, input staging, agent run, archive
                  evaluation and replay
tests/            CPU-only contract, isolation, payload and proxy tests
llm_judge/        integrity review policy
oracle/           trusted reference codec (host-side only, never in Work)
assets/           asset manifest, fetch.sh, public prompts and image input
```

## Upstream and license

- Quant-VideoGen (QuantVideoGen, arXiv 2602.02958), pinned at commit
  `0601468f2dbba6a17ac7086faec6d41527cad188` in `upstream/`; Apache-2.0
  (per `upstream/pyproject.toml`). Its experiment integrations (Self-Forcing,
  LongCat-Video, HY-WorldPlay, LingBot) are included with task-specific
  adapters for device binding and byte accounting
  (`upstream/INTEGRATION_NOTES.md`). Upstream repository:
  [svg-project/Quant-VideoGen](https://github.com/svg-project/Quant-VideoGen).
- Models (weights are downloaded, not redistributed): `Wan-AI/Wan2.1-T2V-1.3B`,
  `gdhe17/Self-Forcing`, `meituan-longcat/LongCat-Video`, `tencent/HY-WorldPlay`,
  `tencent/HunyuanVideo-1.5`, `Qwen/Qwen2.5-VL-7B-Instruct`, `google/byt5-small`,
  `black-forest-labs/FLUX.1-Redux-dev` (gated), ModelScope
  `AI-ModelScope/Glyph-SDXL-v2`; pinned revisions are in
  `configs/paper_protocol.json` and `locks/model-lock.json`. Licenses as
  declared on the Hugging Face model cards:

  | model | license |
  |---|---|
  | `Wan-AI/Wan2.1-T2V-1.3B`, `gdhe17/Self-Forcing`, `Qwen/Qwen2.5-VL-7B-Instruct`, `google/byt5-small` | Apache-2.0 |
  | `meituan-longcat/LongCat-Video` | MIT |
  | `tencent/HY-WorldPlay` (and its official repository images) | Tencent HY-WorldPlay Community License |
  | `tencent/HunyuanVideo-1.5` | Tencent Hunyuan Community License |
  | `black-forest-labs/FLUX.1-Redux-dev` | FLUX.1 [dev] Non-Commercial License |
  | ModelScope `AI-ModelScope/Glyph-SDXL-v2` | see the upstream repository |

  Several of these licenses are not open-source licenses and some carry
  territorial restrictions; check each before use.
- Evaluation: VBench (`Vchitect/VBench` @ `45e79ec14e69a2187202c675d2dbce1a71843d53`,
  Apache-2.0 code), DINO (`facebookresearch/dino` @
  `7c446df5b9f45747937fb0d72314eb9f7b66930a`, Apache-2.0), CLIP, LAION aesthetic
  predictor, MUSIQ (IQA-PyTorch) and LPIPS weights. Prompts come from Movie Gen
  Video Bench. For the evaluation weights and the Movie Gen Video Bench prompts,
  see their upstream sources for license terms.
- The task's own code is under the repository's root `LICENSE` (PolyForm Noncommercial 1.0.0).
