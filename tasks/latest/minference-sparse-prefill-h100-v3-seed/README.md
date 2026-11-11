# MInference Sparse BF16 Prefill Kernel on H100

This task is a single-GPU kernel benchmark for long-context sparse prefill attention in the
style of Microsoft MInference (vertical/slash sparse patterns). The agent optimizes one
five-argument BF16 causal attention call with native grouped-query attention (32 query heads,
8 KV heads, head dimension 128) on real Llama-3.1-8B-Instruct activations, and must stay inside
a fixed numerical error budget against an FP32 dense reference. It is a "seed" task: the Work
container starts with a working, conservative starter kernel (`kernel_v001`) adapted from
MInference, which the agent improves.

## The task

The agent works in `/app/workdir/minference_prefill_lab` inside the Work container and submits
only these paths:

- `candidates/kernel_vNNN/submission.py`: versioned candidate implementations
  (`kernel_v001` is the starter);
- `FINAL_VERSION`: the number of the candidate to use;
- `notes.md`: free-form experiment notes.

Each candidate must keep the ABI

```
minference_sparse_prefill_bf16(q, k, v, softmax_scale, out) -> None
```

for BF16 native GQA with Q32/KV8, D=128 and any real sequence length N >= 64 (including lengths
that are not a multiple of the tile size). It must leave `q`, `k`, `v` unchanged and fully write
`out` on the caller's CUDA stream. The sparsity pattern is the candidate's choice.

Rules (enforced by static source admission, an LLM source-integrity review and GPU-side checks
in the Judge):

- All input-dependent work (pattern estimation, index conversion, attention, output) happens
  inside every call. Only compiled code and shape-only empty scratch may persist across calls;
  outputs or patterns from the warm-up call or earlier calls must not be reused.
- Not allowed: existing vendor attention kernels, dynamic or native library loading, file or
  network I/O, timing or stream manipulation, `torch.compile`, inspecting the Judge context,
  private data, nonces or hidden GPU memory, and hard-coding case identities.
- Allowed: Torch/Triton numerical code, a self-written Triton dense fallback, and fixed
  input-independent sparse masks, as long as the numerical gates pass.
- Protected runtime, benchmark and controller files must not be read or altered.

Inside the Work container, `tools/run_public_tests.sh --candidate-version N` runs the public
suite (18 cases on a different model and prompts from the Judge), `--quick` a small subset, and
`--boundaries-only` only the small-N boundary cases without loading the model.
`tools/check_source.sh --candidate-version N` runs CPU-side source admission.
`sforge-submit` sends the current candidates to the Judge.

The local launcher gives the agent a 12-hour budget by default (`--hours` / `--timeout`).
Submissions have a cooldown of 300 s between them (`EVAL_INTERVAL`).

## Evaluation

Kernel inputs are real Q/K/V activations captured from Llama-3.1-8B-Instruct (FlashAttention-2
capture backend) on RULER-style prompts. The suite is a 48-group matrix: four RULER task types
(`niah_single_2`, `niah_multikey_1`, `vt`, `fwe`) x three context budgets (32K, 64K, 128K) x four
layers (1, 4, 15, 27). Each group has three trials, each on a different real prompt with its real
length N (2048 <= N < context budget). Before each trial an unscored warm-up call runs on layer 2
of the same shape. Q/K/V for one prompt at a time are generated in the Judge, integrity-checked and
deleted again (at most 8 GiB scratch); this preparation is not timed.

Gates (all must pass, otherwise the submission is invalid with score 0):

- **Fidelity.** For each call the output is compared with an FP32 exact causal native-GQA
  reference on sampled query rows, separately for 512 uniformly sampled rows and the 64 tail
  rows. The error is the energy-normalized
  `NRMSE = sqrt(sum((out - ref)^2) / max(sum(ref^2), 1e-12))`. Limits: global NRMSE <= 0.25 and
  worst-head NRMSE <= 0.50, for both row samples. The limits are in
  `qualification/numerical_policy.json` (in Work, the same limits are in
  `/workspace/benchmark/public_kernel.py`). They are an engineering budget for approximate attention defined by this task, not an
  upstream threshold and not a measure of end-to-end answer quality.
- **Boundaries.** Eleven small lengths (64, 65, 511, 1889, 2047, 2048, 2049, 4095, 4096, 4097,
  8193) are checked on the full output; they are not timed.
- **Integrity.** Output fully written and finite, inputs unchanged; static source admission and
  the LLM integrity review (`llm_judge/POLICY.md`) pass.

**Score** (maximize, unit: dense-equivalent TFLOPS): for each trial,
`2*32*128*N*(N+1) / max(host wall time, CUDA event time) / 1e12`; the median of the three trials per
group; the equal-weight geometric mean over the 48 groups. The timed region covers pattern
estimation, index conversion, attention, any dense repair, the output write and synchronization.
It is a kernel throughput score, not an end-to-end model speedup.

Interactive submissions (`sforge-submit`) and the final evaluation run the same private 48-group
suite on one H100. The public suite in Work uses a different model, tasks, lengths and layers. After the
agent's budget ends, the final Judge evaluates the candidate named by `FINAL_VERSION`. Among a
run's results, a valid result ranks above an invalid one, then by score. One Judge run takes about
half an hour, mostly model loading and Q/K/V preparation; the Judge timeout is 7200 s.

## Hardware

Each run uses two GPUs on one host: one for Work and one for the Judge (every submission and the
final). Work and Judge each get 8 vCPUs and 48 GiB of host memory. The task's execution profile
requires, per container, one NVIDIA GPU with at least 80 GB of memory and compute capability 9.0.

| hardware type | status |
|---|---|
| `h100-94gb-400w` (H100 NVL 94 GB) | qualified |
| `h100-80gb-700w` (H100 SXM 80 GB) | qualified |

## Running

1. **Assets.** `tasks/latest/minference-sparse-prefill-h100-v3-seed/assets/fetch.sh` downloads
   into `INFRATHON_ASSETS_ROOT`:
   - `gradientai/Llama-3-8B-Instruct-262k` at a pinned revision from Hugging Face (about 15 GiB;
     used by Work);
   - `NousResearch/Meta-Llama-3.1-8B-Instruct` at a pinned revision from Hugging Face (about
     15 GiB; used by the Judge);
   - the public prompt set (Work) and the final prompt set (Judge only) from the Infrathon
     dataset (the final prompts come from its `judge/` part and are not in the repository).
     `scripts/generate_public_prompts.py` regenerates the public set and its suite settings.

   The models and prompt sets are not baked into any image. The launcher mounts them read-only at
   run time: the public model and prompts into Work, the Llama-3.1 model and the final prompts into
   the Judge. The Judge refuses prompt files with more than one hard link, so the launcher
   mounts run-scoped private copies of the prompt sets.
2. **Images.** This task declares `images/release.json`, so pull with

   ```bash
   INFRATHON_ASSETS_ROOT=<asset root> scripts/infrathon/pull_images.sh minference-sparse-prefill-h100-v3-seed
   ```

   which pulls the published Work and Judge images for the host's hardware type and writes the
   build record the launcher checks.

   To build from source (paths relative to the task directory):
   1. Rebuild the software parent image from the official PyTorch base and nine pinned packages
      (needs network access to PyPI and the flash-attention GitHub release):
      `scripts/build_software_parent.sh [TAG]`. It verifies the parent against
      `references/software-parent/` and prints two variables, `ON_DEMAND_SOFTWARE_IMAGE` and
      `SOFTWARE_PARENT_VERIFICATION`; export both.
   2. Point the build at the frozen plan and asset contract in `qualification/`:

      ```bash
      export FINAL_PERFORMANCE_PLAN="$PWD/qualification/final-performance-plan.json"     # absolute paths; run from the task directory
      export JUDGE_ASSET_CONTRACT="$PWD/qualification/external-assets-contract.json"
      export BUILD_ROOT=<empty build directory>   # default: build/kernel-only-thin-contexts
      ```
   3. Run `scripts/build_images.sh` (`--validate-only` stops after staging;
      `--work-only` / `--judge-only` build one image).

   An image built on a rebuilt parent is a new image identity, not the published one.
3. **Run.**

   ```bash
   # agent run: Work on GPU 0, Judge on GPU 1, 12-hour budget, effort max (the defaults)
   launch/minference-v3/run.sh --model MODEL [--work-gpu 0 --judge-gpu 1] --tmux

   # evaluation only: one archive through the final Judge path, no agent
   launch/minference-v3/run.sh --judge-gpu 1 --eval-archive FILE [--skip-integrity-review]
   ```

   `--work-gpu`, `--judge-gpu`, `--hours`, `--effort` and `--judge-port` (default 18497)
   change the defaults. `--dry-run` resolves and checks images, assets, GPUs and the Judge port
   without starting anything. `--skip-integrity-review` is only for reference or replay
   archives. Results go to `outputs/<task>/runs/<run>/completion.json`.

Agent runs need a Codex agent runtime (`launch/build-agent-runtime.sh`), a version store
(`SFORGE_VERSION_STORE_ROOT`, see `launch/mlsysforge.env.example`), model credentials
(`--auth` for a Codex `auth.json`, or a `MODEL_API_BASE_URL` endpoint), and the LLM integrity
review configured (`SFORGE_LLM_JUDGE_MODE=enforce` and the `SFORGE_LLM_JUDGE_*` settings; see
`launch/mlsysforge.env.example`). The Work container has only an internal Docker network and
reaches the outside through a restricted CONNECT proxy (`scripts/restricted_proxy.py`).

## Files

- `images/work/`: Work image: the agent workspace (`workdir/minference_prefill_lab`, starter
  candidate, public test tools), `AGENT_TASK.md`, the public benchmark runner and submit commands.
- `images/judge/`: Judge image: candidate worker, source admission, the verifier that measures
  fidelity and throughput.
- `images/parent/`: Dockerfile of the shared software parent image.
- `runtime/`: trusted runtime shared by Work and Judge (single-prompt Q/K/V preparation,
  fidelity checks, timing).
- `judge/`: Judge entry that turns verifier output into the structured result.
- `llm_judge/POLICY.md`: the source-integrity review policy.
- `activations/recipe/`: how the RULER prompts and model activations are produced.
- `qualification/`: numerical policy and the public/external asset contracts.
- `portable/`: portable task, suite and protocol manifests (public, interactive, final).
- `references/`: pinned upstream MInference files, the FlashAttention-3 dense baseline build,
  software-parent provenance.
- `oracle/`: reference solutions, never shown to agents. `kernel_v001` is the seed (byte-identical to
  the Work starter, about 197 TFLOPS on an H100 NVL Judge). `FINAL_VERSION` selects `kernel_v002`,
  the strongest MInference 1.0 paper solution that passes the Judge: upstream config plus the seed's
  per-call error probe, which recomputes densely any head whose sampled NRMSE in a 16-row band
  exceeds 0.30 (a repair trigger protecting the 0.50 worst-head limit; the 0.25 global gate still
  applies). Valid at 364.04 and 363.54 TFLOPS in two H100 NVL Judge runs.
- `paper_reference/`: the measurement record for the paper candidates (provenance, run ids, and the
  strict upstream port, which fails the fidelity gate).
- `scripts/`: image builds, the local launcher side (`run_local_v3.sh`,
  `sforge_local_entry_v3.py`), evaluation of archives, validators.
- `tasks/`: the SForge task spec (`minference_sparse_prefill_h100_seed_v3.json`).
- `assets/`: `manifest.json` and `fetch.sh`.

## Upstream and license

- **MInference** (Microsoft), commit `a4eb395f949ea39e871f9bc586d683390692c6be`, MIT License.
  The starter's last-64 vertical/slash pattern estimator, index converter and sparse kernel are
  adapted from it (budgets scaled to 768 vertical / 3072 slash, native GQA, non-tile-aligned N,
  plus a per-call error probe with a self-written Triton dense head repair). The RULER prompt
  generator vendored in that repository produces the task's prompts. Pinned upstream files are
  listed in `references/upstream/manifest.json`. The 48-group matrix, error budget and score are
  defined by this task, not by upstream.
- **Llama-3.1-8B-Instruct** weights (`NousResearch/Meta-Llama-3.1-8B-Instruct`, revision
  `d10aef7999a2b5ba950ab3974312feeedbfe0b77`): a mirror of Meta's Llama 3.1 8B Instruct with
  byte-identical weights, under the Llama 3.1 Community License. The official `meta-llama`
  repository is gated; users must comply with Meta's license.
- **FlashAttention-3** (Dao-AILab/flash-attention, commit
  `060c9188beec3a8b62b33a3bfa6d5d2d44975fab`, tag v2.8.3; CUTLASS
  `dc4817921edda44a549197ff3a9dcf5df0636e7b`): a dense baseline for author measurements only; it
  is not in the Work or Judge images. FlashAttention and CUTLASS are BSD-3-Clause at these
  commits.
- The RULER prompt generator comes from the copy vendored in the MInference repository; see the
  upstream MInference and RULER repositories for its license.
- The task's own code is under the repository's root `LICENSE` (PolyForm Noncommercial 1.0.0).
