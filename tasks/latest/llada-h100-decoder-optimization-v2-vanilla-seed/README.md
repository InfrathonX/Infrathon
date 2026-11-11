# LLaDA Decoder Optimization on H100 (stratified, vanilla seed)

This task optimizes inference for LLaDA-8B-Instruct, a masked-diffusion language model that
generates text by iteratively unmasking a fixed-length token buffer. The agent receives a
correct but deliberately slow deterministic generation engine (`decoder_v001`, which
re-evaluates the full model at every step) and must raise generation throughput while keeping
answer quality, determinism, and request isolation. It is a "seed" task: the starter is a
working, unaccelerated ("vanilla") baseline to improve, and the decoding strategy is open.

## The task

The Work container's working directory is `/app/workdir/decoder_lab`. The agent creates
versioned candidates under `solutions/decoder_vNNN/candidate.py` and selects one with
`FINAL_VERSION` (a decimal integer from 1 to 999). Only `solutions/`, `FINAL_VERSION` and
`notes.md` are captured; changes to the runtime, tests or installed packages are not part of a
submission. `decoder_v001` should stay unchanged as a recovery point.

The selected module must keep this interface:

```python
def build(model, tokenizer):            # called once; model and tokenizer are read-only
    return engine

@torch.inference_mode()
def generate(self, prompt_ids, spec):   # spec: runtime.api.GenerationSpec
    return runtime.api.GenerationResult(token_ids=output, trace={})
```

`prompt_ids` is an int64 `[1, prompt_length]` tensor on the GPU; `spec` gives the generation
length, block length, temperature and mask-token ID. The output must have shape
`[1, prompt_length + gen_length]`, stay on the same device as int64, preserve the prompt bit
for bit, fill every requested slot (also after EOS) with tokens chosen from the model's own
predictions, and contain no mask tokens. Exact token equality with the starter is not
required.

Fixed runtime: checkpoint `GSAI-ML/LLaDA-8B-Instruct` at revision
`6059b30c9531767ce47e0ca6a8161a6bb0c67693` (8.0 B parameters, BF16, mounted read-only),
batch size 1, maximum sequence 4,096 tokens, PyTorch 2.7.1 / CUDA 12.8 / transformers 4.49.0.

Rules enforced by the Judge and its source admission check:

- Source is plain Python or Triton: only `candidate.py` and `__init__.py` per version
  directory; no symlinks, binaries, encoded source, data files or dynamically loaded code.
  The retained source tree is at most 2 MiB and `notes.md` at most 128 KiB.
- Allowed imports: `__future__`, `collections`, `contextlib`, `dataclasses`, `functools`,
  `itertools`, `math`, `runtime` (only `runtime.api`), `torch`, `triton`, `typing`.
- No reflection beyond the starter's fixed `getattr(obj, "mlsysforge_allow_cpu", default)`
  probe; no dynamic imports, file access, subprocesses, frame/hook access or dunder
  introspection; no replacing `model.forward` or trusted runtime objects.
- Do not train, quantize, replace, reload or mutate the weights, and do not load another
  model. Every request must actually run the supplied model.
- Repeated identical calls must give identical output. No mutable state may carry between
  requests, including key/value, hidden-state or logit caches of a shared prompt prefix
  (such as few-shot examples); such caches are rebuilt within each request.
- Do not pad, overwrite or remap generated slots, and do not suppress or replace EOS to
  change the measured output length.
- Do not hard-code visible prompts, answers, token sequences or example shapes, and do not
  try to identify evaluation inputs.

A required LLM integrity review (`llm_judge/POLICY.md`) rejects submissions that violate
the scope, for example canned output in place of real model-based generation.

The launcher's default agent budget is 24 hours (`TASK_TIMEOUT=86400`; change it with
`--hours` or `--timeout`). One Judge run scores all 520 generations and takes about 50-55
minutes on the H100 NVL, so a run with `--hours 1` leaves room for about one
round of feedback. A Judge run still in progress when the budget ends is allowed to finish
before the final snapshot, and the independent final evaluation takes about as long again.

Visible tests in the Work container: `bash /workspace/public_tests/run_public_tests.sh`
with `--cpu-only` (source policy and interface checks on a small synthetic model), `--quick`
(one target per dataset on the real checkpoint), or no flag (the full visible suite of 40
generations, ten per dataset, over train or synthetic targets, disjoint from the evaluation data);
`--candidate-version N` tests a retained version without changing `FINAL_VERSION`.

## Evaluation

The Judge runs a frozen, stratified workload of 520 generations: 130 official-test targets
from each of GSM8K, MATH, HumanEval and MBPP, spread over 12 benchmark rows
(generation lengths 256, 512 and 1024, block length 32, 0- to 8-shot prompts; 216,320
requested output slots in total). Each target appears in exactly one row. The case
selection and order are Judge-only.

For each row `r`, throughput is

```text
TP_r = sum(useful output tokens) / sum(controller-measured seconds)
```

Useful tokens are counted through EOS/stop; for HumanEval the full buffer is decoded and the
non-EOS slots are counted. Timing starts after trusted prompt preparation and CUDA
synchronization and covers the whole worker round trip (allocations, cache setup, all model
evaluations, token selection, state reset, result validation and transfer back); it excludes
checkpoint load, tokenization, warm-up, answer parsing and tests. The score is the geometric
mean over the 12 rows, in tokens per second, higher is better, with no baseline
normalization:

```text
score = exp(mean_r(log(TP_r)))
```

A submission is valid only if all hard gates pass; an invalid submission scores 0. The gates
are: output contract, determinism, cache isolation, coverage of all 32 model layers,
sampled weight integrity, source policy, case/row/overall timeouts, finite timing, the
memory cap, and the quality gate. Quality is measured per case (answer correct, answer
parseable, useful output length) relative to a frozen reference decoder run on the same GPU
model. The quality gate fails when:

- the minimum row or aggregate retention falls below a threshold derived from precommitted
  positive and degraded-negative control submissions;
- any of the aggregate retentions (correct, parsed, useful tokens) over all 520 generations
  is below 90%;
- any benchmark's pooled 130-generation retention is below 75%;
- any useful-token retention exceeds 150% (this blocks inflating throughput by moving EOS).

The memory cap is device-relative: `floor(total device memory in MiB) - 7,168 MiB`, measured
on the GPU the Judge runs on.

Interactive submissions (`sforge-submit --details` during the work session) run the same
complete 520-generation evaluation, gates and deadlines as the final evaluation and return
only a redacted aggregate result (validity and score, no per-example details). There is a
300-second cooldown between submissions and no cap on their number; the run does not
auto-evaluate. After the session ends, the captured state is evaluated once more as the
final. The run's result is the best valid score across submissions and this final
evaluation. Per-row deadlines are 420 s per case and 900 s per row, with a 3,600 s inner and
3,900 s outer limit for a whole evaluation.

Scores are comparable only within one measurement profile (GPU model, compute capability,
driver, image ids).

## Hardware

By default a run uses two GPUs: Work runs on GPU 0 and the Judge on GPU 1 (`--work-gpu`,
`--judge-gpu`), so the Judge measures on its own GPU while Work keeps running. With
`--shared-gpu` Work and Judge share one whole GPU serially, and before each Judge run a
root-owned `.sforge/prepare_submit.sh` hook checks for remaining GPU holders and the host
pauses the Work container.

| hardware type | support |
|---|---|
| `h100-94gb-400w` (H100 NVL 94 GB) | qualified |
| `h100-80gb-700w` (H100 SXM 80 GB) | qualified |

The Judge carries a quality reference for each GPU model and selects the one for the GPU it
observes; a GPU model without a reference is refused rather than compared against another
device's numbers. The capability envelope is compute capability 8.0 or newer and at least
34 GiB (34,816 MiB) of device memory.

## Running

1. Assets: `tasks/latest/llada-h100-decoder-optimization-v2-vanilla-seed/assets/fetch.sh`
   (files go under `INFRATHON_ASSETS_ROOT`). It downloads `GSAI-ML/LLaDA-8B-Instruct` at the
   pinned revision (about 16 GB) into the Hugging Face cache and the Judge-only data
   (hidden cases, per-GPU quality references, and the Judge's copies of the frozen
   workload/memory/quality policies that are also in the task root) from the dataset's
   `judge/` part; the hidden cases and references are not in the repository, and the build
   reads all of them from `INFRATHON_ASSETS_ROOT`. Point `LLADA_MODEL_REPO` at the model's cache repository
   folder (`models--GSAI-ML--LLaDA-8B-Instruct`; `scripts/infrathon/assets.py hub-cache` prints
   the cache root). It is mounted read-only into both containers.
2. Images:
   `INFRATHON_PULL=1 tasks/latest/llada-h100-decoder-optimization-v2-vanilla-seed/scripts/build_images.sh`
   pulls the published Work and public Judge images and builds the Judge holdout layer
   locally; without `INFRATHON_PULL=1` it builds everything from source.
3. Quality-control archives: the launcher checks the precommitted reference and control
   submissions in `quality_controls.json` before a formal run. They are not shipped (neither
   in the repository nor in the dataset); regenerate them deterministically (a few seconds, no
   download) into the task's `artifacts/`. `--stress-extra-mib 61440` is the size of the extra
   GPU buffer in the near-cap memory-stress archive; it must stay 61440 to match the committed
   archive hashes:

   ```bash
   python3 tasks/latest/llada-h100-decoder-optimization-v2-vanilla-seed/scripts/make_reference_archives.py \
     --task-root tasks/latest/llada-h100-decoder-optimization-v2-vanilla-seed --stress-extra-mib 61440
   ```

   Without them the dry run stops with "quality control archive path is unsafe or missing".
   The layout check then prints two `stale-calibration signal` warnings. They are expected:
   the quality references were calibrated with earlier copies of the SForge runtime and some
   task scripts, so the check cannot reproduce them byte for byte from the current tree. The
   Judge re-checks everything it scores with, so the warnings do not affect scoring; they become
   errors only with `LLADA_STRICT_REFERENCE=1`.
4. Run:

   ```bash
   LLADA_MODEL_REPO=/path/to/models--GSAI-ML--LLaDA-8B-Instruct \
   launch/llada-h100/run.sh --model <model> --auth /path/to/auth.json \
     [--work-gpu N --judge-gpu M]
   ```

   Defaults: Work on GPU 0, Judge on GPU 1, 24 hours, effort `max`, Judge port 18384;
   `--work-gpu`, `--judge-gpu`, `--hours`, `--effort` and `--judge-port` change them.
   `--shared-gpu` runs Work and Judge on one GPU, chosen with `--gpu N` (default 0) or
   `--device-uuid UUID`; either flag alone also selects shared mode.
   `--agent claude-code --claude-token FILE` runs Claude Code instead of Codex. `--dry-run`
   resolves and checks the plan (model, agent runtime, images, integrity review, GPU, port)
   without starting anything; `--tmux` runs detached. Options can also come from
   `launch/mlsysforge.env` (see `launch/mlsysforge.env.example`).

   To evaluate one stored submission archive without an agent:
   `JUDGE_GPU=<idx> tasks/latest/llada-h100-decoder-optimization-v2-vanilla-seed/scripts/eval_archive.sh <archive.tar.gz> [run-id]`.
   Batch replay is not provided; use `eval_archive.sh` per archive.

The launcher checks for the agent runtime and credentials (a mode-0600 Codex `auth.json`
or a Claude token file), the model repository, current images, the required LLM integrity
review settings (`SFORGE_LLM_JUDGE_*`), and that the GPUs are free (UUID lock, no other
processes or containers on them).

## Files

```text
images/work/       agent instructions (AGENT_TASK.md), seed workspace with decoder_v001,
                   trusted runtime, visible tests and public cases
images/judge/      Judge Dockerfiles, run_judge.py, verifier (controller, worker, quality,
                   source admission, memory sampling); Judge-only private_data/ comes from the assets
judge/             Judge adapter source
tasks/             SForge task definition (judge command, timeouts, submit paths)
portable/          portable task contract and suite manifests
llm_judge/         integrity review policy
host_reference/    host-only calibration and control sources
scripts/           image build, evaluation, qualification and validation tools
assets/            asset manifest and fetch script
tests/             task-local contract tests
*_policy.json, quality_controls.json   frozen workload, memory and quality-control policies
```

## Upstream and license

- Model: `GSAI-ML/LLaDA-8B-Instruct` at `6059b30c9531767ce47e0ca6a8161a6bb0c67693`.
  MIT (per its model card).
- Workload rows follow the configurations of [NVlabs/Fast-dLLM](https://github.com/NVlabs/Fast-dLLM)
  at `a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b` (Apache-2.0); prompts follow
  [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
  v0.4.8 (commit `6d2abda4fd171e68a8789330c4149e37c1ca0bda`, MIT).
- Datasets (Hugging Face, pinned revisions):

  | dataset | revision | license |
  |---|---|---|
  | `openai/gsm8k` (`main`) | `e53f048856ff4f594e959d75785d2c2d37b678ee` | MIT |
  | `EleutherAI/hendrycks_math` | `21a5633873b6a120296cce3e2df9d5550074f4a3` | MIT |
  | `openai/openai_humaneval` | `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544` | MIT |
  | `google-research-datasets/mbpp` (`full`) | `4bb6404fdc6cacfda99d4ac4205087b89d32030c` | CC-BY-4.0 |
- The task's own code is under the repository's root `LICENSE` (PolyForm Noncommercial 1.0.0).
