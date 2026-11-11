# EAGLE-3 Speculative Decoding Engine for Qwen3-8B

The agent optimizes a complete EAGLE-3 draft-tree speculative-decoding engine for a
frozen BF16 `Qwen/Qwen3-8B` target and a frozen EAGLE-3 draft head
(`Tengyunw/qwen3_8b_eagle3`). The goal is maximum generated tokens per second for
lossless greedy decoding on one GPU, while every emitted token stays consistent with the
target's greedy choice. This is a seed task: the workspace starts with `engine_v001`, a
working port of the upstream EAGLE engine (dynamic draft tree, target tree verification,
posterior acceptance and KV commit, tree policy 60 tokens / depth 7 / top-k 10).

## The task

The agent works in `/app/workdir/spec_lab` inside the Work container. The submitted
paths are:

```text
solutions/engine_vNNN/   candidate.py plus optional Python, CUDA, C++, headers, config
FINAL_VERSION            integer selecting the evaluated engine_vNNN
notes.md                 the agent's working notes
```

ABI (`runtime/api.py`, read-only):

- `candidate.build(target, draft, tokenizer, contract)` returns an engine. It receives
  the frozen target, the frozen draft head, a read-only tokenizer and an
  `EngineContract`. Build is untimed but capped at 600 seconds.
- `engine.generate(prompt_ids, GenerationSpec)` returns `runtime.api.GenerationResult`.
  Inputs are one to eight int64 CUDA prompt tensors.
- Each output keeps the prompt prefix bit-exact, emits at least one in-vocabulary token,
  stops at and includes the first EOS, and otherwise emits exactly `max_new_tokens`.
  Prompt plus generation never exceeds 3,968 tokens; engines must provision positions
  through 4,096. Results must be deterministic and independent of request order.

The agent may change tree policy, target and draft execution, attention, KV storage and
commit, CUDA graphs, Triton or CUDA kernels (built from shipped source with
`torch.utils.cpp_extension` into `build/` or `/tmp`), `torch.compile`, batching,
allocation, and draft-side precision (repacking or quantizing a copy of the draft head
is allowed). Target-side numerical changes are allowed only as far as the oracle below
accepts them.

The Judge rejects, by static source admission and process checks:

- files other than Python, CUDA/C++, headers, JSON, Markdown, text and TOML; files over
  1 MiB; a selected version over 16 MiB; symlinks and binaries;
- network modules, direct subprocess use, `os.system`, pickle and `torch.load`, dynamic
  eval/exec/import, frame or live-object introspection, environment-driven routing,
  `sys.modules` control, and monkey-patching of the trusted runtime;
- imports of runtime modules other than `runtime.api`, and imports from another
  `engine_vNNN` (all dependencies live in the selected version);
- reads outside the workspace, model and `/tmp` roots, and writes outside `build/` and
  `/tmp`;
- mutating the handed-over tensors or the read-only model mounts; Python helper threads
  still alive when `generate` returns.

Returning a previously computed output for a request, special-casing disclosed prompts,
or faking timing or completion is out of scope (see `llm_judge/POLICY.md`).

Public tests in the Work container (`/workspace/public_tests/run_public_tests.sh` with
`--cpu-only`, `--quick` or `--full`) run the engine on public prompts with the public
oracle policy; they are diagnostics and never produce a score. The launcher's default
agent budget is 24 hours (`TASK_TIMEOUT=86400`).

## Evaluation

The workload has six cells:

| cell | batch | max new tokens | thinking |
| --- | ---: | ---: | --- |
| `chat_bs1` | 1 | 512 | off |
| `math_bs1` | 1 | 512 | off |
| `code_bs1` | 1 | 512 | off |
| `longctx_bs1` (1.5-3k-token prompts) | 1 | 256 | off |
| `mixed_bs8` | 8 | 512 | off |
| `think_bs1` | 1 | 1,024 | on |

Judge prompts come from MT-Bench, GSM8K, HumanEval and UltraChat-200k, rendered with Qwen's
chat template. The public set that ships in the Work image is a smaller smoke workload
with its own cells, budgets and prompt sources, and shares no prompt with the Judge
(`scripts/prepare_public_prompts.py` rebuilds it from pinned dataset revisions):

| public cell | batch | max new tokens | thinking | prompts (+ warm-up) | source |
| --- | ---: | ---: | --- | ---: | --- |
| `chat_bs2` | 2 | 384 | off | 4 (+2) | UltraChat-200k `test_sft` |
| `math_bs4` | 4 | 256 | off | 8 (+4) | GSM8K `train` |
| `code_bs3` | 3 | 384 | off | 6 (+3) | MBPP sanitized `test` |
| `longctx_bs2` (0.8-1.4k-token prompts) | 2 | 128 | off | 4 (+2) | UltraChat-200k `test_sft`, concatenated |
| `think_bs2` | 2 | 768 | on | 4 (+2) | GSM8K `train` |
| `mixed_bs6` | 6 | 320 | off | 12 (+6) | chat/math/code pools above |

Public throughput uses the same formula but is not comparable to Judge throughput.

**Metric.** For each cell, `T_c = total generated tokens / total generate wall seconds`,
measured by the Judge's controller around each call with CUDA synchronized before and
after, so prefill is timed. Build, one warm-up per shape, reference work and determinism
reruns are not timed. The score is the geometric mean of `T_c` over the six cells, in
tokens/s, raw and uncapped (no baseline normalization). Raw throughput is comparable only
within one hardware type.

**Greedy-consistency oracle.** After the candidate's worker exits, the Judge loads a
fresh eager-attention BF16 target and teacher-forces the candidate's own trajectory. It
requires:

- O1: for every emitted token, `max(logits) - logits[token] <= 24.0`;
- O2 and tiers: global argmax-disagreement rates across the whole run below eight
  calibrated margin-CDF limits (the public values are in
  `runtime/policy_public.json`; the Judge uses them as a floor and relaxes them only up to
  3x the disagreement rate of the unmodified seed measured in the same evaluation).

The oracle tolerates BF16 reduction-order ties and draft-side changes; it does not prove
bit-exact equality with one reference stream or that target weights are untouched.

**Other gates.** O3/O4 (prefix, dtype, shape, ID range, EOS and budget), O5 (selected
prompts rerun in reverse order must reproduce exactly), source admission, protected-file
and process ordering, per-request wall and memory caps, device-idle at return, and
completeness. Wall and memory caps are relative to the unmodified seed, which the Judge
runs on its own device at the start of every evaluation: up to 8x the seed's
extrapolated wall and 1.10x the seed's peak memory plus 8 GiB. A submission is valid
only if every gate passes; any failure, exception or timeout scores zero.

**Submissions.** Scheduled automatic evaluation is disabled. The agent calls
`sforge-submit` to have the Judge grade the current `FINAL_VERSION` on the complete
private final suite; it returns the score and gate vector and is rate-limited by a
cooldown. At the deadline the workspace is frozen as `final_archive.tar.gz` and
evaluated once more on the same suite by the host, after Work has stopped. The run's
result is the best valid score across submissions and this final evaluation. The Judge
outer timeout is 5,400 seconds. The submission cooldown defaults to 1,800 seconds (the
task script's `EVAL_INTERVAL`); manual submissions are otherwise unlimited.

## Hardware

Two GPUs per run by default: Work runs on GPU 0 and the Judge on GPU 1 (`--work-gpu`,
`--judge-gpu`), so the Judge measures on its own GPU while Work keeps running. With
`--shared-gpu` Work and Judge share one GPU serially: before each Judge run the host pauses
the Work container and resumes it afterwards. The capability floor is compute capability 8.0 and
34 GiB of device memory; Work and Judge each get 8 CPUs and 64 GiB of host RAM.

| hardware type | status |
| --- | --- |
| `h100-94gb-400w` (H100 NVL 94 GB) | qualified |
| `h100-80gb-700w` (H100 SXM 80 GB) | qualified |

## Running

1. **Assets.** `tasks/latest/eagle3-spec-decode-engine-qwen3-8b-h100-v1-seed/assets/fetch.sh`
   fetches into `INFRATHON_ASSETS_ROOT` (default `~/.cache/infrathon/assets`) and
   verifies:
   - `Qwen/Qwen3-8B` at revision `b968826d9c46dd6066d109eabc6255188de91218` from Hugging
     Face, kept in the Hugging Face cache (about 16.4 GB);
   - the converted EAGLE-3 head (safetensors, about 0.8 GB; produced by
     `scripts/convert_head.py` from `Tengyunw/qwen3_8b_eagle3` at
     `2a1059d51f622b8cad7d7d72840153ffea5488a0`) from the Infrathon dataset;
   - the Judge-only prompt sets, commitments, reference streams and policy under
     `judge/private_data/` from the dataset's `judge/` part (these files are not in the
     repository; the build reads them from `INFRATHON_ASSETS_ROOT`).

   Both models are mounted read-only at `/opt/models/qwen3-8b-hf` and
   `/opt/models/eagle3-head` (`EAGLE3_TARGET_REPO`, `EAGLE3_HEAD_DIR` override the host
   paths).
2. **Images.**
   `INFRATHON_PULL=1 tasks/latest/eagle3-spec-decode-engine-qwen3-8b-h100-v1-seed/scripts/build_images.sh`
   pulls the published Work and public Judge images and builds the Judge holdout layer
   locally. Without `INFRATHON_PULL=1` it builds both images from source
   (`pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel`, Transformers 4.53.1).
3. **Run.**

   ```bash
   launch/eagle3-spec-decode/run.sh --model MODEL_ID [--work-gpu N --judge-gpu M] [--dry-run]
   ```

   Defaults: Work on GPU 0, Judge on GPU 1, 24 hours, effort `max`, Judge port 18386;
   `--work-gpu`, `--judge-gpu`, `--hours`, `--effort` and `--judge-port` change them.
   `--shared-gpu` runs Work and Judge on one GPU, chosen with `--gpu N` (default 0) or
   `--device-uuid UUID`; either flag alone also selects shared mode.

   `--agent codex` (default, `--auth auth.json`) or `--agent claude-code`
   (`--claude-token FILE`); `--config FILE` loads a launch config
   (`launch/mlsysforge.env.example`); `--dry-run` checks the plan without touching the
   GPU. The launcher expects a prebuilt agent runtime (`launch/build-agent-runtime.sh`)
   and starts its own restricted proxy sidecar for the model API; the Work container also reaches
   public HTTP(S) hosts through the run's egress proxy and the agent may use web search,
   while the Judge is offline. The required LLM integrity review is
   configured with the `SFORGE_LLM_JUDGE_*` keys in the launch config.

   Eval-only and replay:

   ```bash
   # evaluate one archive (final_archive.tar.gz or submission.tar.gz)
   JUDGE_GPU=N tasks/latest/eagle3-spec-decode-engine-qwen3-8b-h100-v1-seed/scripts/eval_archive.sh ARCHIVE
   # re-score saved checkpoints or every retained engine_vNNN of a finished run
   launch/eagle3-spec-decode/replay.sh --origin-run RUN_ID --measurement-set NAME [--per-version] [--json]
   ```

## Files

```text
images/work/        Work image: AGENT_TASK.md, public tests, workdir/spec_lab
                    (runtime/ trusted API and public oracle, solutions/engine_v001 seed,
                    public_prompts/ with public reference outputs)
images/judge/       public Judge (run_judge.py, verify_eagle3.py, verifier/, seed
                    reference) and Dockerfile.private for the holdout layer
host_reference/     host-only controls and reference implementations; not in any image
oracle/             reference candidates (FINAL_VERSION: EAGLE-3 in SGLang, engine_v002,
                    not admissible; ADMISSIBLE_VERSION: the seed) and
                    paper_reference/forced_judge/ (local-only Judge image and runner
                    that scores engine_v002 with source admission and the LLM review off)
portable/           portable task contract: suites, oracle, selection, resources
scripts/            build, run, eval, replay, preflight and validation scripts
tasks/              SForge task spec and benchmark metadata
tests/              CPU-only contract and hardening tests
llm_judge/          integrity-review policy
assets/             asset manifest and fetch.sh
```

## Reference solutions (oracle)

`oracle/FINAL_VERSION` selects the strongest published implementation: EAGLE-3 in
SGLang, the serving integration the EAGLE-3 authors point to. It is not admissible
under the task rules (subprocesses, sockets, environment routing), so it is scored by
a local-only Judge (`oracle/paper_reference/forced_judge/`: the private Judge image
plus an isolated SGLang virtualenv) with source admission and the LLM integrity review
turned off; every other gate is unchanged. The memory gate measures only the worker
process, not the SGLang server process, so it does not cover this candidate's full
memory use. `oracle/ADMISSIBLE_VERSION` selects the seed. Scores on an H100 NVL Judge,
private final suite (details in `oracle/notes.md`):

| oracle candidate | admissible | Judge | score (tokens/s) | gates |
| --- | --- | --- | --- | --- |
| engine_v001: seed = upstream EAGLE HF-eager port (best admissible) | yes | normal | 122.63 | all pass |
| engine_v002: EAGLE-3 in SGLang 0.5.20, the head's published tree (32 tokens / depth 6 / top-k 10), deterministic mode (`FINAL_VERSION`) | no | source admission and LLM review off | 414.16, 415.10 (two Judge runs) | all other gates pass; the memory gate sees only the worker process |

## Upstream and license

- EAGLE (EAGLE-3 engine), commit `cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b`, Apache-2.0.
  The seed ports its draft/verify/accept/commit algorithm behind the task ABI. No upstream checkout enters
  either image.
- `Qwen/Qwen3-8B` at `b968826d9c46dd6066d109eabc6255188de91218`, Apache-2.0.
- `Tengyunw/qwen3_8b_eagle3` at `2a1059d51f622b8cad7d7d72840153ffea5488a0`, MIT.
- Prompt sources: MT-Bench (Apache-2.0), GSM8K (MIT), HumanEval (MIT),
  UltraChat-200k (MIT); the public set also uses MBPP (CC-BY-4.0).

The task's own code is under the repository's root `LICENSE` (PolyForm Noncommercial 1.0.0).
