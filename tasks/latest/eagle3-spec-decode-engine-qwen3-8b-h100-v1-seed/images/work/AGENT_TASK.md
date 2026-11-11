# Optimize the EAGLE-3 speculative-decoding engine

Maximize generated tokens/s on one CUDA device for lossless greedy decoding with frozen Qwen3-8B BF16 target weights and the frozen Tengyunw EAGLE-3 head. The complete `solutions/engine_v001` seed faithfully implements upstream's dynamic draft tree, target tree verification, posterior acceptance, and KV commit at policy 60/7/10. Submit `solutions/engine_vNNN/`, select it in `FINAL_VERSION`, and keep `notes.md` current.

## Contract and score

`candidate.py` exports `build(target, draft, tokenizer, contract)`. The result is an engine whose `generate(prompt_ids, GenerationSpec)` returns `runtime.api.GenerationResult`. Inputs are one to eight CUDA int64 prompt tensors. Preserve every prompt; emit at least one token; stop at and include the first contract EOS, otherwise emit exactly the requested budget. Every emitted token must satisfy the private teacher-forced greedy oracle.

The Judge scores several cells covering chat, math, code, long context, a mixed batch and thinking-on math, with batch sizes from 1 to 8 and max-new budgets up to 1024 tokens. Timed generation includes prefill. Cell throughput is total generated tokens / total synchronized wall time; the uncapped score is their geometric mean. Build and warm-up are untimed. The Judge separately enforces O1 hard logit margin, O2 aggregate non-argmax rate, the frozen eight-level margin-CDF, O3 termination, O4 shape/prefix/range, and O5 determinism/isolation, plus wall, memory, and device-idle caps. The wall and memory caps are not frozen numbers from one GPU: the Judge runs the unmodified seed engine on its own device at the start of every evaluation and admits up to 8x the seed's extrapolated full-length batch wall and 1.10x the seed's peak plus 8 GiB. The tiers have global-run scope; the public policy exposes the exact calibrated values, which the Judge uses as a floor and relaxes only to 3x the same-run seed disagreement rate.

## Allowed source surface

Python and sibling `.py`, `.cu`, `.cuh`, `.cpp`, `.h`, `.json`, `.md`, `.txt`, and `.toml` files are allowed, at most 1 MiB each. Torch, Triton, `torch.compile`, CUDA graphs/streams/events, and `torch.utils.cpp_extension` builds from shipped source into `build/` or `/tmp/torch_extensions` are allowed. Network modules, direct subprocess use, pickle `torch.load`, `os.system`, reads outside the workspace/model/`/tmp` roots, writes outside `build/` or `/tmp`, environment-driven run routing, `sys.modules` or `runtime.*` monkey-patching, frame introspection, and non-source binaries are rejected. `getattr` on `torch`/`triton` namespaces with a constant name is allowed for feature detection (`getattr(torch.nn.attention, "sdpa_kernel", None)`); a non-constant name, or a name routing into a forbidden call such as `torch.load`, is still rejected. File paths are checked by their static prefix: the leading constant of an f-string/`%`/`format` template, constant or `__file__`-derived `os.path.join`/`Path` segments (`Path(__file__).parent / name`, `os.path.dirname(__file__)`) are resolved and admitted when that prefix lies inside an allowed root for the mode; a path that is only a bare variable is rejected. Files you write under `/tmp` or `build/` may be read back. Join candidate Python helper threads before `generate` returns. The Judge joins and charges all queued CUDA work to that same request.

The target and head mounts are read-only. Repacking or quantizing copies of the draft is allowed. Target numerical changes must still pass the oracle. Do not mutate handed-over tensors.

## Public validation

```bash
/workspace/public_tests/run_public_tests.sh --cpu-only
/workspace/public_tests/run_public_tests.sh --quick --candidate-version 1
/workspace/public_tests/run_public_tests.sh --full --candidate-version 1
```

The public suite is a smoke workload, not a copy of the Judge workload. Its six cells are `chat_bs2`, `math_bs4`, `code_bs3`, `longctx_bs2`, `think_bs2` and `mixed_bs6` with max-new budgets 384, 256, 384, 128, 768 and 320 (38 prompts in total, built from UltraChat-200k test_sft, GSM8K train and MBPP). Batch sizes, budgets and prompts all differ from the Judge's and no prompt is shared with the Judge, so public tok/s is not comparable to Judge tok/s.

Before the first score-producing run after a change, use `sforge-save-version --label <hypothesis>`.

## Getting a score

The public suite never produces a score. It runs your engine against prompts you can read, with public threshold values, and its geometric mean is a development signal only.

```bash
sforge-submit          # grade one round on the complete frozen private final suite
```

`sforge-submit` is the only way to see how the Judge scores you before the deadline. It evaluates the complete frozen final prompt set with the private policy, returns the accepted score and the gate vector, and is rate-limited by a cooldown, so submit your first real candidate early and keep submitting as you improve rather than saving it all for the end. Scheduled automatic evaluation is disabled: if you never call `sforge-submit`, the run records zero graded rounds.

At the deadline the workspace freezes as `final_archive.tar.gz` and is evaluated once again against the same complete frozen final prompt set. Accepted interactive submissions are retained as `submission.tar.gz`. Your final result is the best valid score you reach within the time budget, so keep `FINAL_VERSION` on your strongest candidate.

## Recommended workflow

1. Read `runtime/api.py`, seed `candidate.py`, and `config.py`; trace one
   draft/verify/accept/commit cycle before changing it.
2. Run CPU-only contract checks. Copy the seed to the next unused three-digit
   version directory; never edit a measured version in place.
3. State one hypothesis in `notes.md` and save the version before its first
   score-producing run.
4. Use `--quick` for iteration. Inspect per-cell throughput, acceptance length,
   cycles, prefill, peak allocation, oracle tail, and exact-stream evidence.
5. Run `--full` before interactive submission. Re-run prior best and new
   candidate when a change is near documented timing noise.
6. Select only an already saved version in `FINAL_VERSION`.

Do not tune exclusively to disclosed public prompts or reference streams.
Anchor and final data are disjoint, and the private oracle follows the
candidate’s own trajectory instead of requiring one stored stream.

## Important implementation invariants

- Target feature capture is the input to layers 2, 18, and 33 in that order.
- Tree position IDs include prior sequence length; clear the temporary tree
  mask immediately after target verification.
- Retrieve indices and sorted tree tokens remain synchronized.
- Target KV commit copies accepted path rows into contiguous sequence slots for
  every layer before advancing cache length.
- The last accepted target token and matching fused hidden state feed the next
  draft cycle.
- Contract truncation handles accepted paths crossing EOS or generation budget.
- Persistent graph/cache buffers reset correctly for different prompt lengths
  and request order.
- Batch 8 is one call, not permission to change per-sequence greedy semantics.

## Request reuse and trusted runtime boundary

All scored prompts are distinct by rendered token sequence, across all cells
and mixed-batch groups, and disjoint from public, other private slices and
warm-up content. Persistent buffers, prefix/KV caches, compiled artifacts and
CUDA graphs are legitimate reuse. Returning a previously completed output for
an identical request is not a measured optimisation. O5 deliberately repeats
requests outside the score window to check equality; it does not establish
absence of output caches.

Candidate runtime imports are limited to `runtime.api`, the documented ABI.
Imports of internal runtime modules are rejected, even without mutation.
Imports from another implementation version are rejected: all submitted
implementation dependencies must live in the selected version. Dtype selection
with `getattr(torch, 'bfloat16')` and writes beneath `/tmp` or
`/app/workdir/spec_lab/build` are allowed. Reads/writes are checked after path
normalisation, including Path receivers, aliases and traversal. Pickle loading,
GC enumeration of live objects, and clock/runtime mutation are forbidden.

The private controller keeps future requests in its own process and sends one
batch only after build has finished and the previous batch has completed.
Its own wall clock measures generation through validation, output
materialisation, all-stream CUDA synchronisation, serialization and response
transfer. Candidate clock changes cannot shorten that interval. Queued work is
joined and charged to the same call; current-stream pending status and join
latency are recorded before/around the join. The current-stream query does not
claim to enumerate all streams. Live candidate Python threads fail the idle
gate. Build, warmup, oracle and O5 work remain outside the scored interval.

## Submitting on the single-accelerator profile

By default Work and the Judge run on separate GPUs. In the single-accelerator
profile they share one GPU: `sforge-submit` automatically invokes the
root-owned `.sforge/prepare_submit.sh` immediately before archive/HTTP
admission and refuses submission if that hook fails. Wait for every public
test and compiler process to finish first. You may run the hook manually to
diagnose remaining holders. The host then pauses this Work container while the
Judge runs and resumes it afterwards. In either layout the formal runner
disables in-run automatic hidden evaluation: only your manual
`sforge-submit --details` submissions run Judge while you work, and the
timeout-boundary archive is judged independently after Work ends.
