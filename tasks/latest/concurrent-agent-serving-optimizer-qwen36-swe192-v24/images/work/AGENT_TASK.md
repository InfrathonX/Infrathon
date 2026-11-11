# Optimize Concurrent Coding-Agent Inference — v2.4

Work in `/app` and modify `/app/submission` in place.

Implement a transparent online scheduling gateway that improves the aggregate
rate at which many concurrent, multi-turn coding agents produce valid
executable actions. The starter is a protocol-correct, program-aware admission
gateway; `ARCHITECTURE.md` describes its design.
Treat the starter as a point to improve on or replace, not as the known
optimum. Direct, the reference Judge compares against, opens every request
immediately and has no cross-workflow monitoring, admission, pause, or
restoration policy.

## Architectural objective

This is a workflow-level serving-systems problem, not merely an HTTP
rate-limiting problem.

A fixed global concurrency semaphore, FIFO queue, or body-size SJF policy is a
useful control baseline, but should not be treated as the intended final
architecture unless strong evidence shows that it models the multi-turn
workflow lifecycle and hybrid-cache pressure adequately.

A single
sub-percent improvement should be treated as baseline evidence, not as
completion.

Build a self-contained online monitoring and control plane inside the Candidate
Gateway. It must infer workload state only from traffic it legitimately proxies
and the lifecycle notification described below. Judge does not provide live KV,
GPU, queue, workload, or model-server telemetry to your process.

Do not hard-code the workflow count, the concurrency, or the warm-up and
scored-window timing into scheduling decisions. Capacity limits may come from
`/app/RUNTIME_CONTRACT.json`; workload size and timing must be inferred from
observed traffic, so the policy stays correct if the workload size or the
window changes.

## Deliverables

Only `/app/submission` is submitted. Keep this executable entry point:

```text
/app/submission/launch.sh
```

Judge launches it as:

```bash
BACKEND_URL=<Judge-selected OpenAI-compatible /v1 endpoint> \
PORT=9000 \
/app/submission/launch.sh
```

`launch.sh` must stay in the foreground and listen on `0.0.0.0:${PORT}`.
Always use the supplied `BACKEND_URL`; never hard-code an ordinary backend
address. Keep runtime state in memory or under `/tmp`, not in submitted source.

The gateway must implement:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`

Both streaming and non-streaming chat completions are exercised.

Maintain `/app/submission/ARCHITECTURE.md` as a concise architecture and
experiment ledger. It is part of the required engineering workflow but does
not replace a working implementation and is not rewarded for prose complexity.

## Read the run contracts first

Before designing or evaluating a candidate, read completely:

```text
/app/RUNTIME_CONTRACT.json
/app/RUN_BUDGET.json
```

`RUNTIME_CONTRACT.json` contains fixed public facts about this model-server
boot: model shape, vLLM limits, and conservative hybrid-cache page geometry. It
contains no live telemetry, private workload identity, reference policy, or
hidden order.

`RUN_BUDGET.json` is overwritten by the formal runner before your session. It
contains the actual agent timeout, hidden-submission cooldown, and approximate
hidden-evaluation duration for this run. Do not assume a fixed budget; use the
injected value.

## Scoring contract

Correctness and transparency are binary gates. If any required gate fails, the
submission is invalid and scores zero. Besides the protocol and one-to-one
backend audit checks, Judge applies these hard gates, where Direct means the
Direct evidence selected for this run:

- `paired_request_success`: request success rate at least
  max(0.98, Direct's rate - 0.02);
- `paired_valid_step_ratio`: valid-step ratio at least
  max(0.90, Direct's ratio - 0.06);
- `candidate_p99_gateway_wait`: p99 gateway wait at most 0.4 times the scored
  window (720 s for the 1800-second hidden window);
- `paired_backend_work`: backend tokens per valid step at least 0.90 times
  Direct's.

For a valid submission, the only primary
score is `valid_steps_per_min`: aggregate valid executable coding-agent steps
per minute. Higher is better.

GPU utilization, raw requests per second, progress by only a favored subset of
workflows, and code or architecture complexity are not the objective.

Judge runs a private fixed-window Candidate workload against the same fixed
model-serving system for every submission. The hidden workload has 192 distinct
SWE-bench Lite workflows, concurrency 192, a concealed committed order, 300
seconds of warmup, and an exact 1800-second scored window. Candidate is compared
diagnostically with the Direct evidence selected for this run under the same
stable serving contract and committed hidden workload. Its observed rate is
reported by Judge; do not assume a cross-run number or model-server boot. This
is candidate-only screening, not a fresh paired A/B rerun.

A score below Direct can still be protocol-valid. Preserve valid checkpoints
and distinguish correctness from performance.

## Fixed serving system

The backend is one fixed OpenAI-compatible vLLM 0.24.0 server hosting
`Qwen/Qwen3.6-27B`, tensor-parallel across two CUDA devices with compute
capability 8.0 or newer. Model weights use BF16
and KV storage uses FP8. Prefix caching and chunked prefill are enabled. vLLM
admits at most 192 sequences and has an 8,192-token batched-token ceiling per
engine iteration.

The dense 27B text backbone has 64 layers and repeats this four-layer pattern
16 times:

```text
3 x (Gated DeltaNet linear attention -> FFN)
1 x (gated full attention -> FFN)
```

The 16 full-attention layers retain token-growing context state. The 48 Gated
DeltaNet layers carry recurrent/convolution state with a material fixed cost
per live program. The public runtime contract expresses the conservative page
geometry used to reason about these two costs. That geometry is a measurement
of this model-server boot, not a universal constant — the contract marks it
`measured_on_profile` and the launcher checks it against the live backend
before your session — so plan against the values in the contract you were
given rather than any number quoted elsewhere. A single logical KV-token
total, raw request bytes, or current backend request count alone is not a
complete cross-program admission model.

Only the gateway is in scope. You cannot change the model server, weights,
runner, agents, sandboxes, GPU configuration, or runtime contract.

## Multi-turn program lifecycle

The workload contains many distinct coding tasks. Every agent alternates
between model inference and variable-duration tool work, then returns
asynchronously with a longer conversation.

Requests sharing one opaque `X-Agent-Run-ID` belong to one long-lived agent
program. Treat the identifier only as an equality key; never interpret it as
task metadata or assume a particular arrival order.

Reason explicitly about observable transitions such as:

```text
NEW
  -> REASONING / request pending or in flight
  -> ACTING / tool execution between model calls
  -> PAUSED / temporarily held by the gateway under resource pressure
  -> REASONING
  -> ...
  -> RELEASED
```

You may use a different internal state machine, but it must handle request
offer, queueing, admission, response completion, cancellation, disconnect, and
release consistently. Backend requests currently in flight are not necessarily
the same thing as the broader live workflow/cache working set.

The runner may send a lifecycle notification after a workflow's final model
call:

```http
POST /programs/release
Content-Type: application/json
X-Agent-Run-ID: <opaque workflow ID>

{"program_id":"<same opaque workflow ID>"}
```

Use it to discard bounded per-program state when useful. It is not a model
request and must not be forwarded to `BACKEND_URL`. A 2xx response acknowledges
it; returning 404 or 405 is also allowed.

## Candidate-owned online monitoring

The Gateway may maintain bounded in-memory state derived from:

- request arrival, queue, admission, upstream-open, first-response, and
  completion times;
- request method/path, raw size, and opaque end-to-end headers;
- read-only parsing of request JSON, including accumulated message/context
  size, streaming mode, and already-declared token limits;
- request or step ordinal observed for the same opaque workflow key;
- response status, chunk cadence, complete-stream duration, and usage fields
  already present in the matching response;
- cancellation, timeout, disconnect, and `/programs/release` events.

Monitoring and scheduling state must have explicit memory and lifecycle bounds.
Parsing is observational only: forward the original request and response
semantics unchanged. Do not modify a request to obtain usage or telemetry.

The Candidate must not depend on or attempt to access:

- a Docker or model Unix socket;
- host processes, GPU APIs, vLLM `/metrics`, or private controller state;
- private workload prompts, task identities, nonce, committed order, or
  reference implementation;
- extra probe requests to the model backend.

## Transparency requirements

For every accepted `/v1/models` or `/v1/chat/completions` request, make exactly
one matching request to `BACKEND_URL` and faithfully return that response.

Do not modify or replace:

- messages, model names, sampling parameters, stream options, or token limits;
- response content, usage, finish reasons, status, or errors;
- streaming chunk content or ordering.

Do not synthesize, cache, replay, truncate, merge, retry, skip, or speculate
model responses. Preserve end-to-end and benchmark tracing headers. Propagate
client cancellation, timeout, and disconnect promptly to the one matching
upstream request. Do not create hidden background backend work.

At the trusted end of a fixed measurement window, Judge may disconnect queued
or in-flight requests. The benchmark accounts for those boundary cancellations
separately. This is not permission to drop or short-circuit ordinary requests.

## Required architecture-first workflow

Establish transport correctness first, then build an explicit resource and
workflow model before performance optimization.

Before the first performance submission, update `ARCHITECTURE.md` with:

1. a resource model separating fixed per-program, context-growing, and
   request-transient costs;
2. a program state machine and cleanup rules;
3. a conservative capacity/admission invariant;
4. the scheduling unit, queue ordering, pause, and restoration policy;
5. bounded-wait, starvation, cancellation, and deadline behavior;
6. at least three materially different architecture families considered;
7. one falsifiable experiment row with expected evidence and a rejection rule.

Use a request-level semaphore/FIFO implementation only as a control. Before
spending hidden evaluations on adjacent scalar settings, implement and evaluate
at least one program-aware or adaptive architecture.

Before every public or hidden performance run, record the exact candidate
checkpoint/archive/hash and a concise description of what changed. Do not
delay a useful evaluation while constructing a bespoke checklist of secondary
metrics, causal thresholds, or rejection rules.

After every result, record the outcome, keep/reject decision, and exact location
of the protected best. Restore the protected best immediately after an
unsuccessful experiment.

## Architecture questions to answer

Use these as design questions, not as a prescribed algorithm:

1. Is the scheduling unit an HTTP request or a long-lived agent program?
2. How do REASONING and ACTING programs consume different resources?
3. How should fixed recurrent-state cost and rounded attention-context cost be
   combined into a conservative online estimate?
4. How can request growth and already-present response usage calibrate that
   estimate over time?
5. Under pressure, which programs should remain active, pause, or resume?
6. Which pending program should be admitted next, and why should that improve
   valid agent steps rather than only raw request throughput?
7. How are early workflow progress, deep-context efficiency, bounded waiting,
   and starvation balanced?
8. Which invariants remain true during cancellation, disconnect, release, and
   the fixed-window boundary?

## Local correctness checks

Run the deterministic loopback protocol smoke test after every meaningful
transport or scheduler change:

```bash
bash /workspace/protocol_tests/run_protocol_smoke.sh
```

It launches the current `launch.sh` against a small fake backend and checks
health, models forwarding, non-streaming bytes, streaming SSE ordering, error
passthrough, opaque headers, one-to-one forwarding, concurrency, cancellation,
and lifecycle isolation. It produces no performance score and does not model
the real coding-agent workload.

Also run:

```bash
python3 -m py_compile /app/submission/*.py
bash -n /app/submission/launch.sh
```

Add focused deterministic tests for your resource accounting, state
transitions, selection, cancellation, and cleanup logic. Keep development
tests outside the submitted directory or excluded from the runtime entry path.

## Public evaluation

Verify the public plane:

```bash
kvbench doctor
```

The public profile `public-swe192-distinct` runs 192 distinct SWE-Bench Lite
MiniSWE workflows at concurrency 192, at most 100 agent steps each. The first
300 seconds are an unscored steady-state warmup; only the following exact
600-second scored window is measured. Infrastructure/protocol setup is outside those
two intervals.

Record the frozen public Direct anchor, then evaluate the current snapshot:

```bash
kvbench run --mode baseline --profile public-swe192-distinct
kvbench run --mode candidate --profile public-swe192-distinct
kvbench compare
kvbench compare --json
```

The trusted launcher selects the public Direct evidence for this run and
provisions its non-secret rate as `KVBENCH_PUBLIC_DIRECT_RATE`; `kvbench run
--mode baseline` records that value locally. Do not assume a cross-run numeric
constant. Public and hidden evaluations share the serving system and are
serialized. Public snapshots and safe results are stored under
`/app/public_evals`.

Public differs from hidden in scored-window duration. It is an advisory
measurement, not a promotion gate. Once local protocol checks pass and Public
returns an authoritative `valid` result, a materially distinct, restorable
candidate is eligible for one Hidden submission. A Public score below Direct
is useful directional evidence but is not, by itself, a reason to forbid
Hidden evaluation.

Do not invent additional numeric prerequisites from Public diagnostics such as
prefix-hit rate, queue depth, or a required margin over Direct. Those fields
help interpret a result; only the hard gates listed under "Scoring contract"
decide validity. Skip Hidden only for a concise,
concrete reason: the Public result is invalid, local correctness failed, the
candidate is a duplicate, or the regression is clearly catastrophic. Avoid a
long chain of Public-only architectures: after at most two consecutive valid,
materially distinct Public candidates, send the stronger one to Hidden before
continuing exploration.

## Hidden iterative feedback

Once a complete architectural milestone passes local checks, submit it with:

```bash
sforge-submit --details
```

Your final result is the best valid score you reach within the time budget.

Use:

```bash
sforge-submit --list
```

to map results to checkpoints. Manual hidden submissions are count-unlimited,
but successful submissions obey the cooldown in `RUN_BUDGET.json`. Evaluations
are serialized and each normally takes about 35–45 minutes, so duplicate
snapshots and nearby one-literal sweeps consume the opportunity to evaluate a
materially different architecture.

A useful hidden milestone may change one or more of:

- the resource model;
- program state machine;
- admission invariant;
- scheduling/pause/restoration policy;
- fairness or lifecycle mechanism.

This list guides exploration; it is not a hidden-submission eligibility gate.
A coherent parameter bundle motivated by Public or prior Hidden evidence can
also be worth evaluating. Nearby one-literal sweeps are usually a poor use of
time, but do not withhold Hidden solely because a valid candidate missed a
self-imposed Public threshold.

Do not fit an optimum, interpolate a curve, or claim causality from one
sub-percent result. Judge feedback intentionally does not disclose private
prompts, task identities, per-request traces, committed order, nonce,
controller internals, or reference source.

## Final checkpoint

Keep `/app/submission` on the strongest known Judge-valid implementation except
while creating a restorable experiment snapshot. Write code changes promptly;
automatic background evaluation is disabled, but the formal runner captures an
immutable archive at the actual timeout boundary and evaluates it afterward.

Before the harness-reported deadline:

- restore the protected best;
- rerun compilation, shell syntax, focused tests, and protocol smoke;
- confirm `launch.sh` is executable;
- confirm monitoring state is bounded and all release/cancel paths are safe;
- update `ARCHITECTURE.md` with the final checkpoint hash and evidence.

## Clean-room policy

Do not copy an external implementation. The submitted gateway must have no
runtime dependency on the internet.

Do not attempt to reach the model service or host resources directly, inspect
evaluation state, discover private workloads, or obtain a reference
implementation. Judge remains the only path to live public or hidden
evaluation.
