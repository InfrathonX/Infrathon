# Paged-GQA Decode Lab

## Submission evaluation policy

Every `sforge-submit` runs the evaluation on hidden test cases. `sforge-submit --details` returns the gate states, the score and the four family scores. Hidden cases and per-case results stay on the Judge. Your final result is the best valid score you reach within the time budget, so keep `FINAL_VERSION` on your strongest candidate.

Read `AGENT_TASK.md` before editing. The only submitted surface is:

```text
candidates/kernel_vNNN/submission.py
FINAL_VERSION
notes.md
```

`kernel_v001/submission.py` starts at zero bytes. The selected module must
expose the exact seven-argument `paged_gqa_decode_bf16` ABI, launch at least
one reachable Triton kernel, write the BF16 `out` tensor in place, return
`None`, write nothing outside `out` and its own scratch, and leave all inputs
unchanged.

The support domain is wider than the public cases: any `B` from 1 to 256, any
`S_q` from 1 to 16, any GQA group with `H_q <= 128` (a power of two
or not), page sizes 1 to 256, cache lengths up to 1,048,576, page tables wider
than needed, page pools larger than `B * W` with repeated (shared) page ids,
and NaN or Inf in cache slots no query may see. Scored calls look like a
serving engine's decode steps: very uneven lengths in one batch, shared prefix
pages, long contexts, and sequences of consecutive steps in which `B` and every
length change between calls (compilation during those timed calls counts). A sensible first version is one program per
request/query head with FP32 online softmax that already follows four rules:

1. pad a group size or `S_q` that is not a power of two up to the next power of two and mask
   the padded rows on load and store;
2. use `page_table.shape[1]` (or its stride) as the table row stride, never
   `k_cache.shape[0] // B`, and compute cache offsets in 64 bits;
3. zero or mask V for `j >= L_b` (`0 * NaN` is NaN);
4. mask K logits with `tl.where`, not an additive `-inf` (`NaN + (-inf)` is
   NaN), with a finite mask value far below `-4.6e4` if you use one.

Logical key `j` is gathered through `page_table[b, j // page_size]`; query
head `h` maps to `h // (H_q // H_kv)`; query row `i` stops at
`cache_seqlens[b] - S_q + i`. Shape-keyed fast paths are fine as long as every
path is correct for every input that can reach it, but a kernel that compiles
a new variant for every batch size, length or table width pays for it in the
sequence family; safe same-device Torch
scratch allocation and multiple Triton launches are allowed.

Development commands:

```bash
./tools/run_one.sh --version 1
./tools/run_public_tests.sh --quick --candidate-version 1
./tools/run_public_tests.sh --candidate-version 1
./tools/run_public_tests.sh --family stream --candidate-version 1
./tools/run_public_tests.sh --edge --candidate-version 1
./tools/run_public_tests.sh --sanitize --candidate-version 1
```

The first three select exactly 1, 3, and 127 public cases; `--family` selects
one family (decode, spec, long or stream). `--edge`
runs the public edge cases natively with guard bands and input checks and
reports each case; `--sanitize` runs their sanitizer subset under
compute-sanitizer memcheck (it prints `MEMCHECK UNAVAILABLE` and exits 2 where
the sanitizer cannot attach). Run `--sanitize` before `--edge` whenever a
kernel is new: memcheck reports an out-of-bounds access before it reaches the
GPU. Public output provides disclosed geometry, correctness summaries,
per-case edge results and performance aggregates only; it never runs the
hidden cases.

Compute in BF16 and FP32 only; FP16, FP8, INT8 and FP4 count as
approximations even within tolerance. Each call computes only its own output:
no caching across calls, no work for later calls, no work after the call
returns, and no patching of imported modules, clocks, CUDA events or global
numeric settings.

Host code obeys a small statically checked subset: no host `for`/`while`/
`try`, comprehensions, `lambda`, f-strings, or default arguments; Torch is
limited to metadata and scratch allocation; every host helper must be reachable
from the entrypoint; and `@triton.jit` takes no arguments while tuning goes
through `@triton.autotune` with `triton.Config`. Kernel bodies are unrestricted
Triton. `AGENT_TASK.md` lists the whole contract -- read it before writing the
first version.

Keep a known-correct version before changing schedules. Record measurements in
`notes.md`, then point `FINAL_VERSION` to the strongest clean candidate and
run the full public suite, the edge suite and the sanitizer suite. Only Torch
metadata/scratch allocation and self-contained Triton are allowed in the
submission: no network use, copied source, existing attention kernel, private stream, host
Torch tensor math, FP16/FP8/INT8/FP4 path, dynamic loading, or evaluator
introspection.

Before each `sforge-submit --details`, wait for all public-test and compiler
workers to exit.
