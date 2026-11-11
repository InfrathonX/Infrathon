# Forced-oracle Judge: official EAGLE-3 in SGLang

LOCAL-ONLY tooling to get the real Judge result of the strongest official
EAGLE-3 implementation, SGLang's EAGLE-3 speculative decoding (the serving
integration the EAGLE authors point to), which the agent rules do not admit
(subprocess, sockets, environment routing; SGLang is not in the images). Agents
are always judged by the unmodified `:local` image; this never retags or pushes it.

## Provenance

- Paper: EAGLE-3, Li et al., arXiv:2503.01840. Reference implementation
  https://github.com/SafeAILab/EAGLE @ `cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b`
  (the seed `engine_v001` is a statement-level port of it).
- Engine: SGLang https://github.com/sgl-project/sglang, PyPI `sglang==0.5.20`
  (torch 2.13.0+cu130, transformers 5.12.1, sglang-kernel 0.4.7, flashinfer 0.6.18),
  installed with `uv==0.9.30 pip install --prerelease=allow` into an isolated
  virtualenv `/opt/sglang-venv`, so the Judge's own torch 2.8.0 / transformers
  4.53.1 (used by the seed reference and the teacher-forced oracle) is untouched.
- Models: the task's frozen mounts, `Qwen/Qwen3-8B` @ `b968826d...` and the
  converted `Tengyunw/qwen3_8b_eagle3` @ `2a1059d5...` head (same key names and
  `config.json`, `LlamaForCausalLMEagle3`), both read-only.
- Draft tree: the head's model-card SGLang command (`--speculative-num-steps 6
  --speculative-eagle-topk 10 --speculative-num-draft-tokens 32`), CUDA graphs,
  default Hopper attention backend; plus `--enable-deterministic-inference`
  (needed for O5, see below) and `--disable-radix-cache`.
- Glue: `oracle/solutions/engine_v002/candidate.py`.

## Files

| file | role |
| --- | --- |
| `Dockerfile` | private Judge `:local` + SGLang venv + admission override |
| `source_admission_forced.py` | wraps the enforced `verifier/source_admission.py` (kept as `source_admission_enforced.py`), records its errors as `forced_bypass_errors` and admits the candidate; `protected_hashes.txt` is rebound for that one file (and the enforced copy is added), every other protected identity is checked unchanged |
| `forced_entry.py` | runs the task's own `scripts/sforge_h100_entry.py`, adding `--cpuset-cpus` for the Judge |
| `run_forced_oracle.sh` | same `sforge eval` call and Judge env as `scripts/eval_archive.sh`, with a temporary task-spec copy whose `judge.image_tag` is `forced-oracle`; `FORCED_TAG=local` runs an admissible control through the normal Judge |
| `make_forced_oracle_archive.sh` | packs `solutions/engine_v001` (Work seed) + `oracle/solutions/engine_v*`, `FINAL_VERSION` = argument or `oracle/FINAL_VERSION` |

## Build and run

```bash
docker build -t eagle3_spec_decode.judge.eagle3_spec_decode_engine_qwen3_8b_h100_v1:forced-oracle \
  oracle/paper_reference/forced_judge
oracle/paper_reference/forced_judge/make_forced_oracle_archive.sh /tmp/eagle3_oracle_v2.tar.gz 2
JUDGE_GPU=7 EAGLE3_TARGET_REPO=~/.cache/huggingface/hub/models--Qwen--Qwen3-8B \
  EAGLE3_HEAD_DIR=/path/to/converted/head MLSYSFORGE_ROOT=/path/to/sforge-checkout \
  oracle/paper_reference/forced_judge/run_forced_oracle.sh /tmp/eagle3_oracle_v2.tar.gz forced-eagle3-v2-gpu7-<stamp>
```

Image used: `sha256:b0bdd34c84105b8b7d915214dde833539716c865041d36862d35f27fb5bafc01`,
FROM `:local` `sha256:1b0e376474a8aed59314bd3586147a60c9554dc4c19220cb1d27fb8c47454563`.
A Judge run takes about 25 minutes (seed reference, candidate, two oracle passes).
Reports: `$MLSYSFORGE_ROOT/outputs/<task>/runs/<run id>/runs/<run id>/<task id>/submissions/manual-1/`
(`report.json`; `eagle3_spec_decode_private.json` in `private_judge_artifacts.tar.gz`).

## Results (H100 NVL GPU 7, cpuset 28-31,60-63, final holdout, 2026-10-29)

See `oracle/notes.md` for the full table. Forced engine_v002: 414.16
(`forced-eagle3-v2-gpu7-20261029-213711`) and 415.10
(`forced-eagle3-v2-gpu7-r2-20261029-214543`) tokens/s, valid, all 11 Judge
gates PASSED. Judged `candidate.py` sha256
`c5916bf0e1d5b96765cf57ea2ca58baeb41c70d75218d78f90b8cf7df75ed4f7`
(archive sha256 `d8aa66304b2fb2e606452e84e94206004a73009964d9aa3d2fd5d4310e97e235`).

Bypassed: source admission (7 enforced errors: `http.client`, `socket`,
`subprocess` imports, environment-variable routing) and the LLM integrity review
(`SFORGE_LLM_JUDGE_MODE=off`). Caveat on `memory_ok`: the gate reads the worker
process's `torch.cuda.max_memory_allocated` (17.16 GiB, the handed-in target and
draft), not the SGLang server process, which loads its own copy of the weights
and a KV pool (`--mem-fraction-static 0.55`, about 52 GB of the 94 GB card). By
device-level accounting the engine would exceed the 27.7-30.0 GiB same-run cap,
so treat the memory gate as not meaningfully enforced for this candidate. The
task Judge has no compute-sanitizer stage.

Why `--enable-deterministic-inference`: without it SGLang's batch-size-dependent
kernels made the Judge's O5 rerun (two prompts submitted together) diverge from
the batch-1 streams on 5 of 6 public cells; with it all six reruns reproduce
exactly, at about 10-15% lower bs1 throughput (public-prompt probe: 380/445/354
tokens/s chat/math/code without, 325/384/351 with).
