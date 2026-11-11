# Host-only qualification oracle

FINAL_VERSION is 2 (`scorer_v002`, measured below). Both versions use the faithful helper-A port of Fast KVzip's `Weight` gate from `Janghyun1230/FastKVzip` at `e04afaa`; they differ in the trailing local-window rule (see Candidates). `scorer_v001/assets.pt` losslessly repacks the released MIT-licensed `Jang-Hyun/Fast-KVzip` `qwen3-8b/q4_dim16_sink16.pt` weights into 252 tensors, and `scorer_v002/assets.ref` reuses that asset. The pinned upstream SHA-256 is `8e0fd73396e487f5b7d01556ed9c119a686d0326a3517ae30168f48a8c92ed2a`; the deterministic canonical repack SHA-256 is `722aa9641a2ef8ce51655d4c26188c69e97d9e4679154ad882f9fe64e2e9ff39`; and the bounded lossless transport SHA-256 is `300de4f36e3f09c31117fe66a02512db40f9c981eda365d2d1e8213fc147d20e` (126,333,495 bytes). `scripts/make_oracle_archive.sh` reproduces both the git-ignored canonical asset and its transport (it runs `scripts/prepare_oracle_assets.py` with the repository virtualenv, `ORACLE_ASSET_PYTHON` to override) and writes `artifacts/oracle.tar.gz` (scorer_v002 plus the scorer_v001 it references, FINAL_VERSION 2). `ORACLE_VERSION=1` still packs the old oracle as `artifacts/oracle-v001.tar.gz`. This oracle is never copied into the Work image.

## Candidates

| version | what | window rule | H100 NVL 94 GB score (real Judge, final holdout, 1,812 rows) |
|---|---|---|---|
| `scorer_v001` (old oracle) | Fast KVzip released gate, FastKVzip repo rule | `int(0.02 n)` trailing tokens when n < 16000, otherwise 4096; window scores set to the global max | 82.74869749649393, valid, 13/13 gates (2 GPUs, run `nvl-kvzip-diag-oracle-20261020-133450`; bit-identical on 1 GPU and in run `kvzip-nvl-oracle-20261023-070906`); 82.83 on H100 SXM 80 GB |
| `scorer_v002` (FINAL_VERSION; best admissible paper solution, also the strongest paper solution) | the same released gate and weights, NVIDIA kvpress rule | `max(1, int(0.02 n))` when n < 32000, otherwise 4096; window scores set to 1.0 | **92.80859136442182**, valid, 13/13 gates (2 GPUs, run `forced-kvzip-v002-20261029-210914`) |

No forced (admission-bypassing) run was needed: v002 passes the Judge's normal source admission, and its only asset is the transport asset that every oracle run exempts with `KV_SCORING_HOST_ORACLE=1` (the same exemption v001 uses). The LLM integrity review was off (`SFORGE_LLM_JUDGE_MODE=off`), as for every oracle run.

v002 run `forced-kvzip-v002-20261029-210914` (2026-10-29, H100 NVL GPUs 0 and 1, wall 7,226 s, judge runtime 7,196 s): ratio_010 0.7998 / ratio_020 0.9685 / ratio_030 1.0160; groups contextual_qa 0.8665, redundancy 1.0123, retrieval 0.9054; overhead suite 0.052, item max 1.669, 0 item violations. Compare v001: 0.642 / 0.849 / 0.992; groups 0.858 / 0.902 / 0.722; overhead suite 0.051. The judged archive (sha256 `0959e9c6...`, built with `ORACLE_VERSION=2 scripts/make_oracle_archive.sh` before FINAL_VERSION changed) holds the same `scorer_v002/submission.py` (`90ed853a...`), `assets.ref` (`b32d46a5...`), `scorer_v001/submission.py` (`e4e20916...`) and transport asset (`300de4f3...`) that the default archive now packs (FINAL_VERSION 2 inside both); only `notes.md` differs. The `forced-` run-id prefix is the batch naming only; no gate was bypassed. It was run from the development tree's `scripts/eval_archive.sh` with Judge image `d8ca23feabe0`. That image has the same scoring code and holdout as the published image `99981140b92b`, and v001 gave a bit-identical score on both images.

`scorer_v001` window detail: `prefill/eval.py` and `prefill/model/wrapper.py` of FastKVzip `e04afaa` (2% below 16000 tokens, else a fixed 4096-token window, set to the maximum). On the held-out suite, that 4096-token window covers the whole per-head budget at r = 0.1 on 783 of 1,812 rows (334 at r = 0.2). On those rows the gate falls back to pure recency, which matches v001's ratio profile (0.642 / 0.849 / 0.992) and its weak retrieval group (0.722).

`scorer_v002` is the strongest official configuration of the same method:

- Gate math: FastKVzip `e04afaa` `prefill/attention/gate.py` `Weight.forward`, identical to NVIDIA/kvpress `2d66e74` `kvpress/presses/fastkvzip_press.py` `FastKVzipGate.forward`.
- RMSNorm: upstream `Qwen3RMSNorm` order (normalise in fp32, cast to bf16, multiply by the fp32 weight). v001 casts before multiplying.
- Window: kvpress `FastKVzipPress._score_fast` lines 252-261. This is the integration that the Fast KVzip README links to, and the one on the kvpress leaderboard.
- kvpress's `n_sink=4` protection has no effect here, because those tokens are in the always-visible system prompt.
- Weights: the same released checkpoint. `assets.ref` points to `scorer_v001`, so the only shipped asset is the transport asset that the Judge exempts under `KV_SCORING_HOST_ORACLE=1`.
- Nothing is tuned on this task, and no agent idea is used.
- CPU parity with v001 outside the window: max |Δ| ≤ 0.0054, top-10% overlap 0.986 to 0.997.
- Provenance: `Janghyun1230/FastKVzip@e04afaa` (arXiv 2601.17668); `NVIDIA/kvpress@2d66e74`; `snu-mllab/KVzip@5d84729` (the underlying protocol). KVzip's own reconstruction scorer cannot run under this ABI: it needs about 2x prefill, and the overhead cap is 25%.

Reproduce the v002 measurement (about 3.8 h on one H100 NVL, about 2 h on two):

```bash
scripts/make_oracle_archive.sh   # artifacts/oracle.tar.gz: FINAL_VERSION 2, scorer_v002 plus scorer_v001
KV_SCORING_HOST_ORACLE=1 SFORGE_LLM_JUDGE_MODE=off JUDGE_GPUS=<i,j> \
  scripts/eval_archive.sh artifacts/oracle.tar.gz <run-id>
```

The packed `scorer_v002/submission.py` (sha256 `90ed853a...`), `assets.ref` (`b32d46a5...`) and `scorer_v001` files are byte-identical to the paper-vs-agent comparison archive (`paper_submission.tar.gz`, sha256 `4d3f6c87...`, not shipped in this repository). Only `notes.md` differs. The Judge's source admission accepts the v002 archive (ok, final_version 2, no errors or warnings).
