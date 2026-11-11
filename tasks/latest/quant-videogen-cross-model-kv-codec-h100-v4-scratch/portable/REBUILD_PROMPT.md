# Rebuild and run the Quant-VideoGen Cross-Model KV Codec v4 on a fresh multi-GPU host

This is the **operational layer** for this task: how to assemble the external
inputs, build the images, derive the hidden inputs for the 17 cases, verify
everything, judge an archive and launch an agent run on a clean host with one
Work GPU and one to eight admissible Judge GPUs. The task itself (ABI, score,
rules) is summarised in the task `README.md` and the agent instructions in
`images/work/AGENT_TASK.md`.

The release config `configs/h100_scratch_v4_release.json` is
`status: "calibrated"`: every golden and init SHA-256, each case's
`bf16_vbench` and `bf16_kv_bytes`, both gate layers and an
`oracle_calibration` block are in it. It is Judge-only. In the public
distribution it is not in the repository; `assets/fetch.sh` places it at
`<INFRATHON_ASSETS_ROOT>/<task directory>/judge/h100_scratch_v4_release.json`, and
`scripts/build_images.sh` reads it from there when `INFRATHON_ASSETS_ROOT` is set.
The scripts that read the release directly (`scripts/lib.sh`,
`scripts/stage_eval_inputs.py`, `scripts/calibrate_gates.py`,
`scripts/refresh_portable_manifests.py`, `scripts/validate_layout.py`) expect
it at `configs/h100_scratch_v4_release.json`.

You may use Codex, Claude, or another build agent, but **do not let the build
agent redesign the benchmark.** Preserve these scientific identities
byte-for-byte:

- the TaskCore and its resource manifest;
- the hidden case table: 6 LongCat-Video cases (MovieGen prompts #12, #185,
  #320, #593, #893, #894, each a frozen 73-frame init continued for 10
  segments, 273 frames) and 11 HY-WorldPlay-8B cases (the 10 official
  `assets/test_case.csv` image/caption pairs plus the QVG `run.sh` case, 285
  frames each, the paper camera trajectory), their pinned inputs and BF16
  goldens (SHA-256 in `configs/h100_scratch_v4_release.json`);
- the calibrated mean gates and per-case floors, the paper protocol, the
  per-model budgets, the cumulative byte accounting in the pinned upstream
  integrations and the metric implementations;
- the public diagnostics (fast ABI/fidelity/compression check and the
  Self-Forcing paper protocol at prefixes 350 and 700), the source admission
  policy and the read-only codec compatibility runtime;
- the agent-facing optimization contract (an empty `submission` start).

Every relative path below is relative to the task directory
`tasks/latest/quant-videogen-cross-model-kv-codec-h100-v4-scratch/`; `<repo>`
is the repository root. Host-private values (GPU UUIDs, ports, directories,
proxy addresses) are written as placeholders — substitute your own.

---

## 1. Host prerequisites

Do `<repo>/docs/HOST_PREREQUISITES.md` first: Docker >= 24 with the NVIDIA
runtime as the **default** runtime, your user in the `docker` group, `uv` plus
the repository venv, `jq flock rsync patch git sha256sum nvidia-smi` on PATH,
and >= 150 GB free on Docker's data root.

On top of that this task needs:

| Need | Why |
| --- | --- |
| `curl docker flock jq nvidia-smi python3 sha256sum ss tar` on PATH | the scripts check each one before touching Docker (`scripts/lib.sh`) |
| `hf` (Hugging Face CLI) and `git` with network access | the artifact root must be assembled with the Hub client, never `wget` (§2) |
| ~196 GiB for the artifact root, on top of the Docker data root | the ~190 GiB of public trees in §2 plus ~6 GB of hidden inputs for the 17 cases |
| Docker data root headroom for the images (base, Work, public Judge, Judge holdout layer) | the Judge image is `FROM` the Work image, which is `FROM` the base |
| Egress for the **build step only** | the base build reaches the registry, PyPI and the pinned flash-attention release; behind a proxy export `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY` (lowercase spellings work too) and `scripts/build_images.sh` forwards them as Docker's predefined build args |
| `umask 022` in the launching shell | a docker-group shell's default umask makes the run root group-writable and the model-lock safety check then refuses it (§6.2) |
| Host RAM for concurrent cases | one LongCat case peaks at about 37 GiB of process memory and one HY case at about 34 GiB; eight cards running six LongCat and two HY cases need roughly 290 GiB plus the metric processes (an 8-GPU oracle run peaked at about 272 GiB in the Judge cgroup) |

The agent CLI runtime and agent credentials are needed only for agent runs
(§7), not for rebuilding and qualifying the task.

---

## 2. External inputs

Everything external lives under one read-only tree named by `QVG_ARTIFACT_ROOT`
(`scripts/lib.sh`; missing or not a directory makes `require_artifact_root`
die). A correct fresh root is about **196 GiB**: the public trees below plus
the hidden inputs `scripts/stage_eval_inputs.py` derives for the 17 cases.

**Assemble it with `hf download --revision <sha> --local-dir <path>`, never
`wget` or a plain `git clone`**: `scripts/verify_models.py` requires the
`.cache/huggingface/` revision evidence that only the Hub client writes. Keep
`.git` for the four git checkouts — the verifier runs `git rev-parse HEAD`
before and after each walk. Do not chmod or repack anything afterwards; file
modes are part of `tree_sha256`.

Revisions come from `configs/paper_protocol.json` (`model_revisions`,
`mlsysforge_supplemental_revisions`, `vbench_assets_sha256`); the path bindings
and the per-tree digests are in `scripts/verify_models.py` and
`locks/model-lock.json`. `$ROOT` below is `QVG_ARTIFACT_ROOT`.

| Input | Source @ pin | Locked size | SHA | Expected at | Availability |
| --- | --- | --- | --- | --- | --- |
| Wan2.1 T2V 1.3B DiT / VAE / umT5-XXL | `Wan-AI/Wan2.1-T2V-1.3B` @ `37ec512624d61f7aa208f7ea8140a131f93afc9a` | 17.6 GB | `tree_sha256` in `locks/model-lock.json` | `$ROOT/staging/self-forcing/Wan2.1-T2V-1.3B` | **public** |
| Self-Forcing DMD checkpoint (`checkpoints/self_forcing_dmd.pt` only) | `gdhe17/Self-Forcing` @ `2f8b779212da279d212c22a509b66ad6552f350e` | 5.7 GB | `tree_sha256` | `$ROOT/staging/self-forcing/Self-Forcing` | **public** |
| LongCat-Video 13B (dit, text_encoder, vae, tokenizer, lora) | `meituan-longcat/LongCat-Video` @ `03b55529b1d1d4045f5fbe14d65c8c6e8116b278` | 83.3 GB | `tree_sha256` | `$ROOT/staging/longcat/LongCat-Video` | **public** |
| HY-WorldPlay AR distilled action model (`ar_distilled_action_model/model.safetensors` only) | `tencent/HY-WorldPlay` @ `f4c29235647707b571479a69b569e4166f9f5bf8` | 34.2 GB | `tree_sha256` | `$ROOT/staging/hy-worldplay-8b/HY-WorldPlay` | **public** |
| HunyuanVideo-1.5 backbone subset (`transformer/480p_i2v/`, `vae/`, `scheduler/`) | `tencent/HunyuanVideo-1.5` @ `9b49404b3f5df2a8f0b31df27a0c7ab872e7b038` | 38.3 GB | `tree_sha256` | `$ROOT/staging/hy-worldplay-8b/HunyuanVideo-1.5` | **public** |
| Qwen2.5-VL-7B-Instruct LLM text encoder | `Qwen/Qwen2.5-VL-7B-Instruct` @ `cc594898137f460bfe9f0759e9844b3ce807cfb5` | 16.6 GB | `tree_sha256` | `$ROOT/staging/hy-worldplay-8b/Qwen2.5-VL-7B-Instruct` | **public** |
| ByT5 glyph text encoder (4 files only) | `google/byt5-small` @ `68377bdc18a2ffec8a0533fef03b1c513a4dd49d` | 1.2 GB | `tree_sha256` | `$ROOT/staging/hy-worldplay-8b/byt5-small` | **public** |
| SigLIP vision encoder | `black-forest-labs/FLUX.1-Redux-dev` @ `c95859fbf7703ca4d6824b4da4407d7cd0434f81` | 1.1 GB | `tree_sha256` | `$ROOT/staging/hy-worldplay-8b/FLUX.1-Redux-dev` | **gated** — accept the licence and export `HF_TOKEN` first |
| Glyph-ByT5 mapper / byt5 model / UNet LoRA | ModelScope `AI-ModelScope/Glyph-SDXL-v2` @ `b51a8604c03c9a4e71a389f4c7deeb526320f2e6` | 1.9 GB | pinned commit + `tree_sha256` | `$ROOT/staging/hy-worldplay-8b/Glyph-SDXL-v2` | **public** (git, keep `.git`) |
| Official HY-WorldPlay repo: the 11 HY case images (`assets/img/1.png`…`10.png`, with `5.jpeg` and `6.jpeg`, plus `img/test.png`) and the captions in `assets/test_case.csv`; not mounted itself | `github.com/Tencent-Hunyuan/HY-WorldPlay` @ `1588e1336e842b03b0a7860c654ebd7c46bb065e` | 23.6 MB | pinned commit; every staged image by sha256 in the release | `$ROOT/staging/hy-worldplay-official` | **public** (git, keep `.git`); Tencent HY-WorldPlay Community License, the same licence as the model |
| VBench evaluator source | `github.com/Vchitect/VBench` @ `45e79ec14e69a2187202c675d2dbce1a71843d53` | 261 MB | pinned commit | `$ROOT/staging/vbench` | **public** (git, keep `.git`) |
| DINO model definition (loaded with `source='local'`) | `github.com/facebookresearch/dino` @ `7c446df5b9f45747937fb0d72314eb9f7b66930a` | 25.9 MB | pinned commit | `$ROOT/staging/vbench-cache/dino_model/facebookresearch_dino_main` | **public** (git, keep `.git`) |
| 8 loose VBench / LPIPS weights: CLIP ViT-B-32 and ViT-L-14, DINO ViT-B/16 (**twice** — the local path and the `torch.hub` path look in different directories), the LAION aesthetic head, MUSIQ-SPAQ, AlexNet, VGG16 | URLs hard-coded in the pinned VBench source | 2.9 GB | per-file sha256 in `configs/paper_protocol.json` (`vbench_assets_sha256`) | `$ROOT/staging/vbench-cache/{clip_model,dino_model,aesthetic_model,pyiqa_model,torch/hub/checkpoints}` | **public** (direct download) |
| Hidden case inputs and BF16 goldens (17 cases) | **generated**: `scripts/stage_eval_inputs.py --gpus <list>` walks the release case table inside the Judge image with the Judge's read-only model mounts. LongCat: `judge/run_longcat.py --mode bf16 --make-init` gives each prompt's 73-frame `init.mp4`, then `--mode bf16 --segments 10` continued from it gives the 273-frame golden (`expandable_segments` on); HY: `judge/run_hy_worldplay_8b.py --mode bf16` with the case image, caption and paper pose gives the 285-frame golden. Case `lc-012`'s init is pinned to `7108816a…` and is never regenerated; `hy-qvg`'s golden is pinned to `a8ee4657…`, so a regeneration must reproduce it bit for bit or is refused (an existing copy can be carried with `--reuse`). The release's inits and goldens were produced with these runners on 8 x H100 NVL, each with its BF16 VBench | about 6 GB (each golden is about 0.33–0.34 GB of uint8 frames) | sha256 per file in `configs/h100_scratch_v4_release.json`; the Judge refuses a mismatch. The script writes a `MANIFEST.json` (sha256 + device/driver) and reports whether the bytes reproduce the release; a non-Hopper device is a new measurement profile and needs a release refresh | `$ROOT/hidden/{longcat/<idx3>,hy/<nn\|qvg>}/**` → `/eval/hidden` in the Judge | derived from the public pins (GPU capture, recipe-bound) |
| Hidden prompt suite and calibrated thresholds | the 1003 Movie Gen Video Bench prompts at `judge/private/prompts/MovieGenVideoBench_extended.txt` (Judge-only path; `.dockerignore` excludes `judge/private/`, `validate_layout.py` checks it); the six case prompts are lines 12, 185, 320, 593, 893 and 894 (1-based); the thresholds are in the release config | 315 KB | bound through the release digest (`prompt_file_sha256`, per-case `prompt_index` and `prompt`) | `scripts/stage_eval_inputs.py` copies it to `$ROOT/hidden/prompts/…` | only two records of the suite, #1 and #6, are Work-visible, in `assets/public_prompts.txt` |
| Public Self-Forcing reference outputs the public diagnostics compare against | **generated**: `scripts/stage_eval_inputs.py --only public-eval` runs the Work image's `run_self_forcing_paper.py --mode bf16` and the pristine reference codec (`--mode candidate --vbench`) on the two public prompts | 3.9 GB | `public-eval/self-forcing/MANIFEST.json` (frames sha256 + device); zero-weight diagnostics, bytes differ per GPU by design | `$ROOT/public-eval/self-forcing/{bf16,qvg-reference}/**` → `/eval/public` in Work | derived from the public pins |
| Work base image | built here from `images/work/base/Dockerfile` + `images/work/base/requirements.lock` | — | upstream image pinned by manifest digest; every lock line an exact version, the one URL requirement additionally by sha256 | `BASE_WORK_IMAGE`, default `quant_videogen_h100_scratch_v4.base.<task_id>:local` | **built from this repository** (set `BASE_WORK_IMAGE` only to substitute an externally supplied base, which is then validated, not built) |
| Agent CLI runtime | `<repo>/launch/build-agent-runtime.sh` | — | `tree_sha256` in the runtime marker | `SFORGE_AGENT_RUNTIME_DIR` | host-provided; both images are agent-free and the CLI is bind-mounted read-only at run time |
| Agent credentials | task owner | — | — | `CODEX_AUTH_JSON_PATH` (default `$HOME/.codex/auth.json`) | owner-provided; must be a regular non-symlink file, mode 600, owned by the launching user |

The exact download command list is:

```bash
# --- Hugging Face snapshots (8 trees, 184.4 GiB) -----------------------------
hf download Wan-AI/Wan2.1-T2V-1.3B \
  --revision 37ec512624d61f7aa208f7ea8140a131f93afc9a \
  --local-dir "$ROOT/staging/self-forcing/Wan2.1-T2V-1.3B"
hf download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt \
  --revision 2f8b779212da279d212c22a509b66ad6552f350e \
  --local-dir "$ROOT/staging/self-forcing/Self-Forcing"
hf download meituan-longcat/LongCat-Video \
  --revision 03b55529b1d1d4045f5fbe14d65c8c6e8116b278 \
  --local-dir "$ROOT/staging/longcat/LongCat-Video"
hf download tencent/HY-WorldPlay ar_distilled_action_model/model.safetensors \
  --revision f4c29235647707b571479a69b569e4166f9f5bf8 \
  --local-dir "$ROOT/staging/hy-worldplay-8b/HY-WorldPlay"
hf download tencent/HunyuanVideo-1.5 \
  --revision 9b49404b3f5df2a8f0b31df27a0c7ab872e7b038 \
  --include "vae/*" "scheduler/*" "transformer/480p_i2v/*" \
  --local-dir "$ROOT/staging/hy-worldplay-8b/HunyuanVideo-1.5"
hf download Qwen/Qwen2.5-VL-7B-Instruct \
  --revision cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --local-dir "$ROOT/staging/hy-worldplay-8b/Qwen2.5-VL-7B-Instruct"
hf download google/byt5-small \
  config.json pytorch_model.bin special_tokens_map.json tokenizer_config.json \
  --revision 68377bdc18a2ffec8a0533fef03b1c513a4dd49d \
  --local-dir "$ROOT/staging/hy-worldplay-8b/byt5-small"
# gated: accept the licence and export HF_TOKEN first
hf download black-forest-labs/FLUX.1-Redux-dev \
  --revision c95859fbf7703ca4d6824b4da4407d7cd0434f81 \
  --local-dir "$ROOT/staging/hy-worldplay-8b/FLUX.1-Redux-dev"

# --- git checkouts (4 trees, 2.1 GiB; keep .git) -----------------------------
git clone https://www.modelscope.cn/AI-ModelScope/Glyph-SDXL-v2.git \
  "$ROOT/staging/hy-worldplay-8b/Glyph-SDXL-v2"
git -C "$ROOT/staging/hy-worldplay-8b/Glyph-SDXL-v2" \
  checkout b51a8604c03c9a4e71a389f4c7deeb526320f2e6
git clone https://github.com/Tencent-Hunyuan/HY-WorldPlay.git \
  "$ROOT/staging/hy-worldplay-official"
git -C "$ROOT/staging/hy-worldplay-official" \
  checkout 1588e1336e842b03b0a7860c654ebd7c46bb065e
git clone https://github.com/Vchitect/VBench.git "$ROOT/staging/vbench"
git -C "$ROOT/staging/vbench" \
  checkout 45e79ec14e69a2187202c675d2dbce1a71843d53
git clone https://github.com/facebookresearch/dino.git \
  "$ROOT/staging/vbench-cache/dino_model/facebookresearch_dino_main"
git -C "$ROOT/staging/vbench-cache/dino_model/facebookresearch_dino_main" \
  checkout 7c446df5b9f45747937fb0d72314eb9f7b66930a

# --- loose VBench / LPIPS weights (8 files, 2.7 GiB) -------------------------
# VBench's own downloader writes into $VBENCH_CACHE_DIR, which is a read-only
# mount at run time, so the cache must be complete before the first run.
wget -P "$ROOT/staging/vbench-cache/clip_model" \
  https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
wget -P "$ROOT/staging/vbench-cache/clip_model" \
  https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
wget -P "$ROOT/staging/vbench-cache/dino_model" \
  https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth
wget -P "$ROOT/staging/vbench-cache/torch/hub/checkpoints" \
  https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth
wget -P "$ROOT/staging/vbench-cache/aesthetic_model/emb_reader" \
  https://raw.githubusercontent.com/LAION-AI/aesthetic-predictor/main/sa_0_4_vit_l_14_linear.pth
wget -P "$ROOT/staging/vbench-cache/pyiqa_model" \
  https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/musiq_spaq_ckpt-358bb6af.pth
wget -P "$ROOT/staging/vbench-cache/torch/hub/checkpoints" \
  https://download.pytorch.org/models/alexnet-owt-7be5be79.pth
wget -P "$ROOT/staging/vbench-cache/torch/hub/checkpoints" \
  https://download.pytorch.org/models/vgg16-397923af.pth
```

`dino_vitbase16_pretrain.pth` is deliberately downloaded twice: VBench's local
DINO path and the `torch.hub` path look in different directories.

Nothing in the root is owner-provided. Once the public staging trees pass
`scripts/verify_models.sh`, one command derives the rest:

```bash
QVG_ARTIFACT_ROOT="$ROOT" \
  python3 scripts/stage_eval_inputs.py --gpus 0,1,2,3   # prompts + 17 cases + public-eval
```

It needs the built Judge and Work images (§5). Each listed card runs one case
at a time and holds both host lock files (`/tmp/sforge-gpu-<uuid>.lock`,
`/tmp/mlsysforge-gpu-<uuid>.lock`) while it does. The cost is dominated by the
six LongCat cases: with eight cases at once on 8 x H100 NVL, each 10-segment
BF16 continuation takes about 3 hours (9,993–11,220 s) after its init
(1,071–3,530 s).
Goldens are GPU captures: on the qualification profile they reproduce the
release digests byte-for-byte; a non-Hopper device is a new measurement
profile and requires regenerating goldens **and** gates (release refresh).

**Never overwrite a frozen LongCat init with a regenerated one.** Generating
the 73-frame init from scratch is not stable on the same image: independent
regenerations can differ from the frozen file. The init is generated once,
frozen by digest, and every later golden continues from the frozen file.
Continuing from a frozen init is byte-reproducible for one segment; the
ten-segment continuation has not been checked separately. The HY side is
bit-deterministic (`hy-qvg` and `hy-01` regenerate identically). LongCat
oracle frames are not: two oracle runs of the calibrated release differ by up
to 0.09 dB PSNR in 5 of 6 LongCat cases (§6.5).

Do not add a `staging/hy-worldplay/` tree (Wan-5B weights): it is in no
manifest and no mount plan, and `paper_protocol.json` names
`Wan2.2-TI2V-5B-Diffusers` as a forbidden substitute.

**Mount plan** (`scripts/sforge_runtime_entry.py`, all read-only; a missing host
path fails the run before the container starts). Work gets four mounts:
`staging/self-forcing` → `/opt/qvg-src/ckpts/Self-Forcing`, `staging/vbench` →
`/opt/vbench`, `staging/vbench-cache` → `/opt/vbench-cache`, `public-eval` →
`/eval/public`. Judge gets twelve: `staging/longcat/LongCat-Video` →
`/models/hidden/longcat`; the three HunyuanVideo-1.5 subdirectories and
`HY-WorldPlay/ar_distilled_action_model` under
`/models/hidden/hy-worldplay/{base/transformer,base/vae,base/scheduler,action}`;
`Qwen2.5-VL-7B-Instruct`, `byt5-small` and `Glyph-SDXL-v2` under
`/models/hidden/hy-worldplay/base/text_encoder/{llm,byt5-small,Glyph-SDXL-v2}`;
`FLUX.1-Redux-dev` → `/models/hidden/hy-worldplay/base/vision_encoder/siglip`;
`staging/vbench` and `staging/vbench-cache` as for Work; and `hidden` →
`/eval/hidden`, laid out as

```text
/eval/hidden/
  prompts/MovieGenVideoBench_extended.txt
  longcat/<idx3>/{init.mp4, bf16-10seg.frames.pt}       # 012 185 320 593 893 894
  hy/<nn|qvg>/{image.<png|jpeg>, bf16-285.frames.pt}    # 01..10, qvg
```

The HY case images live only there; they are not baked into the Judge image.

The Judge container runs with `network_mode=none` and
`HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` / `DIFFUSERS_OFFLINE=1`, so
anything not on disk is a hard failure rather than a lazy download.

---

## 3. GPU envelope and what enforces it

v4 has two roles with different device counts. **Work** gets one CUDA device of
its own. The **Judge** gets one to eight identical CUDA devices and shards the
17 cases over them; the score does not depend on the count, only the wall time
does.

| Item | Value | Enforced by |
| --- | --- | --- |
| Judge device count | `judge.min_gpu_count..judge.max_gpu_count` = 1..8 identical devices (same name and memory), distinct UUIDs, all idle at start | `judge/evaluate_hidden.py` hardware probe, run in a subprocess so the controller keeps no CUDA context |
| Work device count | exactly one | `scripts/sforge_runtime_entry.py` |
| Vendor | `vendor = nvidia` | `scripts/lib.sh` |
| Compute capability | `>= 8.0` | `judge/evaluate_hidden.py` and `scripts/lib.sh` |
| Device memory (each Judge GPU) | `>= judge.min_device_memory_bytes` = 84,825,604,096 bytes (79 GiB, like the other 80 GB-class tasks; an H100 SXM 80 GB reports 85,019,590,656 bytes to torch) from the release | `judge/evaluate_hidden.py`; the task JSON execution profile `cuda-sm80-79gib-sharded-judge` and `scripts/lib.sh` (`GPU_MIN_TOTAL_MIB` 80896) agree, asserted by `tests/test_portable_contract.py`, `tests/test_capability_envelope.py` and `tests/test_eval_archive.py` |
| Device binding | every Judge GPU stays visible to every process; `CUDA_VISIBLE_DEVICES` is never set. A case's runner gets `QVG_DEVICE_INDEX=<index>` and `CUDA_DEVICE_ORDER=PCI_BUS_ID`, and the pinned upstream entrypoints pick that device. The runner records `other_device_allocated_bytes` (current or peak reservations on every other GPU) and `cuda_device_index`; non-zero bytes, a device other than the slot's, or a `calls` count other than the case's segments (LongCat) or context passes (HY, 17 at 285 frames) fails the candidate, and the source policy rejects `set_device`, `torch.cuda.device` and the allocator-statistics resets | `judge/run_longcat.py`, `judge/run_hy_worldplay_8b.py`, the upstream patches, `judge/evaluate_hidden.py` |
| Whole GPU, canonical UUID | `GPU-…` UUIDs only; `all`, `void`, `none` and MIG identities are rejected. Work gets exactly one; the Judge gets the list in the host variable `QVG_JUDGE_GPU_UUIDS` (space separated) | `scripts/sforge_runtime_entry.py` |
| Idle at admission | each listed card idle, with both host lock files (`/tmp/sforge-gpu-<uuid>.lock`, `/tmp/mlsysforge-gpu-<uuid>.lock`) held for the whole evaluation | `scripts/eval_archive.sh`, `scripts/lib.sh` |
| Features | `cuda`, `bf16`, `triton`, `read-only-model-mount` | the software ABI document `portable/software-abi.json` |
| Container resources | task JSON `work` / `judge`; the Judge's CPU/memory share for a given Judge GPU count comes from `scripts/lib.sh judge_plan` (sized at 32 CPUs / 448 GiB for eight GPUs). Each Judge GPU slot gets `threads_per_slot` = 4 CPUs through `taskset`, with `OMP_NUM_THREADS`/`MKL_NUM_THREADS` set to match | task JSON; release `judge`; `scripts/lib.sh` |

**Measured peaks** (H100 NVL):

| Run | GPU peak | Process memory | Wall |
| --- | --- | --- | --- |
| LongCat, 10 segments, BF16 golden (`expandable_segments` on) | torch peak 89,096,133,120 B (83.0 GiB) | about 37 GiB per case | 9,993–11,220 s with eight cases at once |
| LongCat, 10 segments, BF16 without `expandable_segments` | out of memory in segment 2 (fragmentation) | — | — |
| LongCat 73-frame init, BF16 | 44,731,942,400 B | — | 1,071–3,530 s |
| HY-WorldPlay-8B, 285 frames, BF16 golden | torch peak 97.39e9–97.43e9 B | about 34 GiB per case | — |
| LongCat oracle in the Judge (candidate mode), eight cases at once | — | — | stage 2,691–2,829 s; about 48–50 min with metrics |
| HY oracle in the Judge (candidate mode), eight cases at once | — | — | stage 595–871 s; about 11–18 min with metrics |
| Whole Judge, 17 cases on 8 GPUs, oracle | — | — | 3,861 s (controller bound 8,720 s) |
| VBench on 273 frames | — | — | 66 s (4 CPU) / 116 s (2 CPU) |

**Measured peaks** (H100 SXM 80 GB, official oracle in the Judge, six cards):

| Run | GPU peak | Wall |
| --- | --- | --- |
| LongCat case | torch peak 69,174,532,608 B; 71,895 MiB on the card (of 81,559) | stage 1,703–1,733 s |
| HY-WorldPlay-8B case | torch peak 56,134,619,648 B; 57,983 MiB on the card | stage 296–304 s |
| Whole Judge, 17 cases on 6 GPUs | — | 2,613–2,620 s (controller bound 11,240 s) |

**The floor covers the Judge, not the BF16 goldens.** The Judge never runs
BF16: it compares a candidate's frames with goldens generated once, on H100
NVL, and frozen. Only that BF16 generation needs a 94 GB-class card (HY peaks
at 97.4e9 bytes, LongCat at 89.1e9); in the Judge the official codec peaks at
75.4e9 bytes per card. `min_device_memory_bytes` = 79 GiB therefore admits
both H100 types, which share the goldens. On an 80 GB card a candidate has
about 9.4 GiB more than the official codec on a LongCat case (about 23 GiB on
HY) for extra persistent KV or scratch; beyond that it runs out of memory there
but not on a 94 GB card, so an extreme submission can be valid on one type and
not the other. The GPU model name is never a gate; name, UUID and total memory
are recorded as measurement provenance only.

**Allocator settings in the Judge.** LongCat runners default to
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; without it the BF16
10-segment run fragments and fails in segment 2. With `QVG_CUDA_RESERVE=1`
(passed through by the controller together with
`QVG_CUDA_RESERVE_HEADROOM_BYTES`), each runner process instead reserves its
card in one allocation (4 GiB headroom) with the plain allocator
(`expandable_segments:False`), `empty_cache` becomes a no-op and peak
statistics are reset so `peak_cuda_bytes` is real use; this is meant for
GPU-sharing layers such as HAMi that serialise expandable-segment growth
across slots. `judge/judge_shards.py` sets `CUDA_DEVICE_SM_LIMIT=0` for every
slot, which keeps HAMi's utilization watcher from throttling every
`cudaMalloc` when several slots are busy. Neither changes task semantics.

---

## 4. What each deployment captures, and what is drawn per build — by design

**Nothing is redrawn per build.** The hidden suite is a frozen 17-case table
bound by SHA-256 to the release; rebuilding the images never changes a prompt,
an image, a pose, a seed or a golden. The five sampled LongCat prompts were
drawn once, with `random.Random("qvg-v4-longcat-20261006").sample` over the
1000 prompts left after excluding the public #1 and #6 and case #12, and the
result is frozen in the release. The public suite is three fixed zero-weight
diagnostics. Randomness is neither in case selection at build time nor in the
reference: the reference is the set of pre-computed, digest-pinned BF16
goldens.

What a deployment does capture for itself:

| Captured per deployment | Why that is the design, and what it means for comparing scores |
| --- | --- |
| The **artifact root** — public trees re-downloaded at their pinned revisions, the hidden and public evaluation trees derived from them by `scripts/stage_eval_inputs.py` | The public part is too large to vendor; the hidden prompt suite is in-tree on a Judge-only path, the HY case images come from the pinned official repo, and the goldens are recipe-bound captures. `scripts/verify_models.sh` is the only definition of "this root is correct": it pins every tree by revision/commit and every loose weight by sha256, so two deployments that both pass it hold the same scored bytes. |
| The **image IDs** (base, Work, Judge) | The base is built from a digest-pinned public image plus a fully pinned requirement lock, so two deployments get functionally equivalent images, **not byte-identical** ones: apt security point releases move with the build date. None of them touches CUDA, Python or a scored path. The exact image IDs are recorded in the run manifest and in the measurement-profile comparability key. |
| `model-lock.json`, written per run | Proof that the mounted score-affecting surface was the pinned one at the moment of the run. |
| The `measurement_profile` comparability key | Hashes only stable fields: GPU model / compute capability / memory, driver, exact image IDs, software ABI, calibration protocol and the scheduler-issued calibration epoch. The physical UUIDs and observed clocks go to `deployment_attestation` instead, so an audit trail exists without polluting the comparison key. The Judge GPU count is not part of it: it changes wall time only. |

**Consequence for comparing scores.** The score is a pure persistent-byte ratio
with no timing term, summed over every compression call and every case of a
model, so raw scores are comparable across devices, across Judge fleet sizes
and across deployments that share an `evaluation_contract_sha256`. **Validity
is not**: the per-case wall gates are absolute deployment budgets, and the
fidelity/VBench gates are bound to the numerics of the qualification profile
(Hopper + FA3). Moving to a non-Hopper device requires regenerating goldens
and gates with the same protocol; this task does not do that automatically.
Scores of this v4 suite are not comparable with scores of earlier versions of
the task.

---

## 5. Build

Order matters: the Judge image is `FROM` the Work image, which is `FROM` the
base, and `scripts/build_images.sh` builds them in that order, followed by the
host-local Judge holdout layer.

```bash
cd <repo>/tasks/latest/quant-videogen-cross-model-kv-codec-h100-v4-scratch
umask 022
export QVG_ARTIFACT_ROOT=/absolute/path/to/quant-videogen     # see §2
# Leave BASE_WORK_IMAGE unset: the base is built from this repository.
# Behind an egress proxy, also export HTTP_PROXY / HTTPS_PROXY / NO_PROXY.

# 1) accept the artifact root and write the model lock
bash scripts/verify_models.sh

# 2) pure CPU/meta checks (§6.1)
python3 -B -m pytest -p no:cacheprovider tests -q
python3 -B scripts/validate_layout.py .

# 3) base -> work -> judge -> judge holdout layer
bash scripts/build_images.sh

# 4) the fail-closed proxy chain, including a real CLI smoke
RUN_CODEX=1 MODEL=<model> bash scripts/network_preflight.sh

# 5) derive the hidden inputs for the 17 cases (§2), then re-verify the root
python3 scripts/stage_eval_inputs.py --gpus <i,j,...>
bash scripts/verify_models.sh

# 6) only when re-binding the release (new goldens): copy release_values from
#    <root>/hidden/MANIFEST.json into configs/h100_scratch_v4_release.json (the
#    controller refuses a null golden or init digest in any status), refresh
#    the manifests (the suite manifests copy the case table from the release),
#    rebuild (the Judge image bakes the release)
python3 scripts/refresh_portable_manifests.py
bash scripts/build_images.sh

# 7) the trusted positive control on the hidden suite, sharded over the cards
JUDGE_GPUS=<i,j,...> bash scripts/run_oracle.sh
```

What step 3 does, in order:

1. **base** — `images/work/base/Dockerfile` starts at
   `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` pinned by manifest digest and
   installs `images/work/base/requirements.lock` with `pip install --no-deps`.
   Every line of that lock is an exact version (the one URL requirement also
   carries its sha256), so no resolver decision happens at build time.
   `images/work/base/check_environment.py` runs inside the build and fails it
   if the closure does not import or a score-affecting version has drifted. The
   base adds the apt tools the images actually use (`setpriv` and `pkill` for
   the Judge candidate sandbox, `curl` for the network preflight, `ninja-build`
   for `TORCH_EXTENSIONS_DIR`, `ffmpeg`, `git`, `jq`, `ripgrep`), creates the
   `agent` account with no sudoers entry, and copies the public Self-Forcing
   integration and the public metrics. It carries no model weights, no
   reference codec, no starter code and no agent CLI, and `/opt/mlsysforge` is
   a real directory rather than a symlink shim.
2. **work** — `images/work/Dockerfile` builds `FROM` the base and installs the
   read-only compatibility runtime, the public tests, the prompt asset and the
   root-owned cooperative submit hook, then asserts in the same layer that no
   codec reference, evaluator surface, hidden-workload integration or agent
   toolchain is present and that `/workspace/submission` is genuinely empty.
3. **judge** — `images/judge/Dockerfile` builds `FROM` the Work image and
   restores the protected model integrations, runners, metrics and evaluator
   root-only, creates the eight per-slot candidate accounts (UIDs/GIDs
   23001–23008) and makes `/eval` and `/models/hidden` root-only. It carries no
   HY case image; those come from `/eval/hidden`.
4. **judge holdout layer** — `images/judge/Dockerfile.private` builds on the
   public Judge image and adds the Judge-only release config from its own
   build context (the committed `configs/h100_scratch_v4_release.json`, or
   `<INFRATHON_ASSETS_ROOT>/<task directory>/judge/h100_scratch_v4_release.json`
   when `INFRATHON_ASSETS_ROOT` is set).

The script then runs the images once and fails the build unless `codex`,
`node` and `npm` are all absent, and writes `docker image inspect` output to
`<repo>/outputs/<task-directory>/build/images.json`. Keep that file: the image
IDs it records are part of the measurement-profile comparability key.

`BASE_WORK_IMAGE` may name an externally supplied base image, which is then
only validated with `docker image inspect`; leave it unset for the
repository-built base. `scripts/build_images.sh` re-execs itself through
`sg docker` when the current shell is not in the docker group.

---

## 6. Post-build verification and expected bands

### 6.1 Pure CPU (no GPU needed)

```bash
python3 -B -m pytest -p no:cacheprovider tests -q
python3 -B scripts/validate_layout.py .
```

- Tests: run them with the repository venv interpreter; all must pass.
- Layout: `layout OK: <task directory>`; any problem is reported as
  `layout validation failed:` plus a bullet list.

`tests/test_portable_contract.py` recomputes the scientific resource digest,
the delivery/build-resource digest, the manifest byte digests and the real
suite case counts (17) from the release, the evaluator, the Judge adapters and
the policy. `scripts/validate_layout.py` additionally checks that the base is
self-contained: a digest-pinned `FROM`, every lock line pinned, the
base-then-work-then-judge order, the `BASE_WORK_IMAGE` override path, and no
agent toolchain. Treat any mismatch as a hard stop. The task JSON is
deliberately not a member of either resource manifest because it is the root
document binding their digests; each manifest also excludes itself to avoid a
recursive hash.

After editing any file in the build-resource manifest — this file included —
run `python3 -B scripts/refresh_portable_manifests.py` and confirm with
`--check` (no output, exit 0). It recomputes digests and **never** adds or
removes a resource entry: `delivery.build_resource_manifest_sha256` in the task
JSON changes, the scientific `portable.task_core.resource_manifest_sha256` must
not.

### 6.2 Artifact root acceptance

```bash
QVG_ARTIFACT_ROOT=/absolute/path/to/quant-videogen bash scripts/verify_models.sh
```

Writes `<repo>/outputs/<task-directory>/verification/model-lock.json` with
`"status": "verified"` — 8 Hugging Face bundles at their pinned revisions, 4 git
bundles at their pinned commits, 8 loose weights by sha256, 2 mounted trees. It
refuses a root containing a symlink, a non-regular entry, an `.incomplete`
download, a tree whose files change while being hashed, a git checkout whose
`HEAD` is not the pinned commit or moves during the walk, or a loose weight
whose sha256 differs. It also refuses to overwrite an existing output, so
delete the previous file before re-running, and it refuses an output parent
directory that is group- or other-writable (`model lock output parent is
unsafe`) — hence the `umask 022`.

### 6.3 Network chain

```bash
RUN_CODEX=1 MODEL=<model> bash scripts/network_preflight.sh
```

Verifies the whole fail-closed chain — Work has no direct egress, a refused
CONNECT target is answered `403`, an admitted target is reachable through the
restricted proxy — including one real CLI smoke.

### 6.4 Trusted positive control (needs the GPUs)

```bash
JUDGE_GPUS=<i,j,...> bash scripts/run_oracle.sh
```

It packs the repository-private `oracle/` tree as a submission and evaluates
it through `scripts/eval_archive.sh`: the listed cards are resolved to UUIDs,
both host lock files are held for each, each card must be idle and inside the
envelope, and the Judge shards the 17 cases over them. The oracle exists only
on the host side and never enters the Work image; it is admitted by its exact
tree digest.

**While a release is `calibrating`** the run reports `valid=false` with
reason `uncalibrated_release` by design. A complete calibration run has
`outcome=evaluated`, `reason=uncalibrated_release`,
`evaluation_status=uncalibrated`, `candidate_invalid=false` and
**`infrastructure_invalid=true`** (by design: `run_judge.py` refuses an
uncalibrated result without it, so the harness counts such a report as an
infrastructure failure). What matters is that the status is `uncalibrated`
and that the private JSON carries every per-case reading; those readings are
the calibration input. `scripts/run_oracle.sh` checks exactly this.

**On a calibrated release** the run must report `valid=true`,
`timed_out=false`, `candidate_invalid=false`, `infrastructure_invalid=false`
and a score inside the host's quality-file `oracle_calibration.score_band`
(6.55–7.10 for the official Quant-VideoGen oracle on both hardware types). The
per-model byte sums depend only on shapes and on the oracle's deterministic
payload layout, so the compression score (6.8170; LongCat 6.8796, HY 6.7549)
repeats to the last digit on every card. The score itself moves by up to about
±2.2% because VBench scores identical frames differently run to run: four full
runs scored 6.8255 and 6.7782 on 6 x H100 NVL and 6.9846 and 6.6795 on
6 x H100 SXM 80 GB, all against the NVL reference. `bf16_kv_bytes` is
233,098,444,800 per LongCat case and 160,101,826,560 per HY case.

The zero point of the quality term, `models.<model>.quality_reference`, is
measured the same way, once per hardware type: the per-model case means (PSNR,
SSIM, LPIPS and the four VBench dimensions) of two full Judge runs of the
official oracle on that type, averaged, so the oracle sits at `q ≈ 0` there.
It is frozen with the oracle's provenance and score band in the Judge-only
asset `hw/<type>/judge/h100_scratch_v4_quality.json`, not in the release:
the release, its gates and the BF16 goldens are shared by every type.
`scripts/build_images.sh` stages the host's file beside the release in the
Judge holdout layer (`/opt/mlsysforge/configs/h100_scratch_v4_quality.json`,
passed to the controller as `--quality`), and the controller refuses a
calibrated release without one. The two references agree closely (HY fidelity
bit-identical, LongCat within 0.03 dB and 0.5 VBench point).

Wall time on eight cards: about 3,861 s for the whole Judge, against a
controller bound of 8,720 s. Six LongCat cases run side by side (stage
2,691–2,829 s each) while HY cases queue on the other two cards and backfill
the LongCat cards (stage 595–871 s each). On one card the same suite runs
serially; the derived controller bound is 62,940 s (about 17.5 hours).

### 6.5 Calibration (required once per release, needs the GPUs)

The gates are the scientific contract and moving one is an owner decision,
never a deployment-time edit. The rule, implemented by
`scripts/calibrate_gates.py`:

1. Run the trusted oracle on the complete 17-case suite **at least twice** on
   the qualification profile and calibrate from all runs: every reading is the
   worst over runs, with the mean−3σ (or mean+3σ) term applied. Generation is
   not sigma 0 end to end: HY frames repeat bit-for-bit, but LongCat frames
   drift slightly between oracle runs (PSNR changes ≤ 0.09 dB in 5 of 6
   cases), and VBench scores identical frames differently from run to run
   (per-case σ about 0.2–0.35 points on HY, 0.4–0.6 on LongCat; LongCat 6-case
   means move by up to 0.65 points). A single run with fixed margins is not
   enough: it under-states the LongCat VBench mean variation.
2. **Mean gates (per model, on the case means):** PSNR −0.25 dB, SSIM
   −0.0075, LPIPS +0.0075; each VBench dimension's allowed drop is the
   observed drop of the oracle's case mean below the mean BF16 VBench (at
   least 0, worst / 3σ over runs) plus 1.0 point.
3. **Per-case floors (per model, from the worst case over runs):** PSNR −1.0 dB, SSIM
   −0.03, LPIPS +0.03; each VBench dimension's allowed drop is
   2 × (worst per-case drop, at least 0, + 0.25).
4. Write each case's `bf16_vbench` (VBench on its golden) and `bf16_kv_bytes`
   (from the codec-mode oracle profile; shapes only) into the release. Never
   copy `bf16_kv_bytes` from a BF16 golden profile: for HY the BF16 mode
   counts the post-RoPE/PRoPE cache plus the BF16 window, not the `encode`
   inputs, so its number differs.
5. Controls through the full Judge as ordinary candidates (an RTN-INT2 codec
   should be rejected, an RTN-INT4 codec should pass) are optional and are not
   part of the shipped release.
6. `python3 scripts/calibrate_gates.py <qvg-private.json> <qvg-private.json> [...] --write` flips the
   release to `calibrated` (it accepts only complete runs of a `calibrating`
   release); then refresh the suite manifests and portable digests and
   rebuild the Judge.

Oracle readings (case means) and the current gates, rounded; the release holds
the exact values:

| Model | Reading (case mean) | Layer | PSNR ≥ | SSIM ≥ | LPIPS ≤ | VBench allowed drop AQ / BG / IQ / SC |
| --- | --- | --- | ---: | ---: | ---: | --- |
| LongCat (6) | PSNR 23.66, SSIM 0.870, LPIPS 0.096 | mean | 23.387 | 0.8627 | 0.1039 | 1.873 / 2.685 / 6.174 / 4.196 |
| LongCat (6) | | case floor | 18.261 | 0.7119 | 0.1620 | 7.420 / 6.365 / 28.659 / 10.570 |
| HY-WorldPlay-8B (11) | PSNR 31.06, SSIM 0.945, LPIPS 0.046 | mean | 30.805 | 0.9372 | 0.0534 | 2.055 / 1.511 / 2.415 / 1.984 |
| HY-WorldPlay-8B (11) | | case floor | 24.700 | 0.8357 | 0.1304 | 7.289 / 2.773 / 7.342 / 4.528 |

Over ten LongCat continuation segments the oracle's VBench drifts per case in
both directions (for example `lc-320` image_quality 62.5 → 49.5, `lc-893`
63.8 → 66.7), so the LongCat per-case VBench floors are loose (image_quality
allows a drop of about 29 points) and the mean VBench gate (about 6.2 for
image_quality) is the meaningful one. The LongCat aesthetic mean gate is 1.87:
the 1.0-point margin plus the worst drop over the calibration runs (worst /
3σ).

**HY rollout determinism.** `hyvideo/generate.py` calls
`seed_everything(args.seed)` before `generate_video()`, mirroring LongCat.
Without it the autoregressive rollout ranks history clips by a Monte-Carlo FOV
estimate drawn from an unseeded `torch.rand`, so near-ties swap memory clips
between runs while the KV shapes and byte counts stay identical.
`tests/test_hy_rollout_determinism.py` pins this.

---

## 7. Formal run

```bash
launch/quant-videogen/run.sh --model <model> --effort <level> \
  --work-gpu <G> --judge-gpus <A,B,...> --judge-port <port> \
  [--hours N] [--agent codex|claude-code] [--dry-run] [--tmux]
```

`launch/quant-videogen/run.sh` normalises the shared CLI knobs and calls
`scripts/run_agent.sh`, which owns the GPU locks, preflight, image and model
checks, the sidecar network, the Judge server and the independent final Judge.
Its contract:

- Work gets exactly one GPU; every interactive Judge and the independent final
  Judge run the complete 17-case hidden suite spread over the Judge GPUs
  (`QVG_JUDGE_GPU_UUIDS`). Both host lock names of every GPU are held for the
  whole run, final included.
- The Judge's CPU/memory share and its timeout come from
  `scripts/lib.sh judge_plan` for the Judge GPU count; the task JSON's
  `judge.eval_timeout` (9000 s) is sized for eight GPUs, and
  `scripts/sforge_agent_entry.py` carries the planned timeout into the Judge
  server and the Work-side `sforge-submit` wait.
- Automatic evaluation is off; only a manual `sforge-submit` triggers the
  Judge, with a 300 s submission cooldown. Every scored submission runs the
  complete 17-case suite (about an hour on eight cards); the archive at the
  deadline is judged once more on the same suite.
- The default agent budget is 24 hours (`TASK_TIMEOUT=86400`; `--hours` /
  `--timeout` override it).
- Work sits on an internal Docker network whose only way out is a restricted
  CONNECT sidecar. The task JSON sets `internet: true`, so public hosts on
  ports 80/443 are allowed and internal addresses are not.
- Agents: `codex` (default; `CODEX_AUTH_JSON_PATH`, or an OpenAI-compatible
  `MODEL_API_BASE_URL` plus a mode-0600 `MODEL_API_KEY_FILE` relayed by the
  sidecar) or `claude-code` (`--claude-token`). `QVG_ARTIFACT_ROOT` and
  `SFORGE_AGENT_RUNTIME_DIR` must be set, and the `SFORGE_LLM_JUDGE_*`
  integrity review must be configured with `SFORGE_LLM_JUDGE_MODE=enforce`;
  the run refuses otherwise.
- `--dry-run` (`QVG_DRY_RUN=1`) resolves and prints the plan without taking a
  lock or starting a container.

Evaluate one archive without an agent with
`JUDGE_GPUS=<i,j,...> scripts/eval_archive.sh ARCHIVE.tar.gz`, and replay saved
submissions of an earlier run with `launch/quant-videogen/replay.sh`
(delegates to `scripts/replay_candidates.sh`).

The time limits have limited headroom under eight-way concurrency: the
per-case stage budgets are HY 1500 s and LongCat 4500 s, about 1.6–1.7x the
slowest oracle case with eight cases running at once; the 8-GPU controller
bound is 8720 s, inside `eval_timeout` 9000 s.

---

## 8. Run artifacts, success criteria and triage

### 8.1 Where things land

A local `scripts/eval_archive.sh` (or `scripts/run_oracle.sh`) evaluation
writes under

```text
<repo>/outputs/quant-videogen-cross-model-kv-codec-h100-v4-scratch/runs/<RUN_ID>/
```

with the SForge report (aggregate `valid` / `score` only) and
`private_judge_artifacts.tar.gz`, which holds the Judge's root-private result
JSON: per-model sums and checks, every case's readings, its slot, UID, GPU
UUID, port, timings and log paths, the schedule makespan and its bound, and
`first_error` on failure.

### 8.2 Success criteria

A complete evaluation means the controller ran every `case_id` in the release
exactly once and wrote its private JSON. `valid` is a **separate** question:
it says whether the candidate was accepted. A candidate-level failure reports
`candidate_invalid = true` and `infrastructure_invalid = false`. Before a score
counts, both models must clear their mean gates (PSNR, SSIM, LPIPS and the
four VBench drops) and every case must clear its floors and its wall gate
(LongCat 4500 s, HY 1500 s), on top of `source_admission`,
`candidate_isolation`, `hardware_envelope`, the frozen `bf16_kv_bytes`,
`other_device_allocated_bytes == 0`, `cuda_device_index` equal to the slot's
device and a full `calls` count. Two runs are comparable when their
`evaluation_contract_sha256` and `measurement_profile.comparability_key_sha256`
agree.

### 8.3 What to look at first on failure

| Symptom | Look here first |
| --- | --- |
| It never started | the script's own output: a GPU lock already held, a card not idle or outside the envelope, mixed GPU models, an unverified artifact root, stale images |
| `model lock output parent is unsafe` | the launching shell's umask; restart with `umask 022` |
| `infrastructure_invalid` | the private JSON's `first_error` and the failing case's control record: input hashes, isolation, a metric child, the controller budget, or a cgroup OOM |
| `candidate_invalid` | the failing case's record: non-zero exit, stage timeout, output tree, `bf16_kv_bytes` mismatch, memory left on another GPU, or a quality gate |
| A LongCat case runs out of memory | check that `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (or `QVG_CUDA_RESERVE=1`) reached the runner; without either the BF16 10-segment run fragments and fails in segment 2 |
| Every allocation or model load crawls under a GPU-sharing layer | HAMi's utilization watcher: check that the slot environment has `CUDA_DEVICE_SM_LIMIT=0` |
| A wall gate fails only with many cards | CPU, host RAM or read contention between concurrent cases; compare the case's wall with a one-card run before blaming the candidate |
| A trusted metric returned an error | an intermittent `infrastructure_error` from a metric child is possible; re-running the same archive normally comes back valid |
| The score looks wrong but the run is valid | check that the images are current, then re-verify the artifact root |

### 8.4 What to re-measure on another GPU

1. **Re-measure the envelope.** Sample NVML `memory.used` at 1 Hz on every
   Judge card through one full trusted evaluation, and on the Work card through
   one full public run. The floor in the release, the task JSON profile and
   `scripts/lib.sh` move together, and the images are rebuilt.
2. **Do not re-measure the score.** It is a device- and fleet-size-independent
   byte ratio; nothing in it needs recalibration for a new card.
3. **Same model:** change the GPU list; keep the artifact root and the images
   unchanged for the whole evaluation (the Judge re-verifies the full
   score-affecting mount surface before and after every evaluation).
4. **Different model, still inside the envelope:** the goldens and gates are
   calibrated on Hopper + FA3 numerics, and the upstream attention layer
   switches FA3/FA2 on a GPU name containing `h100`/`hopper`. A non-Hopper
   device requires regenerating all 17 goldens **and** recalibrating with the
   same protocol (§6.5).
5. **A much slower card or a busier host:** `models.<m>.stage_seconds` is a
   deployment budget, not a calibrated value. Raising it changes no scored
   input or quality gate, but the controller bound is derived from it, and the
   task JSON `judge.eval_timeout` must stay at or above the eight-card bound.
6. **Anything that changes the comparability key** — GPU model, driver, an
   image ID, the software ABI, the calibration protocol or the campaign epoch —
   creates a new measurement profile even when the portable TaskCore digest is
   unchanged. Record the new key with the run.
