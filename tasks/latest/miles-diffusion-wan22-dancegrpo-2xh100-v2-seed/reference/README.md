# Reference implementation (host only; never in an image)

The reference solution is kept as patches against the task's upstream bases; `scripts/build_images.sh`
never copies this directory.

- `miles_diffusion_wan.patch`: the reference changes to miles_diffusion, including `task_entry.py` and
  `engine_overlay/` (a copy of the engine overlay). Apply it to a `git archive 29fde6ee` export of miles_diffusion.
- `sglang_wan.patch`: the engine overlay source. Apply it to a `python/` export of sglang-miles-h3 `859a278e`.
- Qualification runs the reference as a submission archive (`repo/` + `notes.md`) through the real SForge evaluation:
  `scripts/rescore.py --archive <ref.tar.gz> --label <label>`.
- H100 NVL (94 GB): `miles_diffusion_wan_h100.patch` applies on top of the reference tree (`git apply` after
  `miles_diffusion_wan.patch`). It changes only where memory goes, not what is sampled, scored or trained:
  activation checkpointing in the FSDP trainer, the training cache returned before each `serve` publication, and the
  colocated HPSv3 worker's weights packed after loading and its scoring cache returned after each call. Without it the
  reference runs out of memory on 94 GB cards (replay: trainer ~79 GiB + Judge engine 15.4 + HPSv3 21.7 per card); with
  it the per-card peak is 87780 MiB of 95830. Its qualification record is `runtime/qualification.json`.
- H100 SXM (80 GB): `miles_diffusion_wan_h100_80g.patch` applies on top of the H100 NVL tree (after
  `miles_diffusion_wan_h100.patch`). It again changes only where memory goes: in `run` the run's own engine keeps the UMT5
  text encoder in host memory between prompts (`--sglang-cpu-offload-components text_encoder`), and on cards below
  90 GiB the colocated HPSv3 worker's weights leave the GPU between scorings (in `run` reloaded from the checkpoint,
  in `serve` held in one registered host buffer). Per-card peaks are 79993 MiB (RUN) and 72012 MiB (timed SERVE
  replays) of 81559. Its qualification record is `runtime/qualification.dual-cuda-sm90-nccl-80g.json`.
