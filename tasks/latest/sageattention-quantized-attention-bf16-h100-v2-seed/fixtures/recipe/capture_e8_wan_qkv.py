#!/usr/bin/env python3
"""Capture the e8 real-activation edge fixture named by ``RECIPE.json``.

The script is self-contained by construction: it imports ``torch`` and the
pinned upstream Wan2.1 checkout that ``build_fixture.sh`` places on the path,
and nothing else.  It never reads this repository, another task, a calibration
lab tree or any absolute host path -- every location is a command-line argument
and every scientific constant comes from ``RECIPE.json``.

What it reproduces (faithful port of the original 2026-09-01 lab capture):

  * ``WanT2V.generate``'s UniPC branch, offload-free: 50 steps, shift 5.0,
    guide scale 5.0, 832x480, 81 frames, ``seq_len`` 32760, seed 4100, the
    recipe's prompt.
  * ``WanModel.forward`` (t2v path) inlined so the block loop is visible.
  * The capture hook: ``wan.modules.model.flash_attention`` is wrapped, and a
    forward-pre/post hook pair on every ``block.self_attn`` records which block
    is executing.  The wrapper therefore sees exactly the post-RoPE, qk-normed
    Q/K/V of one self-attention call, in ``[B, L, H, D]`` layout.
  * The slice: the full ``[1, 32760, 12, 128]`` capture is copied to the host in
    bfloat16, then the first 8192 tokens are cloned out.

Only denoising step 0 is needed, so the pipeline aborts as soon as the
block-15 / step-0 / conditional Q/K/V have been copied to the host.  Nothing
executed after that point can influence them, so no scheduler step, no
unconditional branch and no VAE decode ever runs.

Usage (see ``build_fixture.sh``; this is expected to run inside the builder
container with exactly one visible GPU)::

    capture_e8_wan_qkv.py --recipe RECIPE.json --wan-src /opt/wan2.1 \\
        --checkpoint /models/wan21-t2v-1.3b --output-dir /out
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import sys
import time
from typing import Any

import torch


class CaptureComplete(Exception):
    """Raised to abort the pipeline once the requested Q/K/V are on the host."""


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def log(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, default=str), flush=True)


def driver_version() -> str | None:
    """Best-effort NVIDIA driver version for the capture manifest."""

    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        value = result.stdout.strip().splitlines()[0].strip()
        if value:
            return value
    except Exception:  # pragma: no cover - informational only
        pass
    try:
        return str(torch._C._cuda_getDriverVersion())
    except Exception:  # pragma: no cover - informational only
        return None


def verify_checkpoint(checkpoint: pathlib.Path, recipe: dict) -> None:
    """Fail closed unless every pinned snapshot file is byte-for-byte right."""

    for name, expected in sorted(recipe["model"]["files"].items()):
        path = checkpoint / name
        if not path.is_file():
            raise RuntimeError(f"pinned checkpoint file is missing: {name}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"pinned checkpoint file {name} has SHA-256 {actual}, "
                f"recipe requires {expected}"
            )
    log(
        "checkpoint_verified",
        repo=recipe["model"]["repo"],
        revision=recipe["model"]["revision"],
        files=len(recipe["model"]["files"]),
    )


class CaptureManager:
    """Wraps Wan2.1's attention entry point and stops at the recipe's call."""

    def __init__(self, capture: dict, tokens: int) -> None:
        self.block_index = int(capture["block_index"])
        self.step_index = int(capture["step_index"])
        self.branch = str(capture["cfg_branch"])
        self.tokens = tokens
        self.call_context: dict[str, Any] | None = None
        self.active_block: int | None = None
        self.grid_sizes: torch.Tensor | None = None
        self.seq_lens: torch.Tensor | None = None
        self.payload: dict[str, torch.Tensor] | None = None
        self.original_flash = None

    def install(self, wan_model_module, blocks) -> None:
        self.original_flash = wan_model_module.flash_attention

        def wrapped_flash(*args, **kwargs):
            q = kwargs.get("q", args[0] if len(args) > 0 else None)
            k = kwargs.get("k", args[1] if len(args) > 1 else None)
            v = kwargs.get("v", args[2] if len(args) > 2 else None)
            ctx = self.call_context
            if (
                ctx is not None
                and self.active_block == self.block_index
                and int(ctx["step_index"]) == self.step_index
                and str(ctx["branch"]) == self.branch
            ):
                self.save(q, k, v)
                raise CaptureComplete()
            return self.original_flash(*args, **kwargs)

        wan_model_module.flash_attention = wrapped_flash

        for block_index, block in enumerate(blocks):
            def pre_hook(module, args, kwargs=None, index=block_index):
                self.active_block = index
                # WanAttentionBlock calls self_attn(x, seq_lens, grid_sizes, freqs).
                kw = kwargs or {}
                self.seq_lens = kw.get("seq_lens", args[1] if len(args) > 1 else None)
                self.grid_sizes = kw.get("grid_sizes", args[2] if len(args) > 2 else None)

            def post_hook(module, args, output, index=block_index):
                if self.active_block == index:
                    self.active_block = None

            block.self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
            block.self_attn.register_forward_hook(post_hook)

    def save(self, q, k, v) -> None:
        # The copies synchronize this capture with the GPU.  Store exact values
        # as passed into attention, standardized to the fixture's bfloat16.
        self.payload = {
            name: tensor.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
            for name, tensor in (("q", q), ("k", k), ("v", v))
        }


def forward_t2v(model, sinusoidal_embedding_1d, amp, x, t, context, seq_len):
    """Faithful inline port of ``WanModel.forward`` (t2v path)."""

    device = model.patch_embedding.weight.device
    if model.freqs.device != device:
        model.freqs = model.freqs.to(device)

    x = [model.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x
    ])

    with amp.autocast(dtype=torch.float32):
        e = model.time_embedding(sinusoidal_embedding_1d(model.freq_dim, t).float())
        e0 = model.time_projection(e).unflatten(1, (6, model.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

    context_lens = None
    context = model.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(model.text_len - u.size(0), u.size(1))])
            for u in context
        ])
    )

    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=model.freqs,
        context=context,
        context_lens=context_lens,
    )

    for block in model.blocks:
        x = block(x, **kwargs)

    x = model.head(x, e)
    x = model.unpatchify(x, grid_sizes)
    return [u.float() for u in x]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, type=pathlib.Path)
    parser.add_argument("--wan-src", required=True, type=pathlib.Path)
    parser.add_argument("--checkpoint", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    args = parser.parse_args(argv)

    recipe_bytes = args.recipe.read_bytes()
    recipe = json.loads(recipe_bytes.decode("utf-8"))
    recipe_sha256 = hashlib.sha256(recipe_bytes).hexdigest()
    capture = recipe["capture"]
    tensors = recipe["tensors"]
    tokens = int(tensors["shape"][1])

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected exactly one visible GPU, got {torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(0)
    driver = driver_version()
    log(
        "environment",
        python=sys.version.split()[0],
        torch=torch.__version__,
        cuda=torch.version.cuda,
        driver=driver,
        device_name=properties.name,
        compute_capability=f"{properties.major}.{properties.minor}",
        recipe_id=recipe["recipe_id"],
        recipe_sha256=recipe_sha256,
    )

    verify_checkpoint(args.checkpoint, recipe)

    wan_src = str(args.wan_src)
    if wan_src not in sys.path:
        sys.path.insert(0, wan_src)
    import torch.cuda.amp as amp
    import wan.modules.attention as wan_attention_module
    import wan.modules.model as wan_model_module
    from wan.configs import WAN_CONFIGS
    from wan.modules.model import WanModel, sinusoidal_embedding_1d
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae import WanVAE
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    for forbidden in recipe["builder"]["forbidden_packages"]:
        if forbidden == "flash_attn_interface" and wan_attention_module.FLASH_ATTN_3_AVAILABLE:
            raise RuntimeError(
                "flash_attn_interface (FA3) is installed; wan attention would "
                "silently prefer it and the capture would not follow the recipe"
            )
    if not wan_attention_module.FLASH_ATTN_2_AVAILABLE:
        raise RuntimeError("flash_attn (FA2) is required by the recipe and is absent")

    config = WAN_CONFIGS[capture["wan_task"]]
    size = tuple(capture["size_wh"])
    frame_num = int(capture["frame_num"])
    sampling_steps = int(capture["sampling_steps"])
    shift = float(capture["shift"])
    seed = int(capture["seed"])
    prompt = capture["prompt"]

    checkpoint = str(args.checkpoint)
    started = time.perf_counter()
    text_encoder = T5EncoderModel(
        text_len=config.text_len,
        dtype=config.t5_dtype,
        device=torch.device("cpu"),
        checkpoint_path=os.path.join(checkpoint, config.t5_checkpoint),
        tokenizer_path=os.path.join(checkpoint, config.t5_tokenizer),
        shard_fn=None,
    )
    text_encoder.model.to(device)
    vae = WanVAE(vae_pth=os.path.join(checkpoint, config.vae_checkpoint), device=device)
    model = WanModel.from_pretrained(checkpoint)
    model.eval().requires_grad_(False)
    model.to(device)
    if len(model.blocks) != int(capture["num_blocks"]):
        raise RuntimeError(
            f"model has {len(model.blocks)} blocks, recipe requires "
            f"{capture['num_blocks']}"
        )
    log("models_loaded", seconds=time.perf_counter() - started)

    manager = CaptureManager(capture, tokens)
    manager.install(wan_model_module, model.blocks)

    # ---- faithful port of WanT2V.generate (unipc branch, offload-free) ----
    F = frame_num
    target_shape = (
        vae.model.z_dim,
        (F - 1) // config.vae_stride[0] + 1,
        size[1] // config.vae_stride[1],
        size[0] // config.vae_stride[2],
    )
    seq_len = math.ceil(
        (target_shape[2] * target_shape[3])
        / (config.patch_size[1] * config.patch_size[2])
        * target_shape[1]
    )
    if seq_len != int(capture["seq_len"]):
        raise RuntimeError(f"seq_len {seq_len} != recipe {capture['seq_len']}")
    if int(capture["step_index"]) != 0:
        raise RuntimeError(
            "this capture script only supports denoising step 0: the pipeline is "
            "aborted at the end of the first step so that no work which cannot "
            "influence the fixture is ever executed"
        )

    n_prompt = config.sample_neg_prompt

    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)

    context = text_encoder([prompt], device)
    context_null = text_encoder([n_prompt], device)

    noise = [
        torch.randn(
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=torch.float32,
            device=device,
            generator=seed_g,
        )
    ]

    aborted = False
    generate_started = time.perf_counter()
    with amp.autocast(dtype=config.param_dtype), torch.no_grad():
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=config.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        sample_scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
        timesteps = sample_scheduler.timesteps
        timesteps_float = [float(t) for t in timesteps]

        latents = noise
        arg_c = {"context": context, "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}

        try:
            for step_index, t in enumerate(timesteps):
                latent_model_input = latents
                timestep = torch.stack([t])
                base_ctx = {
                    "step_index": step_index,
                    "total_steps": sampling_steps,
                    "timestep": timesteps_float[step_index],
                }
                for branch, arg in (("cond", arg_c), ("uncond", arg_null)):
                    manager.call_context = dict(base_ctx, branch=branch)
                    try:
                        forward_t2v(
                            model,
                            sinusoidal_embedding_1d,
                            amp,
                            latent_model_input,
                            timestep,
                            arg["context"],
                            arg["seq_len"],
                        )
                    finally:
                        manager.call_context = None
                        manager.active_block = None
                raise RuntimeError(
                    "the recipe's capture point was never reached; the pipeline "
                    "must abort inside the first denoising step"
                )
        except CaptureComplete:
            aborted = True
    if not aborted or manager.payload is None:
        raise RuntimeError("capture did not produce a payload")
    log("capture_reached", seconds=time.perf_counter() - generate_started)

    full_shape = tuple(manager.payload["q"].shape)
    if list(full_shape) != list(tensors["capture_shape"]):
        raise RuntimeError(
            f"captured shape {full_shape} != recipe {tensors['capture_shape']}"
        )

    timestep_value = timesteps_float[int(capture["step_index"])]
    if timestep_value != float(capture["timestep_value"]):
        raise RuntimeError(
            f"timestep {timestep_value} != recipe {capture['timestep_value']}"
        )

    payload: dict[str, Any] = {
        name: manager.payload[name][:, :tokens].clone().contiguous()
        for name in tensors["names"]
    }
    meta = {
        "prompt": prompt,
        "prompt_index": int(capture["prompt_index"]),
        "block_index": int(capture["block_index"]),
        "step_index": int(capture["step_index"]),
        "timestep_value": timestep_value,
        "grid_sizes": (
            manager.grid_sizes.detach().cpu().tolist()
            if manager.grid_sizes is not None else None
        ),
        "seq_lens": (
            manager.seq_lens.detach().cpu().tolist()
            if manager.seq_lens is not None else None
        ),
        "cfg_branch": str(capture["cfg_branch"]),
        "conditional": str(capture["cfg_branch"]) == "cond",
        "capture_time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "synthetic_drive": False,
        "tensor_layout": tensors["layout"],
        "tensor_dtype": tensors["dtype"],
        "post_rope_qk": bool(capture["post_rope_qk"]),
        "qk_norm": bool(capture["qk_norm"]),
        "sampling_steps": sampling_steps,
        "size_wh": list(size),
        "frame_num": frame_num,
        "seed": seed,
        "slice": tensors["slice"],
        "recipe_id": recipe["recipe_id"],
        "recipe_sha256": recipe_sha256,
        "model_repo": recipe["model"]["repo"],
        "model_revision": recipe["model"]["revision"],
        "upstream_commit": recipe["upstream"]["commit"],
    }
    payload["meta"] = meta

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = args.output_dir / recipe["fixture"]["name"]
    manifest_path = args.output_dir / recipe["fixture"]["manifest"]
    for path in (fixture_path, manifest_path):
        if path.exists():
            raise RuntimeError(f"refusing to overwrite existing output: {path}")
    torch.save(payload, fixture_path)

    fixture_sha256 = sha256_file(fixture_path)
    manifest = {
        "schema_version": 1,
        "fixture": recipe["fixture"]["name"],
        "sha256": fixture_sha256,
        "size_bytes": fixture_path.stat().st_size,
        "recipe_id": recipe["recipe_id"],
        "recipe_sha256": recipe_sha256,
        "captured_utc": meta["capture_time_utc"],
        "gpu_name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "driver_version": driver,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "model_repo": recipe["model"]["repo"],
        "model_revision": recipe["model"]["revision"],
        "upstream_commit": recipe["upstream"]["commit"],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    statistics = {}
    for name in tensors["names"]:
        value = payload[name].float()
        statistics[name] = {
            "shape": list(payload[name].shape),
            "rms": float(value.square().mean().sqrt()),
            "abs_max": float(value.abs().max()),
            "finite_fraction": float(torch.isfinite(value).to(torch.float64).mean()),
        }
    log(
        "complete",
        fixture=str(fixture_path),
        sha256=fixture_sha256,
        size_bytes=manifest["size_bytes"],
        statistics=statistics,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
