#!/usr/bin/env python3
"""Capture the e7 edge fixture: one real Wan2.1-T2V-1.3B QAD stage-1 training step.

The fixture is no longer bound by the SHA-256 of one host's bytes.  It is bound by
this recipe: every deployment runs this script once at build time and keeps its own
capture.  `RECIPE.json` is the frozen identity (model revision, upstream commit,
prompt, seeds, geometry, block, sigma, slicing and the sanity bands); this file is
the only executable part of it.

Self-containment rule (docs/SOURCE_PROVENANCE.md): the script may import the pinned
upstream checkout and nothing else.  No task module, no lab directory, no absolute
host path.  Everything else it needs -- the two monkeypatches and the latent
sampler -- is re-implemented here from the recipe description.

What it reproduces, in two stages.

Stage 1, the latent `x0` the training step starts from.  A flow-matching sample of
the pinned model itself on the capture prompt: UniPC-30, shift 3.0, classifier-free
guidance 5.0, seed 4100, at the recipe geometry.  Self-attention runs with a
float64 real-arithmetic rotary embedding (bit-comparable with upstream's complex
`rope_apply`, and the form the reference capture used) and torch SDPA in bf16; the
cross-attention path is routed to the same SDPA call, so no fused attention package
is required or admitted.

Stage 2, the training step itself, a FastVideo-style flow-matching step with the
weighting scheme pinned instead of sampled:

    sigma  = recipe.training_step.sigma          (0.5; scheduler index 500)
    t      = recipe.training_step.timestep       (500.0)
    eps    ~ N(0, 1), generator seeded at recipe.training_step.noise_seed
    x_t    = (1 - sigma) * x0 + sigma * eps      bf16, requires_grad
    pred   = WanModel(x_t, t, context)           autocast bf16, full checkpointing
    target = eps - x0                            (precondition_outputs = False)
    loss   = F.mse_loss(pred.float(), target.float())
    loss.backward()

In the self-attention of the captured block, `q, k` (post qk-RMSNorm, post rotary
embedding), `v` and the attention output `o` are recorded on the forward, and a
tensor hook on `o` records `do = dL/do` on the backward.  The hook writes into a
separate dict because non-reentrant activation checkpointing runs the block forward
twice: the second run replaces the recorded activations, while the hook that fires
belongs to the first, graph-carrying run.

The first `kept_tokens` query rows (and the same key/value rows) are kept, in the
model's own `[B, S, H, D]` bf16 layout.  The runtime permutes to `[B, H, S, D]`.

Usage (build_fixture.sh drives this inside the builder container):

  python capture_e7_wan_grad.py \
      --recipe RECIPE.json \
      --upstream /opt/upstream/Wan2.1 \
      --snapshot /models/Wan2.1-T2V-1.3B \
      --out fixtures/e7_wan13b_grad_b29_t1_s8192.pt \
      --manifest fixtures/e7_wan13b_grad_b29_t1_s8192.manifest.json
"""

from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_util


# --------------------------------------------------------------------------- helpers ---
def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(tensor: torch.Tensor) -> dict[str, float]:
    """The three quantities the sanity bands are expressed in."""
    wide = tensor.detach().float()
    finite = torch.isfinite(wide)
    return {
        "rms": float(wide.masked_fill(~finite, 0.0).pow(2).mean().sqrt()),
        "abs_max": float(wide.masked_fill(~finite, 0.0).abs().max()),
        "finite_fraction": float(finite.float().mean()),
    }


def check_band(name: str, measured: dict[str, float], band: dict[str, float]) -> None:
    if not band["rms_min"] <= measured["rms"] <= band["rms_max"]:
        raise SystemExit(
            f"{name}: rms {measured['rms']:.6e} outside "
            f"[{band['rms_min']:.6e}, {band['rms_max']:.6e}]"
        )
    if not band["abs_max_min"] <= measured["abs_max"] <= band["abs_max_max"]:
        raise SystemExit(
            f"{name}: abs_max {measured['abs_max']:.6e} outside "
            f"[{band['abs_max_min']:.6e}, {band['abs_max_max']:.6e}]"
        )
    if measured["finite_fraction"] < band["finite_fraction_min"]:
        raise SystemExit(
            f"{name}: finite fraction {measured['finite_fraction']:.6f} below "
            f"{band['finite_fraction_min']:.6f}"
        )


def verify_snapshot(snapshot: Path, recipe: dict) -> None:
    """Every declared model file must be present with the declared digest."""
    for relative, expected in sorted(recipe["model"]["files"].items()):
        path = snapshot / relative
        if not path.is_file():
            raise SystemExit(f"model snapshot is missing {relative}: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise SystemExit(f"model file {relative} has sha256 {actual}, expected {expected}")
    log(f"model snapshot verified: {len(recipe['model']['files'])} file(s)")


def verify_upstream(upstream: Path, recipe: dict) -> None:
    """The upstream checkout must be the pinned commit."""
    expected = recipe["upstream"]["commit"]
    stamp = upstream / "UPSTREAM_COMMIT"
    if stamp.is_file():
        actual = stamp.read_text(encoding="utf-8").strip()
    else:
        head = (upstream / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            actual = (upstream / ".git" / head[5:]).read_text(encoding="utf-8").strip()
        else:
            actual = head
    if actual != expected:
        raise SystemExit(f"upstream checkout is at {actual}, expected {expected}")
    log(f"upstream checkout verified: {expected}")


# ------------------------------------------------------------------- upstream patches ---
def rope_apply_real(x, grid_sizes, freqs):
    """Real-arithmetic float64 equivalent of `wan.modules.model.rope_apply`.

    `x` is `[B, S, N, D]`; `freqs` is the complex table upstream builds, already
    split-concatenated as (t | h | w).  Upstream multiplies complex64 pairs; this
    does the identical rotation in float64 and returns float32.  The reference
    capture's latent was sampled through this form, so the recipe pins it.
    """
    n, c = x.size(2), x.size(3) // 2
    f_t, f_h, f_w = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        length = f * h * w
        table = torch.cat(
            [
                f_t[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                f_h[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                f_w[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(length, 1, c)
        cos = table.real.to(torch.float64)
        sin = table.imag.to(torch.float64)
        head = x[i, :length].to(torch.float64).reshape(length, n, c, 2)
        real, imaginary = head[..., 0], head[..., 1]
        rotated = torch.stack(
            [real * cos - imaginary * sin, real * sin + imaginary * cos], dim=-1
        ).flatten(2)
        if length < x.size(1):
            rotated = torch.cat([rotated, x[i, length:].to(torch.float64)])
        output.append(rotated)
    return torch.stack(output).float()


def sdpa_bshd(q, k, v):
    """Non-causal scaled dot-product attention on `[B, S, H, D]` tensors."""
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    )
    return out.transpose(1, 2).contiguous()


def install_sampling_patches(wan_model_module, dtype):
    """Stage-1 patches: float64 real rotary embedding, every attention through SDPA."""

    def patched_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0,
                          softmax_scale=None, q_scale=None, causal=False,
                          window_size=(-1, -1), deterministic=False,
                          dtype_=torch.bfloat16, version=None):
        assert not causal and window_size == (-1, -1)
        return sdpa_bshd(q.to(dtype), k.to(dtype), v.to(dtype))

    def patched_self_attention(self, x, seq_lens, grid_sizes, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d).to(dtype)
        q = rope_apply_real(q, grid_sizes, freqs).to(dtype)
        k = rope_apply_real(k, grid_sizes, freqs).to(dtype)
        return self.o(sdpa_bshd(q, k, v).flatten(2))

    wan_model_module.flash_attention = patched_attention
    wan_model_module.WanSelfAttention.forward = patched_self_attention


def install_capture_patches(wan_model_module, dtype, record):
    """Stage-2 patches: upstream rotary embedding, SDPA, and the recorder on one block.

    `record(kind, index, q, k, v, o)` is called inside the self-attention and
    cross-attention forwards with the tensors in `[B, S, H, D]`.
    """

    def patched_self_attention(self, x, seq_lens, grid_sizes, freqs):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d).to(dtype)
        q = wan_model_module.rope_apply(q, grid_sizes, freqs).to(dtype)
        k = wan_model_module.rope_apply(k, grid_sizes, freqs).to(dtype)
        o = sdpa_bshd(q, k, v)
        record("self", getattr(self, "_block_index", None), q, k, v, o)
        return self.o(o.flatten(2))

    def patched_cross_attention(self, x, context, context_lens):
        b, n, d = x.size(0), self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(b, -1, n, d).to(dtype)
        k = self.norm_k(self.k(context)).view(b, -1, n, d).to(dtype)
        v = self.v(context).view(b, -1, n, d).to(dtype)
        o = sdpa_bshd(q, k, v)
        record("cross", getattr(self, "_block_index", None), q, k, v, o)
        return self.o(o.flatten(2))

    wan_model_module.WanSelfAttention.forward = patched_self_attention
    wan_model_module.WanT2VCrossAttention.forward = patched_cross_attention


def enable_full_checkpointing(model):
    """`enable_gradient_checkpointing_type: full`: every transformer block recomputes."""
    for block in model.blocks:
        original = block.forward

        def wrapped(*args, _original=original, **kwargs):
            if torch.is_grad_enabled():
                return checkpoint_util.checkpoint(
                    _original, *args, use_reentrant=False, **kwargs
                )
            return _original(*args, **kwargs)

        block.forward = wrapped


# ------------------------------------------------------------------------------ stages ---
@torch.no_grad()
def sample_latent(model, context, context_null, recipe, sequence_length, device):
    """Stage 1: the flow-matching sample the training step starts from."""
    from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

    spec = recipe["latent"]
    generator = torch.Generator(device=device)
    generator.manual_seed(int(spec["seed"]))
    latent = torch.randn(
        *spec["shape"], dtype=torch.float32, device=device, generator=generator
    )
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000, shift=1, use_dynamic_shifting=False
    )
    scheduler.set_timesteps(int(spec["steps"]), device=device, shift=float(spec["shift"]))
    guidance = float(spec["guidance_scale"])
    started = time.time()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for step, timestep in enumerate(scheduler.timesteps):
            stacked = torch.stack([timestep])
            conditional = model([latent], t=stacked, context=context,
                                seq_len=sequence_length)[0]
            unconditional = model([latent], t=stacked, context=context_null,
                                  seq_len=sequence_length)[0]
            prediction = unconditional + guidance * (conditional - unconditional)
            latent = scheduler.step(
                prediction.unsqueeze(0), timestep, latent.unsqueeze(0),
                return_dict=False, generator=generator,
            )[0].squeeze(0)
            if step % 5 == 0:
                log(f"  sample step {step}/{spec['steps']} t={float(timestep):.1f} "
                    f"|x|={latent.float().norm():.1f} ({time.time() - started:.1f}s)")
    latent = latent.float()
    log(f"latent sampled in {time.time() - started:.1f}s: {tuple(latent.shape)} "
        f"mean={latent.mean():.4f} std={latent.std():.4f}")
    return latent


def run_training_step(model, context, x0, recipe, sequence_length, device, dtype):
    """Stage 2: one training step, with the recorder installed on the captured block."""
    import wan.modules.model as wan_model_module
    from diffusers import FlowMatchEulerDiscreteScheduler

    capture = recipe["capture"]
    step_spec = recipe["training_step"]
    kind = capture["attn_kind"]
    block_index = int(capture["block_index"])
    kept = int(capture["kept_tokens"])

    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def record(recorded_kind, index, q, k, v, o):
        if recorded_kind != kind or index != block_index:
            return
        for name, tensor in (("q", q), ("k", k), ("v", v), ("o", o)):
            activations[name] = tensor[:, :kept].detach().to(dtype).cpu().clone()

        def hook(grad):
            gradients["do"] = grad[:, :kept].detach().to(dtype).cpu().clone()
            gradients["do_norm_full"] = float(grad.detach().float().norm())
            return None

        if o.requires_grad:
            o.register_hook(hook)

    install_capture_patches(wan_model_module, dtype, record)

    scheduler = FlowMatchEulerDiscreteScheduler(shift=1.0)
    index = int(step_spec["scheduler_index"])
    sigma = float(scheduler.sigmas.float()[index])
    timestep = float(scheduler.timesteps.float()[index])
    if abs(sigma - float(step_spec["sigma"])) > 1e-6:
        raise SystemExit(f"scheduler index {index} gives sigma {sigma}, recipe says "
                         f"{step_spec['sigma']}")
    if abs(timestep - float(step_spec["timestep"])) > 1e-6:
        raise SystemExit(f"scheduler index {index} gives timestep {timestep}, recipe says "
                         f"{step_spec['timestep']}")

    generator = torch.Generator(device=device)
    generator.manual_seed(int(step_spec["noise_seed"]))
    noise = torch.randn(x0.shape, generator=generator, device=device, dtype=torch.float32)
    target = noise - x0
    x_t = ((1.0 - sigma) * x0 + sigma * noise).to(torch.bfloat16).requires_grad_(True)

    log(f"training step: sigma={sigma:.4f} timestep={timestep:.1f} block={block_index} "
        f"kind={kind} kept_tokens={kept}")
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        prediction = model([x_t], t=torch.tensor([timestep], device=device),
                           context=context, seq_len=sequence_length)[0]
    forward_seconds = time.time() - started
    loss = F.mse_loss(prediction.float(), target.float())
    started = time.time()
    loss.backward()
    backward_seconds = time.time() - started
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    log(f"  forward {forward_seconds:.1f}s backward {backward_seconds:.1f}s "
        f"loss={loss.item():.6f} peak={peak:.1f} GiB "
        f"|dL/dx_t|={x_t.grad.float().norm():.4e}")

    if "do" not in gradients:
        raise SystemExit("no attention-output gradient was recorded")
    if set(activations) != {"q", "k", "v", "o"}:
        raise SystemExit(f"incomplete activation record: {sorted(activations)}")
    tensors = {name: activations[name].contiguous() for name in ("q", "k", "v", "o")}
    tensors["do"] = gradients["do"].contiguous()
    # Everything measured -- the loss, the gradient norms, the wall clock, the peak
    # memory -- belongs to the capture manifest, not to the saved payload.  The payload
    # therefore carries only the recipe's own identity, and two captures of the same
    # recipe on the same host produce a byte-identical file.
    measurements = {
        "loss": float(loss.item()),
        "grad_x_t_norm": float(x_t.grad.float().norm()),
        "do_norm_full_sequence": float(gradients["do_norm_full"]),
    }
    timing = {
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "peak_gib": peak,
    }
    return tensors, measurements, timing


# -------------------------------------------------------------------------------- main ---
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Capture the e7 edge fixture.")
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True,
                        help="checkout of the pinned upstream repository")
    parser.add_argument("--snapshot", type=Path, required=True,
                        help="pinned model snapshot directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)

    recipe_path = args.recipe.resolve()
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    recipe_sha256 = sha256_file(recipe_path)
    verify_upstream(args.upstream.resolve(), recipe)
    verify_snapshot(args.snapshot.resolve(), recipe)

    sys.path.insert(0, str(args.upstream.resolve()))
    from wan.configs import WAN_CONFIGS
    import wan.modules.model as wan_model_module
    from wan.modules.model import WanModel
    from wan.modules.t5 import T5EncoderModel

    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    dtype = getattr(torch, recipe["capture"]["dtype"].split(".")[-1])

    geometry = recipe["geometry"]
    sequence_length = int(geometry["seq_len"])
    latent_t, latent_h, latent_w = (
        int(geometry["latent_t"]), int(geometry["latent_h"]), int(geometry["latent_w"])
    )
    expected_sequence = latent_t * (latent_h // 2) * (latent_w // 2)
    if expected_sequence != sequence_length:
        raise SystemExit(f"recipe geometry is inconsistent: {expected_sequence} != "
                         f"{sequence_length}")
    log(f"geometry: {geometry['frames']} frames -> T={latent_t}; latent "
        f"{latent_h}x{latent_w}; S={sequence_length}")

    config = WAN_CONFIGS[recipe["model"]["config_key"]]
    install_sampling_patches(wan_model_module, dtype)

    log("loading the transformer")
    started = time.time()
    model = WanModel.from_pretrained(str(args.snapshot), torch_dtype=torch.bfloat16)
    model = model.to(device).eval().requires_grad_(False)
    for index, block in enumerate(model.blocks):
        block.self_attn._block_index = index
        block.cross_attn._block_index = index
    log(f"loaded in {time.time() - started:.1f}s: dim={model.dim} heads={model.num_heads} "
        f"layers={model.num_layers} head_dim={model.dim // model.num_heads}")
    if model.num_heads != recipe["capture"]["shape"][2]:
        raise SystemExit("model head count does not match the recipe")

    log("loading the text encoder")
    started = time.time()
    encoder = T5EncoderModel(
        text_len=config.text_len, dtype=config.t5_dtype, device=device,
        checkpoint_path=str(args.snapshot / config.t5_checkpoint),
        tokenizer_path=str(args.snapshot / config.t5_tokenizer),
    )
    context = [c.to(device) for c in encoder([recipe["prompt"]], device)]
    context_null = [c.to(device) for c in encoder([config.sample_neg_prompt], device)]
    log(f"text encoder done in {time.time() - started:.1f}s: context "
        f"{tuple(context[0].shape)}")
    del encoder
    gc.collect()
    torch.cuda.empty_cache()

    x0 = sample_latent(model, context, context_null, recipe, sequence_length, device)
    del context_null
    gc.collect()
    torch.cuda.empty_cache()

    enable_full_checkpointing(model)
    tensors, measurements, timing = run_training_step(
        model, context, x0, recipe, sequence_length, device, dtype
    )

    shape = tuple(recipe["capture"]["shape"])
    measured: dict[str, dict[str, float]] = {}
    for name in recipe["capture"]["tensors"]:
        tensor = tensors[name]
        if tuple(tensor.shape) != shape:
            raise SystemExit(f"{name} has shape {tuple(tensor.shape)}, expected {shape}")
        if tensor.dtype != dtype:
            raise SystemExit(f"{name} has dtype {tensor.dtype}, expected {dtype}")
        measured[name] = stats(tensor)
        check_band(name, measured[name], recipe["sanity_bands"][name])
        log(f"  {name}: rms={measured[name]['rms']:.6e} "
            f"abs_max={measured[name]['abs_max']:.6e} "
            f"finite={measured[name]['finite_fraction']:.6f}")

    payload = {name: tensors[name] for name in recipe["capture"]["tensors"]}
    payload["meta"] = {
        "recipe_id": recipe["recipe_id"],
        "recipe_sha256": recipe_sha256,
        "model_repo": recipe["model"]["repo_id"],
        "model_revision": recipe["model"]["revision"],
        "upstream_repo": recipe["upstream"]["repo"],
        "upstream_commit": recipe["upstream"]["commit"],
        "attn_kind": recipe["capture"]["attn_kind"],
        "block_index": recipe["capture"]["block_index"],
        "sigma": recipe["training_step"]["sigma"],
        "timestep": recipe["training_step"]["timestep"],
        "frames": recipe["geometry"]["frames"],
        "size_wh": [recipe["geometry"]["width"], recipe["geometry"]["height"]],
        "grid_sizes": recipe["geometry"]["grid_sizes"],
        "seq_len": sequence_length,
        "kept_tokens": recipe["capture"]["kept_tokens"],
        "tensor_layout": recipe["capture"]["tensor_layout"],
        "dtype": recipe["capture"]["dtype"],
        "post_rope_qk": recipe["capture"]["post_rope_qk"],
        "qk_norm": recipe["capture"]["qk_norm"],
        "num_heads": int(model.num_heads),
        "head_dim": int(model.dim // model.num_heads),
        "loss": recipe["training_step"]["loss"],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, args.out)
    fixture_sha256 = sha256_file(args.out)
    log(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.0f} MB) sha256={fixture_sha256}")

    capability = torch.cuda.get_device_capability(0)
    manifest = {
        "schema_version": 1,
        "identity": "per-deployment-capture",
        "recipe": {"id": recipe["recipe_id"], "sha256": recipe_sha256,
                   "path": "fixtures/recipe/RECIPE.json"},
        "fixture": {
            "name": args.out.name,
            "sha256": fixture_sha256,
            "bytes": int(args.out.stat().st_size),
            "shape": list(shape),
            "dtype": recipe["capture"]["dtype"],
            "tensors": list(recipe["capture"]["tensors"]),
        },
        "capture": {
            "captured_at": datetime.datetime.now(datetime.timezone.utc)
            .replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": f"{capability[0]}.{capability[1]}",
            "driver_version": _driver_version(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "measurements": measurements,
            "timing": timing,
            "tensor_stats": measured,
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_name(args.manifest.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, args.manifest)
    log(f"wrote {args.manifest}")
    return 0


def _driver_version() -> str:
    try:
        import subprocess

        return subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()[0]
    except Exception:  # pragma: no cover - informational field only
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
