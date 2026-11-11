"""Independent SD3.5-M oracle for the miles_diffusion DiffusionOPD task (author / Judge side).

Nothing here imports miles_diffusion or sglang, so it can check either of them:
- DiT: diffusers ``SD3Transformer2DModel``; ``strict_fp32()`` turns off TF32 and non-math SDPA.
- LoRA: explicit merge ``W = W0 + (alpha/r) * B @ A`` computed in fp64 from the PEFT safetensors and
  stored in the model dtype; ``lora_scope`` swaps a merged set in and restores W0 byte-exactly.
- Text conditioning: diffusers ``StableDiffusion3Pipeline.encode_prompt`` (CLIP-L/G penultimate hidden
  state + pooled, T5 at max 256 tokens) with fp32 encoders.
- Transition: fp64 transcription of the ``cps`` branch of sglang-miles-h3@859a278e
  ``scheduler_rl_mixin.flow_sde_sampling``; KL between two CPS transitions from the same state is
  ``mean((mu_S - mu_T)^2) / (2 s_t^2)`` (element mean over the D latent elements = true KL / D).
- Noise: canonical keys ``(case_seed, sample, step)`` so honest samplers agree (conventions A16).
"""

from __future__ import annotations

import contextlib
import json
import math
import os
from dataclasses import dataclass

import torch
from safetensors.torch import load_file

PEFT_PREFIX = "base_model.model."


def strict_fp32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


# ----------------------------------------------------------------------------------------- LoRA
@dataclass
class Lora:
    """PEFT LoRA: module name -> (A [r, in], B [out, r]) in fp32, and the alpha/r scale."""

    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]]
    scale: float
    source: str = ""

    @classmethod
    def load(cls, path: str) -> "Lora":
        """``path`` is a PEFT adapter dir (adapter_config.json + adapter_model.safetensors) or a
        bare safetensors file with PEFT key names (then r=32, alpha=64 are the task contract)."""
        if os.path.isdir(path):
            cfg = json.load(open(os.path.join(path, "adapter_config.json")))
            r, alpha = int(cfg["r"]), float(cfg["lora_alpha"])
            if cfg.get("use_rslora") or cfg.get("use_dora") or cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
                raise ValueError(f"unsupported adapter config in {path}")
            tensors = load_file(os.path.join(path, "adapter_model.safetensors"))
        else:
            r, alpha = 32, 64.0
            tensors = load_file(path)
        pairs: dict[str, list] = {}
        for key, value in tensors.items():
            name = key[len(PEFT_PREFIX):] if key.startswith(PEFT_PREFIX) else key
            for tag, slot in ((".lora_A.weight", 0), (".lora_B.weight", 1)):
                if name.endswith(tag):
                    pairs.setdefault(name[: -len(tag)], [None, None])[slot] = value.float()
                    break
            else:
                raise ValueError(f"unexpected LoRA key {key}")
        out = {}
        for module, (a, b) in pairs.items():
            if a is None or b is None or a.shape[0] != r or b.shape[1] != r:
                raise ValueError(f"bad LoRA pair for {module}")
            out[module] = (a, b)
        return cls(out, alpha / r, path)


class SD3Oracle:
    def __init__(self, model_dir: str, device: str = "cuda", dtype: torch.dtype = torch.float32):
        from diffusers import SD3Transformer2DModel

        self.model_dir = model_dir
        self.device = torch.device(device)
        self.dtype = dtype
        self.transformer = (
            SD3Transformer2DModel.from_pretrained(os.path.join(model_dir, "transformer"), torch_dtype=dtype)
            .to(self.device)
            .eval()
            .requires_grad_(False)
        )
        self._modules = dict(self.transformer.named_modules())
        self._w0: dict[str, torch.Tensor] = {}
        self._active: str | None = None

    # -- LoRA merge ---------------------------------------------------------------------------
    @torch.no_grad()
    def _merge(self, lora: Lora) -> None:
        for name, (a, b) in lora.pairs.items():
            linear = self._modules[name]
            if name not in self._w0:
                self._w0[name] = linear.weight.detach().clone()
            w0 = self._w0[name].to(torch.float64)
            delta = lora.scale * (b.to(self.device, torch.float64) @ a.to(self.device, torch.float64))
            linear.weight.copy_((w0 + delta).to(self.dtype))

    @torch.no_grad()
    def _restore(self) -> None:
        for name, w0 in self._w0.items():
            self._modules[name].weight.copy_(w0)

    @contextlib.contextmanager
    def lora_scope(self, lora: Lora | None, tag: str = ""):
        """Run with base weights (lora=None) or with one explicitly merged LoRA."""
        if self._active is not None:
            raise RuntimeError("nested lora_scope")
        self._active = tag or "active"
        try:
            if lora is not None:
                self._merge(lora)
            yield self
        finally:
            self._restore()
            self._active = None

    # -- forward ------------------------------------------------------------------------------
    def velocity(self, x: torch.Tensor, timestep: torch.Tensor, prompt_embeds: torch.Tensor,
                 pooled: torch.Tensor, autocast_dtype: torch.dtype | None = None) -> torch.Tensor:
        """DiT velocity prediction (guidance 1.0: single conditional branch). ``timestep`` is the
        scheduler timestep (sigma * 1000), exactly what the engine feeds its DiT."""
        ctx = (torch.autocast("cuda", dtype=autocast_dtype) if autocast_dtype is not None
               else contextlib.nullcontext())
        with ctx:
            out = self.transformer(
                hidden_states=x.to(self.device, self.dtype),
                timestep=timestep.to(self.device, torch.float32).expand(x.shape[0]),
                encoder_hidden_states=prompt_embeds.to(self.device, self.dtype),
                pooled_projections=pooled.to(self.device, self.dtype),
                return_dict=False,
            )[0]
        return out.float()


# ------------------------------------------------------------------------------ text + schedule
class SD3TextEncoder:
    def __init__(self, model_dir: str, device: str = "cuda"):
        from diffusers import StableDiffusion3Pipeline

        self.pipe = StableDiffusion3Pipeline.from_pretrained(
            model_dir, transformer=None, vae=None, torch_dtype=torch.float32
        ).to(device)

    @torch.no_grad()
    def encode(self, prompts: list[str], max_sequence_length: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
        pe, _, pooled, _ = self.pipe.encode_prompt(
            prompt=prompts, prompt_2=None, prompt_3=None, do_classifier_free_guidance=False,
            max_sequence_length=max_sequence_length,
        )
        return pe.float(), pooled.float()


def sd3_schedule(model_dir: str, num_steps: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    """(timesteps [num_steps], sigmas [num_steps+1]) of the pinned FlowMatchEuler scheduler (shift 3)."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(os.path.join(model_dir, "scheduler"))
    sched.set_timesteps(num_steps)
    return sched.timesteps.float().cpu(), sched.sigmas.float().cpu()


# ----------------------------------------------------------------------------------- CPS in fp64
def cps_std(sigma_next: float, eta: float) -> float:
    return float(sigma_next) * math.sin(eta * math.pi / 2)


def cps_mean(x: torch.Tensor, v: torch.Tensor, sigma: float, sigma_next: float, eta: float) -> torch.Tensor:
    """sglang-miles-h3 cps: mu = (x - s v)(1 - s') + (x + (1 - s) v) sqrt(s'^2 - std^2), in fp64."""
    x64, v64 = x.to(torch.float64), v.to(torch.float64)
    std = cps_std(sigma_next, eta)
    pred_x0 = x64 - sigma * v64
    noise_est = x64 + v64 * (1.0 - sigma)
    return pred_x0 * (1.0 - sigma_next) + noise_est * math.sqrt(max(sigma_next**2 - std**2, 0.0))


def kl_over_d(mu_s: torch.Tensor, mu_t: torch.Tensor, sigma_next: float, eta: float) -> torch.Tensor:
    """Per-sample KL(N(mu_s, s^2 I) || N(mu_t, s^2 I)) / D, D = elements per sample."""
    std = cps_std(sigma_next, eta)
    diff = mu_s.to(torch.float64) - mu_t.to(torch.float64)
    return (diff**2).flatten(1).mean(dim=1) / (2.0 * std * std)


# ------------------------------------------------------------------------------ canonical noise
def canonical_normal(shape: tuple[int, ...], *keys: int) -> torch.Tensor:
    """Standard normal on CPU from a Philox-free, order-independent key: the seed is a stable hash of
    the integer keys, so every honest sampler draws identical noise for (case, sample, step)."""
    h = 1469598103934665603
    for k in keys:
        h = ((h ^ (int(k) & 0xFFFFFFFFFFFFFFFF)) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    g = torch.Generator(device="cpu").manual_seed(h & 0x7FFFFFFFFFFFFFFF)
    return torch.randn(shape, generator=g, dtype=torch.float32)


# ------------------------------------------------------------------------------ LoRA gradients
@contextlib.contextmanager
def lora_grad_scope(oracle: "SD3Oracle", lora: Lora):
    """Attach ``lora`` UNMERGED as forward hooks (y += scale * x A^T B^T) with A, B as fp32 leaf tensors that
    require grad, so the oracle can differentiate a loss w.r.t. the student LoRA. Yields {module: (A, B)}."""
    params, handles = {}, []
    for name, (a, b) in lora.pairs.items():
        a_ = a.to(oracle.device, torch.float32).clone().requires_grad_(True)
        b_ = b.to(oracle.device, torch.float32).clone().requires_grad_(True)
        params[name] = (a_, b_)

        def hook(module, inputs, output, a_=a_, b_=b_):
            x = inputs[0]
            return output + lora.scale * torch.nn.functional.linear(torch.nn.functional.linear(x.float(), a_), b_).to(output.dtype)

        handles.append(oracle._modules[name].register_forward_hook(hook))
    try:
        yield params
    finally:
        for h in handles:
            h.remove()
