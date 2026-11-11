"""HPSv3 video-frame scorer (author transcription; Judge oracle + reference reward).

Transcribed from verl-omni b8b67c9a ``verl_omni/utils/reward_score/hpsv3_reward.py`` (Apache-2.0), which follows the
official MizzenAI/HPSv3 inferencer (MIT). Contract (task reward definition):

- weights ``MizzenAI/HPSv3@4f81e3e0 HPSv3.safetensors`` (Qwen2-VL layout; the repo's own config.json says Qwen2.5-VL
  and must NOT be used); model built from ``Qwen/Qwen2-VL-7B-Instruct@eed13092`` config + processor, tokenizer extended
  by the special token ``<|Reward|>``; ranknet head Linear(3584,1024)-ReLU-Dropout-Linear(1024,16)-ReLU-Linear(16,2)
  kept in fp32, backbone bf16; strict load.
- input: uint8 RGB frames 0, 4, 8 of the decoded 9-frame video; each frame resized with PIL BICUBIC to
  ``smart_resize(h, w, factor=28, min_pixels=max_pixels=256*28*28)`` (704x1280 -> 308x588).
- positions: sequential 1-D position ids for all tokens (official forward passes position_ids=None to the
  text decoder under transformers 4.45.2; no M-RoPE) — reproduces the official scorer within its batch noise (S3c).
- chat: one user message [image, text = INSTRUCTION.format(text_prompt=prompt) + PROMPT_WITH_SPECIAL_TOKEN],
  ``apply_chat_template(add_generation_prompt=True)``, processor padding=True (right), no truncation.
- score: rm_head output at the ``<|Reward|>`` position, column 0 (mu). Video reward = 0.1 * mean of the 3 frame mus.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

INSTRUCTION = (
    "\nYou are tasked with evaluating a generated image based on Visual Quality and Text"
    " Alignment and give a overall score to estimate the human preference. Please provide"
    " a rating from 0 to 10, with 0 being the worst and 10 being the best. \n"
    "\n"
    "**Visual Quality:**  \n"
    "Evaluate the overall visual quality of the image. The following sub-dimensions should be considered:\n"
    "- **Reasonableness:** The image should not contain any significant biological or logical errors,"
    " such as abnormal body structures or nonsensical environmental setups.\n"
    "- **Clarity:** Evaluate the sharpness and visibility of the image. The image should be clear and"
    " easy to interpret, with no blurring or indistinct areas.\n"
    "- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other"
    " visual elements (e.g., hair, clothing, shadows).\n"
    "- **Aesthetic and Creativity:** Assess the artistic aspects of the image, including the color"
    " scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should"
    " convey a sense of harmony and balance.\n"
    "- **Safety:** The image should not contain harmful or inappropriate content, such as political,"
    " violent, or adult material. If such content is present, the image quality and satisfaction score"
    " should be the lowest possible. \n"
    "\n"
    "**Text Alignment:**  \n"
    "Assess how well the image matches the textual prompt across the following sub-dimensions:\n"
    "- **Subject Relevance** Evaluate how accurately the subject(s) in the image (e.g., person, animal,"
    " object) align with the textual description. The subject should match the description in terms of"
    " number, appearance, and behavior.\n"
    "- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate"
    " how well the image adheres to this style.\n"
    "- **Contextual Consistency**: Assess whether the background, setting, and surrounding elements in"
    " the image logically fit the scenario described in the prompt. The environment should support and"
    " enhance the subject without contradictions.\n"
    "- **Attribute Fidelity**: Check if specific attributes mentioned in the prompt (e.g., colors,"
    " clothing, accessories, expressions, actions) are faithfully represented in the image. Minor"
    " deviations may be acceptable, but critical attributes should be preserved.\n"
    "- **Semantic Coherence**: Evaluate whether the overall meaning and intent of the prompt are"
    " captured in the image. The generated content should not introduce elements that conflict with or"
    " distort the original description.\n"
    "Textual prompt - {text_prompt}\n"
    "\n\n"
)
PROMPT_WITH_SPECIAL_TOKEN = """
Please provide the overall ratings of this image: <|Reward|>

END
"""
REWARD_TOKEN = "<|Reward|>"
IMAGE_FACTOR = 28
PIXELS = 256 * 28 * 28
FRAME_INDICES = (0, 4, 8)
REWARD_SCALE = 0.1


def smart_resize(height: int, width: int, factor: int = IMAGE_FACTOR, min_pixels: int = PIXELS, max_pixels: int = PIXELS):
    if max(height, width) / min(height, width) > 200:
        raise ValueError("aspect ratio > 200")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def resize_frame(frame_hwc_uint8: np.ndarray) -> Image.Image:
    if frame_hwc_uint8.dtype != np.uint8 or frame_hwc_uint8.ndim != 3 or frame_hwc_uint8.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 uint8 RGB, got {frame_hwc_uint8.dtype} {frame_hwc_uint8.shape}")
    img = Image.fromarray(frame_hwc_uint8).convert("RGB")
    w, h = img.size
    rh, rw = smart_resize(h, w)
    return img.resize((rw, rh), Image.BICUBIC)


def build_messages(images: Sequence[Image.Image], prompts: Sequence[str]):
    return [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img, "min_pixels": PIXELS, "max_pixels": PIXELS},
                    {"type": "text", "text": INSTRUCTION.format(text_prompt=p) + PROMPT_WITH_SPECIAL_TOKEN},
                ],
            }
        ]
        for img, p in zip(images, prompts, strict=True)
    ]


def load_processor(base_dir: str):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(base_dir, padding_side="right")
    processor.tokenizer.add_special_tokens({"additional_special_tokens": [REWARD_TOKEN]})
    return processor


def processor_inputs(processor, images: Sequence[Image.Image], prompts: Sequence[str]):
    msgs = build_messages(images, prompts)
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # Images are already resized to the contract size; the processor's own smart_resize is then the identity.
    return processor(text=text, images=list(images), padding=True, return_tensors="pt")


def _remap_state_dict(state_dict, model_keys):
    if any("model.language_model" in k for k in model_keys) and not any("language_model" in k for k in state_dict):
        out = {}
        for key, value in state_dict.items():
            if key.startswith("visual."):
                out["model." + key] = value
            elif key.startswith("model."):
                out["model.language_model." + key[len("model.") :]] = value
            else:
                out[key] = value
        return out
    return state_dict


def build_model(checkpoint: str, base_dir: str, reward_token_id: int, vocab_size: int, device="cpu",
                backbone_dtype=torch.bfloat16, attn_implementation: str = "sdpa"):
    from safetensors.torch import load_file
    from transformers import AutoConfig, Qwen2VLForConditionalGeneration

    config = AutoConfig.from_pretrained(base_dir)
    config._attn_implementation = attn_implementation

    class HPSv3RewardModel(Qwen2VLForConditionalGeneration):
        def __init__(self, cfg):
            super().__init__(cfg)
            hidden = cfg.text_config.hidden_size if hasattr(cfg, "text_config") else cfg.hidden_size
            self.rm_head = nn.Sequential(
                nn.Linear(hidden, 1024), nn.ReLU(), nn.Dropout(0.05), nn.Linear(1024, 16), nn.ReLU(), nn.Linear(16, 2)
            )

        def reward(self, input_ids, attention_mask, pixel_values, image_grid_thw, mm_token_type_ids=None):
            # Official HPSv3 semantics (MizzenAI inferencer, transformers 4.45.2): the text decoder is called with
            # position_ids=None, i.e. sequential 1-D positions for every token (no M-RoPE). transformers>=5 would
            # otherwise compute M-RoPE positions for the image tokens, which is a different function (S3c).
            bsz, seqlen = input_ids.shape
            position_ids = torch.arange(seqlen, device=input_ids.device).view(1, 1, seqlen).expand(3, bsz, seqlen)
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                use_cache=False,
            )
            h = out.last_hidden_state
            mask = input_ids == reward_token_id
            if not bool((mask.sum(dim=1) == 1).all()):
                raise RuntimeError("expected exactly one reward token per sequence")
            pooled = h[mask].float()  # [B, hidden], one position per row (row-major order)
            return self.rm_head(pooled)  # [B, 2] fp32

    # Same construction order as verl-omni: build (fp32) -> resize embeddings -> .to(bf16) (this also casts the
    # rotary inv_freq buffers) -> rm_head back to fp32 -> strict load of the bf16 checkpoint.
    with torch.device(device):
        model = HPSv3RewardModel(config)
    model.resize_token_embeddings(vocab_size)
    model.to(backbone_dtype)
    model.rm_head.to(torch.float32)
    sd = load_file(checkpoint, device="cpu")
    sd = _remap_state_dict(sd, model.state_dict().keys())
    model.load_state_dict(sd, strict=True)
    del sd
    model.eval().requires_grad_(False)
    return model


class HPSv3Scorer:
    def __init__(self, checkpoint: str, base_dir: str, device="cuda", backbone_dtype=torch.bfloat16,
                 attn_implementation="sdpa"):
        self.processor = load_processor(base_dir)
        tok = self.processor.tokenizer
        self.reward_token_id = tok.convert_tokens_to_ids(REWARD_TOKEN)
        self.device = device
        self.model = build_model(checkpoint, base_dir, self.reward_token_id, len(tok), device, backbone_dtype,
                                 attn_implementation)

    @torch.no_grad()
    def score_images(self, images: Sequence[Image.Image], prompts: Sequence[str], batch_size: int = 16) -> torch.Tensor:
        """Returns fp32 [N] mu (rm_head column 0)."""
        out = []
        for s in range(0, len(images), batch_size):
            inp = processor_inputs(self.processor, images[s : s + batch_size], prompts[s : s + batch_size])
            inp = {k: v.to(self.device) for k, v in inp.items()}
            logits = self.model.reward(
                inp["input_ids"], inp["attention_mask"], inp["pixel_values"], inp["image_grid_thw"],
                inp.get("mm_token_type_ids"),
            )
            out.append(logits[:, 0].float().cpu())
        return torch.cat(out)

    def score_videos(self, videos_uint8: np.ndarray, prompts: Sequence[str], batch_size: int = 16):
        """videos_uint8: [V, 9, H, W, 3] uint8. Returns (frame_mu [V, 3] fp32, reward [V] fp64)."""
        frames, fprompts = [], []
        for v, p in zip(videos_uint8, prompts, strict=True):
            for f in FRAME_INDICES:
                frames.append(resize_frame(np.ascontiguousarray(v[f])))
                fprompts.append(p)
        mu = self.score_images(frames, fprompts, batch_size).view(len(prompts), len(FRAME_INDICES))
        return mu, REWARD_SCALE * mu.double().mean(dim=1)
