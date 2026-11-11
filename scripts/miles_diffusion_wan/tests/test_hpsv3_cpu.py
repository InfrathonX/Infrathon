"""CPU tests of the HPSv3 transcription against verl-omni (template bytes, resize, processor inputs, strict load).

The full-model test is slow (~8B params on CPU) and runs only with WAN_SLOW=1.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

# WAN_ASSETS: the Wan assets root (MDWAN_ASSETS in scripts/miles_common/host_config.py); WAN_REFS: the pinned upstream
# checkouts (default <WAN_ASSETS>/refs). Without them the tests that need them are skipped.
ASSETS = Path(os.environ["WAN_ASSETS"]) if os.environ.get("WAN_ASSETS") else None
REFS = Path(os.environ["WAN_REFS"]) if os.environ.get("WAN_REFS") else (ASSETS / "refs" if ASSETS else None)
if REFS is None:
    pytest.skip("set WAN_ASSETS (the Wan assets root, MDWAN_ASSETS) or WAN_REFS (the pinned upstream checkouts)",
                allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hpsv3_scorer as S  # noqa: E402

VO_FILE = REFS / "verl-omni/verl_omni/utils/reward_score/hpsv3_reward.py"
QWEN = ASSETS / "Qwen2-VL-7B-Instruct-config" if ASSETS else Path("/nonexistent/WAN_ASSETS-unset")
CKPT = ASSETS / "HPSv3/HPSv3.safetensors" if ASSETS else Path("/nonexistent/WAN_ASSETS-unset")


def _vo_module():
    if "verl.utils.device" not in sys.modules:
        for name in ("verl", "verl.utils"):
            sys.modules.setdefault(name, types.ModuleType(name))
        dev = types.ModuleType("verl.utils.device")
        dev.get_device_name = lambda: "cpu"
        sys.modules["verl.utils.device"] = dev
    spec = importlib.util.spec_from_file_location("vo_hpsv3_reward", VO_FILE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["vo_hpsv3_reward"] = mod
    spec.loader.exec_module(mod)
    return mod


def _video(seed=0, h=704, w=1280, t=9):
    rng = np.random.default_rng(seed)
    # smooth-ish content so bicubic resizing is not pure noise
    base = rng.integers(0, 256, size=(t, h // 16, w // 16, 3), dtype=np.uint8)
    return np.repeat(np.repeat(base, 16, axis=1), 16, axis=2)


def test_template_bytes_match_verl_omni():
    vo = _vo_module()
    assert S.INSTRUCTION == vo._INSTRUCTION
    assert S.PROMPT_WITH_SPECIAL_TOKEN == vo._PROMPT_WITH_SPECIAL_TOKEN
    assert "<|Reward|>" in S.PROMPT_WITH_SPECIAL_TOKEN


@pytest.mark.parametrize("hw", [(704, 1280), (480, 832), (1280, 704), (100, 100), (28, 5000)])
def test_smart_resize_matches_verl_omni(hw):
    vo = _vo_module()
    assert S.smart_resize(*hw) == vo._smart_resize(*hw, min_pixels=S.PIXELS, max_pixels=S.PIXELS)


def test_contract_frame_size():
    assert S.smart_resize(704, 1280) == (308, 588)


def test_frame_resize_matches_verl_omni_fetch():
    vo = _vo_module()
    v = _video()
    for f in S.FRAME_INDICES:
        ours = np.asarray(S.resize_frame(v[f]))
        theirs = np.asarray(
            vo._fetch_image({"image": vo._to_pil_hwc(v[f]), "min_pixels": S.PIXELS, "max_pixels": S.PIXELS})
        )
        assert ours.shape == (308, 588, 3) and np.array_equal(ours, theirs)


def test_verl_omni_frame_extraction_is_0_4_8():
    vo = _vo_module()
    v = torch.from_numpy(_video()).permute(0, 3, 1, 2).contiguous()  # TCHW, the VO reward-manager contract
    frames = vo._extract_frames(v, frame_interval=4)
    assert len(frames) == 3
    for pil, f in zip(frames, S.FRAME_INDICES):
        assert np.array_equal(np.asarray(pil), _video()[f])


@pytest.mark.skipif(not QWEN.exists(), reason="Qwen2-VL config not staged (set WAN_ASSETS)")
def test_processor_inputs_match_verl_omni_path():
    vo = _vo_module()
    proc = S.load_processor(str(QWEN))
    rid = proc.tokenizer.convert_tokens_to_ids(S.REWARD_TOKEN)
    assert len(proc.tokenizer) == 151658 and rid == 151657
    v = _video(1)
    prompts = ["a cat surfing a wave at sunset", "A much longer prompt " * 40]
    imgs = [S.resize_frame(v[0]), S.resize_frame(v[4])]
    ours = S.processor_inputs(proc, imgs, prompts)
    # verl-omni path: messages -> _process_vision_info (PIL bicubic) -> processor(text, images, padding=True)
    msgs = [
        [{"role": "user", "content": [
            {"type": "image", "image": vo._to_pil_hwc(v[f]), "min_pixels": S.PIXELS, "max_pixels": S.PIXELS},
            {"type": "text", "text": vo._INSTRUCTION.format(text_prompt=p) + vo._PROMPT_WITH_SPECIAL_TOKEN}]}]
        for f, p in zip((0, 4), prompts)
    ]
    theirs = proc(
        text=proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
        images=vo._process_vision_info(msgs), padding=True, return_tensors="pt", videos_kwargs={"do_rescale": True},
    )
    assert torch.equal(ours["input_ids"], theirs["input_ids"])
    assert torch.equal(ours["attention_mask"], theirs["attention_mask"])
    assert torch.equal(ours["image_grid_thw"], theirs["image_grid_thw"])
    assert torch.equal(ours["pixel_values"], theirs["pixel_values"])
    assert ((ours["input_ids"] == rid).sum(dim=1) == 1).all()
    assert ours["image_grid_thw"].tolist() == [[1, 22, 42], [1, 22, 42]]  # 231 merged visual tokens per frame
    n_img = (ours["input_ids"] == proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")).sum(dim=1)
    assert n_img.tolist() == [231, 231]


@pytest.mark.skipif(os.environ.get("WAN_SLOW") != "1" or not CKPT.exists(), reason="set WAN_SLOW=1")
def test_strict_load_and_forward_cpu():
    torch.set_num_threads(min(64, os.cpu_count() or 8))
    sc = S.HPSv3Scorer(str(CKPT), str(QWEN), device="cpu")
    v = np.stack([_video(2), _video(3)])
    mu, reward = sc.score_videos(v, ["a red car driving on a road", "a red car driving on a road"], batch_size=6)
    print("frame mu", mu.tolist(), "reward", reward.tolist())
    assert mu.shape == (2, 3) and torch.isfinite(mu).all()
    # batch invariance of the transcription itself (padding right, one reward token per row)
    mu1 = sc.score_images([S.resize_frame(v[0][0])], ["a red car driving on a road"], batch_size=1)
    assert abs(float(mu1[0]) - float(mu[0, 0])) < 0.05
