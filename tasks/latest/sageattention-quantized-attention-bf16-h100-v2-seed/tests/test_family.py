from __future__ import annotations

import pytest
from conftest import load_module

# The generator is exercised through the trusted runtime, so this module needs
# Torch.  The repository's lightweight-check interpreter does not carry it;
# skip rather than fail collection there.
torch = pytest.importorskip("torch")

runtime=load_module("qattn_runtime_family","runtime/quant_attention_runtime.py")

def test_generator_is_byte_deterministic() -> None:
    one=runtime.generate_family(torch,1,512,4,2,64,123,"raw",device="cpu")
    two=runtime.generate_family(torch,1,512,4,2,64,123,"raw",device="cpu")
    for left,right in zip(one[:3],two[:3],strict=True): assert torch.equal(left.view(torch.uint8),right.view(torch.uint8))
    assert one[3] == two[3]

def test_token_scale_precedes_bias_and_smoothing_shrinks_raw_range() -> None:
    _q,k,_v,params=runtime.generate_family(torch,1,4096,4,2,64,991,"raw",device="cpu",dtype=torch.float32)
    centered=k-k.mean(dim=1,keepdim=True)
    assert float(k.abs().amax()/centered.abs().amax()) >= 2.0
    # Bias channels are token-independent after generation: their means are
    # many standard errors from zero in the raw regime.
    ratio=(k.mean(1).abs()/k.std(1).clamp_min(1e-6)).amax()
    assert float(ratio) > 6.0 and params["regime"] == "raw"

BAND_SHAPE=(1,256,12,128)

def _band_tensors(scale: float=1.0) -> dict[str,object]:
    """Synthetic q/k/v whose statistics sit on the recipe's reference values.

    Mock tensors only: a flat field at the reference RMS with one element
    pushed to the reference abs-max, so both statistics land mid-band without
    a GPU, a model or the real capture.
    """
    recipe=runtime.load_recipe()
    out: dict[str,object]={}
    for name,band in recipe["sanity_bands"]["tensors"].items():
        value=torch.full(BAND_SHAPE,float(band["rms_reference"])*scale,dtype=torch.float32)
        value[0,0,0,0]=float(band["abs_max_reference"])*scale
        out[name]=value
    return out

def test_fixture_sanity_bands_accept_a_reference_like_capture() -> None:
    recipe=runtime.load_recipe()
    runtime._check_fixture_bands(torch,_band_tensors(),recipe)

def test_fixture_sanity_bands_reject_a_rescaled_capture() -> None:
    recipe=runtime.load_recipe()
    with pytest.raises(RuntimeError,match="outside recipe band"):
        runtime._check_fixture_bands(torch,_band_tensors(scale=100.0),recipe)
    with pytest.raises(RuntimeError,match="outside recipe band"):
        runtime._check_fixture_bands(torch,_band_tensors(scale=0.001),recipe)

def test_fixture_sanity_bands_reject_nonfinite_values() -> None:
    recipe=runtime.load_recipe()
    tensors=_band_tensors()
    tensors["k"][0,0,0,0]=float("nan")
    with pytest.raises(RuntimeError,match="finite fraction"):
        runtime._check_fixture_bands(torch,tensors,recipe)

def test_fixture_sanity_bands_cover_every_tensor(monkeypatch: pytest.MonkeyPatch) -> None:
    recipe=runtime.load_recipe()
    assert set(recipe["sanity_bands"]["tensors"]) == {"q","k","v"}
    for name in ("q","k","v"):
        tensors=_band_tensors()
        tensors[name]=tensors[name]*50.0
        with pytest.raises(RuntimeError,match=f"e8 {name} "):
            runtime._check_fixture_bands(torch,tensors,recipe)
