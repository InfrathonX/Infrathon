from pathlib import Path
import gzip
import io
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from conftest import load_module

assets = load_module("kv_prepare_oracle_assets_test", "scripts/prepare_oracle_assets.py")
oracle = load_module("kv_oracle_transport_test", "oracle/candidates/scorer_v001/submission.py")
runtime = load_module("kv_runtime_transport_test", "runtime/kv_scoring_runtime.py")


def synthetic_checkpoint():
    layers = []
    for layer in range(2):
        layers.append({name: torch.arange(layer + 1, layer + 4, dtype=torch.float32) for name in assets.TENSOR_NAMES})
    return {"module": layers}


def test_repack_is_deterministic_on_synthetic_state(tmp_path: Path):
    first = assets.repack_state_dict(synthetic_checkpoint())
    second = assets.repack_state_dict(synthetic_checkpoint())
    assert list(first) == list(second)
    assert all(torch.equal(first[key], second[key]) for key in first)
    path_a = tmp_path / "a/assets.pt"; path_b = tmp_path / "b/assets.pt"
    digest_a = assets.save_repacked(first, path_a); digest_b = assets.save_repacked(second, path_b)
    assert digest_a == digest_b


def test_transport_repack_is_lossless(tmp_path: Path):
    original = {
        "fastkvzip/layer0/q_proj.weight": torch.arange(24, dtype=torch.bfloat16).reshape(4, 6),
        "fastkvzip/layer0/q_norm.weight": torch.arange(4, dtype=torch.float32),
    }
    source = tmp_path / "source.pt"; transport = tmp_path / "transport.pt"
    torch.save(original, source)
    assets.save_transport_repacked(source, transport)
    assert transport.stat().st_size <= 128 * 1024 * 1024
    packed = torch.load(io.BytesIO(gzip.decompress(transport.read_bytes())), map_location="cpu", weights_only=True)
    decoded = oracle.KVScorer({}, packed).assets
    assert list(decoded) == list(original)
    assert all(torch.equal(decoded[key], original[key]) for key in original)

    ops = SimpleNamespace(load=lambda source_value, **_kwargs: torch.load(source_value, map_location="cpu", weights_only=True))
    runtime_loaded = runtime._load_assets(torch, ops, transport)
    assert all(torch.equal(runtime_loaded[key], packed[key]) for key in packed)
