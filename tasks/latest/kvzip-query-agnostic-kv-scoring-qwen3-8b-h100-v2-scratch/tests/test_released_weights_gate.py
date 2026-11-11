"""Released-weights gate on synthetic gates shaped like Fast KVzip's (CPU only)."""
from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from conftest import TASK_ROOT, load_module

gate = load_module("kv_released_weights_gate_test", "images/judge/released_weights_gate.py")
LAYERS = 2


def released_gate(seed: int = 7) -> dict:
    """Two layers of q_proj [64,4096] / k_proj [16,4096] plus the small tensors."""
    g = torch.Generator().manual_seed(seed); out = {}
    for layer in range(LAYERS):
        p = f"fastkvzip/layer{layer}/"
        out[p + "q_proj.weight"] = (0.01 * torch.randn(64, 4096, generator=g)).to(torch.bfloat16)
        out[p + "q_proj.bias"] = (0.1 * torch.randn(64, generator=g)).to(torch.bfloat16)
        out[p + "k_proj.weight"] = (0.01 * torch.randn(16, 4096, generator=g)).to(torch.bfloat16)
        out[p + "q_norm.weight"] = 1 + 0.01 * torch.randn(16, generator=g)
        out[p + "k_norm.weight"] = 1 + 0.01 * torch.randn(16, generator=g)
        out[p + "k_base"] = 0.005 * torch.randn(8, 1, 16, 16, generator=g)
        out[p + "b"] = (0.1 * torch.randn(8, 1, 4, generator=g)).to(torch.bfloat16)
    return out


RELEASED = released_gate()
RELEASED_BYTES = io.BytesIO(); torch.save(RELEASED, RELEASED_BYTES)
REFERENCE = gate.build_reference([{"name": "synthetic", "sha256": gate.sha256_bytes(RELEASED_BYTES.getvalue())}], [RELEASED])


def run(assets: dict) -> dict:
    buffer = io.BytesIO(); torch.save(assets, buffer)
    payload = buffer.getvalue()
    return gate.check(payload, gate.load_tensors(payload), REFERENCE)


def rows_only(assets: dict) -> dict:
    """Drop the small tensors so only row matching can catch a copy."""
    return {k: v for k, v in assets.items() if v.dim() >= 2 and v.shape[-1] == 4096}


def int8_rowwise(t):
    w = t.float(); scale = w.abs().amax(dim=-1, keepdim=True) / 127
    return (w / scale).round().clamp(-127, 127).to(torch.int8), scale


def b01_layout(src: dict) -> dict:
    """What the B01 run shipped: q+k concatenated, int8 per-row, renamed keys."""
    out = {}
    for layer in range(LAYERS):
        p = f"fastkvzip/layer{layer}/"
        q, s = int8_rowwise(torch.cat([src[p + "q_proj.weight"], src[p + "k_proj.weight"]]))
        out[f"{layer}.weight"] = q; out[f"{layer}.scale"] = s
        out[f"{layer}.k_base"] = src[p + "k_base"].reshape(8, 16, 16); out[f"{layer}.b"] = src[p + "b"].float().reshape(8, 4)
    return out


def byte_planes(src: dict) -> dict:
    """The oracle transport packing (prepare_oracle_assets.save_transport_repacked)."""
    return {f"packed/{k}": v.contiguous().view(torch.uint8).reshape(*v.shape, v.element_size()).movedim(-1, 0).contiguous() for k, v in src.items()}


def test_reference_holds_no_weights() -> None:
    assert set(REFERENCE) == {"schema", "row_dim", "sketch_dim", "sketch_seed", "sources", "file_sha256", "tensor_digests", "row_sketches", "projection"}
    assert REFERENCE["row_sketches"].shape == (LAYERS * 80, gate.SKETCH_DIM) and REFERENCE["row_sketches"].dtype == torch.float16


@pytest.mark.parametrize("name,make", [
    ("exact copy", lambda s: dict(s)),
    ("fp32 cast", lambda s: {k: v.float() for k, v in s.items()}),
    ("bf16 cast of rows only", lambda s: {k: v.to(torch.bfloat16) for k, v in rows_only(s).items()}),
    ("per-row scaled", lambda s: {k: v.float() * torch.linspace(0.5, 3.0, v.shape[0]).unsqueeze(1) for k, v in rows_only(s).items()}),
    ("sign-flipped rows", lambda s: {k: -v.float() for k, v in rows_only(s).items()}),
    ("transposed", lambda s: {k: v.float().t().contiguous() for k, v in rows_only(s).items()}),
    ("reshaped", lambda s: {k: v.reshape(-1, 2048, 8) for k, v in rows_only(s).items()}),
    ("keys renamed and reordered", lambda s: {f"t{i}": s[k] for i, k in enumerate(reversed(list(s)))}),
    ("B01 layout (concat + int8 + scale)", b01_layout),
    ("row-permuted fp16", lambda s: {k: v[torch.randperm(v.shape[0], generator=torch.Generator().manual_seed(1))].half() for k, v in rows_only(s).items()}),
    ("oracle byte planes", byte_planes),
    ("raw bytes as int8", lambda s: {k: v.contiguous().view(torch.int8) for k, v in rows_only(s).items()}),
    ("fine-tuned copy (30% noise)", lambda s: {k: v.float() + 0.3 * v.float().std() * torch.randn(v.shape, generator=torch.Generator().manual_seed(2)) for k, v in rows_only(s).items()}),
])
def test_released_gate_derivatives_are_rejected(name: str, make) -> None:
    result = run(make(RELEASED))
    assert result["ok"] is False, (name, result)


def test_exact_copy_is_caught_three_ways() -> None:
    result = run(dict(RELEASED))
    assert result["exact_tensor_matches"] == len(RELEASED) and result["matched_rows"] == result["reference_rows"]
    assert gate.check(RELEASED_BYTES.getvalue(), RELEASED, REFERENCE)["reasons"][0].startswith("asset file is byte-identical")


@pytest.mark.parametrize("name,make", [
    ("independent gate, same shapes", lambda: released_gate(seed=99)),
    ("independent gate, B01 layout", lambda: b01_layout(released_gate(seed=123))),
    ("trained-like gate sharing a common direction", lambda: {
        k: (v.float() + 0.01 * torch.ones(4096)) if v.dim() == 2 else v for k, v in released_gate(seed=5).items()}),
    ("positional table and no 4096-wide rows", lambda: {"pos": torch.linspace(0, 1, 32768), "w": torch.randn(36, 8)}),
    ("small tensors only, all ones", lambda: {"q_norm": torch.ones(16), "k_norm": torch.ones(16)}),
])
def test_independent_assets_are_admitted(name: str, make) -> None:
    result = run(make())
    assert result["ok"] is True, (name, result)
    assert result["max_row_cosine"] < gate.ROW_COSINE_THRESHOLD


def test_a_few_copied_rows_stay_below_the_count_floor() -> None:
    few = {"rows": RELEASED["fastkvzip/layer0/q_proj.weight"][: gate.MIN_MATCHED_ROWS - 1].float() * 2}
    result = run(few)
    assert result["ok"] is True and result["matched_rows"] == gate.MIN_MATCHED_ROWS - 1
    more = {"rows": RELEASED["fastkvzip/layer0/q_proj.weight"][: gate.MIN_MATCHED_ROWS].float() * 2}
    assert run(more)["ok"] is False


def test_schema_mismatch_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="schema"):
        gate.check(b"x", {}, {**REFERENCE, "sketch_dim": 128})


def test_cli_check_and_describe(tmp_path: Path) -> None:
    asset = tmp_path / "assets.pt"; torch.save(b01_layout(RELEASED), asset)
    reference = io.BytesIO(); torch.save(REFERENCE, reference)
    script = str(TASK_ROOT / "images/judge/released_weights_gate.py")
    checked = subprocess.run([sys.executable, "-I", script, "check", str(asset)], input=reference.getvalue(), capture_output=True, check=True)
    verdict = json.loads(checked.stdout.decode().strip().splitlines()[-1])
    assert verdict["ok"] is False and verdict["matched_rows"] == LAYERS * 80
    described = subprocess.run([sys.executable, "-I", script, "describe", str(asset)], capture_output=True, check=True, text=True)
    lines = [json.loads(line) for line in described.stdout.splitlines()]
    assert lines[0]["tensor_count"] == len(b01_layout(RELEASED)) and lines[1]["name"] == "0.weight" and lines[1]["dtype"] == "int8"
    assert "file_sha256" in lines[0] and all("sha256" in entry for entry in lines[1:])


def test_gzip_assets_are_read_like_the_runtime(tmp_path: Path) -> None:
    import gzip
    buffer = io.BytesIO(); torch.save(rows_only(RELEASED), buffer)
    asset = tmp_path / "assets.pt"; asset.write_bytes(gzip.compress(buffer.getvalue()))
    payload = gate.read_asset_bytes(asset)
    assert gate.check(payload, gate.load_tensors(payload), REFERENCE)["ok"] is False
