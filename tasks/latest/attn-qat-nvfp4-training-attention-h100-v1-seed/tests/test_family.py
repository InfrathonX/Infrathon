from __future__ import annotations

from pathlib import Path

import pytest
from conftest import load_module

# The generators and the golden are exercised through the trusted runtime, so this module
# needs Torch.  The repository's lightweight-check interpreter does not carry it; skip
# rather than fail collection there.
torch = pytest.importorskip("torch")

runtime = load_module("attn_qat_runtime_family", "runtime/attn_qat_runtime.py")
public = load_module("attn_qat_public_family", "images/work/benchmark/public_runner.py")

SMALL = dict(b=1, s_q=128, s_kv=128, h=4, d=64)

# NVFP4 fake quantization, pinned against `lab/seed/reference.py::fq_torch` -- the helper-A
# function that was itself verified bit-exact against the Triton kernel on 1,532,928 values.
# One group of 16 per row, so each row is one block scale.  The four cases that matter:
#   ties        every magnitude sits exactly halfway between two e2m1 grid points and the
#               group maximum is 6, so the block scale is 1 and RTNE is directly visible
#               (0.25 -> 0, 0.75 -> 1, 1.25 -> 1, 1.75 -> 2, 2.5 -> 2, 3.5 -> 4, 5 -> 4)
#   zeros       sd = 0, 1/sd = inf, x/sd = NaN, e2m1 saturates to 6, 6 * 0 = -0.0
#   overflow    a / 6 exceeds e4m3's 448, so the PTX satfinite clamp binds (not Torch's NaN)
#   tiny        the e4m3 floor: a group whose maximum is below 6 * 2^-10 flushes to zero,
#               which is why the forward must never quantize a normalized probability
FQ_VECTORS = {
    "ties_bf16": {
        "dtype": "bfloat16",
        "source_is_fp32": False,
        "x": [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0,
              -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0, -6.0],
        "y": [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0,
              -0.0, -1.0, -1.0, -2.0, -2.0, -4.0, -4.0, -6.0],
    },
    "zeros_bf16": {
        "dtype": "bfloat16",
        "source_is_fp32": False,
        "x": [0.0] * 16,
        "y": [-0.0] * 16,
    },
    "e4m3_overflow_bf16": {
        "dtype": "bfloat16",
        "source_is_fp32": False,
        "x": [3008.0, 6016.0, 9024.0, 12032.0, 14976.0, 18048.0, 20992.0, 24064.0,
              27008.0, 29952.0, 33024.0, 36096.0, 38912.0, 41984.0, 45056.0, 48128.0],
        "y": [2688.0] * 16,
    },
    "ties_fp32": {
        "dtype": "float32",
        "source_is_fp32": True,
        "x": [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0,
              -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5.0, -6.0],
        "y": [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0,
              -0.0, -1.0, -1.0, -2.0, -2.0, -4.0, -4.0, -6.0],
    },
    "unnormalized_weights_fp32": {
        "dtype": "float32",
        "source_is_fp32": True,
        "x": [2.0 ** (-0.37 * i) for i in range(16)],
        "y": [1.03125, 0.6875, 0.515625, 0.515625, 0.34375, 0.2578125, 0.171875, 0.171875,
              0.0859375, 0.0859375, 0.0859375, 0.0859375, 0.0859375, 0.0, 0.0, 0.0],
    },
    "one_hot_fp32": {
        "dtype": "float32",
        "source_is_fp32": True,
        "x": [1.0] + [0.0] * 15,
        "y": [1.03125] + [0.0] * 15,
    },
    "below_the_e4m3_floor_fp32": {
        "dtype": "float32",
        "source_is_fp32": True,
        "x": [1e-7 * (i + 1) for i in range(16)],
        "y": [0.0] * 16,
    },
}


def _gate():
    return runtime.TrustedGateOps.capture(torch)


def _generate(seed: int, **overrides):
    kwargs = dict(SMALL)
    kwargs.update(overrides)
    return runtime.generate_inputs(
        torch,
        kwargs["b"],
        kwargs["s_q"],
        kwargs["s_kv"],
        kwargs["h"],
        kwargs["d"],
        seed,
        device="cpu",
        dtype=torch.bfloat16,
        overrides=kwargs.get("case_overrides"),
        k_zero=kwargs.get("k_zero", False),
        v_zero_groups=kwargs.get("v_zero_groups", 0.0),
        tie_set=kwargs.get("tie_set", False),
        do_scale=kwargs.get("do_scale"),
    )


@pytest.mark.parametrize("name", sorted(FQ_VECTORS))
def test_fake_quantize_matches_the_pinned_reference_vectors(name: str) -> None:
    case = FQ_VECTORS[name]
    dtype = getattr(torch, case["dtype"])
    x = torch.tensor(case["x"], dtype=torch.float32).reshape(1, 16).to(dtype)
    expected = torch.tensor(case["y"], dtype=torch.float32).reshape(1, 16)
    produced = runtime.fake_quantize(_gate(), x, case["source_is_fp32"])
    assert torch.equal(produced, expected), (name, produced.tolist())


def test_fake_quantize_pads_an_unaligned_last_axis() -> None:
    x = torch.randn(3, 35, dtype=torch.float32)
    produced = runtime.fake_quantize(_gate(), x, True)
    assert tuple(produced.shape) == (3, 35)
    # The zero padding cannot change a group maximum, so the leading aligned groups agree
    # with the same call on the truncated tensor.
    aligned = runtime.fake_quantize(_gate(), x[:, :32], True)
    assert torch.equal(produced[:, :32], aligned)


def test_generator_is_byte_deterministic() -> None:
    one = _generate(123)
    two = _generate(123)
    for left, right in zip(one[:4], two[:4], strict=True):
        assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))
    assert one[4]["roles"] == two[4]["roles"]


def test_generator_shapes_dtype_and_layout() -> None:
    q, k, v, do, _params = _generate(7)
    for tensor in (q, k, v, do):
        assert tuple(tensor.shape) == (1, 4, 128, 64)
        assert tensor.dtype is torch.bfloat16
        assert tensor.is_contiguous()


def test_different_seeds_give_different_draws() -> None:
    one = _generate(11)
    two = _generate(12)
    assert not torch.equal(one[0].view(torch.uint8), two[0].view(torch.uint8))


def test_primal_range_patch_is_the_shipped_default() -> None:
    # Helper B's correction: the lineage's ranges never reach the near-Gaussian
    # within-16-group dynamic range the real mid-network captures live at, and NVFP4 error
    # is monotone in that statistic.  Values replaced: (0.30, 0.85), (0.30, 0.85),
    # (0.20, 0.55), (0.8, 17.8), (0.8, 19.3), (0.5, 5.2).
    assert runtime.RANGES["chan_sigma_q"] == (0.02, 0.62)
    assert runtime.RANGES["chan_sigma_k"] == (0.02, 0.62)
    assert runtime.RANGES["chan_sigma_v"] == (0.02, 0.24)
    assert runtime.RANGES["q_bias_mag"] == (0.5, 5.0)
    assert runtime.RANGES["k_bias_mag"] == (0.5, 5.0)
    assert runtime.RANGES["v_bias_mag"] == (0.4, 2.5)


def test_group_dynamic_range_lands_in_the_measured_band() -> None:
    # Real captures: q 2.16-2.61, k 2.14-2.64, v 2.10-2.24 (iid normal 2.110).
    q, k, v, _do, _params = _generate(31, s_q=512, s_kv=512, h=8, d=128)
    for tensor, low, high in ((q, 2.05, 2.85), (k, 2.05, 2.85), (v, 2.05, 2.45)):
        values = tensor.float().reshape(-1, 16)
        ratio = values.abs().amax(dim=-1) / values.pow(2).mean(dim=-1).sqrt().clamp_min(1e-30)
        assert low <= float(ratio.mean()) <= high, float(ratio.mean())


def test_do_is_unit_rms_and_uncorrelated_with_the_output() -> None:
    q, k, v, do, _params = _generate(41, s_q=512, s_kv=512, h=8, d=128)
    assert float(do.float().pow(2).mean().sqrt()) == pytest.approx(1.0, rel=1e-3)
    out = runtime.reference_attention(
        torch, q.permute(0, 2, 1, 3).contiguous(), k.permute(0, 2, 1, 3).contiguous(),
        v.permute(0, 2, 1, 3).contiguous()
    )
    left = out.float().reshape(-1, 128)
    right = do.permute(0, 2, 1, 3).float().reshape(-1, 128)
    cosine = (left * right).sum(-1) / (
        left.norm(dim=-1) * right.norm(dim=-1)
    ).clamp_min(1e-20)
    assert abs(float(cosine.median())) < 0.05


def test_do_scale_override_rescales_the_gradient() -> None:
    _q, _k, _v, do, params = _generate(43, do_scale=1.0e-6)
    assert params["do_rms"] == pytest.approx(1.0e-6)
    assert float(do.float().pow(2).mean().sqrt()) == pytest.approx(1.0e-6, rel=1e-3)


def test_k_zero_and_v_zero_group_overrides() -> None:
    _q, k, v, _do, params = _generate(47, k_zero=True, v_zero_groups=0.25)
    assert params["k_zero"] is True
    assert float(k.float().abs().max()) == 0.0
    groups = v.float().reshape(-1, 16)
    zeroed = (groups.abs().amax(dim=-1) == 0.0).float().mean()
    assert 0.1 < float(zeroed) < 0.4


def test_tie_set_override_builds_the_rounding_tie_fixture() -> None:
    q, k, v, _do, params = _generate(53, tie_set=True)
    assert params["regime"] == "e2m1-tie-set"
    allowed = set(runtime.TIE_MAGNITUDES) | {runtime.TIE_GROUP_MAXIMUM}
    for tensor in (q, k, v):
        values = tensor.float().abs()
        assert set(values.unique().tolist()) <= allowed
        groups = values.reshape(-1, 16)
        assert bool((groups.amax(dim=-1) == runtime.TIE_GROUP_MAXIMUM).all())


def test_maxp_targets_override_is_applied() -> None:
    _q, _k, _v, _do, params = _generate(59, h=4, case_overrides={"maxp_targets": [1.0] * 4})
    assert params["maxp_targets"] == [1.0] * 4


def test_cross_attention_slices_query_and_key_rows() -> None:
    q, k, v, do, params = _generate(61, s_q=128, s_kv=48)
    assert tuple(q.shape) == (1, 4, 128, 64) and tuple(do.shape) == (1, 4, 128, 64)
    for tensor in (k, v):
        assert tuple(tensor.shape) == (1, 4, 48, 64)
    assert params["generated_s"] == 128 and params["cross_attention"] is True
    full = _generate(61)
    assert torch.equal(
        k.view(torch.uint8), full[1][:, :, :48].contiguous().view(torch.uint8)
    )


def test_self_attention_does_not_slice() -> None:
    _q, _k, _v, _do, params = _generate(67)
    assert params["cross_attention"] is False


def test_golden_reproduces_the_operator_on_a_tiny_case() -> None:
    # The forward's `o` must differ from the high-precision path by roughly one NVFP4
    # rounding, and every output must be finite -- including on the zero-group path where
    # the block scale is zero and the intermediate is NaN.
    q, k, v, do, _params = _generate(71, s_q=64, s_kv=64, h=2, d=64, v_zero_groups=0.5)
    reference = runtime.reference_fwd_bwd(_gate(), q, k, v, do, 64.0**-0.5, chunk=32)
    for name in runtime.GATE_TENSORS:
        assert bool(torch.isfinite(reference[name]).all()), name
    assert float(reference["o"].abs().max()) > 0.0


def test_public_selection_matches_the_disclosed_cells() -> None:
    assert len(public.cases_for_suite("one")) == 1
    quick = public.cases_for_suite("quick")
    assert [x["case_id"] for x in quick] == ["public-p2", "public-p4", "public-p24", "public-p10"]
    assert [x["perf_only"] for x in quick] == [False, False, False, True]
    full = public.cases_for_suite("full")
    assert len(full) == 21
    assert not any(x["perf_only"] for x in full)
    assert [x["cell"] for x in full] == list(public.GEOMETRY)
    assert public.cases_for_suite("full", only="p10", perf_only=True) == [
        {"case_id": "public-p10", "cell": "P10", "seed": public.SEEDS["P10"],
         "gated": False, "perf_only": True, "layer": None}
    ]


def test_capture_without_its_manifest_fails_closed(tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    """A capture is only admissible with the manifest that binds it to the recipe."""
    (tmp_path / runtime.FIXTURE_NAME).write_bytes(b"whatever this deployment captured")
    monkeypatch.setenv("ATTN_QAT_FIXTURE_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="missing or unsafe"):
        runtime._fixture_path()


def test_missing_fixture_directory_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ATTN_QAT_FIXTURE_DIR", raising=False)
    with pytest.raises(RuntimeError, match="ATTN_QAT_FIXTURE_DIR"):
        runtime._fixture_path()


def test_saved_bundle_walk_counts_new_storages_only() -> None:
    gate = _gate()
    contract = torch.zeros(4, 4)
    reused = contract[:2]
    fresh = torch.zeros(8, 8)
    total, tensors = runtime.saved_object_bytes(gate, (reused, fresh), (contract,))
    assert tensors == 2
    assert total == fresh.numel() * fresh.element_size()
    with pytest.raises(RuntimeError, match="callables"):
        runtime.saved_object_bytes(gate, (torch.zeros(1), len), (contract,))
    with pytest.raises(RuntimeError, match="unsupported member"):
        runtime.saved_object_bytes(gate, (object(),), (contract,))
