"""Score math pins: logical bytes, weighted geomean, envelope, guards."""

from __future__ import annotations

import math

import pytest

from conftest import _load

runtime = _load("kv_runtime_s", "runtime/kv_transfer_runtime.py")


def _ids(n: int, nb: int = 512) -> list[int]:
    return list(range(n))


def _wave(**overrides) -> dict:
    wave = {
        "layout": "split", "nl": 32, "nh": 8, "hs": 128, "chunk_tokens": 256,
        "num_objects": 4, "direction": "store", "skip": 0, "nb": 512,
    }
    wave.update(overrides)
    wave["ids"] = _ids(wave["num_objects"] * wave["chunk_tokens"] // 16, wave["nb"])
    return wave


def test_wave_logical_bytes_by_hand() -> None:
    wave = _wave()
    # 4 objects x 16 blocks x 2 kv x 32 layers x 16 tokens x 8 heads x 128 x 2B
    assert runtime.wave_logical_bytes(wave) == 64 * 2 * 32 * 16 * 8 * 128 * 2

    fused = dict(wave, layout="fused")
    # kv=1 but the per-head width doubles: identical byte volume by construction.
    assert runtime.wave_logical_bytes(fused) == runtime.wave_logical_bytes(wave)

    skipped = dict(wave, skip=16)
    assert runtime.wave_logical_bytes(skipped) == (
        runtime.wave_logical_bytes(wave) * 48 // 64
    )
    half = _wave(hs=64)
    assert runtime.wave_logical_bytes(half) * 2 == runtime.wave_logical_bytes(wave)
    assert runtime.wave_geometry(_wave(hs=64, layout="fused"))["hs"] == 128


def _row(family: str, score: float, stratum: str = "s0", tier: str = "other") -> dict:
    return {"scored": True, "family": family, "case_score": score, "stratum": stratum, "tier": tier}


def test_weighted_aggregate() -> None:
    rows = [
        _row("device-large", 1400.0), _row("device-large", 1000.0),
        _row("device-small", 500.0),
        _row("host-large", 52.0), _row("host-small", 40.0), _row("host-duplex", 70.0),
        _row("trace", 50.0), _row("overlap", 30.0),
    ]
    summary = runtime.aggregate(rows)
    device = math.sqrt(math.sqrt(1400.0 * 1000.0) * 500.0)
    host = (52.0 * 40.0 * 70.0) ** (1 / 3)
    expected = math.exp(0.35 * math.log(device) + 0.25 * math.log(host)
                        + 0.10 * math.log(50.0) + 0.30 * math.log(30.0))
    assert math.isclose(summary["score_gbps"], expected, rel_tol=1e-12)
    assert math.isclose(summary["group_scores"]["device"], device, rel_tol=1e-12)
    assert math.isclose(summary["family_scores"]["overlap"], 30.0, rel_tol=1e-12)


def test_strata_and_tiers_count_once() -> None:
    """A stratum with many cases weighs as much as one with a single case, and
    each tier counts once however many strata it has."""
    rows = [_row("trace", 100.0, "a", "paper")] * 9 + [_row("trace", 25.0, "b", "paper")]
    assert math.isclose(runtime.family_score(rows), 50.0, rel_tol=1e-12)
    rows = ([_row("trace", 100.0, f"o{i}", "other") for i in range(9)]
            + [_row("trace", 25.0, "p", "paper")])
    assert math.isclose(runtime.family_score(rows), 50.0, rel_tol=1e-12)


def test_small_calls_cannot_be_averaged_away() -> None:
    """Many large cases do not dilute one small family: each family counts once."""
    few = [_row("device-large", 1400.0), _row("device-small", 200.0)]
    many = [_row("device-large", 1400.0)] * 50 + [_row("device-small", 200.0)]
    base = [_row("host-large", 52.0), _row("host-small", 40.0), _row("host-duplex", 70.0),
            _row("trace", 50.0), _row("overlap", 30.0)]
    assert math.isclose(runtime.aggregate(few + base)["score_gbps"],
                        runtime.aggregate(many + base)["score_gbps"], rel_tol=1e-12)


def test_aggregate_requires_every_family() -> None:
    rows = [_row("device-large", 1000.0)]
    with pytest.raises(RuntimeError):
        runtime.aggregate(rows)
    summary = runtime.aggregate(rows, require_full_groups=False)
    assert math.isclose(summary["score_gbps"], 1000.0, rel_tol=1e-12)


def test_group_weights_sum_to_one() -> None:
    assert math.isclose(sum(runtime.GROUP_WEIGHTS.values()), 1.0, rel_tol=0.0)


def _probe(device: float = 1763.0, host: float = 57.6) -> dict:
    return {
        "protocol": runtime.ENVELOPE_PROTOCOL,
        "probe_bytes": runtime.ENVELOPE_PROBE_BYTES,
        "repeats": runtime.ENVELOPE_PROBE_REPEATS,
        "margin": runtime.ENVELOPE_MARGIN,
        "device_gbps": device,
        "h2d_gbps": host * 0.8,
        "d2h_gbps": host * 0.7,
        "zerocopy_h2d_gbps": host,
        "zerocopy_d2h_gbps": host * 0.9,
        "host_gbps": host,
        "duplex_gbps": host * 1.4,
    }


def test_envelope_comes_from_the_same_run_probe() -> None:
    """No frozen bandwidth number: the envelope is whatever the trusted probe measured."""
    probe = _probe()
    assert runtime.case_envelope_gbps({"arm": "device"}, probe) == 1763.0
    assert runtime.case_envelope_gbps({"arm": "host"}, probe) == 57.6
    assert runtime.case_envelope_gbps({"arm": "trace"}, probe) == 57.6
    assert math.isclose(runtime.case_envelope_gbps({"arm": "duplex"}, probe), 57.6 * 1.4)
    faster = _probe(device=4000.0, host=120.0)  # e.g. H200 / PCIe Gen6 host
    assert runtime.case_envelope_gbps({"arm": "device"}, faster) == 4000.0
    assert runtime.case_envelope_gbps({"arm": "host"}, faster) == 120.0
    assert runtime.validate_envelopes(probe)["device_gbps"] == 1763.0
    assert "DEVICE_ENVELOPE_GBPS" not in dir(runtime)


@pytest.mark.parametrize(
    "mutate",
    (
        lambda p: p.__setitem__("protocol", "other"),
        lambda p: p.__setitem__("probe_bytes", 1),
        lambda p: p.__setitem__("margin", 2.0),
        lambda p: p.__setitem__("device_gbps", runtime.ENVELOPE_MIN_DEVICE_GBPS - 1.0),
        lambda p: p.__setitem__("host_gbps", 1.0),
        lambda p: p.__setitem__("h2d_gbps", float("nan")),
        lambda p: p.__setitem__("host_gbps", 999.0),
        lambda p: p.__setitem__("duplex_gbps", 10.0),
        lambda p: p.pop("duplex_gbps"),
    ),
)
def test_broken_probe_records_are_rejected(mutate) -> None:
    probe = _probe()
    mutate(probe)
    with pytest.raises(RuntimeError):
        runtime.validate_envelopes(probe)


def test_native_arch_follows_the_device() -> None:
    import types

    def fake(major, minor):
        return types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_capability=lambda i=0: (major, minor)))

    assert runtime.native_arch(fake(9, 0)) == "sm_90a"
    assert runtime.native_arch(fake(8, 0)) == "sm_80"
    assert runtime.native_arch(fake(8, 6)) == "sm_86"
    assert runtime.native_arch(fake(10, 0)) == "sm_100a"
    argv = runtime.compile_argv(runtime.Path("a.cu"), runtime.Path("a.so"), runtime.Path("inc"), "sm_86")
    assert "-arch=sm_86" in argv and not any("sm_90a" in item for item in argv)
    with pytest.raises(ValueError):
        runtime.compile_argv(runtime.Path("a.cu"), runtime.Path("a.so"), runtime.Path("inc"), "sm_90a -O0")


def _case(**overrides) -> dict:
    case = {"case_id": "x", "family": "device-large", "arm": "device", "foreground": None,
            "stratum": "s0", "tier": "other", "waves": [_wave()], "trials": 3, "seed": 1}
    case.update(overrides)
    return case


def test_validate_case_guards() -> None:
    ok = runtime.validate_case(_case(), scored=True)
    assert ok["waves"][0]["num_objects"] == 4
    small = _wave(num_objects=1, chunk_tokens=16)
    assert runtime.validate_case(_case(family="device-small", waves=[small]), scored=True)
    with pytest.raises(ValueError, match="does not match family"):
        runtime.validate_case(_case(family="device-small"), scored=True)
    with pytest.raises(ValueError, match="does not belong"):
        runtime.validate_case(_case(family="host-large"), scored=True)
    with pytest.raises(ValueError, match="exceeds 128 waves"):
        runtime.validate_case(_case(waves=[small] * 129, family="device-small"), scored=True)
    with pytest.raises(ValueError, match="must share"):
        runtime.validate_case(_case(waves=[_wave(), _wave(direction="retrieve")]), scored=True)
    with pytest.raises(ValueError, match="foreground"):
        runtime.validate_case(_case(family="overlap", arm="overlap"), scored=True)
    with pytest.raises(ValueError, match="foreground"):
        runtime.validate_case(_case(foreground="gemm"), scored=True)
    assert runtime.validate_case(_case(family="overlap", arm="overlap", foreground="hbm"),
                                 scored=True)
    # A direct case may skip a different prefix on every call.
    assert runtime.validate_case(_case(waves=[_wave(), _wave(skip=3)]), scored=True)
    # A trace mixes call classes on one pool, but never two pools.
    mixed = [_wave(), _wave(direction="retrieve", num_objects=1)]
    assert runtime.validate_case(_case(family="trace", arm="trace", waves=mixed), scored=True)
    with pytest.raises(ValueError, match="must share"):
        runtime.validate_case(_case(family="trace", arm="trace", waves=[_wave(), _wave(nl=36)]),
                              scored=True)
    store = _wave(nb=1024)
    load = dict(_wave(nb=1024, direction="retrieve"), ids=list(range(512, 576)))
    assert runtime.validate_case(_case(family="host-duplex", arm="duplex", waves=[store, load]),
                                 scored=True)
    with pytest.raises(ValueError, match="disjoint"):
        runtime.validate_case(_case(family="host-duplex", arm="duplex",
                                    waves=[store, dict(load, ids=store["ids"])]), scored=True)
    with pytest.raises(ValueError, match="stratum or tier"):
        runtime.validate_case(_case(tier="hidden"), scored=True)
    with pytest.raises(ValueError, match="exceeds"):
        big = _wave(nl=128, nh=32, chunk_tokens=1024, nb=4096)
        runtime.validate_case(_case(family="host-large", arm="host", waves=[big] * 4), scored=True)


def test_wave_validation_ranges() -> None:
    base = _wave(nl=4, nh=4, chunk_tokens=64, num_objects=2, nb=64, direction="retrieve")
    assert runtime.validate_wave(base)["nl"] == 4
    assert runtime.validate_wave(dict(base, nl=128))["nl"] == 128
    for bad in (
        dict(base, nl=0), dict(base, nl=129), dict(base, nh=0),
        dict(base, nh=33), dict(base, hs=96), dict(base, num_objects=0),
        dict(base, num_objects=5), dict(base, chunk_tokens=17), dict(base, skip=9),
        dict(base, chunk_tokens=1024),  # 128 blocks > nb=64
        dict(base, layout="mla"), dict(base, direction="up"),
        dict(base, ids=base["ids"][:-1]), dict(base, ids=[0] * 8),
        dict(base, ids=list(range(57, 65))), dict(base, ids=[-1] + list(range(7))),
    ):
        with pytest.raises(ValueError):
            runtime.validate_wave(bad)


def test_edge_ids_have_the_shapes_a_live_pool_produces() -> None:
    assert runtime.edge_ids("reverse", 4, 64, 1) == sorted(runtime.edge_ids("reverse", 4, 64, 1),
                                                         reverse=True)
    ends = runtime.edge_ids("ends", 6, 100, 1)
    assert 0 in ends and 99 in ends
    runs = runtime.edge_ids("runs", 64, 4096, 3)
    assert len(set(runs)) == 64
