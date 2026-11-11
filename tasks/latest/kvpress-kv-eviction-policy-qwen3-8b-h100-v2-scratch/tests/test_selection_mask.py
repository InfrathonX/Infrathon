from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import kv_eviction_runtime as rt  # noqa: E402


def _selection(n_kv: int, length: int, n_kept: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return torch.stack([torch.randperm(length, generator=generator)[:n_kept] for _ in range(n_kv)]).unsqueeze(0)


@pytest.mark.parametrize("length", [64, 65, 71, 1000, rt.MAX_CONTEXT_TOKENS])
def test_round_trip_gives_ascending_positions(length: int) -> None:
    idx = _selection(8, length, 64)
    packed = rt.pack_selection(torch, idx, length)
    assert packed.dtype == torch.uint8 and tuple(packed.shape) == (8, -(-length // 8))
    assert torch.equal(rt.unpack_selection(torch, packed, length, 8, 64), idx.sort(-1).values)


def test_one_bit_per_position_is_much_smaller_than_int64() -> None:
    length = 13_000
    stored = sum(rt.pack_selection(torch, _selection(8, length, int(length * (1 - r))), length).numel() for r in rt.RATIOS)
    as_int64 = sum(8 * int(length * (1 - r)) * 8 for r in rt.RATIOS)
    assert as_int64 / stored > 15


def test_rejects_malformed_masks() -> None:
    length, n_kept = 65, 64
    packed = rt.pack_selection(torch, _selection(8, length, n_kept), length)
    assert rt.unpack_selection(torch, packed.to(torch.int64), length, 8, n_kept) is None
    assert rt.unpack_selection(torch, packed[:7], length, 8, n_kept) is None
    assert rt.unpack_selection(torch, packed.tolist(), length, 8, n_kept) is None
    assert rt.unpack_selection(torch, packed, length, 8, n_kept + 1) is None
    padding = packed.clone(); padding[0, -1] |= 1  # position 71 lies past the 65-token context
    assert rt.unpack_selection(torch, padding, length, 8, n_kept) is None
    short = rt.pack_selection(torch, _selection(8, length, n_kept - 1), length)
    assert rt.unpack_selection(torch, short, length, 8, n_kept) is None


def test_handoff_estimate_covers_a_full_holdout_in_single_digit_gigabytes() -> None:
    items = [{"meta": {"context_tokens": 13_000}, "context": ""}] * 5000
    assert 5e9 < rt.handoff_bytes(items) < 1e10
    assert rt.handoff_bytes([{"context": "x" * 4000}]) == len(rt.RATIOS) * (36 * 8 * 126 + (1 << 16))


def test_warmup_tensors_match_shapes_and_strides_but_not_content() -> None:
    ops = rt.TrustedOps.capture(torch)
    hidden = torch.randn(1, 40, 64, dtype=torch.bfloat16); queries = torch.randn(1, 40, 32, 16, dtype=torch.bfloat16).transpose(1, 2)
    keys = torch.randn(1, 8, 40, 16, dtype=torch.bfloat16); values = torch.randn(1, 8, 40, 16, dtype=torch.bfloat16)
    pe = (torch.randn(1, 40, 16), torch.randn(1, 40, 16)); weights = {"q_proj": torch.randn(4, 4)}
    class C: pass
    capture = C(); capture.hidden_states = [hidden]; capture.queries = [queries]; capture.keys = [keys]; capture.values = [values]
    capture.position_embeddings = pe; capture.weights = [weights]
    a = rt._decoy_inputs(ops, capture, 0, torch.Generator().manual_seed(rt.decoy_seed("f00001")))
    b = rt._decoy_inputs(ops, capture, 0, torch.Generator().manual_seed(rt.decoy_seed("f00001")))
    for got, ref in zip(a[:4], (hidden, queries, keys, values)):
        assert got.shape == ref.shape and got.stride() == ref.stride() and got.dtype == ref.dtype and not torch.equal(got, ref)
    assert all(torch.equal(x, y) for x, y in zip(a[:4], b[:4]))
    assert torch.equal(a[4][0], pe[0]) and a[4][0].data_ptr() != pe[0].data_ptr() and a[5] is weights
    assert rt.decoy_seed("f00001") != rt.decoy_seed("f00002")


def test_masks_digest_identifies_the_selection_not_its_order() -> None:
    idx = _selection(8, 1000, 64, seed=1)
    a = [rt.pack_selection(torch, idx, 1000)]
    b = [rt.pack_selection(torch, idx.flip(-1), 1000)]  # same kept set, other order
    c = [rt.pack_selection(torch, _selection(8, 1000, 64, seed=2), 1000)]
    assert rt.masks_digest(a) == rt.masks_digest(b) != rt.masks_digest(c) and len(rt.masks_digest(a)) == 64
