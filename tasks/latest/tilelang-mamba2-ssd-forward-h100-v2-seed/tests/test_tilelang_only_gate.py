"""The runtime TileLang-only operator record, exercised on CPU tensors.

A TileLang kernel launches through TileLang's own adapter and dispatches no ATen
operator, so the record of a TileLang-only call holds allocations and views
only.  Host torch compute in any spelling -- a function, a tensor method, an
operator, a dunder method -- dispatches a compute operator and fails the gate.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from conftest import _load  # noqa: E402

runtime = _load("mamba2_ssd_runtime", "runtime/mamba2_ssd_runtime.py")


def record(fn) -> list[str]:
    recorder = runtime.torch_op_recorder_class()()
    with recorder:
        fn()
    return list(recorder.ops)


def allocation_and_views() -> None:
    x = torch.empty((2, 8, 4, 6), dtype=torch.float32)
    torch.zeros((3,))
    torch.full((2, 2), 1.5)
    torch.empty_like(x)
    x.view(2, 8, 24)
    x.reshape(16, 24)
    x.permute(0, 2, 1, 3)
    x.transpose(1, 2)
    x[:, 1:3, 0]
    x.unsqueeze(0).squeeze(0)
    x[:, :1].expand(2, 8, 4, 6)
    x.split(4, dim=1)
    x.narrow(1, 0, 4)
    x.contiguous()
    x.view(torch.int32)


def test_allocations_and_views_pass() -> None:
    ops = record(allocation_and_views)
    assert ops
    assert runtime.check_torch_ops(ops, "views") == sorted(set(ops))


X = torch.randn(2, 8, 4, 6)

COMPUTE = {
    "elementwise_operator": lambda: X * 2.0,
    "elementwise_method": lambda: X.exp(),
    "scan": lambda: X.cumsum(1),
    "dunder_matmul": lambda: X[0, 0].__matmul__(X[0, 0].t()),
    "dtype_conversion": lambda: X.to(torch.float64),
    "copy": lambda: torch.empty_like(X).copy_(X),
    "noncontiguous_reshape": lambda: X.permute(0, 2, 1, 3).reshape(2, 4, 48),
    "host_read": lambda: X[0, 0, 0, 0].item(),
    "tensor_fill": lambda: torch.empty(3).fill_(X[0, 0, 0, 0]),
}


@pytest.mark.parametrize("label", sorted(COMPUTE))
def test_host_compute_fails_the_gate(label: str) -> None:
    ops = record(COMPUTE[label])
    with pytest.raises(runtime.GateFailure) as raised:
        runtime.check_torch_ops(ops, label)
    assert raised.value.gate == "tilelang_only_ok"


def test_allowlist_holds_no_compute_operator() -> None:
    for name in runtime.ALLOWED_TORCH_OPS:
        assert name.startswith("aten::")
        for compute in ("mm", "mul", "add", "exp", "cumsum", "copy_", "clone", "_to_copy", "item"):
            assert name.split("::")[1].split(".")[0] != compute, name
