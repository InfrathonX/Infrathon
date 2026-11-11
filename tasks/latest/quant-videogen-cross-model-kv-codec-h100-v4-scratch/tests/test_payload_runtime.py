from __future__ import annotations

import importlib.util
from collections import OrderedDict
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[1]
PAYLOAD_PATH = ROOT / "runtime/quant_videogen/payload.py"


def load_payload_module():
    spec = importlib.util.spec_from_file_location("qvg_scratch_payload", PAYLOAD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scalars_and_keys_are_charged_bytes() -> None:
    """Scalars are not free: KV data must not travel as text or Python numbers."""
    payload = load_payload_module()
    # keys "shape"/"scale"/"ok" = 5 + 5 + 2, three ints and one float at 8, one bool at 1
    assert payload.payload_nbytes({"shape": [1, 2, 3], "scale": 0.5, "ok": True}) == 45
    assert payload.payload_nbytes("x" * 100) == 100
    assert payload.payload_nbytes(None) == 0


def test_nested_and_quantized_tensors_are_rejected_before_the_cuda_check() -> None:
    import torch

    payload = load_payload_module()
    nested = torch.nested.nested_tensor([torch.zeros(2), torch.zeros(3)])
    with pytest.raises(ValueError, match="dense strided"):
        payload.payload_nbytes([nested])
    quantized = torch.quantize_per_tensor(torch.zeros(4), 0.1, 0, torch.qint8)
    with pytest.raises(ValueError, match="quantized"):
        payload.payload_nbytes([quantized])
    sparse = torch.zeros(3, 3).to_sparse()
    with pytest.raises(ValueError, match="dense strided"):
        payload.payload_nbytes([sparse])


def test_payload_rejects_cycles_and_custom_objects() -> None:
    payload = load_payload_module()
    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="cycle"):
        payload.payload_nbytes(cyclic)
    with pytest.raises(TypeError, match="unsupported payload object"):
        payload.payload_nbytes(object())


def test_payload_rejects_cpu_tensor() -> None:
    payload = load_payload_module()
    with pytest.raises(ValueError, match="CUDA"):
        payload.payload_nbytes(torch.zeros(1))


# ---------------------------------------------------------------------------
# v4: the counter rejects payload tensors that are not plain torch.Tensor or
# that carry state (autograd, Python attributes, hooks) it cannot see.  Each of
# these checks runs before the CUDA-residency rule, so they are reachable on a
# CPU tensor without monkeypatching; the existing CUDA-only rule stays intact.
# ---------------------------------------------------------------------------


def test_payload_rejects_tensor_subclass() -> None:
    payload = load_payload_module()
    param = torch.nn.Parameter(torch.zeros(1))
    assert type(param) is not torch.Tensor and isinstance(param, torch.Tensor)
    with pytest.raises(ValueError, match="subclass"):
        payload.payload_nbytes(param)


def test_payload_rejects_tensor_requiring_grad() -> None:
    payload = load_payload_module()
    with pytest.raises(ValueError, match="require grad"):
        payload.payload_nbytes(torch.zeros(1, requires_grad=True))


def test_payload_rejects_tensor_with_autograd_graph() -> None:
    payload = load_payload_module()
    # A non-leaf tensor carries a grad_fn (an autograd graph that references its
    # inputs); such a tensor necessarily also requires grad, so either guard is
    # an acceptable rejection.
    live = torch.zeros(2, requires_grad=True) * 2
    assert live.grad_fn is not None
    with pytest.raises(ValueError, match="require grad|grad_fn"):
        payload.payload_nbytes(live)


def test_payload_rejects_tensor_with_saved_grad() -> None:
    payload = load_payload_module()
    tensor = torch.zeros(3)
    tensor.grad = torch.ones(3)
    assert not tensor.requires_grad and tensor.grad is not None
    with pytest.raises(ValueError, match=r"saved \.grad"):
        payload.payload_nbytes(tensor)


def test_payload_rejects_tensor_with_python_attributes() -> None:
    payload = load_payload_module()
    tensor = torch.zeros(3)
    tensor.sidecar = torch.ones(3)  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="attributes"):
        payload.payload_nbytes(tensor)


def test_payload_rejects_tensor_with_backward_hooks() -> None:
    payload = load_payload_module()
    tensor = torch.zeros(3)
    tensor._backward_hooks = OrderedDict({0: (lambda *a: None)})
    with pytest.raises(ValueError, match="backward hooks"):
        payload.payload_nbytes(tensor)


def test_payload_rejects_tensor_with_post_accumulate_grad_hooks() -> None:
    payload = load_payload_module()
    tensor = torch.zeros(3)
    tensor._post_accumulate_grad_hooks = OrderedDict({0: (lambda *a: None)})
    with pytest.raises(ValueError, match="post-accumulate-grad hooks"):
        payload.payload_nbytes(tensor)


def test_payload_attribute_checks_run_before_the_cuda_check() -> None:
    # A plain CPU tensor passes every new type/attribute check and fails only
    # the CUDA residency rule -- proving the new checks admit plain tensors.
    payload = load_payload_module()
    with pytest.raises(ValueError, match="CUDA"):
        payload.payload_nbytes(torch.zeros(4))


def test_payload_counts_plain_control_like_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    # The paper controls (identity, rtn-int2, rtn-int4) return plain tensors and
    # small scalar metadata.  Monkeypatch only the CUDA residency rule so the
    # full counter runs on a CPU tensor tree and confirms those outputs pass the
    # new type/attribute rules and are charged their full storage once.
    payload = load_payload_module()
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    q = torch.zeros((1, 2, 3, 8), dtype=torch.uint8)
    s = torch.zeros((1, 2, 3, 1), dtype=torch.uint8)
    packed_k = {"q": q, "s": s, "bits": 2, "shape": (1, 2, 3, 128)}
    packed_v = {"q": q, "s": s, "bits": 2, "shape": (1, 2, 3, 128)}
    # Shared storage once, plus each dict's keys (q, s, bits, shape = 11 bytes)
    # and scalars (bits = 8, four shape ints = 32).
    expected = q.untyped_storage().nbytes() + s.untyped_storage().nbytes() + 2 * (11 + 8 + 32)
    assert payload.payload_nbytes((packed_k, packed_v)) == expected
