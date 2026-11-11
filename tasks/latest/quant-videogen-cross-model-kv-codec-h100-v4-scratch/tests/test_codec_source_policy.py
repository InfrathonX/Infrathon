from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "policy/codec_source_policy.py"
ORACLE = ROOT / "oracle"
REPOSITORY_ROOT = next(parent for parent in ROOT.parents if (parent / "sforge").is_dir())
CONTROL_CODECS = ROOT / "tests/fixtures/control_codecs"


def load_policy():
    spec = importlib.util.spec_from_file_location("qvg_h100_scratch_policy", POLICY_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_submission(tmp_path: Path) -> Path:
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "mlsysforge_codec.py").write_text(
        """from __future__ import annotations
import torch

def encode(k, v):
    return {"data": k.to(torch.int8)}, {"data": v.to(torch.int8)}

def decode(packed_k, packed_v):
    return packed_k["data"].to(torch.bfloat16), packed_v["data"].to(torch.bfloat16)
""",
        encoding="utf-8",
    )
    return submission


def append_source(path: Path, source: str) -> None:
    path.write_text(path.read_text(encoding="utf-8") + source, encoding="utf-8")


def test_minimal_scratch_codec_passes_policy(tmp_path: Path) -> None:
    result = load_policy().validate_submission(make_submission(tmp_path))
    assert result == {
        "valid": True,
        "policy_schema_version": 4,
        "files": 1,
        "python_files": 1,
        "checked_files": 1,
        "source_bytes": result["source_bytes"],
        "trusted_tree": False,
    }


def test_exact_private_oracle_tree_is_trusted() -> None:
    result = load_policy().validate_submission(ORACLE)
    assert result["valid"] is True
    assert result["trusted_tree"] is True
    assert result["checked_files"] == 0


@pytest.mark.parametrize(
    "payload,fragment",
    [
        ("\nopen('ambient-state', 'w')\n", "ambient call"),
        ("\nimport subprocess\n", "outside the codec allowlist"),
        ("\nSTASH = []\n", "module or class state"),
        ("\ndef copy_out(k):\n    return k.cpu()\n", "offload"),
        ("\ndef raw_pointer(k):\n    return k.data_ptr()\n", "serialization/offload"),
        ("\nLOCATION = '/eval/protected/result'\n", "protected evaluator surface"),
        ("\ndef mutate_global(k):\n    global STASH\n    STASH = k\n", "module-global"),
        # The sharded Judge's other-device check reads the allocator's peaks.
        ("\ndef move(k):\n    torch.cuda.set_device(1)\n", "serialization/offload"),
        ("\ndef hide(k):\n    torch.cuda.reset_peak_memory_stats(1)\n", "serialization/offload"),
        ("\ndef hide(k):\n    torch.cuda.memory.reset_max_memory_cached()\n", "serialization/offload"),
        ("\ndef move(k):\n    with torch.cuda.device(1):\n        return k\n", "current CUDA device"),
    ],
)
def test_policy_rejects_ambient_state_and_bypasses(
    tmp_path: Path, payload: str, fragment: str
) -> None:
    policy = load_policy()
    submission = make_submission(tmp_path)
    append_source(submission / "mlsysforge_codec.py", payload)
    with pytest.raises(policy.PolicyError, match=fragment):
        policy.validate_submission(submission)


def test_policy_accepts_optional_codec_package_and_notes(tmp_path: Path) -> None:
    submission = make_submission(tmp_path)
    codec = submission / "codec"
    codec.mkdir()
    (codec / "__init__.py").write_text("BITS = 4\n", encoding="utf-8")
    (submission / "notes.md").write_text("simple int8 baseline\n", encoding="utf-8")
    result = load_policy().validate_submission(submission)
    assert result["python_files"] == 2


def test_policy_accepts_normal_cuda_helpers_and_triton_device_load(tmp_path: Path) -> None:
    submission = make_submission(tmp_path)
    codec = submission / "codec"
    codec.mkdir()
    (codec / "kernel.py").write_text(
        """import collections
import functools
import itertools
import operator
import torch
import triton
import triton.language as tl

@triton.jit
def copy_kernel(source, target, count: tl.constexpr):
    offsets = tl.arange(0, count)
    values = tl.load(source + offsets)
    tl.store(target + offsets, values)

def synchronize():
    torch.cuda.synchronize()
""",
        encoding="utf-8",
    )
    result = load_policy().validate_submission(submission)
    assert result["python_files"] == 2


def test_policy_accepts_triton_language_from_import(tmp_path: Path) -> None:
    submission = make_submission(tmp_path)
    codec = submission / "codec"
    codec.mkdir()
    (codec / "kernel.py").write_text(
        """import triton
from triton import language as tl

@triton.jit
def read_kernel(source, count: tl.constexpr):
    return tl.load(source + tl.arange(0, count))
""",
        encoding="utf-8",
    )
    assert load_policy().validate_submission(submission)["python_files"] == 2


def test_policy_ignores_generated_python_cache_files(tmp_path: Path) -> None:
    submission = make_submission(tmp_path)
    root_cache = submission / "__pycache__"
    nested_cache = submission / "codec/__pycache__"
    root_cache.mkdir()
    nested_cache.mkdir(parents=True)
    (root_cache / "mlsysforge_codec.cpython-311.pyc").write_bytes(b"generated")
    (nested_cache / "kernel.cpython-312.pyo").write_bytes(b"generated")
    result = load_policy().validate_submission(submission)
    assert result["files"] == 1
    assert result["python_files"] == 1


def test_policy_accepts_local_helper_class_state(tmp_path: Path) -> None:
    submission = make_submission(tmp_path)
    codec = submission / "codec"
    codec.mkdir()
    (codec / "quantizer.py").write_text(
        """class Quantizer:
    def __init__(self, bits):
        self.bits = bits

    def encode(self, tensor):
        self.last_shape = tensor.shape
        return tensor
""",
        encoding="utf-8",
    )
    assert load_policy().validate_submission(submission)["python_files"] == 2


def test_policy_still_rejects_non_triton_load(tmp_path: Path) -> None:
    policy = load_policy()
    submission = make_submission(tmp_path)
    append_source(
        submission / "mlsysforge_codec.py",
        "\ndef deserialize(torch, path):\n    return torch.load(path)\n",
    )
    with pytest.raises(policy.PolicyError, match="serialization/offload"):
        policy.validate_submission(submission)


def test_policy_rejects_files_outside_source_closure(tmp_path: Path) -> None:
    policy = load_policy()
    submission = make_submission(tmp_path)
    (submission / "payload.bin").write_bytes(b"not codec source")
    with pytest.raises(policy.PolicyError, match="outside the codec source closure"):
        policy.validate_submission(submission)


def test_policy_rejects_links(tmp_path: Path) -> None:
    policy = load_policy()
    submission = make_submission(tmp_path)
    codec = submission / "codec"
    codec.mkdir()
    (codec / "escape.py").symlink_to("/etc/passwd")
    with pytest.raises(policy.PolicyError, match="links are forbidden"):
        policy.validate_submission(submission)


def test_policy_requires_only_the_two_function_abi(tmp_path: Path) -> None:
    policy = load_policy()
    submission = make_submission(tmp_path)
    (submission / "mlsysforge_codec.py").write_text(
        "def encode(k, v):\n    return k, v\n", encoding="utf-8"
    )
    with pytest.raises(policy.PolicyError, match="encode and decode"):
        policy.validate_submission(submission)


@pytest.mark.parametrize("terms", [500, 700, 5000, 100_000])
def test_deeply_nested_source_is_a_policy_rejection(tmp_path: Path, terms: int) -> None:
    """A source too deep to parse or check is the submission's rejection, never an evaluator fault."""
    policy = load_policy()
    submission = make_submission(tmp_path)
    chain = " + ".join(["k"] * terms)
    (submission / "mlsysforge_codec.py").write_text(
        f"def encode(k, v):\n    x = {chain}\n    return x, v\n\n"
        "def decode(packed_k, packed_v):\n    return packed_k, packed_v\n",
        encoding="utf-8",
    )
    with pytest.raises(policy.PolicyError, match="cannot (parse|check) mlsysforge_codec.py"):
        policy.validate_submission(submission)


def test_controller_admission_rejects_a_deep_source_as_a_candidate_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sys.path.insert(0, str(ROOT / "judge"))
    import evaluate_hidden

    monkeypatch.setattr(evaluate_hidden, "SOURCE_POLICY", POLICY_PATH)
    submission = make_submission(tmp_path)
    chain = " + ".join(["k"] * 700)
    (submission / "mlsysforge_codec.py").write_text(
        f"def encode(k, v):\n    x = {chain}\n    return x, v\n\n"
        "def decode(packed_k, packed_v):\n    return packed_k, packed_v\n",
        encoding="utf-8",
    )
    with pytest.raises(evaluate_hidden.CandidateFailure) as caught:
        evaluate_hidden.admit_submission(submission)
    assert evaluate_hidden.failure_kind(caught.value) == "candidate_failure"


# ---------------------------------------------------------------------------
# v4: close the two reviewer bypasses -- stdlib re-export surfaces reachable
# through an allowed import root, and alias laundering of a live module object.
# ---------------------------------------------------------------------------

REVIEWER_BYPASSES = [
    # (1) torch.sys.modules rebinding the in-process KV accounting.
    (
        "\ndef rebind(k):\n"
        "    torch.sys.modules['hyvideo.modules.attention']._qvg_pass_stats = k\n",
        "torch.sys",
    ),
    # (1) torch re-exports stdlib modules as plain attributes.
    ("\ndef esc(k):\n    torch.os.system('id')\n    return k\n", "standard-library module: torch.os"),
    (
        "\ndef esc(k):\n    return torch.importlib.import_module('socket')\n",
        "standard-library module: torch.importlib",
    ),
    ("\ndef esc(k):\n    return torch.ctypes.CDLL('libc.so.6')\n", "standard-library module: torch.ctypes"),
    # (1) torch.builtins.getattr is a master key even as a chained call.
    (
        "\ndef esc(k):\n    return torch.builtins.getattr(k, 'storage')\n",
        "ambient call: torch.builtins.getattr",
    ),
    ("\ndef esc(k):\n    return torch.builtins.setattr(k, 'x', 1)\n", "ambient call: torch.builtins.setattr"),
    # (1) nested stdlib re-export (torch.cuda.os).
    ("\ndef esc(k):\n    torch.cuda.os.system('id')\n    return k\n", "standard-library module: torch.os"),
    ("\ndef esc(k):\n    return torch.inspect.stack()\n", "standard-library module: torch.inspect"),
    ("\ndef esc(k):\n    return torch.threading.Thread\n", "standard-library module: torch.threading"),
    ("\ndef esc(k):\n    return torch.platform.system()\n", "standard-library module: torch.platform"),
    # (2) alias laundering: binding the live module out of the checked chain.
    ("\ndef esc(k):\n    t = torch\n    return t._C\n", "laundering"),
    (
        "\ndef esc(k):\n    c = torch.cuda\n    with c.device(1):\n        return k\n",
        "laundering",
    ),
    ("\ndef esc(k):\n    d = {'t': torch}\n    return d['t']._C\n", "laundering"),
    ("\ndef esc(t=torch):\n    return t.sys\n", "laundering"),
    ("\ndef esc(k):\n    g = lambda t=torch: t.sys\n    return g\n", "laundering"),
    ("\ndef esc(k):\n    return (t := torch).sys\n", "laundering"),
    ("\ndef esc(k):\n    for t in [torch]:\n        t.os.system('id')\n    return k\n", "laundering"),
    # (2) from-imports off torch stay rejected outright.
    ("\nfrom torch import cuda\ndef esc(k):\n    cuda.set_device(1)\n    return k\n", "from-imports from torch"),
]


@pytest.mark.parametrize("payload,fragment", REVIEWER_BYPASSES)
def test_policy_rejects_reviewer_bypasses(tmp_path: Path, payload: str, fragment: str) -> None:
    policy = load_policy()
    submission = make_submission(tmp_path)
    append_source(submission / "mlsysforge_codec.py", payload)
    with pytest.raises(policy.PolicyError, match=fragment):
        policy.validate_submission(submission)


LEGAL_IDIOMS = [
    # dtype value-reads on an import alias are allowed (explicit leaf allowlist).
    "\ndef dt(k):\n    return k.to(torch.bfloat16), k.to(torch.int8), k.to(torch.uint8)\n",
    "\ndef mk(k):\n    return torch.empty(k.shape, dtype=torch.uint8, device=k.device)\n",
    "\ndef check(k):\n    assert isinstance(k, torch.Tensor)\n    return k\n",
    "\ndef sync(k):\n    torch.cuda.synchronize()\n    return k\n",
]


@pytest.mark.parametrize("payload", LEGAL_IDIOMS)
def test_policy_admits_common_alias_idioms(tmp_path: Path, payload: str) -> None:
    submission = make_submission(tmp_path)
    append_source(submission / "mlsysforge_codec.py", payload)
    assert load_policy().validate_submission(submission)["valid"] is True


def test_policy_admits_torch_functional_and_triton_language_aliases(tmp_path: Path) -> None:
    submission = make_submission(tmp_path)
    codec = submission / "codec"
    codec.mkdir()
    (codec / "kernel.py").write_text(
        """import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def pad_kernel(source, count: tl.constexpr):
    offsets = tl.arange(0, count)
    tl.store(source + offsets, tl.load(source + offsets))


def pad(x):
    return F.pad(x, (0, 1))
""",
        encoding="utf-8",
    )
    assert load_policy().validate_submission(submission)["python_files"] == 2


@pytest.mark.parametrize("control", ["identity", "rtn-int2", "rtn-int4"])
def test_control_codecs_are_admitted(control: str) -> None:
    """The paper-alignment control codecs must stay admissible under v4."""
    result = load_policy().validate_submission(CONTROL_CODECS / control)
    assert result["valid"] is True
    assert result["trusted_tree"] is False


def _patch_freeze(monkeypatch: pytest.MonkeyPatch, module) -> None:
    # The Judge freezes the admitted tree with os.chown(0, 0), which needs root;
    # the policy decision runs before it, so neutralise only the freeze here.
    monkeypatch.setattr(module, "_freeze_tree", lambda *a, **k: None)


@pytest.mark.parametrize("control", ["identity", "rtn-int2", "rtn-int4"])
def test_controls_and_oracle_admitted_through_judge_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, control: str
) -> None:
    sys.path.insert(0, str(ROOT / "judge"))
    import evaluate_hidden

    monkeypatch.setattr(evaluate_hidden, "SOURCE_POLICY", POLICY_PATH)
    _patch_freeze(monkeypatch, evaluate_hidden)
    assert evaluate_hidden.admit_submission(CONTROL_CODECS / control)["valid"] is True


def test_oracle_tree_admitted_through_judge_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sys.path.insert(0, str(ROOT / "judge"))
    import evaluate_hidden

    monkeypatch.setattr(evaluate_hidden, "SOURCE_POLICY", POLICY_PATH)
    _patch_freeze(monkeypatch, evaluate_hidden)
    result = evaluate_hidden.admit_submission(ORACLE)
    assert result["valid"] is True and result["trusted_tree"] is True


def test_reviewer_bypass_rejected_as_candidate_failure_through_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sys.path.insert(0, str(ROOT / "judge"))
    import evaluate_hidden

    monkeypatch.setattr(evaluate_hidden, "SOURCE_POLICY", POLICY_PATH)
    _patch_freeze(monkeypatch, evaluate_hidden)
    submission = make_submission(tmp_path)
    append_source(
        submission / "mlsysforge_codec.py",
        "\ndef esc(k):\n    return torch.os.system('id')\n",
    )
    with pytest.raises(evaluate_hidden.CandidateFailure):
        evaluate_hidden.admit_submission(submission)


# ---------------------------------------------------------------------------
# v4: close five reviewer gaps in the source contract -- memoizing decorators
# (persistent hidden state), autograd side channels, indirect name-driven
# lookup, dunder-like string literals, and subclassing torch/triton types --
# plus wildcard-import hardening that keeps the operator/functools surface
# statically visible.  Rule 1 (the payload byte counter) is exercised in
# tests/test_payload_runtime.py.
# ---------------------------------------------------------------------------

NEW_RULE_REJECTIONS = [
    # (2) memoizing decorators keep objects alive across encode calls.
    ("\nimport functools\n@functools.lru_cache\ndef memo(k):\n    return k\n", "memoizer"),
    ("\nimport functools\n@functools.cache\ndef memo(k):\n    return k\n", "memoizer"),
    (
        "\nimport functools\nclass Keeper:\n    @functools.cached_property\n    def value(self):\n        return 1\n",
        "memoizer",
    ),
    ("\nfrom functools import lru_cache\n@lru_cache(maxsize=8)\ndef memo(k):\n    return k\n", "memoizer"),
    ("\nfrom functools import cache as keep\n@keep\ndef memo(k):\n    return k\n", "memoizer"),
    # (3) autograd side channels.
    ("\ndef g(k):\n    with torch.enable_grad():\n        return k\n", "autograd"),
    ("\ndef g(k):\n    torch.set_grad_enabled(True)\n    return k\n", "autograd"),
    ("\nimport torch.autograd\n", "forbidden torch surface"),
    ("\ndef g(k):\n    torch.autograd.backward(k)\n    return k\n", "torch surface"),
    ("\ndef g(k):\n    k.requires_grad_(True)\n    return k\n", "requires_grad_"),
    ("\ndef g(k):\n    return torch.empty(k.shape, device=k.device, requires_grad=True)\n", "requires_grad="),
    ("\ndef g(k):\n    k.requires_grad = True\n    return k\n", "autograd side channel"),
    ("\ndef g(k):\n    k.grad = k\n    return k\n", "autograd side channel"),
    # (4) indirect, name-driven attribute access / mutation and dunder literals.
    ("\nimport operator\ndef g(k):\n    return operator.attrgetter('storage')(k)\n", "indirect-lookup"),
    ("\nimport operator\ndef g(k):\n    return operator.methodcaller('cpu')(k)\n", "indirect-lookup"),
    ("\nimport operator\ndef g(k):\n    operator.setitem({}, 'x', k)\n    return k\n", "indirect-lookup"),
    ("\nimport operator\ndef g(k):\n    operator.delitem({'x': 1}, 'x')\n    return k\n", "indirect-lookup"),
    ("\nfrom operator import attrgetter\ndef g(k):\n    return attrgetter('storage')(k)\n", "indirect-lookup"),
    ("\ndef g(k):\n    label = '__globals__'\n    return k, label\n", "dunder-like string"),
    # (5) subclassing / metaclassing torch or triton types.
    ("\nclass Sneaky(torch.Tensor):\n    pass\n", "subclassing a torch"),
    ("\nimport triton\nclass Sneaky(triton.language.tensor):\n    pass\n", "subclassing a torch/triton"),
    ("\nclass Sneaky(metaclass=torch.Tensor):\n    pass\n", "metaclass"),
    # hardening: a wildcard import hides which names it binds.
    ("\nfrom operator import *\n", "wildcard"),
    ("\nfrom functools import *\n", "wildcard"),
]


@pytest.mark.parametrize("payload,fragment", NEW_RULE_REJECTIONS)
def test_policy_rejects_v4_new_rule_gaps(tmp_path: Path, payload: str, fragment: str) -> None:
    policy = load_policy()
    submission = make_submission(tmp_path)
    append_source(submission / "mlsysforge_codec.py", payload)
    with pytest.raises(policy.PolicyError, match=fragment):
        policy.validate_submission(submission)


NEW_RULE_LEGAL = [
    # functools/operator may be imported and used for non-memoizing,
    # non-name-driven work.
    "\nimport functools\ndef g(k):\n    return functools.reduce(lambda a, b: a, [k, k])\n",
    "\nimport functools\nimport operator\ndef g(k):\n    return k\n",
    # Subclassing a non-torch/triton allowed type stays fine.
    "\nfrom enum import Enum\nclass Bits(Enum):\n    LOW = 2\n    HIGH = 4\n",
    # detach() drops grad and is a plain, allowed tensor op.
    "\ndef g(k):\n    return k.detach()\n",
]


@pytest.mark.parametrize("payload", NEW_RULE_LEGAL)
def test_policy_admits_v4_legal_idioms(tmp_path: Path, payload: str) -> None:
    submission = make_submission(tmp_path)
    append_source(submission / "mlsysforge_codec.py", payload)
    assert load_policy().validate_submission(submission)["valid"] is True


def test_v4_new_rule_gap_rejected_as_candidate_failure_through_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sys.path.insert(0, str(ROOT / "judge"))
    import evaluate_hidden

    monkeypatch.setattr(evaluate_hidden, "SOURCE_POLICY", POLICY_PATH)
    _patch_freeze(monkeypatch, evaluate_hidden)
    submission = make_submission(tmp_path)
    append_source(
        submission / "mlsysforge_codec.py",
        "\nimport functools\n@functools.lru_cache\ndef memo(k):\n    return k\n",
    )
    with pytest.raises(evaluate_hidden.CandidateFailure) as caught:
        evaluate_hidden.admit_submission(submission)
    assert evaluate_hidden.failure_kind(caught.value) == "candidate_failure"
