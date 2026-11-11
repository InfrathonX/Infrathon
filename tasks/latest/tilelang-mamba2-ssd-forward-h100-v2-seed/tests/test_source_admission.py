"""The admission policy, exercised against a real submission and hostile edits.

Every hostile variant below is a concrete route out of the tile DSL, taken from
an inventory of the escape routes in the pinned 0.1.13 wheel.
The point of the test is not that each string is forbidden -- it is that a
submission which reaches native code, an existing kernel, the compiler, or the
harness never gets executed.
"""

from __future__ import annotations

import pytest

from conftest import TASK_ROOT, _load

admission = _load("ssd_source_admission", "images/judge/source_admission.py")

SEED = TASK_ROOT / "images/work/workdir/mamba2_ssd_lab/candidates/kernel_v001/submission.py"

HOSTILE = {
    # raw source and external kernels
    "inline_cuda_source": "import tilelang.language as T\nk = T.CUDASourceCodeKernel(source_code_or_path='__global__ void f(){}')\n",
    "import_source_pragma": "import tilelang.language as T\ndef f():\n    T.import_source('extern \"C\" void g();')\n",
    "kernel_prelude": "import tilelang.language as T\ndef f():\n    with T.Kernel(1, prelude='#include <x.h>'):\n        pass\n",
    "call_kernel_triton": "import tilelang.language as T\ndef f(k):\n    T.call_kernel(k, (1,))\n",
    # external / packed / intrinsic calls
    "call_extern": "import tilelang.language as T\ndef f():\n    return T.call_extern('float', 'my_gemm')\n",
    "call_packed": "import tilelang.language as T\ndef f():\n    return T.call_packed('x')\n",
    "call_intrin": "import tilelang.language as T\ndef f():\n    return T.call_intrin('float', 'x')\n",
    # raw hardware intrinsics, including behind an alias
    "raw_ptx": "import tilelang.language as T\ndef f():\n    T.ptx_ldmatrix('float16', False, 4, '.b16', 0, 0, 0, 0)\n",
    "ptx_alias": "from tilelang.language import ptx_wgmma_ss as w\ndef f():\n    w()\n",
    "mma_intrinsic": "import tilelang.language as T\ndef f():\n    T.mma_sync()\n",
    # pointer fabrication
    "address_of": "import tilelang.language as T\ndef f(a):\n    return T.address_of(a)\n",
    "tensor_from_addr": "import tilelang.language as T\ndef f(p):\n    return T.make_tensor_from_addr(p, (1,))\n",
    # compiler redirection
    "compile_flags": "import tilelang\n@tilelang.jit(compile_flags=['-I/tmp'])\ndef f():\n    pass\n",
    "compile_flag_pass_key": "import tilelang\n@tilelang.jit(pass_configs={'tl.device_compile_flags': '-I/tmp'})\ndef f():\n    pass\n",
    "custom_lower_pass": "import tilelang\n@tilelang.jit(pass_configs={'tirx.add_lower_pass': None})\ndef f():\n    pass\n",
    "codegen_postproc": "import tilelang\ntilelang.register_cuda_postproc(lambda src, t: src)\n",
    "postproc_callback": "from tilelang.engine.callback import register_cuda_postproc_callback as r\n@r\ndef f(src, t):\n    return src\n",
    "opaque_pass_configs": "import tilelang\ndef build(cfg):\n    return tilelang.jit(pass_configs=cfg)\n",
    # backend and target
    "cutedsl_backend": "import tilelang\n@tilelang.jit(execution_backend='cutedsl')\ndef f():\n    pass\n",
    "target_override": "import tilelang\n@tilelang.jit(target='hip')\ndef f():\n    pass\n",
    # prebuilt artefacts and bundled TVM
    "from_database": "import tilelang\ndef f():\n    return tilelang.JITKernel.from_database(None, '', '', '/tmp/x.so', [], None, None, None, 'cython')\n",
    "bundled_tvm": "import tvm\ndef f():\n    return tvm.get_global_func('x')\n",
    "tvm_via_tilelang": "import tilelang\ndef f():\n    return tilelang.tvm.runtime.load_module('/tmp/x.so')\n",
    "contrib_nvcc": "from tilelang.contrib import nvcc\ndef f(src):\n    return nvcc.compile_cuda(src)\n",
    # existing kernels
    "torch_matmul": "import torch\ndef f(a, b):\n    return torch.matmul(a, b)\n",
    "matmul_via_alias": "import torch\nmm = torch.matmul\ndef f(a, b):\n    return mm(a, b)\n",
    "matmul_operator": "def f(a, b):\n    return a @ b\n",
    "einsum": "import torch\ndef f(a, b):\n    return torch.einsum('ij,jk->ik', a, b)\n",
    "mamba_ssm": "from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined as m\ndef f(*a):\n    return m(*a)\n",
    "triton": "import triton.language as tl\ndef f():\n    return tl.dot\n",
    # generic Python escapes
    "ctypes": "import ctypes\ndef f():\n    return ctypes.CDLL('/tmp/x.so')\n",
    "cpp_extension": "import torch\ndef f():\n    return torch.utils.cpp_extension.load_inline(name='x', cpp_sources='')\n",
    "torch_ops": "import torch\ndef f():\n    return torch.ops.mylib.myop\n",
    "dynamic_getattr": "import tilelang\ndef f(n):\n    return getattr(tilelang, n)\n",
    "star_import": "from tilelang.language import *\n",
    "unlisted_import": "import subprocess\n",
    # harness tampering
    "clock_rebinding": "import torch\ntorch.cuda.synchronize = lambda *a, **k: None\n",
    "env_mutation": "import os\nos.environ['TILELANG_CACHE_DIR'] = '/tmp/mine'\n",
    "env_putenv": "import os\ndef f():\n    os.putenv('TL_CUTLASS_PATH', '/tmp')\n",
    "harness_import": "import mamba2_ssd_runtime\n",
    "private_stream": "import torch\ndef f():\n    return torch.cuda.stream(torch.cuda.Stream())\n",
    # autotuning inside a submission
    "runtime_autotune": "import tilelang\n@tilelang.autotune(configs=[{'a': 1}])\ndef f():\n    pass\n",
    # v2: routes found by probing the v1 rules (2026-10-06)
    "dunder_matmul_method": "def f(a, b):\n    return a.__matmul__(b)\n",
    "dunder_matmul_unbound": "import torch\ndef f(a, b):\n    return torch.Tensor.__matmul__(a, b)\n",
    "dunder_via_type": "def f(a, b):\n    return type(a).__matmul__(a, b)\n",
    "module_dict_lookup": "import tilelang.language as T\ndef f():\n    return T.__dict__['call_extern']\n",
    "operator_attrgetter": "import operator\nimport tilelang.language as T\ndef f():\n    return operator.attrgetter('call_extern')(T)\n",
    "operator_methodcaller": "from operator import methodcaller\ndef f(a):\n    return methodcaller('exp')(a)\n",
    "getattr_as_value": "import functools\nimport tilelang\ndef f(n):\n    return functools.partial(getattr, tilelang)(n)\n",
    "torch_linalg": "import torch\ndef f(a, b):\n    return torch.linalg.multi_dot([a, b])\n",
    "torch_addr": "import torch\ndef f(s, a, b):\n    return torch.addr(s, a, b)\n",
    "torch_exp_cumsum": "import torch\ndef f(a):\n    return torch.cumsum(torch.exp(a), 0)\n",
    "torch_nn_functional": "import torch\ndef f(a):\n    return torch.nn.functional.pad(a, (0, 1))\n",
    "torch_private_namespace": "import torch\ndef f():\n    return torch._C\n",
    "torch_dispatch_mode_import": "from torch.utils._python_dispatch import TorchDispatchMode\n",
    "torch_name_import": "from torch import cumsum\n",
    "torch_memory_snapshot": "import torch\ndef f():\n    return torch.cuda.memory_snapshot()\n",
    "time_import": "import time\n",
}

# Ordinary code a TileLang submission legitimately contains; none may be rejected.
LEGITIMATE = {
    "allocation_surface": (
        "import torch\n"
        "def f(x):\n"
        "    a = torch.empty((2, 3), device=x.device, dtype=torch.bfloat16)\n"
        "    b = torch.zeros_like(x)\n"
        "    c = torch.full((4,), 0.0, dtype=torch.float32, device=x.device)\n"
        "    s = torch.cuda.current_stream()\n"
        "    return a, b, c, s, x.view(-1), x.permute(0, 1), x.shape, torch.float64\n"
    ),
    "main_guard": "if __name__ == '__main__':\n    pass\n",
    "no_grad_decorator": "import torch\n@torch.no_grad()\ndef f(x):\n    return x\n",
    "alloc_tmem": "import tilelang.language as T\ndef f():\n    return T.alloc_tmem\n",
}


@pytest.fixture
def workspace(tmp_path):
    def build(extra: str = "") -> object:
        candidate = tmp_path / "candidates/kernel_v001"
        candidate.mkdir(parents=True, exist_ok=True)
        (candidate / "submission.py").write_text(SEED.read_text() + "\n\n" + extra)
        (tmp_path / "FINAL_VERSION").write_text("1\n")
        (tmp_path / "notes.md").write_text("# notes\n")
        return admission.validate_submission(tmp_path)

    return build


def test_seed_is_admitted(workspace) -> None:
    report = workspace()
    assert report["ok"], report["errors"]
    assert report["candidate_version"] == 1
    assert report["selected_sha256"]


@pytest.mark.parametrize("label", sorted(HOSTILE))
def test_hostile_variant_is_rejected(workspace, label: str) -> None:
    report = workspace(HOSTILE[label])
    assert not report["ok"], f"{label} was admitted"


def test_seed_defines_the_entry_point() -> None:
    assert "def ssd_forward(" in SEED.read_text()


def test_binary_and_symlink_entries_are_rejected(tmp_path) -> None:
    candidate = tmp_path / "candidates/kernel_v001"
    candidate.mkdir(parents=True)
    (candidate / "submission.py").write_text("def ssd_forward(*a):\n    return None\n")
    (candidate / "kernel.so").write_bytes(b"\x7fELF")
    (tmp_path / "FINAL_VERSION").write_text("1\n")
    (tmp_path / "notes.md").write_text("")
    report = admission.validate_submission(tmp_path)
    assert not report["ok"]
    assert any("kernel.so" in error for error in report["errors"])


@pytest.mark.parametrize("label", sorted(LEGITIMATE))
def test_legitimate_variant_is_admitted(workspace, label: str) -> None:
    report = workspace(LEGITIMATE[label])
    assert report["ok"], (label, report["errors"])


def test_oracle_is_admitted(tmp_path) -> None:
    oracle = TASK_ROOT / "oracle/candidates/kernel_v001/submission.py"
    candidate = tmp_path / "candidates/kernel_v001"
    candidate.mkdir(parents=True)
    (candidate / "submission.py").write_text(oracle.read_text())
    (tmp_path / "FINAL_VERSION").write_text("1\n")
    (tmp_path / "notes.md").write_text("")
    report = admission.validate_submission(tmp_path)
    assert report["ok"], report["errors"]


def test_selected_oracle_archive_is_admitted(tmp_path) -> None:
    """The oracle archive (every candidate, FINAL_VERSION, notes.md) passes Judge admission."""
    import shutil

    oracle = TASK_ROOT / "oracle"
    version = int((oracle / "FINAL_VERSION").read_text(encoding="utf-8").strip())
    root = tmp_path / "archive"
    shutil.copytree(oracle / "candidates", root / "candidates")
    for member in ("FINAL_VERSION", "notes.md"):
        shutil.copyfile(oracle / member, root / member)
    report = admission.validate_submission(root)
    assert report["ok"], report["errors"]
    assert report["final_version"] == version


def test_judge_and_work_admission_copies_are_identical() -> None:
    judge = (TASK_ROOT / "images/judge/source_admission.py").read_bytes()
    work = (TASK_ROOT / "images/work/benchmark/source_admission.py").read_bytes()
    assert judge == work
