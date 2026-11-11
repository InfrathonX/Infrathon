from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from conftest import LAB_ROOT, TASK_ROOT
from runtime.source_policy import AdmissionError, admit


def _lab_with_candidate(tmp_path: Path, source: str) -> Path:
    lab = tmp_path / "orpo_lab"
    shutil.copytree(LAB_ROOT, lab)
    (lab / "solutions/objective_v001/candidate.py").write_text(source, encoding="utf-8")
    return lab


def test_seed_is_admitted():
    report = admit(LAB_ROOT)
    assert report == {"final_version": 1, "versions": [1]}


def test_forbidden_import_is_rejected(tmp_path):
    lab = _lab_with_candidate(tmp_path, "import os\n")
    with pytest.raises(AdmissionError, match="outside the allowlist"):
        admit(lab)


def test_symlink_payload_is_rejected(tmp_path):
    lab = tmp_path / "orpo_lab"
    shutil.copytree(LAB_ROOT, lab)
    candidate = lab / "solutions/objective_v001/candidate.py"
    candidate.unlink()
    candidate.symlink_to("/etc/passwd")
    with pytest.raises(AdmissionError):
        admit(lab)


@pytest.mark.parametrize(
    "source",
    [
        "import torch\ntorch.cuda.synchronize = lambda *_: None\n",
        "import torch\ntorch.backends.cuda.matmul.allow_tf32 = True\n",
        "class Objective:\n    def mutate(self):\n        self.control.synchronize = None\n",
        "import triton\ntriton.runtime.driver.active = None\n",
    ],
)
def test_external_or_nested_attribute_writes_are_rejected(tmp_path, source):
    lab = _lab_with_candidate(tmp_path, source)
    with pytest.raises(AdmissionError, match="external or nested attributes"):
        admit(lab)


@pytest.mark.parametrize(
    "source",
    [
        "import torch\ntorch.cuda.reset_peak_memory_stats()\n",
        "import torch\ntorch.cuda.memory.reset_peak_memory_stats()\n",
        "import torch\ntorch.cuda.max_memory_reserved()\n",
        "import torch\ntorch.cuda.synchronize()\n",
        "from torch import cuda as accelerator\naccelerator.Event()\n",
        "from torch.cuda import Event as TimerEvent\nTimerEvent()\n",
        "from torch.cuda import reset_peak_memory_stats as reset\nreset()\n",
        "from torch.cuda.memory import reset_peak_memory_stats as reset\nreset()\n",
        "import torch as framework\nstream = framework.cuda.current_stream()\n",
        "import torch\ngraph = torch.cuda.CUDAGraph()\n",
        "import torch\nfree, total = torch.cuda.mem_get_info()\n",
    ],
)
def test_cuda_measurement_and_control_access_is_rejected(tmp_path, source):
    lab = _lab_with_candidate(tmp_path, source)
    with pytest.raises(AdmissionError, match="control-plane"):
        admit(lab)


def test_cuda_namespace_cannot_escape_through_a_default_argument(tmp_path):
    source = """\
from torch import cuda

def tamper(control=cuda):
    control.reset_peak_memory_stats()
"""
    lab = _lab_with_candidate(tmp_path, source)
    with pytest.raises(AdmissionError, match="capturing the torch.cuda namespace"):
        admit(lab)


@pytest.mark.parametrize(
    "source",
    [
        "import torch.distributed as dist\ndist.barrier()\n",
        "from torch import distributed as dist\ndist.all_reduce(None)\n",
        "import torch\ntorch.distributed.destroy_process_group()\n",
        "from torch.utils import benchmark\n",
        "import torch\nwith torch.profiler.profile():\n    pass\n",
        "import torch\nwith torch.autograd.profiler.profile():\n    pass\n",
        "import torch\ntorch._C._cuda_resetPeakMemoryStats()\n",
        "from torch.utils.cpp_extension import load\n",
        "import time\ntime.perf_counter()\n",
    ],
)
def test_distributed_profiler_and_time_control_is_rejected(tmp_path, source):
    lab = _lab_with_candidate(tmp_path, source)
    with pytest.raises(AdmissionError):
        admit(lab)


def test_direct_self_and_autograd_context_state_remain_allowed(tmp_path):
    source = """\
import torch

class LocalObjective(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 2.0

class LocalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.scale = 2.0
        return value
"""
    assert admit(_lab_with_candidate(tmp_path, source))["final_version"] == 1


def test_torch_compile_amp_and_triton_kernel_surface_remain_allowed(tmp_path):
    source = """\
import torch
import triton
import triton.language as tl

def eager(value):
    return value.sin()

compiled = torch.compile(eager, fullgraph=True)
cuda_available = torch.cuda.is_available()

@triton.jit
def kernel(source, destination, size: tl.constexpr):
    offsets = tl.arange(0, size)
    values = tl.load(source + offsets)
    tl.store(destination + offsets, values)
"""
    assert admit(_lab_with_candidate(tmp_path, source))["final_version"] == 1


def test_declarative_qwen3_patch_spec_is_source_admitted(tmp_path):
    source = """\
import torch
from runtime.api import Qwen3ModelPatch

def rope(query, key, cos, sin, unsqueeze_dim=1):
    return query, key

def rms_norm_forward(self, hidden_states):
    return hidden_states * self.weight

def swiglu_forward(self, hidden_states):
    return torch.nn.functional.linear(hidden_states, self.weight)

def build_model_patch(contract):
    return Qwen3ModelPatch(
        rope=rope,
        rms_norm_forward=rms_norm_forward,
        swiglu_forward=swiglu_forward,
    )
"""
    assert admit(_lab_with_candidate(tmp_path, source))["final_version"] == 1


@pytest.mark.parametrize(
    "imperative_export",
    [
        "def optimize_model(model, contract):\n    return model\n",
        "optimize_model = lambda model, contract: model\n",
        (
            "def __getattr__(name):\n"
            "    if name == 'optimize_model':\n"
            "        return lambda model, contract: model\n"
            "    raise AttributeError(name)\n"
        ),
        (
            "def dynamic_export(name):\n"
            "    if name == 'optimize_model':\n"
            "        return lambda model, contract: model\n"
            "    raise AttributeError(name)\n"
            "__getattr__ = dynamic_export\n"
        ),
    ],
)
def test_participant_imperative_model_hook_is_rejected(tmp_path, imperative_export):
    source = "import torch\n\n" + imperative_export
    lab = _lab_with_candidate(tmp_path, source)
    with pytest.raises(AdmissionError, match="optimize_model|__getattr__"):
        admit(lab)


def test_public_and_judge_policy_bridges_are_byte_identical():
    public = LAB_ROOT / "runtime/source_policy.py"
    judge = TASK_ROOT / "images/judge/verifier/source_admission.py"
    assert public.read_bytes() == judge.read_bytes()
