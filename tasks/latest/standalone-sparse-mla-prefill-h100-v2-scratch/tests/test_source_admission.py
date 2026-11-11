from __future__ import annotations

import os
from pathlib import Path

import pytest

from conftest import TASK_ROOT, load_module


admission = load_module("standalone_source_admission", "images/judge/source_admission.py")


VALID_SOURCE = """\
import torch
import triton
import triton.language as tl

def sparse_mla_prefill(q, kv, indices, sm_scale, attn_sink, topk_length, out, max_logits, lse):
    return None
"""


def make_submission(root: Path, *, selected: int = 1, selected_source: str = VALID_SOURCE) -> Path:
    (root / "candidates/kernel_v001").mkdir(parents=True)
    (root / "candidates/kernel_v001/submission.py").write_text(
        selected_source if selected == 1 else "import torch\ndef rejected_experiment(a, b): return torch.mm(a, b)\n",
        encoding="utf-8",
    )
    if selected != 1:
        selected_dir = root / f"candidates/kernel_v{selected:03d}"
        selected_dir.mkdir()
        (selected_dir / "submission.py").write_text(selected_source, encoding="utf-8")
    (root / "FINAL_VERSION").write_text(f"{selected}\n", encoding="utf-8")
    (root / "notes.md").write_text("candidate notes\n", encoding="utf-8")
    return root


def test_normal_submission_scans_only_selected_version(tmp_path: Path) -> None:
    root = make_submission(tmp_path, selected=2)
    (root / "candidates/kernel_v001/submission.py").write_bytes(b"")
    result = admission.validate_submission(root)
    assert result["ok"] is True
    assert result["final_version"] == 2
    assert result["num_candidate_versions"] == 2
    assert result["candidate_version"] == 2
    assert result["selected_path"] == str(root / "candidates/kernel_v002/submission.py")
    assert result["selected_relative_path"] == "candidates/kernel_v002/submission.py"
    assert len(result["selected_sha256"]) == 64


def test_selected_existing_gemm_is_rejected(tmp_path: Path) -> None:
    root = make_submission(
        tmp_path,
        selected_source="import torch\ndef sparse_mla_prefill(a, b, *args): return torch.matmul(a, b)\n",
    )
    result = admission.validate_submission(root)
    assert result["ok"] is False
    assert any("existing GEMM API" in error for error in result["errors"])


def test_locally_authored_helper_name_is_not_a_style_failure(tmp_path: Path) -> None:
    root = make_submission(
        tmp_path,
        selected_source=(
            "def matmul(*args): return None\n"
            "def sparse_mla_prefill(*args): return matmul(*args)\n"
        ),
    )
    assert admission.validate_submission(root)["ok"] is True


def test_selected_empty_initial_candidate_is_invalid(tmp_path: Path) -> None:
    root = make_submission(tmp_path, selected_source="")
    result = admission.validate_submission(root)
    assert result["ok"] is False
    assert any("selected source must not be empty" in error for error in result["errors"])


def test_extra_path_and_symlink_fail_closed(tmp_path: Path) -> None:
    root = make_submission(tmp_path)
    (root / "unexpected.py").write_text("pass\n", encoding="utf-8")
    source = root / "candidates/kernel_v001/submission.py"
    source.unlink()
    source.symlink_to(root / "notes.md")
    result = admission.validate_submission(root)
    assert result["ok"] is False
    assert any("unexpected root entry" in error for error in result["errors"])
    assert any("symlink is forbidden" in error for error in result["errors"])


def test_selected_private_stream_is_rejected_without_style_rules(tmp_path: Path) -> None:
    root = make_submission(
        tmp_path,
        selected_source=(
            "import torch\n"
            "def sparse_mla_prefill(*args):\n"
            "    stream = torch.cuda.Stream()\n"
            "    return None\n"
        ),
    )
    result = admission.validate_submission(root)
    assert result["ok"] is False
    assert any("private CUDA stream API" in error for error in result["errors"])


def test_selected_trusted_harness_access_is_rejected(tmp_path: Path) -> None:
    sources = (
        "import sparse_mla_prefill_runtime\ndef sparse_mla_prefill(*args): return None\n",
        "import candidate_worker\ndef sparse_mla_prefill(*args): return None\n",
        "import public_runner\ndef sparse_mla_prefill(*args): return None\n",
        "import __main__\ndef sparse_mla_prefill(*args): return None\n",
        (
            "import sys\n"
            "def sparse_mla_prefill(*args):\n"
            "    return sys.modules['sparse_mla_prefill_runtime']._reference(*args)\n"
        ),
    )
    for index, source in enumerate(sources):
        root = make_submission(tmp_path / str(index), selected_source=source)
        result = admission.validate_submission(root)
        assert result["ok"] is False
        assert any(
            "forbidden import" in error or "trusted harness access" in error
            for error in result["errors"]
        )


def test_selected_frame_introspection_import_is_rejected(tmp_path: Path) -> None:
    root = make_submission(
        tmp_path,
        selected_source=(
            "import inspect\n"
            "def sparse_mla_prefill(*args):\n"
            "    return inspect.currentframe()\n"
        ),
    )
    result = admission.validate_submission(root)
    assert result["ok"] is False
    assert any("forbidden import" in error for error in result["errors"])


def test_workspace_mode_allows_only_the_fixed_non_submission_entries(tmp_path: Path) -> None:
    root = make_submission(tmp_path)
    for directory in ("build", "tools"):
        (root / directory).mkdir()
    (root / "AGENT_TASK.md").write_text("trusted instruction\n", encoding="utf-8")
    (root / "README_AGENT.md").write_text("trusted readme\n", encoding="utf-8")
    assert admission.validate_submission(root)["ok"] is False
    result = admission.validate_submission(root, 1, workspace_mode=True)
    assert result["ok"] is True
    assert result["candidate_version"] == 1


HEADER = "import torch\nimport triton\nimport triton.language as tl\n"

V3_REJECTED = {
    "timer_patch": "import time\ntime.perf_counter = lambda: 0.0",
    "event_patch": "torch.cuda.Event.elapsed_time = lambda self, other: 0.0",
    "imported_class_patch": "from torch import Tensor\nTensor.fill_ = len",
    "alias_patch": "t = torch\nt.isfinite = len",
    "alias_attribute_patch": "ev = torch.cuda.Event\nev.elapsed_time = len",
    "container_alias_patch": "d = {'m': torch}\nd['m'].isfinite = len",
    "computed_target": "[torch][0].isfinite = len",
    "setattr": "setattr(torch, 'isfinite', len)",
    "vars": "vars(torch)['isfinite'] = len",
    "dict_dunder": "x = torch.__dict__",
    "getattr_gemm": "getattr(torch, 'm' + 'm')",
    "backends_flag": "torch.backends.cuda.matmul.allow_tf32 = True",
    "global_setter": "torch.set_float32_matmul_precision('medium')",
    "rng": "torch.manual_seed(0)",
    "cuda_graph": "g = torch.cuda.CUDAGraph()",
    "save": "torch.save({}, 'x')",
    "triton_compile": "triton.compile(None)",
    "load_binary": "triton.runtime.driver.active.utils.load_binary('k', b'', 0, 0)",
    "os_import": "import os",
    "torch_ops_import": "from torch import ops",
    "private_namespace": "torch._scaled_mm",
    "star_import": "from torch import *",
    "jiterator": "torch.cuda.jiterator",
    "matmul_operator": "c = a @ b",
}

V3_ALLOWED = {
    "set_allocator": "triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device='cuda'))",
    "device_properties": "sm = torch.cuda.get_device_properties(0).multi_processor_count\nn = triton.runtime.driver.active.utils.get_device_properties(0)",
    "local_cache": "_CACHE = {}\n_CACHE[(1, 2)] = 3\nclass Plan:\n    def __init__(self):\n        self.tiles = 1",
    "do_bench": "import triton.testing\nt = triton.testing.do_bench(lambda: None)",
    "libdevice": "from triton.language.extra import libdevice",
    "main_guard": "if __name__ == '__main__':\n    pass",
    "scratch_tensor": "buf = torch.empty(4, device='cuda')\nbuf[0] = 1",
    "config_object": "cfg = triton.Config({'BM': 64}, num_warps=4)\nx = cfg.kwargs",
    "math_functools": "import math\nimport functools\n@functools.lru_cache(None)\ndef f(x):\n    return math.ceil(x / 128)",
}


def test_v2_rejects_patching_introspection_and_global_state(tmp_path: Path) -> None:
    for index, (name, body) in enumerate(V3_REJECTED.items()):
        root = make_submission(tmp_path / str(index), selected_source=HEADER + body + "\n")
        result = admission.validate_submission(root)
        assert result["ok"] is False, name


def test_v2_allows_the_documented_implementation_surface(tmp_path: Path) -> None:
    for index, (name, body) in enumerate(V3_ALLOWED.items()):
        source = VALID_SOURCE + body + "\n"
        root = make_submission(tmp_path / str(index), selected_source=source)
        result = admission.validate_submission(root)
        assert result["ok"] is True, (name, result["errors"])


def test_v2_oracle_is_admitted_and_the_v1_oracle_only_for_warnings(tmp_path: Path) -> None:
    task_root = Path(__file__).resolve().parents[1]
    # Admit exactly what `scripts/make_oracle_archive.sh --admissible` ships to the
    # normal Judge: the ADMISSIBLE_VERSION candidate, notes.md and a FINAL_VERSION
    # selecting it.  oracle/paper_reference/ is a host-only harness that never
    # enters the archive.
    import shutil

    admissible = int((task_root / "oracle/ADMISSIBLE_VERSION").read_text(encoding="utf-8"))
    selected = f"candidates/kernel_v{admissible:03d}/submission.py"
    shipped = tmp_path / "oracle"
    (shipped / selected).parent.mkdir(parents=True)
    shutil.copy2(task_root / "oracle" / selected, shipped / selected)
    shutil.copy2(task_root / "oracle/notes.md", shipped / "notes.md")
    (shipped / "FINAL_VERSION").write_text(f"{admissible}\n", encoding="utf-8")
    result = admission.validate_submission(shipped)
    assert result["ok"] is True, result["errors"]
    assert result["selected_relative_path"] == selected

    # FINAL_VERSION selects the FlashMLA paper candidate, which only the local
    # forced-oracle Judge (oracle/paper_reference/forced_judge/) admits.
    final = int((task_root / "oracle/FINAL_VERSION").read_text(encoding="utf-8"))
    if final != admissible:
        forced = admission._scan_selected_source(
            (task_root / f"oracle/candidates/kernel_v{final:03d}/submission.py").read_text(encoding="utf-8"),
            "forced",
        )
        assert any("forbidden import: 'flash_mla" in error for error in forced), forced
    # v1 is archived (2026-10-07) under tasks/archive/.
    v1 = task_root.parents[1] / "archive/standalone-sparse-mla-prefill-h100-v1-scratch/oracle/candidates/kernel_v001/submission.py"
    if v1.is_file():
        assert admission._scan_selected_source(v1.read_text(encoding="utf-8"), "v1") == [
            "v1:3: forbidden import: 'warnings'"
        ]


TASK_REJECTED = {
    "flash_mla_import": "import flash_mla",
    "sdpa": "torch.nn.functional.scaled_dot_product_attention(q, k, v)",
    "flash_mla_call": "x.flash_mla_sparse_fwd(q, kv, idx)",
    "bmm": "torch.bmm(a, b)",
    "einsum": "torch.einsum('ij,jk->ik', a, b)",
    "linear": "torch.nn.functional.linear(a, b)",
    "fp8_dtype_triton": "x = tl.float8e4nv",
    "fp8_dtype_torch": "x = torch.float8_e4m3fn",
    "fp8_dtype_string": "x = a.to('float8_e5m2')",
    "fp4_dtype": "x = torch.float4_e2m1fn_x2",
    "dot_scaled": "y = tl.dot_scaled(a, s, 'e4m3', b, t, 'e4m3')",
    "kernel_name_prefix": "@triton.jit\ndef _Zkernel(x):\n    pass",
    "kernel_name_gemm": "@triton.jit\ndef _sparse_gemm_kernel(x):\n    pass",
    "kernel_name_cutlass": "def cutlass_like(x):\n    pass",
    "kernel_name_nvjet": "def NvJet_style(x):\n    pass",
    "renaming": "import functools\nf = functools.wraps(len)(lambda: 0)",
    "warnings_import": "import warnings",
    "time_import": "import time",
    "threading_import": "import threading",
    "os_environ": "import os\nx = os.environ",
    "memory_snapshot": "torch.cuda.memory_snapshot()",
    "memory_namespace": "from torch.cuda import memory",
}

TASK_ALLOWED = {
    "fp16_probabilities": "p = torch.float16\nq = tl.float16",
    "bf16_dot": "x = tl.bfloat16",
    "gluon": "from triton.experimental import gluon\nfrom triton.experimental.gluon import language as gl",
    "runtime_errors": "import triton.runtime.errors\nE = triton.runtime.errors.OutOfResources",
    "kernel_names": "@triton.jit\ndef _sparse_mla_fwd(x):\n    pass\n@triton.jit\ndef _combine(x):\n    pass",
    "docstring_mentions": '"""No FP8 or float8 path; the GEMM is tl.dot."""',
}


def test_task_specific_rejections(tmp_path: Path) -> None:
    for index, (name, body) in enumerate(TASK_REJECTED.items()):
        root = make_submission(tmp_path / str(index), selected_source=HEADER + body.replace("\\n", "\n") + "\n")
        result = admission.validate_submission(root)
        assert result["ok"] is False, name


def test_task_specific_allowances(tmp_path: Path) -> None:
    for index, (name, body) in enumerate(TASK_ALLOWED.items()):
        source = VALID_SOURCE + body.replace("\\n", "\n") + "\n"
        root = make_submission(tmp_path / str(index), selected_source=source)
        result = admission.validate_submission(root)
        assert result["ok"] is True, (name, result["errors"])


# The 38 agent kernels with a valid v1 Judge report live outside the repository
# (2026-10-06 probe evidence, cands/*.py); point SMLA_HISTORICAL_CANDIDATES at
# that directory to check that the v2 allowlist admits every one of them.
HISTORICAL_CANDIDATES = os.environ.get("SMLA_HISTORICAL_CANDIDATES", "")


@pytest.mark.skipif(
    not HISTORICAL_CANDIDATES or not Path(HISTORICAL_CANDIDATES).is_dir(),
    reason="set SMLA_HISTORICAL_CANDIDATES to the historical kernel directory",
)
def test_v1_judge_valid_historical_kernels_remain_admitted() -> None:
    rejected = {}
    for path in sorted(Path(HISTORICAL_CANDIDATES).glob("*.py")):
        if path.stem.startswith("planted_"):
            continue
        errors = admission._scan_selected_source(path.read_text(encoding="utf-8"), path.name)
        # The v1 oracle copy differs from the v2 oracle only by its warnings import.
        if errors and errors != [f"{path.name}:3: forbidden import: 'warnings'"]:
            rejected[path.name] = errors
    assert rejected == {}


def test_work_and_judge_admission_copies_are_byte_identical() -> None:
    assert (TASK_ROOT / "images/work/benchmark/source_admission.py").read_bytes() == (
        TASK_ROOT / "images/judge/source_admission.py"
    ).read_bytes()


def test_sanitizer_excluded_names_match_the_verifier_configuration() -> None:
    verifier = load_module("admission_verifier", "images/judge/verify_sparse_mla.py")
    substrings = tuple(pattern.split("=", 1)[1] for pattern in verifier.SANITIZER_KERNEL_EXCLUDES if pattern.startswith("kns="))
    assert substrings == admission.FORBIDDEN_KERNEL_NAME_SUBSTRINGS
    assert "regex=^_Z" in verifier.SANITIZER_KERNEL_EXCLUDES and admission.FORBIDDEN_KERNEL_NAME_PREFIXES == ("_z",)
