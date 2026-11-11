from __future__ import annotations

from pathlib import Path

from conftest import load_module


admission = load_module("standalone_source_admission", "images/judge/source_admission.py")


VALID_SOURCE = """\
import torch
import triton
import triton.language as tl

def grouped_fp8_gemm_nt_masked(a, b, a_scale, b_scale, m_sizes, expected_m_hint, out):
    return None

def grouped_fp8_gemm_nt_psum(a, b, a_scale, b_scale, group_ends, expected_m_hint, out):
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
        selected_source="import torch\ndef grouped_fp8_gemm_nt_masked(a, b, *args): return torch.matmul(a, b)\n",
    )
    result = admission.validate_submission(root)
    assert result["ok"] is False
    assert any("existing GEMM API" in error for error in result["errors"])


def test_locally_authored_helper_name_is_not_a_style_failure(tmp_path: Path) -> None:
    root = make_submission(
        tmp_path,
        selected_source=(
            "def matmul(*args): return None\n"
            "def grouped_fp8_gemm_nt_masked(*args): return matmul(*args)\n"
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
            "def grouped_fp8_gemm_nt_masked(*args):\n"
            "    stream = torch.cuda.Stream()\n"
            "    return None\n"
        ),
    )
    result = admission.validate_submission(root)
    assert result["ok"] is False
    assert any("private CUDA stream API" in error for error in result["errors"])


def test_selected_trusted_harness_access_is_rejected(tmp_path: Path) -> None:
    sources = (
        "import grouped_fp8_runtime\ndef grouped_fp8_gemm_nt_masked(*args): return None\n",
        "import candidate_worker\ndef grouped_fp8_gemm_nt_masked(*args): return None\n",
        "import public_runner\ndef grouped_fp8_gemm_nt_masked(*args): return None\n",
        "import __main__\ndef grouped_fp8_gemm_nt_masked(*args): return None\n",
        (
            "import sys\n"
            "def grouped_fp8_gemm_nt_masked(*args):\n"
            "    return sys.modules['grouped_fp8_runtime']._reference(*args)\n"
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
            "def grouped_fp8_gemm_nt_masked(*args):\n"
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


def test_v3_rejects_patching_introspection_and_global_state(tmp_path: Path) -> None:
    for index, (name, body) in enumerate(V3_REJECTED.items()):
        root = make_submission(tmp_path / str(index), selected_source=HEADER + body + "\n")
        result = admission.validate_submission(root)
        assert result["ok"] is False, name


def test_v3_allows_the_documented_implementation_surface(tmp_path: Path) -> None:
    for index, (name, body) in enumerate(V3_ALLOWED.items()):
        source = VALID_SOURCE + body + "\n"
        root = make_submission(tmp_path / str(index), selected_source=source)
        result = admission.validate_submission(root)
        assert result["ok"] is True, (name, result["errors"])


def test_v3_oracle_is_admitted(tmp_path) -> None:
    # Stage exactly what `scripts/make_oracle_archive.sh --admissible` packs for the
    # normal Judge (candidates/, notes.md, FINAL_VERSION = oracle/ADMISSIBLE_VERSION);
    # oracle/paper_reference/ is a host-only harness and never part of the archive.
    import shutil

    oracle_root = Path(__file__).resolve().parents[1] / "oracle"
    admissible = int((oracle_root / "ADMISSIBLE_VERSION").read_text(encoding="utf-8"))
    staged = tmp_path / "oracle"
    staged.mkdir()
    shutil.copytree(oracle_root / "candidates", staged / "candidates")
    shutil.copy2(oracle_root / "notes.md", staged / "notes.md")
    shutil.copy2(oracle_root / "ADMISSIBLE_VERSION", staged / "FINAL_VERSION")
    result = admission.validate_submission(staged)
    assert result["ok"] is True, result["errors"]
    assert result["selected_relative_path"] == f"candidates/kernel_v{admissible:03d}/submission.py"


def test_v3_forced_paper_oracle_is_selected_but_not_admitted() -> None:
    # oracle/FINAL_VERSION selects the upstream DeepGEMM candidate; only the local
    # forced-oracle Judge (oracle/paper_reference/forced_judge/) admits it.
    oracle_root = Path(__file__).resolve().parents[1] / "oracle"
    final = int((oracle_root / "FINAL_VERSION").read_text(encoding="utf-8"))
    admissible = int((oracle_root / "ADMISSIBLE_VERSION").read_text(encoding="utf-8"))
    for version in (final, admissible):
        assert (oracle_root / f"candidates/kernel_v{version:03d}/submission.py").is_file()
    if final != admissible:
        result = admission.validate_submission(oracle_root, final)
        assert result["ok"] is False
        assert any("deep_gemm" in error for error in result["errors"]), result["errors"]
