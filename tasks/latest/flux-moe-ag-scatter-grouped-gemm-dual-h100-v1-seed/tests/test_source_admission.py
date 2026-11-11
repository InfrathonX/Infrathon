from __future__ import annotations

from pathlib import Path

import pytest

from conftest import load_module


admission = load_module("moeag_source_admission", "images/judge/source_admission.py")


VALID_SOURCE = """\
import torch
import triton
import triton.language as tl

@triton.jit
def _copy_kernel(source, destination, n: tl.constexpr):
    offsets = tl.arange(0, n)
    tl.store(destination + offsets, tl.load(source + offsets))

def _launch_copy(source, destination):
    _copy_kernel[(1,)](source, destination, n=128)

def moe_ag_scatter_gemm_bf16(
    x_local, x_peers, w, sorted_token_ids, sorted_expert_ids, expert_offsets, token_offsets, out, workspace, peer_workspaces, peer_ptrs, workspace_ptrs, rank, world_size
):
    _launch_copy(x_local, out)
    return None
"""


def make_submission(
    root: Path,
    *,
    selected: int = 1,
    selected_source: str = VALID_SOURCE,
) -> Path:
    initial = root / "candidates/kernel_v001"
    initial.mkdir(parents=True)
    initial_source = selected_source if selected == 1 else "# retained old experiment\n"
    (initial / "submission.py").write_text(initial_source, encoding="utf-8")
    if selected != 1:
        selected_dir = root / f"candidates/kernel_v{selected:03d}"
        selected_dir.mkdir()
        (selected_dir / "submission.py").write_text(selected_source, encoding="utf-8")
    (root / "FINAL_VERSION").write_text(f"{selected}\n", encoding="utf-8")
    (root / "notes.md").write_text("candidate notes\n", encoding="utf-8")
    return root


def decision_for(tmp_path: Path, source: str) -> dict[str, object]:
    return admission.validate_submission(make_submission(tmp_path, selected_source=source))


def assert_rejected(tmp_path: Path, source: str) -> None:
    result = decision_for(tmp_path, source)
    assert result["ok"] is False, result
    assert result["errors"], result


def source_with_body(body: str, imports: str = "") -> str:
    indented = "\n".join(f"    {line}" if line else "" for line in body.splitlines())
    return f"""\
{imports}import triton
import triton.language as tl

@triton.jit
def _kernel(x_local, out):
    value = tl.load(x_local)
    tl.store(out, value)

def moe_ag_scatter_gemm_bf16(
    x_local, x_peers, w, sorted_token_ids, sorted_expert_ids, expert_offsets, token_offsets, out, workspace, peer_workspaces, peer_ptrs, workspace_ptrs, rank, world_size
):
{indented}
    _kernel[(1,)](x_local, out)
    return None
"""


def source_with_helper(
    helper_name: str,
    helper_parameters: str,
    helper_body: str,
    entry_body: str,
) -> str:
    helper_lines = "\n".join(f"    {line}" for line in helper_body.splitlines())
    entry_lines = "\n".join(f"    {line}" for line in entry_body.splitlines())
    return f"""\
import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(x_local, out):
    tl.store(out, tl.load(x_local))

def {helper_name}({helper_parameters}):
{helper_lines}

def moe_ag_scatter_gemm_bf16(
    x_local, x_peers, w, sorted_token_ids, sorted_expert_ids, expert_offsets, token_offsets, out, workspace, peer_workspaces, peer_ptrs, workspace_ptrs, rank, world_size
):
{entry_lines}
    return None
"""


def source_with_host_helpers(helpers: str, entry_body: str) -> str:
    helper_source = helpers.strip()
    entry_lines = "\n".join(f"    {line}" for line in entry_body.splitlines())
    return f"""\
import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(x_local, out):
    tl.store(out, tl.load(x_local))

{helper_source}

def moe_ag_scatter_gemm_bf16(
    x_local, x_peers, w, sorted_token_ids, sorted_expert_ids, expert_offsets, token_offsets, out, workspace, peer_workspaces, peer_ptrs, workspace_ptrs, rank, world_size
):
{entry_lines}
    return None
"""


@pytest.mark.parametrize(
    "language_import",
    (
        "import triton.language as tl\n",
        "from triton import language as tl\n",
    ),
)
def test_direct_torch_triton_and_triton_language_imports_are_allowed(
    tmp_path: Path, language_import: str
) -> None:
    source = VALID_SOURCE.replace(
        "import triton.language as tl\n", language_import
    )
    assert decision_for(tmp_path, source)["ok"] is True


def test_safe_metadata_scratch_allocation_and_multiple_triton_launches_are_allowed(
    tmp_path: Path,
) -> None:
    source = """\
import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(source, destination, n: tl.constexpr):
    offsets = tl.arange(0, n)
    tl.store(destination + offsets, tl.load(source + offsets))

def moe_ag_scatter_gemm_bf16(
    x_local, x_peers, w, sorted_token_ids, sorted_expert_ids, expert_offsets, token_offsets, out, workspace, peer_workspaces, peer_ptrs, workspace_ptrs, rank, world_size
):
    width = x_local.shape[-1]
    rows = x_local.shape[0] * x_local.shape[1]
    scratch = torch.empty((rows, width), device=x_local.device, dtype=x_local.dtype)
    second = torch.empty_like(scratch)
    _kernel[(rows,)](x_local, scratch, n=width)
    _kernel[(rows,)](scratch, second, n=width)
    _kernel[(rows,)](second, out, n=width)
    return None
"""
    assert decision_for(tmp_path, source)["ok"] is True


def test_normal_submission_scans_only_selected_version(tmp_path: Path) -> None:
    root = make_submission(tmp_path, selected=2)
    (root / "candidates/kernel_v001/submission.py").write_bytes(b"")
    result = admission.validate_submission(root)
    assert result["ok"] is True
    assert result["final_version"] == 2
    assert result["num_candidate_versions"] == 2
    assert result["candidate_version"] == 2
    assert result["selected_relative_path"] == "candidates/kernel_v002/submission.py"
    assert len(str(result["selected_sha256"])) == 64


@pytest.mark.parametrize(
    "parameters",
    (
        "x_local, peer_ptrs, w, sorted_token_ids, sorted_token_ids, out",
        "x_local, x_peers, w, sorted_token_ids, sorted_expert_ids, expert_offsets, token_offsets, out, workspace, peer_workspaces, peer_ptrs, workspace_ptrs, rank, world_size, extra",
        "x_local, kv, indices, sm_scale, attn_sink, topk_length, out, max_logits, lse",
        "*args",
        "x_local, x_peers, w, sorted_token_ids, sorted_expert_ids, expert_offsets, token_offsets, out, workspace, peer_workspaces, peer_ptrs, workspace_ptrs, rank, world_size=None",
        "x_local, peer_ptrs, w, sorted_token_ids, sorted_token_ids, *, rank, out",
    ),
)
def test_selected_entrypoint_requires_exact_ten_parameter_abi(
    tmp_path: Path, parameters: str
) -> None:
    source = VALID_SOURCE.replace(
        "x_local, x_peers, w, sorted_token_ids, sorted_expert_ids, expert_offsets, token_offsets, out, workspace, peer_workspaces, peer_ptrs, workspace_ptrs, rank, world_size\n",
        parameters + "\n",
    )
    assert_rejected(tmp_path, source)


@pytest.mark.parametrize(
    "source",
    (
        "def other(x_local, x_peers, w, sorted_token_ids, sorted_expert_ids, expert_offsets, token_offsets, out, workspace, peer_workspaces, peer_ptrs, workspace_ptrs, rank, world_size):\n    return None\n",
        "def moe_ag_scatter_gemm_bf16(x_local, x_peers, w, sorted_token_ids, sorted_expert_ids, expert_offsets, token_offsets, out, workspace, peer_workspaces, peer_ptrs, workspace_ptrs, rank, world_size):\n    return None\n",
        (
            "import triton\nimport triton.language as tl\n"
            "@triton.jit\ndef _unused(x_local, out):\n    tl.store(out, tl.load(x_local))\n"
            "def moe_ag_scatter_gemm_bf16(x_local, x_peers, w, sorted_token_ids, sorted_expert_ids, expert_offsets, token_offsets, out, workspace, peer_workspaces, peer_ptrs, workspace_ptrs, rank, world_size):\n"
            "    return None\n"
        ),
    ),
)
def test_entrypoint_reachable_call_graph_must_launch_a_triton_kernel(
    tmp_path: Path, source: str
) -> None:
    assert_rejected(tmp_path, source)


@pytest.mark.parametrize(
    "statement",
    (
        "temporary = torch.softmax(x_local, dim=-1)",
        "temporary = x_local.softmax(dim=-1)",
        "temporary = torch.matmul(x_local, peer_ptrs)",
        "temporary = x_local.matmul(peer_ptrs)",
        "temporary = torch.nn.functional.scaled_dot_product_attention(x_local, peer_ptrs, w)",
        "temporary = x_local + peer_ptrs",
        "out.copy_(x_local)",
        "out.zero_()",
        "x_local.add_(1)",
        "out[0] = x_local[0]",
    ),
)
def test_torch_data_paths_host_arithmetic_and_host_mutation_are_rejected(
    tmp_path: Path, statement: str
) -> None:
    assert_rejected(tmp_path, source_with_body(statement, "import torch\n"))


@pytest.mark.parametrize(
    "module",
    ("flash_attn", "flashinfer", "xformers", "vllm", "sgl_kernel"),
)
def test_external_attention_package_imports_are_rejected(
    tmp_path: Path, module: str
) -> None:
    assert_rejected(tmp_path, source_with_body("pass", f"import {module}\n"))


@pytest.mark.parametrize(
    ("statement", "imports"),
    (
        ("open('payload.py')", ""),
        ("eval('1 + 1')", ""),
        ("exec('value = 1')", ""),
        ("compile('1', '<candidate>', 'eval')", ""),
        ("__import__('os')", ""),
        ("importlib.import_module('os')", "import importlib\n"),
        ("torch.ops.vendor.kernel(x_local)", "import torch\n"),
        ("torch.utils.cpp_extension.load_inline(name='x', cpp_sources='')", "import torch.utils.cpp_extension\n"),
        ("subprocess.run(['true'])", "import subprocess\n"),
        ("os.system('true')", "import os\n"),
        ("socket.create_connection(('127.0.0.1', 1))", "import socket\n"),
        ("requests.get('https://example.invalid')", "import requests\n"),
        ("pathlib.Path('/tmp/payload').read_text()", "import pathlib\n"),
    ),
)
def test_dynamic_extension_process_network_and_filesystem_escape_is_rejected(
    tmp_path: Path, statement: str, imports: str
) -> None:
    assert_rejected(tmp_path, source_with_body(statement, imports))


@pytest.mark.parametrize(
    "statement",
    (
        "__builtins__['open']('payload.py')",
        "__builtins__['__import__']('os')",
        "torch.__dict__['ops'].vendor.kernel(x_local)",
        "torch.nn.functional.__dict__['scaled_dot_product_attention'](x_local, peer_ptrs, w)",
        "torch.cuda.__dict__['Stream']()",
    ),
)
def test_subscript_call_chain_bypasses_are_rejected(
    tmp_path: Path, statement: str
) -> None:
    assert_rejected(tmp_path, source_with_body(statement, "import torch\n"))


@pytest.mark.parametrize(
    "statement",
    (
        "torch.__dict__['matmul'] = lambda left, right: left",
        "torch.__dict__.update({'matmul': lambda left, right: left})",
        "torch.matmul = lambda left, right: left",
        "del torch.matmul",
    ),
)
def test_imported_module_mutation_including_dictionary_routes_is_rejected(
    tmp_path: Path, statement: str
) -> None:
    assert_rejected(tmp_path, source_with_body(statement, "import torch\n"))


@pytest.mark.parametrize(
    "statement",
    (
        "temporary = x_local.__class__",
        "temporary = moe_ag_scatter_gemm_bf16.__globals__",
        "temporary = (1).__class__.__mro__",
        (
            "try:\n"
            "    raise RuntimeError('probe')\n"
            "except RuntimeError as exc:\n"
            "    temporary = exc.__traceback__.tb_frame.f_back.f_locals"
        ),
    ),
)
def test_dunder_traceback_frame_and_local_traversal_are_rejected(
    tmp_path: Path, statement: str
) -> None:
    assert_rejected(tmp_path, source_with_body(statement))


@pytest.mark.parametrize("dtype",("torch.float16","torch.float8_e4m3fn","torch.int8","torch.bfloat16","torch.float32"))
def test_allowed_scratch_dtypes_are_admitted(tmp_path: Path,dtype: str) -> None:
    source=source_with_body(f"temporary = torch.empty_like(x_local, dtype={dtype})")
    assert decision_for(tmp_path,source)["ok"] is True

def test_host_tensor_mean_is_rejected(tmp_path: Path) -> None:
    assert_rejected(tmp_path,source_with_body("temporary = peer_ptrs.mean(dim=1)"))


@pytest.mark.parametrize(
    "statement",
    (
        "torch.cuda.Stream()",
        "torch.cuda.set_stream(None)",
        "torch.cuda.ExternalStream(0)",
    ),
)
def test_private_stream_apis_are_rejected(tmp_path: Path, statement: str) -> None:
    assert_rejected(tmp_path, source_with_body(statement, "import torch\n"))


def test_layout_rejects_extra_python_sources_and_symlinks(tmp_path: Path) -> None:
    root = make_submission(tmp_path)
    (root / "unexpected.py").write_text("pass\n", encoding="utf-8")
    candidate = root / "candidates/kernel_v001"
    (candidate / "helper.py").write_text("pass\n", encoding="utf-8")
    source = candidate / "submission.py"
    source.unlink()
    source.symlink_to(root / "notes.md")
    result = admission.validate_submission(root)
    assert result["ok"] is False
    assert any("unexpected root entry" in error for error in result["errors"])
    assert any("unexpected candidate entry" in error for error in result["errors"])
    assert any("symlink is forbidden" in error for error in result["errors"])


def test_selected_empty_initial_candidate_is_invalid(tmp_path: Path) -> None:
    result = admission.validate_submission(make_submission(tmp_path, selected_source=""))
    assert result["ok"] is False
    assert any("selected source must not be empty" in error for error in result["errors"])


@pytest.mark.parametrize(
    "contents",
    (b"", b"0\n", b"01\n", b"1000\n", b"1 \n", b"1\n\n", b"true\n"),
)
def test_final_version_requires_one_canonical_integer(
    tmp_path: Path, contents: bytes
) -> None:
    root = make_submission(tmp_path)
    (root / "FINAL_VERSION").write_bytes(contents)
    result = admission.validate_submission(root)
    assert result["ok"] is False
    assert any("FINAL_VERSION" in error for error in result["errors"])


@pytest.mark.parametrize("override", (0, 1000, True, "1"))
def test_candidate_version_override_has_the_same_bounds(
    tmp_path: Path, override: object
) -> None:
    result = admission.validate_submission(make_submission(tmp_path), override)
    assert result["ok"] is False
    assert any("candidate_version" in error for error in result["errors"])


def test_workspace_mode_allows_only_fixed_non_submission_entries(tmp_path: Path) -> None:
    root = make_submission(tmp_path)
    for directory in ("build", "tools"):
        (root / directory).mkdir()
    (root / "AGENT_TASK.md").write_text("trusted instruction\n", encoding="utf-8")
    (root / "README_AGENT.md").write_text("trusted readme\n", encoding="utf-8")
    assert admission.validate_submission(root)["ok"] is False
    result = admission.validate_submission(root, 1, workspace_mode=True)
    assert result["ok"] is True
    assert result["candidate_version"] == 1


@pytest.mark.parametrize(
    ("helper_name", "entry_body"),
    (
        (
            "launch",
            "launch = torch.nn.functional.scaled_dot_product_attention\n"
            "launch(x_local, peer_ptrs, w)",
        ),
        (
            "copy",
            "copy = torch.Tensor.copy_\ncopy(out, x_local)",
        ),
        (
            "helper",
            "helper: object = torch.softmax\nhelper(x_local, dim=-1)",
        ),
        (
            "helper",
            "(helper := torch.softmax)\nhelper(x_local, dim=-1)",
        ),
        (
            "helper",
            "for helper in [torch.softmax]:\n    pass\nhelper(x_local, dim=-1)",
        ),
    ),
)
def test_reachable_helper_identity_cannot_be_shadowed_by_runtime_bindings(
    tmp_path: Path, helper_name: str, entry_body: str
) -> None:
    source = source_with_helper(
        helper_name,
        "x_local, out",
        "_kernel[(1,)](x_local, out)",
        entry_body,
    )
    assert_rejected(tmp_path, source)


def test_module_assignment_cannot_replace_a_certified_helper(tmp_path: Path) -> None:
    source = source_with_helper(
        "helper",
        "x_local, out",
        "_kernel[(1,)](x_local, out)",
        "helper(x_local, out)",
    ).replace(
        "\ndef moe_ag_scatter_gemm_bf16(",
        "\nhelper = 0\n\ndef moe_ag_scatter_gemm_bf16(",
    )
    assert_rejected(tmp_path, source)


def test_nested_scope_parameter_cannot_shadow_a_certified_helper(
    tmp_path: Path,
) -> None:
    source = source_with_helper(
        "helper",
        "x_local, out",
        "_kernel[(1,)](x_local, out)",
        "callback = lambda helper: helper\nhelper(x_local, out)",
    )
    assert_rejected(tmp_path, source)


@pytest.mark.parametrize(
    "statement",
    (
        "import os",
        "import pathlib",
        "import socket",
        "import subprocess",
        "import importlib",
        "import inspect",
        "import traceback",
        "from os import system",
    ),
)
def test_imports_inside_candidate_functions_are_always_rejected(
    tmp_path: Path, statement: str
) -> None:
    assert_rejected(tmp_path, source_with_body(statement))


@pytest.mark.parametrize("condition", ("False", "0", "None", "()", "[]", "{}", "''", "not True"))
def test_constant_dead_triton_launch_does_not_certify_entrypoint(
    tmp_path: Path, condition: str
) -> None:
    source = source_with_helper(
        "_unused",
        "x_local, out",
        "_kernel[(1,)](x_local, out)",
        f"if {condition}:\n    _kernel[(1,)](x_local, out)",
    )
    assert_rejected(tmp_path, source)


def test_launch_after_unconditional_return_does_not_certify_entrypoint(
    tmp_path: Path,
) -> None:
    source = source_with_body("return None\n_kernel[(1,)](x_local, out)")
    assert_rejected(tmp_path, source)


def test_helper_call_after_unconditional_return_is_not_reachable(
    tmp_path: Path,
) -> None:
    source = VALID_SOURCE.replace(
        "    _launch_copy(x_local, out)\n    return None\n",
        "    return None\n    _launch_copy(x_local, out)\n",
    )
    assert_rejected(tmp_path, source)


def test_live_conditional_specializations_can_each_launch_triton(tmp_path: Path) -> None:
    source = source_with_body(
        "if x_local.shape[0] > 1:\n"
        "    _kernel[(1,)](x_local, out)\n"
        "else:\n"
        "    _kernel[(1,)](x_local, out)"
    )
    assert decision_for(tmp_path, source)["ok"] is True


@pytest.mark.parametrize(
    ("parameters", "helper_body", "entry_body"),
    (
        ("value, /", "temporary = value + value", "helper(x_local)\n_kernel[(1,)](x_local, out)"),
        ("*values", "temporary = values[0] + values[0]", "helper(x_local)\n_kernel[(1,)](x_local, out)"),
        ("*, value", "temporary = value + value", "helper(value=x_local)\n_kernel[(1,)](x_local, out)"),
        ("**values", "temporary = values['value'] + values['value']", "helper(value=x_local)\n_kernel[(1,)](x_local, out)"),
    ),
)
def test_tensor_taint_reaches_every_python_parameter_kind(
    tmp_path: Path, parameters: str, helper_body: str, entry_body: str
) -> None:
    source = source_with_helper("helper", parameters, helper_body, entry_body)
    assert_rejected(tmp_path, source)


def test_scalar_metadata_helper_is_allowed_in_a_triton_grid(tmp_path: Path) -> None:
    source = source_with_helper(
        "grid_for",
        "elements",
        "return triton.cdiv(elements, 128)",
        "grid = grid_for(x_local.numel())\n_kernel[(grid,)](x_local, out)",
    )
    assert decision_for(tmp_path, source)["ok"] is True


def test_tensor_taint_propagates_through_multiple_helper_calls(tmp_path: Path) -> None:
    source = source_with_helper(
        "outer",
        "value",
        "inner(value)",
        "outer(x_local)\n_kernel[(1,)](x_local, out)",
    ).replace(
        "\ndef outer(value):",
        "\ndef inner(value, /):\n"
        "    temporary = value + value\n\n"
        "def outer(value):",
    )
    assert_rejected(tmp_path, source)


@pytest.mark.parametrize(
    "pattern",
    (
        "helper",
        "[*helper]",
        "{'shape': _, **helper}",
    ),
)
def test_structural_pattern_capture_cannot_shadow_a_certified_helper(
    tmp_path: Path, pattern: str
) -> None:
    subject = "{'shape': x_local.shape}" if "shape" in pattern else "x_local.shape"
    source = source_with_helper(
        "helper",
        "x_local, out",
        "_kernel[(1,)](x_local, out)",
        f"match {subject}:\n    case {pattern}:\n        pass\nhelper(x_local, out)",
    )
    assert_rejected(tmp_path, source)


@pytest.mark.parametrize(
    "entry_body",
    (
        "try:\n"
        "    return None\n"
        "except RuntimeError:\n"
        "    _kernel[(1,)](x_local, out)",
        "while True:\n    pass\n_kernel[(1,)](x_local, out)",
        "if 1 == 2:\n    _kernel[(1,)](x_local, out)",
        "unused = [_kernel[(1,)](x_local, out) for _ in ()]",
    ),
)
def test_statically_unreachable_launch_forms_do_not_certify_entrypoint(
    tmp_path: Path, entry_body: str
) -> None:
    source = source_with_helper(
        "_unused",
        "x_local, out",
        "_kernel[(1,)](x_local, out)",
        entry_body,
    )
    assert_rejected(tmp_path, source)


@pytest.mark.parametrize(
    "condition",
    (
        "('x' and 'y') == 'y'",
        "('' or 'y') == 'y'",
    ),
)
def test_live_literal_bool_comparison_preserves_operand_value(
    tmp_path: Path, condition: str
) -> None:
    # The conditional launch must be the only launch in the module, so the
    # assertion below cannot be satisfied by some other live launch.
    source = source_with_host_helpers(
        "", f"if {condition}:\n    _kernel[(1,)](x_local, out)"
    )
    assert decision_for(tmp_path, source)["ok"] is True


def test_named_expression_cannot_hide_tensor_taint_from_a_helper(
    tmp_path: Path,
) -> None:
    source = source_with_helper(
        "host_math",
        "value",
        "temporary = value + value",
        "host_math((value := x_local))\n_kernel[(1,)](x_local, out)",
    )
    assert_rejected(tmp_path, source)


def test_branch_join_keeps_possible_tensor_taint(tmp_path: Path) -> None:
    source = source_with_helper(
        "host_math",
        "value",
        "temporary = value + value",
        "value = x_local\n"
        "if rank > 0:\n"
        "    value = 1\n"
        "host_math(value)\n"
        "_kernel[(1,)](x_local, out)",
    )
    assert_rejected(tmp_path, source)


def test_oracle_source_with_kernel_loop_remains_allowed(tmp_path: Path) -> None:
    oracle = (
        Path(__file__).resolve().parents[1]
        / "oracle/candidates/kernel_v001/submission.py"
    )
    assert decision_for(tmp_path, oracle.read_text(encoding="utf-8"))["ok"] is True


@pytest.mark.parametrize(
    ("helpers", "entry_body"),
    (
        (
            "def produce():\n"
            "    return torch.empty((1,), device='cuda')",
            "value = produce()\ntemporary = value + value\n_kernel[(1,)](x_local, out)",
        ),
        (
            "def produce(count):\n"
            "    return torch.empty((count,), device='cuda')",
            "value = produce(1)\ntemporary = value + value\n_kernel[(1,)](x_local, out)",
        ),
        (
            "def produce(value):\n    return value",
            "value = produce(x_local)\ntemporary = value + value\n_kernel[(1,)](x_local, out)",
        ),
        (
            "def produce(flag):\n"
            "    if flag > 0:\n"
            "        return 1\n"
            "    return torch.empty((1,), device='cuda')",
            "value = produce(rank)\n"
            "temporary = value + value\n"
            "_kernel[(1,)](x_local, out)",
        ),
        (
            "def inner(count):\n"
            "    return torch.empty((count,), device='cuda')\n\n"
            "def outer(count):\n"
            "    return inner(count)",
            "value = outer(1)\ntemporary = value + value\n_kernel[(1,)](x_local, out)",
        ),
    ),
)
def test_context_aware_helper_return_summary_blocks_host_tensor_math(
    tmp_path: Path, helpers: str, entry_body: str
) -> None:
    assert_rejected(tmp_path, source_with_host_helpers(helpers, entry_body))


def test_unknown_recursive_helper_return_fails_closed(tmp_path: Path) -> None:
    source = source_with_host_helpers(
        "def recurse(value):\n    return recurse(value)",
        "value = recurse(1)\n_kernel[(1,)](x_local, out)",
    )
    assert_rejected(tmp_path, source)


def test_dead_tensor_return_does_not_taint_scalar_helper(tmp_path: Path) -> None:
    source = source_with_host_helpers(
        "def grid_for(elements):\n"
        "    if False:\n"
        "        return torch.empty((1,), device='cuda')\n"
        "    return triton.cdiv(elements, 128)",
        "grid = grid_for(x_local.numel())\n_kernel[(grid,)](x_local, out)",
    )
    assert decision_for(tmp_path, source)["ok"] is True


@pytest.mark.parametrize(
    "statement",
    (
        "value = 0\nvalue += x_local",
        "rank += x_local",
        "x_local += rank",
        "out += rank",
    ),
)
def test_host_augmented_assignment_is_rejected(
    tmp_path: Path, statement: str
) -> None:
    assert_rejected(tmp_path, source_with_body(statement))


@pytest.mark.parametrize(
    "statement",
    (
        "head, *values = (0, x_local)\ntemporary = values[0] + values[0]",
        "[*values] = [x_local]\ntemporary = values[0] + values[0]",
    ),
)
def test_host_starred_tensor_unpack_is_rejected(
    tmp_path: Path, statement: str
) -> None:
    assert_rejected(tmp_path, source_with_body(statement))


def test_host_starred_call_binding_is_rejected(tmp_path: Path) -> None:
    source = source_with_helper(
        "grid_for",
        "elements",
        "return triton.cdiv(elements, 128)",
        "grid = grid_for(*(x_local.numel(),))\n_kernel[(grid,)](x_local, out)",
    )
    assert_rejected(tmp_path, source)


@pytest.mark.parametrize(
    "entry_body",
    (
        "if 1 == 2 < rank:\n    _kernel[(1,)](x_local, out)",
        "if rank or True:\n"
        "    return None\n"
        "else:\n"
        "    _kernel[(1,)](x_local, out)",
        "if rank and False:\n    _kernel[(1,)](x_local, out)",
    ),
)
def test_absorbing_truth_values_do_not_certify_dead_launches(
    tmp_path: Path, entry_body: str
) -> None:
    source = source_with_helper(
        "_unused",
        "x_local, out",
        "_kernel[(1,)](x_local, out)",
        entry_body,
    )
    assert_rejected(tmp_path, source)


@pytest.mark.parametrize(
    "entry_body",
    (
        "if not (rank or True):\n    _kernel[(1,)](x_local, out)",
        "if not (rank and False):\n"
        "    return None\n"
        "else:\n"
        "    _kernel[(1,)](x_local, out)",
        "if not not (rank or True):\n"
        "    return None\n"
        "else:\n"
        "    _kernel[(1,)](x_local, out)",
        "if not not (rank and False):\n    _kernel[(1,)](x_local, out)",
    ),
)
def test_nested_not_truth_does_not_certify_dead_launches(
    tmp_path: Path, entry_body: str
) -> None:
    source = source_with_helper(
        "_unused",
        "x_local, out",
        "_kernel[(1,)](x_local, out)",
        entry_body,
    )
    assert_rejected(tmp_path, source)


@pytest.mark.parametrize(
    "condition",
    (
        "not (rank and False)",
        "not not (rank or True)",
    ),
)
def test_nested_not_truth_keeps_live_launches(
    tmp_path: Path, condition: str
) -> None:
    # The conditional launch must be the only launch in the module, so the
    # assertion below cannot be satisfied by some other live launch.
    source = source_with_host_helpers(
        "", f"if {condition}:\n    _kernel[(1,)](x_local, out)"
    )
    assert decision_for(tmp_path, source)["ok"] is True


@pytest.mark.parametrize(
    "comparison",
    (
        "1000 is 1000",
        "1000 is not 1000",
    ),
)
def test_host_identity_comparisons_are_outside_the_scheduling_subset(
    tmp_path: Path, comparison: str
) -> None:
    assert_rejected(tmp_path, source_with_body(f"identity = {comparison}"))


def test_supported_equality_and_live_shape_specialization_remain_allowed(
    tmp_path: Path,
) -> None:
    source = source_with_body(
        "if 1000 == 1000 and x_local.shape[0] > 1:\n"
        "    _kernel[(1,)](x_local, out)\n"
        "else:\n"
        "    _kernel[(1,)](x_local, out)"
    )
    assert decision_for(tmp_path, source)["ok"] is True


def test_identity_comparison_inside_triton_kernel_remains_allowed(
    tmp_path: Path,
) -> None:
    source = """\
import triton
import triton.language as tl

@triton.jit
def _kernel(x_local, out, flag: tl.constexpr):
    if flag is None:
        tl.store(out, tl.load(x_local))

def moe_ag_scatter_gemm_bf16(
    x_local, x_peers, w, sorted_token_ids, sorted_expert_ids, expert_offsets, token_offsets, out, workspace, peer_workspaces, peer_ptrs, workspace_ptrs, rank, world_size
):
    _kernel[(1,)](x_local, out, flag=None)
    return None
"""
    assert decision_for(tmp_path, source)["ok"] is True


# --------------------------------------------------------------------------
# In-process callback routes.
#
# Triton runs several participant-supplied callables itself, at launch time,
# outside the entrypoint call graph: heuristics values, autotune pruners and
# benchmarks, Config pre-hooks, jit launch metadata, and a callable grid.  A
# host function reached only that way used to escape the reachable-call policy
# pass entirely, so it could rebind interpreter state -- for example
# torch.Tensor.__sub__ -- and drive the trusted correctness gate to zero while
# the kernel wrote nothing.  Every route below must fail closed.
# --------------------------------------------------------------------------

CALLBACK_HOOK = """\
def _hook(args):
    setattr(torch.Tensor, "__sub__", torch.mul)
    return 16
"""

KERNEL_BODY = """\
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.zeros([BLOCK], dtype=tl.bfloat16), mask=offs < N)
"""

ENTRY_BODY = """\
def moe_ag_scatter_gemm_bf16(
    x_local, x_peers, w, sorted_token_ids, sorted_expert_ids, expert_offsets, token_offsets, out, workspace, peer_workspaces, peer_ptrs, workspace_ptrs, rank, world_size
):
    _kernel[(1,)](out, out.numel(), BLOCK=4096)
    return None
"""


def callback_source(decorators: str, hook: str = CALLBACK_HOOK) -> str:
    return f"""\
import torch
import triton
import triton.language as tl

{hook}
{decorators}def _kernel(out_ptr, N, BLOCK: tl.constexpr, BLOCK_M: tl.constexpr = None):
{KERNEL_BODY}
{ENTRY_BODY}"""


@pytest.mark.parametrize(
    "decorators",
    (
        '@triton.heuristics({"BLOCK_M": _hook})\n@triton.jit\n',
        "@triton.autotune(configs=[triton.Config({}, num_warps=4)], key=[],"
        ' prune_configs_by={"early_config_prune": _hook})\n@triton.jit\n',
        "@triton.autotune(configs=[triton.Config({}, num_warps=4)], key=[],"
        " do_bench=_hook)\n@triton.jit\n",
        "@triton.autotune(configs=[triton.Config({}, num_warps=4, pre_hook=_hook)],"
        " key=[])\n@triton.jit\n",
        "@triton.jit(launch_metadata=_hook)\n",
    ),
)
def test_triton_decorator_callback_routes_are_rejected(
    tmp_path: Path, decorators: str
) -> None:
    assert_rejected(tmp_path, callback_source(decorators))


def test_callable_grid_route_is_rejected(tmp_path: Path) -> None:
    source = """\
import torch
import triton
import triton.language as tl

def _grid(meta):
    setattr(torch.Tensor, "__sub__", torch.mul)
    return (1,)

@triton.jit
def _kernel(x_local, out):
    tl.store(out, tl.load(x_local))

def moe_ag_scatter_gemm_bf16(
    x_local, x_peers, w, sorted_token_ids, sorted_expert_ids, expert_offsets, token_offsets, out, workspace, peer_workspaces, peer_ptrs, workspace_ptrs, rank, world_size
):
    _kernel[_grid](x_local, out)
    return None
"""
    assert_rejected(tmp_path, source)


def test_callable_grid_body_is_policed_like_any_other_callee(
    tmp_path: Path,
) -> None:
    """A callable grid is legal, but its body is not an unchecked host scope."""
    legal = """\
import torch
import triton
import triton.language as tl

def _grid(meta):
    return (meta["BLOCK"],)

@triton.jit
def _kernel(x_local, out, BLOCK: tl.constexpr):
    tl.store(out, tl.load(x_local))

def moe_ag_scatter_gemm_bf16(
    x_local, x_peers, w, sorted_token_ids, sorted_expert_ids, expert_offsets, token_offsets, out, workspace, peer_workspaces, peer_ptrs, workspace_ptrs, rank, world_size
):
    _kernel[_grid](x_local, out, BLOCK=64)
    return None
"""
    assert decision_for(tmp_path / "legal", legal)["ok"] is True
    assert_rejected(
        tmp_path / "tainted",
        legal.replace(
            '    return (meta["BLOCK"],)',
            '    torch.sum(x_local)\n    return (meta["BLOCK"],)',
        ),
    )


def test_unreachable_host_function_is_rejected(tmp_path: Path) -> None:
    """Dead host code is never given kind bindings, so it must not exist."""
    source = source_with_helper(
        "_dead",
        "value",
        "return value + 1",
        "_kernel[(1,)](x_local, out)",
    )
    result = decision_for(tmp_path, source)
    assert result["ok"] is False, result
    assert any("unreachable host function" in error for error in result["errors"]), result


@pytest.mark.parametrize(
    "call",
    (
        'setattr(torch.Tensor, "__sub__", torch.mul)',
        'getattr(torch, "mul")',
        'eval("1")',
        'exec("pass")',
        "torch.load(out)",
        "torch.compile(out)",
    ),
)
def test_dynamic_and_interpreter_mutation_calls_are_rejected_anywhere(
    tmp_path: Path, call: str
) -> None:
    """The blocklist is global: reachability must never gate it."""
    assert_rejected(tmp_path, source_with_body(f"{call}\n_kernel[(1,)](x_local, out)"))


def test_dunder_string_constant_is_rejected(tmp_path: Path) -> None:
    """A dunder hidden in a string is the same surface as a dunder attribute."""
    assert_rejected(tmp_path, source_with_body('name = "__sub__"\n_kernel[(1,)](x_local, out)'))


# Flux MoE-specific policy surface from IMPLEMENTATION_PLAN.md Task 2.
def test_torch_distributed_import_is_rejected(tmp_path: Path) -> None:
    assert_rejected(
        tmp_path,
        source_with_body("pass", "import torch.distributed as dist\n"),
    )


def test_host_peer_workspace_copy_is_rejected(tmp_path: Path) -> None:
    assert_rejected(tmp_path, source_with_body("workspace_ptrs.copy_(workspace)"))


def test_host_moe_matmul_is_rejected(tmp_path: Path) -> None:
    assert_rejected(tmp_path, source_with_body("temporary = x_local @ w[0].t()"))


def test_environment_read_is_rejected(tmp_path: Path) -> None:
    assert_rejected(
        tmp_path,
        source_with_body('temporary = os.environ["TOKEN"]', "import os\n"),
    )


def test_release_atomic_and_peer_store_are_admitted(tmp_path: Path) -> None:
    source = VALID_SOURCE.replace(
        "    tl.store(destination + offsets, tl.load(source + offsets))\n",
        (
            "    value = tl.load(source + offsets)\n"
            "    tl.store(destination + offsets, value)\n"
            "    tl.store(source + offsets, value)\n"
            '    tl.atomic_xchg(destination + offsets, 1, sem="release", scope="sys")\n'
        ),
    )
    assert decision_for(tmp_path, source)["ok"] is True


def test_workspace_integer_view_and_slice_are_admitted(tmp_path: Path) -> None:
    source = VALID_SOURCE.replace(
        "    _launch_copy(x_local, out)\n",
        "    flags = workspace[:128].view(torch.int32)\n    _launch_copy(x_local, flags)\n",
    )
    assert decision_for(tmp_path, source)["ok"] is True


def test_kernel_acquire_spin_is_admitted(tmp_path: Path) -> None:
    source = VALID_SOURCE.replace(
        "    tl.store(destination + offsets, tl.load(source + offsets))\n",
        (
            '    while tl.atomic_add(destination, 0, sem="acquire", scope="gpu") == 0:\n'
            "        pass\n"
            "    tl.store(destination + offsets, tl.load(source + offsets))\n"
        ),
    )
    assert decision_for(tmp_path, source)["ok"] is True


def test_host_per_expert_launch_loop_is_rejected(tmp_path: Path) -> None:
    assert_rejected(
        tmp_path,
        source_with_body("for expert in range(8):\n    _kernel[(1,)](x_local, out)"),
    )
