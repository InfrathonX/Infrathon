from __future__ import annotations

from pathlib import Path

import pytest

from conftest import load_module


admission = load_module("attn_qat_source_admission", "images/judge/source_admission.py")

ABI = "q, k, v, softmax_scale, causal, o"
BACKWARD_ABI = "q, k, v, softmax_scale, causal, o, saved, do, dq, dk, dv"
# Every template ships both entry points: each one has to launch a kernel on its own call
# graph, so a source that only defines the forward is rejected for a reason that has
# nothing to do with what the test is probing.
BACKWARD_SOURCE = """

def attn_qat_bwd(
    q, k, v, softmax_scale, causal, o, saved, do, dq, dk, dv
):
    _backward_kernel[(1,)](q, dq)
    return None
"""
BACKWARD_KERNEL = """

@triton.jit
def _backward_kernel(source, destination):
    tl.store(destination, tl.load(source))
"""


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

def attn_qat_fwd(
    q, k, v, softmax_scale, causal, o
):
    _launch_copy(q, o)
    return None
"""


def _with_backward(source: str) -> str:
    """Give a template the backward entry point unless it already declares its own.

    Both entry points are required and each must launch a kernel on its own call graph, so a
    template that only carries the forward would be rejected for a reason unrelated to what
    the test is probing.
    """
    if "def attn_qat_bwd(" in source or not source.strip():
        return source
    return source + BACKWARD_KERNEL + BACKWARD_SOURCE


def make_submission(
    root: Path,
    *,
    selected: int = 1,
    selected_source: str = VALID_SOURCE,
    with_backward: bool = True,
) -> Path:
    initial = root / "candidates/kernel_v001"
    initial.mkdir(parents=True)
    if with_backward:
        selected_source = _with_backward(selected_source)
    initial_source = selected_source if selected == 1 else "# retained old experiment\n"
    (initial / "submission.py").write_text(initial_source, encoding="utf-8")
    if selected != 1:
        selected_dir = root / f"candidates/kernel_v{selected:03d}"
        selected_dir.mkdir()
        (selected_dir / "submission.py").write_text(selected_source, encoding="utf-8")
    (root / "FINAL_VERSION").write_text(f"{selected}\n", encoding="utf-8")
    (root / "notes.md").write_text("candidate notes\n", encoding="utf-8")
    return root


def decision_for(tmp_path: Path, source: str, with_backward: bool = True) -> dict[str, object]:
    return admission.validate_submission(
        make_submission(tmp_path, selected_source=source, with_backward=with_backward)
    )


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
def _kernel(q, out):
    value = tl.load(q)
    tl.store(out, value)

def attn_qat_fwd(
    q, k, v, softmax_scale, causal, o
):
{indented}
    _kernel[(1,)](q, o)
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
def _kernel(q, out):
    tl.store(out, tl.load(q))

def {helper_name}({helper_parameters}):
{helper_lines}

def attn_qat_fwd(
    q, k, v, softmax_scale, causal, o
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
def _kernel(q, out):
    tl.store(out, tl.load(q))

{helper_source}

def attn_qat_fwd(
    q, k, v, softmax_scale, causal, o
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

def attn_qat_fwd(
    q, k, v, softmax_scale, causal, o
):
    width = q.shape[-1]
    rows = q.shape[0] * q.shape[1]
    scratch = torch.empty((rows, width), device=q.device, dtype=q.dtype)
    second = torch.empty_like(scratch)
    _kernel[(rows,)](q, scratch, n=width)
    _kernel[(rows,)](scratch, second, n=width)
    _kernel[(rows,)](second, o, n=width)
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
        # right arity, wrong names
        "q, k, v, scale, out",
        # right names, wrong order
        "q, k, v, o, softmax_scale",
        # extra parameter
        "q, k, v, softmax_scale, causal, o, extra",
        # the backward's parameters on the forward
        "q, k, v, softmax_scale, causal, o, saved, do, dq, dk, dv",
        # a different task's ABI
        "q, kv, indices, sm_scale, attn_sink, topk_length, o, max_logits, lse",
        "*args",
        # defaults and keyword-only forms are not the contract
        "q, k, v, softmax_scale, causal, o=None",
        "q, k, v, softmax_scale, *, o",
    ),
)
def test_selected_forward_requires_the_exact_five_parameter_abi(
    tmp_path: Path, parameters: str
) -> None:
    source = VALID_SOURCE.replace(
        ABI + "\n",
        parameters + "\n",
    )
    assert_rejected(tmp_path, source)


@pytest.mark.parametrize(
    "parameters",
    (
        # missing the saved bundle
        "q, k, v, softmax_scale, causal, o, do, dq, dk, dv",
        # the gradients in the wrong order
        "q, k, v, softmax_scale, causal, o, saved, do, dk, dq, dv",
        # extra parameter
        "q, k, v, softmax_scale, causal, o, saved, do, dq, dk, dv, extra",
        "**kwargs",
    ),
)
def test_selected_backward_requires_the_exact_ten_parameter_abi(
    tmp_path: Path, parameters: str
) -> None:
    source = VALID_SOURCE + BACKWARD_KERNEL + BACKWARD_SOURCE.replace(
        BACKWARD_ABI + "\n", parameters + "\n"
    )
    assert_rejected(tmp_path, source)


def test_a_forward_without_a_backward_is_rejected(tmp_path: Path) -> None:
    result = decision_for(tmp_path, VALID_SOURCE, with_backward=False)
    assert result["ok"] is False, result
    assert any("missing attn_qat_bwd entrypoint" in e for e in result["errors"]), result


def test_each_entrypoint_needs_its_own_kernel_launch(tmp_path: Path) -> None:
    # A backward that only reads the bundle and returns has no reachable kernel of its own.
    source = VALID_SOURCE + """

def attn_qat_bwd(
    q, k, v, softmax_scale, causal, o, saved, do, dq, dk, dv
):
    return None
"""
    result = decision_for(tmp_path, source)
    assert result["ok"] is False, result
    assert any(
        "attn_qat_bwd call graph must launch" in error for error in result["errors"]
    ), result


@pytest.mark.parametrize(
    "name",
    ("attn_qat", "attn_qat_forward", "attnqat_fwd", "fwd"),
)
def test_entrypoint_name_must_be_exact(tmp_path: Path, name: str) -> None:
    source = VALID_SOURCE.replace("def attn_qat_fwd(", f"def {name}(")
    result = decision_for(tmp_path, source)
    assert result["ok"] is False, result
    assert any("missing attn_qat_fwd entrypoint" in e for e in result["errors"]), result


@pytest.mark.parametrize(
    "source",
    (
        "def other(" + ABI + "):\n    return None\n",
        "def attn_qat_fwd(" + ABI + "):\n    return None\n",
        (
            "import triton\nimport triton.language as tl\n"
            "@triton.jit\ndef _unused(q, out):\n    tl.store(out, tl.load(q))\n"
            "def attn_qat_fwd(" + ABI + "):\n"
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
        "temporary = torch.softmax(q, dim=-1)",
        "temporary = q.softmax(dim=-1)",
        "temporary = torch.matmul(q, k)",
        "temporary = q.matmul(k)",
        "temporary = torch.nn.functional.scaled_dot_product_attention(q, k, v)",
        "temporary = q + k",
        "o.copy_(q)",
        "o.zero_()",
        "q.add_(1)",
        "o[0] = q[0]",
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
        ("torch.ops.vendor.kernel(q)", "import torch\n"),
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
        "torch.__dict__['ops'].vendor.kernel(q)",
        "torch.nn.functional.__dict__['scaled_dot_product_attention'](q, k, v)",
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
        "temporary = q.__class__",
        "temporary = attn_qat_fwd.__globals__",
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
    source=source_with_body(f"temporary = torch.empty_like(q, dtype={dtype})")
    assert decision_for(tmp_path,source)["ok"] is True

def test_host_tensor_mean_is_rejected(tmp_path: Path) -> None:
    assert_rejected(tmp_path,source_with_body("temporary = k.mean(dim=1)"))


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
            "launch(q, k, v)",
        ),
        (
            "copy",
            "copy = torch.Tensor.copy_\ncopy(o, q)",
        ),
        (
            "helper",
            "helper: object = torch.softmax\nhelper(q, dim=-1)",
        ),
        (
            "helper",
            "(helper := torch.softmax)\nhelper(q, dim=-1)",
        ),
        (
            "helper",
            "for helper in [torch.softmax]:\n    pass\nhelper(q, dim=-1)",
        ),
    ),
)
def test_reachable_helper_identity_cannot_be_shadowed_by_runtime_bindings(
    tmp_path: Path, helper_name: str, entry_body: str
) -> None:
    source = source_with_helper(
        helper_name,
        "q, out",
        "_kernel[(1,)](q, o)",
        entry_body,
    )
    assert_rejected(tmp_path, source)


def test_module_assignment_cannot_replace_a_certified_helper(tmp_path: Path) -> None:
    source = source_with_helper(
        "helper",
        "q, out",
        "_kernel[(1,)](q, o)",
        "helper(q, o)",
    ).replace(
        "\ndef attn_qat_fwd(",
        "\nhelper = 0\n\ndef attn_qat_fwd(",
    )
    assert_rejected(tmp_path, source)


def test_nested_scope_parameter_cannot_shadow_a_certified_helper(
    tmp_path: Path,
) -> None:
    source = source_with_helper(
        "helper",
        "q, out",
        "_kernel[(1,)](q, o)",
        "callback = lambda helper: helper\nhelper(q, o)",
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
        "q, out",
        "_kernel[(1,)](q, o)",
        f"if {condition}:\n    _kernel[(1,)](q, o)",
    )
    assert_rejected(tmp_path, source)


def test_launch_after_unconditional_return_does_not_certify_entrypoint(
    tmp_path: Path,
) -> None:
    source = source_with_body("return None\n_kernel[(1,)](q, o)")
    assert_rejected(tmp_path, source)


def test_helper_call_after_unconditional_return_is_not_reachable(
    tmp_path: Path,
) -> None:
    source = VALID_SOURCE.replace(
        "    _launch_copy(q, o)\n    return None\n",
        "    return None\n    _launch_copy(q, o)\n",
    )
    assert_rejected(tmp_path, source)


def test_live_conditional_specializations_can_each_launch_triton(tmp_path: Path) -> None:
    source = source_with_body(
        "if q.shape[0] > 1:\n"
        "    _kernel[(1,)](q, o)\n"
        "else:\n"
        "    _kernel[(1,)](q, o)"
    )
    assert decision_for(tmp_path, source)["ok"] is True


@pytest.mark.parametrize(
    ("parameters", "helper_body", "entry_body"),
    (
        ("value, /", "temporary = value + value", "helper(q)\n_kernel[(1,)](q, o)"),
        ("*values", "temporary = values[0] + values[0]", "helper(q)\n_kernel[(1,)](q, o)"),
        ("*, value", "temporary = value + value", "helper(value=q)\n_kernel[(1,)](q, o)"),
        ("**values", "temporary = values['value'] + values['value']", "helper(value=q)\n_kernel[(1,)](q, o)"),
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
        "grid = grid_for(q.numel())\n_kernel[(grid,)](q, o)",
    )
    assert decision_for(tmp_path, source)["ok"] is True


def test_tensor_taint_propagates_through_multiple_helper_calls(tmp_path: Path) -> None:
    source = source_with_helper(
        "outer",
        "value",
        "inner(value)",
        "outer(q)\n_kernel[(1,)](q, o)",
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
    subject = "{'shape': q.shape}" if "shape" in pattern else "q.shape"
    source = source_with_helper(
        "helper",
        "q, out",
        "_kernel[(1,)](q, o)",
        f"match {subject}:\n    case {pattern}:\n        pass\nhelper(q, o)",
    )
    assert_rejected(tmp_path, source)


@pytest.mark.parametrize(
    "entry_body",
    (
        "try:\n"
        "    return None\n"
        "except RuntimeError:\n"
        "    _kernel[(1,)](q, o)",
        "while True:\n    pass\n_kernel[(1,)](q, o)",
        "if 1 == 2:\n    _kernel[(1,)](q, o)",
        "unused = [_kernel[(1,)](q, o) for _ in ()]",
    ),
)
def test_statically_unreachable_launch_forms_do_not_certify_entrypoint(
    tmp_path: Path, entry_body: str
) -> None:
    source = source_with_helper(
        "_unused",
        "q, out",
        "_kernel[(1,)](q, o)",
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
        "", f"if {condition}:\n    _kernel[(1,)](q, o)"
    )
    assert decision_for(tmp_path, source)["ok"] is True


def test_named_expression_cannot_hide_tensor_taint_from_a_helper(
    tmp_path: Path,
) -> None:
    source = source_with_helper(
        "host_math",
        "value",
        "temporary = value + value",
        "host_math((value := q))\n_kernel[(1,)](q, o)",
    )
    assert_rejected(tmp_path, source)


def test_branch_join_keeps_possible_tensor_taint(tmp_path: Path) -> None:
    source = source_with_helper(
        "host_math",
        "value",
        "temporary = value + value",
        "value = q\n"
        "if softmax_scale > 0:\n"
        "    value = 1\n"
        "host_math(value)\n"
        "_kernel[(1,)](q, o)",
    )
    assert_rejected(tmp_path, source)


def test_oracle_source_with_kernel_loop_remains_allowed(tmp_path: Path) -> None:
    oracle = (
        Path(__file__).resolve().parents[1]
        / "oracle/candidates/kernel_v001/submission.py"
    )
    assert decision_for(tmp_path, oracle.read_text(encoding="utf-8"))["ok"] is True


def test_selected_oracle_archive_is_admitted(tmp_path: Path) -> None:
    """The oracle archive (FINAL_VERSION, notes.md, selected source) passes Judge admission."""
    import shutil

    oracle = Path(__file__).resolve().parents[1] / "oracle"
    version = int((oracle / "FINAL_VERSION").read_text(encoding="utf-8").strip())
    selected = f"candidates/kernel_v{version:03d}/submission.py"
    root = tmp_path / "archive"
    for member in ("FINAL_VERSION", "notes.md", selected):
        (root / member).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(oracle / member, root / member)
    result = admission.validate_submission(root)
    assert result["ok"] is True, result["errors"]
    assert result["final_version"] == version


@pytest.mark.parametrize(
    ("helpers", "entry_body"),
    (
        (
            "def produce():\n"
            "    return torch.empty((1,), device='cuda')",
            "value = produce()\ntemporary = value + value\n_kernel[(1,)](q, o)",
        ),
        (
            "def produce(count):\n"
            "    return torch.empty((count,), device='cuda')",
            "value = produce(1)\ntemporary = value + value\n_kernel[(1,)](q, o)",
        ),
        (
            "def produce(value):\n    return value",
            "value = produce(q)\ntemporary = value + value\n_kernel[(1,)](q, o)",
        ),
        (
            "def produce(flag):\n"
            "    if flag > 0:\n"
            "        return 1\n"
            "    return torch.empty((1,), device='cuda')",
            "value = produce(softmax_scale)\n"
            "temporary = value + value\n"
            "_kernel[(1,)](q, o)",
        ),
        (
            "def inner(count):\n"
            "    return torch.empty((count,), device='cuda')\n\n"
            "def outer(count):\n"
            "    return inner(count)",
            "value = outer(1)\ntemporary = value + value\n_kernel[(1,)](q, o)",
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
        "value = recurse(1)\n_kernel[(1,)](q, o)",
    )
    assert_rejected(tmp_path, source)


def test_dead_tensor_return_does_not_taint_scalar_helper(tmp_path: Path) -> None:
    source = source_with_host_helpers(
        "def grid_for(elements):\n"
        "    if False:\n"
        "        return torch.empty((1,), device='cuda')\n"
        "    return triton.cdiv(elements, 128)",
        "grid = grid_for(q.numel())\n_kernel[(grid,)](q, o)",
    )
    assert decision_for(tmp_path, source)["ok"] is True


@pytest.mark.parametrize(
    "statement",
    (
        "value = 0\nvalue += q",
        "softmax_scale += q",
        "q += softmax_scale",
        "o += softmax_scale",
    ),
)
def test_host_augmented_assignment_is_rejected(
    tmp_path: Path, statement: str
) -> None:
    assert_rejected(tmp_path, source_with_body(statement))


@pytest.mark.parametrize(
    "statement",
    (
        "head, *values = (0, q)\ntemporary = values[0] + values[0]",
        "[*values] = [q]\ntemporary = values[0] + values[0]",
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
        "grid = grid_for(*(q.numel(),))\n_kernel[(grid,)](q, o)",
    )
    assert_rejected(tmp_path, source)


@pytest.mark.parametrize(
    "entry_body",
    (
        "if 1 == 2 < softmax_scale:\n    _kernel[(1,)](q, o)",
        "if softmax_scale or True:\n"
        "    return None\n"
        "else:\n"
        "    _kernel[(1,)](q, o)",
        "if softmax_scale and False:\n    _kernel[(1,)](q, o)",
    ),
)
def test_absorbing_truth_values_do_not_certify_dead_launches(
    tmp_path: Path, entry_body: str
) -> None:
    source = source_with_helper(
        "_unused",
        "q, out",
        "_kernel[(1,)](q, o)",
        entry_body,
    )
    assert_rejected(tmp_path, source)


@pytest.mark.parametrize(
    "entry_body",
    (
        "if not (softmax_scale or True):\n    _kernel[(1,)](q, o)",
        "if not (softmax_scale and False):\n"
        "    return None\n"
        "else:\n"
        "    _kernel[(1,)](q, o)",
        "if not not (softmax_scale or True):\n"
        "    return None\n"
        "else:\n"
        "    _kernel[(1,)](q, o)",
        "if not not (softmax_scale and False):\n    _kernel[(1,)](q, o)",
    ),
)
def test_nested_not_truth_does_not_certify_dead_launches(
    tmp_path: Path, entry_body: str
) -> None:
    source = source_with_helper(
        "_unused",
        "q, out",
        "_kernel[(1,)](q, o)",
        entry_body,
    )
    assert_rejected(tmp_path, source)


@pytest.mark.parametrize(
    "condition",
    (
        "not (softmax_scale and False)",
        "not not (softmax_scale or True)",
    ),
)
def test_nested_not_truth_keeps_live_launches(
    tmp_path: Path, condition: str
) -> None:
    # The conditional launch must be the only launch in the module, so the
    # assertion below cannot be satisfied by some other live launch.
    source = source_with_host_helpers(
        "", f"if {condition}:\n    _kernel[(1,)](q, o)"
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
        "if 1000 == 1000 and q.shape[0] > 1:\n"
        "    _kernel[(1,)](q, o)\n"
        "else:\n"
        "    _kernel[(1,)](q, o)"
    )
    assert decision_for(tmp_path, source)["ok"] is True


def test_identity_comparison_inside_triton_kernel_remains_allowed(
    tmp_path: Path,
) -> None:
    source = """\
import triton
import triton.language as tl

@triton.jit
def _kernel(q, out, flag: tl.constexpr):
    if flag is None:
        tl.store(out, tl.load(q))

def attn_qat_fwd(
    q, k, v, softmax_scale, causal, o
):
    _kernel[(1,)](q, o, flag=None)
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
def attn_qat_fwd(
    q, k, v, softmax_scale, causal, o
):
    _kernel[(1,)](o, o.numel(), BLOCK=4096)
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
def _kernel(q, out):
    tl.store(out, tl.load(q))

def attn_qat_fwd(
    q, k, v, softmax_scale, causal, o
):
    _kernel[_grid](q, o)
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
def _kernel(q, out, BLOCK: tl.constexpr):
    tl.store(out, tl.load(q))

def attn_qat_fwd(
    q, k, v, softmax_scale, causal, o
):
    _kernel[_grid](q, o, BLOCK=64)
    return None
"""
    assert decision_for(tmp_path / "legal", legal)["ok"] is True
    assert_rejected(
        tmp_path / "tainted",
        legal.replace(
            '    return (meta["BLOCK"],)',
            '    torch.sum(q)\n    return (meta["BLOCK"],)',
        ),
    )


def test_unreachable_host_function_is_rejected(tmp_path: Path) -> None:
    """Dead host code is never given kind bindings, so it must not exist."""
    source = source_with_helper(
        "_dead",
        "value",
        "return value + 1",
        "_kernel[(1,)](q, o)",
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
        "torch.load(o)",
        "torch.compile(o)",
    ),
)
def test_dynamic_and_interpreter_mutation_calls_are_rejected_anywhere(
    tmp_path: Path, call: str
) -> None:
    """The blocklist is global: reachability must never gate it."""
    assert_rejected(tmp_path, source_with_body(f"{call}\n_kernel[(1,)](q, o)"))


def test_dunder_string_constant_is_rejected(tmp_path: Path) -> None:
    """A dunder hidden in a string is the same surface as a dunder attribute."""
    assert_rejected(tmp_path, source_with_body('name = "__sub__"\n_kernel[(1,)](q, o)'))


AUTOTUNE_SOURCE = """\
import triton
import triton.language as tl

@triton.autotune(
    configs=[triton.Config({"BLOCK": 64}, num_warps=4)],
    key=["n"],
)
@triton.jit
def _kernel(q, out, n, BLOCK: tl.constexpr):
    tl.store(out, tl.load(q))

def attn_qat_fwd(
    q, k, v, softmax_scale, causal, o
):
    _kernel[(1,)](q, o, q.numel())
    return None
"""

HEURISTICS_SOURCE = """\
import triton
import triton.language as tl

@triton.heuristics({"BLOCK": lambda args: 64})
@triton.jit
def _kernel(q, out, BLOCK: tl.constexpr):
    tl.store(out, tl.load(q))

def attn_qat_fwd(
    q, k, v, softmax_scale, causal, o
):
    _kernel[(1,)](q, o)
    return None
"""


def test_autotune_decorator_is_rejected_with_a_named_message(tmp_path: Path) -> None:
    """A search that compiles during evaluation is not a bounded kernel set."""
    result = decision_for(tmp_path, AUTOTUNE_SOURCE)
    assert result["ok"] is False, result
    assert any("triton.autotune is not permitted" in e for e in result["errors"]), result


def test_heuristics_decorator_is_rejected_with_a_named_message(tmp_path: Path) -> None:
    """Heuristics values are participant callables run at launch time."""
    result = decision_for(tmp_path, HEURISTICS_SOURCE)
    assert result["ok"] is False, result
    assert any("triton.heuristics is not permitted" in e for e in result["errors"]), result


@pytest.mark.parametrize(
    "statement",
    (
        "torch.softmax(q, dim=-1)",
        "temporary = torch.softmax(q, dim=-1)",
        "temporary = torch.matmul(q, k)",
        "temporary = torch.nn.functional.scaled_dot_product_attention(q, k, v)",
    ),
)
def test_host_attention_math_cannot_produce_the_output(
    tmp_path: Path, statement: str
) -> None:
    """The whole operator is the timed kernel's contract, not a host post-pass."""
    assert_rejected(
        tmp_path, source_with_body(f"{statement}\n_kernel[(1,)](q, o)", "import torch\n")
    )


def test_host_copy_into_the_output_is_rejected(tmp_path: Path) -> None:
    assert_rejected(tmp_path, source_with_body("o.copy_(q)\n_kernel[(1,)](q, o)"))


@pytest.mark.parametrize("name", ("q", "k", "v", "o"))
def test_forward_parameters_are_tensor_kinded_not_scalar(
    tmp_path: Path, name: str
) -> None:
    """Only `softmax_scale` is a scalar; every other parameter is tensor-tainted."""
    assert_rejected(
        tmp_path, source_with_body(f"temporary = {name} + {name}\n_kernel[(1,)](q, o)")
    )


@pytest.mark.parametrize("name", ("do", "dq", "dk", "dv"))
def test_backward_parameters_are_tensor_kinded_not_scalar(
    tmp_path: Path, name: str
) -> None:
    source = VALID_SOURCE + BACKWARD_KERNEL + f"""

def attn_qat_bwd(
    q, k, v, softmax_scale, causal, o, saved, do, dq, dk, dv
):
    temporary = {name} + {name}
    _backward_kernel[(1,)](q, dq)
    return None
"""
    assert_rejected(tmp_path, source)


@pytest.mark.parametrize(
    "body",
    (
        # indexing the bundle recovers a tensor, and tensor math on the host is rejected
        "temporary = saved[0] + saved[1]",
        # so does unpacking it
        "first, second = saved\ntemporary = first * 2",
        # the bundle itself is as opaque as a tensor
        "temporary = saved + saved",
        "count = len(saved)\ntemporary = saved[0].sum()",
    ),
)
def test_saved_bundle_is_opaque_on_the_host(tmp_path: Path, body: str) -> None:
    lines = "\n".join(f"    {line}" for line in body.splitlines())
    source = VALID_SOURCE + BACKWARD_KERNEL + f"""

def attn_qat_bwd(
    q, k, v, softmax_scale, causal, o, saved, do, dq, dk, dv
):
{lines}
    _backward_kernel[(1,)](q, dq)
    return None
"""
    assert_rejected(tmp_path, source)


def test_saved_bundle_may_be_unpacked_and_handed_to_a_kernel(tmp_path: Path) -> None:
    source = VALID_SOURCE + BACKWARD_KERNEL + """

def attn_qat_bwd(
    q, k, v, softmax_scale, causal, o, saved, do, dq, dk, dv
):
    quantized_q, quantized_k = saved
    _backward_kernel[(1,)](quantized_q, dq)
    _backward_kernel[(1,)](quantized_k, dk)
    return None
"""
    assert decision_for(tmp_path, source)["ok"] is True


def test_forward_may_return_the_saved_bundle_but_the_backward_may_not(
    tmp_path: Path,
) -> None:
    forward_returns = VALID_SOURCE.replace(
        "    _launch_copy(q, o)\n    return None\n",
        "    _launch_copy(q, o)\n    scratch = torch.empty_like(q)\n    return (scratch,)\n",
    )
    assert decision_for(tmp_path / "forward", forward_returns)["ok"] is True
    backward_returns = VALID_SOURCE + BACKWARD_KERNEL + """

def attn_qat_bwd(
    q, k, v, softmax_scale, causal, o, saved, do, dq, dk, dv
):
    _backward_kernel[(1,)](q, dq)
    return dq
"""
    result = decision_for(tmp_path / "backward", backward_returns)
    assert result["ok"] is False, result
    assert any("attn_qat_bwd must return None" in error for error in result["errors"])


def test_softmax_scale_stays_a_scalar(tmp_path: Path) -> None:
    """Host arithmetic on the one scalar parameter is legitimate."""
    source = source_with_body("scale = softmax_scale * 2.0\n_kernel[(1,)](q, o)")
    assert decision_for(tmp_path, source)["ok"] is True


# ---------------------------------------------------------------------------------------
# Additions for this ABI: the Triton scratch allocator, host-side TMA descriptors, the
# inline-PTX and target-routing bans, and Torch autograd.
# ---------------------------------------------------------------------------------------

CONFORMING_ALLOCATOR = """
def _scratch(size, alignment, stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")
"""


def allocator_source(callback: str, setter: str = "triton.set_allocator(_scratch)") -> str:
    """A valid submission whose forward installs `callback` as the Triton allocator."""
    return f"""\
import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(source, destination):
    tl.store(destination, tl.load(source))
{callback}

def attn_qat_fwd(
    q, k, v, softmax_scale, causal, o
):
    {setter}
    _kernel[(1,)](q, o)
    return None
"""


def test_conforming_scratch_allocator_is_admitted(tmp_path: Path) -> None:
    result = decision_for(tmp_path, allocator_source(CONFORMING_ALLOCATOR))
    assert result["ok"] is True, result


def test_conforming_scratch_allocator_may_carry_a_docstring(tmp_path: Path) -> None:
    # The shipped seed's callback documents itself.  A leading string constant executes
    # nothing, so it is dropped before the one-statement body check.
    callback = """
def _scratch(size, alignment, stream):
    "Scratch allocator for the device-side TMA descriptors."
    return torch.empty(size, dtype=torch.int8, device="cuda")
"""
    result = decision_for(tmp_path, allocator_source(callback))
    assert result["ok"] is True, result


def test_allocator_callback_with_a_fourth_parameter_is_rejected(tmp_path: Path) -> None:
    # Triton calls the callback with exactly (size, alignment, stream); a fourth
    # parameter means the name does not denote the allocator route at all.
    callback = """
def _scratch(size, alignment, stream, extra):
    return torch.empty(size, dtype=torch.int8, device="cuda")
"""
    assert_rejected(tmp_path, allocator_source(callback))


def test_allocator_callback_defined_inside_the_entrypoint_is_rejected(
    tmp_path: Path,
) -> None:
    # A nested definition is not a module-level function, so the exact-shape check never
    # sees it and the installation fails closed.
    source = """\
import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(source, destination):
    tl.store(destination, tl.load(source))


def attn_qat_fwd(
    q, k, v, softmax_scale, causal, o
):
    def _scratch(size, alignment, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")
    triton.set_allocator(_scratch)
    _kernel[(1,)](q, o)
    return None
"""
    assert_rejected(tmp_path, source)


def test_allocator_device_cannot_be_a_call(tmp_path: Path) -> None:
    # A call in the device slot is a second execution route inside the callback.
    callback = """
def _scratch(size, alignment, stream):
    return torch.empty(size, dtype=torch.int8, device=torch.cuda.current_device())
"""
    assert_rejected(tmp_path, allocator_source(callback))


@pytest.mark.parametrize(
    "callback",
    (
        # wrong parameter count
        "\ndef _scratch(size, alignment):\n"
        "    return torch.empty(size, dtype=torch.int8, device='cuda')\n",
        # more than one statement: the callback becomes a general execution route
        "\ndef _scratch(size, alignment, stream):\n"
        "    total = size * 2\n"
        "    return torch.empty(total, dtype=torch.int8, device='cuda')\n",
        # a size that is not the first parameter
        "\ndef _scratch(size, alignment, stream):\n"
        "    return torch.empty(alignment, dtype=torch.int8, device='cuda')\n",
        # the wrong scratch dtype
        "\ndef _scratch(size, alignment, stream):\n"
        "    return torch.empty(size, dtype=torch.float32, device='cuda')\n",
        # a different allocation call
        "\ndef _scratch(size, alignment, stream):\n"
        "    return torch.zeros(size, dtype=torch.int8, device='cuda')\n",
        # missing the dtype
        "\ndef _scratch(size, alignment, stream):\n"
        "    return torch.empty(size, device='cuda')\n",
        # a decorated callback
        "\n@triton.jit\ndef _scratch(size, alignment, stream):\n"
        "    return torch.empty(size, dtype=torch.int8, device='cuda')\n",
    ),
)
def test_non_conforming_allocator_callbacks_are_rejected(
    tmp_path: Path, callback: str
) -> None:
    assert_rejected(tmp_path, allocator_source(callback))


@pytest.mark.parametrize(
    "setter",
    (
        # a lambda is not a module-level function
        "triton.set_allocator(lambda size, alignment, stream: q)",
        # a name that is not a declared function
        "triton.set_allocator(torch.empty)",
        # keyword form, so the callback identity is not a plain name
        "triton.set_allocator(fn=_scratch)",
        # more than one argument
        "triton.set_allocator(_scratch, _scratch)",
    ),
)
def test_non_conforming_allocator_installations_are_rejected(
    tmp_path: Path, setter: str
) -> None:
    assert_rejected(tmp_path, allocator_source(CONFORMING_ALLOCATOR, setter))


def test_host_side_tensor_descriptors_are_admitted(tmp_path: Path) -> None:
    source = """\
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

@triton.jit
def _kernel(descriptor, destination):
    destination.store([0, 0], descriptor.load([0, 0]))

def attn_qat_fwd(
    q, k, v, softmax_scale, causal, o
):
    source_descriptor = TensorDescriptor.from_tensor(q, [16, 16])
    destination_descriptor = TensorDescriptor.from_tensor(o, [16, 16])
    _kernel[(1,)](source_descriptor, destination_descriptor)
    return None
"""
    assert decision_for(tmp_path, source)["ok"] is True


def test_a_descriptor_is_opaque_on_the_host(tmp_path: Path) -> None:
    source = """\
import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

@triton.jit
def _kernel(source, destination):
    tl.store(destination, tl.load(source))

def attn_qat_fwd(
    q, k, v, softmax_scale, causal, o
):
    descriptor = TensorDescriptor.from_tensor(q, [16, 16])
    total = descriptor + descriptor
    _kernel[(1,)](q, o)
    return None
"""
    assert_rejected(tmp_path, source)


@pytest.mark.parametrize(
    "module",
    (
        "from triton.tools.tensor_descriptor import TensorDescriptor as TD",
        "from triton.tools import tensor_descriptor",
        "from torch.nn.functional import softmax",
    ),
)
def test_other_from_imports_stay_rejected(tmp_path: Path, module: str) -> None:
    assert_rejected(tmp_path, f"{module}\n" + VALID_SOURCE)


def test_inline_asm_elementwise_is_rejected_with_a_named_message(tmp_path: Path) -> None:
    source = """\
import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(source, destination):
    value = tl.inline_asm_elementwise(
        "cvt.rn.satfinite.e2m1x2.f32 $0, $1, $2;",
        "=r,r,r",
        [tl.load(source)],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    tl.store(destination, value)

def attn_qat_fwd(
    q, k, v, softmax_scale, causal, o
):
    _kernel[(1,)](q, o)
    return None
"""
    result = decision_for(tmp_path, source)
    assert result["ok"] is False, result
    assert any("inline_asm_elementwise is not permitted" in e for e in result["errors"])


def test_target_info_routing_is_rejected_with_a_named_message(tmp_path: Path) -> None:
    source = """\
import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(source, destination):
    if tl.target_info.cuda_capability_geq(10, 0):
        tl.store(destination, tl.load(source))
    else:
        tl.store(destination, tl.load(source))

def attn_qat_fwd(
    q, k, v, softmax_scale, causal, o
):
    _kernel[(1,)](q, o)
    return None
"""
    result = decision_for(tmp_path, source)
    assert result["ok"] is False, result
    assert any("target_info is not permitted" in e for e in result["errors"])


@pytest.mark.parametrize(
    "statement",
    (
        "torch.autograd.grad(o, q)",
        "temporary = torch.autograd.Function",
        "q.requires_grad_(True)",
        "temporary = q.grad",
    ),
)
def test_autograd_surface_is_rejected(tmp_path: Path, statement: str) -> None:
    assert_rejected(tmp_path, source_with_body(f"{statement}\n_kernel[(1,)](q, o)",
                                               "import torch\n"))


@pytest.mark.parametrize(
    "method",
    ("sort", "argsort", "item", "tolist", "numpy", "cpu", "float"),
)
def test_unlisted_kernel_methods_stay_rejected(tmp_path: Path, method: str) -> None:
    source = f"""\
import torch
import triton
import triton.language as tl

@triton.jit
def _kernel(source, destination):
    value = tl.load(source)
    tl.store(destination, value.{method}())

def attn_qat_fwd(
    q, k, v, softmax_scale, causal, o
):
    _kernel[(1,)](q, o)
    return None
"""
    assert_rejected(tmp_path, source)
