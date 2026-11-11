#!/usr/bin/env python3
"""Structural and selected-source admission for the TileLang SSD task.

The submitted compute must be TileLang.  That cannot be established by
checking that a file imports ``tilelang``: the package deliberately exposes
raw CUDA/C++ source injection, external and packed calls, compiler
callbacks, custom lowering passes, device compile flags, the bundled TVM,
native compiler helpers, and alternative execution backends.  The policy
below is an import **allowlist** plus a denylist over every one of those
routes, inventoried against the pinned 0.1.13 wheel in
``iteration_docs/tilelang-mamba2-ssd-h100-v1/TILELANG_SURFACE.md``.

v2 adds three rules found by probing v1: no double-underscore attribute
access (``x.__matmul__``, ``T.__dict__[...]``), no dynamic attribute access
through ``operator`` or by passing ``getattr`` and friends around as values,
and an allowlist for the ``torch`` surface: allocation, dtypes, devices and
stream queries only.  Tensor *methods* cannot be resolved statically; the
runtime TileLang-only operator record covers them.

This is a diagnostic boundary for code-producing agents and a statement of
the public policy in AGENT_TASK.md.  It is not a Python sandbox, and it does
not replace the container's read-only mounts, absent network, private cache
root, fixed compiler environment, and the runtime operator record.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import stat
import sys
from pathlib import Path
from typing import Any


CANDIDATE_RE = re.compile(r"^kernel_v([0-9]{3})$")
FINAL_VERSION_RE = re.compile(rb"(?:[1-9][0-9]{0,2})\n?\Z")
MAX_VERSIONS = 64
MAX_SOURCE_BYTES = 512 * 1024
MAX_NOTES_BYTES = 64 * 1024
MAX_TOTAL_BYTES = 4 * 1024 * 1024

# --- policy ---------------------------------------------------------------
# Imports are allowlisted at the root, then pruned inside ``tilelang`` itself.
ALLOWED_IMPORT_ROOTS = {
    "abc", "collections", "contextlib", "dataclasses", "enum", "functools",
    "itertools", "math", "numbers", "operator", "tilelang", "torch",
    "typing", "warnings",
}
# The whole torch surface a submission may name.  Everything a submission does
# with torch is allocate its own buffers, describe dtypes and devices, and ask
# for the current stream; all compute belongs in TileLang.
ALLOWED_TORCH_NAMES = frozenset(
    {
        "torch.empty", "torch.empty_like", "torch.empty_strided", "torch.zeros",
        "torch.zeros_like", "torch.full", "torch.full_like",
        "torch.device", "torch.dtype", "torch.Size", "torch.Tensor",
        "torch.no_grad", "torch.inference_mode",
        "torch.cuda.current_stream", "torch.cuda.current_device",
        "torch.cuda.device_count", "torch.cuda.get_device_capability",
        "torch.cuda.get_device_properties", "torch.cuda.is_available",
        "torch.cuda.synchronize",
        "torch.float32", "torch.float", "torch.float16", "torch.half",
        "torch.bfloat16", "torch.float64", "torch.double",
        "torch.float8_e4m3fn", "torch.float8_e5m2",
        "torch.int8", "torch.int16", "torch.int32", "torch.int", "torch.int64",
        "torch.long", "torch.uint8", "torch.bool",
    }
)
# Dynamic attribute access that would turn a string into any name at all.
FORBIDDEN_DYNAMIC_ACCESS = {"operator.attrgetter", "operator.methodcaller"}
# Double-underscore names a submission may still use.
ALLOWED_DUNDER_NAMES = {"__name__", "__doc__", "__qualname__", "__all__"}
# Subtrees of an otherwise allowed root that carry an escape out of the DSL.
FORBIDDEN_IMPORT_PREFIXES = (
    "tilelang.autotuner",
    "tilelang.cache",
    "tilelang.contrib",
    "tilelang.engine",
    "tilelang.env",
    "tilelang.ir",
    "tilelang.jit.adapter",
    "tilelang.profiler",
    "tilelang.testing",
    "tilelang.tools",
    "tilelang.transform",
    "tilelang.tvm",
    "torch.utils.cpp_extension",
    # Trusted harness modules a submission must never reach.
    "candidate_worker",
    "mamba2_ssd_runtime",
    "public_runner",
    "source_admission",
)
# Every fast vendor route to a matrix product.  ``gemm`` is deliberately absent:
# ``T.gemm`` is the intended tensor-core primitive for this task.
FORBIDDEN_GEMM_TAILS = {
    "addbmm", "addmm", "addmv", "baddbmm", "bmm", "chain_matmul", "conv1d",
    "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d", "einsum",
    "inner", "linear", "matmul", "mm", "mv", "tensordot",
    "_convolution", "_int_mm", "_scaled_mm",
}
# Whole existing SSM / linear-attention kernels, not just their GEMMs.
FORBIDDEN_OPERATOR_TAILS = {
    "chunk_gated_delta_rule",
    "mamba_chunk_scan_combined",
    "mamba_split_conv1d_scan_combined",
    "scaled_dot_product_attention",
    "selective_scan_fn",
    "ssd_chunk_scan_combined",
    "ssd_selective_scan",
    "_chunk_cumsum_fwd",
    "_chunk_scan_fwd",
    "_chunk_state_fwd",
    "_state_passing_fwd",
}
# TileLang routes that emit, import, or call code the tile DSL did not generate.
FORBIDDEN_TILELANG_TAILS = {
    # raw source and external kernels
    "cudasourcecodekernel", "import_source", "call_kernel",
    # external / packed / intrinsic calls
    "call_cpacked", "call_cpacked_lowered", "call_extern", "call_intrin",
    "call_llvm_intrin", "call_llvm_pure_intrin", "call_packed",
    "call_packed_lowered", "call_pure_extern", "call_tir",
    "tvm_call_cpacked", "tvm_call_cpacked_lowered", "tvm_call_packed",
    "tvm_call_packed_lowered", "anylist_setitem_call_cpacked",
    "anylist_setitem_call_packed",
    # pointer fabrication
    "access_ptr", "address_of", "make_tensor", "make_tensor_from_addr",
    # build / lower / artefact routes
    "from_database", "par_compile", "export_library", "export_ptx",
    "export_sass", "export_sources", "load_lib", "load_module",
    "get_global_func", "register_func", "register_global_func",
    # autotuning belongs to the Work phase, not to a submission
    "autotune", "autotuner", "from_kernel",
}
# Prefixes of exported names that are inline PTX or raw hardware intrinsics.
# ``T.gemm``/``T.copy`` lower to wgmma and TMA on their own; reaching past them
# would make the task's "no inline PTX" rule incoherent.
FORBIDDEN_TILELANG_TAIL_PREFIXES = ("mma_", "ptx_", "tcgen05_", "tvm_mma")
# Exact dotted names, so that a string method such as ``name.lower()`` is safe.
FORBIDDEN_EXACT_CALLS = {
    "tilelang.compile",
    "tilelang.engine.lower",
    "tilelang.jitkernel",
    "tilelang.lower",
    "tilelang.par_compile",
    "torch.compile",
    "torch.jit.load",
    "torch.load",
}
FORBIDDEN_BUILTIN_CALLS = {
    "__import__", "compile", "delattr", "eval", "exec", "getattr", "globals",
    "locals", "open", "setattr", "type", "vars",
}
FORBIDDEN_PROCESS_OR_LOAD_TAILS = {
    "cdll", "dlopen", "execv", "execve", "fork", "forkpty", "load_inline",
    "loadlibrary", "popen", "putenv", "pydll", "spawnl", "spawnle", "spawnlp",
    "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe", "system",
    "unsetenv",
}
FORBIDDEN_PRIVATE_STREAM_NAMES = {
    "torch.cuda.externalstream",
    "torch.cuda.set_stream",
    "torch.cuda.stream",
}
# Keyword arguments on the TileLang surface that redirect compilation itself.
FORBIDDEN_TILELANG_KEYWORDS = {
    "compile_flags",
    "execution_backend",
    "prelude",
    "source_code_or_path",
    "target",
    "target_host",
}
# Pass-configuration keys that inject flags or a participant-authored pass.
FORBIDDEN_PASS_CONFIG_KEYS = {
    "tir_add_lower_pass",
    "tirx.add_lower_pass",
    "tl.device_compile_flags",
    "tl_device_compile_flags",
}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _require_directory(path: Path, root: Path, errors: list[str]) -> bool:
    label = _relative(path, root)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        errors.append(f"missing required directory: {label}")
        return False
    except OSError as exc:
        errors.append(f"cannot inspect {label}: {exc}")
        return False
    if stat.S_ISLNK(metadata.st_mode):
        errors.append(f"symlink is forbidden: {label}")
        return False
    if not stat.S_ISDIR(metadata.st_mode):
        errors.append(f"required directory is not a directory: {label}")
        return False
    return True


def _list_directory(path: Path, root: Path, errors: list[str]) -> list[Path]:
    try:
        return sorted(path.iterdir(), key=lambda entry: entry.name)
    except OSError as exc:
        errors.append(f"cannot list {_relative(path, root)}: {exc}")
        return []


def _read_text(
    path: Path,
    root: Path,
    errors: list[str],
    *,
    max_bytes: int,
    require_nonempty: bool,
) -> tuple[str | None, bytes | None, int]:
    label = _relative(path, root)
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        errors.append(f"missing required file: {label}")
        return None, None, 0
    except OSError as exc:
        errors.append(f"cannot inspect {label}: {exc}")
        return None, None, 0
    if stat.S_ISLNK(metadata.st_mode):
        errors.append(f"symlink is forbidden: {label}")
        return None, None, 0
    if not stat.S_ISREG(metadata.st_mode):
        errors.append(f"required file is not regular: {label}")
        return None, None, 0
    if metadata.st_nlink != 1:
        errors.append(f"hard-linked file is forbidden: {label}")
    if metadata.st_size > max_bytes:
        errors.append(f"file exceeds {max_bytes} bytes: {label}")
        return None, None, metadata.st_size
    if require_nonempty and metadata.st_size == 0:
        errors.append(f"file must not be empty: {label}")
        return None, b"", 0
    try:
        data = path.read_bytes()
    except OSError as exc:
        errors.append(f"cannot read {label}: {exc}")
        return None, None, metadata.st_size
    if len(data) != metadata.st_size:
        errors.append(f"file changed while being inspected: {label}")
    if data.startswith(b"\xef\xbb\xbf"):
        errors.append(f"UTF-8 BOM is forbidden: {label}")
    if b"\x00" in data:
        errors.append(f"NUL byte is forbidden: {label}")
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        errors.append(f"file is not valid UTF-8: {label}: {exc}")
        return None, data, len(data)
    return text, data, len(data)


def _dotted_name(node: ast.AST, aliases: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        prefix = _dotted_name(node.value, aliases)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return None


def _is_forbidden_import(name: str) -> bool:
    lowered = name.lower()
    return any(
        lowered == prefix or lowered.startswith(prefix + ".")
        for prefix in FORBIDDEN_IMPORT_PREFIXES
    )


def _is_unlisted_import(name: str) -> bool:
    root = name.split(".", 1)[0]
    return bool(root) and root not in ALLOWED_IMPORT_ROOTS


class _SelectedPolicy(ast.NodeVisitor):
    def __init__(self, relative_path: str, tree: ast.Module) -> None:
        self.relative_path = relative_path
        self.aliases: dict[str, str] = {}
        self.errors: list[str] = []
        self.dict_constants: dict[str, ast.Dict] = {}
        # Attribute nodes that are the inner part of a longer chain; the torch
        # allowlist is applied to complete chains only.
        self.inner_attributes: set[int] = set()
        for statement in tree.body:
            if not isinstance(statement, ast.Assign) or not isinstance(statement.value, ast.Dict):
                continue
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    self.dict_constants[target.id] = statement.value

    def _error(self, node: ast.AST, category: str, token: str) -> None:
        line = getattr(node, "lineno", 1)
        self.errors.append(
            f"{self.relative_path}:{line}: forbidden {category}: {token!r}"
        )

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for item in node.names:
            local = item.asname or item.name.split(".", 1)[0]
            self.aliases[local] = item.name
            if _is_forbidden_import(item.name):
                self._error(node, "import", item.name)
            elif _is_unlisted_import(item.name):
                self._error(node, "import outside the allowed set", item.name)
            elif item.name.startswith("torch."):
                self._error(node, "torch submodule outside the allocation surface", item.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        module = node.module or ""
        if node.level:
            self._error(node, "relative import", "." * node.level + module)
        if _is_forbidden_import(module):
            self._error(node, "import", module)
        elif _is_unlisted_import(module):
            self._error(node, "import outside the allowed set", module)
        if module.startswith("torch."):
            self._error(node, "torch submodule outside the allocation surface", module)
        for item in node.names:
            if item.name == "*":
                self._error(node, "star import", f"{module}.*")
            local = item.asname or item.name
            resolved = f"{module}.{item.name}" if module else item.name
            self.aliases[local] = resolved
            if module == "torch" and resolved not in ALLOWED_TORCH_NAMES:
                self._error(node, "torch name outside the allocation surface", resolved)
            if resolved.lower() in FORBIDDEN_DYNAMIC_ACCESS:
                self._error(node, "dynamic attribute access", resolved)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:  # noqa: N802
        if isinstance(node.op, ast.MatMult):
            self._error(node, "existing GEMM operator", "@")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        if (
            node.attr.startswith("__")
            and node.attr.endswith("__")
            and node.attr not in ALLOWED_DUNDER_NAMES
        ):
            self._error(node, "double-underscore attribute access", node.attr)
        exact = _dotted_name(node, self.aliases) or ""
        if id(node) not in self.inner_attributes:
            if exact.startswith("torch.") and exact not in ALLOWED_TORCH_NAMES:
                self._error(node, "torch name outside the allocation surface", exact)
        if isinstance(node.value, ast.Attribute):
            self.inner_attributes.add(id(node.value))
        if exact.lower() in FORBIDDEN_DYNAMIC_ACCESS:
            self._error(node, "dynamic attribute access", exact)
        name = exact.lower()
        if name == "sys.modules" or name.startswith("sys.modules."):
            self._error(node, "trusted harness access", name)
        if name == "torch.ops" or name.startswith("torch.ops."):
            self._error(node, "compiled extension access", name)
        if name == "torch.utils.cpp_extension" or name.startswith(
            "torch.utils.cpp_extension."
        ):
            self._error(node, "runtime extension compilation", name)
        if name == "torch.classes" or name.startswith("torch.classes."):
            self._error(node, "compiled extension access", name)
        if _is_forbidden_import(name):
            self._error(node, "TileLang internal access", name)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        exact = _dotted_name(node, self.aliases) or ""
        name = exact.lower()
        if name == "sys.modules":
            self._error(node, "trusted harness access", name)
        if (
            node.id.startswith("__")
            and node.id.endswith("__")
            and node.id not in ALLOWED_DUNDER_NAMES
        ):
            self._error(node, "double-underscore name", node.id)
        if isinstance(node.ctx, ast.Load):
            # A dynamic builtin passed around as a value (functools.partial,
            # map, a lookup table) is as good as calling it.
            if name in FORBIDDEN_BUILTIN_CALLS:
                self._error(node, "dynamic code or file loading", exact)
            if name in FORBIDDEN_DYNAMIC_ACCESS:
                self._error(node, "dynamic attribute access", exact)
            if exact.startswith("torch.") and exact not in ALLOWED_TORCH_NAMES:
                self._error(node, "torch name outside the allocation surface", exact)
        self.generic_visit(node)

    def _module_rooted(self, node: ast.AST) -> str | None:
        """Return the dotted name of an attribute chain rooted at an import."""
        if not isinstance(node, ast.Attribute):
            return None
        root: ast.AST = node
        while isinstance(root, ast.Attribute):
            root = root.value
        if not isinstance(root, ast.Name) or root.id not in self.aliases:
            return None
        return _dotted_name(node, self.aliases)

    def _check_assign_target(self, target: ast.AST) -> None:
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._check_assign_target(element)
            return
        if isinstance(target, ast.Subscript):
            base = (_dotted_name(target.value, self.aliases) or "").lower()
            if base.endswith("environ"):
                self._error(target, "environment mutation", base)
            return
        name = self._module_rooted(target)
        if name is not None:
            # Rebinding an attribute of an imported module would let a
            # submission replace a primitive the trusted harness itself uses.
            self._error(target, "imported-module mutation", name)

    def _record_alias(self, target: ast.AST, value: ast.AST) -> None:
        """Follow ``f = torch.matmul`` so a later ``f(...)`` still resolves."""
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(
            value, (ast.Tuple, ast.List)
        ):
            if len(target.elts) == len(value.elts):
                for element, item in zip(target.elts, value.elts):
                    self._record_alias(element, item)
            return
        if not isinstance(target, ast.Name):
            return
        if isinstance(value, (ast.Name, ast.Attribute)):
            resolved = _dotted_name(value, self.aliases)
            if resolved and resolved != target.id:
                self.aliases[target.id] = resolved
                return
        self.aliases.pop(target.id, None)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        for target in node.targets:
            self._check_assign_target(target)
            self._record_alias(target, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        self._check_assign_target(node.target)
        if node.value is not None:
            self._record_alias(node.target, node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
        self._check_assign_target(node.target)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:  # noqa: N802
        for target in node.targets:
            self._check_assign_target(target)
        self.generic_visit(node)

    def _check_pass_configs(self, node: ast.AST) -> None:
        """Validate a literal ``pass_configs=`` mapping.

        A non-literal key cannot be checked without executing the submission,
        so it is rejected rather than assumed benign.
        """
        if isinstance(node, ast.Name) and node.id in self.dict_constants:
            node = self.dict_constants[node.id]
        if not isinstance(node, ast.Dict):
            self._error(node, "non-literal pass_configs mapping", "pass_configs")
            return
        for key in node.keys:
            if key is None:
                self._error(node, "pass_configs dictionary unpacking", "**")
                continue
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                token = key.value
            else:
                resolved = _dotted_name(key, self.aliases)
                if resolved is None:
                    self._error(key, "non-literal pass_configs key", "pass_configs")
                    continue
                token = resolved.rsplit(".", 1)[-1]
            if token.lower() in FORBIDDEN_PASS_CONFIG_KEYS:
                self._error(key, "pass configuration", token)

    def _check_keywords(self, node: ast.Call, name: str) -> None:
        """Keyword policy, applied only on the TileLang surface.

        ``target=`` is an ordinary word in ordinary code; it is only a redirect
        of the compiler when it lands on a TileLang entry point, so the check is
        scoped to resolved ``tilelang``/``T`` receivers rather than to any call.
        """
        lowered = name.lower()
        on_tilelang = lowered == "tilelang" or lowered.startswith("tilelang.")
        for keyword in node.keywords:
            if keyword.arg is None:
                if on_tilelang:
                    self._error(node, "keyword unpacking on a TileLang call", f"{name}(**...)")
                continue
            if keyword.arg == "pass_configs":
                self._check_pass_configs(keyword.value)
            elif on_tilelang and keyword.arg.lower() in FORBIDDEN_TILELANG_KEYWORDS:
                self._error(node, "compilation redirect", f"{name}({keyword.arg}=...)")

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        name = _dotted_name(node.func, self.aliases) or ""
        lowered = name.lower()
        tail = lowered.rsplit(".", 1)[-1]
        if lowered in FORBIDDEN_BUILTIN_CALLS:
            self._error(node, "dynamic code or file loading", name)
        if lowered in FORBIDDEN_EXACT_CALLS:
            self._error(node, "compiled or external code loading", name)
        if "." in lowered and tail in FORBIDDEN_GEMM_TAILS:
            self._error(node, "existing GEMM API", name)
        if tail in FORBIDDEN_OPERATOR_TAILS:
            self._error(node, "existing SSM or attention kernel", name)
        if tail in FORBIDDEN_PROCESS_OR_LOAD_TAILS:
            self._error(node, "process or dynamic loading API", name)
        if tail in FORBIDDEN_TILELANG_TAILS:
            self._error(node, "route out of the tile DSL", name)
        if tail.startswith(FORBIDDEN_TILELANG_TAIL_PREFIXES):
            self._error(node, "raw hardware intrinsic", name)
        if tail.startswith("register_") and "postproc" in tail:
            # A generated-code postprocessor is handed the emitted CUDA source
            # and may return anything at all in its place.
            self._error(node, "generated-code postprocessor", name)
        if any(part == "environ" for part in lowered.split(".")):
            self._error(node, "environment mutation", name)
        if lowered in FORBIDDEN_PRIVATE_STREAM_NAMES or tail == "externalstream":
            self._error(node, "private CUDA stream API", name)
        self._check_keywords(node, name)
        self.generic_visit(node)


def _scan_selected_source(text: str, relative_path: str) -> list[str]:
    try:
        tree = ast.parse(text, filename=relative_path)
    except SyntaxError as exc:
        line = exc.lineno or 1
        return [f"{relative_path}:{line}: Python syntax error: {exc.msg}"]
    visitor = _SelectedPolicy(relative_path, tree)
    visitor.visit(tree)
    return visitor.errors


def validate_submission(
    root: Path,
    candidate_version: int | None = None,
    *,
    workspace_mode: bool = False,
) -> dict[str, Any]:
    """Return a deterministic, JSON-serializable admission decision."""
    root = Path(root)
    errors: list[str] = []
    warnings: list[str] = []
    versions: list[int] = []
    sources: dict[int, tuple[str, bytes, str]] = {}
    selected_version: int | None = None
    total_bytes = 0

    if not _require_directory(root, root, errors):
        return {
            "ok": False,
            "final_version": None,
            "candidate_version": candidate_version,
            "num_candidate_versions": 0,
            "versions": [],
            "selected_path": None,
            "selected_relative_path": None,
            "selected_sha256": None,
            "errors": errors,
            "warnings": warnings,
        }

    allowed_root_entries = {"candidates", "FINAL_VERSION", "notes.md"}
    if workspace_mode:
        allowed_root_entries.update({"AGENT_TASK.md", "README_AGENT.md", "build", "tools", ".sforge"})
        # .sforge/ is the root-owned cooperative pre-submit hook of the single-accelerator
        # profile; submit_paths never packs it, so Judge mode stays strict.
    for entry in _list_directory(root, root, errors):
        if entry.name not in allowed_root_entries:
            errors.append(f"unexpected root entry: {entry.name}")

    final_text, final_bytes, final_size = _read_text(
        root / "FINAL_VERSION",
        root,
        errors,
        max_bytes=16,
        require_nonempty=True,
    )
    total_bytes += final_size
    if final_text is not None and final_bytes is not None:
        if FINAL_VERSION_RE.fullmatch(final_bytes) is None:
            errors.append("FINAL_VERSION must contain one canonical integer from 1 to 999")
        else:
            selected_version = int(final_text.rstrip("\n"))

    _, _, notes_size = _read_text(
        root / "notes.md",
        root,
        errors,
        max_bytes=MAX_NOTES_BYTES,
        require_nonempty=False,
    )
    total_bytes += notes_size

    candidates = root / "candidates"
    if _require_directory(candidates, root, errors):
        for candidate_dir in _list_directory(candidates, root, errors):
            match = CANDIDATE_RE.fullmatch(candidate_dir.name)
            if match is None:
                errors.append(f"unexpected candidates entry: {_relative(candidate_dir, root)}")
                continue
            version = int(match.group(1))
            if version == 0:
                errors.append(f"candidate version 000 is forbidden: {_relative(candidate_dir, root)}")
                continue
            if not _require_directory(candidate_dir, root, errors):
                continue
            versions.append(version)
            entries = _list_directory(candidate_dir, root, errors)
            for entry in entries:
                if entry.name != "submission.py":
                    errors.append(f"unexpected candidate entry: {_relative(entry, root)}")
            source_path = candidate_dir / "submission.py"
            source_text, source_bytes, source_size = _read_text(
                source_path,
                root,
                errors,
                max_bytes=MAX_SOURCE_BYTES,
                require_nonempty=False,
            )
            total_bytes += source_size
            if source_text is not None and source_bytes is not None:
                sources[version] = (
                    source_text,
                    source_bytes,
                    _relative(source_path, root),
                )

    versions.sort()
    if not versions:
        errors.append("at least one kernel version is required")
    if len(versions) > MAX_VERSIONS:
        errors.append(f"submission retains {len(versions)} versions; maximum is {MAX_VERSIONS}")
    if selected_version is not None and selected_version not in versions:
        errors.append(f"FINAL_VERSION selects missing kernel_v{selected_version:03d}")
    if total_bytes > MAX_TOTAL_BYTES:
        errors.append(f"submission text exceeds {MAX_TOTAL_BYTES} total bytes")

    requested_version = selected_version
    if candidate_version is not None:
        if (
            isinstance(candidate_version, bool)
            or not isinstance(candidate_version, int)
            or not 1 <= candidate_version <= 999
        ):
            errors.append("candidate_version must be an integer from 1 to 999")
            requested_version = None
        else:
            requested_version = candidate_version
    if requested_version is not None and requested_version not in versions:
        errors.append(f"selected candidate kernel_v{requested_version:03d} is missing")

    selected_path: str | None = None
    selected_relative_path: str | None = None
    selected_sha256: str | None = None
    if requested_version is not None and requested_version in sources:
        selected_text, selected_bytes, selected_relative_path = sources[requested_version]
        selected_path = str(root / selected_relative_path)
        if not selected_bytes:
            errors.append(f"selected source must not be empty: {selected_relative_path}")
        else:
            errors.extend(_scan_selected_source(selected_text, selected_relative_path))
        selected_sha256 = _sha256(selected_bytes)

    if versions:
        missing = sorted(set(range(min(versions), max(versions) + 1)) - set(versions))
        if missing:
            warnings.append(f"candidate version sequence has gaps: {missing}")

    return {
        "ok": not errors,
        "final_version": selected_version,
        "candidate_version": requested_version,
        "num_candidate_versions": len(versions),
        "versions": versions,
        "selected_path": selected_path,
        "selected_relative_path": selected_relative_path,
        "selected_sha256": selected_sha256,
        "errors": errors,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate-version", type=int)
    parser.add_argument("--workspace-mode", action="store_true")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    payload = validate_submission(
        args.root,
        args.candidate_version,
        workspace_mode=args.workspace_mode,
    )
    encoded = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if args.json_output is not None:
        try:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(encoded, encoding="utf-8")
        except OSError as exc:
            print(f"failed to write --json-output: {exc}", file=sys.stderr)
            return 3
    print(encoded, end="")
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
