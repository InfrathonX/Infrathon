#!/usr/bin/env python3
"""Structural and selected-source admission for the standalone K-grouped wgrad v2 task."""

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

# These checks express the short, public policy in AGENT_TASK.md.  They are a
# diagnostic boundary for normal code-producing agents, not a Python sandbox;
# the runtime's harness-integrity sentinel and the LLM integrity review are
# separate layers.  The rules are an allowlist (the same rules as the dynamic
# grouped FP8 v3 task): a selected submission may
# import only the modules below, may not patch anything it imported, and may
# not reach introspection, dynamic code, file, process, stream or global
# configuration machinery.
ALLOWED_IMPORT_ROOTS = frozenset(
    {
        "__future__",
        "collections",
        "dataclasses",
        "enum",
        "functools",
        "itertools",
        "math",
        "torch",
        "triton",
        "typing",
    }
)
# Sub-namespaces of allowed packages that are not part of the implementation
# surface: compiled-extension, operator-registry, compiler-internal, graph,
# distributed, serialization and global-configuration machinery.
FORBIDDEN_NAMESPACE_PREFIXES = (
    "torch._",
    "torch.backends",
    "torch.compiler",
    "torch.cuda.graphs",
    "torch.cuda.jiterator",
    "torch.cuda.memory",
    "torch.distributed",
    "torch.export",
    "torch.fx",
    "torch.hub",
    "torch.jit",
    "torch.library",
    "torch.multiprocessing",
    "torch.onnx",
    "torch.ops",
    "torch.overrides",
    "torch.package",
    "torch.random",
    "torch.utils",
    "triton.backends",
    "triton.compiler",
    "triton.knobs",
    "triton.tools",
)
FORBIDDEN_GEMM_TAILS = {
    "_scaled_mm",
    "addmm",
    "addbmm",
    "baddbmm",
    "bmm",
    "einsum",
    "linear",
    "matmul",
    "mm",
    "mv",
    "tensordot",
}
FORBIDDEN_NAMES = {
    "__builtins__",
    "__import__",
    "__loader__",
    "__spec__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "exit",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "quit",
    "setattr",
    "vars",
}
ALLOWED_DUNDER_ATTRIBUTES = {"__init__", "__name__"}
ALLOWED_DUNDER_NAMES = {"__name__"}
FORBIDDEN_EXACT_CALLS = {
    "torch.compile",
    "torch.cuda.cudagraph",
    "torch.cuda.graph",
    "torch.cuda.make_graphed_callables",
    "torch.cuda.manual_seed",
    "torch.cuda.manual_seed_all",
    "torch.cuda.seed",
    "torch.cuda.seed_all",
    "torch.cuda.set_device",
    "torch.cuda.set_per_process_memory_fraction",
    "torch.cuda.set_rng_state",
    "torch.cuda.set_rng_state_all",
    "torch.cuda.set_sync_debug_mode",
    "torch.load",
    "torch.manual_seed",
    "torch.save",
    "torch.seed",
    "torch.use_deterministic_algorithms",
    "triton.compile",
}
FORBIDDEN_CALL_PREFIXES = ("torch.set_",)
FORBIDDEN_PROCESS_OR_LOAD_TAILS = {
    "cdll",
    "dlopen",
    "execv",
    "execve",
    "fork",
    "forkpty",
    "load_binary",
    "load_inline",
    "load_library",
    "loadlibrary",
    "popen",
    "pydll",
    "spawnl",
    "spawnle",
    "spawnlp",
    "spawnlpe",
    "spawnv",
    "spawnve",
    "spawnvp",
    "spawnvpe",
    "system",
}
FORBIDDEN_PRIVATE_STREAM_NAMES = {
    "torch.cuda.externalstream",
    "torch.cuda.set_stream",
    "torch.cuda.stream",
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


def _forbidden_namespace(name: str) -> str | None:
    lowered = name.lower()
    for prefix in FORBIDDEN_NAMESPACE_PREFIXES:
        if lowered == prefix or lowered.startswith(prefix if prefix.endswith("_") else prefix + "."):
            return prefix
    return None


def _import_allowed(module: str) -> bool:
    root = module.split(".", 1)[0]
    return root in ALLOWED_IMPORT_ROOTS and _forbidden_namespace(module) is None


def _target_root(node: ast.AST) -> ast.AST:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node


class _SelectedPolicy(ast.NodeVisitor):
    def __init__(self, relative_path: str) -> None:
        self.relative_path = relative_path
        self.aliases: dict[str, str] = {}
        self.imported: set[str] = set()
        self.errors: list[str] = []

    def _error(self, node: ast.AST, category: str, token: str) -> None:
        line = getattr(node, "lineno", 1)
        self.errors.append(
            f"{self.relative_path}:{line}: forbidden {category}: {token!r}"
        )

    # -- imports ---------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for item in node.names:
            local = item.asname or item.name.split(".", 1)[0]
            self.aliases[local] = item.name if item.asname else local
            self.imported.add(local)
            if not _import_allowed(item.name):
                self._error(node, "import", item.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        module = node.module or ""
        if node.level:
            self._error(node, "relative import", "." * node.level + module)
        elif not _import_allowed(module):
            self._error(node, "import", module)
        for item in node.names:
            if item.name == "*":
                self._error(node, "star import", module)
                continue
            full = f"{module}.{item.name}" if module else item.name
            if module != "__future__" and _forbidden_namespace(full) is not None:
                self._error(node, "import", full)
            local = item.asname or item.name
            self.aliases[local] = full
            self.imported.add(local)
        self.generic_visit(node)

    # -- names and attributes ------------------------------------------
    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if node.id in FORBIDDEN_NAMES:
            self._error(node, "name", node.id)
        elif (
            node.id.startswith("__")
            and node.id.endswith("__")
            and node.id not in ALLOWED_DUNDER_NAMES
        ):
            self._error(node, "dunder name", node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        if (
            node.attr.startswith("__")
            and node.attr.endswith("__")
            and node.attr not in ALLOWED_DUNDER_ATTRIBUTES
        ):
            self._error(node, "dunder attribute", node.attr)
        name = _dotted_name(node, self.aliases) or ""
        root = name.split(".", 1)[0]
        if root in ALLOWED_IMPORT_ROOTS and isinstance(_target_root(node), ast.Name):
            prefix = _forbidden_namespace(name)
            if prefix is not None:
                self._error(node, "namespace", name)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:  # noqa: N802
        if isinstance(node.op, ast.MatMult):
            self._error(node, "existing GEMM operator", "@")
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
        if isinstance(node.op, ast.MatMult):
            self._error(node, "existing GEMM operator", "@=")
        self._check_target(node.target)
        self.generic_visit(node)

    # -- assignment targets --------------------------------------------
    def _check_target(self, target: ast.AST) -> None:
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._check_target(element)
            return
        if isinstance(target, ast.Starred):
            self._check_target(target.value)
            return
        if not isinstance(target, (ast.Attribute, ast.Subscript)):
            return
        root = _target_root(target)
        if not isinstance(root, ast.Name):
            self._error(target, "assignment to a computed object", ast.unparse(target)[:80])
        elif root.id in self.imported:
            self._error(target, "assignment to an imported module or object", ast.unparse(target)[:80])

    def _aliases_import(self, value: ast.AST | None) -> bool:
        """True when ``value`` hands out an imported object rather than a new one."""
        if value is None:
            return False
        if isinstance(value, (ast.Name, ast.Attribute)):
            root = _target_root(value)
            return isinstance(root, ast.Name) and root.id in self.imported
        if isinstance(value, (ast.Tuple, ast.List, ast.Set)):
            return any(self._aliases_import(element) for element in value.elts)
        if isinstance(value, ast.Dict):
            return any(self._aliases_import(element) for element in value.values)
        if isinstance(value, ast.IfExp):
            return self._aliases_import(value.body) or self._aliases_import(value.orelse)
        if isinstance(value, ast.BoolOp):
            return any(self._aliases_import(element) for element in value.values)
        if isinstance(value, ast.NamedExpr):
            return self._aliases_import(value.value)
        return False

    def _taint(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self.imported.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._taint(element)
        elif isinstance(target, ast.Starred):
            self._taint(target.value)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        for target in node.targets:
            self._check_target(target)
            if self._aliases_import(node.value):
                self._taint(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        self._check_target(node.target)
        if self._aliases_import(node.value):
            self._taint(node.target)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:  # noqa: N802
        if self._aliases_import(node.value):
            self._taint(node.target)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:  # noqa: N802
        for target in node.targets:
            self._check_target(target)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        self._check_target(node.target)
        self.generic_visit(node)

    visit_AsyncFor = visit_For  # noqa: N815

    def visit_With(self, node: ast.With) -> None:  # noqa: N802
        for item in node.items:
            if item.optional_vars is not None:
                self._check_target(item.optional_vars)
        self.generic_visit(node)

    visit_AsyncWith = visit_With  # noqa: N815

    def visit_comprehension(self, node: ast.comprehension) -> None:  # noqa: N802
        self._check_target(node.target)
        self.generic_visit(node)

    # -- calls -----------------------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        name = _dotted_name(node.func, self.aliases) or ""
        lowered = name.lower()
        tail = lowered.rsplit(".", 1)[-1]
        if lowered in FORBIDDEN_EXACT_CALLS or any(
            lowered.startswith(prefix) for prefix in FORBIDDEN_CALL_PREFIXES
        ):
            self._error(node, "global state, graph, RNG, serialization or compile API", name)
        if (
            "." in lowered
            and tail in FORBIDDEN_GEMM_TAILS
            and lowered != "tl.dot"
            and not lowered.endswith(".tl.dot")
        ):
            self._error(node, "existing GEMM API", name)
        if tail in FORBIDDEN_PROCESS_OR_LOAD_TAILS:
            self._error(node, "process or dynamic loading API", name)
        if lowered in FORBIDDEN_PRIVATE_STREAM_NAMES or tail == "externalstream":
            self._error(node, "private CUDA stream API", name)
        self.generic_visit(node)


def _scan_selected_source(text: str, relative_path: str) -> list[str]:
    try:
        tree = ast.parse(text, filename=relative_path)
    except SyntaxError as exc:
        line = exc.lineno or 1
        return [f"{relative_path}:{line}: Python syntax error: {exc.msg}"]
    visitor = _SelectedPolicy(relative_path)
    visitor.visit(tree)
    # A forbidden attribute chain is reported once, at its outermost node.
    return list(dict.fromkeys(visitor.errors))


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
