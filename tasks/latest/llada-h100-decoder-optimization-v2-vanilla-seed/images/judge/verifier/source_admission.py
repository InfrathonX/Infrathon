"""Fail-closed admission for the narrow Python/Triton submission surface."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any


VERSION_RE = re.compile(r"decoder_v([0-9]{3})")
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_NOTES_BYTES = 128 * 1024
ALLOWED_SOLUTION_FILES = {"__init__.py", "candidate.py"}
ALLOWED_IMPORT_ROOTS = {
    "__future__", "collections", "contextlib", "dataclasses", "functools",
    "itertools", "math", "runtime", "torch", "triton", "typing",
}
FORBIDDEN_CALLS = {
    "__import__", "breakpoint", "compile", "delattr", "dir", "eval", "exec",
    "getattr", "globals", "input", "locals", "open", "setattr", "vars",
}
SAFE_GETATTR_ATTRIBUTES = {"mlsysforge_allow_cpu"}
FORBIDDEN_ATTRIBUTES = {
    "__builtins__", "__code__", "__dict__", "__globals__", "__subclasses__",
    "_forward_hooks", "f_back", "f_builtins", "f_code", "f_globals", "f_locals",
    "load_inline", "load_library", "register_forward_hook", "tb_frame", "tb_next",
    "cpp_extension", "from_file", "load", "save",
}
FORBIDDEN_PATH_PREFIXES = {"/logs/", "/opt/"}


class AdmissionError(ValueError):
    pass


def locate_lab(app_dir: Path) -> Path:
    candidates = (
        app_dir / "workdir/decoder_lab",
        app_dir / "environment/workdir/decoder_lab",
        app_dir / "app/workdir/decoder_lab",
        app_dir,
    )
    for candidate in candidates:
        if (candidate / "solutions").is_dir() and (candidate / "FINAL_VERSION").is_file():
            return candidate
    raise AdmissionError(f"cannot locate decoder_lab below {app_dir}")


def _python_policy(
    path: Path,
    source: str,
    *,
    allow_model_forward_assignment: bool = False,
) -> None:
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise AdmissionError(f"invalid Python source: {path.name}") from exc
    # Record only the documented constant-string feature probes so the generic
    # forbidden-name pass can continue rejecting aliases, dynamic names, and
    # every other reflective use.
    allowed_getattr_names: set[int] = set()
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
        ):
            continue
        safe = (
            len(node.args) == 3
            and not node.keywords
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
            and node.args[1].value in SAFE_GETATTR_ATTRIBUTES
        )
        if not safe:
            raise AdmissionError(f"forbidden call 'getattr' in {path.name}")
        allowed_getattr_names.add(id(node.func))

    for node in ast.walk(tree):
        roots: list[str] = []
        if isinstance(node, ast.Import):
            roots = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots = [node.module.split(".")[0]]
        for root in roots:
            if root not in ALLOWED_IMPORT_ROOTS:
                raise AdmissionError(f"import {root!r} is outside the candidate allowlist in {path.name}")
        if isinstance(node, ast.Import) and any(alias.name == "runtime" for alias in node.names):
            raise AdmissionError(f"candidate may import only runtime.api in {path.name}")
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] == "runtime" and node.module != "runtime.api":
                raise AdmissionError(f"candidate may import only runtime.api in {path.name}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS and id(node.func) not in allowed_getattr_names:
                raise AdmissionError(f"forbidden call {node.func.id!r} in {path.name}")
        if (
            isinstance(node, ast.Name)
            and node.id in FORBIDDEN_CALLS
            and id(node) not in allowed_getattr_names
        ):
            raise AdmissionError(f"forbidden name {node.id!r} in {path.name}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in FORBIDDEN_ATTRIBUTES:
                raise AdmissionError(f"forbidden attribute {node.attr!r} in {path.name}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if any(node.value.startswith(prefix) for prefix in FORBIDDEN_PATH_PREFIXES):
                raise AdmissionError(f"forbidden service path literal in {path.name}")
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(
                isinstance(target, ast.Attribute)
                and (
                    target.attr == "_forward_hooks"
                    or (
                        target.attr == "forward"
                        and not allow_model_forward_assignment
                    )
                )
                for target in targets
            ):
                raise AdmissionError(f"candidate may not replace model audit attributes in {path.name}")


def admit(
    lab_root: Path,
    selected_version_override: int | None = None,
) -> dict[str, Any]:
    final_path = lab_root / "FINAL_VERSION"
    if final_path.is_symlink() or not final_path.is_file():
        raise AdmissionError("FINAL_VERSION must be a regular file")
    final_text = final_path.read_text(encoding="utf-8").strip()
    if re.fullmatch(r"[1-9][0-9]{0,2}", final_text) is None:
        raise AdmissionError("FINAL_VERSION must be one integer from 1 through 999")
    final_version = int(final_text)
    if selected_version_override is not None:
        if (
            not isinstance(selected_version_override, int)
            or isinstance(selected_version_override, bool)
            or not 1 <= selected_version_override <= 999
        ):
            raise AdmissionError("selected version override must be from 1 through 999")
        final_version = selected_version_override
    notes_path = lab_root / "notes.md"
    if notes_path.is_symlink() or not notes_path.is_file():
        raise AdmissionError("notes.md must be a regular file")
    if notes_path.stat().st_size > MAX_NOTES_BYTES:
        raise AdmissionError("notes.md exceeds 128 KiB")
    try:
        notes_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise AdmissionError("notes.md must be readable UTF-8 text") from exc

    solutions = lab_root / "solutions"
    if solutions.is_symlink() or not solutions.is_dir():
        raise AdmissionError("solutions must be a regular directory")
    versions: list[int] = []
    total_bytes = final_path.stat().st_size + notes_path.stat().st_size
    file_count = 2
    for path in sorted(solutions.rglob("*")):
        if path.is_symlink():
            raise AdmissionError(f"submission contains symlink: {path.relative_to(lab_root)}")
        if path.is_dir():
            continue
        if not path.is_file() or path.name not in ALLOWED_SOLUTION_FILES:
            raise AdmissionError(f"unsupported submission file: {path.relative_to(lab_root)}")
        size = path.stat().st_size
        total_bytes += size
        file_count += 1
        if total_bytes > MAX_SOURCE_BYTES:
            raise AdmissionError("submission source exceeds 2 MiB")
        version_match = VERSION_RE.fullmatch(path.parent.name)
        if version_match is None:
            raise AdmissionError(f"invalid version directory: {path.parent.name}")
        is_selected = int(version_match.group(1)) == final_version
        _python_policy(
            path,
            path.read_text(encoding="utf-8"),
            # Retained experiments are never imported or executed by the
            # loader.  Permit a rejected compile experiment to keep its own
            # top-level forward wrapper while preserving the audit-mutation
            # ban for both files in the selected package.
            allow_model_forward_assignment=not is_selected,
        )
    for directory in solutions.iterdir():
        if not directory.is_dir():
            raise AdmissionError("solutions may contain only version directories")
        match = VERSION_RE.fullmatch(directory.name)
        if match is None:
            raise AdmissionError(f"invalid version directory: {directory.name}")
        version = int(match.group(1))
        versions.append(version)
        candidate = directory / "candidate.py"
        if not candidate.is_file() or candidate.is_symlink():
            raise AdmissionError(f"version {version} has no regular candidate.py")
    if final_version not in versions:
        raise AdmissionError("FINAL_VERSION does not select a retained version")
    return {
        "final_version": final_version,
        "versions": sorted(versions),
        "file_count": file_count,
        "source_bytes": total_bytes,
    }
