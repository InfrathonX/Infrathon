#!/usr/bin/env python3
"""Judge-only static admission checks for clean-room Python/Triton candidates.

This is compatibility hardening, not a Python sandbox. The hidden verifier
repeats the same scan before importing any candidate.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


VERSION_RE = re.compile(r"^flash_kmeans_v([0-9]{3})\.py$")
ALLOWED_IMPORT_ROOTS = {"__future__", "functools", "math", "torch", "triton", "typing"}
FORBIDDEN_IMPORT_ROOTS = {
    "builtins",
    "cffi",
    "ctypes",
    "cuml",
    "faiss",
    "fast_pytorch_kmeans",
    "fastkmeans",
    "flash_kmeans",
    "gc",
    "importlib",
    "inspect",
    "io",
    "joblib",
    "marshal",
    "multiprocessing",
    "numpy",
    "os",
    "pathlib",
    "pickle",
    "requests",
    "resource",
    "shelve",
    "sklearn",
    "socket",
    "subprocess",
    "sys",
    "tempfile",
    "threading",
    "time",
    "urllib",
}
FORBIDDEN_CALL_NAMES = {
    "__import__",
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
    "print",
    "quit",
    "setattr",
    "vars",
}
FORBIDDEN_ATTRIBUTE_SUFFIXES = {
    "add_dll_directory",
    "chmod",
    "compile",
    "connect",
    "dump",
    "dumps",
    "from_file",
    "frombuffer",
    "load",
    "load_library",
    "loads",
    "open",
    "read",
    "read_bytes",
    "read_text",
    "recv",
    "remove",
    "rename",
    "replace",
    "reset_accumulated_memory_stats",
    "reset_max_memory_allocated",
    "reset_max_memory_cached",
    "reset_peak_host_memory_stats",
    "reset_peak_memory_stats",
    "rmdir",
    "save",
    "send",
    "manual_seed",
    "set_default_device",
    "set_default_dtype",
    "set_float32_matmul_precision",
    "set_num_interop_threads",
    "set_num_threads",
    "use_deterministic_algorithms",
    "setdefault",
    "set_stream",
    "stream",
    "externalstream",
    "event",
    "cudagraph",
    "symlink_to",
    "to_file",
    "unlink",
    "update_wrapper",
    "write",
    "write_bytes",
    "write_text",
}
FORBIDDEN_ATTRIBUTE_FRAGMENTS = {
    "_c._dl",
    "cpp_extension",
    "dlopen",
    "distributed",
    "library",
    "ops.load",
    "utils.data",
}
FORBIDDEN_GLOBAL_SETTERS = {
    "torch.manual_seed",
    "torch.set_default_device",
    "torch.set_default_dtype",
    "torch.set_float32_matmul_precision",
    "torch.set_num_interop_threads",
    "torch.set_num_threads",
    "torch.use_deterministic_algorithms",
}
FORBIDDEN_BRIDGE_SEGMENTS = {
    "_c",
    "builtins",
    "cffi",
    "ctypes",
    "gc",
    "importlib",
    "inspect",
    "io",
    "marshal",
    "modules",
    "os",
    "pathlib",
    "pickle",
    "socket",
    "subprocess",
    "sys",
}
RESULT_MARKERS = {
    "FLASH_KMEANS_RESULT",
    "Start Structured Result",
    "End Structured Result",
    "reward.json",
    "flash_kmeans_results.json",
}
MAX_SOURCE_BYTES = 2_000_000


def _attribute_name(node: ast.AST) -> str:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return ".".join(reversed(parts))


def _problem(path: Path, root: Path, node: ast.AST | None, message: str) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "line": int(getattr(node, "lineno", 1)),
        "message": message,
    }


def discover_candidates(root: Path) -> tuple[dict[int, Path], list[dict[str, Any]]]:
    kernels = root / "kernels"
    candidates: dict[int, Path] = {}
    problems: list[dict[str, Any]] = []
    if not kernels.is_dir() or kernels.is_symlink():
        return candidates, [
            {"path": "kernels", "line": 1, "message": "missing kernels directory"}
        ]
    for path in sorted(kernels.rglob("*")):
        if path.is_symlink():
            problems.append(_problem(path, root, None, "symlinks are not admitted"))
            continue
        if path.is_dir():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.parent != kernels or not VERSION_RE.fullmatch(path.name):
            problems.append(
                _problem(
                    path,
                    root,
                    None,
                    "kernel files must be named kernels/flash_kmeans_vNNN.py",
                )
            )
            continue
        version = int(VERSION_RE.fullmatch(path.name).group(1))
        if version in candidates:
            problems.append(_problem(path, root, None, "duplicate candidate version"))
        candidates[version] = path
    return candidates, problems


def selected_version(root: Path) -> int | None:
    path = root / "FINAL_VERSION"
    if not path.is_file() or path.is_symlink():
        return None
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    return int(text) if re.fullmatch(r"[0-9]{1,3}", text) else None


def scan_candidate(path: Path, root: Path) -> list[dict[str, Any]]:
    problems: list[dict[str, Any]] = []
    try:
        if path.stat().st_size > MAX_SOURCE_BYTES:
            return [_problem(path, root, None, "candidate source is too large")]
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        return [_problem(path, root, exc if isinstance(exc, SyntaxError) else None, "invalid Python source")]

    for marker in RESULT_MARKERS:
        if marker.lower() in source.lower():
            problems.append(_problem(path, root, None, "result-marker/output forgery text is prohibited"))
            break

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_name = alias.name.split(".", 1)[0]
                if root_name in FORBIDDEN_IMPORT_ROOTS or root_name not in ALLOWED_IMPORT_ROOTS:
                    problems.append(_problem(path, root, node, f"prohibited import: {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root_name = module.split(".", 1)[0]
            lowered_module = module.lower()
            if (
                node.level
                or root_name in FORBIDDEN_IMPORT_ROOTS
                or root_name not in ALLOWED_IMPORT_ROOTS
                or any(fragment in lowered_module for fragment in FORBIDDEN_ATTRIBUTE_FRAGMENTS)
            ):
                problems.append(_problem(path, root, node, f"prohibited import: {module or '<relative>'}"))
            for alias in node.names:
                imported = alias.name.lower()
                if (
                    imported in FORBIDDEN_ATTRIBUTE_SUFFIXES
                    or imported in {name.rsplit(".", 1)[-1] for name in FORBIDDEN_GLOBAL_SETTERS}
                    or imported in FORBIDDEN_BRIDGE_SEGMENTS
                    or imported in FORBIDDEN_CALL_NAMES
                    or imported.startswith("__")
                ):
                    problems.append(
                        _problem(
                            path,
                            root,
                            node,
                            f"prohibited imported symbol: {alias.name}",
                        )
                    )
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALL_NAMES:
                problems.append(_problem(path, root, node, f"prohibited call: {node.func.id}"))
            elif isinstance(node.func, ast.Attribute):
                dotted = _attribute_name(node.func)
                lowered = dotted.lower()
                segments = set(lowered.split("."))
                triton_primitive = (
                    node.func.attr.lower() in {"load", "store"}
                    and (
                        lowered.startswith("tl.")
                        or lowered.startswith("triton.language.")
                    )
                )
                if (
                    (
                        node.func.attr.lower() in FORBIDDEN_ATTRIBUTE_SUFFIXES
                        and not triton_primitive
                    )
                    or node.func.attr.lower() in FORBIDDEN_CALL_NAMES
                    or bool(segments & FORBIDDEN_BRIDGE_SEGMENTS)
                    or any(fragment in lowered for fragment in FORBIDDEN_ATTRIBUTE_FRAGMENTS)
                    or lowered in FORBIDDEN_GLOBAL_SETTERS
                    or (
                        lowered.startswith("torch.set_")
                        and not lowered.endswith(".set_")
                    )
                ):
                    problems.append(_problem(path, root, node, f"prohibited call: {dotted}"))
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)):
            targets: list[ast.AST] = []
            if isinstance(node, ast.Assign):
                targets.extend(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets.append(node.target)
            elif isinstance(node, ast.AugAssign):
                targets.append(node.target)
            else:
                targets.extend(node.targets)
            if any(isinstance(target, ast.Attribute) for target in targets):
                problems.append(
                    _problem(
                        path,
                        root,
                        node,
                        "attribute assignment is prohibited (monkeypatch hardening)",
                    )
                )
        elif isinstance(node, ast.Attribute):
            dotted = _attribute_name(node).lower()
            triton_primitive = (
                node.attr.lower() in {"load", "store"}
                and (
                    dotted.startswith("tl.")
                    or dotted.startswith("triton.language.")
                )
            )
            if (
                node.attr.startswith("__")
                or node.attr.lower() in FORBIDDEN_CALL_NAMES
                or bool(set(dotted.split(".")) & FORBIDDEN_BRIDGE_SEGMENTS)
                or (
                    node.attr.lower() in FORBIDDEN_ATTRIBUTE_SUFFIXES
                    and not triton_primitive
                )
                or any(
                    fragment in dotted
                    for fragment in FORBIDDEN_ATTRIBUTE_FRAGMENTS
                )
                or dotted in FORBIDDEN_GLOBAL_SETTERS
            ):
                problems.append(
                    _problem(
                        path,
                        root,
                        node,
                        "prohibited attribute/module-bridge reference",
                    )
                )
        elif isinstance(node, ast.Name) and (
            node.id in FORBIDDEN_CALL_NAMES
            or node.id == "__builtins__"
            or node.id.startswith("__")
        ):
            problems.append(_problem(path, root, node, f"prohibited builtin/dunder reference: {node.id}"))

    functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for required in ("assign", "update"):
        if required not in functions:
            problems.append(_problem(path, root, None, f"missing required function: {required}"))
    return problems


def scan_submission(root: Path) -> dict[str, Any]:
    root = root.resolve()
    candidates, problems = discover_candidates(root)
    if not candidates:
        problems.append(
            {
                "path": "kernels",
                "line": 1,
                "message": (
                    "no candidate implementation found; create "
                    "kernels/flash_kmeans_v001.py"
                ),
            }
        )
    version = selected_version(root)
    if version is None:
        problems.append(
            {"path": "FINAL_VERSION", "line": 1, "message": "FINAL_VERSION must contain one integer"}
        )
    elif version not in candidates:
        problems.append(
            {
                "path": "FINAL_VERSION",
                "line": 1,
                "message": "FINAL_VERSION does not select an admitted candidate",
            }
        )
    for path in candidates.values():
        problems.extend(scan_candidate(path, root))
    return {
        "schema_version": 1,
        "status": "clean" if not problems else "rejected",
        "selected_version": version,
        "candidate_versions": sorted(candidates),
        "problems": problems,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    result = scan_submission(args.root)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if result["status"] == "clean":
        print(
            "Source admission: clean; selected "
            f"v{int(result['selected_version']):03d}."
        )
        return 0
    print("SOURCE ADMISSION REJECTED")
    for problem in result["problems"]:
        print(f"  {problem['path']}:{problem['line']}: {problem['message']}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
