#!/usr/bin/env python3
"""Fail-closed model-level source admission shared by Work and Judge."""

from __future__ import annotations

import ast
import hashlib
import re
import posixpath
from pathlib import Path
from typing import Any

VERSION_RE = re.compile(r"engine_v([0-9]{3})")
FINAL_RE = re.compile(rb"[1-9][0-9]{0,2}\n?")
ALLOWED_SUFFIXES = {
    ".py",
    ".cu",
    ".cuh",
    ".cpp",
    ".h",
    ".json",
    ".md",
    ".txt",
    ".toml",
}
MAX_FILE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024
FORBIDDEN_IMPORT_ROOTS = {
    "aiohttp",
    "builtins",
    "http",
    "httpx",
    "importlib",
    "inspect",
    "pickle",
    "_pickle",
    "multiprocessing",
    "requests",
    "socket",
    "subprocess",
    "urllib",
}
FORBIDDEN_CALLS = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "os.fork",
    "os.forkpty",
    "os.popen",
    "os.posix_spawn",
    "os.posix_spawnp",
    "os.setpgid",
    "os.setpgrp",
    "os.setsid",
    "os.system",
    "pickle.loads",
    "pickle.load",
    "_pickle.loads",
    "_pickle.load",
    "gc.get_objects",
    "gc.get_referrers",
    "gc.get_referents",
    "sys._getframe",
    "sys._current_frames",
    "torch.load",
    "torch.serialization.load",
}
FORBIDDEN_CALL_PREFIXES = ("os.exec", "os.spawn")
INTROSPECTION_ATTRIBUTES = {
    "__globals__",
    "__closure__",
    "__code__",
    "__subclasses__",
    "gi_frame",
    "cr_frame",
    "f_back",
    "f_locals",
    "f_globals",
}
# Constant-name getattr on these namespaces is ordinary feature detection
# (``getattr(torch.nn.attention, "sdpa_kernel", None)``); every other protected
# root keeps rejecting dynamic lookups.
FEATURE_LOOKUP_ROOTS = ("torch", "triton")
GETATTR_BLOCKED_ATTRIBUTES = INTROSPECTION_ATTRIBUTES | {"__dict__", "__builtins__"}
POLICY_ROOT = "/app/workdir/spec_lab"
# /tmp/ is an admitted write root, so candidates may read back their own files
# there (this also covers /tmp/torch_extensions/). The workspace read root
# already contains build/ and the candidate's own version directory.
SAFE_READ_PREFIXES = ("/app/workdir/spec_lab/", "/opt/models/", "/tmp/")
SAFE_WRITE_PREFIXES = ("/app/workdir/spec_lab/build/", "/tmp/")
PATH_CONSTRUCTORS = {
    "pathlib.Path",
    "pathlib.PosixPath",
    "pathlib.PurePath",
    "pathlib.PurePosixPath",
}
PATH_PASSTHROUGH = {"os.path.abspath", "os.path.realpath", "os.path.normpath", "os.fspath", "str"}


def _dotted(node: ast.AST, aliases: dict[str, str]) -> str | None:
    """Resolve a simple attribute/call route through known aliases."""
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        prefix = _dotted(node.value, aliases)
        return f"{prefix}.{node.attr}" if prefix else None
    if isinstance(node, ast.Call):
        return _dotted(node.func, aliases)
    return None


def _forbidden_call(route: str) -> bool:
    """Return whether a resolved call can launch or deserialize code."""
    return route in FORBIDDEN_CALLS or route.startswith(
        FORBIDDEN_CALL_PREFIXES
    )


def _rooted(route: str, roots: tuple[str, ...]) -> bool:
    """Return whether a dotted route is one of roots or an attribute under one."""
    return any(route == root or route.startswith(root + ".") for root in roots)


def _feature_lookup(base: str, attribute: str | None) -> bool:
    """Admit ``getattr(<torch/triton namespace>, "<constant name>")``.

    The name must be a static constant (feature detection such as
    ``getattr(torch.cuda, "synchronize")``); a name that cannot be resolved,
    an introspection/dictionary attribute, or a route into a forbidden call
    (``getattr(torch, "load")``, ``getattr(torch, "serialization")``) is not.
    """
    if attribute is None or not _rooted(base, FEATURE_LOOKUP_ROOTS):
        return False
    target = f"{base}.{attribute}"
    if attribute in GETATTR_BLOCKED_ATTRIBUTES or _forbidden_call(target):
        return False
    return not any(item.startswith(target + ".") for item in FORBIDDEN_CALLS)


def _template_prefix(template: str, marker: str) -> tuple[str, bool] | None:
    """Static prefix of a ``%``/``str.format`` template; None when it is empty."""
    head = template.split(marker, 1)[0]
    if marker not in template:
        return template, True
    return (head, False) if head else None


def _join_prefixes(
    parts: list[tuple[str, bool] | None],
) -> tuple[str, bool] | None:
    """Join resolved path segments, stopping at the first dynamic one."""
    if not parts or parts[0] is None:
        return None
    prefix, complete = parts[0]
    for part in parts[1:]:
        if not complete:
            break
        if part is None:
            return posixpath.join(prefix, ""), False
        prefix, complete = posixpath.join(prefix, part[0]), part[1]
    return prefix, complete


def _under_roots(target: tuple[str, bool], prefixes: tuple[str, ...]) -> bool:
    """Check a resolved path, or its static prefix, against admitted roots."""
    path, complete = target
    normalized = posixpath.normpath(posixpath.join(POLICY_ROOT, path))
    if complete:
        return any(
            normalized == prefix.rstrip("/") or normalized.startswith(prefix)
            for prefix in prefixes
        )
    # A dynamic tail may extend the last component (``f"/tmp{name}"``), so only
    # a static prefix that already lies strictly inside a root is admitted.
    if path.endswith("/"):
        normalized = posixpath.join(normalized, "")
    return any(normalized.startswith(prefix) for prefix in prefixes)


def _aliases(tree: ast.AST) -> dict[str, str]:
    """Collect import aliases and propagate simple assignment aliases."""
    aliases: dict[str, str] = {}
    assignments: list[tuple[str, ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                root = item.name.split(".", 1)[0]
                aliases[item.asname or root] = item.name if item.asname else root
        elif isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                aliases[item.asname or item.name] = (
                    f"{node.module}.{item.name}"
                )
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name):
                assignments.append((node.targets[0].id, node.value))
        elif isinstance(node, ast.AnnAssign) and isinstance(
            node.target,
            ast.Name,
        ):
            if node.value is not None:
                assignments.append((node.target.id, node.value))
    for _iteration in range(min(len(assignments) + 1, 16)):
        changed = False
        for target, value in assignments:
            route = _dotted(value, aliases)
            if route and aliases.get(target) != route:
                aliases[target] = route
                changed = True
        if not changed:
            break
    return aliases


def _read_regular(path: Path, root: Path, maximum: int) -> bytes:
    """Read one bounded regular file whose resolved path stays under root."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"not a regular file: {path.name}")
    resolved = path.resolve(strict=True)
    if root != resolved and root not in resolved.parents:
        raise ValueError(f"path escapes submission: {path.name}")
    data = path.read_bytes()
    if len(data) > maximum:
        raise ValueError(f"file exceeds {maximum} bytes: {path.name}")
    return data


class Policy(ast.NodeVisitor):
    """Reject source routes that can reach the host control plane."""

    def __init__(self, relative: str) -> None:
        """Create a policy visitor for one displayed relative path."""
        self.relative = relative
        self.errors: list[str] = []
        self.aliases: dict[str, str] = {}
        self.paths: dict[str, tuple[str, bool]] = {}

    def fail(self, node: ast.AST, message: str) -> None:
        """Record a stable line-oriented admission error."""
        self.errors.append(f"{self.relative}:{getattr(node, 'lineno', 1)}: {message}")

    def check_import(self, node, module):
        """Expose only the public ABI and the selected implementation package."""
        if module == "runtime" or (module.startswith("runtime.") and module != "runtime.api"):
            self.fail(node, "trusted-runtime internal import is forbidden; use runtime.api")
        if module.startswith("solutions."):
            selected = self.relative.split("/")[:2]
            package = ".".join(selected) if len(selected) == 2 else ""
            if not package or not (module == package or module.startswith(package + ".")):
                self.fail(node, "import outside selected version is forbidden")

    def path_value(self, node):
        """Resolve a fully literal string through variables, receivers and joins."""
        resolved = self.path_prefix(node)
        return resolved[0] if resolved is not None and resolved[1] else None

    def path_prefix(self, node):
        """Resolve a path expression to its static leading prefix.

        Returns ``(prefix, complete)``; ``complete`` is False when the
        expression has a dynamic tail (an f-string field, an unresolved join
        segment, a ``%``/``format`` argument). ``__file__`` resolves to the
        scanned file's own location under the policy root. A bare unresolved
        variable yields None.
        """
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value, True
        if isinstance(node, ast.Name):
            if node.id == "__file__" and self.aliases.get("__file__", "__file__") == "__file__":
                return posixpath.join(POLICY_ROOT, self.relative), True
            return self.paths.get(node.id)
        if isinstance(node, ast.JoinedStr):
            head = node.values[0] if node.values else None
            if isinstance(head, ast.Constant) and isinstance(head.value, str):
                return head.value, len(node.values) == 1
            return None
        if isinstance(node, ast.Attribute):
            base = self.path_prefix(node.value)
            if base is not None and base[1] and node.attr == "parent":
                return posixpath.dirname(base[0]), True
            return None
        if isinstance(node, ast.Call):
            route = _dotted(node.func, self.aliases) or ""
            method = node.func.attr if isinstance(node.func, ast.Attribute) else None
            if route in PATH_CONSTRUCTORS or route == "os.path.join":
                return _join_prefixes([self.path_prefix(arg) for arg in node.args])
            if method == "joinpath":
                parts = [self.path_prefix(node.func.value), *(self.path_prefix(arg) for arg in node.args)]
                return _join_prefixes(parts)
            if route == "os.path.dirname" and len(node.args) == 1:
                base = self.path_prefix(node.args[0])
                return (posixpath.dirname(base[0]), True) if base is not None and base[1] else None
            if route in PATH_PASSTHROUGH and len(node.args) == 1:
                return self.path_prefix(node.args[0])
            if method in {"resolve", "absolute"} and not node.args:
                return self.path_prefix(node.func.value)
            template = node.func.value if method == "format" else None
            if isinstance(template, ast.Constant) and isinstance(template.value, str):
                return _template_prefix(template.value, "{")
            return None
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Add, ast.Mod)):
            left = self.path_prefix(node.left)
            if left is None:
                return None
            if isinstance(node.op, ast.Mod):
                return _template_prefix(left[0], "%") if left[1] else left
            right = self.path_prefix(node.right)
            if isinstance(node.op, ast.Div):
                return _join_prefixes([left, right])
            if not left[1]:
                return left
            return (left[0], False) if right is None else (left[0] + right[0], right[1])
        return None

    def visit_FunctionDef(self, node):
        """Keep function-local names separate from imported module aliases."""
        aliases, paths = self.aliases.copy(), self.paths.copy()
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            self.aliases[argument.arg] = "<local>"
            self.paths.pop(argument.arg, None)
        self.generic_visit(node)
        self.aliases, self.paths = aliases, paths

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Import(self, node: ast.Import) -> None:
        """Reject imports of networking, process, and introspection modules."""
        for item in node.names:
            root = item.name.split(".", 1)[0]
            self.check_import(node, item.name)
            if root in FORBIDDEN_IMPORT_ROOTS:
                self.fail(node, f"forbidden import {item.name!r}")
            self.aliases[item.asname or root] = item.name if item.asname else root

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Apply the same root policy to from-imports and retain aliases."""
        module = node.module or ""
        root = module.split(".", 1)[0]
        self.check_import(node, module)
        if node.level > 1:
            self.fail(node, "relative import outside selected version is forbidden")
        if not node.level and root in FORBIDDEN_IMPORT_ROOTS:
            self.fail(node, f"forbidden import {module!r}")
        # Explicitly allow the compiler surface; its process use is trusted
        # Torch code rather than candidate-controlled subprocess access.
        for item in node.names:
            self.aliases[item.asname or item.name] = (
                f"{module}.{item.name}" if module else item.name
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Reject dynamic code, process launch, routing, and unsafe paths."""
        route = _dotted(node.func, self.aliases) or ""
        if _forbidden_call(route):
            self.fail(node, f"forbidden call {route!r}")
        if route in {"os.getenv", "os.environ.get", "os.environ.__getitem__"}:
            self.fail(node, "environment-variable routing is forbidden")
        if route == "vars" and node.args:
            base = _dotted(node.args[0], self.aliases) or ""
            if base in {"os", "runtime", "sys", "time", "torch", "gc"} or base.startswith(("os.", "runtime.", "sys.", "time.", "torch.")):
                self.fail(node, "control-plane module dictionary access is forbidden")
        if route in {"getattr", "setattr", "delattr"} and node.args:
            base = _dotted(node.args[0], self.aliases) or ""
            attribute = self.path_value(node.args[1]) if len(node.args) > 1 else None
            if base == "gc" and (attribute is None or attribute == "__dict__" or _forbidden_call(f"gc.{attribute}")):
                self.fail(node, "GC enumeration lookup is forbidden")
            protected = base in {"os", "runtime", "sys", "time", "torch"} or base.startswith(("os.", "runtime.", "sys.", "time.", "torch."))
            if protected and not (route == "getattr" and _feature_lookup(base, attribute)):
                self.fail(node, "dynamic control-plane lookup or mutation is forbidden")
        path_methods = {"open", "read_bytes", "read_text", "write_bytes", "write_text"}
        method = node.func.attr if isinstance(node.func, ast.Attribute) else None
        receiver = self.path_prefix(node.func.value) if method in path_methods else None
        builtin_open = route in {"open", "io.open"}
        path_call = route.startswith("pathlib.") and route.rsplit(".", 1)[-1] in path_methods
        if builtin_open or receiver is not None or path_call:
            kwargs = {item.arg: item.value for item in node.keywords}
            path = self.path_prefix(node.args[0]) if builtin_open and node.args else self.path_prefix(kwargs.get("file")) if builtin_open else receiver
            mode_index = 1 if builtin_open else 0
            mode_node = kwargs.get("mode", node.args[mode_index] if len(node.args) > mode_index else ast.Constant("r"))
            mode = mode_node.value if isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str) else None
            if method in {"write_bytes", "write_text"}:
                mode = "w"
            if path is None or mode is None:
                self.fail(node, "filesystem path or mode cannot be resolved")
            else:
                prefixes = SAFE_WRITE_PREFIXES if any(flag in mode for flag in "wax+") else SAFE_READ_PREFIXES
                if not _under_roots(path, prefixes):
                    self.fail(node, "filesystem path is outside admitted roots")
        if route.startswith("ctypes."):
            for arg in node.args:
                if (
                    isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                    and arg.value.startswith("/")
                    and not arg.value.startswith(
                        ("/app/workdir/", "/tmp/torch_extensions/")
                    )
                ):
                    self.fail(
                        node,
                        (
                            "ctypes may not load outside workspace or Torch "
                            "extension roots"
                        ),
                    )
        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Reject frame/function introspection and trusted module dictionaries."""
        route = _dotted(node, self.aliases) or ""
        if node.attr in INTROSPECTION_ATTRIBUTES:
            self.fail(node, "frame/function introspection is forbidden")
        if node.attr == "__dict__" and route.startswith(("runtime.", "time.", "sys.", "gc.")):
            self.fail(node, "trusted module dictionary access is forbidden")
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Reject direct environment and module-table access."""
        route = _dotted(node.value, self.aliases) or ""
        if route in {"sys.modules", "os.environ"}:
            self.fail(node, f"forbidden control-plane access {route!r}")
        self.generic_visit(node)

    def _assignment(self, target: ast.AST) -> None:
        """Reject assignments into the trusted runtime or module table."""
        if isinstance(target, ast.Name):
            return  # Rebinding a local name does not mutate a module.
        while isinstance(target, ast.Subscript):
            target = target.value
        route = _dotted(target, self.aliases) or ""
        if route.startswith(("runtime.", "sys.modules", "time.", "torch.cuda.")):
            self.fail(target, "trusted-runtime monkey-patching is forbidden")

    def bind_assignment(self, target, value):
        """Carry semantic aliases and literal paths forward in this scope."""
        if isinstance(target, ast.Name):
            path = self.path_prefix(value)
            route = _dotted(value, self.aliases)
            self.aliases[target.id] = route or "<local>"
            if path is None:
                self.paths.pop(target.id, None)
            else:
                self.paths[target.id] = path

    def visit_Assign(self, node: ast.Assign) -> None:
        """Check every ordinary assignment target."""
        for target in node.targets:
            self._assignment(target)
        self.generic_visit(node)
        for target in node.targets:
            self.bind_assignment(target, node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Check an annotated assignment target."""
        self._assignment(node.target)
        self.generic_visit(node)
        self.bind_assignment(node.target, node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Check an augmented assignment target."""
        self._assignment(node.target)
        self.generic_visit(node)


def scan_python(path: Path, relative: str) -> list[str]:
    """Parse and scan one selected Python source file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
    except (UnicodeDecodeError, SyntaxError) as exc:
        return [f"{relative}: invalid Python: {exc}"]
    policy = Policy(relative)
    policy.visit(tree)
    return policy.errors


def validate_submission(
    root: str | Path,
    candidate_version: int | None = None,
    *,
    workspace_mode: bool = False,
) -> dict[str, Any]:
    """Validate and describe the selected candidate source tree."""
    if not isinstance(workspace_mode, bool):
        raise TypeError("workspace_mode must be bool")
    if candidate_version is not None and (
        isinstance(candidate_version, bool)
        or not isinstance(candidate_version, int)
        or not 1 <= candidate_version <= 999
    ):
        raise TypeError("candidate_version must be an integer from 1 to 999")
    root = Path(root)
    errors: list[str] = []
    versions: list[int] = []
    selected = None
    selected_sha = None
    try:
        resolved = root.resolve(strict=True)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("submission root must be a real directory")
        final = _read_regular(root / "FINAL_VERSION", resolved, 16)
        if FINAL_RE.fullmatch(final) is None:
            raise ValueError("FINAL_VERSION must be a canonical integer 1..999")
        selected = int(final)
        _read_regular(root / "notes.md", resolved, MAX_FILE_BYTES)
        solutions = root / "solutions"
        if solutions.is_symlink() or not solutions.is_dir():
            raise ValueError("solutions must be a real directory")
        for entry in sorted(solutions.iterdir()):
            match = VERSION_RE.fullmatch(entry.name)
            if match is None:
                errors.append(f"unexpected solutions entry: {entry.name}")
                continue
            if entry.is_symlink() or not entry.is_dir():
                errors.append(f"version must be a real directory: {entry.name}")
                continue
            versions.append(int(match.group(1)))
        requested = selected if candidate_version is None else int(candidate_version)
        version_dir = solutions / f"engine_v{requested:03d}"
        if requested not in versions:
            errors.append(f"selected engine_v{requested:03d} is missing")
        else:
            digest = hashlib.sha256()
            total = 0
            for path in sorted(version_dir.rglob("*")):
                if path.is_dir() and not path.is_symlink():
                    continue
                relative = path.relative_to(root).as_posix()
                if path.is_symlink() or not path.is_file():
                    errors.append(f"unsafe candidate entry: {relative}")
                    continue
                if path.suffix.lower() not in ALLOWED_SUFFIXES:
                    errors.append(f"non-source candidate file: {relative}")
                    continue
                try:
                    data = _read_regular(path, resolved, MAX_FILE_BYTES)
                except (OSError, ValueError) as exc:
                    errors.append(f"{relative}: {exc}")
                    continue
                total += len(data)
                digest.update(relative.encode() + b"\0" + data)
                if path.suffix == ".py":
                    errors.extend(scan_python(path, relative))
            if total > MAX_TOTAL_BYTES:
                errors.append("selected version exceeds 16 MiB")
            if not (version_dir / "candidate.py").is_file():
                errors.append("selected version requires candidate.py")
            selected_sha = digest.hexdigest()
    except (OSError, ValueError) as exc:
        errors.append(str(exc))
        requested = candidate_version
    return {
        "ok": not errors,
        "final_version": selected,
        "candidate_version": requested,
        "versions": sorted(versions),
        "num_candidate_versions": len(versions),
        "selected_sha256": selected_sha,
        "errors": list(dict.fromkeys(errors)),
        "warnings": [],
        "mode": "workspace" if workspace_mode else "archive",
    }
