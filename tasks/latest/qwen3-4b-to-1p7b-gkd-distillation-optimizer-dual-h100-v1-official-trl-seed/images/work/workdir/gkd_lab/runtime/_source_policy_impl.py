"""Canonical fail-closed admission for the Python/Triton candidate surface."""

from __future__ import annotations

import ast
import re
from pathlib import Path


VERSION_RE = re.compile(r"objective_v([0-9]{3})")
MAX_SOURCE_BYTES = 2 * 1024 * 1024
MAX_NOTES_BYTES = 128 * 1024
ALLOWED_SOLUTION_FILES = {"__init__.py", "candidate.py"}
ALLOWED_IMPORT_ROOTS = {
    "__future__",
    "collections",
    "contextlib",
    "dataclasses",
    "functools",
    "itertools",
    "math",
    "runtime",
    "torch",
    "triton",
    "typing",
}
FORBIDDEN_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}
FORBIDDEN_ATTRIBUTES = {
    "__builtins__",
    "__code__",
    "__dict__",
    "__globals__",
    "__subclasses__",
    "f_back",
    "f_builtins",
    "f_code",
    "f_globals",
    "f_locals",
    "from_file",
    "load_inline",
    "load_library",
    "register_forward_hook",
    "save",
    "tb_frame",
    "tb_next",
}
SAFE_DUNDER_ATTRIBUTES = {"__init__"}
ALLOWED_ATTRIBUTE_WRITE_RECEIVERS = {"self", "ctx"}
FORBIDDEN_PARTICIPANT_EXPORTS = {"optimize_model"}
FORBIDDEN_DYNAMIC_MODULE_EXPORTS = {"__getattr__"}

# Candidate objectives never own distributed orchestration or profilers.  Ban
# both direct imports and qualified access through ``torch`` aliases.
FORBIDDEN_IMPORT_PREFIXES = {
    "time",
    "torch._C",
    "torch.utils.cpp_extension",
    "torch.distributed",
    "torch.profiler",
    "torch.utils.benchmark",
    "torch.autograd.profiler",
}
FORBIDDEN_QUALIFIED_PREFIXES = {
    "torch._C",
    "torch.utils.cpp_extension",
    "torch.distributed",
    "torch.profiler",
    "torch.utils.benchmark",
    "torch.autograd.profiler",
}

# This is deliberately a control-plane list, not a ban on CUDA tensor math,
# torch.compile, AMP, custom autograd, or Triton kernels.
FORBIDDEN_CUDA_CONTROL_MEMBERS = {
    "CUDAGraph",
    "Event",
    "ExternalStream",
    "Stream",
    "caching_allocator_alloc",
    "caching_allocator_delete",
    "change_current_allocator",
    "current_device",
    "current_stream",
    "default_stream",
    "device",
    "elapsed_time",
    "empty_cache",
    "get_allocator_backend",
    "graph",
    "graph_pool_handle",
    "ipc_collect",
    "make_graphed_callables",
    "max_memory_allocated",
    "max_memory_cached",
    "max_memory_reserved",
    "mem_get_info",
    "memory_allocated",
    "memory_cached",
    "memory_reserved",
    "memory_snapshot",
    "memory_stats",
    "memory_stats_as_nested_dict",
    "memory_summary",
    "reset_accumulated_host_memory_stats",
    "reset_accumulated_memory_stats",
    "reset_max_memory_allocated",
    "reset_max_memory_cached",
    "reset_peak_host_memory_stats",
    "reset_peak_memory_stats",
    "set_device",
    "set_per_process_memory_fraction",
    "set_stream",
    "stream",
    "synchronize",
}
FORBIDDEN_CUDA_CONTROL_NAMESPACES = {"graphs", "memory", "profiler"}


class AdmissionError(ValueError):
    pass


def locate_lab(app_dir: Path) -> Path:
    candidates = (
        app_dir / "workdir/gkd_lab",
        app_dir / "environment/workdir/gkd_lab",
        app_dir / "app/workdir/gkd_lab",
        app_dir,
    )
    for candidate in candidates:
        if (candidate / "solutions").is_dir() and (candidate / "FINAL_VERSION").is_file():
            return candidate
    raise AdmissionError(f"cannot locate gkd_lab below {app_dir}")


def _module_is_forbidden(name: str) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_IMPORT_PREFIXES)


def _import_aliases(tree: ast.AST) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.asname:
                    aliases[imported.asname] = imported.name
                else:
                    root = imported.name.split(".", 1)[0]
                    aliases[root] = root
        elif isinstance(node, ast.ImportFrom) and node.module:
            for imported in node.names:
                if imported.name == "*":
                    continue
                aliases[imported.asname or imported.name] = f"{node.module}.{imported.name}"
    return aliases


def _qualified_name(node: ast.AST, aliases: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = _qualified_name(node.value, aliases)
        return None if base is None else f"{base}.{node.attr}"
    return None


def _propagate_torch_aliases(tree: ast.AST, aliases: dict[str, str]) -> None:
    assignments: list[tuple[str, ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                assignments.append((target.id, node.value))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            assignments.append((node.target.id, node.value))
    for _ in range(len(assignments) + 1):
        changed = False
        for target, value in assignments:
            qualified = _qualified_name(value, aliases)
            if qualified and (qualified == "torch" or qualified.startswith("torch.")):
                if aliases.get(target) != qualified:
                    aliases[target] = qualified
                    changed = True
        if not changed:
            break


def _is_forbidden_qualified(name: str) -> bool:
    if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_QUALIFIED_PREFIXES):
        return True
    cuda_prefix = "torch.cuda."
    if not name.startswith(cuda_prefix):
        return False
    components = name[len(cuda_prefix) :].split(".")
    return bool(
        components
        and (
            components[0] in FORBIDDEN_CUDA_CONTROL_NAMESPACES
            or any(component in FORBIDDEN_CUDA_CONTROL_MEMBERS for component in components)
        )
    )


def _direct_candidate_state_write(node: ast.Attribute) -> bool:
    return isinstance(node.value, ast.Name) and node.value.id in ALLOWED_ATTRIBUTE_WRITE_RECEIVERS


def _reject_imperative_model_hook(tree: ast.Module) -> None:
    """Reserve the legacy host-only ``optimize_model`` name from participants."""

    forbidden_names = FORBIDDEN_PARTICIPANT_EXPORTS | FORBIDDEN_DYNAMIC_MODULE_EXPORTS
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for imported in node.names:
                bound_name = imported.asname or imported.name.split(".", 1)[0]
                if bound_name in forbidden_names:
                    name = bound_name
                    break
            else:
                continue
        else:
            continue
        if name == "optimize_model":
            raise AdmissionError(
                "participant optimize_model exports are forbidden; "
                "use build_model_patch(contract)"
            )
        if name == "__getattr__":
            # PEP 562 could otherwise synthesize optimize_model only when the
            # trusted loader asks for that module attribute.
            raise AdmissionError("candidate module __getattr__ hooks are forbidden")


def _check_python(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise AdmissionError(f"candidate file is not regular: {path}")
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise AdmissionError(f"candidate file is too large: {path.name}")
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        raise AdmissionError(f"invalid Python source: {path.name}") from exc
    _reject_imperative_model_hook(tree)
    aliases = _import_aliases(tree)
    _propagate_torch_aliases(tree, aliases)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}

    for node in ast.walk(tree):
        roots: list[str] = []
        imported_modules: list[str] = []
        if isinstance(node, ast.Import):
            roots = [alias.name.split(".")[0] for alias in node.names]
            imported_modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots = [node.module.split(".")[0]]
            imported_modules = [f"{node.module}.{alias.name}" for alias in node.names]
        for root in roots:
            if root not in ALLOWED_IMPORT_ROOTS:
                raise AdmissionError(f"import {root!r} is outside the allowlist")
        for imported_module in imported_modules:
            if _module_is_forbidden(imported_module):
                raise AdmissionError(f"control-plane import {imported_module!r} is forbidden")
            if _is_forbidden_qualified(imported_module):
                raise AdmissionError(f"control-plane import {imported_module!r} is forbidden")
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] == "runtime" and node.module != "runtime.api":
                raise AdmissionError("candidate may import only runtime.api")
        if isinstance(node, ast.Import) and any(alias.name == "runtime" for alias in node.names):
            raise AdmissionError("candidate may import only runtime.api")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                raise AdmissionError(f"forbidden call {node.func.id!r}")
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_CALLS:
            raise AdmissionError(f"forbidden name {node.id!r}")
        if isinstance(node, ast.Name):
            qualified_name = _qualified_name(node, aliases)
            if qualified_name != node.id and _is_forbidden_qualified(qualified_name):
                raise AdmissionError(f"control-plane access {qualified_name!r} is forbidden")
            if qualified_name == "torch.cuda":
                parent = parents.get(node)
                if not (isinstance(parent, ast.Attribute) and parent.value is node):
                    raise AdmissionError("capturing the torch.cuda namespace is forbidden")
        if isinstance(node, ast.Attribute):
            if (
                node.attr.startswith("__")
                and node.attr not in SAFE_DUNDER_ATTRIBUTES
            ) or node.attr in FORBIDDEN_ATTRIBUTES:
                raise AdmissionError(f"forbidden attribute {node.attr!r}")
            if isinstance(node.ctx, (ast.Store, ast.Del)) and not _direct_candidate_state_write(node):
                raise AdmissionError("writes to external or nested attributes are forbidden")
            qualified = _qualified_name(node, aliases)
            if qualified and _is_forbidden_qualified(qualified):
                raise AdmissionError(f"control-plane access {qualified!r} is forbidden")
            # Do not allow a CUDA module handle to escape into a local/default
            # argument and later evade qualified-name checks.
            if qualified == "torch.cuda":
                parent = parents.get(node)
                if not (isinstance(parent, ast.Attribute) and parent.value is node):
                    raise AdmissionError("capturing the torch.cuda namespace is forbidden")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith(("/logs/", "/opt/", "/proc/", "/sys/")):
                raise AdmissionError("forbidden absolute path literal")


def admit(lab_root: Path, requested_version: int | None = None) -> dict[str, object]:
    lab_root = lab_root.resolve()
    solutions = lab_root / "solutions"
    if solutions.is_symlink() or not solutions.is_dir():
        raise AdmissionError("solutions must be a regular directory")
    retained: list[int] = []
    for entry in sorted(solutions.iterdir()):
        match = VERSION_RE.fullmatch(entry.name)
        if not match or entry.is_symlink() or not entry.is_dir():
            raise AdmissionError(f"unexpected solutions entry: {entry.name}")
        version = int(match.group(1))
        children = list(entry.iterdir())
        caches = [item for item in children if item.name == "__pycache__"]
        for cache in caches:
            if cache.is_symlink() or not cache.is_dir():
                raise AdmissionError("__pycache__ must be a regular directory")
            for artifact in cache.iterdir():
                if (
                    artifact.is_symlink()
                    or not artifact.is_file()
                    or artifact.suffix != ".pyc"
                    or artifact.stat().st_size > MAX_SOURCE_BYTES
                ):
                    raise AdmissionError("unexpected bytecode cache artifact")
        names = {item.name for item in children if item.name != "__pycache__"}
        if names != ALLOWED_SOLUTION_FILES:
            raise AdmissionError(f"unexpected files in {entry.name}: {sorted(names)}")
        for name in sorted(names):
            _check_python(entry / name)
        retained.append(version)
    if not retained or retained != list(range(1, max(retained) + 1)):
        raise AdmissionError("candidate versions must be contiguous from v001")
    marker = (lab_root / "FINAL_VERSION").read_text(encoding="utf-8").strip()
    if not marker.isascii() or not marker.isdigit():
        raise AdmissionError("FINAL_VERSION is invalid")
    selected = int(marker) if requested_version is None else requested_version
    if selected not in retained:
        raise AdmissionError("selected candidate version does not exist")
    notes = lab_root / "notes.md"
    if notes.is_symlink() or not notes.is_file() or notes.stat().st_size > MAX_NOTES_BYTES:
        raise AdmissionError("notes.md is missing, linked, or too large")
    return {"final_version": selected, "versions": retained}
