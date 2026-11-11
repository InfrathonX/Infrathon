#!/usr/bin/env python3
"""Fail-closed source and layout admission for KV-scoring policies."""
from __future__ import annotations
import argparse, ast, hashlib, json, re, stat, sys
from pathlib import Path
from typing import Any

FINAL_VERSION_RE = re.compile(rb"(?:[1-9][0-9]{0,2})\n?\Z")
CANDIDATE_RE = re.compile(r"scorer_v([0-9]{3})\Z")
MAX_VERSIONS = 64
MAX_SOURCE_BYTES = 512 * 1024
MAX_ASSET_BYTES = 128 * 1024 * 1024
MAX_NOTES_BYTES = 64 * 1024
ALLOWED_IMPORTS = {"math", "typing", "dataclasses", "torch", "torch.nn.functional", "triton", "triton.language"}
FORBIDDEN_BUILTIN_CALLS = {"open", "exec", "eval", "compile", "input", "breakpoint", "__import__", "getattr", "setattr", "delattr", "globals", "locals", "vars", "dir", "help", "memoryview"}
FORBIDDEN_LITERAL_MARKERS = ("/opt/models", "kv_scoring_runtime", "candidate_worker", "public_runner", "/logs/verifier", "__builtins__", "__class__", "__globals__", "__subclasses__", "tb_frame", "f_back")
FORBIDDEN_ROOTS = {"os", "sys", "pathlib", "subprocess", "socket", "urllib", "requests", "importlib", "inspect", "pickle", "marshal", "ctypes", "multiprocessing", "threading", "asyncio"}
FORBIDDEN_CALL_PREFIXES = (
    "torch.save", "torch.load", "torch.compile", "torch.jit", "torch.hub",
    "torch.distributed", "torch.utils.cpp_extension", "torch.ops.load_library",
    "torch.cuda.Stream", "torch.cuda.stream", "torch.cuda.graph", "torch.cuda.CUDAGraph",
    "torch.cuda.set_rng_state", "torch.cuda.manual_seed", "torch.manual_seed",
    "torch.set_default", "torch.use_deterministic_algorithms", "torch.from_file",
    "torch.package", "torch.multiprocessing", "torch.serialization",
    # the shard's GPU is the current device; allocator counters prove no other GPU is used
    "torch.cuda.set_device", "torch.cuda.device", "torch.cuda.reset_peak_memory_stats",
    "torch.cuda.reset_accumulated_memory_stats", "torch.cuda.reset_max_memory_allocated",
    "torch.cuda.reset_max_memory_cached",
)
SELECT_PARAMS = {"hidden_states", "queries", "keys", "values", "position_embeddings", "weights"}
MUTATING_TAILS = {"copy_", "fill_", "zero_", "add_", "sub_", "mul_", "div_", "scatter_", "index_copy_", "masked_fill_", "resize_", "set_"}
EXPECTED = {
    "__init__": ["self", "spec", "assets"],
    "score": ["self", "layer_idx", "hidden_states", "queries", "keys", "values", "position_embeddings", "weights", "ctx_range"],
}

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _dotted(node: ast.AST) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr); node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id); return ".".join(reversed(parts))
    return None

class PolicyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.imported_roots: set[str] = set()
        self.alias_paths: dict[str, str] = {}

    def resolved(self, dotted: str) -> str:
        if not dotted: return dotted
        root, dot, tail = dotted.partition(".")
        return self.alias_paths.get(root, root) + (dot + tail if dot else "")

    def forbidden_path(self, dotted: str) -> bool:
        return dotted.split('.', 1)[0] in FORBIDDEN_ROOTS or any(dotted == p or dotted.startswith(p + '.') for p in FORBIDDEN_CALL_PREFIXES)

    def error(self, node: ast.AST, message: str) -> None:
        self.errors.append(f"line {getattr(node, 'lineno', '?')}: {message}")

    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            if alias.name not in ALLOWED_IMPORTS:
                self.error(node, f"forbidden import: {alias.name}")
            self.imported_roots.add(alias.asname or alias.name.split('.')[0])
            self.alias_paths[alias.asname or alias.name.split('.')[0]] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        module = node.module or ""
        if node.level or module not in ALLOWED_IMPORTS:
            self.error(node, f"forbidden import: {'.' * node.level}{module}")
        for alias in node.names:
            self.imported_roots.add(alias.asname or alias.name)
            self.alias_paths[alias.asname or alias.name] = f"{module}.{alias.name}"
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> Any: self.error(node, "global is forbidden")
    def visit_Nonlocal(self, node: ast.Nonlocal) -> Any: self.error(node, "nonlocal is forbidden")

    def visit_Try(self, node: ast.Try) -> Any:
        if any(isinstance(x, (ast.Import, ast.ImportFrom)) for x in ast.walk(node)):
            self.error(node, "try/except around import is forbidden")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if "__" in node.attr:
            self.error(node, "dunder attribute access is forbidden")
        dotted = _dotted(node) or ""
        if self.forbidden_path(self.resolved(dotted)):
            self.error(node, f"forbidden process/load reference: {self.resolved(dotted)}")
        if any(part in {"gi_frame", "cr_frame", "tb_frame", "f_back", "f_globals", "f_locals"} for part in dotted.split('.')):
            self.error(node, "frame/traceback access is forbidden")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        if isinstance(node.ctx, ast.Load) and node.id in self.alias_paths and self.forbidden_path(self.alias_paths[node.id]):
            self.error(node, f"forbidden aliased reference: {self.alias_paths[node.id]}")

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, str) and any(x in node.value for x in FORBIDDEN_LITERAL_MARKERS):
            self.error(node, "forbidden literal marker")

    def visit_Call(self, node: ast.Call) -> Any:
        dotted = self.resolved(_dotted(node.func) or "")
        tail = dotted.rsplit('.', 1)[-1]
        if dotted in FORBIDDEN_BUILTIN_CALLS or (isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_BUILTIN_CALLS):
            self.error(node, f"forbidden builtin call: {dotted}")
        if self.forbidden_path(dotted):
            self.error(node, f"forbidden exact/process/load call: {dotted}")
        if tail in MUTATING_TAILS:
            root = dotted.split('.', 1)[0]
            if root in SELECT_PARAMS:
                self.error(node, f"obvious input mutation: {dotted}")
        self.generic_visit(node)

    def _assignment(self, node: ast.AST, targets: list[ast.AST]) -> None:
        for target in targets:
            dotted = _dotted(target)
            if dotted and dotted.split('.', 1)[0] in self.imported_roots:
                self.error(node, f"forbidden imported-module mutation: {dotted}")
            if dotted and dotted.startswith("torch.backends."):
                self.error(node, "torch.backends assignment is forbidden")

    def visit_Assign(self, node: ast.Assign) -> Any:
        self._assignment(node, list(node.targets))
        resolved = self.resolved(_dotted(node.value) or "")
        for target in node.targets:
            if isinstance(target, ast.Name) and resolved: self.alias_paths[target.id] = resolved
        self.generic_visit(node)
    def visit_AnnAssign(self, node: ast.AnnAssign) -> Any:
        self._assignment(node, [node.target]); self.generic_visit(node)
    def visit_AugAssign(self, node: ast.AugAssign) -> Any:
        self._assignment(node, [node.target]); self.generic_visit(node)

def _scan_selected_source(text: str, relative: str) -> list[str]:
    try:
        tree = ast.parse(text, filename=relative)
    except (SyntaxError, ValueError) as exc:
        return [f"source parse failed: {exc}"]
    visitor = PolicyVisitor(); visitor.visit(tree)
    policies = [x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "KVScorer"]
    if len(policies) != 1:
        visitor.errors.append("module must define exactly one class KVScorer")
        return visitor.errors
    methods = {x.name: x for x in policies[0].body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for name, expected in EXPECTED.items():
        fn = methods.get(name)
        if fn is None:
            visitor.errors.append(f"KVScorer missing method {name}"); continue
        if isinstance(fn, ast.AsyncFunctionDef):
            visitor.errors.append(f"{name} must not be async")
        args = fn.args
        actual = [x.arg for x in args.posonlyargs + args.args]
        if actual != expected or args.vararg or args.kwarg or args.kwonlyargs:
            visitor.errors.append(f"{name} parameter names differ: expected {expected}")
    return visitor.errors

def _regular(path: Path, root: Path, errors: list[str], *, max_bytes: int, required: bool = True) -> bytes | None:
    try:
        st = path.lstat()
    except FileNotFoundError:
        if required: errors.append(f"missing file: {path.relative_to(root)}")
        return None
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        errors.append(f"not a regular non-symlink file: {path.relative_to(root)}"); return None
    if st.st_size > max_bytes:
        errors.append(f"file too large: {path.relative_to(root)}"); return None
    return path.read_bytes()

def _workspace_cache_entry(path: Path) -> bool:
    return bool(
        (path.name == "__pycache__" and path.is_dir() and not path.is_symlink())
        or (path.name.endswith(".pyc") and path.is_file() and not path.is_symlink())
    )

def validate_submission(root: Path, candidate_version: int | None = None, *, workspace_mode: bool = False) -> dict[str, Any]:
    root = Path(root).absolute()
    errors: list[str] = []; warnings: list[str] = []
    result: dict[str, Any] = {"ok": False, "final_version": None, "candidate_version": None, "num_candidate_versions": 0, "versions": [], "selected_path": None, "selected_relative_path": None, "selected_sha256": None, "selected_assets_path": None, "errors": errors, "warnings": warnings}
    if not root.is_dir() or root.is_symlink():
        errors.append("submission root must be a real directory"); return result
    allowed = {"candidates", "FINAL_VERSION", "notes.md"}
    # Workspace mode inspects the live Work lab, whose root also carries the
    # task-owned files the image bakes in (AGENT_TASK.md, README_AGENT.md,
    # tools/, the root-owned .sforge/ pre-submit hook of the single-accelerator
    # profile) and the agent's scratch build/ directory. submit_paths strips
    # every one of them before the Judge sees a submission, so Judge mode
    # (the default) stays strict.
    if workspace_mode: allowed |= {"AGENT_TASK.md", "README_AGENT.md", "build", "tools", ".sforge"}
    for p in root.iterdir():
        if workspace_mode and _workspace_cache_entry(p): continue
        if p.name not in allowed: errors.append(f"unexpected root entry: {p.name}")
    final_raw = _regular(root / "FINAL_VERSION", root, errors, max_bytes=16)
    selected: int | None = None
    if final_raw is not None:
        if FINAL_VERSION_RE.fullmatch(final_raw) is None: errors.append("FINAL_VERSION must contain one canonical integer from 1 to 999")
        else: selected = int(final_raw)
    _regular(root / "notes.md", root, errors, max_bytes=MAX_NOTES_BYTES)
    candidates = root / "candidates"
    if not candidates.is_dir() or candidates.is_symlink(): errors.append("candidates must be a real directory")
    versions: list[int] = []; sources: dict[int, tuple[bytes, Path]] = {}; asset_paths: dict[int, Path] = {}; asset_refs: dict[int, str] = {}
    if candidates.is_dir() and not candidates.is_symlink():
        for entry in candidates.iterdir():
            match = CANDIDATE_RE.fullmatch(entry.name)
            if not match:
                errors.append(f"unexpected candidates entry: {entry.name}"); continue
            version = int(match.group(1)); versions.append(version)
            if not entry.is_dir() or entry.is_symlink(): errors.append(f"candidate is not a real directory: {entry.name}"); continue
            names = {x.name for x in entry.iterdir() if not (workspace_mode and _workspace_cache_entry(x))}
            if not names <= {"submission.py", "assets.pt", "assets.ref"}: errors.append(f"unexpected candidate entry in {entry.name}")
            if {"assets.pt", "assets.ref"} <= names: errors.append(f"{entry.name} may contain only one of assets.pt or assets.ref")
            raw = _regular(entry / "submission.py", root, errors, max_bytes=MAX_SOURCE_BYTES)
            if raw is not None: sources[version] = (raw, entry / "submission.py")
            if "assets.pt" in names:
                if _regular(entry / "assets.pt", root, errors, max_bytes=MAX_ASSET_BYTES) is not None: asset_paths[version] = entry / "assets.pt"
            if "assets.ref" in names:
                ref_raw = _regular(entry / "assets.ref", root, errors, max_bytes=32)
                if ref_raw is not None:
                    try: ref = ref_raw.decode("ascii").strip()
                    except UnicodeDecodeError: errors.append(f"{entry.name}/assets.ref must be ASCII")
                    else:
                        if re.fullmatch(r"scorer_v[0-9]{3}", ref) is None: errors.append(f"{entry.name}/assets.ref must contain one scorer_vNNN")
                        else: asset_refs[version] = ref
    versions.sort(); result["versions"] = versions; result["num_candidate_versions"] = len(versions); result["final_version"] = selected
    if not versions: errors.append("at least one scorer version is required")
    if len(versions) > MAX_VERSIONS: errors.append(f"submission retains {len(versions)} versions; maximum is {MAX_VERSIONS}")
    if selected is not None and selected not in versions: errors.append(f"FINAL_VERSION selects missing scorer_v{selected:03d}")
    for version, ref in asset_refs.items():
        target = int(ref[-3:])
        if target == version: errors.append(f"scorer_v{version:03d}/assets.ref may not reference itself")
        elif target in asset_refs: errors.append(f"scorer_v{version:03d}/assets.ref chains are forbidden")
        elif target not in asset_paths: errors.append(f"scorer_v{version:03d}/assets.ref target must hold a regular assets.pt")
    requested = selected
    if candidate_version is not None:
        if isinstance(candidate_version, bool) or not isinstance(candidate_version, int) or not 1 <= candidate_version <= 999:
            errors.append("candidate_version must be an integer from 1 to 999"); requested = None
        else: requested = candidate_version
    result["candidate_version"] = requested
    if requested is not None and requested not in versions: errors.append(f"selected candidate scorer_v{requested:03d} is missing")
    if requested in sources:
        raw, path = sources[requested]
        result["selected_path"] = str(path); result["selected_relative_path"] = str(path.relative_to(root)); result["selected_sha256"] = _sha256(raw)
        if not raw: errors.append("selected source must not be empty")
        else:
            try: text = raw.decode("utf-8")
            except UnicodeDecodeError: errors.append("selected source must be UTF-8")
            else: errors.extend(_scan_selected_source(text, result["selected_relative_path"]))
        if requested in asset_paths: result["selected_assets_path"] = str(asset_paths[requested])
        elif requested in asset_refs:
            target = int(asset_refs[requested][-3:])
            if target in asset_paths: result["selected_assets_path"] = str(asset_paths[target])
    missing = sorted(set(range(min(versions), max(versions) + 1)) - set(versions)) if versions else []
    if missing: warnings.append(f"candidate version sequence has gaps: {missing}")
    result["ok"] = not errors
    return result

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True); p.add_argument("--candidate-version", type=int); p.add_argument("--workspace-mode", action="store_true")
    a = p.parse_args(argv); result = validate_submission(a.root, a.candidate_version, workspace_mode=a.workspace_mode)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0 if result["ok"] else 2

if __name__ == "__main__": raise SystemExit(main())
