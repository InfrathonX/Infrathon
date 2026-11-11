#!/usr/bin/env python3
"""Participant-visible source admission for a self-contained CUDA codec.

This gate deliberately checks a small, understandable source contract. Runtime
checks remain authoritative for payload structure, memory, fidelity, state,
and timing.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path


MAX_FILES = 64
MAX_TOTAL_BYTES = 4 * 1024 * 1024
POLICY_SCHEMA_VERSION = 4
REQUIRED_FILES = {"mlsysforge_codec.py"}
OPTIONAL_TOP_LEVEL_FILES = {"notes.md"}
TRUSTED_ORACLE_TREE_SHA256 = (
    "83aeca827a5496f25d99c7801cb82aaabb3aab25936f2bca16d0d194244d3cef"
)

ALLOWED_IMPORT_ROOTS = {
    "__future__",
    "codec",
    "collections",
    "dataclasses",
    "einops",
    "enum",
    "functools",
    "itertools",
    "math",
    "operator",
    "torch",
    "triton",
    "typing",
}
FORBIDDEN_CALL_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
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
    "as_subclass",
    "cpu",
    "data_ptr",
    "dump",
    "dumps",
    "from_file",
    "load",
    "loads",
    "manual_seed",
    "manual_seed_all",
    "numpy",
    "record_stream",
    "register_hook",
    "register_post_accumulate_grad_hook",
    "requires_grad_",
    # The sharded Judge keeps every GPU visible and fails a case that left
    # allocator memory (current or peak) on another device; a codec may not
    # move the process to another device or erase that peak record.
    "reset_accumulated_memory_stats",
    "reset_max_memory_allocated",
    "reset_max_memory_cached",
    "reset_peak_memory_stats",
    "retain_grad",
    "save",
    "set_",
    "set_default_device",
    "set_device",
    "set_default_dtype",
    "share_memory_",
    "storage",
    "tolist",
    "untyped_storage",
    "use_deterministic_algorithms",
}
FORBIDDEN_LITERAL_FRAGMENTS = {
    "/eval",
    "/logs",
    "/models",
    "/opt/qvg-judge",
    "golden",
    "private_judge",
    "release.json",
    "report.json",
    "submission.tar",
}
FORBIDDEN_TORCH_SURFACES = {
    "_C",
    "autograd",
    "backends",
    "classes",
    "compile",
    "compiler",
    "distributed",
    "hub",
    "jit",
    "library",
    "multiprocessing",
    "ops",
    "package",
    "profiler",
    "serialization",
    "utils",
}
FORBIDDEN_TRITON_SURFACES = {"_C", "backends", "runtime"}
# Autograd state changes made through a function call (``torch.enable_grad``,
# ``torch.set_grad_enabled``) rather than a tensor method; an autograd graph or
# saved gradient is persistent tensor state the byte counter never sees.
FORBIDDEN_GRAD_FUNCTIONS = {"enable_grad", "set_grad_enabled"}
# Assignment to one of these attributes is an autograd side channel
# (``t.requires_grad = True``, ``t.grad = other``) even when the receiver is a
# local tensor, so it is rejected independently of the locality rules.
FORBIDDEN_ATTRIBUTE_STORES = {"grad", "requires_grad"}
# Memoizing decorators keep their results -- and any tensors those results
# reference -- alive across every later call.  All of these live under
# ``functools`` (an allowed import root); none is a legitimate stateless codec
# helper, so each is rejected wherever a functools alias reaches it or it is
# imported by name.
FORBIDDEN_MEMOIZERS = {"cache", "cached_property", "lru_cache"}
# ``operator`` is an allowed import root, and its string arguments are opaque to
# this checker, so these name-driven attribute-access / mutation helpers would
# launder a forbidden attribute or item operation past the static surface.
FORBIDDEN_OPERATOR_FUNCTIONS = {"attrgetter", "delitem", "methodcaller", "setitem"}
# A string literal shaped like a dunder (``__class__``, ``__globals__`` ...) is
# only ever useful to drive a name-based lookup past the static checks.
DUNDER_LITERAL = re.compile(r"__\w+__")
MUTATING_METHODS = {
    "add",
    "append",
    "clear",
    "discard",
    "extend",
    "insert",
    "pop",
    "popitem",
    "remove",
    "reverse",
    "setdefault",
    "sort",
    "update",
}
# Every standard-library top-level module name, plus ``builtins``.  A chain
# element that names one of these on an imported-module alias is a re-exported
# escape hatch (``torch.os``, ``torch.sys``, ``torch.ctypes``, ``torch.builtins``,
# ``triton.importlib`` ...) and is rejected wherever it appears.  Only chain
# *elements* are tested, never the import root itself, so an allowed stdlib root
# such as ``operator`` or ``collections`` is unaffected.
STDLIB_MODULE_NAMES = frozenset(sys.stdlib_module_names) | {"builtins"}
# The only leaf attributes an imported-module alias chain may be *read as a
# value* (anything other than the base of a longer chain, a call target, or a
# decorator): numeric dtypes, numeric constants, and the handful of types used
# for ``isinstance``.  These are plain constants, not module objects, so they
# cannot be laundered into an unchecked attribute chain.  torch, triton.language
# and numpy spellings are unioned; collisions are harmless.
ALLOWED_VALUE_LEAVES = frozenset(
    {
        # torch / numpy / triton numeric dtypes
        "bfloat16",
        "bool",
        "bool_",
        "byte",
        "cdouble",
        "cfloat",
        "chalf",
        "char",
        "complex32",
        "complex64",
        "complex128",
        "complex256",
        "double",
        "float",
        "float_",
        "float16",
        "float32",
        "float64",
        "float128",
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "half",
        "int",
        "int_",
        "int1",
        "int8",
        "int16",
        "int32",
        "int64",
        "intp",
        "long",
        "short",
        "uint1",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "uintp",
        # triton.language dtype spellings
        "constexpr",
        "float8e4b8",
        "float8e4b15",
        "float8e4nv",
        "float8e5",
        "float8e5b16",
        "pi32_t",
        "void",
        # numeric constants
        "e",
        "inf",
        "nan",
        "newaxis",
        "pi",
        # types used in isinstance / annotations
        "Size",
        "Tensor",
        "dtype",
    }
)


class PolicyError(ValueError):
    """A participant artifact violates the published source contract."""


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    message: str


def _attribute_chain(node: ast.Attribute) -> tuple[str | None, list[str]]:
    parts = [node.attr]
    value = node.value
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.reverse()
        return value.id, parts
    return None, []


def _root_identifier(node: ast.AST) -> str | None:
    value = node
    while isinstance(value, (ast.Attribute, ast.Subscript)):
        value = value.value
    return value.id if isinstance(value, ast.Name) else None


def _annotate_parents(tree: ast.AST) -> None:
    """Record each node's syntactic parent and field so a use site's position
    (attribute base, call target, decorator, or a plain value) can be decided
    statically.  Done iteratively so a deeply nested source does not recurse."""
    for node in ast.walk(tree):
        for field, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        item._qvg_parent = node  # type: ignore[attr-defined]
                        item._qvg_field = field  # type: ignore[attr-defined]
            elif isinstance(value, ast.AST):
                value._qvg_parent = node  # type: ignore[attr-defined]
                value._qvg_field = field  # type: ignore[attr-defined]


def _parent_and_field(node: ast.AST) -> tuple[ast.AST | None, str | None]:
    return getattr(node, "_qvg_parent", None), getattr(node, "_qvg_field", None)


def _is_attribute_base(node: ast.AST) -> bool:
    """True when ``node`` is the ``value`` of an enclosing ``Attribute`` -- i.e.
    it is the base of a longer chain that is itself checked at its own top."""
    parent, field = _parent_and_field(node)
    return isinstance(parent, ast.Attribute) and field == "value"


def _is_call_target(node: ast.AST) -> bool:
    parent, field = _parent_and_field(node)
    return isinstance(parent, ast.Call) and field == "func"


def _is_decorator(node: ast.AST) -> bool:
    parent, field = _parent_and_field(node)
    return (
        isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and field == "decorator_list"
    )


def _statically_immutable(node: ast.AST) -> bool:
    if isinstance(node, (ast.Constant, ast.Name, ast.Attribute)):
        return True
    if isinstance(node, ast.Tuple):
        return all(_statically_immutable(item) for item in node.elts)
    if isinstance(node, ast.Subscript):
        return _statically_immutable(node.value) and _statically_immutable(node.slice)
    if isinstance(node, ast.UnaryOp):
        return _statically_immutable(node.operand)
    if isinstance(node, ast.BinOp):
        return _statically_immutable(node.left) and _statically_immutable(node.right)
    if isinstance(node, ast.BoolOp):
        return all(_statically_immutable(item) for item in node.values)
    return False


def _is_main_guard(node: ast.If) -> bool:
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


class _Bindings(ast.NodeVisitor):
    def __init__(self, arguments: set[str]) -> None:
        self.names = set(arguments)
        self.globals: set[str] = set()
        self.nonlocals: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            self.names.add(alias.asname or alias.name.split(".", 1)[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        for alias in node.names:
            self.names.add(alias.asname or alias.name)

    def visit_Global(self, node: ast.Global) -> None:  # noqa: N802
        self.globals.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:  # noqa: N802
        self.nonlocals.update(node.names)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self.names.add(node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        return


class SourceVisitor(ast.NodeVisitor):
    def __init__(self, relative_path: str) -> None:
        self.relative_path = relative_path
        self.findings: list[Finding] = []
        self.scopes: list[tuple[str, frozenset[str], str | None]] = [
            ("module", frozenset(), None)
        ]
        self.torch_aliases: set[str] = set()
        self.triton_aliases: set[str] = set()
        # Every local name bound to an imported module or imported symbol.
        self.import_aliases: set[str] = set()
        # The subset that is bound to a *module object* (``import X``,
        # ``import X.Y as Z``, ``from triton import language``): only these may
        # launder a live module into an unchecked attribute chain, so only these
        # are forbidden from appearing as a plain value.
        self.module_aliases: set[str] = set()
        # Local names bound to the ``functools`` / ``operator`` modules, so a
        # ``functools.lru_cache`` / ``operator.attrgetter`` chain can be caught.
        self.functools_aliases: set[str] = set()
        self.operator_aliases: set[str] = set()
        # Bare names pulled in by ``from functools import lru_cache`` /
        # ``from operator import attrgetter`` that are forbidden on their own.
        self.forbidden_direct_names: set[str] = set()

    def collect_aliases(self, tree: ast.AST) -> None:
        """Pre-pass: gather every import-bound name before checking use sites, so
        a chain is resolved against its alias regardless of statement order."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    local = alias.asname or root
                    self.import_aliases.add(local)
                    self.module_aliases.add(local)
                    if root == "torch":
                        self.torch_aliases.add(local)
                    elif root == "triton":
                        self.triton_aliases.add(local)
                    elif root == "functools":
                        self.functools_aliases.add(local)
                    elif root == "operator":
                        self.operator_aliases.add(local)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    continue
                module = node.module or ""
                root = module.split(".", 1)[0]
                for alias in node.names:
                    local = alias.asname or alias.name
                    self.import_aliases.add(local)
                    if root == "torch":
                        # from-imports off torch are rejected outright; mark the
                        # name a module alias for defence in depth regardless.
                        self.module_aliases.add(local)
                    if root == "triton" and module == "triton" and alias.name == "language":
                        self.triton_aliases.add(local)
                        self.module_aliases.add(local)
                    if root == "functools" and alias.name in FORBIDDEN_MEMOIZERS:
                        self.forbidden_direct_names.add(local)
                    if root == "operator" and alias.name in FORBIDDEN_OPERATOR_FUNCTIONS:
                        self.forbidden_direct_names.add(local)

    def reject(self, node: ast.AST, message: str) -> None:
        self.findings.append(
            Finding(self.relative_path, int(getattr(node, "lineno", 0)), message)
        )

    def visit_If(self, node: ast.If) -> None:  # noqa: N802
        if _is_main_guard(node):
            for statement in node.orelse:
                self.visit(statement)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for alias in node.names:
            root = alias.name.split(".", 1)[0]
            local = alias.asname or root
            if root not in ALLOWED_IMPORT_ROOTS:
                self.reject(node, f"import is outside the codec allowlist: {alias.name}")
            parts = alias.name.split(".")
            if root == "torch":
                self.torch_aliases.add(local)
                if len(parts) > 1 and parts[1] in FORBIDDEN_TORCH_SURFACES:
                    self.reject(node, f"import reaches a forbidden torch surface: {alias.name}")
            elif root == "triton":
                self.triton_aliases.add(local)
                if len(parts) > 1 and parts[1] in FORBIDDEN_TRITON_SURFACES:
                    self.reject(node, f"import reaches a forbidden Triton surface: {alias.name}")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        if node.level:
            return
        module = node.module or ""
        root = module.split(".", 1)[0]
        if root not in ALLOWED_IMPORT_ROOTS:
            self.reject(node, f"import is outside the codec allowlist: {module}")
        # A wildcard import hides which names it binds, so a forbidden
        # ``operator``/``functools`` helper (or anything else) could reach the
        # module namespace untracked.
        if any(alias.name == "*" for alias in node.names):
            self.reject(node, f"wildcard imports are forbidden: from {module} import *")
        if root == "torch":
            self.reject(node, "from-imports from torch are forbidden")
        if root == "triton" and module == "triton":
            for alias in node.names:
                if alias.name == "language":
                    self.triton_aliases.add(alias.asname or alias.name)
        if root == "triton" and any(
            part in FORBIDDEN_TRITON_SURFACES for part in module.split(".")[1:]
        ):
            self.reject(node, f"import reaches a forbidden Triton surface: {module}")
        if root == "functools":
            for alias in node.names:
                if alias.name in FORBIDDEN_MEMOIZERS:
                    self.forbidden_direct_names.add(alias.asname or alias.name)
        if root == "operator":
            for alias in node.names:
                if alias.name in FORBIDDEN_OPERATOR_FUNCTIONS:
                    self.forbidden_direct_names.add(alias.asname or alias.name)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
            if argument.annotation is not None:
                self.visit(argument.annotation)
        defaults = [
            *node.args.defaults,
            *(item for item in node.args.kw_defaults if item is not None),
        ]
        for default in defaults:
            if not _statically_immutable(default):
                self.reject(default, "mutable or computed function defaults are forbidden")
            self.visit(default)
        if node.returns is not None:
            self.visit(node.returns)

        arguments = {
            argument.arg
            for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        }
        if node.args.vararg is not None:
            arguments.add(node.args.vararg.arg)
        if node.args.kwarg is not None:
            arguments.add(node.args.kwarg.arg)
        bindings = _Bindings(arguments)
        for statement in node.body:
            bindings.visit(statement)
        locals_ = bindings.names - bindings.globals - bindings.nonlocals
        receiver = node.args.args[0].arg if self.scopes[-1][0] == "class" and node.args.args else None
        self.scopes.append(("function", frozenset(locals_), receiver))
        for statement in node.body:
            self.visit(statement)
        self.scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        for decorator in node.decorator_list:
            self.visit(decorator)
        # Subclassing a torch/triton type (``class C(torch.Tensor)``) -- or
        # using one as a metaclass -- yields an object whose storage/size/device
        # behaviour can diverge from a plain tensor and slip past the byte
        # counter, so reject any base or metaclass rooted at a torch/triton
        # import alias.
        subtype_aliases = self.torch_aliases | self.triton_aliases
        for base in node.bases:
            if _root_identifier(base) in subtype_aliases:
                self.reject(node, "subclassing a torch/triton type is forbidden")
        for keyword in node.keywords:
            if keyword.arg == "metaclass" and _root_identifier(keyword.value) in subtype_aliases:
                self.reject(node, "a torch/triton metaclass is forbidden")
        for base in node.bases:
            self.visit(base)
        self.scopes.append(("class", frozenset(), None))
        for statement in node.body:
            self.visit(statement)
        self.scopes.pop()

    def _check_target(self, target: ast.AST, node: ast.AST) -> None:
        # Writing ``.grad`` or ``.requires_grad`` is an autograd side channel
        # regardless of whose tensor the receiver is, so reject it before the
        # ordinary locality rules.
        if isinstance(target, ast.Attribute) and target.attr in FORBIDDEN_ATTRIBUTE_STORES:
            self.reject(node, f"assigning to .{target.attr} is a forbidden autograd side channel")
        kind, locals_, receiver = self.scopes[-1]
        if kind != "function":
            if isinstance(target, (ast.Attribute, ast.Subscript)):
                self.reject(node, "module/class persistent mutation is forbidden")
            return
        if not isinstance(target, (ast.Attribute, ast.Subscript)):
            return
        root = _root_identifier(target)
        if root == receiver:
            return
        elif root is None or root not in locals_:
            self.reject(node, "non-local attribute/container mutation is forbidden")

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        if self.scopes[-1][0] in {"module", "class"} and not _statically_immutable(node.value):
            self.reject(node, "mutable/computed module or class state is forbidden")
        for target in node.targets:
            self._check_target(target, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        if self.scopes[-1][0] in {"module", "class"} and (
            node.value is None or not _statically_immutable(node.value)
        ):
            self.reject(node, "mutable/computed module or class state is forbidden")
        self._check_target(node.target, node)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
        if self.scopes[-1][0] in {"module", "class"}:
            self.reject(node, "module/class persistent mutation is forbidden")
        self._check_target(node.target, node)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:  # noqa: N802
        self.reject(node, "module-global mutation is forbidden")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:  # noqa: N802
        self.reject(node, "nonlocal persistent mutation is forbidden")

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        direct = node.func.id if isinstance(node.func, ast.Name) else None
        attribute = node.func.attr if isinstance(node.func, ast.Attribute) else None
        if direct in FORBIDDEN_CALL_NAMES:
            self.reject(node, f"forbidden dynamic or ambient call: {direct}")
        root, call_chain = (
            _attribute_chain(node.func)
            if isinstance(node.func, ast.Attribute)
            else (None, [])
        )
        # A forbidden ambient name is a master key whether it is reached bare
        # (``getattr(...)``) or as the final attribute of an import-rooted chain
        # (``torch.builtins.getattr(...)``, ``torch.os.system(...)``).
        if root in self.import_aliases and call_chain and call_chain[-1] in FORBIDDEN_CALL_NAMES:
            self.reject(
                node,
                f"forbidden dynamic or ambient call: {root}.{'.'.join(call_chain)}",
            )
        # ``load`` is a serialization/ambient capability on ordinary Python
        # and torch objects, but it is the normal device-memory read primitive
        # in ``triton.language`` kernels.  Keep the broad prohibition while
        # admitting precisely that kernel operation.
        triton_device_load = attribute == "load" and root in self.triton_aliases
        if attribute in FORBIDDEN_ATTRIBUTES and not triton_device_load:
            self.reject(node, f"forbidden serialization/offload call: {attribute}")
        values = [*node.args, *(keyword.value for keyword in node.keywords)]
        if any(
            isinstance(value, ast.Constant)
            and isinstance(value.value, str)
            and value.value.lower() == "cpu"
            for value in values
        ):
            self.reject(node, "CPU offload is forbidden")
        for keyword in node.keywords:
            if (
                keyword.arg == "device"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
                and keyword.value.value.lower() == "cpu"
            ):
                self.reject(node, "CPU offload is forbidden")
            # Enabling grad on a freshly created tensor is an autograd side
            # channel just like calling ``requires_grad_()``.
            if keyword.arg == "requires_grad":
                self.reject(node, "the requires_grad= keyword is forbidden")
        if isinstance(node.func, ast.Attribute) and (
            attribute in MUTATING_METHODS or (attribute or "").endswith("_")
        ):
            kind, locals_, receiver = self.scopes[-1]
            root = _root_identifier(node.func.value)
            if kind != "function":
                self.reject(node, "module/class persistent mutation call is forbidden")
            elif root == receiver:
                pass
            elif root is None or root not in locals_:
                self.reject(node, "non-local mutation call is forbidden")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if node.id in FORBIDDEN_CALL_NAMES:
            self.reject(node, f"forbidden dynamic or ambient name: {node.id}")
        if node.id.startswith("__"):
            self.reject(node, f"dunder identifier access is forbidden: {node.id}")
        # A memoizing decorator or name-driven operator helper pulled in by
        # ``from functools import lru_cache`` / ``from operator import
        # attrgetter`` is forbidden wherever its bare name is used.
        if isinstance(node.ctx, ast.Load) and node.id in self.forbidden_direct_names:
            self.reject(node, f"forbidden memoizer or indirect-lookup helper: {node.id}")
        # A live module alias may only be the base of a checked attribute chain,
        # a call target, or a decorator.  Anywhere else (``t = torch``, a dict
        # value, a default, an iterable, a walrus) it escapes into an unchecked
        # name, so reject it.
        if (
            isinstance(node.ctx, ast.Load)
            and node.id in self.module_aliases
            and not (
                _is_attribute_base(node) or _is_call_target(node) or _is_decorator(node)
            )
        ):
            self.reject(node, f"import alias used as a value is forbidden (laundering): {node.id}")

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        if node.attr.startswith("__"):
            self.reject(node, f"dunder attribute access is forbidden: {node.attr}")
        root, chain = _attribute_chain(node)
        triton_device_load = node.attr == "load" and root in self.triton_aliases
        if node.attr in FORBIDDEN_ATTRIBUTES and not triton_device_load:
            self.reject(node, f"forbidden serialization/offload attribute: {node.attr}")
        # ``torch.enable_grad`` / ``torch.set_grad_enabled`` (as a call or a
        # context manager) turn autograd back on and build a persistent graph.
        if node.attr in FORBIDDEN_GRAD_FUNCTIONS:
            self.reject(node, f"forbidden autograd state change: {node.attr}")
        # A memoizing decorator reached through a functools alias keeps tensors
        # alive across calls; a name-driven operator helper launders attribute
        # access or mutation past the static surface.
        if root in self.functools_aliases and chain and chain[-1] in FORBIDDEN_MEMOIZERS:
            self.reject(node, f"forbidden memoizer: {root}.{'.'.join(chain)}")
        if root in self.operator_aliases and chain and chain[-1] in FORBIDDEN_OPERATOR_FUNCTIONS:
            self.reject(node, f"forbidden indirect-lookup helper: {root}.{'.'.join(chain)}")
        if root in self.import_aliases and chain:
            # (a) Any chain element that names a standard-library module is a
            # re-exported escape hatch (torch.os, torch.sys, torch.ctypes,
            # torch.importlib, torch.builtins, triton.inspect, ...).
            stdlib_hit = next((part for part in chain if part in STDLIB_MODULE_NAMES), None)
            if stdlib_hit is not None:
                self.reject(
                    node,
                    f"import alias reaches a standard-library module: {root}.{stdlib_hit}",
                )
            # A private/native element anywhere on an alias chain (``torch._C``,
            # ``t._C`` once t is resolved) is never a legitimate codec surface.
            private_hit = next((part for part in chain if part.startswith("_")), None)
            if private_hit is not None:
                self.reject(
                    node,
                    f"forbidden private/native surface on import alias: {root}.{private_hit}",
                )
            if root in self.torch_aliases and chain[0] in FORBIDDEN_TORCH_SURFACES:
                self.reject(node, f"forbidden dynamic/native torch surface: {root}.{chain[0]}")
            if root in self.triton_aliases and chain[0] in FORBIDDEN_TRITON_SURFACES:
                self.reject(node, f"forbidden dynamic/native Triton surface: {root}.{chain[0]}")
            if chain[-2:] == ["cuda", "device"]:
                self.reject(node, "switching the current CUDA device is forbidden")
            # (b) At the top of a *module*-alias chain (not a base, call target
            # or decorator), the chain is being read as a value.  Allow that only
            # for an explicit allowlist of non-module constants (dtypes etc.); any
            # other value-read could launder a module object out of the chain.
            if (
                root in self.module_aliases
                and not _is_attribute_base(node)
                and not (_is_call_target(node) or _is_decorator(node))
                and chain[-1] not in ALLOWED_VALUE_LEAVES
            ):
                self.reject(
                    node,
                    "import alias used as a value is forbidden (laundering): "
                    f"{root}.{'.'.join(chain)}",
                )
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:  # noqa: N802
        if isinstance(node.value, str):
            lowered = node.value.lower()
            for fragment in FORBIDDEN_LITERAL_FRAGMENTS:
                if fragment in lowered:
                    self.reject(node, f"literal references a protected evaluator surface: {fragment}")
            # A dunder-shaped string literal only drives a name-based lookup
            # (``getattr(x, "__class__")``, ``operator.attrgetter("__globals__")``)
            # past the static checks, so reject it wherever it appears.
            dunder = DUNDER_LITERAL.search(node.value)
            if dunder is not None:
                self.reject(node, f"dunder-like string literal is forbidden: {dunder.group()}")


def _relative_files(submission: Path) -> list[Path]:
    if not submission.is_dir() or submission.is_symlink():
        raise PolicyError("submission must be a regular directory")
    files: list[Path] = []
    for path in sorted(submission.rglob("*")):
        relative = path.relative_to(submission).as_posix()
        if path.is_symlink():
            raise PolicyError(f"links are forbidden: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise PolicyError(f"special files are forbidden: {relative}")
        parts = Path(relative).parts
        if "__pycache__" in parts or path.suffix in {".pyc", ".pyo"}:
            continue
        files.append(path)
    return files


def _is_python_source(relative: str) -> bool:
    return relative == "mlsysforge_codec.py" or (
        relative.startswith("codec/") and relative.endswith(".py")
    )


def _tree_digest(submission: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    ordered = sorted(files, key=lambda path: path.relative_to(submission).as_posix().encode("utf-8"))
    for path in ordered:
        relative = path.relative_to(submission).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _parse_source(payload: bytes, relative: str) -> ast.AST:
    """Parse submitted source; every failure caused by its bytes is a PolicyError.

    ast.parse raises RecursionError (or MemoryError) on deeply nested source and
    ValueError on some malformed input; those are the submission's rejection,
    never an evaluator fault.
    """
    try:
        return ast.parse(payload.decode("utf-8"), filename=relative)
    except (UnicodeError, SyntaxError, ValueError, RecursionError, MemoryError) as exc:
        raise PolicyError(f"cannot parse {relative}: {type(exc).__name__}: {exc}") from exc


def validate_submission(submission: Path) -> dict[str, object]:
    submission = Path(submission)
    files = _relative_files(submission)
    if len(files) > MAX_FILES:
        raise PolicyError(f"submission has too many files: {len(files)} > {MAX_FILES}")

    names = {path.relative_to(submission).as_posix() for path in files}
    missing = sorted(REQUIRED_FILES - names)
    if missing:
        raise PolicyError(f"required codec files are missing: {missing}")

    total_bytes = 0
    python_files: list[tuple[str, bytes]] = []
    for path in files:
        relative = path.relative_to(submission).as_posix()
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise PolicyError(f"cannot read submitted file: {relative}: {exc}") from exc
        total_bytes += len(payload)
        if relative in OPTIONAL_TOP_LEVEL_FILES:
            continue
        if not _is_python_source(relative):
            raise PolicyError(f"file is outside the codec source closure: {relative}")
        python_files.append((relative, payload))
    if total_bytes > MAX_TOTAL_BYTES:
        raise PolicyError(f"submission source is too large: {total_bytes} > {MAX_TOTAL_BYTES} bytes")

    trusted_tree = _tree_digest(submission, files) == TRUSTED_ORACLE_TREE_SHA256
    findings: list[Finding] = []
    if not trusted_tree:
        for relative, payload in python_files:
            tree = _parse_source(payload, relative)
            _annotate_parents(tree)
            visitor = SourceVisitor(relative)
            visitor.collect_aliases(tree)
            try:
                visitor.visit(tree)
            except (RecursionError, MemoryError) as exc:
                # The visitor recurses once per AST level; a source nested too
                # deeply to check is the submission's own rejection.
                raise PolicyError(f"cannot check {relative}: source is nested too deeply") from exc
            findings.extend(visitor.findings)
    if findings:
        rendered = "; ".join(
            f"{finding.path}:{finding.line}: {finding.message}"
            for finding in findings[:20]
        )
        if len(findings) > 20:
            rendered += f"; ... {len(findings) - 20} more"
        raise PolicyError(rendered)

    try:
        entry_source = (submission / "mlsysforge_codec.py").read_bytes()
    except OSError as exc:
        raise PolicyError(f"cannot parse mlsysforge_codec.py: {exc}") from exc
    mlsysforge_tree = _parse_source(entry_source, "mlsysforge_codec.py")
    functions = {
        node.name for node in mlsysforge_tree.body if isinstance(node, ast.FunctionDef)
    }
    if not {"encode", "decode"}.issubset(functions):
        raise PolicyError("mlsysforge_codec.py must expose encode and decode")

    return {
        "valid": True,
        "policy_schema_version": POLICY_SCHEMA_VERSION,
        "files": len(files),
        "python_files": len(python_files),
        "checked_files": 0 if trusted_tree else len(python_files),
        "source_bytes": total_bytes,
        "trusted_tree": trusted_tree,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = validate_submission(args.submission)
    except PolicyError as exc:
        result = {
            "valid": False,
            "policy_schema_version": POLICY_SCHEMA_VERSION,
            "error": str(exc),
        }
        print(json.dumps(result, sort_keys=True))
        raise SystemExit(2)
    print(json.dumps(result, indent=2 if args.json else None, sort_keys=True))


if __name__ == "__main__":
    main()
