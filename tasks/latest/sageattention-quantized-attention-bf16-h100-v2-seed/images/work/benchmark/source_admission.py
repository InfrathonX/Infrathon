#!/usr/bin/env python3
"""Fail-closed layout and selected-source admission for quantized attention."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import stat
import sys
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any


CANDIDATE_RE = re.compile(r"^kernel_v([0-9]{3})$")
FINAL_VERSION_RE = re.compile(rb"(?:[1-9][0-9]{0,2})\n?\Z")
MAX_VERSIONS = 64
MAX_SOURCE_BYTES = 512 * 1024
MAX_NOTES_BYTES = 64 * 1024
MAX_TOTAL_BYTES = 4 * 1024 * 1024

EXPECTED_PARAMETERS = (
    "q",
    "k",
    "v",
    "is_causal",
    "softmax_scale",
    "out",
)
ALLOWED_IMPORTS = {"math", "typing", "torch", "triton", "triton.language"}
ALLOWED_KERNEL_DECORATORS = {
    "triton.jit",
    "triton.autotune",
}
ALLOWED_AUTOTUNE_KEYWORDS = {"configs", "key", "rep", "warmup"}
ALLOWED_CONFIG_KEYWORDS = {"maxnreg", "num_ctas", "num_stages", "num_warps"}
SAFE_HOST_BUILTINS = {
    "abs",
    "bool",
    "float",
    "int",
    "len",
    "max",
    "min",
    "range",
    "round",
    "tuple",
}
SAFE_HOST_MATH = {
    "math.ceil",
    "math.floor",
    "math.log2",
    "math.sqrt",
}
SAFE_TENSOR_METADATA = {"dim", "is_contiguous", "numel", "size", "stride"}
SAFE_TENSOR_ATTRIBUTES = {"device", "dtype", "is_cuda", "ndim", "shape"}
SAFE_TORCH_ALLOCATORS = {
    "torch.empty",
    "torch.empty_like",
    "torch.empty_strided",
    "torch.full",
    "torch.full_like",
    "torch.zeros",
    "torch.zeros_like",
}
FORBIDDEN_LITERAL_MARKERS = (
    "/logs/verifier",
    "/run/",
    "/tests",
    "/workspace/benchmark",
    "candidate_worker",
    "cutlass",
    "cutesdsl",
    "flash_attn",
    "flash_mla",
    "flashinfer",
    "http://",
    "https://",
    "quant_attention_runtime",
    "public_runner",
    "xformers",
)

# Enforced over the whole module by ``_PolicyVisitor``, independent of any
# reachability or kind inference, so a callable the entrypoint never calls is
# still denied the interpreter-mutation and dynamic-loading surface.
FORBIDDEN_BUILTIN_CALLS = {
    "__import__",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "locals",
    "open",
    "setattr",
    "vars",
}
FORBIDDEN_EXACT_CALLS = {"torch.compile", "torch.jit.load", "torch.load"}
FORBIDDEN_PROCESS_OR_LOAD_TAILS = {
    "cdll",
    "dlopen",
    "execv",
    "execve",
    "fork",
    "forkpty",
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
        return f"{prefix}.{node.attr}" if prefix else None
    return None


def _decorator_name(node: ast.AST, aliases: dict[str, str]) -> str | None:
    return _dotted_name(node.func if isinstance(node, ast.Call) else node, aliases)


def _reject_unsafe_decorator_arguments(
    decorator: ast.AST,
    name: str | None,
    aliases: dict[str, str],
    functions: dict[str, ast.FunctionDef],
    relative_path: str,
    errors: list[str],
) -> None:
    """Close the Triton decorator routes that run a callback in-process.

    ``triton.heuristics`` values, ``autotune(prune_configs_by=...)`` pruners,
    ``Config(pre_hook=...)`` hooks, and ``jit(launch_metadata=...)`` all call
    participant code at launch time, outside every reachable-call policy pass.
    """
    for item in ast.walk(decorator):
        if isinstance(item, ast.Name) and item.id in functions:
            errors.append(
                f"{relative_path}:{item.lineno}: "
                f"forbidden decorator callback: {item.id!r}"
            )
    if not isinstance(decorator, ast.Call):
        return
    if name == "triton.jit" and (decorator.args or decorator.keywords):
        errors.append(
            f"{relative_path}:{decorator.lineno}: "
            "triton.jit takes no decorator arguments"
        )
    for call in (item for item in ast.walk(decorator) if isinstance(item, ast.Call)):
        call_name = _dotted_name(call.func, aliases)
        if call_name == "triton.autotune":
            allowed, limit = ALLOWED_AUTOTUNE_KEYWORDS, 2
        elif call_name == "triton.Config":
            allowed, limit = ALLOWED_CONFIG_KEYWORDS, 1
        else:
            continue
        if len(call.args) > limit:
            errors.append(
                f"{relative_path}:{call.lineno}: "
                f"too many positional {call_name} arguments"
            )
        for keyword in call.keywords:
            if keyword.arg is None or keyword.arg not in allowed:
                errors.append(
                    f"{relative_path}:{call.lineno}: "
                    f"forbidden {call_name} keyword: {keyword.arg!r}"
                )


def _kernel_launch(node: ast.Call, kernels: set[str]) -> str | None:
    if not isinstance(node.func, ast.Subscript):
        return None
    target = node.func.value
    if isinstance(target, ast.Name) and target.id in kernels:
        return target.id
    return None


def _safe_annotation(node: ast.AST, aliases: dict[str, str]) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str) or node.value is None
    return _dotted_name(node, aliases) in {
        "bool",
        "float",
        "int",
        "str",
        "torch.Tensor",
        "triton.language.constexpr",
    }


def _safe_module_value(node: ast.AST, aliases: dict[str, str]) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (str, int, float, bool, type(None)))
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_safe_module_value(item, aliases) for item in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            (key is None or _safe_module_value(key, aliases))
            and _safe_module_value(value, aliases)
            for key, value in zip(node.keys, node.values)
        )
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _safe_module_value(node.operand, aliases)
    if isinstance(node, ast.Call) and _dotted_name(node.func, aliases) in {
        "triton.Config",
        "triton.language.constexpr",
    }:
        return all(_safe_module_value(arg, aliases) for arg in node.args) and all(
            keyword.arg is not None and _safe_module_value(keyword.value, aliases)
            for keyword in node.keywords
        )
    return False


def _target_names(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for item in node.elts:
            names.update(_target_names(item))
        return names
    if isinstance(node, ast.Starred):
        return _target_names(node.value)
    return set()


def _reject_callable_shadowing(
    tree: ast.Module,
    functions: dict[str, ast.FunctionDef],
    relative_path: str,
    errors: list[str],
) -> None:
    """Keep every statically certified callable bound to its declared body."""
    protected = set(functions)

    def record(node: ast.AST, names: set[str]) -> None:
        for name in sorted(names & protected):
            errors.append(
                f"{relative_path}:{getattr(node, 'lineno', 1)}: "
                f"forbidden certified-callable shadowing: {name!r}"
            )

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                record(target, _target_names(target))
        elif isinstance(node, ast.AnnAssign):
            record(node.target, _target_names(node.target))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            record(
                node,
                {
                    item.asname or item.name.split(".", 1)[0]
                    for item in node.names
                },
            )

    for function in functions.values():
        for node in ast.walk(function):
            if isinstance(node, ast.arg):
                record(node, {node.arg})
            elif isinstance(node, ast.Name) and isinstance(
                node.ctx, (ast.Store, ast.Del)
            ):
                record(node, {node.id})
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                record(
                    node,
                    {
                        item.asname or item.name.split(".", 1)[0]
                        for item in node.names
                    },
                )
            elif isinstance(node, ast.ExceptHandler) and node.name is not None:
                record(node, {node.name})
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node is not function:
                    record(node, {node.name})


class _Kind(Enum):
    SCALAR = auto()
    METADATA = auto()
    UNKNOWN = auto()
    TENSOR = auto()
    DATA = auto()


@dataclass
class _FunctionResult:
    calls: set[str]
    launches: set[str]
    call_node_ids: set[int]
    return_node_ids: set[int]
    falls_through: bool


_UNKNOWN_LITERAL = object()


class _Truth(Enum):
    FALSE = auto()
    TRUE = auto()
    UNKNOWN = auto()


def _literal_value(node: ast.AST) -> object:
    """Evaluate only inert literals needed for executable-path analysis."""
    if isinstance(node, ast.Constant) and isinstance(
        node.value, (str, bytes, int, float, complex, bool, type(None))
    ):
        return node.value
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        values = [_literal_value(item) for item in node.elts]
        if any(value is _UNKNOWN_LITERAL for value in values):
            return _UNKNOWN_LITERAL
        try:
            if isinstance(node, ast.Tuple):
                return tuple(values)
            if isinstance(node, ast.List):
                return values
            return set(values)
        except (TypeError, ValueError):
            return _UNKNOWN_LITERAL
    if isinstance(node, ast.Dict):
        if any(key is None for key in node.keys):
            return _UNKNOWN_LITERAL
        keys = [_literal_value(key) for key in node.keys]
        values = [_literal_value(value) for value in node.values]
        if any(value is _UNKNOWN_LITERAL for value in (*keys, *values)):
            return _UNKNOWN_LITERAL
        try:
            return dict(zip(keys, values))
        except (TypeError, ValueError):
            return _UNKNOWN_LITERAL
    if isinstance(node, ast.UnaryOp):
        value = _literal_value(node.operand)
        if value is _UNKNOWN_LITERAL:
            return _UNKNOWN_LITERAL
        try:
            if isinstance(node.op, ast.Not):
                return not value
            if isinstance(node.op, ast.UAdd):
                return +value  # type: ignore[operator]
            if isinstance(node.op, ast.USub):
                return -value  # type: ignore[operator]
            if isinstance(node.op, ast.Invert):
                return ~value  # type: ignore[operator]
        except (TypeError, ValueError, OverflowError):
            return _UNKNOWN_LITERAL
    if isinstance(node, ast.BoolOp):
        for item in node.values[:-1]:
            value = _literal_value(item)
            if value is _UNKNOWN_LITERAL:
                return _UNKNOWN_LITERAL
            try:
                truth = bool(value)
            except (TypeError, ValueError):
                return _UNKNOWN_LITERAL
            if isinstance(node.op, ast.And) and not truth:
                return value
            if isinstance(node.op, ast.Or) and truth:
                return value
        return _literal_value(node.values[-1])
    if isinstance(node, ast.Compare):
        truth = _comparison_truth(node)
        if truth is _Truth.TRUE:
            return True
        if truth is _Truth.FALSE:
            return False
    return _UNKNOWN_LITERAL


def _comparison_pair_truth(
    left_node: ast.AST, operator: ast.cmpop, right_node: ast.AST
) -> _Truth:
    left = _literal_value(left_node)
    right = _literal_value(right_node)
    if left is _UNKNOWN_LITERAL or right is _UNKNOWN_LITERAL:
        return _Truth.UNKNOWN
    try:
        if isinstance(operator, ast.Eq):
            matches = left == right
        elif isinstance(operator, ast.NotEq):
            matches = left != right
        elif isinstance(operator, ast.Lt):
            matches = left < right  # type: ignore[operator]
        elif isinstance(operator, ast.LtE):
            matches = left <= right  # type: ignore[operator]
        elif isinstance(operator, ast.Gt):
            matches = left > right  # type: ignore[operator]
        elif isinstance(operator, ast.GtE):
            matches = left >= right  # type: ignore[operator]
        elif isinstance(operator, ast.In):
            matches = left in right  # type: ignore[operator]
        elif isinstance(operator, ast.NotIn):
            matches = left not in right  # type: ignore[operator]
        else:
            return _Truth.UNKNOWN
    except (TypeError, ValueError, OverflowError):
        return _Truth.UNKNOWN
    return _Truth.TRUE if matches else _Truth.FALSE


def _comparison_truth(node: ast.Compare) -> _Truth:
    saw_unknown = False
    left = node.left
    for operator, right in zip(node.ops, node.comparators):
        truth = _comparison_pair_truth(left, operator, right)
        if truth is _Truth.FALSE:
            return _Truth.FALSE
        saw_unknown = saw_unknown or truth is _Truth.UNKNOWN
        left = right
    return _Truth.UNKNOWN if saw_unknown else _Truth.TRUE


def _condition_truth(node: ast.AST) -> _Truth:
    """Return sound abstract truth without conflating it with operand values."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        truth = _condition_truth(node.operand)
        if truth is _Truth.TRUE:
            return _Truth.FALSE
        if truth is _Truth.FALSE:
            return _Truth.TRUE
        return _Truth.UNKNOWN
    if isinstance(node, ast.BoolOp):
        truths = [_condition_truth(value) for value in node.values]
        if isinstance(node.op, ast.And):
            if _Truth.FALSE in truths:
                return _Truth.FALSE
            return (
                _Truth.TRUE
                if all(item is _Truth.TRUE for item in truths)
                else _Truth.UNKNOWN
            )
        if _Truth.TRUE in truths:
            return _Truth.TRUE
        return (
            _Truth.FALSE
            if all(item is _Truth.FALSE for item in truths)
            else _Truth.UNKNOWN
        )
    if isinstance(node, ast.Compare):
        return _comparison_truth(node)
    value = _literal_value(node)
    if value is _UNKNOWN_LITERAL:
        return _Truth.UNKNOWN
    try:
        return _Truth.TRUE if bool(value) else _Truth.FALSE
    except (TypeError, ValueError):
        return _Truth.UNKNOWN


def _literal_truth(node: ast.AST) -> bool | None:
    truth = _condition_truth(node)
    if truth is _Truth.TRUE:
        return True
    if truth is _Truth.FALSE:
        return False
    return None


_ALLOWED_HOST_STATEMENTS = (
    ast.AnnAssign,
    ast.Assert,
    ast.Assign,
    ast.Expr,
    ast.If,
    ast.Pass,
    ast.Return,
)
_ALLOWED_HOST_EXPRESSIONS = (
    ast.Attribute,
    ast.BinOp,
    ast.BoolOp,
    ast.Call,
    ast.Compare,
    ast.Constant,
    ast.Dict,
    ast.IfExp,
    ast.List,
    ast.Name,
    ast.Set,
    ast.Slice,
    ast.Subscript,
    ast.Tuple,
    ast.UnaryOp,
)


def _reject_unsupported_host_syntax(
    function: ast.FunctionDef,
    relative_path: str,
    errors: list[str],
) -> None:
    """Enforce the small host subset; Triton JIT bodies are scanned elsewhere."""
    for statement in function.body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Compare) and any(
                isinstance(operator, (ast.Is, ast.IsNot))
                for operator in node.ops
            ):
                errors.append(
                    f"{relative_path}:{node.lineno}: forbidden host identity comparison"
                )
            elif isinstance(node, ast.stmt) and not isinstance(
                node, _ALLOWED_HOST_STATEMENTS
            ):
                errors.append(
                    f"{relative_path}:{getattr(node, 'lineno', function.lineno)}: "
                    f"forbidden host syntax: {type(node).__name__!r}"
                )
            elif isinstance(node, ast.expr) and not isinstance(
                node, _ALLOWED_HOST_EXPRESSIONS
            ):
                errors.append(
                    f"{relative_path}:{getattr(node, 'lineno', function.lineno)}: "
                    f"forbidden host expression: {type(node).__name__!r}"
                )


class _ReachableCallVisitor(ast.NodeVisitor):
    def __init__(self, host_functions: set[str], kernels: set[str]) -> None:
        self.host_functions = host_functions
        self.kernels = kernels
        self.calls: set[str] = set()
        self.launches: set[str] = set()
        self.call_node_ids: set[int] = set()
        self.return_node_ids: set[int] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
        return

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        launch = _kernel_launch(node, self.kernels)
        if launch is not None:
            self.launches.add(launch)
            for item in ast.walk(node.func.slice):
                if isinstance(item, ast.Name) and item.id in self.host_functions:
                    self.calls.add(item.id)
                    self.call_node_ids.add(id(item))
        elif isinstance(node.func, ast.Name) and node.func.id in self.host_functions:
            self.calls.add(node.func.id)
            self.call_node_ids.add(id(node))
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:  # noqa: N802
        for value in node.values:
            self.visit(value)
            truth = _literal_truth(value)
            if isinstance(node.op, ast.And) and truth is False:
                break
            if isinstance(node.op, ast.Or) and truth is True:
                break

    def visit_Compare(self, node: ast.Compare) -> None:  # noqa: N802
        self.visit(node.left)
        left = node.left
        for operator, right in zip(node.ops, node.comparators):
            self.visit(right)
            if _comparison_pair_truth(left, operator, right) is _Truth.FALSE:
                break
            left = right

    def visit_IfExp(self, node: ast.IfExp) -> None:  # noqa: N802
        self.visit(node.test)
        truth = _literal_truth(node.test)
        if truth is True:
            self.visit(node.body)
        elif truth is False:
            self.visit(node.orelse)
        else:
            self.visit(node.body)
            self.visit(node.orelse)


def _reachable_function_result(
    function: ast.FunctionDef,
    host_functions: set[str],
    kernels: set[str],
) -> _FunctionResult:
    visitor = _ReachableCallVisitor(host_functions, kernels)

    def visit_block(statements: list[ast.stmt]) -> bool:
        falls_through = True
        for statement in statements:
            if not falls_through:
                break
            falls_through = visit_statement(statement)
        return falls_through

    def visit_statement(statement: ast.stmt) -> bool:
        if isinstance(statement, ast.Return):
            visitor.return_node_ids.add(id(statement))
            if statement.value is not None:
                visitor.visit(statement.value)
            return False
        if isinstance(statement, ast.Raise):
            if statement.exc is not None:
                visitor.visit(statement.exc)
            if statement.cause is not None:
                visitor.visit(statement.cause)
            return False
        if isinstance(statement, (ast.Break, ast.Continue)):
            return False
        if isinstance(statement, ast.If):
            visitor.visit(statement.test)
            truth = _literal_truth(statement.test)
            if truth is True:
                return visit_block(statement.body)
            if truth is False:
                return visit_block(statement.orelse)
            body_falls = visit_block(statement.body)
            else_falls = visit_block(statement.orelse) if statement.orelse else True
            return body_falls or else_falls
        if isinstance(statement, ast.While):
            visitor.visit(statement.test)
            truth = _literal_truth(statement.test)
            if truth is False:
                return visit_block(statement.orelse)
            visit_block(statement.body)
            visit_block(statement.orelse)
            return True
        if isinstance(statement, (ast.For, ast.AsyncFor)):
            visitor.visit(statement.iter)
            empty = isinstance(statement.iter, (ast.Tuple, ast.List, ast.Set)) and not statement.iter.elts
            if not empty:
                visit_block(statement.body)
            visit_block(statement.orelse)
            return True
        if isinstance(statement, (ast.With, ast.AsyncWith)):
            for item in statement.items:
                visitor.visit(item.context_expr)
            return visit_block(statement.body)
        if isinstance(statement, ast.Try):
            body_falls = visit_block(statement.body)
            handler_falls = [visit_block(handler.body) for handler in statement.handlers]
            else_falls = visit_block(statement.orelse) if statement.orelse else body_falls
            final_falls = visit_block(statement.finalbody) if statement.finalbody else True
            return final_falls and (else_falls or any(handler_falls))
        if isinstance(statement, ast.Assert):
            visitor.visit(statement.test)
            if statement.msg is not None:
                visitor.visit(statement.msg)
            return _literal_truth(statement.test) is not False
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return True
        visitor.visit(statement)
        return True

    falls_through = visit_block(function.body)
    return _FunctionResult(
        visitor.calls,
        visitor.launches,
        visitor.call_node_ids,
        visitor.return_node_ids,
        falls_through,
    )


def _join_kinds(*values: _Kind) -> _Kind:
    for kind in (
        _Kind.DATA,
        _Kind.TENSOR,
        _Kind.UNKNOWN,
        _Kind.METADATA,
        _Kind.SCALAR,
    ):
        if kind in values:
            return kind
    return _Kind.SCALAR


def _computed_kind(values: list[_Kind]) -> _Kind:
    joined = _join_kinds(*values)
    if joined in {_Kind.TENSOR, _Kind.DATA}:
        return _Kind.DATA
    return joined


def _call_parameter_kinds(
    function: ast.FunctionDef,
    call: ast.Call,
    kind_of: Any,
) -> dict[str, _Kind]:
    positional = [*function.args.posonlyargs, *function.args.args]
    keyword_only = list(function.args.kwonlyargs)
    result = {
        argument.arg: _Kind.SCALAR for argument in (*positional, *keyword_only)
    }
    if function.args.vararg is not None:
        result[function.args.vararg.arg] = _Kind.SCALAR
    if function.args.kwarg is not None:
        result[function.args.kwarg.arg] = _Kind.SCALAR

    position = 0
    for argument in call.args:
        if isinstance(argument, ast.Starred):
            kind = kind_of(argument.value)
            for parameter in positional[position:]:
                result[parameter.arg] = _join_kinds(result[parameter.arg], kind)
            if function.args.vararg is not None:
                name = function.args.vararg.arg
                result[name] = _join_kinds(result[name], kind)
            position = len(positional)
            continue
        kind = kind_of(argument)
        if position < len(positional):
            name = positional[position].arg
        elif function.args.vararg is not None:
            name = function.args.vararg.arg
        else:
            position += 1
            continue
        result[name] = _join_kinds(result[name], kind)
        position += 1

    named = {argument.arg for argument in (*positional, *keyword_only)}
    for keyword in call.keywords:
        kind = kind_of(keyword.value)
        if keyword.arg is None:
            for name in named:
                result[name] = _join_kinds(result[name], kind)
            if function.args.kwarg is not None:
                name = function.args.kwarg.arg
                result[name] = _join_kinds(result[name], kind)
        elif keyword.arg in named:
            result[keyword.arg] = _join_kinds(result[keyword.arg], kind)
        elif function.args.kwarg is not None:
            name = function.args.kwarg.arg
            result[name] = _join_kinds(result[name], kind)
    return result


class _PolicyVisitor(ast.NodeVisitor):
    def __init__(
        self, relative_path: str, aliases: dict[str, str], errors: list[str]
    ) -> None:
        self.relative_path = relative_path
        self.aliases = aliases
        self.errors = errors

    def _error(self, node: ast.AST, category: str, token: object = "") -> None:
        line = getattr(node, "lineno", 1)
        self.errors.append(
            f"{self.relative_path}:{line}: forbidden {category}: {token!r}"
        )

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        lowered = node.attr.lower()
        if node.attr.startswith("__") or node.attr.endswith("__"):
            self._error(node, "dunder/frame introspection", node.attr)
        if lowered in {"tb_frame", "f_back", "f_builtins", "f_globals", "f_locals"}:
            self._error(node, "traceback/frame traversal", node.attr)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if node.id.startswith("__") or node.id.endswith("__"):
            self._error(node, "dunder/frame introspection", node.id)

    def _reject_forbidden_call(self, node: ast.Call) -> None:
        """Deny interpreter-mutation and dynamic-loading calls anywhere."""
        if isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_BUILTIN_CALLS:
            self._error(node, "dynamic code or interpreter mutation", node.func.id)
        name = _dotted_name(node.func, self.aliases)
        if name is None:
            return
        lowered = name.lower()
        if lowered in FORBIDDEN_BUILTIN_CALLS:
            self._error(node, "dynamic code or interpreter mutation", name)
        if lowered in FORBIDDEN_EXACT_CALLS:
            self._error(node, "dynamic code or file loading", name)
        if lowered.rsplit(".", 1)[-1] in FORBIDDEN_PROCESS_OR_LOAD_TAILS:
            self._error(node, "process or library loading", name)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        self._reject_forbidden_call(node)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:  # noqa: N802
        if not isinstance(node.value, str):
            return
        lowered = node.value.strip().lower()
        stripped = node.value.strip()
        if stripped.startswith("__") or stripped.endswith("__"):
            self._error(node, "dunder/frame introspection", stripped)
        marker = next(
            (item for item in FORBIDDEN_LITERAL_MARKERS if item in lowered), None
        )
        if marker is not None:
            self._error(node, "trusted/external literal", marker)

    def visit_Global(self, node: ast.Global) -> None:  # noqa: N802
        self._error(node, "global state", node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:  # noqa: N802
        self._error(node, "nonlocal state", node.names)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        self._error(node, "class definition", node.name)
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:  # noqa: N802
        if isinstance(node.op, ast.MatMult):
            self._error(node, "existing GEMM operator", "@")
        self.generic_visit(node)

    def _import_root(self, node: ast.AST) -> str | None:
        current = node
        while isinstance(current, (ast.Attribute, ast.Subscript)):
            current = current.value
        if isinstance(current, ast.Name) and current.id in self.aliases:
            return self.aliases[current.id]
        return None

    def _check_import_mutation(self, target: ast.AST) -> None:
        if isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self._check_import_mutation(item)
            return
        root = self._import_root(target)
        if root is not None:
            self._error(target, "imported-module mutation", root)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        for target in node.targets:
            self._check_import_mutation(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        self._check_import_mutation(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
        self._check_import_mutation(node.target)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:  # noqa: N802
        for target in node.targets:
            self._check_import_mutation(target)
        self.generic_visit(node)


class _KernelVisitor(_PolicyVisitor):
    def __init__(
        self,
        relative_path: str,
        aliases: dict[str, str],
        errors: list[str],
        kernels: set[str],
    ) -> None:
        super().__init__(relative_path, aliases, errors)
        self.kernels = kernels

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        self._reject_forbidden_call(node)
        name = _dotted_name(node.func, self.aliases)
        allowed = False
        if name is not None:
            allowed = (
                name.startswith("triton.language.")
                or name in self.kernels
                or name in SAFE_HOST_BUILTINS
                or name == "triton.cdiv"
                or name.rsplit(".", 1)[-1] == "to"
            )
        if isinstance(node.func, ast.Attribute) and node.func.attr == "to":
            allowed = True
        if not allowed:
            self._error(node, "kernel call", name or type(node.func).__name__)
        self.generic_visit(node)


class _HostVisitor(_PolicyVisitor):
    def __init__(
        self,
        relative_path: str,
        aliases: dict[str, str],
        errors: list[str],
        function: ast.FunctionDef,
        functions: dict[str, ast.FunctionDef],
        host_functions: set[str],
        kernels: set[str],
        initial_kinds: dict[str, _Kind],
        return_summaries: dict[str, _Kind],
    ) -> None:
        super().__init__(relative_path, aliases, errors)
        self.functions = functions
        self.host_functions = host_functions
        self.kernels = kernels
        self.return_summaries = return_summaries
        self.calls: set[str] = set()
        self.launches: set[str] = set()
        self.call_bindings: dict[int, tuple[str, dict[str, _Kind]]] = {}
        self.return_node_kinds: dict[int, _Kind] = {}
        arguments = [
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        ]
        if function.args.vararg is not None:
            arguments.append(function.args.vararg)
        if function.args.kwarg is not None:
            arguments.append(function.args.kwarg)
        self.kinds = {
            argument.arg: initial_kinds.get(argument.arg, _Kind.SCALAR)
            for argument in arguments
        }

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._error(node, "nested function", node.name)

    def visit_Return(self, node: ast.Return) -> None:  # noqa: N802
        kind = _Kind.SCALAR if node.value is None else self._kind(node.value)
        self.return_node_kinds[id(node)] = kind
        if node.value is not None:
            self.visit(node.value)

    def _kind(self, node: ast.AST) -> _Kind:
        if isinstance(node, ast.Name):
            return self.kinds.get(node.id, _Kind.SCALAR)
        if isinstance(node, ast.Constant):
            return _Kind.SCALAR
        if isinstance(node, ast.Attribute):
            base = self._kind(node.value)
            if base is _Kind.TENSOR and node.attr in SAFE_TENSOR_ATTRIBUTES:
                return _Kind.METADATA
            return _Kind.DATA if base in {_Kind.TENSOR, _Kind.DATA} else base
        if isinstance(node, ast.Subscript):
            base = self._kind(node.value)
            if base is _Kind.METADATA:
                return _Kind.METADATA
            if base is _Kind.UNKNOWN:
                return _Kind.UNKNOWN
            return _Kind.DATA if base in {_Kind.TENSOR, _Kind.DATA} else _Kind.SCALAR
        if isinstance(node, ast.Call):
            name = _dotted_name(node.func, self.aliases)
            if name in SAFE_TORCH_ALLOCATORS:
                return _Kind.TENSOR
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in SAFE_TENSOR_METADATA
            ):
                base = self._kind(node.func.value)
                if base is _Kind.TENSOR:
                    return _Kind.METADATA
                if base is _Kind.UNKNOWN:
                    return _Kind.UNKNOWN
            if _kernel_launch(node, self.kernels) is not None:
                return _Kind.SCALAR
            if name in self.host_functions:
                return self.return_summaries.get(str(name), _Kind.UNKNOWN)
            if name in SAFE_HOST_BUILTINS | SAFE_HOST_MATH | {"triton.cdiv"}:
                kinds = [
                    self._kind(value)
                    for value in (*node.args, *(item.value for item in node.keywords))
                ]
                return _computed_kind(kinds)
            return _Kind.DATA
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            kinds = [self._kind(item) for item in node.elts]
        elif isinstance(node, ast.Dict):
            kinds = [self._kind(item) for item in node.values]
        elif isinstance(node, ast.UnaryOp):
            kinds = [self._kind(node.operand)]
        elif isinstance(node, (ast.BinOp, ast.BoolOp, ast.Compare, ast.IfExp)):
            kinds = [
                self._kind(child)
                for child in ast.iter_child_nodes(node)
                if isinstance(child, ast.expr)
            ]
        else:
            return _Kind.SCALAR
        return _computed_kind(kinds)

    def _bind(self, target: ast.AST, kind: _Kind) -> None:
        if isinstance(target, ast.Name):
            self.kinds[target.id] = _join_kinds(
                self.kinds.get(target.id, _Kind.SCALAR), kind
            )
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self._bind(item, kind)

    def _reject_mutating_target(self, target: ast.AST) -> None:
        if isinstance(target, (ast.Attribute, ast.Subscript)):
            self._error(target, "host/imported-module mutation", ast.unparse(target))
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self._reject_mutating_target(item)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        kind = self._kind(node.value)
        for target in node.targets:
            self._reject_mutating_target(target)
            self._bind(target, kind)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        kind = self._kind(node.value) if node.value is not None else _Kind.SCALAR
        self._reject_mutating_target(node.target)
        self._bind(node.target, kind)
        if node.value is not None:
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
        self._reject_mutating_target(node.target)
        if self._kind(node.target) in {_Kind.TENSOR, _Kind.DATA}:
            self._error(node, "host tensor mutation", ast.unparse(node.target))
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:  # noqa: N802
        for target in node.targets:
            self._reject_mutating_target(target)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:  # noqa: N802
        if self._kind(node.value) in {_Kind.TENSOR, _Kind.DATA}:
            self._error(node, "host tensor data access", ast.unparse(node))
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        base = self._kind(node.value)
        if base in {_Kind.TENSOR, _Kind.DATA} and node.attr not in (
            SAFE_TENSOR_ATTRIBUTES | SAFE_TENSOR_METADATA
        ):
            self._error(node, "host tensor data access", node.attr)
        super().visit_Attribute(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:  # noqa: N802
        if self._kind(node) is _Kind.DATA:
            self._error(node, "host tensor computation", ast.unparse(node))
        super().visit_BinOp(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:  # noqa: N802
        if self._kind(node) is _Kind.DATA:
            self._error(node, "host tensor computation", ast.unparse(node))
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:  # noqa: N802
        if self._kind(node.test) in {_Kind.TENSOR, _Kind.DATA}:
            self._error(node, "host tensor control flow", ast.unparse(node.test))
        self.visit(node.test)
        incoming = dict(self.kinds)
        for statement in node.body:
            self.visit(statement)
        body_kinds = dict(self.kinds)
        self.kinds = dict(incoming)
        for statement in node.orelse:
            self.visit(statement)
        else_kinds = dict(self.kinds)
        self.kinds = {
            name: _join_kinds(
                incoming.get(name, _Kind.SCALAR),
                body_kinds.get(name, _Kind.SCALAR),
                else_kinds.get(name, _Kind.SCALAR),
            )
            for name in incoming.keys() | body_kinds.keys() | else_kinds.keys()
        }

    def visit_While(self, node: ast.While) -> None:  # noqa: N802
        if self._kind(node.test) in {_Kind.TENSOR, _Kind.DATA}:
            self._error(node, "host tensor control flow", ast.unparse(node.test))
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        kind = self._kind(node.iter)
        if kind in {_Kind.TENSOR, _Kind.DATA}:
            self._error(node, "host tensor iteration", ast.unparse(node.iter))
        self._bind(node.target, kind)
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        kind = self._kind(node.iter)
        if kind in {_Kind.TENSOR, _Kind.DATA}:
            self._error(node, "host tensor iteration", ast.unparse(node.iter))
        self._bind(node.target, kind)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:  # noqa: N802
        if self._kind(node) is _Kind.DATA:
            self._error(node, "host tensor comparison", ast.unparse(node))
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:  # noqa: N802
        if self._kind(node) is _Kind.DATA:
            self._error(node, "host tensor boolean computation", ast.unparse(node))
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:  # noqa: N802
        if self._kind(node.test) in {_Kind.TENSOR, _Kind.DATA}:
            self._error(node, "host tensor control flow", ast.unparse(node.test))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        self._reject_forbidden_call(node)
        launch = _kernel_launch(node, self.kernels)
        if launch is not None:
            self.launches.add(launch)
            # Triton calls a callable grid in-process at launch, so a host
            # function named in the grid slot is a real execution route and
            # must be policed like any other callee.  Its only argument is the
            # meta mapping of compile-time constants, hence SCALAR bindings.
            for item in ast.walk(node.func.slice):
                if isinstance(item, ast.Name) and item.id in self.host_functions:
                    self.calls.add(item.id)
                    callee = self.functions[item.id]
                    self.call_bindings[id(item)] = (
                        item.id,
                        {
                            argument.arg: _Kind.SCALAR
                            for argument in (
                                *callee.args.posonlyargs,
                                *callee.args.args,
                                *callee.args.kwonlyargs,
                            )
                        },
                    )
            self.visit(node.func.slice)
            for value in (*node.args, *(item.value for item in node.keywords)):
                self.visit(value)
            return

        # A callable recovered with ``mapping[key]`` is never a reviewed host
        # or Triton route. This closes __builtins__ and module.__dict__ paths.
        if isinstance(node.func, ast.Subscript):
            self._error(node, "subscript call route", ast.unparse(node.func))
            self.generic_visit(node)
            return

        name = _dotted_name(node.func, self.aliases)
        allowed = False
        if name in self.host_functions:
            self.calls.add(str(name))
            self.call_bindings[id(node)] = (
                str(name),
                _call_parameter_kinds(self.functions[str(name)], node, self._kind),
            )
            allowed = True
        elif name in SAFE_TORCH_ALLOCATORS:
            kinds = [
                self._kind(value)
                for value in (*node.args, *(item.value for item in node.keywords))
            ]
            tensor_positions = {
                index
                for index, kind in enumerate(kinds)
                if kind in {_Kind.TENSOR, _Kind.DATA}
            }
            allowed = name.endswith("_like") and tensor_positions <= {0}
            if not name.endswith("_like"):
                allowed = not tensor_positions
        elif name in SAFE_HOST_BUILTINS | SAFE_HOST_MATH | {"triton.cdiv"}:
            kinds = [
                self._kind(value)
                for value in (*node.args, *(item.value for item in node.keywords))
            ]
            allowed = not any(kind in {_Kind.TENSOR, _Kind.DATA} for kind in kinds)
        elif isinstance(node.func, ast.Attribute):
            allowed = (
                self._kind(node.func.value) is _Kind.TENSOR
                and node.func.attr in SAFE_TENSOR_METADATA
            )
        if not allowed:
            self._error(node, "host call", name or type(node.func).__name__)
        self.generic_visit(node)


def _scan_selected_source(text: str, relative_path: str) -> list[str]:
    try:
        tree = ast.parse(text, filename=relative_path)
    except SyntaxError as exc:
        return [
            f"{relative_path}:{exc.lineno or 1}: Python syntax error: {exc.msg}"
        ]

    errors: list[str] = []
    aliases: dict[str, str] = {}
    functions: dict[str, ast.FunctionDef] = {}
    kernels: set[str] = set()

    # Import validation is allowlist-based so unknown filesystem, process,
    # network, extension, and trusted-harness modules fail closed.
    for index, node in enumerate(tree.body):
        if (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name not in ALLOWED_IMPORTS:
                    errors.append(
                        f"{relative_path}:{node.lineno}: forbidden import: {item.name!r}"
                    )
                expected_alias = "tl" if item.name == "triton.language" else None
                if item.asname != expected_alias:
                    errors.append(
                        f"{relative_path}:{node.lineno}: forbidden import alias: {item.asname!r}"
                    )
                aliases[item.asname or item.name.split(".", 1)[0]] = item.name
        elif isinstance(node, ast.ImportFrom):
            future = (
                node.level == 0
                and node.module == "__future__"
                and len(node.names) == 1
                and node.names[0].name == "annotations"
                and node.names[0].asname is None
            )
            triton_language = (
                node.level == 0
                and node.module == "triton"
                and len(node.names) == 1
                and node.names[0].name == "language"
                and node.names[0].asname == "tl"
            )
            if not (future or triton_language):
                errors.append(
                    f"{relative_path}:{node.lineno}: forbidden from-import: {node.module!r}"
                )
            if triton_language:
                aliases["tl"] = "triton.language"
        elif isinstance(node, ast.AsyncFunctionDef):
            errors.append(
                f"{relative_path}:{node.lineno}: forbidden async function: {node.name!r}"
            )
        elif isinstance(node, ast.FunctionDef):
            if node.name in functions:
                errors.append(
                    f"{relative_path}:{node.lineno}: duplicate function: {node.name!r}"
                )
            functions[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            # Checked after aliases have been collected.
            continue
        else:
            errors.append(
                f"{relative_path}:{getattr(node, 'lineno', 1)}: forbidden module statement: {type(node).__name__!r}"
            )

    top_level_imports = {
        id(node) for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and id(node) not in top_level_imports:
            errors.append(f"{relative_path}:{node.lineno}: forbidden nested import")

    for function in functions.values():
        decorator_names = {
            _decorator_name(decorator, aliases) for decorator in function.decorator_list
        }
        if "triton.jit" in decorator_names:
            kernels.add(function.name)
        for decorator in function.decorator_list:
            name = _decorator_name(decorator, aliases)
            if name not in ALLOWED_KERNEL_DECORATORS:
                errors.append(
                    f"{relative_path}:{decorator.lineno}: forbidden decorator: {name!r}"
                )
            for call in (
                item for item in ast.walk(decorator) if isinstance(item, ast.Call)
            ):
                call_name = _dotted_name(call.func, aliases)
                if call_name not in ALLOWED_KERNEL_DECORATORS | {"triton.Config"}:
                    errors.append(
                        f"{relative_path}:{call.lineno}: forbidden decorator call: {call_name!r}"
                    )
            _reject_unsafe_decorator_arguments(
                decorator, name, aliases, functions, relative_path, errors
            )
        arguments = [
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        ]
        for annotation in (
            *(
                argument.annotation
                for argument in arguments
                if argument.annotation is not None
            ),
            *((function.returns,) if function.returns is not None else ()),
        ):
            if not _safe_annotation(annotation, aliases):
                errors.append(
                    f"{relative_path}:{annotation.lineno}: forbidden function annotation"
                )
        defaults = [
            *function.args.defaults,
            *(item for item in function.args.kw_defaults if item is not None),
        ]
        if defaults:
            errors.append(
                f"{relative_path}:{defaults[0].lineno}: forbidden function default"
            )

    _reject_callable_shadowing(tree, functions, relative_path, errors)

    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if any(not isinstance(target, ast.Name) for target in targets):
            errors.append(
                f"{relative_path}:{node.lineno}: forbidden module mutation target"
            )
        if value is None or not _safe_module_value(value, aliases):
            errors.append(
                f"{relative_path}:{node.lineno}: forbidden module state"
            )

    target = functions.get("quantized_attention_bf16")
    if target is None:
        errors.append(f"{relative_path}:1: missing quantized_attention_bf16 entrypoint")
    else:
        parameters = tuple(argument.arg for argument in target.args.args)
        if (
            parameters != EXPECTED_PARAMETERS
            or target.args.posonlyargs
            or target.args.vararg is not None
            or target.args.kwarg is not None
            or target.args.kwonlyargs
            or target.args.defaults
            or any(item is not None for item in target.args.kw_defaults)
        ):
            errors.append(
                f"{relative_path}:{target.lineno}: entrypoint parameters must exactly equal {EXPECTED_PARAMETERS!r}"
            )
        for returned in (
            item for item in ast.walk(target) if isinstance(item, ast.Return)
        ):
            if returned.value is not None and not (
                isinstance(returned.value, ast.Constant) and returned.value.value is None
            ):
                errors.append(
                    f"{relative_path}:{returned.lineno}: entrypoint must return None"
                )

    _PolicyVisitor(relative_path, aliases, errors).visit(tree)
    host_functions = set(functions) - kernels
    for name in sorted(host_functions):
        _reject_unsupported_host_syntax(functions[name], relative_path, errors)
    results = {
        name: _reachable_function_result(function, host_functions, kernels)
        for name, function in functions.items()
        if name in host_functions
    }
    for name, function in functions.items():
        if name in kernels:
            visitor = _KernelVisitor(
                relative_path, aliases, errors, kernels
            )
            for statement in function.body:
                visitor.visit(statement)

    if target is not None and target.name in host_functions:
        initial = {
            argument.arg: (
                _Kind.SCALAR
                if argument.arg in {"is_causal", "softmax_scale"}
                else _Kind.TENSOR
            )
            for argument in (
                *target.args.posonlyargs,
                *target.args.args,
                *target.args.kwonlyargs,
            )
        }
        inferred: dict[str, dict[str, _Kind]] = {target.name: initial}
        return_summaries: dict[str, _Kind] = {}
        converged = False
        max_rounds = max(8, len(host_functions) * 8)
        for _ in range(max_rounds):
            next_inferred: dict[str, dict[str, _Kind]] = {target.name: initial}
            next_returns: dict[str, _Kind] = {}
            for name in sorted(inferred):
                result = results.get(name)
                if result is None:
                    continue
                visitor = _HostVisitor(
                    relative_path,
                    aliases,
                    [],
                    functions[name],
                    functions,
                    host_functions,
                    kernels,
                    inferred[name],
                    return_summaries,
                )
                for statement in functions[name].body:
                    visitor.visit(statement)

                return_kinds = [
                    kind
                    for node_id, kind in visitor.return_node_kinds.items()
                    if node_id in result.return_node_ids
                ]
                if result.falls_through:
                    return_kinds.append(_Kind.SCALAR)
                next_returns[name] = (
                    _join_kinds(*return_kinds)
                    if return_kinds
                    else _Kind.UNKNOWN
                )

                for node_id, (callee, binding) in visitor.call_bindings.items():
                    if node_id not in result.call_node_ids:
                        continue
                    previous = next_inferred.get(callee)
                    if previous is None:
                        next_inferred[callee] = dict(binding)
                        continue
                    merged: dict[str, _Kind] = {}
                    for parameter in previous.keys() | binding.keys():
                        if parameter in previous and parameter in binding:
                            merged[parameter] = _join_kinds(
                                previous[parameter], binding[parameter]
                            )
                        elif parameter in previous:
                            merged[parameter] = previous[parameter]
                        else:
                            merged[parameter] = binding[parameter]
                    next_inferred[callee] = merged

            for name in next_inferred:
                next_returns.setdefault(name, _Kind.UNKNOWN)
            if next_inferred == inferred and next_returns == return_summaries:
                inferred = next_inferred
                return_summaries = next_returns
                converged = True
                break
            inferred = next_inferred
            return_summaries = next_returns

        if not converged:
            errors.append(
                f"{relative_path}:{target.lineno}: host parameter/return summaries did not converge"
            )
        for name in sorted(inferred):
            if return_summaries.get(name, _Kind.UNKNOWN) is _Kind.UNKNOWN:
                errors.append(
                    f"{relative_path}:{functions[name].lineno}: "
                    f"unknown recursive or unstable host return: {name!r}"
                )

        # Every host function must be reachable from the entrypoint.  An
        # unreachable one is never given kind bindings, so the policy pass below
        # would skip it entirely while Triton decorator and grid routes can
        # still execute it.
        for name in sorted(host_functions - set(inferred)):
            errors.append(
                f"{relative_path}:{functions[name].lineno}: "
                f"unreachable host function: {name!r}"
            )

        for name in sorted(inferred):
            visitor = _HostVisitor(
                relative_path,
                aliases,
                errors,
                functions[name],
                functions,
                host_functions,
                kernels,
                inferred[name],
                return_summaries,
            )
            for statement in functions[name].body:
                visitor.visit(statement)

    if target is not None:
        pending = [target.name]
        seen: set[str] = set()
        launches: set[str] = set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            result = results.get(name)
            if result is None:
                continue
            launches.update(result.launches)
            pending.extend(sorted(result.calls - seen))
        if not launches:
            errors.append(
                f"{relative_path}:{target.lineno}: entrypoint call graph must launch a declared @triton.jit kernel"
            )

    return list(dict.fromkeys(errors))


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
