#!/usr/bin/env python3
"""Fail-closed AST admission for the MInference Python/Triton submission.

This scanner enforces the documented source policy for ordinary submitted
code.  It is a policy boundary, not a Python sandbox; the Judge must still run
admitted code in the separately hardened worker and image boundary.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ENTRYPOINT = "minference_sparse_prefill_bf16"
ENTRYPOINT_ARGUMENTS = (
    "q",
    "k",
    "v",
    "softmax_scale",
    "out",
)
MAX_SOURCE_BYTES = 512 * 1024
MAX_NOTES_BYTES = 64 * 1024
MAX_WORKSPACE_BYTES = MAX_SOURCE_BYTES + MAX_NOTES_BYTES
MAX_AST_NODES = 50_000
MAX_AST_DEPTH = 256
MAX_LITERAL_BYTES = 64 * 1024

ALLOWED_IMPORT_ROOTS = {"__future__", "math", "torch", "triton", "typing"}
# Imports expose Python modules, including their transitive imports and private
# native operators.  A root-name allowlist therefore cannot define the admitted
# numerical surface.  Only these exact public APIs/namespaces may be referenced;
# aliases are resolved to the same names by the AST visitor below.  Extending
# this surface requires reviewing an API, rather than inheriting new APIs from
# an installed Torch/Triton version.
TORCH_NUMERICAL_APIS = set("""
abs absolute acos acosh add addbmm addmm addmv addr all amax amin aminmax any
arange argmax argmin argsort as_strided as_tensor asin asinh atan atan2 atanh
baddbmm bmm broadcast_to bucketize cat ceil chunk clamp clamp_max clamp_min
clone concat concatenate contiguous cos cosh count_nonzero cummax cummin
cumprod cumsum diag diag_embed diagonal diff div dot einsum empty empty_like
empty_strided eq equal erf erfc exp exp2 expm1 expand_copy eye finfo flatten
flip floor floor_divide fmod full full_like gather ge greater greater_equal
gt hstack iinfo index_add index_copy index_fill index_select isclose isfinite
isinf isnan le lerp less less_equal linspace log log10 log1p log2 logaddexp
logaddexp2 logical_and logical_not logical_or logical_xor logsumexp lt masked_fill
masked_select matmul max maximum mean median min minimum mm movedim moveaxis
mul multiply mv nan_to_num narrow ne neg negative nonzero norm ones ones_like
outer permute pow prod reciprocal remainder repeat_interleave reshape roll
round rsqrt scatter scatter_add scatter_reduce searchsorted select sigmoid sign
sin sinh softmax sort split sqrt square squeeze stack std sub subtract sum
t take take_along_dim tan tanh tensor tile topk transpose tril triu trunc
unbind unique unique_consecutive unsqueeze var vdot vstack where zeros zeros_like
""".split())
TORCH_TENSOR_APIS = set("""
add_ as_strided bmm clone contiguous copy_ dim expand fill_ flatten float
index_select masked_fill_ matmul mm mul_ numel permute repeat reshape size
softmax sort squeeze stride sub_ to transpose unsqueeze view zero_
""".split())
TORCH_FUNCTIONAL_APIS = {"log_softmax", "normalize", "pad", "softmax"}
TORCH_LINALG_APIS = set("""
cholesky cholesky_ex cond cross det diagonal eig eigh eigvals eigvalsh inv
inv_ex lstsq matmul matrix_norm matrix_power matrix_rank multi_dot norm pinv
qr slogdet solve solve_ex solve_triangular svd svdvals tensorinv tensorsolve
vecdot vector_norm
""".split())
TORCH_VALUES = set("""
Tensor bfloat16 bool complex64 complex128 device dtype float16 float32 float64
half inf int8 int16 int32 int64 long nan short uint8
""".split())
TRITON_NUMERICAL_APIS = {"cdiv", "jit", "next_power_of_2"}
TRITON_LANGUAGE_APIS = set("""
abs advance arange argmax argmin associative_scan assume atomic_add atomic_and
atomic_cas atomic_max atomic_min atomic_or atomic_xchg atomic_xor broadcast
broadcast_to cast cat cdiv ceil clamp cos cumprod cumsum div_rn dot dot_scaled
exp exp2 expand_dims fdiv flip floor fma full gather histogram interleave join
load log log2 make_block_ptr max max_contiguous maximum min minimum multiple_of
num_programs permute program_id range ravel reduce reshape rsqrt sigmoid sin
sort split sqrt sqrt_rn static_range store sum swizzle2d trans where xor_sum zeros
zeros_like
""".split())
TRITON_LANGUAGE_MATH_APIS = set("""
abs ceil cos div_rn erf exp exp2 floor fma log log2 rsqrt sin sqrt sqrt_rn tanh
""".split())
TRITON_LANGUAGE_VALUES = set("""
bfloat16 constexpr float16 float32 float64 float8e4b15 float8e4nv float8e5
int1 int8 int16 int32 int64 uint8 uint16 uint32 uint64
""".split())
MATH_NUMERICAL_APIS = set("""
acos acosh asin asinh atan atan2 atanh ceil comb copysign cos cosh degrees dist
erf erfc exp exp2 expm1 fabs factorial floor fmod frexp fsum gamma gcd hypot
isclose isfinite isinf isnan isqrt lcm ldexp lgamma log log10 log1p log2
modf nextafter perm pow prod radians remainder sin sinh sqrt tan tanh trunc ulp
""".split())
TYPING_VALUES = set("""
Any Callable Dict Final Iterable Iterator List Literal Mapping NamedTuple
Optional Protocol Sequence Set Tuple Type Union TYPE_CHECKING
""".split())
ALLOWED_NUMERICAL_CALLS = (
    {f"torch.{name}" for name in TORCH_NUMERICAL_APIS}
    | {f"torch.Tensor.{name}" for name in TORCH_TENSOR_APIS}
    | {f"torch.nn.functional.{name}" for name in TORCH_FUNCTIONAL_APIS}
    | {f"torch.linalg.{name}" for name in TORCH_LINALG_APIS}
    | {"torch.Tensor", "torch.device"}
    | {f"triton.{name}" for name in TRITON_NUMERICAL_APIS}
    | {f"math.{name}" for name in MATH_NUMERICAL_APIS}
    | {"typing.cast"}
)
for _language_prefix in ("tl", "triton.language"):
    ALLOWED_NUMERICAL_CALLS.update(
        f"{_language_prefix}.{name}" for name in TRITON_LANGUAGE_APIS
    )
    ALLOWED_NUMERICAL_CALLS.update(
        f"{_language_prefix}.math.{name}" for name in TRITON_LANGUAGE_MATH_APIS
    )
ALLOWED_NUMERICAL_REFERENCES = (
    ALLOWED_NUMERICAL_CALLS
    | {"torch", "torch.nn", "torch.nn.functional", "torch.linalg", "triton",
       "triton.language", "triton.language.math", "tl", "tl.math", "math", "typing"}
    | {f"torch.{name}" for name in TORCH_VALUES}
    | {f"math.{name}" for name in ("e", "inf", "nan", "pi", "tau")}
    | {f"typing.{name}" for name in TYPING_VALUES}
)
for _language_prefix in ("tl", "triton.language"):
    ALLOWED_NUMERICAL_REFERENCES.update(
        f"{_language_prefix}.{name}" for name in TRITON_LANGUAGE_VALUES
    )
BOUNDED_API_ROOTS = {"math", "torch", "triton", "tl", "typing"}
FORBIDDEN_PACKAGE_PREFIXES = (
    "flash_attn",
    "flashattention",
    "flashinfer",
    "minference",
    "sgl_kernel",
    "sglang",
    "vllm",
    "xformers",
)
# Public dense linear algebra (matmul, bmm, einsum, softmax, topk, ...) is
# admitted in v2: the candidate owns its pattern estimator.  Only private
# fused-GEMM entry points stay closed.
FORBIDDEN_MATH_TAILS = {
    "_int_mm",
    "_scaled_mm",
}
FORBIDDEN_ATTENTION_TAILS = {
    "_scaled_dot_product_cudnn_attention",
    "_scaled_dot_product_efficient_attention",
    "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_flash_attention_for_cpu",
    "_scaled_dot_product_fused_attention_overrideable",
    "flash_attn_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_varlen_func",
    "flash_attn_with_kvcache",
    "flash_mla_sparse_fwd",
    "scaled_dot_product_attention",
    "sdpa_kernel",
    "sparse_attn_prefill",
}
FORBIDDEN_DYNAMIC_NAMES = {
    "__import__",
    "compile",
    "delattr",
    "eval",
    "exec",
    "getattr",
    "globals",
    "locals",
    "setattr",
    "vars",
}
FORBIDDEN_NATIVE_EXACT = {
    "torch.compile",
    "torch.jit.load",
    "torch.jit.script",
    "torch.jit.trace",
    "torch.jit.trace_module",
    "torch.classes.load_library",
    "torch.ops.load_library",
    "triton.compile",
}
FORBIDDEN_NATIVE_TAILS = {
    "cdll",
    "dlopen",
    "load_inline",
    "load_library",
    "loadlibrary",
    "pydll",
}
FORBIDDEN_FILE_TAILS = {
    "from_file",
    "open",
    "read",
    "read_bytes",
    "read_text",
    "tofile",
    "write",
    "write_bytes",
    "write_text",
}
FORBIDDEN_CUDA_TAILS = {
    "current_stream",
    "default_stream",
    "event",
    "externalstream",
    "record_stream",
    "set_stream",
    "stream",
    "synchronize",
    "wait_event",
    "wait_stream",
}
FORBIDDEN_TIMING_TAILS = {
    "benchmark",
    "do_bench",
    "elapsed_time",
    "monotonic",
    "monotonic_ns",
    "perf_counter",
    "perf_counter_ns",
    "process_time",
    "time",
    "time_ns",
}
FORBIDDEN_POINTER_TAILS = {
    "_cdata",
    "_typed_storage",
    "data_ptr",
    "id",
    "storage",
    "storage_offset",
    "untyped_storage",
}
FORBIDDEN_INTROSPECTION_ATTRIBUTES = {
    "__builtins__",
    "__class__",
    "__code__",
    "__cuda_array_interface__",
    "__dict__",
    "__globals__",
    "__subclasses__",
    "f_back",
    "f_builtins",
    "f_code",
    "f_globals",
    "f_locals",
    "tb_frame",
    "tb_next",
}
FORBIDDEN_NAMESPACE_PREFIXES = ("torch.cuda", "torch.hub", "triton.runtime")
MUTATING_METHOD_TAILS = {
    "add",
    "append",
    "clear",
    "discard",
    "extend",
    "insert",
    "pop",
    "popitem",
    "remove",
    "setdefault",
    "update",
}
INDIRECT_TARGET = "<indirect-call-target>"
ALLOWED_BUILTIN_CALLS = {
    "Exception",
    "RuntimeError",
    "TypeError",
    "ValueError",
    "abs",
    "bool",
    "dict",
    "enumerate",
    "float",
    "int",
    "len",
    "list",
    "max",
    "min",
    "range",
    "set",
    "sorted",
    "sum",
    "tuple",
    "zip",
}
ALLOWED_METADATA_METHOD_TAILS = {"get", "numel", "sort", "stride", "to"}


def _issue(
    code: str,
    path: str,
    message: str,
    *,
    node: ast.AST | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {"code": code, "path": path, "message": message}
    if node is not None:
        item["line"] = int(getattr(node, "lineno", 1))
        item["column"] = int(getattr(node, "col_offset", 0))
    if token is not None:
        item["token"] = token
    return item


def _possible_names(node: ast.AST, aliases: dict[str, set[str]]) -> set[str]:
    if isinstance(node, ast.Name):
        return set(aliases.get(node.id, {node.id}))
    if isinstance(node, ast.Attribute):
        prefixes = _possible_names(node.value, aliases)
        return {f"{prefix}.{node.attr}" for prefix in prefixes}
    if isinstance(node, ast.IfExp):
        return _possible_names(node.body, aliases) | _possible_names(
            node.orelse, aliases
        )
    return {INDIRECT_TARGET}


def _display_name(node: ast.AST, aliases: dict[str, set[str]]) -> str:
    return "|".join(sorted(_possible_names(node, aliases)))


def _root_name(node: ast.AST) -> ast.Name | None:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node if isinstance(node, ast.Name) else None


def _targets(target: ast.AST) -> Iterable[ast.AST]:
    if isinstance(target, (ast.Tuple, ast.List)):
        for item in target.elts:
            yield from _targets(item)
    else:
        yield target


def _target_names(target: ast.AST) -> Iterable[str]:
    for item in _targets(target):
        if isinstance(item, ast.Name):
            yield item.id


def _package_forbidden(name: str) -> bool:
    lowered = name.lower()
    return any(
        lowered == prefix or lowered.startswith(prefix + ".")
        for prefix in FORBIDDEN_PACKAGE_PREFIXES
    )


class _Policy(ast.NodeVisitor):
    def __init__(self, path: str) -> None:
        self.path = path
        self.aliases: dict[str, set[str]] = {}
        self.import_roots: set[str] = set()
        self.local_functions: set[str] = set()
        self.triton_kernels: set[str] = set()
        self.errors: list[dict[str, Any]] = []

    def error(self, code: str, node: ast.AST, message: str, token: str) -> None:
        self.errors.append(_issue(code, self.path, message, node=node, token=token))

    def _add_alias(self, local: str, names: set[str]) -> None:
        self.aliases.setdefault(local, set()).update(names)

    def _mark_unknown_target(self, target: ast.AST) -> None:
        for name in _target_names(target):
            self._add_alias(name, {INDIRECT_TARGET})

    def _check_numerical_reference(self, node: ast.AST, names: set[str]) -> None:
        for name in sorted(names):
            if (
                name.split(".", 1)[0] in BOUNDED_API_ROOTS
                and name not in ALLOWED_NUMERICAL_REFERENCES
            ):
                self.error(
                    "unadmitted_api",
                    node,
                    "reference is outside the bounded numerical API allowlist",
                    name,
                )

    def visit_Module(self, node: ast.Module) -> None:  # noqa: N802
        self.local_functions.update(
            statement.name
            for statement in ast.walk(node)
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        self.generic_visit(node)

    def _exact_triton_jit(self, node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "triton"
            and node.attr == "jit"
            and self.aliases.get("triton", {"triton"}) == {"triton"}
        )

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.local_functions.add(node.name)
        arguments = [
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ]
        if node.args.vararg is not None:
            arguments.append(node.args.vararg)
        if node.args.kwarg is not None:
            arguments.append(node.args.kwarg)
        for argument in arguments:
            self._add_alias(argument.arg, {INDIRECT_TARGET})
        for decorator in node.decorator_list:
            if self._exact_triton_jit(decorator):
                self.triton_kernels.add(node.name)
            else:
                self.error(
                    "executable_decorator",
                    decorator,
                    "only the exact @triton.jit kernel decorator is permitted",
                    ast.unparse(decorator),
                )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        self._visit_function(node)

    def visit_AsyncFunctionDef(  # noqa: N802
        self, node: ast.AsyncFunctionDef
    ) -> None:
        self._visit_function(node)

    def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
        for item in node.names:
            root = item.name.split(".", 1)[0]
            local = item.asname or root
            self._add_alias(local, {item.name})
            self.import_roots.add(root)
            self._check_numerical_reference(node, {item.name})
            if _package_forbidden(item.name) or root not in ALLOWED_IMPORT_ROOTS:
                self.error(
                    "forbidden_import",
                    node,
                    "import is outside the candidate allowlist",
                    item.name,
                )
            if item.name.startswith("torch.utils.cpp_extension"):
                self.error(
                    "native_loading",
                    node,
                    "runtime extension compilation is forbidden",
                    item.name,
                )
            if any(
                item.name == prefix or item.name.startswith(prefix + ".")
                for prefix in FORBIDDEN_NAMESPACE_PREFIXES
            ):
                self.error(
                    "forbidden_namespace",
                    node,
                    "the imported runtime-control namespace is forbidden",
                    item.name,
                )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
        module = node.module or ""
        root = module.split(".", 1)[0]
        if node.level:
            self.error(
                "forbidden_import",
                node,
                "relative imports are forbidden",
                "." * node.level + module,
            )
        if not module or _package_forbidden(module) or root not in ALLOWED_IMPORT_ROOTS:
            self.error(
                "forbidden_import",
                node,
                "import is outside the candidate allowlist",
                module or "<relative>",
            )
        if root:
            self.import_roots.add(root)
        for item in node.names:
            if item.name == "*":
                self.error(
                    "wildcard_import",
                    node,
                    "wildcard imports prevent conservative name resolution",
                    f"{module}.*",
                )
                continue
            local = item.asname or item.name
            resolved = f"{module}.{item.name}" if module else item.name
            self._add_alias(local, {resolved})
            self._check_numerical_reference(node, {resolved})
            if any(
                resolved == prefix or resolved.startswith(prefix + ".")
                for prefix in FORBIDDEN_NAMESPACE_PREFIXES
            ):
                self.error(
                    "forbidden_namespace",
                    node,
                    "the imported runtime-control namespace is forbidden",
                    resolved,
                )
        if module.startswith("torch.utils.cpp_extension"):
            self.error(
                "native_loading",
                node,
                "runtime extension compilation is forbidden",
                module,
            )

    def _record_alias(self, target: ast.AST, value: ast.AST) -> None:
        if isinstance(target, (ast.Tuple, ast.List)) and isinstance(
            value, (ast.Tuple, ast.List)
        ):
            if len(target.elts) == len(value.elts):
                for left, right in zip(target.elts, value.elts, strict=True):
                    self._record_alias(left, right)
            return
        if not isinstance(target, ast.Name):
            return
        if isinstance(value, (ast.Name, ast.Attribute, ast.IfExp)):
            resolved = _possible_names(value, self.aliases)
        elif isinstance(value, (ast.Lambda, ast.Subscript, ast.Call)):
            resolved = {INDIRECT_TARGET}
        else:
            return
        resolved.discard(target.id)
        if resolved:
            self._add_alias(target.id, resolved)

    def _check_mutation(self, target: ast.AST) -> None:
        for item in _targets(target):
            root = _root_name(item)
            if root is None or isinstance(item, ast.Name):
                continue
            resolved = _possible_names(root, self.aliases)
            if any(name.split(".", 1)[0] in self.import_roots for name in resolved):
                token = _display_name(item, self.aliases)
                self.error(
                    "module_mutation",
                    item,
                    "attributes of imported modules may not be mutated",
                    token,
                )

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        for target in node.targets:
            self._check_mutation(target)
            self._record_alias(target, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        self._check_mutation(node.target)
        if node.value is not None:
            self._record_alias(node.target, node.value)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:  # noqa: N802
        self._check_mutation(node.target)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:  # noqa: N802
        self._record_alias(node.target, node.value)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:  # noqa: N802
        for target in node.targets:
            self._check_mutation(target)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:  # noqa: N802
        self._mark_unknown_target(node.target)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:  # noqa: N802
        self._mark_unknown_target(node.target)
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:  # noqa: N802
        self._mark_unknown_target(node.target)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:  # noqa: N802
        for item in node.items:
            if item.optional_vars is not None:
                self._mark_unknown_target(item.optional_vars)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:  # noqa: N802
        for item in node.items:
            if item.optional_vars is not None:
                self._mark_unknown_target(item.optional_vars)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:  # noqa: N802
        if node.name is not None:
            self._add_alias(node.name, {INDIRECT_TARGET})
        self.generic_visit(node)

    def _check_forbidden_reference(self, node: ast.AST, names: set[str]) -> None:
        for name in sorted(names):
            lowered = name.lower()
            tail = lowered.rsplit(".", 1)[-1]
            if (
                tail in FORBIDDEN_MATH_TAILS
                and "." in lowered
                and lowered not in {"tl.dot", "triton.language.dot"}
            ):
                self.error(
                    "forbidden_math",
                    node,
                    "references to private fused matrix multiplication are forbidden",
                    name,
                )
            if tail in FORBIDDEN_ATTENTION_TAILS and "." in lowered:
                self.error(
                    "forbidden_attention",
                    node,
                    "references to existing fused attention are forbidden",
                    name,
                )
            if lowered in FORBIDDEN_DYNAMIC_NAMES:
                self.error(
                    "dynamic_code",
                    node,
                    "references to dynamic import/reflection/code are forbidden",
                    name,
                )
            if lowered == "open":
                self.error("file_io", node, "references to file I/O are forbidden", name)

    def visit_Attribute(self, node: ast.Attribute) -> None:  # noqa: N802
        self._check_numerical_reference(node, _possible_names(node, self.aliases))
        names = {name.lower() for name in _possible_names(node, self.aliases)}
        shown = "|".join(sorted(names))
        self._check_forbidden_reference(node, names)
        if any(name == "torch.ops" or name.startswith("torch.ops.") for name in names):
            self.error(
                "forbidden_dispatch",
                node,
                "torch.ops dispatch is forbidden",
                shown,
            )
        if any(
            name == "torch.utils.cpp_extension"
            or name.startswith("torch.utils.cpp_extension.")
            for name in names
        ):
            self.error(
                "native_loading",
                node,
                "runtime extension compilation is forbidden",
                shown,
            )
        if any(
            name == prefix or name.startswith(prefix + ".")
            for name in names
            for prefix in FORBIDDEN_NAMESPACE_PREFIXES
        ):
            self.error(
                "forbidden_namespace",
                node,
                "torch CUDA/hub and Triton runtime access is forbidden",
                shown,
            )
        if node.attr.startswith("__") or node.attr in FORBIDDEN_INTROSPECTION_ATTRIBUTES:
            self.error(
                "dangerous_dunder",
                node,
                "dunder and reflective attribute access is forbidden",
                shown or node.attr,
            )
        if any(
            name.rsplit(".", 1)[-1] in FORBIDDEN_POINTER_TAILS for name in names
        ) or node.attr in FORBIDDEN_INTROSPECTION_ATTRIBUTES:
            self.error(
                "pointer_inspection",
                node,
                "pointer, storage, or reflective inspection is forbidden",
                shown or node.attr,
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:  # noqa: N802
        if isinstance(node.ctx, ast.Load):
            self._check_numerical_reference(node, _possible_names(node, self.aliases))
            self._check_forbidden_reference(
                node, {name.lower() for name in _possible_names(node, self.aliases)}
            )
        if node.id == "__builtins__" or (
            node.id.startswith("__") and node.id.endswith("__")
        ):
            self.error(
                "dangerous_dunder",
                node,
                "dunder and builtins-table access is forbidden",
                node.id,
            )

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        kernel_name = (
            node.func.value.id
            if isinstance(node.func, ast.Subscript)
            and isinstance(node.func.value, ast.Name)
            else None
        )
        legal_kernel_launch = (
            isinstance(node.func, ast.Subscript)
            and kernel_name in self.triton_kernels
            and self.aliases.get(kernel_name, {kernel_name}) == {kernel_name}
        )
        if not isinstance(node.func, (ast.Name, ast.Attribute)) and not legal_kernel_launch:
            self.error(
                "indirect_call",
                node,
                "call targets must be direct names/attributes or exact Triton kernel launches",
                type(node.func).__name__,
            )

        names = (
            _possible_names(node.func, self.aliases)
            if isinstance(node.func, (ast.Name, ast.Attribute))
            else {node.func.value.id} if legal_kernel_launch else {INDIRECT_TARGET}
        )
        shown = "|".join(sorted(names))
        for name in sorted(names):
            tail = name.rsplit(".", 1)[-1]
            allowed = (
                name in ALLOWED_BUILTIN_CALLS
                or (
                    name in self.local_functions
                    and self.aliases.get(name, {name}) == {name}
                )
                or name in ALLOWED_NUMERICAL_CALLS
                or (
                    name.split(".", 1)[0] not in BOUNDED_API_ROOTS
                    and tail in ALLOWED_METADATA_METHOD_TAILS
                )
            )
            if not allowed:
                self.error(
                    "indirect_call",
                    node,
                    "call target is not a resolved safe API, builtin, or local definition",
                    shown,
                )
        for name in sorted(names):
            lowered = name.lower()
            tail = lowered.rsplit(".", 1)[-1]
            if lowered == INDIRECT_TARGET:
                self.error(
                    "indirect_call",
                    node,
                    "ambiguous callable aliases are forbidden",
                    shown,
                )
            if lowered in FORBIDDEN_DYNAMIC_NAMES:
                self.error(
                    "dynamic_code",
                    node,
                    "dynamic import, reflection, and code execution are forbidden",
                    name,
                )
            if lowered == "id":
                self.error(
                    "pointer_inspection",
                    node,
                    "object identity inspection is forbidden",
                    name,
                )
            if lowered in FORBIDDEN_NATIVE_EXACT or tail in FORBIDDEN_NATIVE_TAILS:
                self.error(
                    "native_loading",
                    node,
                    "native, JIT, extension, or library loading is forbidden",
                    name,
                )
            if (
                tail in FORBIDDEN_MATH_TAILS
                and "." in lowered
                and lowered not in {"tl.dot", "triton.language.dot"}
            ):
                self.error(
                    "forbidden_math",
                    node,
                    "private fused matrix multiplication is forbidden",
                    name,
                )
            if tail in FORBIDDEN_ATTENTION_TAILS and "." in lowered:
                self.error(
                    "forbidden_attention",
                    node,
                    "existing fused attention implementation is forbidden",
                    name,
                )
            if lowered == "torch.ops" or lowered.startswith("torch.ops."):
                self.error(
                    "forbidden_dispatch",
                    node,
                    "torch.ops dispatch is forbidden",
                    name,
                )
            if (
                lowered == "open"
                or lowered in {"torch.load", "torch.save", "torch.from_file"}
                or tail in FORBIDDEN_FILE_TAILS
            ):
                self.error("file_io", node, "file I/O is forbidden", name)
            if any(
                lowered == prefix or lowered.startswith(prefix + ".")
                for prefix in FORBIDDEN_NAMESPACE_PREFIXES
            ):
                self.error(
                    "forbidden_namespace",
                    node,
                    "torch CUDA/hub and Triton runtime access is forbidden",
                    name,
                )
            if (
                lowered.startswith("torch.cuda.") and tail in FORBIDDEN_CUDA_TAILS
            ) or tail == "externalstream":
                self.error(
                    "cuda_control",
                    node,
                    "CUDA stream, event, or synchronization control is forbidden",
                    name,
                )
            if tail in FORBIDDEN_TIMING_TAILS:
                self.error("timing", node, "timing primitives are forbidden", name)

        if isinstance(node.func, ast.Attribute):
            receiver_names = _possible_names(node.func.value, self.aliases)
            if node.func.attr.lower() in MUTATING_METHOD_TAILS and any(
                name.split(".", 1)[0] in self.import_roots
                for name in receiver_names
            ):
                self.error(
                    "module_mutation",
                    node,
                    "mutation calls rooted in imported modules are forbidden",
                    shown,
                )
        self.generic_visit(node)


def _ast_depth(tree: ast.AST) -> int:
    maximum = 0
    stack = [(tree, 1)]
    while stack:
        node, depth = stack.pop()
        maximum = max(maximum, depth)
        if maximum > MAX_AST_DEPTH:
            return maximum
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    return maximum


def _validate_abi(tree: ast.Module, path: str) -> list[dict[str, Any]]:
    definitions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == ENTRYPOINT
    ]
    errors: list[dict[str, Any]] = []
    if len(definitions) != 1:
        errors.append(
            _issue(
                "invalid_abi",
                path,
                f"{ENTRYPOINT} must be defined exactly once",
                token=ENTRYPOINT,
            )
        )
        return errors
    function = definitions[0]
    valid = (
        isinstance(function, ast.FunctionDef)
        and function in tree.body
        and not function.decorator_list
        and not function.args.posonlyargs
        and tuple(argument.arg for argument in function.args.args)
        == ENTRYPOINT_ARGUMENTS
        and not function.args.vararg
        and not function.args.kwonlyargs
        and not function.args.kwarg
        and not function.args.defaults
        and not function.args.kw_defaults
        and not any(
            isinstance(node, (ast.Yield, ast.YieldFrom))
            for node in ast.walk(function)
        )
    )
    if not valid:
        errors.append(
            _issue(
                "invalid_abi",
                path,
                "entrypoint must be one undecorated synchronous top-level function "
                "with the exact five positional arguments, no defaults, and no yield",
                node=function,
                token=ENTRYPOINT,
            )
        )
    for statement in tree.body:
        if statement is function:
            continue
        targets: list[ast.AST] = []
        if isinstance(statement, ast.Assign):
            targets.extend(statement.targets)
        elif isinstance(statement, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets.append(statement.target)
        elif isinstance(statement, ast.Delete):
            targets.extend(statement.targets)
        if any(ENTRYPOINT in set(_target_names(target)) for target in targets):
            errors.append(
                _issue(
                    "invalid_abi",
                    path,
                    "entrypoint may not be rebound or deleted",
                    node=statement,
                    token=ENTRYPOINT,
                )
            )
    return errors


def validate_source_text(source: str, path: str) -> list[dict[str, Any]]:
    """Return deterministic structured policy violations for one source file."""

    label = str(path)
    if not isinstance(source, str):
        return [_issue("invalid_source", label, "source must be text")]
    try:
        encoded = source.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        return [_issue("invalid_utf8", label, "source is not valid UTF-8 text")]
    if len(encoded) > MAX_SOURCE_BYTES:
        return [
            _issue(
                "source_too_large",
                label,
                f"source exceeds {MAX_SOURCE_BYTES} bytes",
            )
        ]
    if not encoded:
        return [_issue("invalid_source", label, "source must not be empty")]
    if encoded.startswith(b"\xef\xbb\xbf") or b"\x00" in encoded:
        return [_issue("unsafe_text", label, "BOM and NUL bytes are forbidden")]
    try:
        tree = ast.parse(source, filename=label)
    except (SyntaxError, ValueError) as exc:
        return [
            _issue(
                "syntax_error",
                label,
                f"invalid Python syntax: {getattr(exc, 'msg', str(exc))}",
            )
        ]
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        return [
            _issue(
                "ast_too_large",
                label,
                f"source AST exceeds {MAX_AST_NODES} nodes",
            )
        ]
    if _ast_depth(tree) > MAX_AST_DEPTH:
        return [
            _issue(
                "ast_too_deep",
                label,
                f"source AST exceeds depth {MAX_AST_DEPTH}",
            )
        ]
    errors = _validate_abi(tree, label)
    for node in nodes:
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            value = node.value.encode("utf-8") if isinstance(node.value, str) else node.value
            if len(value) > MAX_LITERAL_BYTES:
                errors.append(
                    _issue(
                        "literal_too_large",
                        label,
                        f"literal exceeds {MAX_LITERAL_BYTES} bytes",
                        node=node,
                    )
                )
    policy = _Policy(label)
    policy.visit(tree)
    errors.extend(policy.errors)
    return errors


@dataclass(frozen=True)
class ApprovedWorkspace:
    """Immutable bytes approved for Task 5; consumers must never reopen paths."""

    submission_bytes: bytes
    submission_sha256: str
    notes_bytes: bytes | None
    notes_sha256: str | None


class WorkspaceAdmissionError(ValueError):
    """Raised when ``admit_workspace`` cannot produce approved bytes."""

    def __init__(self, errors: list[dict[str, Any]]) -> None:
        super().__init__("workspace admission failed")
        self.errors = errors


def _read_regular_bytes_at(
    directory_fd: int,
    name: str,
    label: str,
    *,
    max_bytes: int,
    required: bool,
) -> tuple[bytes | None, int, list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    try:
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        if required:
            errors.append(_issue("missing_submission", label, "submission.py is required"))
        return None, 0, errors
    except OSError as exc:
        return None, 0, [_issue("unsafe_file", label, f"cannot inspect file: {exc}")]
    if not stat.S_ISREG(metadata.st_mode):
        return None, metadata.st_size, [
            _issue("unsafe_file", label, "file must be a regular non-symlink file")
        ]
    if metadata.st_nlink != 1:
        errors.append(_issue("hardlink", label, "hard-linked files are forbidden"))
    if metadata.st_size > max_bytes:
        errors.append(
            _issue("file_too_large", label, f"file exceeds {max_bytes} bytes")
        )
        return None, metadata.st_size, errors
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        file_fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        errors.append(_issue("unsafe_file", label, f"cannot open file safely: {exc}"))
        return None, metadata.st_size, errors
    try:
        opened = os.fstat(file_fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino, opened.st_size)
            != (metadata.st_dev, metadata.st_ino, metadata.st_size)
        ):
            errors.append(_issue("unsafe_file", label, "file changed before safe open"))
            return None, metadata.st_size, errors
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(file_fd, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        after = os.fstat(file_fd)
    except OSError as exc:
        errors.append(_issue("unsafe_file", label, f"cannot read file safely: {exc}"))
        return None, metadata.st_size, errors
    finally:
        os.close(file_fd)
    if (
        (after.st_dev, after.st_ino, after.st_mode, after.st_nlink, after.st_size)
        != (opened.st_dev, opened.st_ino, opened.st_mode, opened.st_nlink, opened.st_size)
        or len(data) != opened.st_size
    ):
        errors.append(_issue("unsafe_file", label, "file changed during inspection"))
    if data.startswith(b"\xef\xbb\xbf") or b"\x00" in data:
        errors.append(_issue("unsafe_text", label, "BOM and NUL bytes are forbidden"))
    try:
        data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        errors.append(_issue("invalid_utf8", label, "file must be valid UTF-8 text"))
        return None, len(data), errors
    if required and not data:
        errors.append(_issue("invalid_source", label, "submission.py must not be empty"))
    return data, len(data), errors


def inspect_workspace(
    root: Path,
) -> tuple[ApprovedWorkspace | None, list[dict[str, Any]]]:
    """Inspect through stable file descriptors and return the exact approved bytes."""

    workspace = Path(root)
    label = str(workspace)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        directory_fd = os.open(workspace, directory_flags)
    except OSError as exc:
        return None, [_issue("invalid_workspace", label, f"cannot open workspace safely: {exc}")]

    errors: list[dict[str, Any]] = []
    allowed = {"submission.py", "notes.md"}
    total_bytes = 0
    try:
        root_metadata = os.fstat(directory_fd)
        if not stat.S_ISDIR(root_metadata.st_mode):
            return None, [
                _issue("invalid_workspace", label, "workspace root must be a directory")
            ]
        try:
            names = sorted(os.listdir(directory_fd))
        except OSError as exc:
            return None, [
                _issue("invalid_workspace", label, f"cannot list workspace: {exc}")
            ]
        for name in names:
            try:
                entry_metadata = os.stat(
                    name, dir_fd=directory_fd, follow_symlinks=False
                )
            except OSError as exc:
                errors.append(_issue("unsafe_file", name, f"cannot inspect path: {exc}"))
                continue
            total_bytes += int(entry_metadata.st_size)
            if name not in allowed:
                errors.append(
                    _issue(
                        "unexpected_path",
                        name,
                        "workspace permits only submission.py and optional notes.md",
                    )
                )
            if not stat.S_ISREG(entry_metadata.st_mode):
                errors.append(
                    _issue("unsafe_file", name, "workspace paths must be regular files")
                )
            elif entry_metadata.st_nlink != 1:
                errors.append(_issue("hardlink", name, "hard-linked files are forbidden"))
        if total_bytes > MAX_WORKSPACE_BYTES:
            errors.append(
                _issue(
                    "workspace_too_large",
                    label,
                    f"workspace exceeds {MAX_WORKSPACE_BYTES} bytes",
                )
            )

        source, _source_size, source_errors = _read_regular_bytes_at(
            directory_fd,
            "submission.py",
            "submission.py",
            max_bytes=MAX_SOURCE_BYTES,
            required=True,
        )
        errors.extend(source_errors)
        notes, _notes_size, notes_errors = _read_regular_bytes_at(
            directory_fd,
            "notes.md",
            "notes.md",
            max_bytes=MAX_NOTES_BYTES,
            required=False,
        )
        errors.extend(notes_errors)
        if source is not None:
            errors.extend(
                validate_source_text(source.decode("utf-8"), "submission.py")
            )
        try:
            if sorted(os.listdir(directory_fd)) != names:
                errors.append(
                    _issue("unsafe_file", label, "workspace changed during inspection")
                )
        except OSError as exc:
            errors.append(
                _issue("unsafe_file", label, f"cannot recheck workspace: {exc}")
            )
    finally:
        os.close(directory_fd)

    if errors or source is None:
        return None, errors
    approved = ApprovedWorkspace(
        submission_bytes=source,
        submission_sha256=hashlib.sha256(source).hexdigest(),
        notes_bytes=notes,
        notes_sha256=hashlib.sha256(notes).hexdigest() if notes is not None else None,
    )
    return approved, []


def admit_workspace(root: Path) -> ApprovedWorkspace:
    """Return immutable approved bytes; Task 5 must import these, never the path."""

    approved, errors = inspect_workspace(root)
    if approved is None:
        raise WorkspaceAdmissionError(errors)
    return approved


def validate_workspace(root: Path) -> list[dict[str, Any]]:
    """Preserve the list-returning admission interface for existing callers."""

    _approved, errors = inspect_workspace(root)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args(argv)
    errors = validate_workspace(args.root)
    print(json.dumps({"ok": not errors, "errors": errors}, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
