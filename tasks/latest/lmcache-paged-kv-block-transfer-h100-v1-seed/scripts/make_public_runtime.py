#!/usr/bin/env python3
"""Derive the Work image's public runtime from the trusted runtime.

The public runtime measures and verifies exactly like the Judge's (same
buffers, same verification, same timing, same overlap foreground), minus the
Judge-only parts: the same-run bandwidth envelope and its probe, the final
manifest loader, and every comment and docstring.  It reads the public case
table from ``public_cases.json`` next to itself.

  make_public_runtime.py runtime/kv_transfer_runtime.py images/work/benchmark/kv_transfer_runtime.py

Run it with the repository's Python (``.venv``): ``ast.unparse`` formatting
differs between Python versions, and tests/test_contract.py regenerates the
file with the test interpreter and compares bytes.
"""
from __future__ import annotations

import ast
import re
import sys

DROP_NAMES = {
    "ENVELOPE_MARGIN", "ENVELOPE_PROTOCOL", "ENVELOPE_PROBE_BYTES", "ENVELOPE_PROBE_REPEATS",
    "ENVELOPE_ZEROCOPY_SOURCE", "ENVELOPE_MIN_DEVICE_GBPS", "ENVELOPE_MIN_HOST_GBPS",
    "QUALIFICATION_ENVELOPE_GBPS", "validate_envelopes", "measure_envelopes",
    "_ZEROCOPY_PROBE", "_load_zerocopy_probe", "case_envelope_gbps", "FINAL_SUITE_ID",
    "load_manifest",
}
BANNED = re.compile(
    r"Judge-owned|holdout|anchor|upstream|f9addd2e|multi_layer|mp_mem|LMCache|vLLM|HAMi|PyroMind|"
    r"H100|NVL|envelope|[Dd]ecoy|sentinel|Trusted|SCORED_|1763|247\.1|headroom"
)
PUBLIC_SUITE = '''

def public_suite() -> list[dict[str, Any]]:
    return load_case_table(Path(__file__).resolve().parent / "public_cases.json")["cases"]
'''


def _names(node: ast.stmt) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, ast.Assign):
        return {t.id for t in node.targets if isinstance(t, ast.Name)}
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return {node.target.id}
    return set()


class Rewrite(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        _strip_docstring(node)
        if node.name == "run_scored_case":
            node.body = [stmt for stmt in node.body if not _is_envelope_check(stmt)]
            node.args.args = [arg for arg in node.args.args if arg.arg != "envelopes"]
        if node.name == "run_suite":
            node.body = [stmt for stmt in node.body if not _assigns(stmt, "envelopes")]
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        self.generic_visit(node)
        _strip_docstring(node)
        if node.name == "TrustedClock":
            node.name = "Clock"
        return node

    def visit_Dict(self, node: ast.Dict) -> ast.AST:
        self.generic_visit(node)
        keep = [(k, v) for k, v in zip(node.keys, node.values)
                if not (isinstance(k, ast.Constant) and isinstance(k.value, str)
                        and k.value.startswith("envelope"))]
        node.keys = [k for k, _v in keep]
        node.values = [v for _k, v in keep]
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == "envelopes":
            return ast.copy_location(ast.Constant(value=None), node) \
                if isinstance(node.ctx, ast.Load) else node
        if node.id == "TrustedClock":
            node.id = "Clock"
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        if node.annotation is not None:
            node.annotation = self.visit(node.annotation)
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, str):
            node.value = node.value.replace("decoy", "pre-call").replace("TrustedClock", "Clock")
        return node


def _strip_docstring(node: ast.AST) -> None:
    body = getattr(node, "body", [])
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        node.body = body[1:] or [ast.Pass()]


def _assigns(stmt: ast.stmt, name: str) -> bool:
    return isinstance(stmt, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == name for t in stmt.targets)


def _is_envelope_check(stmt: ast.stmt) -> bool:
    return (isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Compare)
            and isinstance(stmt.test.left, ast.Name) and stmt.test.left.id == "envelopes")


def main(src: str, dst: str) -> None:
    tree = ast.parse(open(src, encoding="utf-8").read())
    tree.body = [node for node in tree.body if not (_names(node) & DROP_NAMES)]
    _strip_docstring(tree)
    tree = Rewrite().visit(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_suite":
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and getattr(call.func, "id", "") == "run_scored_case":
                    call.args.pop()
    ast.fix_missing_locations(tree)
    body = ast.unparse(tree)
    body = body.replace("_TrustedClock", "_Clock")
    text = ("#!/usr/bin/env python3\n"
            '"""Public test runtime for the paged-KV block transfer task."""\n'
            + body + "\n" + PUBLIC_SUITE)
    match = BANNED.search(text)
    if match:
        raise SystemExit(f"public runtime still names {match.group()!r}: {text[max(0, match.start() - 60):match.end() + 20]!r}")
    compile(text, dst, "exec")
    open(dst, "w", encoding="utf-8").write(text)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
