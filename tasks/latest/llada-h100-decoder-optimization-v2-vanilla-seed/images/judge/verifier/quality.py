"""Frozen paper-aligned answer parsing and restricted code evaluation."""

from __future__ import annotations

import ast
import os
import pwd
import re
import resource
import secrets
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


SAFE_IMPORTS = {
    "bisect", "collections", "decimal", "fractions", "functools", "heapq",
    "itertools", "math", "re", "statistics", "typing",
}
DANGEROUS_NAMES = {
    "SystemExit", "__builtins__", "__import__", "breakpoint", "compile",
    "delattr", "dir", "eval", "exec", "exit", "getattr", "globals", "input",
    "locals", "open", "quit", "setattr", "vars",
}
DANGEROUS_ATTRS = {
    "_exit", "abort", "chmod", "connect", "execv", "execve", "exit", "fork",
    "get_field", "get_value", "kill", "killpg", "popen", "remove", "rename",
    "replace", "rmdir", "spawn", "system", "terminate", "unlink", "vformat",
    "write_text", "write_bytes",
}


def extract_gsm_answer(text: str) -> str | None:
    strict = re.findall(r"####\s*(-?[$0-9][0-9,]*(?:\.[0-9]+)?)", text)
    values = strict or re.findall(r"-?[$0-9][0-9,]*(?:\.[0-9]+)?", text)
    if not values:
        return None
    return values[-1].replace("$", "").replace(",", "").rstrip(".")


def _fix_fracs(value: str) -> str:
    parts = value.split("\\frac")
    out = parts[0]
    for part in parts[1:]:
        out += "\\frac"
        if not part:
            continue
        if part[0] == "{":
            out += part
        elif len(part) >= 2:
            a, b, rest = part[0], part[1], part[2:]
            out += "{" + a + "}" + (b if b == "{" else "{" + b + "}") + rest
        else:
            out += part
    return out


def normalize_math(value: str) -> str:
    value = value.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    value = value.replace("tfrac", "frac").replace("dfrac", "frac")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("^{\\circ}", "").replace("^\\circ", "")
    value = value.replace("\\$", "").replace("\\%", "").replace("%", "")
    if "\\text{ " in value:
        value = value.split("\\text{ ", 1)[0]
    value = value.replace(" .", " 0.").replace("{.", "{0.")
    value = value.strip()
    if value.startswith("."):
        value = "0" + value
    if value.count("=") == 1 and len(value.split("=", 1)[0]) <= 2:
        value = value.split("=", 1)[1]
    value = re.sub(r"\\sqrt([^\{])", r"\\sqrt{\1}", value)
    value = _fix_fracs(value.replace(" ", ""))
    if value == "0.5":
        value = "\\frac{1}{2}"
    slash = re.fullmatch(r"(-?\d+)/(-?\d+)", value)
    if slash:
        value = f"\\frac{{{slash.group(1)}}}{{{slash.group(2)}}}"
    return value


def extract_math_answer(text: str) -> str | None:
    boxed: list[str] = []
    marker = "\\boxed{"
    cursor = 0
    while True:
        start = text.find(marker, cursor)
        if start < 0:
            break
        content_start = start + len(marker)
        depth = 1
        index = content_start
        while index < len(text) and depth:
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
            index += 1
        if depth == 0:
            boxed.append(text[content_start : index - 1])
            cursor = index
        else:
            break
    if boxed:
        return boxed[-1]
    dollars = re.findall(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$(?!\$)", text, flags=re.DOTALL)
    if dollars:
        return dollars[-1].strip()
    nonempty = [line.strip() for line in text.splitlines() if line.strip()]
    return nonempty[-1] if nonempty else None


def strip_code_fence(text: str) -> str:
    if "```python" in text:
        text = text.split("```python", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _longest_valid_python(text: str) -> str:
    """Mirror Fast-dLLM's HumanEval post-processing at bounded cost."""

    lines = text.replace("\t", "    ").replace("\r\n", "\n").replace("\r", "\n").strip().splitlines()
    lines = lines[:100]
    best = ""
    best_nonempty = 0
    for start in range(len(lines)):
        for end in range(start + 1, len(lines) + 1):
            snippet = "\n".join(lines[start:end])
            try:
                ast.parse(snippet)
            except (SyntaxError, MemoryError):
                continue
            nonempty = sum(bool(line.strip()) for line in lines[start:end])
            if nonempty > best_nonempty:
                best = snippet
                best_nonempty = nonempty
    return best


def _definition_name(node: ast.AST) -> str | None:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name
    if isinstance(node, ast.Assign) and node.targets and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id
    return None


def _definition_dependencies(node: ast.AST) -> set[str]:
    dependencies: set[str] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        for child in ast.iter_child_nodes(current):
            if isinstance(child, ast.Name):
                dependencies.add(child.id)
            elif isinstance(child, ast.Attribute):
                dependencies.add(child.attr)
            else:
                stack.append(child)
    return dependencies


def sanitize_humaneval(text: str, entry_point: str) -> str:
    """Keep the generated entry point and its reachable helper definitions.

    Fast-dLLM V1 evaluates LLaDA-Instruct HumanEval generations only after
    its ``postprocess_code.py`` sanitizer.  Reproducing that contract avoids
    treating markdown, a repeated function prefix, or trailing prose as code.
    """

    code = _longest_valid_python(text)
    tree = ast.parse(code)
    imports: list[ast.AST] = []
    definitions: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(node)
            continue
        name = _definition_name(node)
        if name is None:
            continue
        if isinstance(node, ast.FunctionDef) and not any(
            isinstance(child, ast.Return) for child in ast.walk(node)
        ):
            continue
        definitions[name] = node

    dependency_map = {
        name: _definition_dependencies(node) for name, node in definitions.items()
    }
    reachable: set[str] = set()
    pending = [entry_point]
    while pending:
        name = pending.pop(0)
        if name in reachable:
            continue
        reachable.add(name)
        pending.extend(dependency_map.get(name, set()) - reachable)

    selected = [ast.unparse(node) for node in imports]
    selected.extend(
        ast.unparse(node) for name, node in definitions.items() if name in reachable
    )
    return "\n".join(selected)


def code_is_safe(source: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, MemoryError) as exc:
        return False, f"syntax:{type(exc).__name__}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] not in SAFE_IMPORTS for alias in node.names):
                return False, "unsafe_import"
        elif isinstance(node, ast.ImportFrom):
            if node.module is None or node.module.split(".")[0] not in SAFE_IMPORTS:
                return False, "unsafe_import"
        elif isinstance(node, ast.Name) and node.id in DANGEROUS_NAMES:
            return False, f"unsafe_name:{node.id}"
        elif isinstance(node, ast.Attribute) and (
            node.attr.startswith("_") or node.attr in DANGEROUS_ATTRS
        ):
            return False, f"unsafe_attribute:{node.attr}"
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if "__" in node.value:
                return False, "unsafe_dunder_literal"
    return True, "ok"


def _limits() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
    resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    if os.geteuid() == 0:
        account = pwd.getpwnam("mlsysforge-grader")
        os.setgroups([])
        os.setresgid(account.pw_gid, account.pw_gid, account.pw_gid)
        os.setresuid(account.pw_uid, account.pw_uid, account.pw_uid)
    # RLIMIT_NPROC is charged to the real numeric UID across the host kernel,
    # even across ordinary Docker PID namespaces. The image assigns this
    # private grader a fixed high UID before applying the limit so unrelated
    # host service accounts cannot exhaust its allowance.
    resource.setrlimit(resource.RLIMIT_NPROC, (16, 16))


def run_restricted_python(source: str) -> tuple[bool, str]:
    safe, reason = code_is_safe(source)
    if not safe:
        return False, reason
    # A randomized, controller-chosen terminal exit code proves that all
    # appended tests reached their end.  Plain rc==0 is unsafe: generated code
    # could call a process-exit primitive before any assertion executes.
    completion_code = 64 + secrets.randbelow(128)
    executed_source = source + f"\nraise SystemExit({completion_code})\n"
    with tempfile.TemporaryDirectory(prefix="mlsysforge-code-") as directory:
        script = Path(directory) / "candidate_test.py"
        script.write_text(executed_source, encoding="utf-8")
        if os.geteuid() == 0:
            account = pwd.getpwnam("mlsysforge-grader")
            os.chown(directory, account.pw_uid, account.pw_gid)
            os.chown(script, account.pw_uid, account.pw_gid)
            os.chmod(directory, 0o700)
            os.chmod(script, 0o400)
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-S", str(script)],
                cwd=directory,
                env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=6,
                check=False,
                preexec_fn=_limits,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if result.returncode == completion_code:
        return True, "passed"
    return False, f"exit:{result.returncode}"


def score_completion(case: dict[str, Any], completion: str) -> dict[str, Any]:
    benchmark = case["benchmark"]
    if benchmark == "gsm8k":
        prediction = extract_gsm_answer(completion)
        return {"correct": prediction == str(case["gold"]), "parsed": prediction is not None}
    if benchmark == "math":
        prediction = extract_math_answer(completion)
        return {
            "correct": prediction is not None and normalize_math(prediction) == normalize_math(str(case["gold"])),
            "parsed": prediction is not None,
        }
    code = strip_code_fence(completion)
    if benchmark == "humaneval":
        program = sanitize_humaneval(
            case["prompt"] + "\n" + code,
            str(case["entry_point"]),
        ) + "\n" + case["tests"]
    elif benchmark == "mbpp":
        tests = "\n".join(case["tests"])
        program = case.get("test_setup_code", "") + "\n" + code + "\n" + tests + "\n"
    else:
        raise ValueError(f"unknown benchmark: {benchmark}")
    passed, reason = run_restricted_python(program)
    return {"correct": passed, "parsed": not reason.startswith("syntax"), "code_reason": reason}
