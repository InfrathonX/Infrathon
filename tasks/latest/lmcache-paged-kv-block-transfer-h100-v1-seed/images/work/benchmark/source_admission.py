#!/usr/bin/env python3
"""Structural and selected-source admission for the LMCache KV-transfer task.

A submission is one CUDA translation unit per candidate version.  The scan
below is a diagnostic boundary and a statement of the public policy in
AGENT_TASK.md, layered in front of the frozen compile command, the pre-load
undefined-symbol audit of the compiled library (the hard gate for what a
submission can call), the unprivileged worker process, and the physical
bandwidth envelope.  It is not a C++ parser and does not try to be one: the
rules are conservative, and anything the scan cannot classify is rejected
rather than assumed benign.

Policy summary:

- includes are allowlisted (the contract header and freestanding C++ basics);
- no filesystem, process, thread, loader, signal, or environment API tokens;
- CUDA API tokens are allowlisted to launches, on-stream async copies/sets,
  error queries, and occupancy/attribute introspection — allocation, private
  streams/events, any synchronisation, graphs/capture, peer access, host
  callbacks, and the whole driver API are rejected;
- inline ``asm`` is device-PTX only: every template is checked against x86
  register / syscall fingerprints (host-side PTX would not compile anyway,
  so the two constraints together confine asm to device code);
- ``__managed__`` memory is rejected; the entry point must be defined.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import sys
from pathlib import Path
from typing import Any


CANDIDATE_RE = re.compile(r"^kernel_v([0-9]{3})$")
FINAL_VERSION_RE = re.compile(rb"(?:[1-9][0-9]{0,2})\n?\Z")
MAX_VERSIONS = 64
MAX_SOURCE_BYTES = 512 * 1024
MAX_NOTES_BYTES = 64 * 1024
MAX_TOTAL_BYTES = 4 * 1024 * 1024

# Matched against comment/string-blanked code, where the "C" literal's
# content is blanked but its quotes survive.
ENTRYPOINT_RE = re.compile(
    r'extern\s+"[^"]*"\s+void\s+lmcache_paged_kv_block_transfer\s*\('
)

ALLOWED_INCLUDES = {
    "transfer_abi.h",
    "cuda_runtime.h",
    "stdint.h",
    "cstdint",
    "stddef.h",
    "cstddef",
    "type_traits",
}

# CUDA runtime API tokens a submission may name.  Exact-token allowlist; any
# other cuda* token is rejected.  This mirrors the binary-level
# ALLOWED_CUDA_SYMBOLS in the trusted runtime; a task-local test asserts the
# two agree.
ALLOWED_CUDA_TOKENS = {
    "cudaLaunchKernel", "cudaLaunchKernelExC", "cudaLaunchCooperativeKernel",
    "cudaMemcpyAsync", "cudaMemcpy2DAsync", "cudaMemcpy3DAsync",
    "cudaMemsetAsync", "cudaMemset2DAsync",
    "cudaGetLastError", "cudaPeekAtLastError",
    "cudaGetErrorString", "cudaGetErrorName",
    "cudaFuncSetAttribute", "cudaFuncGetAttributes", "cudaFuncSetCacheConfig",
    "cudaOccupancyMaxActiveBlocksPerMultiprocessor",
    "cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
    "cudaDeviceGetAttribute", "cudaGetDevice", "cudaGetDeviceProperties",
    "cudaGetDeviceProperties_v2",
    # types, enums, and qualifiers (not calls) that legitimate kernels name
    "cudaStream_t", "cudaError_t", "cudaSuccess", "cudaMemcpyKind",
    "cudaMemcpyHostToDevice", "cudaMemcpyDeviceToHost",
    "cudaMemcpyDeviceToDevice", "cudaMemcpyDefault",
    "cudaFuncAttributeMaxDynamicSharedMemorySize",
    "cudaFuncAttributePreferredSharedMemoryCarveout",
    "cudaFuncCachePreferShared", "cudaFuncCachePreferL1",
    "cudaFuncCachePreferEqual", "cudaFuncCachePreferNone",
    "cudaDevAttrMaxSharedMemoryPerBlockOptin",
    "cudaDevAttrMultiProcessorCount",
    "cudaDeviceProp", "cudaFuncAttributes", "cudaLaunchConfig_t",
    "cudaLaunchAttribute", "cudaLaunchAttributeID",
    "cudaLaunchAttributeCooperative",
    "cudaLaunchAttributeClusterDimension",
    "cudaLaunchAttributeClusterSchedulingPolicyPreference",
    "cudaClusterSchedulingPolicySpread", "cudaClusterSchedulingPolicyDefault",
    "cudaClusterSchedulingPolicyLoadBalancing",
}
CUDA_TOKEN_RE = re.compile(r"\bcuda[A-Za-z0-9_]+")
DRIVER_API_RE = re.compile(r"\bcu[A-Z][A-Za-z0-9_]*\s*\(")

# Word-boundary tokens that reach the system: loader, process, filesystem,
# threads, signals, environment, longjmp.  ``exec`` family is matched by
# prefix below.
FORBIDDEN_TOKENS = (
    "dlopen", "dlsym", "dlmopen", "dlclose", "dlerror",
    "system", "popen", "pclose", "fork", "vfork", "clone",
    "fopen", "freopen", "fdopen", "ifstream", "ofstream", "fstream",
    "socket", "connect", "bind", "listen", "accept",
    "mmap", "munmap", "mprotect", "mremap", "brk", "sbrk",
    "ptrace", "syscall", "sysconf", "prctl",
    "setenv", "unsetenv", "putenv", "getenv", "environ", "secure_getenv",
    "signal", "sigaction", "raise", "kill",
    "longjmp", "setjmp", "atexit", "at_quick_exit",
    "asctime", "getchar", "scanf", "gets", "fread", "fwrite", "remove",
    "rename", "tmpfile", "tmpnam",
)
FORBIDDEN_PREFIX_RE = re.compile(r"\b(?:exec[lv]p?e?|spawn|pthread_)[A-Za-z0-9_]*")
FORBIDDEN_SUBSTRINGS = (
    "std::thread", "std::async", "std::jthread", "/proc/", "/sys/", "/dev/",
    "__managed__", "_Pragma",
)

ALLOWED_PRAGMA_RE = re.compile(r"^#\s*pragma\s+(unroll|once|nounroll)\b")
INCLUDE_RE = re.compile(r'^#\s*include\s*(?:<([^>]+)>|"([^"]+)")\s*$')
PREPROC_RE = re.compile(
    r"^#\s*(include|define|undef|if|ifdef|ifndef|elif|else|endif|error|pragma)\b"
)

# Inline asm: templates must not carry x86/syscall fingerprints.  PTX in host
# code fails host compilation, so together with these fingerprints asm is
# confined to device-side PTX.
ASM_RE = re.compile(r"\basm\b")
ASM_FORBIDDEN_RE = re.compile(
    r"%%?(?:r[abcd]x|r[sd]i|r[sb]p|r(?:8|9|1[0-5])|e[abcd]x|e[sd]i|e[sb]p)\b"
    r"|int\s*\$"
    r"|\bsyscall\b"
    r"|\.byte\b"
)


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


def _strip_comments_and_strings(text: str) -> tuple[str, list[str]]:
    """Return code with comments/strings blanked, plus the string literals.

    Blanking preserves offsets and newlines, so line numbers in later scans
    stay meaningful.  String literals are collected for their own scans
    (asm templates, embedded paths).
    """
    out = []
    literals: list[str] = []
    i, n = 0, len(text)
    mode = "code"
    current = []
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if mode == "code":
            if ch == "/" and nxt == "/":
                mode = "line_comment"
                out.append("  ")
                i += 2
                continue
            if ch == "/" and nxt == "*":
                mode = "block_comment"
                out.append("  ")
                i += 2
                continue
            if ch == '"':
                mode = "string"
                current = []
                out.append('"')
                i += 1
                continue
            if ch == "'":
                mode = "char"
                out.append("'")
                i += 1
                continue
            out.append(ch)
            i += 1
        elif mode == "line_comment":
            if ch == "\n":
                mode = "code"
                out.append("\n")
            else:
                out.append(" ")
            i += 1
        elif mode == "block_comment":
            if ch == "*" and nxt == "/":
                mode = "code"
                out.append("  ")
                i += 2
                continue
            out.append("\n" if ch == "\n" else " ")
            i += 1
        elif mode == "string":
            if ch == "\\" and nxt:
                current.append(ch + nxt)
                out.append("  ")
                i += 2
                continue
            if ch == '"':
                mode = "code"
                literals.append("".join(current))
                out.append('"')
                i += 1
                continue
            current.append(ch)
            out.append("\n" if ch == "\n" else " ")
            i += 1
        elif mode == "char":
            if ch == "\\" and nxt:
                out.append("  ")
                i += 2
                continue
            if ch == "'":
                mode = "code"
                out.append("'")
                i += 1
                continue
            out.append(" ")
            i += 1
    return "".join(out), literals


def scan_cuda_source(text: str, relative_path: str) -> list[str]:
    errors: list[str] = []
    code, literals = _strip_comments_and_strings(text)
    lines = code.split("\n")
    raw_lines = text.split("\n")

    if ENTRYPOINT_RE.search(code) is None:
        errors.append(
            f"{relative_path}: entry point "
            '`extern "C" void lmcache_paged_kv_block_transfer(...)` is not defined'
        )

    continuation = False
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if continuation:
            continuation = stripped.endswith("\\")
            continue
        if not stripped.startswith("#"):
            continue
        continuation = stripped.endswith("\\")
        raw_line = raw_lines[number - 1].strip()
        match = PREPROC_RE.match(raw_line)
        if match is None:
            errors.append(f"{relative_path}:{number}: unrecognised preprocessor directive")
            continue
        directive = match.group(1)
        if directive == "include":
            include = INCLUDE_RE.match(raw_line)
            if include is None:
                errors.append(f"{relative_path}:{number}: unparseable #include")
                continue
            target = include.group(1) or include.group(2)
            if target not in ALLOWED_INCLUDES:
                errors.append(
                    f"{relative_path}:{number}: forbidden include: {target!r}"
                )
        elif directive == "pragma":
            if ALLOWED_PRAGMA_RE.match(raw_line) is None:
                errors.append(
                    f"{relative_path}:{number}: forbidden pragma: {raw_line!r}"
                )

    def _line_of(offset: int) -> int:
        return code.count("\n", 0, offset) + 1

    for token in FORBIDDEN_TOKENS:
        for match in re.finditer(rf"\b{re.escape(token)}\b", code):
            errors.append(
                f"{relative_path}:{_line_of(match.start())}: forbidden token: {token!r}"
            )
    for match in FORBIDDEN_PREFIX_RE.finditer(code):
        errors.append(
            f"{relative_path}:{_line_of(match.start())}: forbidden token: {match.group(0)!r}"
        )
    for needle in FORBIDDEN_SUBSTRINGS:
        base = code if needle not in ("/proc/", "/sys/", "/dev/") else "\n".join(literals)
        for match in re.finditer(re.escape(needle), base):
            errors.append(f"{relative_path}: forbidden construct: {needle!r}")
            break

    for match in CUDA_TOKEN_RE.finditer(code):
        token = match.group(0)
        if token not in ALLOWED_CUDA_TOKENS:
            errors.append(
                f"{relative_path}:{_line_of(match.start())}: "
                f"CUDA API token outside the allowlist: {token!r}"
            )
    for match in DRIVER_API_RE.finditer(code):
        errors.append(
            f"{relative_path}:{_line_of(match.start())}: "
            f"CUDA driver API is forbidden: {match.group(0)!r}"
        )

    for match in ASM_RE.finditer(code):
        # The template string(s) belong to the nearest literals; scan every
        # literal for the fingerprints instead of pairing them precisely.
        break
    if ASM_RE.search(code):
        for literal in literals:
            if ASM_FORBIDDEN_RE.search(literal):
                errors.append(
                    f"{relative_path}: inline asm carries a host-ISA or syscall "
                    f"fingerprint: {literal[:80]!r}"
                )

    return sorted(set(errors))


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
        allowed_root_entries.update(
            {"AGENT_TASK.md", "README_AGENT.md", "build", "tools", "transfer_abi.h", ".sforge"}
        )
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
            for entry in _list_directory(candidate_dir, root, errors):
                if entry.name != "submission.cu":
                    errors.append(f"unexpected candidate entry: {_relative(entry, root)}")
            source_path = candidate_dir / "submission.cu"
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
            errors.extend(scan_cuda_source(selected_text, selected_relative_path))
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
