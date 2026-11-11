"""Candidate-aware replay measurement records and performance curves."""

from __future__ import annotations

import dis
import fcntl
import functools
import hashlib
import copy
import io
import importlib.metadata
import json
import math
import os
import platform
import re
import statistics
import stat
import sys
import types
import uuid
from dataclasses import asdict, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path, PurePath
from typing import Any, Mapping, Sequence

from sforge.harness.backend.base import ResolvedImage
from sforge.harness.config import SForgeConfig, get_container_env
from sforge.harness.cpu_affinity import policy_for_role
from sforge.harness.selection import select_best
from sforge.harness.task_spec import TaskSpec
from sforge.harness.version_store import StoredCandidate, StoredVersion


MEASUREMENT_SCHEMA_VERSION = 1
MEASUREMENT_KIND = "sforge-candidate-measurement"
CURVE_KIND = "sforge-candidate-curve"
PLAN_KIND = "sforge-candidate-replay-plan"
_SELECTION_POLICIES = {"pass_rate_first", "score_first", "valid_then_score"}
_SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}")
_CONTEXT_KEY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,127}")
_CANDIDATE_ID_RE = re.compile(r"candidate-([0-9a-f]{64})")
_CONTEXT_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_MAX_CONTEXT_VALUE_LENGTH = 4096
_MAX_REPLAY_JSON_BYTES = 64 * 1024 * 1024
_MAX_LOADED_VALUE_DEPTH = 8
_SENSITIVE_RUNTIME_NAME_FRAGMENTS = (
    "API_KEY",
    "AUTH",
    "COOKIE",
    "CREDENTIAL",
    "NONCE",
    "PASSWORD",
    "PRIVATE_KEY",
    "SECRET",
    "SESSION_KEY",
    "TOKEN",
)
_EPHEMERAL_RUNTIME_NAME_FRAGMENTS = (
    "CACHE",
    "LOCK",
    "LOGGER",
    "MEMO",
)
_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_READ_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(
    os, "O_CLOEXEC", 0
)


class CandidateReplayError(RuntimeError):
    """A replay plan or persisted measurement is inconsistent."""


class ReplayLock:
    """One non-blocking process lease plus a pinned replay root directory."""

    def __init__(
        self,
        descriptor: int,
        root_descriptor: int,
        display_root: Path,
    ) -> None:
        self._descriptor = descriptor
        self._root_descriptor = root_descriptor
        self.display_root = display_root

    @property
    def root(self) -> Path:
        if self._root_descriptor < 0:
            raise CandidateReplayError("replay directory lease is closed")
        return Path(f"/proc/self/fd/{self._root_descriptor}")

    def close(self) -> None:
        if self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1
        if self._root_descriptor >= 0:
            os.close(self._root_descriptor)
            self._root_descriptor = -1

    def __del__(self) -> None:
        self.close()


class PinnedReplayDirectory:
    """A directory descriptor used as an unretargetable host log path."""

    def __init__(self, descriptor: int) -> None:
        self._descriptor = descriptor

    @property
    def path(self) -> Path:
        if self._descriptor < 0:
            raise CandidateReplayError("pinned replay directory is closed")
        return Path(f"/proc/self/fd/{self._descriptor}")

    def close(self) -> None:
        if self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1

    def __enter__(self) -> "PinnedReplayDirectory":
        return self

    def __exit__(self, *unused: object) -> None:
        self.close()


def _open_child_directory(
    parent_descriptor: int,
    name: str,
    *,
    create: bool,
) -> int:
    if name in {"", ".", ".."} or "/" in name:
        raise CandidateReplayError("replay directory component is unsafe")
    created = False
    if create:
        try:
            os.mkdir(name, 0o755, dir_fd=parent_descriptor)
            created = True
        except FileExistsError:
            pass
        except OSError as exc:
            raise CandidateReplayError(
                "replay output path must contain only non-symlink directory components"
            ) from exc
    try:
        descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_descriptor)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise CandidateReplayError(
            "replay output path must contain only non-symlink directory components"
        ) from exc
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise CandidateReplayError(
            "replay output path must contain only non-symlink directory components"
        )
    if created:
        try:
            os.fsync(parent_descriptor)
        except OSError as exc:
            os.close(descriptor)
            raise CandidateReplayError(
                "failed to make replay directory durable"
            ) from exc
    return descriptor


def _anchored_path(path: Path) -> tuple[int, tuple[str, ...]]:
    parts = path.parts
    if len(parts) < 5 or parts[:4] != ("/", "proc", "self", "fd"):
        raise CandidateReplayError("replay writes require a pinned output root")
    try:
        descriptor = int(parts[4])
    except ValueError as exc:
        raise CandidateReplayError("pinned replay descriptor is invalid") from exc
    try:
        if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise CandidateReplayError("pinned replay descriptor is not a directory")
    except OSError as exc:
        raise CandidateReplayError("pinned replay descriptor is closed") from exc
    return descriptor, tuple(parts[5:])


def _open_directory_descriptor(path: Path, *, create: bool) -> int:
    root_descriptor, components = _anchored_path(path)
    current = os.dup(root_descriptor)
    try:
        for component in components:
            child = _open_child_directory(
                current,
                component,
                create=create,
            )
            os.close(current)
            current = child
        return current
    except Exception:
        os.close(current)
        raise


def pin_replay_directory(
    path: Path,
    *,
    create: bool = True,
    must_create: bool = False,
) -> PinnedReplayDirectory:
    if not must_create:
        return PinnedReplayDirectory(
            _open_directory_descriptor(path, create=create)
        )
    if not create:
        raise CandidateReplayError("must_create requires directory creation")
    _, components = _anchored_path(path)
    if not components:
        raise CandidateReplayError("cannot exclusively create the replay root")
    parent_descriptor = _open_directory_descriptor(path.parent, create=True)
    name = components[-1]
    try:
        try:
            os.mkdir(name, 0o700, dir_fd=parent_descriptor)
        except FileExistsError as exc:
            raise CandidateReplayError(
                "replay attempt directory already exists"
            ) from exc
        os.fsync(parent_descriptor)
        child_descriptor = _open_child_directory(
            parent_descriptor,
            name,
            create=False,
        )
        return PinnedReplayDirectory(child_descriptor)
    finally:
        os.close(parent_descriptor)


def acquire_replay_lock(root: Path, *, base: Path | None = None) -> ReplayLock:
    base_path = base if base is not None else root.parent
    try:
        base_path.mkdir(parents=True, exist_ok=True)
        canonical_base = base_path.resolve(strict=True)
        lexical_base = base_path.absolute()
        components = root.absolute().relative_to(lexical_base).parts
    except (OSError, ValueError) as exc:
        raise CandidateReplayError("replay root must be inside its output base") from exc
    try:
        base_descriptor = os.open(canonical_base, _DIRECTORY_FLAGS)
    except OSError as exc:
        raise CandidateReplayError(
            "replay output base must be a non-symlink directory"
        ) from exc

    flags = (
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    lock_identity = hashlib.sha256(
        b"sforge-replay-root-lock-v1\0"
        + "/".join(components).encode("utf-8")
    ).hexdigest()
    lock_name = f".sforge-replay-{lock_identity}.lock"
    descriptor = -1
    root_descriptor = -1
    try:
        descriptor = os.open(
            lock_name,
            flags,
            0o600,
            dir_fd=base_descriptor,
        )
    except OSError as exc:
        os.close(base_descriptor)
        raise CandidateReplayError(
            "replay lock must be a regular non-symlink file"
        ) from exc
    try:
        lock_stat = os.fstat(descriptor)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_uid != os.geteuid():
            raise CandidateReplayError(
                "replay lock must be an owner-controlled regular file"
            )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CandidateReplayError(
                "another process is already running this replay plan"
            ) from exc
        # The lease lives in the stable output base, keyed by the logical root
        # path.  Renaming the replay directory therefore cannot create a second
        # independent lock at the original pathname.
        root_descriptor = os.dup(base_descriptor)
        for component in components:
            child = _open_child_directory(
                root_descriptor,
                component,
                create=True,
            )
            os.close(root_descriptor)
            root_descriptor = child
        os.close(base_descriptor)
        base_descriptor = -1
        return ReplayLock(
            descriptor,
            root_descriptor,
            canonical_base.joinpath(*components),
        )
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        if root_descriptor >= 0:
            os.close(root_descriptor)
        if base_descriptor >= 0:
            os.close(base_descriptor)
        raise


def safe_replay_id(value: str, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise CandidateReplayError(
            f"{field} must be 1-200 letters, digits, dot, underscore, or dash"
        )
    return value


def _validated_context_pair(key: object, value: object) -> tuple[str, str]:
    if not isinstance(key, str) or _CONTEXT_KEY_RE.fullmatch(key) is None:
        raise CandidateReplayError(
            "evaluator context keys must be 1-128 safe ASCII characters"
        )
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_CONTEXT_VALUE_LENGTH
        or _CONTEXT_CONTROL_RE.search(value) is not None
    ):
        raise CandidateReplayError(
            "evaluator context values must be nonempty, control-free strings "
            "of at most 4096 characters"
        )
    return key, value


def parse_runtime_context(values: Sequence[str]) -> dict[str, str]:
    """Parse repeatable ``KEY=VALUE`` provenance without ambiguous keys."""

    result: dict[str, str] = {}
    for raw in values:
        if not isinstance(raw, str) or "=" not in raw:
            raise CandidateReplayError(
                "--evaluator-context must use KEY=VALUE"
            )
        key, value = _validated_context_pair(*raw.split("=", 1))
        if key in result:
            raise CandidateReplayError(
                f"duplicate evaluator context key: {key}"
            )
        result[key] = value
    return dict(sorted(result.items()))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def default_replay_run_id(
    *, task_id: str, origin_run_id: str, measurement_set: str
) -> str:
    identity = f"{task_id}\0{origin_run_id}\0{measurement_set}".encode("utf-8")
    digest = hashlib.sha256(identity).hexdigest()[:12]
    readable = f"{origin_run_id}-replay-{measurement_set}"
    if len(readable) <= 180 and _SAFE_ID_RE.fullmatch(readable) is not None:
        return readable
    return f"candidate-replay-{digest}"


def replay_submission_id(
    *,
    replay_run_id: str,
    measurement_set: str,
    candidate_id: str,
    attempt: int,
) -> str:
    """Return a Docker-safe ID unique across independent replay curves."""

    replay_run_id = safe_replay_id(replay_run_id, field="replay run ID")
    measurement_set = safe_replay_id(measurement_set, field="measurement set")
    match = _CANDIDATE_ID_RE.fullmatch(candidate_id)
    if match is None:
        raise CandidateReplayError("candidate ID is not canonical")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise CandidateReplayError("replay attempt must be a positive integer")
    identity = "\0".join(
        (replay_run_id, measurement_set, candidate_id, str(attempt))
    ).encode("utf-8")
    namespace = hashlib.sha256(identity).hexdigest()[:20]
    return f"replay-{match.group(1)[:10]}-{namespace}-{attempt}"


def input_version(candidate: StoredCandidate) -> StoredVersion:
    """Choose a stable exact archive for one normalized code candidate."""

    # Pin replay provenance to the event that first introduced the logical
    # candidate.  Preferring a checkpoint would let a checkpoint recorded
    # later silently change an already-created replay plan even though the
    # submitted tree itself is identical.
    return candidate.events[0]


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _evaluator_source_sha256() -> str:
    harness_root = Path(__file__).resolve().parent
    repository_package = harness_root.parent
    repository_root = repository_package.parent
    selected = sorted(
        path
        for path in repository_package.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    selected.extend(
        path
        for path in (repository_root / "pyproject.toml", repository_root / "uv.lock")
        if path.exists()
    )
    runtime_packages = sorted(
        (
            str(distribution.metadata.get("Name") or ""),
            str(distribution.version),
        )
        for distribution in importlib.metadata.distributions()
    )
    runtime_identity = json.dumps(
        {
            "python_implementation": platform.python_implementation(),
            "python_version": list(sys.version_info[:3]),
            "packages": runtime_packages,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    digest = hashlib.sha256(b"sforge-candidate-evaluator-runtime-v2\0")
    digest.update(len(runtime_identity).to_bytes(8, "big"))
    digest.update(runtime_identity)
    for path in selected:
        try:
            if path.is_symlink() or not path.is_file():
                raise OSError("runtime source must be a regular non-symlink file")
            payload = path.read_bytes()
            relative = path.relative_to(repository_root).as_posix().encode()
        except (OSError, ValueError) as exc:
            raise CandidateReplayError(
                "SForge evaluator source identity is unavailable"
            ) from exc
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return f"sha256:{digest.hexdigest()}"


def _loaded_sforge_module_names() -> list[str]:
    """Return the already-imported SForge modules whose bytecode is executing."""

    return sorted(
        name
        for name, module in sys.modules.items()
        if (name == "sforge" or name.startswith("sforge."))
        and isinstance(module, types.ModuleType)
    )


def _loaded_module_name_is_allowed(name: str) -> bool:
    return name == "__main__" or name == "sforge" or name.startswith("sforge.")


def _canonical_code_constant(value: object) -> object:
    if isinstance(value, types.CodeType):
        return {"kind": "code", "value": _canonical_code_value(value)}
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_canonical_code_constant(item) for item in value],
        }
    if isinstance(value, frozenset):
        converted = [_canonical_code_constant(item) for item in value]
        converted.sort(
            key=lambda item: json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        return {"kind": "frozenset", "items": converted}
    if value is None:
        return {"kind": "none"}
    if value is Ellipsis:
        return {"kind": "ellipsis"}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": str(value)}
    if isinstance(value, float):
        return {"kind": "float", "value": value.hex()}
    if isinstance(value, complex):
        return {
            "kind": "complex",
            "real": value.real.hex(),
            "imag": value.imag.hex(),
        }
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    if isinstance(value, bytes):
        return {"kind": "bytes", "value": value.hex()}
    raise CandidateReplayError(
        "loaded evaluator code contains an unsupported constant"
    )


def _canonical_code_value(code: types.CodeType) -> dict[str, object]:
    """Represent code without CPython's mutable quickening state."""

    return {
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "code": code.co_code.hex(),
        "constants": [
            _canonical_code_constant(value) for value in code.co_consts
        ],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
        "name": code.co_name,
        "qualname": getattr(code, "co_qualname", code.co_name),
        "firstlineno": code.co_firstlineno,
        "linetable": code.co_linetable.hex(),
        "exceptiontable": getattr(code, "co_exceptiontable", b"").hex(),
    }


def _code_identity_bytes(code: types.CodeType) -> bytes:
    return json.dumps(
        _canonical_code_value(code),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _runtime_name_is_sensitive(name: str) -> bool:
    upper = name.upper()
    return any(fragment in upper for fragment in _SENSITIVE_RUNTIME_NAME_FRAGMENTS)


def _runtime_name_is_ephemeral(name: str) -> bool:
    upper = name.upper()
    return any(fragment in upper for fragment in _EPHEMERAL_RUNTIME_NAME_FRAGMENTS)


def _runtime_binding_is_ephemeral(name: str, value: object) -> bool:
    del value
    return _runtime_name_is_ephemeral(name)


def _runtime_type_identity(value: object) -> dict[str, str]:
    value_type = value if isinstance(value, type) else type(value)
    return {
        "module": value_type.__module__,
        "qualname": value_type.__qualname__,
    }


def _stable_loaded_value(
    value: object,
    *,
    seen: set[int] | None = None,
    depth: int = 0,
) -> object:
    """Return a deterministic JSON identity for loaded runtime state.

    This structure is only fed into the aggregate evaluator digest. It is not
    persisted in the replay plan. Unsupported runtime objects use their type
    identity instead of a repr that may contain a process-local address.
    """

    if seen is None:
        seen = set()
    if isinstance(value, Enum):
        return {
            "enum": {
                "type": _runtime_type_identity(value),
                "name": value.name,
                "value": _stable_loaded_value(
                    value.value,
                    seen=seen,
                    depth=depth + 1,
                ),
            }
        }
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return {"float": "nan"}
        if value == math.inf:
            return {"float": "+inf"}
        if value == -math.inf:
            return {"float": "-inf"}
        return value
    if isinstance(value, bytes):
        return {
            "bytes_length": len(value),
            "bytes_sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, re.Pattern):
        return {
            "regex": {
                "pattern": _stable_loaded_value(
                    value.pattern,
                    seen=seen,
                    depth=depth + 1,
                ),
                "flags": value.flags,
            }
        }
    if isinstance(value, PurePath):
        return {
            "path": {
                "type": _runtime_type_identity(value),
                "value": value.as_posix(),
            }
        }
    if isinstance(value, types.ModuleType):
        return {"module": value.__name__}
    if isinstance(value, type):
        return {"type": _runtime_type_identity(value)}
    if depth >= _MAX_LOADED_VALUE_DEPTH:
        return {"depth_limited_type": _runtime_type_identity(value)}

    tracked = isinstance(
        value,
        (
            tuple,
            list,
            set,
            frozenset,
            dict,
            types.FunctionType,
            functools.partial,
        ),
    ) or (is_dataclass(value) and not isinstance(value, type)) or hasattr(
        value, "__dict__"
    )
    identity = id(value)
    if tracked and identity in seen:
        return {"cycle_type": _runtime_type_identity(value)}
    if tracked:
        seen.add(identity)
    try:
        if isinstance(value, (tuple, list)):
            return {
                "sequence_type": type(value).__name__,
                "items": [
                    _stable_loaded_value(item, seen=seen, depth=depth + 1)
                    for item in value
                ],
            }
        if isinstance(value, (set, frozenset)):
            converted = [
                _stable_loaded_value(item, seen=seen, depth=depth + 1)
                for item in value
            ]
            return {
                "set_type": type(value).__name__,
                "items": sorted(
                    converted,
                    key=lambda item: json.dumps(
                        item,
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                ),
            }
        if isinstance(value, dict):
            entries: list[dict[str, object]] = []
            for key, item in value.items():
                stable_key = _stable_loaded_value(
                    key,
                    seen=seen,
                    depth=depth + 1,
                )
                stable_item: object
                if isinstance(key, str) and _runtime_name_is_sensitive(key):
                    stable_item = {"redacted_sensitive_value": True}
                else:
                    stable_item = _stable_loaded_value(
                        item,
                        seen=seen,
                        depth=depth + 1,
                    )
                entries.append({"key": stable_key, "value": stable_item})
            entries.sort(
                key=lambda item: json.dumps(
                    item,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
            return {"mapping": entries}
        if isinstance(value, functools.partial):
            return {
                "partial": {
                    "function": _stable_loaded_value(
                        value.func,
                        seen=seen,
                        depth=depth + 1,
                    ),
                    "args": _stable_loaded_value(
                        value.args,
                        seen=seen,
                        depth=depth + 1,
                    ),
                    "keywords": _stable_loaded_value(
                        value.keywords or {},
                        seen=seen,
                        depth=depth + 1,
                    ),
                }
            }
        if isinstance(value, types.FunctionType):
            return {
                "function": {
                    "module": value.__module__,
                    "qualname": value.__qualname__,
                    "code_sha256": hashlib.sha256(
                        _code_identity_bytes(value.__code__)
                    ).hexdigest(),
                    "bindings": _function_runtime_bindings(
                        value,
                        seen=seen,
                        depth=depth + 1,
                    ),
                }
            }
        if is_dataclass(value) and not isinstance(value, type):
            dataclass_state: list[dict[str, object]] = []
            for field in fields(value):
                field_value: object
                if _runtime_name_is_sensitive(field.name):
                    field_value = {"redacted_sensitive_value": True}
                else:
                    field_value = _stable_loaded_value(
                        getattr(value, field.name),
                        seen=seen,
                        depth=depth + 1,
                    )
                dataclass_state.append(
                    {"name": field.name, "value": field_value}
                )
            return {
                "dataclass": {
                    "type": _runtime_type_identity(value),
                    "fields": dataclass_state,
                }
            }
        try:
            object_vars = vars(value)
        except TypeError:
            object_vars = None
        if isinstance(object_vars, dict):
            attributes: list[dict[str, object]] = []
            for name, item in sorted(object_vars.items()):
                if _runtime_name_is_sensitive(name):
                    stable_item: object = {"redacted_sensitive_value": True}
                else:
                    stable_item = _stable_loaded_value(
                        item,
                        seen=seen,
                        depth=depth + 1,
                    )
                attributes.append({"name": name, "value": stable_item})
            return {
                "object": {
                    "type": _runtime_type_identity(value),
                    "attributes": attributes,
                }
            }
        return {"object_type": _runtime_type_identity(value)}
    finally:
        if tracked:
            seen.remove(identity)


def _loaded_global_names(code: types.CodeType) -> list[str]:
    """Collect globals loaded by a function and all nested code objects."""

    names = {
        instruction.argval
        for instruction in dis.get_instructions(code)
        if instruction.opname in {"LOAD_GLOBAL", "LOAD_NAME"}
        and isinstance(instruction.argval, str)
    }
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            names.update(_loaded_global_names(constant))
    return sorted(names)


def _function_runtime_bindings(
    function: types.FunctionType,
    *,
    seen: set[int] | None = None,
    depth: int = 0,
) -> dict[str, object]:
    """Commit executable bindings that are not stored in a code object."""

    if seen is None:
        seen = set()
    code = function.__code__
    positional_defaults: list[dict[str, object]] = []
    defaults = function.__defaults__ or ()
    first_default = code.co_argcount - len(defaults)
    for offset, value in enumerate(defaults):
        name = code.co_varnames[first_default + offset]
        positional_defaults.append(
            {
                "name": name,
                "value": (
                    {"redacted_sensitive_value": True}
                    if _runtime_name_is_sensitive(name)
                    else _stable_loaded_value(value, seen=seen, depth=depth + 1)
                ),
            }
        )

    keyword_defaults: list[dict[str, object]] = []
    for name, value in sorted((function.__kwdefaults__ or {}).items()):
        keyword_defaults.append(
            {
                "name": name,
                "value": (
                    {"redacted_sensitive_value": True}
                    if _runtime_name_is_sensitive(name)
                    else _stable_loaded_value(value, seen=seen, depth=depth + 1)
                ),
            }
        )

    closure_values: list[dict[str, object]] = []
    closure = function.__closure__ or ()
    for name, cell in zip(code.co_freevars, closure):
        try:
            value = cell.cell_contents
        except ValueError:
            stable_value: object = {"empty_cell": True}
        else:
            stable_value = (
                {"redacted_sensitive_value": True}
                if _runtime_name_is_sensitive(name)
                else _stable_loaded_value(value, seen=seen, depth=depth + 1)
            )
        closure_values.append({"name": name, "value": stable_value})

    global_values: list[dict[str, object]] = []
    for name in _loaded_global_names(code):
        if name not in function.__globals__ or _runtime_name_is_sensitive(name):
            continue
        if _runtime_binding_is_ephemeral(name, function.__globals__[name]):
            stable_value: object = {
                "ephemeral_binding_type": _runtime_type_identity(
                    function.__globals__[name]
                )
            }
        else:
            stable_value = _stable_loaded_value(
                function.__globals__[name],
                seen=seen,
                depth=depth + 1,
            )
        global_values.append(
            {
                "name": name,
                "value": stable_value,
            }
        )

    return {
        "positional_defaults": positional_defaults,
        "keyword_defaults": keyword_defaults,
        "closure": closure_values,
        "globals": global_values,
    }


def _loaded_evaluator_code_sha256(
    module_names: Sequence[str] | None = None,
) -> str:
    """Hash executing bytecode so import-A/disk-B cannot claim source B alone."""

    names = list(module_names or _loaded_sforge_module_names())
    if names != sorted(set(names)) or any(
        not _loaded_module_name_is_allowed(name) for name in names
    ):
        raise CandidateReplayError("loaded SForge module identity is invalid")
    digest = hashlib.sha256(b"sforge-loaded-evaluator-code-v2\0")

    def add_function(identity: str, function: types.FunctionType) -> None:
        payload = _code_identity_bytes(function.__code__)
        encoded_identity = identity.encode("utf-8")
        digest.update(len(encoded_identity).to_bytes(4, "big"))
        digest.update(encoded_identity)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        bindings = json.dumps(
            _function_runtime_bindings(function),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        digest.update(len(bindings).to_bytes(8, "big"))
        digest.update(bindings)

    def add_runtime_value(identity: str, value: object) -> None:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        encoded_identity = identity.encode("utf-8")
        digest.update(len(encoded_identity).to_bytes(4, "big"))
        digest.update(encoded_identity)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    for module_name in names:
        module = sys.modules.get(module_name)
        if not isinstance(module, types.ModuleType):
            raise CandidateReplayError("loaded SForge module identity changed")
        encoded_module = module_name.encode("utf-8")
        digest.update(len(encoded_module).to_bytes(4, "big"))
        digest.update(encoded_module)
        for name, value in sorted(vars(module).items()):
            if isinstance(value, types.FunctionType):
                add_function(f"{module_name}:{name}", value)
            elif isinstance(value, type) and value.__module__ == module_name:
                for attribute_name, attribute in sorted(vars(value).items()):
                    functions: list[types.FunctionType] = []
                    if isinstance(attribute, (staticmethod, classmethod)):
                        functions = [attribute.__func__]
                    elif isinstance(attribute, types.FunctionType):
                        functions = [attribute]
                    elif isinstance(attribute, property):
                        functions = [
                            function
                            for function in (
                                attribute.fget,
                                attribute.fset,
                                attribute.fdel,
                            )
                            if isinstance(function, types.FunctionType)
                        ]
                    for function in functions:
                        add_function(
                            f"{module_name}:{name}.{attribute_name}",
                            function,
                        )
                    if (
                        attribute_name.startswith("__")
                        or functions
                        or isinstance(attribute, property)
                        or callable(attribute)
                    ):
                        continue
                    stable_attribute: object
                    if _runtime_name_is_sensitive(attribute_name):
                        stable_attribute = {"redacted_sensitive_value": True}
                    elif _runtime_binding_is_ephemeral(
                        attribute_name,
                        attribute,
                    ):
                        stable_attribute = {
                            "ephemeral_binding_type": _runtime_type_identity(
                                attribute
                            )
                        }
                    else:
                        stable_attribute = _stable_loaded_value(attribute)
                    add_runtime_value(
                        f"{module_name}:{name}.{attribute_name}:binding",
                        stable_attribute,
                    )
            elif (
                name.isupper() and not _runtime_name_is_sensitive(name)
            ):
                stable = _stable_loaded_value(value)
                add_runtime_value(f"{module_name}:{name}:binding", stable)
    return f"sha256:{digest.hexdigest()}"


def _loaded_evaluator_bytecode_sha256(
    module_names: Sequence[str] | None = None,
) -> str:
    """Hash loaded executable code without mutable process runtime state."""

    names = list(module_names or _loaded_sforge_module_names())
    if names != sorted(set(names)) or any(
        not _loaded_module_name_is_allowed(name) for name in names
    ):
        raise CandidateReplayError("loaded SForge module identity is invalid")
    digest = hashlib.sha256(b"sforge-loaded-evaluator-bytecode-v1\0")

    def add_function(identity: str, function: types.FunctionType) -> None:
        payload = _code_identity_bytes(function.__code__)
        encoded_identity = identity.encode("utf-8")
        digest.update(len(encoded_identity).to_bytes(4, "big"))
        digest.update(encoded_identity)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    for module_name in names:
        module = sys.modules.get(module_name)
        if not isinstance(module, types.ModuleType):
            raise CandidateReplayError("loaded SForge module identity changed")
        encoded_module = module_name.encode("utf-8")
        digest.update(len(encoded_module).to_bytes(4, "big"))
        digest.update(encoded_module)
        for name, value in sorted(vars(module).items()):
            if isinstance(value, types.FunctionType):
                add_function(f"{module_name}:{name}", value)
            elif isinstance(value, type) and value.__module__ == module_name:
                for attribute_name, attribute in sorted(vars(value).items()):
                    if isinstance(attribute, (staticmethod, classmethod)):
                        functions = [attribute.__func__]
                    elif isinstance(attribute, types.FunctionType):
                        functions = [attribute]
                    elif isinstance(attribute, property):
                        functions = [
                            function
                            for function in (
                                attribute.fget,
                                attribute.fset,
                                attribute.fdel,
                            )
                            if isinstance(function, types.FunctionType)
                        ]
                    else:
                        functions = []
                    for function in functions:
                        add_function(
                            f"{module_name}:{name}.{attribute_name}",
                            function,
                        )
    return f"sha256:{digest.hexdigest()}"


def loaded_main_source_sha256() -> str:
    """Hash the task runtime adapter executing as ``__main__``."""

    main_module = sys.modules.get("__main__")
    main_path_value = getattr(main_module, "__file__", None)
    if not isinstance(main_module, types.ModuleType) or not isinstance(
        main_path_value, str
    ):
        raise CandidateReplayError("task runtime adapter module is unavailable")
    main_path = Path(main_path_value)
    try:
        if main_path.is_symlink() or not main_path.is_file():
            raise OSError("runtime adapter must be a regular non-symlink file")
        return hashlib.sha256(main_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise CandidateReplayError(
            "task runtime adapter source identity is unavailable"
        ) from exc


def loaded_evaluator_snapshot(
    module_names: Sequence[str] | None = None,
    *,
    expected_main_sha256: str | None = None,
) -> dict[str, object]:
    """Freeze loaded evaluator configuration before backend/build mutation."""

    names = list(module_names or _loaded_sforge_module_names())
    if expected_main_sha256 is not None:
        if re.fullmatch(r"[0-9a-f]{64}", expected_main_sha256) is None:
            raise CandidateReplayError(
                "task runtime adapter identity is not a canonical SHA-256"
            )
        observed_main_sha256 = loaded_main_source_sha256()
        if observed_main_sha256 != expected_main_sha256:
            raise CandidateReplayError(
                "loaded task runtime adapter does not match its launcher identity"
            )
        names.append("__main__")
        names = sorted(set(names))
    return {
        "modules": names,
        "runtime_sha256": _loaded_evaluator_code_sha256(names),
        "bytecode_sha256": _loaded_evaluator_bytecode_sha256(names),
    }


def _task_contract(task_spec: TaskSpec) -> dict[str, Any]:
    contract = {
        "task_id": task_spec.task_id,
        "benchmark_name": task_spec.benchmark_name,
        "base_image": task_spec.base_image,
        "base_image_hash": task_spec.base_image_hash,
        "base_image_spec": task_spec.base_image_spec,
        "platform": task_spec.platform,
        "cwd": task_spec.cwd,
        "submit_paths": task_spec.submit_paths,
        "submit_exclude": task_spec.submit_exclude,
        "judge": {
            "eval_cmd": task_spec.judge.eval_cmd,
            "eval_timeout": task_spec.judge.eval_timeout,
            "parser": task_spec.judge.parser,
            "private_artifact_path": task_spec.judge.private_artifact_path,
            "setup_cmds": task_spec.judge.setup_cmds,
            "image_tag": task_spec.judge.image_tag,
            "game_server_cmd": task_spec.judge.game_server_cmd,
            "score_direction": task_spec.judge.score_direction,
            "selection": task_spec.judge.selection,
            "rescale": (
                asdict(task_spec.judge.rescale)
                if task_spec.judge.rescale is not None
                else None
            ),
            "cpu_limit": task_spec.judge.cpu_limit,
            "mem_limit": task_spec.judge.mem_limit,
        },
    }
    if task_spec.judge.private_artifact_paths is not None:
        contract['judge']['private_artifact_paths'] = task_spec.judge.private_artifact_paths
    if getattr(task_spec.judge, 'private_bootstrap', None) is not None:
        contract['judge']['private_bootstrap'] = task_spec.judge.private_bootstrap
    if task_spec.portable is not None:
        contract["portable"] = task_spec.portable.to_dict()
    elif task_spec.capability_envelope is not None:
        contract["capability_envelope"] = task_spec.capability_envelope.to_dict()
    return contract


def _backend_contract(config: SForgeConfig) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "backend": config.backend,
        "judge_network_mode": config.judge_network_mode,
    }
    if config.backend == "k8s":
        kubeconfig_sha256 = None
        if config.k8s_kubeconfig:
            try:
                kubeconfig_sha256 = f"sha256:{hashlib.sha256(Path(config.k8s_kubeconfig).read_bytes()).hexdigest()}"
            except OSError as exc:
                raise CandidateReplayError(
                    "Kubernetes evaluator kubeconfig identity is unavailable"
                ) from exc
        contract.update(
            {
                "namespace": config.k8s_namespace,
                "node_selector": config.k8s_node_selector,
                "image_registry": config.k8s_image_registry,
                "kubeconfig_sha256": kubeconfig_sha256,
            }
        )
    return contract


def evaluator_descriptor(
    task_spec: TaskSpec,
    config: SForgeConfig,
    *,
    timeout: int | None,
    runtime_context: Mapping[str, str] | None = None,
    resolved_image: ResolvedImage | None = None,
    loaded_snapshot: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    effective_timeout = (
        task_spec.judge.eval_timeout if timeout is None else timeout
    )
    if (
        not isinstance(task_spec.judge.selection, str)
        or task_spec.judge.selection not in _SELECTION_POLICIES
    ):
        raise CandidateReplayError("replay selection policy is invalid")
    if (
        not isinstance(task_spec.judge.score_direction, str)
        or task_spec.judge.score_direction not in {"maximize", "minimize"}
    ):
        raise CandidateReplayError("replay score direction is invalid")
    if (
        isinstance(effective_timeout, bool)
        or not isinstance(effective_timeout, int)
        or effective_timeout <= 0
    ):
        raise CandidateReplayError("replay evaluation timeout must be positive")
    evaluator_source_identity = _evaluator_source_sha256()
    snapshot = dict(
        loaded_evaluator_snapshot()
        if loaded_snapshot is None
        else loaded_snapshot
    )
    loaded_evaluator_modules = snapshot.get("modules")
    loaded_evaluator_identity = snapshot.get("runtime_sha256")
    loaded_evaluator_bytecode_identity = snapshot.get("bytecode_sha256")
    if (
        not isinstance(loaded_evaluator_modules, list)
        or loaded_evaluator_modules != sorted(set(loaded_evaluator_modules))
        or any(not isinstance(name, str) for name in loaded_evaluator_modules)
        or not isinstance(loaded_evaluator_identity, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", loaded_evaluator_identity)
        is None
        or not isinstance(loaded_evaluator_bytecode_identity, str)
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            loaded_evaluator_bytecode_identity,
        )
        is None
    ):
        raise CandidateReplayError("loaded evaluator snapshot is invalid")
    if any(
        not _loaded_module_name_is_allowed(name)
        for name in loaded_evaluator_modules
    ):
        raise CandidateReplayError(
            "loaded evaluator snapshot contains an invalid module"
        )
    if _loaded_evaluator_bytecode_sha256(
        loaded_evaluator_modules
    ) != loaded_evaluator_bytecode_identity:
        raise CandidateReplayError(
            "loaded SForge evaluator code changed after its startup snapshot"
        )
    observed_modules = _loaded_sforge_module_names()
    if "__main__" in loaded_evaluator_modules:
        observed_modules = sorted([*observed_modules, "__main__"])
    if observed_modules != loaded_evaluator_modules:
        raise CandidateReplayError(
            "loaded SForge evaluator module set changed after its startup snapshot"
        )
    contract = {
        "task_id": task_spec.task_id,
        "judge_image": task_spec.judge_image_key,
        "eval_cmd_sha256": hashlib.sha256(
            task_spec.judge.eval_cmd.encode("utf-8")
        ).hexdigest(),
        "eval_timeout": effective_timeout,
        "private_artifact_path": task_spec.judge.private_artifact_path,
        "submit_paths": task_spec.submit_paths,
        "submit_exclude": task_spec.submit_exclude,
        "backend": config.backend,
        "judge_cpu_limit": (
            config.judge_cpu_limit
            if config.judge_cpu_limit is not None
            else task_spec.judge.cpu_limit
        ),
        "judge_mem_limit": (
            config.judge_mem_limit
            if config.judge_mem_limit is not None
            else task_spec.judge.mem_limit
        ),
        "score_direction": task_spec.judge.score_direction,
        "selection": task_spec.judge.selection,
        "contract_schema_version": 3,
        "task_contract_sha256": _canonical_sha256(
            _task_contract(task_spec)
        ),
        "backend_contract_sha256": _canonical_sha256(
            _backend_contract(config)
        ),
        "judge_environment_sha256": _canonical_sha256(
            get_container_env(config, include_judge_extra=True)
        ),
        "evaluator_source_sha256": evaluator_source_identity,
        "loaded_evaluator_code_sha256": loaded_evaluator_identity,
        "loaded_evaluator_bytecode_sha256": (
            loaded_evaluator_bytecode_identity
        ),
        "loaded_evaluator_modules": loaded_evaluator_modules,
    }
    if task_spec.portable is not None:
        contract["portable_contract"] = task_spec.portable.digest_descriptor()
    # Normalized, host-independent CPU pinning identity, present only when a
    # Judge policy is set: unpinned descriptors keep their digests, and a
    # pinned replay can never continue an unpinned measurement set.
    judge_cpu_affinity = policy_for_role("judge", config)
    if judge_cpu_affinity is not None:
        contract["judge_cpu_affinity"] = judge_cpu_affinity.policy_id
    if resolved_image is None:
        contract["resolved_judge_image"] = {
            "backend": config.backend,
            "requested_ref": task_spec.judge_image_key,
            "execution_ref": None,
            "content_identity": None,
            "identity_kind": "unresolved-preview",
        }
    else:
        if resolved_image.backend != config.backend:
            raise CandidateReplayError(
                "resolved Judge image backend does not match replay backend"
            )
        if resolved_image.requested_ref != task_spec.judge_image_key:
            raise CandidateReplayError(
                "resolved Judge image does not match the task Judge image"
            )
        if not all(
            isinstance(value, str) and value
            for value in (
                resolved_image.execution_ref,
                resolved_image.content_identity,
                resolved_image.identity_kind,
            )
        ):
            raise CandidateReplayError(
                "resolved Judge image identity is incomplete"
            )
        contract["resolved_judge_image"] = asdict(resolved_image)
    if runtime_context:
        contract["runtime_context"] = dict(
            sorted(
                _validated_context_pair(key, value)
                for key, value in runtime_context.items()
            )
        )
        claimed_image = contract["runtime_context"].get("judge_image_id")
        if (
            claimed_image is not None
            and resolved_image is not None
            and claimed_image != resolved_image.content_identity
        ):
            raise CandidateReplayError(
                "wrapper Judge image identity does not match the image "
                "resolved by SForge"
            )
    if _evaluator_source_sha256() != evaluator_source_identity:
        raise CandidateReplayError(
            "SForge evaluator source changed while its identity was being pinned"
        )
    observed_modules = _loaded_sforge_module_names()
    if "__main__" in loaded_evaluator_modules:
        observed_modules = sorted([*observed_modules, "__main__"])
    if observed_modules != loaded_evaluator_modules or (
        _loaded_evaluator_bytecode_sha256(loaded_evaluator_modules)
        != loaded_evaluator_bytecode_identity
    ):
        raise CandidateReplayError(
            "loaded SForge evaluator code changed while its identity was being pinned"
        )
    canonical = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return {
        **contract,
        "fingerprint": f"sha256:{hashlib.sha256(canonical).hexdigest()}",
    }


def replay_root(
    log_dir: Path,
    *,
    replay_run_id: str,
    task_id: str,
    origin_run_id: str,
    measurement_set: str,
) -> Path:
    return (
        log_dir
        / "runs"
        / replay_run_id
        / task_id
        / "candidate-replays"
        / origin_run_id
        / measurement_set
    )


def candidate_measurement_path(root: Path, candidate: StoredCandidate) -> Path:
    return root / candidate.candidate_id / "measurement.json"


def _encode_replay_json(payload: Mapping[str, Any]) -> bytes:
    try:
        encoded = (
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise CandidateReplayError("failed to encode replay JSON") from exc
    if len(encoded) > _MAX_REPLAY_JSON_BYTES:
        raise CandidateReplayError("replay JSON exceeds the size limit")
    return encoded


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.name in {"", ".", ".."} or "/" in path.name:
        raise CandidateReplayError("replay JSON filename is unsafe")
    encoded = _encode_replay_json(payload)
    directory_fd = _open_directory_descriptor(path.parent, create=True)
    temporary_name = f".{path.name}.{uuid.uuid4().hex}.tmp"
    temporary_fd = -1
    try:
        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o644,
            dir_fd=directory_fd,
        )
        offset = 0
        while offset < len(encoded):
            written = os.write(temporary_fd, encoded[offset:])
            if written <= 0:
                raise OSError("replay JSON write made no progress")
            offset += written
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = -1
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        os.fsync(directory_fd)
    except (OSError, TypeError, ValueError) as exc:
        raise CandidateReplayError("failed to persist replay JSON") from exc
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        finally:
            os.close(directory_fd)


def _read_json(
    path: Path,
    *,
    label: str,
    missing_ok: bool,
) -> dict[str, Any] | None:
    try:
        directory_fd = _open_directory_descriptor(path.parent, create=False)
    except FileNotFoundError:
        if missing_ok:
            return None
        raise CandidateReplayError(f"{label} is missing")
    descriptor = -1
    try:
        try:
            descriptor = os.open(path.name, _READ_FLAGS, dir_fd=directory_fd)
        except FileNotFoundError:
            if missing_ok:
                return None
            raise CandidateReplayError(f"{label} is missing")
        except OSError as exc:
            raise CandidateReplayError(
                f"{label} must be a regular non-symlink file"
            ) from exc
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_size > _MAX_REPLAY_JSON_BYTES
        ):
            raise CandidateReplayError(
                f"{label} must be a bounded regular non-symlink file"
            )
        chunks: list[bytes] = []
        remaining = _MAX_REPLAY_JSON_BYTES
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, remaining + 1))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
            if remaining < 0:
                raise CandidateReplayError(f"{label} exceeds the size limit")
        try:
            value = json.loads(
                b"".join(chunks).decode("utf-8"),
                parse_constant=_reject_nonfinite_json_constant,
            )
            contains_nonfinite = _contains_nonfinite_json_number(value)
        except (
            UnicodeError,
            json.JSONDecodeError,
            ValueError,
            OverflowError,
            RecursionError,
        ) as exc:
            raise CandidateReplayError(f"{label} is unreadable") from exc
        if contains_nonfinite:
            raise CandidateReplayError(f"{label} contains a non-finite number")
        if not isinstance(value, dict):
            raise CandidateReplayError(f"{label} must contain a JSON object")
        return value
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory_fd)


def _reject_nonfinite_json_constant(constant: str) -> None:
    raise ValueError(f"non-finite JSON constant: {constant}")


def _contains_nonfinite_json_number(value: object) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, list):
        return any(_contains_nonfinite_json_number(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_nonfinite_json_number(item) for item in value.values())
    return False


def replay_file_exists(path: Path, *, label: str) -> bool:
    return _read_json(path, label=label, missing_ok=True) is not None


def read_replay_plan(path: Path) -> dict[str, Any]:
    """Read one bounded, no-follow replay plan and validate its identity."""

    value = _read_json(path, label="replay plan", missing_ok=False)
    if (
        value is None
        or value.get("schema_version") != MEASUREMENT_SCHEMA_VERSION
        or value.get("kind") != PLAN_KIND
        or not isinstance(value.get("evaluator"), dict)
    ):
        raise CandidateReplayError("replay plan schema identity is invalid")
    return value


def resolved_image_from_replay_plan(
    plan: Mapping[str, Any],
    *,
    expected_backend: str,
    expected_requested_ref: str,
) -> ResolvedImage:
    """Recover the immutable execution ref committed by an existing plan."""

    evaluator = plan.get("evaluator")
    raw = (
        evaluator.get("resolved_judge_image")
        if isinstance(evaluator, Mapping)
        else None
    )
    expected_keys = {
        "backend",
        "requested_ref",
        "execution_ref",
        "content_identity",
        "identity_kind",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_keys:
        raise CandidateReplayError(
            "replay plan does not contain a valid pinned Judge image"
        )
    values = {key: raw.get(key) for key in expected_keys}
    if any(
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_CONTEXT_VALUE_LENGTH
        or _CONTEXT_CONTROL_RE.search(value) is not None
        for value in values.values()
    ):
        raise CandidateReplayError(
            "replay plan does not contain a valid pinned Judge image"
        )
    if (
        values["backend"] != expected_backend
        or values["requested_ref"] != expected_requested_ref
    ):
        raise CandidateReplayError(
            "replay plan pinned Judge image does not match this task/backend"
        )
    return ResolvedImage(
        backend=values["backend"],
        requested_ref=values["requested_ref"],
        execution_ref=values["execution_ref"],
        content_identity=values["content_identity"],
        identity_kind=values["identity_kind"],
    )


def write_measurement(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_json(path, payload)


def ensure_replay_plan(root: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    """Create one immutable replay plan, or require an exact existing match."""

    path = root / "replay_plan.json"
    observed = _read_json(path, label="replay plan", missing_ok=True)
    if observed is not None:
        comparable = dict(observed)
        comparable.pop("created_at", None)
        if comparable != dict(expected):
            raise CandidateReplayError(
                "existing replay plan changed; choose a new --measurement-set "
                "or --replay-run-id"
            )
        return observed

    payload = {**dict(expected), "created_at": utc_now()}
    _atomic_write_json(path, payload)
    return payload


def read_measurement(path: Path) -> dict[str, Any] | None:
    value = _read_json(path, label="measurement record", missing_ok=True)
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != MEASUREMENT_SCHEMA_VERSION
        or value.get("kind") != MEASUREMENT_KIND
    ):
        raise CandidateReplayError(f"invalid measurement record: {path}")
    return value


VERSION_MEASUREMENT_KIND = "sforge-version-measurement"


def read_version_measurement(path: Path) -> dict[str, Any] | None:
    """Read one per-version replay measurement record."""

    value = _read_json(path, label="version measurement record", missing_ok=True)
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("kind") != VERSION_MEASUREMENT_KIND
    ):
        raise CandidateReplayError(f"invalid version measurement record: {path}")
    return value


def validate_existing_measurement(
    measurement: Mapping[str, Any],
    *,
    candidate: StoredCandidate,
    origin_run_id: str,
    replay_run_id: str,
    measurement_set: str,
    evaluator: Mapping[str, Any],
) -> None:
    try:
        _encode_replay_json(dict(measurement))
    except CandidateReplayError as exc:
        raise CandidateReplayError(
            "measurement contains a non-canonical JSON value"
        ) from exc
    if (
        measurement.get("schema_version") != MEASUREMENT_SCHEMA_VERSION
        or measurement.get("kind") != MEASUREMENT_KIND
    ):
        raise CandidateReplayError("measurement schema identity is invalid")
    candidate_value = measurement.get("candidate")
    if not isinstance(candidate_value, dict):
        raise CandidateReplayError("measurement candidate identity is missing")
    expected = {
        "candidate_id": candidate.candidate_id,
        "content_sha256": candidate.content_sha256,
        "origin_run_id": origin_run_id,
        "replay_run_id": replay_run_id,
        "measurement_set": measurement_set,
    }
    observed = {
        "candidate_id": candidate_value.get("candidate_id"),
        "content_sha256": candidate_value.get("content_sha256"),
        "origin_run_id": measurement.get("origin_run_id"),
        "replay_run_id": measurement.get("replay_run_id"),
        "measurement_set": measurement.get("measurement_set"),
    }
    if observed != expected:
        raise CandidateReplayError(
            "existing measurement belongs to a different replay identity"
        )
    if {
        "sequence": candidate_value.get("sequence"),
        "version": candidate_value.get("version"),
    } != {
        "sequence": candidate.sequence,
        "version": f"v{candidate.sequence}",
    }:
        raise CandidateReplayError("measurement candidate sequence is invalid")
    expected_measurement_id = (
        f"{measurement_set}-{candidate.candidate_id}-attempt-"
        f"{measurement.get('attempt')}"
    )
    if measurement.get("measurement_id") != expected_measurement_id:
        raise CandidateReplayError("measurement ID is invalid")

    attempt = measurement.get("attempt")
    if (
        isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or not 1 <= attempt <= 999_999
    ):
        raise CandidateReplayError("measurement attempt must be a positive integer")

    def valid_timestamp(value: object) -> bool:
        if not isinstance(value, str) or not value.endswith("Z"):
            return False
        try:
            parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
        except ValueError:
            return False
        return parsed.tzinfo is not None

    if not valid_timestamp(measurement.get("started_at")):
        raise CandidateReplayError("measurement start timestamp is invalid")

    version = input_version(candidate)
    expected_input = {
        "version_id": version.version_id,
        "archive_sha256": version.archive_sha256,
        "archive_size": version.archive_size,
        "source": version.source,
        "round_id": version.round_id,
    }
    if measurement.get("input_version") != expected_input:
        raise CandidateReplayError("measurement input archive identity is invalid")

    old_evaluator = measurement.get("evaluator")
    if not isinstance(old_evaluator, dict) or old_evaluator != dict(evaluator):
        raise CandidateReplayError(
            "existing measurement used a different evaluator; choose a new "
            "--measurement-set"
        )

    status = measurement.get("status")
    report = measurement.get("report")
    completed_at = measurement.get("completed_at")
    if status == "running":
        if report is not None or completed_at is not None:
            raise CandidateReplayError("running measurement has terminal fields")
        return
    if status not in {"completed", "failed"}:
        raise CandidateReplayError("measurement status is invalid")
    if not valid_timestamp(completed_at) or not isinstance(report, dict):
        raise CandidateReplayError("terminal measurement is incomplete")
    report_task_id = report.get("task_id")
    expected_task_id = evaluator.get("task_id")
    if report_task_id != expected_task_id:
        raise CandidateReplayError("measurement report task identity is invalid")
    report_submission_id = report.get("submission_id")
    expected_submission_id = replay_submission_id(
        replay_run_id=replay_run_id,
        measurement_set=measurement_set,
        candidate_id=candidate.candidate_id,
        attempt=attempt,
    )
    if report_submission_id != expected_submission_id:
        raise CandidateReplayError(
            "measurement report submission identity is invalid"
        )
    valid = report.get("valid")
    score = report.get("score")
    score_0_100 = report.get("score_0_100")
    pass_rate = report.get("pass_rate")
    if not isinstance(valid, bool):
        raise CandidateReplayError("measurement report validity is invalid")
    if score is not None and _finite_score(score) is None:
        raise CandidateReplayError("measurement report score is invalid")
    if score_0_100 is not None:
        finite_score_0_100 = _finite_score(score_0_100)
        if finite_score_0_100 is None or not 0.0 <= finite_score_0_100 <= 100.0:
            raise CandidateReplayError(
                "measurement report normalized score is invalid"
            )
    if (
        isinstance(pass_rate, bool)
        or not isinstance(pass_rate, (int, float))
        or not math.isfinite(float(pass_rate))
        or not 0.0 <= float(pass_rate) <= 1.0
    ):
        raise CandidateReplayError("measurement report pass rate is invalid")
    for field in ("total_tests", "passed", "failed", "errors"):
        value = report.get(field)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise CandidateReplayError(
                f"measurement report {field} count is invalid"
            )
    timed_out = report.get("timed_out")
    if timed_out is not None and not isinstance(timed_out, bool):
        raise CandidateReplayError("measurement report timeout flag is invalid")
    runtime_seconds = report.get("runtime_seconds")
    if runtime_seconds is not None and (
        isinstance(runtime_seconds, bool)
        or not isinstance(runtime_seconds, (int, float))
        or not math.isfinite(float(runtime_seconds))
        or float(runtime_seconds) < 0.0
    ):
        raise CandidateReplayError("measurement report runtime is invalid")
    metrics = report.get("metrics")
    if metrics is not None and not isinstance(metrics, dict):
        raise CandidateReplayError("measurement report metrics are invalid")
    infrastructure_failure = report_is_infrastructure_failure(report)
    if status == "failed" and not infrastructure_failure:
        raise CandidateReplayError("failed measurement is not an infrastructure failure")
    if status == "completed" and infrastructure_failure:
        raise CandidateReplayError("infrastructure failure was marked completed")


def invalid_terminal_report(
    *,
    task_id: str,
    submission_id: str,
    reason: str = "invalid_terminal_report",
    summary: str = "Evaluator returned an invalid terminal report",
) -> dict[str, Any]:
    """Return a sanitized retryable report for a broken evaluator contract."""

    return {
        "task_id": task_id,
        "submission_id": submission_id,
        "total_tests": 0,
        "passed": 0,
        "failed": 0,
        "errors": 1,
        "pass_rate": 0.0,
        "score": 0.0,
        "score_0_100": None,
        "timed_out": False,
        "runtime_seconds": 0.0,
        "test_details": [],
        "valid": False,
        "summary": summary,
        "details": [],
        "metrics": {
            "evaluation_status": "infrastructure_error",
            "infrastructure_reason": reason,
        },
        "submitted_at": 0.0,
    }


def recover_terminal_attempt(
    measurement_path: Path,
    current: Mapping[str, Any],
    *,
    candidate: StoredCandidate,
    origin_run_id: str,
    replay_run_id: str,
    measurement_set: str,
    evaluator: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Recover a terminal attempt written before top-level promotion crashed."""

    if current.get("status") != "running":
        return None
    attempt = current.get("attempt")
    if isinstance(attempt, bool) or not isinstance(attempt, int):
        raise CandidateReplayError("running measurement attempt is invalid")
    attempt_record = read_measurement(
        measurement_path.parent
        / f"attempt-{attempt:03d}"
        / "measurement.json"
    )
    if attempt_record is None:
        return None
    validate_existing_measurement(
        attempt_record,
        candidate=candidate,
        origin_run_id=origin_run_id,
        replay_run_id=replay_run_id,
        measurement_set=measurement_set,
        evaluator=evaluator,
    )
    if attempt_record.get("status") not in {"completed", "failed"}:
        raise CandidateReplayError("durable attempt record is not terminal")
    for field in (
        "measurement_id",
        "attempt",
        "started_at",
        "candidate",
        "input_version",
        "evaluator",
    ):
        if attempt_record.get(field) != current.get(field):
            raise CandidateReplayError(
                "durable attempt does not match its running measurement"
            )
    return attempt_record


def measurement_record(
    *,
    candidate: StoredCandidate,
    version: StoredVersion,
    origin_run_id: str,
    replay_run_id: str,
    measurement_set: str,
    evaluator: Mapping[str, Any],
    attempt: int,
    status: str,
    started_at: str,
    completed_at: str | None = None,
    report: Mapping[str, Any] | None = None,
    replicates: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": MEASUREMENT_SCHEMA_VERSION,
        "kind": MEASUREMENT_KIND,
        "measurement_id": (
            f"{measurement_set}-{candidate.candidate_id}-attempt-{attempt}"
        ),
        "measurement_set": measurement_set,
        "origin_run_id": origin_run_id,
        "replay_run_id": replay_run_id,
        "attempt": attempt,
        "status": status,
        "started_at": started_at,
        "completed_at": completed_at,
        "candidate": {
            "candidate_id": candidate.candidate_id,
            "content_sha256": candidate.content_sha256,
            "sequence": candidate.sequence,
            "version": f"v{candidate.sequence}",
            "first_seen_at": candidate.created_at,
            "label": candidate.label,
            "event_count": len(candidate.events),
        },
        "input_version": {
            "version_id": version.version_id,
            "archive_sha256": version.archive_sha256,
            "archive_size": version.archive_size,
            "source": version.source,
            "round_id": version.round_id,
        },
        "evaluator": dict(evaluator),
        "report": dict(report) if report is not None else None,
    }
    if replicates is not None:
        # Repeated measurements of one candidate. The record keeps its own
        # report as the representative attempt; the curve derives the median
        # and spread from these samples so benchmark noise is visible instead
        # of being read as a code improvement.
        record["replicates"] = [dict(sample) for sample in replicates]
    return record


def report_is_infrastructure_failure(report: Mapping[str, Any]) -> bool:
    if report.get("timed_out") is True:
        return True
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        return False
    if metrics.get("evaluation_status") in {
        "infrastructure_error",
        "resource_busy",
        "cancelled",
    }:
        return True
    if metrics.get("infrastructure_invalid") is True:
        return True
    failure_stage = metrics.get("failure_stage")
    failure_class = metrics.get("failure_class")
    failure_role = metrics.get("failure_role")
    return (
        failure_stage in {"reference_worker", "lifecycle", "controller"}
        or failure_class in {"gpu_busy", "infrastructure"}
        or (
            failure_stage == "telemetry_validation"
            and failure_class == "invalid_telemetry"
            and failure_role == "reference"
        )
    )


def _finite_score(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    score = float(value)
    return score if math.isfinite(score) else None


def _build_timeline_points(
    *,
    candidates: Sequence[StoredCandidate],
    measurements: Mapping[str, Mapping[str, Any]],
    score_direction: str,
    selection: str,
) -> list[dict[str, Any]]:
    """Project unique candidate measurements onto every physical visit event."""

    ordered_events: list[tuple[StoredVersion, StoredCandidate]] = []
    for candidate in candidates:
        for event in candidate.events:
            if event.event_sequence is None:
                raise CandidateReplayError(
                    f"stored event lacks a stable sequence: {event.version_id}"
                )
            ordered_events.append((event, candidate))
    ordered_events.sort(
        key=lambda item: (
            item[0].task_id,
            item[0].run_id,
            item[0].event_sequence or 0,
            item[0].version_id,
        )
    )

    previous_by_stream: dict[tuple[str, str], str] = {}
    first_seen: dict[tuple[str, str, str], int] = {}
    selection_entries: dict[tuple[str, str], list[dict[str, Any]]] = {}
    timeline: list[dict[str, Any]] = []
    for event, candidate in ordered_events:
        assert event.event_sequence is not None
        stream = (event.task_id, event.run_id)
        previous_candidate_id = previous_by_stream.get(stream)
        candidate_key = (*stream, candidate.candidate_id)
        first_seen_event = first_seen.get(candidate_key)
        if first_seen_event is None:
            relation = "first_seen"
            first_seen_event = event.event_sequence
            first_seen[candidate_key] = first_seen_event
        elif previous_candidate_id == candidate.candidate_id:
            relation = "same_as_previous"
        else:
            relation = "revisit"
        previous_by_stream[stream] = candidate.candidate_id

        measurement = measurements.get(candidate.candidate_id)
        status = (
            str(measurement.get("status", "planned"))
            if measurement is not None
            else "planned"
        )
        report = measurement.get("report") if measurement is not None else None
        if not isinstance(report, dict):
            report = {}
        linked_score = (
            _finite_score(report.get("score"))
            if status == "completed"
            else None
        )
        entries = selection_entries.setdefault(stream, [])
        if status == "completed":
            entries.append(
                {
                    "type": "submission",
                    "status": "completed",
                    "round": f"v{candidate.sequence}",
                    "score": linked_score,
                    "pass_rate": report.get("pass_rate"),
                    "valid": report.get("valid"),
                }
            )
        best = select_best(
            entries,
            score_direction=score_direction,
            policy=selection,
        )
        timeline.append(
            {
                "event_sequence": event.event_sequence,
                "event_id": event.version_id,
                "source": event.source,
                "round_id": event.round_id,
                "label": event.label,
                "saved_at": event.created_at,
                "version": f"v{candidate.sequence}",
                "candidate_sequence": candidate.sequence,
                "candidate_id": candidate.candidate_id,
                "content_sha256": candidate.content_sha256,
                "previous_candidate_id": previous_candidate_id,
                "relation": relation,
                "first_seen_event_sequence": first_seen_event,
                "linked_measurement_status": status,
                "linked_score": linked_score,
                "linked_pass_rate": (
                    report.get("pass_rate") if status == "completed" else None
                ),
                "linked_valid": (
                    report.get("valid") if status == "completed" else None
                ),
                "history_best": _finite_score(best.get("best_score")),
                "history_best_version": best.get("best_round") or None,
            }
        )
    return timeline


def rewrite_selector_archive(
    archive: bytes,
    *,
    selector_file: str,
    version_number: int,
) -> bytes:
    """Return the archive with its selector file naming ``version_number``.

    Per-version replay pins each implementation to the earliest event whose
    tree contains it — usually before the agent moved the selector to it, so
    the stored selector still names an older version. Judges (and their
    isolated deployment children, which independently re-derive the selection
    from the tree) score whatever the selector names, so the counterfactual
    "what if the selector had pointed at this implementation" is asked by
    rewriting the selector itself, not by asking every task's Judge to honor
    a side-channel override. Every other member is preserved byte-for-byte.
    """

    import tarfile as _tarfile

    if not 1 <= version_number <= 999:
        raise CandidateReplayError("selector rewrite version must be 1..999")
    payload = f"{version_number}\n".encode("ascii")
    output = io.BytesIO()
    replaced = False
    with _tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as source:
        with _tarfile.open(fileobj=output, mode="w:gz") as bundle:
            for member in source.getmembers():
                if member.isfile() and member.name == selector_file:
                    rewritten = copy.copy(member)
                    rewritten.size = len(payload)
                    bundle.addfile(rewritten, io.BytesIO(payload))
                    replaced = True
                    continue
                stream = source.extractfile(member) if member.isfile() else None
                bundle.addfile(member, stream)
    if not replaced:
        raise CandidateReplayError(
            f"archive has no selector file {selector_file!r} to rewrite"
        )
    return output.getvalue()


def enumerate_version_implementations(
    *,
    store,
    task_spec,
    run_id: str,
) -> list[dict[str, Any]]:
    """Enumerate every distinct retained implementation across a run's events.

    For tasks whose submitted tree retains one implementation per version and
    a selector file naming the scored one (``version_enumeration`` in the task
    spec), the meaningful hidden curve is one point per implementation — the
    same x-axis the agent's own public measurements used — not one point per
    checkpoint of the whole tree. A checkpoint whose selector never moved
    re-scores the same implementation, and implementations the selector never
    reached are otherwise never hidden-scored at all.

    An implementation is a directory (``directory_pattern``, as in
    candidates/kernel_vNNN/) or a single file (``file_pattern``, as in
    kernels/flash_kmeans_vNNN.py); tasks use one shape or the other.

    Returns one entry per distinct (version number, implementation content)
    pair, each pinned to the earliest stored event containing that exact
    content whose selector file parses (so the Judge's own admission of the
    archive cannot fail on an unrelated broken selector).
    """

    import re as _re
    import tarfile as _tarfile

    enumeration = task_spec.version_enumeration
    if not isinstance(enumeration, dict):
        raise CandidateReplayError(
            "task does not declare version_enumeration; per-version replay "
            "needs the implementation pattern and selector file"
        )
    root = str(enumeration.get("root") or "").strip("/")
    directory_text = str(enumeration.get("directory_pattern") or "")
    file_text = str(enumeration.get("file_pattern") or "")
    selector_name = str(enumeration.get("selector_file") or "")
    if not root or not selector_name:
        raise CandidateReplayError(
            "version_enumeration must declare root and selector_file"
        )
    if bool(directory_text) == bool(file_text):
        raise CandidateReplayError(
            "version_enumeration must declare exactly one of "
            "directory_pattern or file_pattern"
        )
    per_directory = bool(directory_text)
    pattern_key = "directory_pattern" if per_directory else "file_pattern"
    try:
        version_pattern = _re.compile(directory_text or file_text)
    except _re.error as exc:
        raise CandidateReplayError(
            f"invalid version_enumeration {pattern_key}: {exc}"
        ) from exc
    if version_pattern.groups != 1:
        raise CandidateReplayError(
            f"{pattern_key} must contain exactly one capture group with "
            "the version number"
        )

    events = store.list_versions(task=task_spec.task_id, run_id=run_id)
    if not events:
        raise CandidateReplayError(
            "no saved version events matched the task and origin run"
        )
    implementations: dict[tuple[int, str], dict[str, Any]] = {}
    for event in events:
        archive = store.read_archive(event)
        with _tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
            members = [item for item in bundle.getmembers() if item.isfile()]
            selector_raw = None
            per_version: dict[int, list[tuple[str, int, bytes]]] = {}
            for item in members:
                if item.name == selector_name:
                    stream = bundle.extractfile(item)
                    selector_raw = stream.read() if stream else None
                    continue
                parts = item.name.split("/")
                if parts[0] != root:
                    continue
                # A directory implementation owns everything beneath it; a
                # file implementation is exactly one entry under the root.
                if len(parts) < 3 if per_directory else len(parts) != 2:
                    continue
                match = version_pattern.fullmatch(parts[1])
                if match is None:
                    continue
                try:
                    number = int(match.group(1))
                except ValueError:
                    continue
                stream = bundle.extractfile(item)
                if stream is None:
                    continue
                relative = "/".join(parts[2:]) if per_directory else parts[1]
                per_version.setdefault(number, []).append(
                    (
                        relative,
                        item.mode & 0o7777,
                        hashlib.sha256(stream.read()).digest(),
                    )
                )
        # An event whose selector does not parse cannot pass the Judge's own
        # admission, so it cannot serve as the source archive for any
        # implementation; the same content usually also exists in a
        # neighboring event with a healthy selector.
        try:
            selector_value = int((selector_raw or b"").decode().strip())
        except (UnicodeDecodeError, ValueError):
            continue
        if not 1 <= selector_value <= 999 or str(selector_value) != (
            (selector_raw or b"").decode().strip()
        ):
            continue
        for number, records in per_version.items():
            digest = hashlib.sha256()
            for name, mode, file_digest in sorted(records):
                digest.update(name.encode("utf-8"))
                digest.update(mode.to_bytes(4, "big"))
                digest.update(file_digest)
            key = (number, digest.hexdigest())
            if key not in implementations:
                implementations[key] = {
                    "version_number": number,
                    "content_sha256": key[1],
                    "implementation_id": f"impl-v{number:03d}-{key[1][:12]}",
                    "first_event_id": event.version_id,
                    "first_event_round": event.round_id,
                    "first_event_label": event.label,
                    "first_seen_at": event.created_at,
                    "archive_sha256": event.archive_sha256,
                    "file_count": len(records),
                }
    ordered = sorted(
        implementations.values(),
        key=lambda item: (item["first_seen_at"], item["version_number"]),
    )
    for index, item in enumerate(ordered, start=1):
        item["sequence"] = index
    return ordered


def replicate_statistics(
    measurement: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Summarize repeated measurements of one candidate.

    Returns None unless the measurement carries at least two completed
    replicate scores, so single-measurement curves keep their exact previous
    shape.
    """

    if measurement is None:
        return None
    samples = measurement.get("replicates")
    if not isinstance(samples, list):
        return None
    scores = [
        value
        for sample in samples
        if isinstance(sample, Mapping) and sample.get("status") == "completed"
        for value in (_finite_score(sample.get("score")),)
        if value is not None
    ]
    if len(scores) < 2:
        return None
    ordered = sorted(scores)
    return {
        "score_sample_count": len(ordered),
        "score_samples": ordered,
        "score_median": statistics.median(ordered),
        "score_min": ordered[0],
        "score_max": ordered[-1],
        "score_stdev": statistics.stdev(ordered),
    }


def build_curve(
    *,
    candidates: Sequence[StoredCandidate],
    measurements: Mapping[str, Mapping[str, Any]],
    origin_run_id: str,
    replay_run_id: str,
    measurement_set: str,
    evaluator: Mapping[str, Any],
    curve_root: Path,
) -> dict[str, Any]:
    direction = evaluator.get("score_direction", "maximize")
    selection = evaluator.get("selection", "pass_rate_first")
    selection_entries: list[dict[str, Any]] = []
    points: list[dict[str, Any]] = []
    completed_count = 0
    for candidate in candidates:
        measurement = measurements.get(candidate.candidate_id)
        status = (
            measurement.get("status", "planned")
            if measurement is not None
            else "planned"
        )
        report = measurement.get("report") if measurement is not None else None
        if not isinstance(report, dict):
            report = {}
        score = _finite_score(report.get("score")) if status == "completed" else None
        replicate_stats = (
            replicate_statistics(measurement) if status == "completed" else None
        )
        if replicate_stats is not None:
            # Repeated measurements: the curve reports the median so run-to-run
            # benchmark noise cannot masquerade as a better candidate.
            score = replicate_stats["score_median"]
        if status == "completed":
            completed_count += 1
            selection_entries.append(
                {
                    "type": "submission",
                    "status": "completed",
                    "round": f"v{candidate.sequence}",
                    "score": score,
                    "pass_rate": report.get("pass_rate"),
                    "valid": report.get("valid"),
                }
            )
        best = select_best(
            selection_entries,
            score_direction=str(direction),
            policy=str(selection),
        )
        measurement_path = candidate_measurement_path(curve_root, candidate)
        points.append(
            {
                "version": f"v{candidate.sequence}",
                "sequence": candidate.sequence,
                "candidate_id": candidate.candidate_id,
                "content_sha256": candidate.content_sha256,
                "label": candidate.label,
                "first_seen_at": candidate.created_at,
                "event_count": len(candidate.events),
                "status": status,
                "score": score,
                "score_0_100": (
                    report.get("score_0_100")
                    if status == "completed"
                    else None
                ),
                "history_best": _finite_score(best.get("best_score")),
                "history_best_version": best.get("best_round") or None,
                "pass_rate": (
                    report.get("pass_rate") if status == "completed" else None
                ),
                "valid": report.get("valid") if status == "completed" else None,
                "attempt_count": (
                    measurement.get("attempt", 0)
                    if measurement is not None
                    else 0
                ),
                **(replicate_stats or {}),
                "measurement_path": str(
                    measurement_path.relative_to(curve_root)
                ),
            }
        )
    timeline_points = _build_timeline_points(
        candidates=candidates,
        measurements=measurements,
        score_direction=str(direction),
        selection=str(selection),
    )
    return {
        "schema_version": MEASUREMENT_SCHEMA_VERSION,
        "kind": CURVE_KIND,
        "task_id": evaluator.get("task_id"),
        "origin_run_id": origin_run_id,
        "replay_run_id": replay_run_id,
        "measurement_set": measurement_set,
        "candidate_count": len(candidates),
        "event_count": len(timeline_points),
        "completed_count": completed_count,
        "score_direction": direction,
        "evaluator": dict(evaluator),
        "updated_at": utc_now(),
        "points": points,
        "timeline_points": timeline_points,
    }


def write_curve(root: Path, payload: Mapping[str, Any]) -> Path:
    path = root / "curve.json"
    _atomic_write_json(path, payload)
    return path
