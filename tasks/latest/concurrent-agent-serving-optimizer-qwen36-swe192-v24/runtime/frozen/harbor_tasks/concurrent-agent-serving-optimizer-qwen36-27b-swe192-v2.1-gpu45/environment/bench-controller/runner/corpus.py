from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SWEBENCH_CORPUS_ROOT = Path(os.environ.get("SWEBENCH_CORPUS_ROOT", "/opt/kvbench-corpus"))
EXPECTED_DATASET = "princeton-nlp/SWE-Bench_Lite"
EXPECTED_DATASET_REVISION = "6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2"


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    issue: str
    module_name: str
    files: dict[str, str]
    test_command: str = "python3 -m unittest -q"
    workspace_template: str | None = None
    source_digest: str | None = None
    workload_family: str = "synthetic-fixture"
    runtime_image: str | None = None
    runtime_manifest_digest: str | None = None
    runtime_image_digest: str | None = None
    runtime_base_commit: str | None = None
    runtime_head_commit: str | None = None
    verification_command: str | None = None


TEMPLATES = [
    {
        "issue": "`add(a, b)` subtracts its second argument. Correct it without changing the public API.",
        "module": "calculator",
        "source": "def add(a, b):\n    return a - b\n",
        "test": "import unittest\nfrom calculator import add\nclass T(unittest.TestCase):\n    def test_add(self): self.assertEqual(add(7, 5), 12)\n",
    },
    {
        "issue": "`clamp` returns the upper bound for values below the lower bound. Fix boundary behavior.",
        "module": "bounds",
        "source": "def clamp(value, low, high):\n    if value < low:\n        return high\n    if value > high:\n        return high\n    return value\n",
        "test": "import unittest\nfrom bounds import clamp\nclass T(unittest.TestCase):\n    def test_low(self): self.assertEqual(clamp(-4, 0, 9), 0)\n",
    },
    {
        "issue": "`dedupe` loses input order by converting through a set. Preserve first-occurrence order.",
        "module": "collections_util",
        "source": "def dedupe(values):\n    return list(set(values))\n",
        "test": "import unittest\nfrom collections_util import dedupe\nclass T(unittest.TestCase):\n    def test_order(self): self.assertEqual(dedupe([3,1,3,2,1]), [3,1,2])\n",
    },
    {
        "issue": "`parse_bool` treats every non-empty string as true. Support common true/false spellings and reject unknown values.",
        "module": "parsing",
        "source": "def parse_bool(value):\n    return bool(value)\n",
        "test": "import unittest\nfrom parsing import parse_bool\nclass T(unittest.TestCase):\n    def test_false(self): self.assertFalse(parse_bool('false'))\n",
    },
    {
        "issue": "`median` selects one middle item for even-sized inputs. Return the arithmetic mean of both middle values.",
        "module": "statistics_util",
        "source": "def median(values):\n    ordered = sorted(values)\n    return ordered[len(ordered)//2]\n",
        "test": "import unittest\nfrom statistics_util import median\nclass T(unittest.TestCase):\n    def test_even(self): self.assertEqual(median([1,4,2,3]), 2.5)\n",
    },
    {
        "issue": "`extension` returns the full filename when no dot is present. It should return an empty string.",
        "module": "paths",
        "source": "def extension(name):\n    return name.rsplit('.', 1)[-1]\n",
        "test": "import unittest\nfrom paths import extension\nclass T(unittest.TestCase):\n    def test_missing(self): self.assertEqual(extension('README'), '')\n",
    },
    {
        "issue": "`slugify` leaves repeated separators. Collapse whitespace runs to one dash and lowercase the result.",
        "module": "text_util",
        "source": "def slugify(text):\n    return text.lower().replace(' ', '-')\n",
        "test": "import unittest\nfrom text_util import slugify\nclass T(unittest.TestCase):\n    def test_spaces(self): self.assertEqual(slugify('Hello   World'), 'hello-world')\n",
    },
    {
        "issue": "`parse_port` accepts values outside the TCP port range. Raise ValueError unless the value is 1..65535.",
        "module": "networking",
        "source": "def parse_port(value):\n    return int(value)\n",
        "test": "import unittest\nfrom networking import parse_port\nclass T(unittest.TestCase):\n    def test_range(self):\n        with self.assertRaises(ValueError): parse_port('70000')\n",
    },
    {
        "issue": "`chunks` drops the final partial chunk. Keep it when the input length is not divisible by size.",
        "module": "batching",
        "source": "def chunks(values, size):\n    return [values[i:i+size] for i in range(0, len(values)-size+1, size)]\n",
        "test": "import unittest\nfrom batching import chunks\nclass T(unittest.TestCase):\n    def test_tail(self): self.assertEqual(chunks([1,2,3,4,5], 2), [[1,2],[3,4],[5]])\n",
    },
    {
        "issue": "`safe_divide` catches every exception. Only division by zero should return the configured fallback.",
        "module": "numeric",
        "source": "def safe_divide(a, b, fallback=None):\n    try:\n        return a / b\n    except Exception:\n        return fallback\n",
        "test": "import unittest\nfrom numeric import safe_divide\nclass T(unittest.TestCase):\n    def test_type(self):\n        with self.assertRaises(TypeError): safe_divide('x', 2)\n",
    },
]


def make_fixture_specs(namespace: str, count: int) -> list[TaskSpec]:
    specs = []
    for index in range(count):
        template = TEMPLATES[index % len(TEMPLATES)]
        suffix = f"{namespace}-{index:03d}"
        module = template["module"]
        specs.append(
            TaskSpec(
                task_id=suffix,
                issue=template["issue"],
                module_name=module,
                files={
                    f"{module}.py": template["source"],
                    f"test_{module}.py": template["test"],
                    "README.md": f"# Offline coding task {suffix}\n\nRun `python3 -m unittest -q` to check the repository.\n",
                },
            )
        )
    return specs


def _load_swebench_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"SWE-bench Lite manifest is not staged: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise RuntimeError("unsupported SWE-bench Lite manifest schema")
    if manifest.get("dataset") != EXPECTED_DATASET:
        raise RuntimeError(f"unexpected SWE-bench dataset: {manifest.get('dataset')!r}")
    if manifest.get("dataset_revision") != EXPECTED_DATASET_REVISION:
        raise RuntimeError(f"unexpected SWE-bench dataset revision: {manifest.get('dataset_revision')!r}")
    entries = manifest.get("instances")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("SWE-bench Lite manifest has no instances")
    return manifest


def _safe_workspace(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RuntimeError(f"invalid staged workspace path: {relative!r}")
    workspaces = (root / "workspaces").resolve(strict=True)
    source = (workspaces / relative).resolve(strict=True)
    source.relative_to(workspaces)
    if not source.is_dir():
        raise RuntimeError(f"staged workspace is not a directory: {source}")
    return source


def make_swebench_specs(
    namespace: str,
    count: int,
    *,
    partition: str,
    seed: int,
    root: Path = SWEBENCH_CORPUS_ROOT,
) -> list[TaskSpec]:
    manifest = _load_swebench_manifest(root)
    entries = [entry for entry in manifest["instances"] if entry.get("partition") == partition]
    if not entries:
        raise RuntimeError(f"SWE-bench Lite partition is empty: {partition}")
    for entry in entries:
        if not isinstance(entry.get("instance_id"), str) or not entry["instance_id"]:
            raise RuntimeError("SWE-bench Lite manifest contains an invalid instance_id")
        if not isinstance(entry.get("problem_statement"), str) or not entry["problem_statement"].strip():
            raise RuntimeError(f"manifest problem statement is missing: {entry.get('instance_id')}")
        workspace_digest = entry.get("workspace_sha256")
        image_digest = entry.get("image_digest")
        if workspace_digest is None and image_digest is None:
            raise RuntimeError(f"manifest runtime digest is missing: {entry.get('instance_id')}")
        if workspace_digest is not None and (
            not isinstance(workspace_digest, str) or len(workspace_digest) != 64
        ):
            raise RuntimeError(f"manifest workspace digest is invalid: {entry.get('instance_id')}")
        if image_digest is not None and (
            not isinstance(image_digest, str)
            or not image_digest.startswith("sha256:")
            or len(image_digest) != 71
        ):
            raise RuntimeError(f"manifest image digest is invalid: {entry.get('instance_id')}")
        if image_digest is not None:
            if not isinstance(entry.get("image_name"), str) or not entry["image_name"].strip():
                raise RuntimeError(f"manifest image name is invalid: {entry.get('instance_id')}")
            manifest_digest = entry.get("official_manifest_digest")
            if (
                not isinstance(manifest_digest, str)
                or not manifest_digest.startswith("sha256:")
                or len(manifest_digest) != 71
            ):
                raise RuntimeError(f"manifest OCI digest is invalid: {entry.get('instance_id')}")
            base_commit = entry.get("base_commit")
            if (
                not isinstance(base_commit, str)
                or len(base_commit) != 40
                or any(character not in "0123456789abcdef" for character in base_commit.lower())
            ):
                raise RuntimeError(f"manifest base commit is invalid: {entry.get('instance_id')}")
            runtime_head = entry.get("runtime_head_commit")
            if (
                not isinstance(runtime_head, str)
                or len(runtime_head) != 40
                or any(character not in "0123456789abcdef" for character in runtime_head.lower())
            ):
                raise RuntimeError(f"manifest runtime HEAD is invalid: {entry.get('instance_id')}")
    entries.sort(
        key=lambda entry: hashlib.sha256(
            f"{namespace}:{seed}:{entry['instance_id']}".encode()
        ).hexdigest()
    )
    specs: list[TaskSpec] = []
    for ordinal in range(count):
        entry = entries[ordinal % len(entries)]
        rollout = ordinal // len(entries)
        instance_id = str(entry["instance_id"])
        task_id = f"{instance_id}-r{rollout:02d}" if rollout else instance_id
        test_command = entry.get("test_command", "run the narrowest relevant offline test")
        if not isinstance(test_command, str) or not test_command.strip() or len(test_command) > 4096:
            raise RuntimeError(f"manifest test command is invalid: {instance_id}")
        workspace = entry.get("workspace")
        image_name = entry.get("image_name")
        if workspace is None and not isinstance(image_name, str):
            raise RuntimeError(f"manifest has no executable runtime: {instance_id}")
        specs.append(
            TaskSpec(
                task_id=task_id,
                issue=str(entry["problem_statement"]),
                module_name=str(entry.get("repo") or "swebench"),
                files={},
                test_command=test_command,
                workspace_template=(str(_safe_workspace(root, workspace)) if workspace is not None else None),
                source_digest=str(entry.get("workspace_sha256") or entry.get("image_digest")),
                workload_family="swebench-lite",
                runtime_image=str(image_name) if image_name is not None else None,
                runtime_manifest_digest=(
                    str(entry.get("official_manifest_digest")) if image_name is not None else None
                ),
                runtime_image_digest=str(entry.get("image_digest")) if image_name is not None else None,
                runtime_base_commit=str(entry.get("base_commit")) if image_name is not None else None,
                runtime_head_commit=str(entry.get("runtime_head_commit")) if image_name is not None else None,
                verification_command=str(entry.get("verification_command") or "git diff --check"),
            )
        )
    return specs


def swebench_partition_ready(partition: str, minimum_unique_instances: int) -> bool:
    """Cheap startup/readiness check for a fully staged immutable partition."""
    try:
        manifest = _load_swebench_manifest(SWEBENCH_CORPUS_ROOT)
        entries = [entry for entry in manifest["instances"] if entry.get("partition") == partition]
        if len({entry.get("instance_id") for entry in entries}) < minimum_unique_instances:
            return False
        for entry in entries:
            if entry.get("workspace") is not None:
                _safe_workspace(SWEBENCH_CORPUS_ROOT, entry.get("workspace"))
            elif not isinstance(entry.get("image_name"), str):
                return False
            workspace_digest = entry.get("workspace_sha256")
            image_digest = entry.get("image_digest")
            if workspace_digest is None and image_digest is None:
                return False
            if image_digest is not None and (
                not isinstance(image_digest, str)
                or not image_digest.startswith("sha256:")
                or len(image_digest) != 71
            ):
                return False
            if image_digest is not None:
                manifest_digest = entry.get("official_manifest_digest")
                if (
                    not isinstance(manifest_digest, str)
                    or not manifest_digest.startswith("sha256:")
                    or len(manifest_digest) != 71
                ):
                    return False
                base_commit = entry.get("base_commit")
                if (
                    not isinstance(base_commit, str)
                    or len(base_commit) != 40
                    or any(character not in "0123456789abcdef" for character in base_commit.lower())
                ):
                    return False
                runtime_head = entry.get("runtime_head_commit")
                if (
                    not isinstance(runtime_head, str)
                    or len(runtime_head) != 40
                    or any(character not in "0123456789abcdef" for character in runtime_head.lower())
                ):
                    return False
        return True
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
        return False


def make_specs(
    namespace: str,
    count: int,
    *,
    workload_family: str = "synthetic-fixture",
    corpus_partition: str = "fixture",
    seed: int = 0,
) -> list[TaskSpec]:
    if workload_family == "synthetic-fixture":
        return make_fixture_specs(namespace, count)
    if workload_family == "swebench-lite":
        return make_swebench_specs(
            namespace,
            count,
            partition=corpus_partition,
            seed=seed,
        )
    raise ValueError(f"unknown workload family: {workload_family}")


def repository_context(task_id: str, target_bytes: int) -> str:
    if target_bytes <= 0:
        return ""
    lines: list[str] = []
    size = 0
    index = 0
    while size < target_bytes:
        digest = hashlib.sha256(f"{task_id}:{index}".encode()).hexdigest()
        line = (
            f"repo-note {index:05d} {digest}: compatibility behavior is covered by offline unit tests; "
            "preserve unrelated public interfaces and make the smallest correct change.\n"
        )
        lines.append(line)
        size += len(line.encode())
        index += 1
    return "".join(lines)[:target_bytes]


def corpus_fingerprint(
    namespace: str,
    count: int,
    *,
    workload_family: str = "synthetic-fixture",
    corpus_partition: str = "fixture",
    seed: int = 0,
) -> str:
    payload = [
        {
            "task_id": spec.task_id,
            "issue": spec.issue,
            "module_name": spec.module_name,
            "files": spec.files,
            "test_command": spec.test_command,
            "source_digest": spec.source_digest,
            "workload_family": spec.workload_family,
            "runtime_image_digest": spec.runtime_image_digest,
            "runtime_manifest_digest": spec.runtime_manifest_digest,
            "runtime_base_commit": spec.runtime_base_commit,
            "runtime_head_commit": spec.runtime_head_commit,
        }
        for spec in make_specs(
            namespace,
            count,
            workload_family=workload_family,
            corpus_partition=corpus_partition,
            seed=seed,
        )
    ]
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()
