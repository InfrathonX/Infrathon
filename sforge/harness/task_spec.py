from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from sforge.harness.benchmark import BenchmarkMeta
from sforge.harness.constants import (
    SForgeTask,
    START_TEST_OUTPUT,
    END_TEST_OUTPUT,
)
from sforge.harness.portable_spec import (
    CapabilityEnvelope,
    PortableSpec,
    is_full_portable_contract,
    parse_capability_envelope,
    parse_portable_spec,
)
from sforge.harness.score_rescale import RescaleSpec, parse_rescale_spec


@dataclass
class WorkSpec:
    specs_dir: str
    agent_query: str
    setup_cmds: list[str] | None = None
    image_tag: str | None = None
    cpu_limit: int | None = None
    mem_limit: str | None = None


@dataclass
class JudgeSpec:
    eval_cmd: str
    eval_timeout: int
    parser: str
    private_artifact_path: str | None = None
    public_eval_cmd: str | None = None
    public_eval_timeout: int | None = None
    public_private_artifact_path: str | None = None
    setup_cmds: list[str] | None = None
    image_tag: str | None = None
    game_server_cmd: str | None = None
    score_direction: str = "maximize"
    selection: str = "pass_rate_first"
    rescale: RescaleSpec | None = None
    cpu_limit: int | None = None
    mem_limit: str | None = None
    # Private evidence set, including private_artifact_path when present.
    # That legacy primary remains required; extra paths may be absent before
    # execution. Without a primary every declared path is required. Omitted
    # preserves the single file; last preserves positional construction.
    private_artifact_paths: list[str] | None = None
    # Optional trusted runtime bootstrap. Omitted retains existing task behavior.
    private_bootstrap: str | None = None


PRIVATE_JUDGE_ARTIFACT_MAX_FILES = 16


def validate_private_artifact_paths(paths: object) -> list[str]:
    """Only distinct, canonical files directly below the private log root."""
    if not isinstance(paths, list) or len(paths) > PRIVATE_JUDGE_ARTIFACT_MAX_FILES:
        raise ValueError('private Judge artifact paths must be a list of at most 16 files')
    names = set()
    for value in paths:
        if not isinstance(value, str) or any(c in value for c in ('\x00', '\n', '\r')):
            raise ValueError('private Judge artifact path must be a file under /logs/verifier')
        source = PurePosixPath(value)
        if (source.parent != PurePosixPath('/logs/verifier')
                or source.name in {'', '.', '..'} or str(source) != value):
            raise ValueError('private Judge artifact path must name one canonical file under /logs/verifier')
        if source.name in names:
            raise ValueError('private Judge artifact paths must have unique file names')
        names.add(source.name)
    return list(paths)


def _compute_base_image_hash(base_image_key: str, base_image_spec: dict) -> str:
    data = json.dumps({"key": base_image_key, "spec": base_image_spec}, sort_keys=True)
    return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class TaskSpec:
    task_id: str
    name: str
    base_image: str  # key into BENCHMARK.json base_images, e.g. "python310"
    platform: str
    cwd: str
    submit_paths: list[str]
    submit_exclude: list[str]
    work: WorkSpec
    judge: JudgeSpec
    # Submitted files that never affect the Judge score (task notes, evidence
    # logs).  They are archived verbatim but do not contribute to logical
    # candidate identity, so notes-only edits do not create a new replay point.
    candidate_identity_exclude: list[str] = field(default_factory=list)
    # Optional declaration that the submitted tree retains one directory per
    # implementation and a selector file naming the scored one (for example
    # candidates/kernel_vNNN + FINAL_VERSION). Per-version replay uses it to
    # score every retained implementation directly instead of replaying whole
    # checkpoints whose selector never moved. Keys: root, directory_pattern
    # (one capture group with the version number), selector_file.
    version_enumeration: dict | None = None
    benchmark_name: str = ""
    base_image_spec: dict | None = None
    game_mode: bool = False
    internet: bool = True
    # Kept last so adding the optional contract does not shift legacy
    # positional construction of TaskSpec.
    portable: PortableSpec | None = None
    # Lightweight capability envelope for tasks whose ``portable`` object has
    # not been promoted to the full contract.  Exactly one of ``portable`` and
    # ``capability_envelope`` is set when the task declares ``portable``.
    capability_envelope: CapabilityEnvelope | None = None

    @property
    def base_image_tag(self) -> str:
        """Full Docker image name with hash tag, e.g. 'mlsysforge.base.python310:a1b2c3d4e5f6'."""
        return f"{self.benchmark_name}.base.{self.base_image}:{self.base_image_hash[:12]}"

    @property
    def work_image_key(self) -> str:
        tag = self.work.image_tag if self.work.image_tag else self.work_image_hash[:12]
        return f"{self.benchmark_name}.work.{self.task_id}:{tag}"

    @property
    def judge_image_key(self) -> str:
        tag = self.judge.image_tag if self.judge.image_tag else self.judge_image_hash[:12]
        return f"{self.benchmark_name}.judge.{self.task_id}:{tag}"

    @property
    def base_image_hash(self) -> str:
        return _compute_base_image_hash(self.base_image, self.base_image_spec or {})

    @property
    def work_image_hash(self) -> str:
        if self.work.setup_cmds is None:
            raise ValueError(
                f"Task '{self.task_id}': cannot compute work image hash without setup_cmds"
            )
        data = json.dumps({
            "base_hash": self.base_image_hash,
            "platform": self.platform,
            "cwd": self.cwd,
            "setup_cmds": self.work.setup_cmds,
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    @property
    def judge_image_hash(self) -> str:
        if self.judge.setup_cmds is None:
            raise ValueError(
                f"Task '{self.task_id}': cannot compute judge image hash without setup_cmds"
            )
        data = json.dumps({
            "base_hash": self.base_image_hash,
            "platform": self.platform,
            "cwd": self.cwd,
            "setup_cmds": self.judge.setup_cmds,
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    @property
    def work_needs_build(self) -> bool:
        return self.work.setup_cmds is not None

    @property
    def judge_needs_build(self) -> bool:
        return self.judge.setup_cmds is not None

    def base_remote_ref(self, registry: str) -> str:
        return f"{registry}/{self.base_image_tag}"

    def work_remote_ref(self, registry: str) -> str:
        return f"{registry}/{self.work_image_key}"

    def judge_remote_ref(self, registry: str) -> str:
        return f"{registry}/{self.judge_image_key}"

    @property
    def setup_workspace_script(self) -> str:
        if self.work.setup_cmds is None:
            raise ValueError(
                f"Task '{self.task_id}': no setup_cmds for work image (using pre-built image_tag)"
            )
        lines = ["#!/bin/bash", "set -euxo pipefail"] + self.work.setup_cmds
        return "\n".join(lines) + "\n"

    @property
    def setup_judge_script(self) -> str:
        if self.judge.setup_cmds is None:
            raise ValueError(
                f"Task '{self.task_id}': no setup_cmds for judge image (using pre-built image_tag)"
            )
        lines = ["#!/bin/bash", "set -euxo pipefail"] + self.judge.setup_cmds
        return "\n".join(lines) + "\n"

    @property
    def eval_script(self) -> str:
        """Generate eval.sh for the judge container."""
        return self.eval_script_for(self.judge.eval_cmd)

    def eval_script_for(self, eval_cmd: str) -> str:
        """Generate an eval script for a trusted task-declared command."""
        lines = [
            "#!/bin/bash",
            "set -uxo pipefail",
            f'echo "{START_TEST_OUTPUT}"',
            f"{eval_cmd} 2>&1",
            "EXIT_CODE=$?",
            f'echo "{END_TEST_OUTPUT}"',
            'echo "Exit code: $EXIT_CODE"',
            'exit "$EXIT_CODE"',
        ]
        return "\n".join(lines) + "\n"


def _validate_image_tag_consistency(
    task_id: str,
    role: str,
    setup_cmds: list[str] | None,
    image_tag: str | None,
    computed_hash_fn,
) -> None:
    """When both setup_cmds and image_tag are present, verify the hash matches."""
    if setup_cmds is not None and image_tag is not None:
        computed = computed_hash_fn()[:12]
        if computed != image_tag:
            raise ValueError(
                f"Task '{task_id}' {role}: image_tag '{image_tag}' does not match "
                f"hash computed from setup_cmds '{computed}'. "
                f"Update image_tag or setup_cmds to make them consistent."
            )
    if setup_cmds is None and image_tag is None:
        raise ValueError(
            f"Task '{task_id}' {role}: must have either setup_cmds or image_tag (or both)."
        )


def make_task_spec(task_path: Path, benchmark: BenchmarkMeta) -> TaskSpec:
    """Load a TaskSpec from a task JSON file, resolving base image from benchmark metadata."""
    with open(task_path) as f:
        data: SForgeTask = json.load(f)

    base_image_key = data["base_image"]
    base_image_spec = benchmark.base_images.get(base_image_key)
    if base_image_spec is None:
        raise ValueError(
            f"Task '{data['task_id']}' references base_image '{base_image_key}' "
            f"which is not defined in BENCHMARK.json. "
            f"Available: {list(benchmark.base_images.keys())}"
        )

    work_data = data["work"]
    work = WorkSpec(
        specs_dir=work_data.get("specs_dir", ""),
        agent_query=work_data.get("agent_query", ""),
        setup_cmds=work_data.get("setup_cmds"),
        image_tag=work_data.get("image_tag"),
        cpu_limit=work_data.get("cpu_limit"),
        mem_limit=work_data.get("mem_limit"),
    )
    judge_data = data["judge"]
    private_artifact_path = judge_data.get("private_artifact_path")
    private_artifact_paths = judge_data.get("private_artifact_paths")
    private_bootstrap = judge_data.get("private_bootstrap")
    if private_bootstrap not in (None, "minference-ruler-quality-v3", "minference-kernel-prefill-v3"):
        raise ValueError(f"Task '{data['task_id']}': unknown judge.private_bootstrap")
    public_private_artifact_path = judge_data.get("public_private_artifact_path")
    for field_name, artifact_path in (
        ("private_artifact_path", private_artifact_path),
        ("public_private_artifact_path", public_private_artifact_path),
    ):
        if artifact_path is None:
            continue
        try:
            validate_private_artifact_paths([artifact_path])
        except ValueError as exc:
            raise ValueError(
                f"Task '{data['task_id']}': judge.{field_name} must "
                "name one file directly under /logs/verifier"
            ) from exc
    if private_artifact_paths is not None:
        private_artifact_paths = validate_private_artifact_paths(private_artifact_paths)
        if private_artifact_path is not None and private_artifact_path not in private_artifact_paths:
            raise ValueError(f"Task '{data['task_id']}': judge.private_artifact_paths must include its primary private_artifact_path")
    public_eval_cmd = judge_data.get("public_eval_cmd")
    public_eval_timeout = judge_data.get("public_eval_timeout")
    if (public_eval_cmd is None) != (public_eval_timeout is None):
        raise ValueError(
            f"Task '{data['task_id']}': judge.public_eval_cmd and "
            "judge.public_eval_timeout must be configured together"
        )
    if public_private_artifact_path is not None and public_eval_cmd is None:
        raise ValueError(
            f"Task '{data['task_id']}': judge.public_private_artifact_path "
            "requires judge.public_eval_cmd"
        )
    if public_eval_timeout is not None and (
        not isinstance(public_eval_timeout, int)
        or isinstance(public_eval_timeout, bool)
        or public_eval_timeout <= 0
    ):
        raise ValueError(
            f"Task '{data['task_id']}': judge.public_eval_timeout must be positive"
        )
    judge = JudgeSpec(
        eval_cmd=judge_data.get("eval_cmd", ""),
        eval_timeout=judge_data.get("eval_timeout", 600),
        parser=judge_data.get("parser", ""),
        private_artifact_path=private_artifact_path,
        private_artifact_paths=private_artifact_paths,
        private_bootstrap=private_bootstrap,
        public_eval_cmd=public_eval_cmd,
        public_eval_timeout=public_eval_timeout,
        public_private_artifact_path=public_private_artifact_path,
        setup_cmds=judge_data.get("setup_cmds"),
        image_tag=judge_data.get("image_tag"),
        game_server_cmd=judge_data.get("game_server_cmd"),
        score_direction=judge_data.get("score_direction", "maximize"),
        selection=judge_data.get("selection", "pass_rate_first"),
        rescale=parse_rescale_spec(judge_data.get("rescale")),
        cpu_limit=judge_data.get("cpu_limit"),
        mem_limit=judge_data.get("mem_limit"),
    )

    untyped_data: dict[str, object] = dict(data)
    portable: PortableSpec | None = None
    capability_envelope: CapabilityEnvelope | None = None
    if "portable" in untyped_data:
        raw_portable = untyped_data["portable"]
        if is_full_portable_contract(raw_portable):
            portable = parse_portable_spec(raw_portable, task_id=data["task_id"])
        else:
            capability_envelope = parse_capability_envelope(
                raw_portable, task_id=data["task_id"]
            )
    declared_direction: str | None = None
    if portable is not None:
        declared_direction = portable.metrics[portable.defaults.metric_id].direction
    elif capability_envelope is not None:
        default_metric = capability_envelope.default_metric
        if default_metric is not None:
            declared_direction = default_metric.direction
    if declared_direction is not None and declared_direction != judge.score_direction:
        raise ValueError(
            f"Task '{data['task_id']}': portable default metric direction "
            f"'{declared_direction}' does not match "
            f"judge.score_direction '{judge.score_direction}'"
        )

    task_spec = TaskSpec(
        task_id=data["task_id"],
        name=data["name"],
        base_image=base_image_key,
        platform=data["platform"],
        cwd=data["cwd"],
        submit_paths=data["submit_paths"],
        submit_exclude=[e.rstrip("/") for e in data.get("submit_exclude", ["tests/"])],
        candidate_identity_exclude=[
            e.rstrip("/") for e in data.get("candidate_identity_exclude", [])
        ],
        version_enumeration=data.get("version_enumeration"),
        portable=portable,
        capability_envelope=capability_envelope,
        work=work,
        judge=judge,
        benchmark_name=benchmark.name,
        base_image_spec=base_image_spec,
        game_mode=data.get("game_mode", False),
        internet=data.get("internet", True),
    )

    _validate_image_tag_consistency(
        data["task_id"], "work",
        work.setup_cmds, work.image_tag,
        lambda: task_spec.work_image_hash,
    )
    _validate_image_tag_consistency(
        data["task_id"], "judge",
        judge.setup_cmds, judge.image_tag,
        lambda: task_spec.judge_image_hash,
    )

    return task_spec


def load_all_tasks(tasks_dir: Path, benchmark: BenchmarkMeta) -> list[TaskSpec]:
    """Load all task specs from a directory of JSON files."""
    tasks = []
    for task_path in sorted(tasks_dir.glob("*.json")):
        tasks.append(make_task_spec(task_path, benchmark))
    return tasks
