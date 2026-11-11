"""Mandatory LLM integrity review for PyroMind runs.

Before a submission reaches its GPU Judge, the Controller reviews it in a
fresh CPU-only PyroMind Sandbox built from the task's published Judge image
(no GPU request, no task assets, no private Judge payload: those are only
ever installed into GPU Judges).  A Worker-like reviewer agent (Codex from
the pinned agent runtime) inspects the submission against the agent's
instructions and the task's ``llm_judge/POLICY.md``; a verified violation
ends the submission before any GPU work and its feedback goes to the agent.

PyroMind specifics, all owned by this module:

* the agent runtime is published once per runtime version as an immutable
  Storage asset (one deterministic tarball) and mounted read-only into each
  review Sandbox, which unpacks it to ``/opt/agent-runtime``;
* review Sandboxes are named under the run's Sandbox prefix and journaled in
  the run's ``sandbox-events.jsonl``, so the run's existing cleanup and
  reconciliation cover them;
* the baseline (the agent's starting files) is the Worker's initial
  checkpoint archive plus the briefs read from the Worker before its agent
  starts, because no Work image is local to the Controller;
* the network is the platform default, as for Workers (not attested).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tarfile
import tempfile
from pathlib import Path

from sforge.harness.llm_judge.config import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    LLMJudgeConfig,
    LLMJudgeConfigError,
)

REVIEW_RUNTIME_MOUNT = "/mnt/sforge-review-runtime"
REVIEW_RUNTIME_FILE = "runtime.tar.gz"
REVIEW_RUNTIME_PREFIX = "sforge-assets/v2/agent-runtime"
DEFAULT_REVIEW_RESOURCES = {"cpu": 2, "memory": "4Gi", "gpu": 0}
REVIEW_CONCURRENCY = 1
REVIEW_ALLOWANCE_SLACK_SECONDS = 180


def review_resources(deployment) -> dict:
    """CPU-only review Sandbox: optional ``review_resources`` override."""

    resources = dict(DEFAULT_REVIEW_RESOURCES)
    resources.update(deployment.config.get("review_resources") or {})
    from sforge.pyromind.deployment import whole_gibibytes

    cpu = resources.get("cpu")
    if isinstance(cpu, bool) or not isinstance(cpu, int) or cpu < 1:
        raise ValueError("review_resources.cpu must be a positive integer")
    if not whole_gibibytes(resources.get("memory")):
        raise ValueError("review_resources.memory must be a whole number of Gi")
    if resources.get("gpu", 0) != 0:
        raise ValueError("The integrity review Sandbox never requests a GPU")
    return resources


def review_config(model: dict, *, mode: str, explicit: bool, resources: dict,
                  review_timeout_seconds: float = 1800.0,
                  on_error: str = "admit", implicit_model: str | None = None,
                  implicit_reasoning_effort: str | None = None) -> LLMJudgeConfig:
    """Map an agent.env model configuration onto the review configuration.

    ``explicit`` marks a dedicated ``--review-model-config``.  Without one the
    Worker's own configuration is reused; with ChatGPT credentials the review
    then runs the default reviewer model instead of the Worker's. Trusted
    deployment overrides apply only to this implicit selection; a dedicated
    reviewer configuration always keeps its own model and reasoning effort.
    """

    if mode not in ("enforce", "shadow", "off"):
        raise ValueError("Integrity review mode must be enforce, shadow or off")
    if on_error not in ("admit", "block"):
        raise ValueError("review_on_error must be admit or block")
    for field, value in (("review_model", implicit_model),
                         ("review_reasoning_effort", implicit_reasoning_effort)):
        if value is not None and (not isinstance(value, str) or not value.strip()
                                  or any(char in value for char in ('\n', '\r', '\x00'))):
            raise ValueError(field + " must be a nonempty single-line string")
    if mode == "off":
        return LLMJudgeConfig(mode="off")
    auth = model.get("AUTH_MODE")
    if auth not in ("chatgpt", "api-key"):
        raise LLMJudgeConfigError(
            "The integrity reviewer needs a Codex model configuration with ChatGPT "
            "credentials or an API key; pass --review-model-config"
        )
    name, effort = model["MODEL"], model.get("REASONING_EFFORT", "high")
    if auth == "chatgpt" and not explicit:
        # The reviewer model of this host (launch/mlsysforge.env), not the Worker's.
        name, effort = os.environ.get("SFORGE_LLM_JUDGE_MODEL") or DEFAULT_MODEL, DEFAULT_REASONING_EFFORT
    if not explicit:
        name = implicit_model if implicit_model is not None else name
        effort = implicit_reasoning_effort if implicit_reasoning_effort is not None else effort
    config = LLMJudgeConfig(
        mode=mode,
        engine="agent",
        agent="codex",
        agent_runtime_dir=Path(model["SFORGE_AGENT_RUNTIME_DIR"]),
        codex_auth_json=Path(model["CODEX_AUTH_JSON_PATH"]) if auth == "chatgpt" else None,
        api_base_url=model.get("MODEL_API_BASE_URL") if auth == "api-key" else None,
        api_key_file=Path(model["MODEL_API_KEY_FILE"]) if auth == "api-key" else None,
        model=name,
        reasoning_effort=effort,
        image="judge",
        network="platform",
        cpu_limit=resources["cpu"],
        mem_limit=resources["memory"],
        concurrency=REVIEW_CONCURRENCY,
        review_timeout_seconds=review_timeout_seconds,
        on_error=on_error,
    )
    config.check_ready()
    return config


def review_allowance_seconds(config: LLMJudgeConfig, ready_timeout: int) -> int:
    """Extra submit-wait budget: a queued submission is reviewed first."""

    if not config.enabled:
        return 0
    return int(config.review_timeout_seconds) + int(ready_timeout) + REVIEW_ALLOWANCE_SLACK_SECONDS


def runtime_archive(runtime, cache_root: Path) -> tuple[Path, str, int]:
    """A deterministic gzip tarball of the agent runtime (cached locally).

    Entries are sorted, owned by root and dated at the epoch, and gzip
    carries no timestamp, so one runtime version always yields the same
    bytes and therefore the same immutable Storage asset.
    """

    tag = f"{runtime.agent}-{runtime.version}-{(runtime.tree_sha256 or 'untracked')[:12]}"
    directory = cache_root / tag
    archive = directory / REVIEW_RUNTIME_FILE
    receipt = directory / "receipt.json"
    if archive.is_file() and receipt.is_file():
        recorded = json.loads(receipt.read_text())
        if recorded.get("size") == archive.stat().st_size:
            return archive, recorded["sha256"], recorded["size"]
    directory.mkdir(parents=True, exist_ok=True)
    root = Path(runtime.root)
    descriptor, temporary = tempfile.mkstemp(dir=directory, suffix=".partial")
    os.close(descriptor)
    try:
        with open(temporary, "wb") as raw, gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0
        ) as compressed, tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as bundle:
            for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
                info = bundle.gettarinfo(str(path), arcname=path.relative_to(root).as_posix())
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                if info.isfile():
                    with path.open("rb") as stream:
                        bundle.addfile(info, stream)
                elif info.isdir() or info.issym():
                    bundle.addfile(info)
        digest = hashlib.sha256()
        with open(temporary, "rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        size = os.path.getsize(temporary)
        os.replace(temporary, archive)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    receipt.write_text(json.dumps({"sha256": digest.hexdigest(), "size": size}))
    return archive, digest.hexdigest(), size


def stage_runtime(runtime, client, bucket: str, cache_root: Path) -> dict:
    """Publish the runtime tarball as an immutable asset (reused when ready)."""

    from sforge.pyromind.assets import prepare_asset

    archive, digest, size = runtime_archive(runtime, cache_root)
    manifest = {
        "kind": "sforge-agent-runtime-archive",
        "agent": runtime.agent,
        "version": runtime.version,
        "tree_sha256": runtime.tree_sha256,
        "files": [{"path": REVIEW_RUNTIME_FILE, "size": size, "sha256": digest}],
    }
    staged = prepare_asset(client, bucket, archive.parent, manifest,
                           f"{REVIEW_RUNTIME_PREFIX}/{runtime.agent}-{runtime.version}")
    return {"data_prefix": staged["data_prefix"], "archive_sha256": digest,
            "archive_bytes": size, "agent": runtime.agent, "version": runtime.version}


def build_review_engine(config: LLMJudgeConfig, backend, *, judge_image: str,
                        name_prefix: str, staged_runtime: dict):
    """The review engine whose Sandboxes run on ``backend`` (gpu_count=0)."""

    from sforge.harness.llm_judge.agent_engine import (
        AgentReviewEngine,
        BackendSandboxProvider,
        extract_runtime_archive,
    )

    if "@sha256:" not in judge_image:
        raise ValueError("The review Sandbox requires the immutable Judge image digest")
    provider = BackendSandboxProvider(
        backend,
        # The published Judge image without its private payload, by digest:
        # also the image identity cached verdicts depend on.
        resolve_image=lambda task_spec, image_key: judge_image,
        name_prefix=name_prefix,
        runtime_installer=extract_runtime_archive(
            f"{REVIEW_RUNTIME_MOUNT}/{REVIEW_RUNTIME_FILE}"
        ),
        runtime_identity={"runtime_archive_sha256": staged_runtime["archive_sha256"]},
        network="platform-default-not-attested",
    )
    return AgentReviewEngine(config, sandboxes=provider)


def runtime_mount(staged_runtime: dict):
    from pyromind_sdk.client.models import VolumeMount

    return VolumeMount(host_path="/workspace/" + staged_runtime["data_prefix"],
                       mount_path=REVIEW_RUNTIME_MOUNT, read_only=True)


def capture_baseline(task, policy, store, initial_version, read_worker_file):
    """Starting files: the initial checkpoint; briefs: read from the Worker.

    Call before the Worker's agent starts, while its files are still the
    task's own.  ``read_worker_file(path)`` returns bytes or raises.
    """

    from sforge.harness.llm_judge.bundle import ArchiveBaseline

    def read(path: str):
        try:
            return read_worker_file(path)
        except Exception:
            return None

    return ArchiveBaseline.capture(
        task, policy, store.read_archive(initial_version), read,
        source=f"pyromind-initial-checkpoint:{initial_version.version_id}",
    )


def worker_visible_history(history: dict) -> dict:
    """History mirrored to the Worker's read-only ``/shared``: no review audit.

    The Controller keeps the full record under ``logs/`` (not mirrored).
    """

    def strip(entry):
        if isinstance(entry, dict):
            return {key: value for key, value in entry.items() if key != "integrity_review"}
        return entry

    if isinstance(history, list):
        return [strip(entry) for entry in history]
    if not isinstance(history, dict):
        return history
    return {**history, "entries": [strip(entry) for entry in history.get("entries", [])]}


def describe(config: LLMJudgeConfig, *, resources: dict, judge_image: str,
             staged_runtime: dict | None, explicit_model_config: bool) -> dict:
    """run.json metadata (no credential values)."""

    if not config.enabled:
        return {"mode": "off"}
    return {
        "mode": config.mode,
        "agent": config.agent,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "auth_mode": "chatgpt" if config.codex_auth_json else "api-key",
        "model_config": "review-specific" if explicit_model_config else "worker",
        "review_image": judge_image,
        "resources": resources,
        "concurrency": config.concurrency,
        "on_error": config.on_error,
        "network": "platform-default-not-attested",
        "runtime_archive_sha256": (staged_runtime or {}).get("archive_sha256"),
        "lifecycle": "one-new-cpu-sandbox-per-review",
    }


def run_offline_review(args) -> int:
    """``review``: audit archives in real CPU review Sandboxes (no Worker, no GPU).

    The same runtime asset, CPU-only backend, Sandbox naming and cleanup as a
    run, for auditing a final archive or stored candidates and for
    calibrating a task's policy on PyroMind.  Exit 0: nothing rejected; 3: a
    violation; 2: a review produced no verdict.
    """

    import threading
    import time
    import uuid

    from sforge.harness.agent_runtime import load_agent_runtime
    from sforge.harness.backend.pyromind_backend import PyroMindBackend
    from sforge.harness.benchmark import load_benchmark
    from sforge.harness.llm_judge.bundle import HostDirectoryBaseline, NoBaseline
    from sforge.harness.llm_judge.reviewer import IntegrityReviewer, redact
    from sforge.harness.task_spec import load_all_tasks
    from sforge.harness.agent_config import load_credentials, load_model_config
    from sforge.pyromind.cli import write_json
    from sforge.pyromind.deployment import Deployment
    from sforge.pyromind.runner import (
        configure_resource_capture, guard_judge_backend, reconcile_creation_ownership,
        reconcile_run_sandboxes,
    )
    from sforge.pyromind.storage import storage_client

    repo = Path(__file__).resolve().parents[2]
    deployment = Deployment.load(Path(args.deployment), repo)
    task_root = Path(args.task_dir or deployment.local_path('task_directory')).resolve()
    tasks_dir = task_root / 'tasks'
    task = load_all_tasks(tasks_dir, load_benchmark(tasks_dir))[0]
    if task.task_id != deployment.config['task_id']:
        raise ValueError('Task identity differs from deployment profile')
    images = json.loads(Path(args.images_file or deployment.local_path('images_file')).read_text())
    judge_image = images['judge']
    model = load_model_config(Path(args.review_model_config))
    secrets = set(load_credentials(model).secrets) | {os.getenv('PYROMIND_API_KEY', '')}
    resources = review_resources(deployment)
    config = review_config(model, mode='enforce', explicit=True, resources=resources,
                           on_error=deployment.config.get('review_on_error', 'admit'))
    output = Path(args.artifacts).resolve() / (
        'review-' + time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
    output.mkdir(parents=True)
    client, bucket = storage_client()
    staged = stage_runtime(load_agent_runtime(config.agent_runtime_dir), client, bucket,
                           Path.home() / '.cache' / 'sforge-pyromind' / 'review-runtime')
    prefix = 'sf-' + uuid.uuid4().hex[:20]
    backend = PyroMindBackend(event_path=output / 'sandbox-events.jsonl', gpu_count=0,
                              volume_mounts=[runtime_mount(staged)])
    backend.ready_timeout = deployment.sandbox_ready_timeout_seconds
    configure_resource_capture(backend, output / 'resource-usage' / 'review')
    cancelled = threading.Event()
    guard_judge_backend(backend, cancelled, prefix)
    engine = build_review_engine(config, backend, judge_image=judge_image,
                                 name_prefix='review', staged_runtime=staged)
    reviewer = IntegrityReviewer(config, tasks_dir=tasks_dir, cache_dir=None, engine=engine)
    reviewer.validate_tasks([task])
    if args.baseline_dir:
        baseline = HostDirectoryBaseline(Path(args.baseline_dir),
            documents_root=Path(args.documents_root) if args.documents_root else None)
    else:
        baseline = NoBaseline('no --baseline-dir given')
    summary = {'task_id': task.task_id, 'deployment': deployment.config['name'],
               'review': describe(config, resources=resources, judge_image=judge_image,
                                  staged_runtime=staged, explicit_model_config=True),
               'sandbox_name_prefix': prefix, 'reviews': []}
    statuses = set()
    try:
        for index, raw in enumerate(args.archive, 1):
            path = Path(raw)
            review = reviewer.review(task_spec=task, archive=path.read_bytes(), baseline=baseline,
                                     log_dir=output / f'{index:03d}-{path.name.split(".")[0]}')
            statuses.add(review.status)
            summary['reviews'].append({'archive': str(path), 'status': review.status,
                                       'reason': review.reason, 'rules': review.rules,
                                       'feedback': review.feedback_to_agent,
                                       'elapsed_seconds': round(review.elapsed_seconds, 1)})
            print(f'{path}: {review.status.upper()} ({review.reason})', flush=True)
    finally:
        cancelled.set()
        summary['remaining_sandbox_ids'] = reconcile_run_sandboxes(backend, prefix)
        ownership = reconcile_creation_ownership((backend,))
        summary['cleanup_errors'] = ownership['cleanup_errors']
        write_json(output / 'summary.json', json.loads(redact(json.dumps(summary), list(secrets))))
        print(f'REVIEW_DIRECTORY={output}', flush=True)
    if summary['remaining_sandbox_ids'] or summary['cleanup_errors']:
        raise RuntimeError('Review Sandbox cleanup needs attention: see summary.json')
    if 'violation' in statuses:
        return 3
    return 2 if 'error' in statuses else 0
