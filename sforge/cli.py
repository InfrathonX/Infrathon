"""SForge CLI entry point."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import copy
import hashlib
import io
import json
import signal
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import docker
import requests

from sforge.harness import candidate_replay
from sforge.harness.backend.base import ContainerBackend, ResolvedImage
from sforge.harness.config import SForgeConfig, create_backend_from_config, load_config
from sforge.harness.constants import ADMIN_SECRET, DEFAULT_EVAL_INTERVAL
from sforge.harness.cpu_affinity import cpu_affinity_identity
from sforge.harness.docker_build import (
    build_all_images,
    build_work_image,
    build_judge_image,
    pull_all_images,
    push_all_images,
)
from sforge.harness.docker_utils import cleanup_container
from sforge.harness.run_evaluation import judge_submission
from sforge.harness.benchmark import load_benchmark
from sforge.harness.task_spec import TaskSpec, make_task_spec, load_all_tasks
from sforge.harness.version_store import (
    VERSION_SOURCES,
    VersionStore,
    VersionStoreError,
)


def _resolve_task(args, config: SForgeConfig) -> TaskSpec:
    """Resolve a single task. Fails if the user passed more than one ID."""
    specs = _resolve_tasks(args, config)
    if len(specs) > 1:
        print(f"Error: this command takes only one --task, got {len(specs)}")
        sys.exit(1)
    return specs[0]


def _resolve_tasks(args, config: SForgeConfig) -> list[TaskSpec]:
    """Resolve one or more tasks by ID, or all tasks with --all."""
    tasks_dir = config.tasks_dir
    benchmark = load_benchmark(tasks_dir)

    if getattr(args, "all", False):
        specs = load_all_tasks(tasks_dir, benchmark)
        if not specs:
            print(f"No tasks found in {tasks_dir}")
            sys.exit(1)
        return specs

    if not args.task:
        print("Error: --task or --all is required")
        sys.exit(1)

    raw = args.task
    task_ids = raw if isinstance(raw, list) else [raw]
    specs: list[TaskSpec] = []
    for tid in task_ids:
        task_file = tasks_dir / f"{tid}.json"
        if not task_file.exists():
            print(f"Error: task '{tid}' not found at {task_file}")
            sys.exit(1)
        specs.append(make_task_spec(task_file, benchmark))
    return specs


def _make_config(args) -> SForgeConfig:
    """Build config from CLI args."""
    overrides = {}
    for key in (
        "log_dir",
        "tasks_dir",
        "registry",
        "backend",
        "work_cpu_limit",
        "work_mem_limit",
        "judge_cpu_limit",
        "judge_mem_limit",
        "version_store_root",
    ):
        val = getattr(args, key, None)
        if val is not None:
            overrides[key] = val
    return load_config(overrides)


# --- Commands ---


def cmd_build(args):
    """Build work + judge images for one or more tasks (parallel)."""
    config = _make_config(args)
    task_specs = _resolve_tasks(args, config)
    client = docker.from_env()
    verbose = not args.silent and len(task_specs) == 1

    force_rebuild = args.force_rebuild or args.force_rebuild_with_base
    force_rebuild_base = args.force_rebuild_with_base

    if len(task_specs) == 1:
        task_spec = task_specs[0]
        print(f"Building images for task: {task_spec.task_id}")
        base, work, judge = build_all_images(
            task_spec, config, client, force_rebuild=force_rebuild,
            force_rebuild_base=force_rebuild_base,
            verbose=verbose,
        )
        print(f"  Base:  {base}")
        print(f"  Work:  {work}")
        print(f"  Judge: {judge}")
    else:
        print(f"Building images for {len(task_specs)} tasks in parallel...")
        print(f"  (verbose output disabled for multi-task build, use single task or check log files)")

        def _build_one(ts: TaskSpec):
            base, work, judge = build_all_images(
                ts, config, client, force_rebuild=force_rebuild,
                force_rebuild_base=force_rebuild_base,
            )
            return ts.task_id, base, work, judge

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(task_specs)) as ex:
            futures = {ex.submit(_build_one, ts): ts for ts in task_specs}
            for fut in concurrent.futures.as_completed(futures):
                ts = futures[fut]
                try:
                    task_id, base, work, judge = fut.result()
                    print(f"  [{task_id}] OK  base={base}  work={work}  judge={judge}")
                except Exception as e:
                    print(f"  [{ts.task_id}] FAILED: {e}", file=sys.stderr)

    print("Done.")


def cmd_pull(args):
    """Pull pre-built images from a remote registry (parallel)."""
    config = _make_config(args)
    if not config.registry:
        print("Error: --registry or SFORGE_REGISTRY env var required for pull")
        sys.exit(1)

    task_specs = _resolve_tasks(args, config)
    client = docker.from_env()

    if len(task_specs) == 1:
        task_spec = task_specs[0]
        print(f"Pulling images for task: {task_spec.task_id}")
        print(f"  Registry: {config.registry}")
        print(f"  Work:  {task_spec.work_image_key}")
        print(f"  Judge: {task_spec.judge_image_key}")
        base_ok, work_ok, judge_ok = pull_all_images(
            task_spec, config.registry, client
        )
        print(f"  Base:  {'OK' if base_ok else 'FAILED'}")
        print(f"  Work:  {'OK' if work_ok else 'FAILED'}")
        print(f"  Judge: {'OK' if judge_ok else 'FAILED'}")
    else:
        print(f"Pulling images for {len(task_specs)} tasks in parallel...")
        print(f"  Registry: {config.registry}")

        def _pull_one(ts: TaskSpec):
            base_ok, work_ok, judge_ok = pull_all_images(
                ts, config.registry, client
            )
            return ts.task_id, base_ok, work_ok, judge_ok

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(task_specs)) as ex:
            futures = {ex.submit(_pull_one, ts): ts for ts in task_specs}
            for fut in concurrent.futures.as_completed(futures):
                ts = futures[fut]
                try:
                    task_id, base_ok, work_ok, judge_ok = fut.result()
                    status = "OK" if all([base_ok, work_ok, judge_ok]) else "PARTIAL"
                    print(f"  [{task_id}] {status}  base={'OK' if base_ok else 'FAIL'}  work={'OK' if work_ok else 'FAIL'}  judge={'OK' if judge_ok else 'FAIL'}")
                except Exception as e:
                    print(f"  [{ts.task_id}] FAILED: {e}", file=sys.stderr)

    print("Done.")


def cmd_push(args):
    """Push locally-built images to the remote registry (parallel)."""
    config = _make_config(args)
    if not config.registry:
        print("Error: --registry or SFORGE_REGISTRY env var required for push")
        sys.exit(1)

    task_specs = _resolve_tasks(args, config)
    client = docker.from_env()

    if len(task_specs) == 1:
        task_spec = task_specs[0]
        print(f"Pushing images for task: {task_spec.task_id}")
        print(f"  Registry: {config.registry}")
        print(f"  Base hash:  {task_spec.base_image_hash[:12]}")
        print(f"  Work hash:  {task_spec.work_image_hash[:12]}")
        print(f"  Judge hash: {task_spec.judge_image_hash[:12]}")
        base_ok, work_ok, judge_ok = push_all_images(
            task_spec, config.registry, client
        )
        print(f"  Base:  {'OK' if base_ok else 'FAILED'}")
        print(f"  Work:  {'OK' if work_ok else 'FAILED'}")
        print(f"  Judge: {'OK' if judge_ok else 'FAILED'}")
    else:
        print(f"Pushing images for {len(task_specs)} tasks in parallel...")
        print(f"  Registry: {config.registry}")

        def _push_one(ts: TaskSpec):
            base_ok, work_ok, judge_ok = push_all_images(
                ts, config.registry, client
            )
            return ts.task_id, base_ok, work_ok, judge_ok

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(task_specs)) as ex:
            futures = {ex.submit(_push_one, ts): ts for ts in task_specs}
            for fut in concurrent.futures.as_completed(futures):
                ts = futures[fut]
                try:
                    task_id, base_ok, work_ok, judge_ok = fut.result()
                    status = "OK" if all([base_ok, work_ok, judge_ok]) else "PARTIAL"
                    print(f"  [{task_id}] {status}  base={'OK' if base_ok else 'FAIL'}  work={'OK' if work_ok else 'FAIL'}  judge={'OK' if judge_ok else 'FAIL'}")
                except Exception as e:
                    print(f"  [{ts.task_id}] FAILED: {e}", file=sys.stderr)

    print("Done.")



def _run_single_task(
    task_spec: TaskSpec,
    args,
    config: SForgeConfig,
    backend,
    run_id: str,
    verbose: bool = False,
    shutdown_event: threading.Event | None = None,
) -> dict | None:
    """Run agent on one task. Returns a summary dict."""
    run_log_dir = config.log_dir / "runs" / run_id / task_spec.task_id

    if not args.agent:
        print("Error: --agent is required")
        sys.exit(1)

    from sforge.harness.agent import create_agent
    from sforge.harness.run_agent import run_agent

    agent = create_agent(args.agent, config)
    effective_timeout = args.timeout or config.agent_timeout or agent.timeout
    disable_stop_hook = getattr(args, "disable_stop_hook", False)
    disable_auto_eval = getattr(args, "disable_auto_eval", False)
    disable_auto_resume = getattr(args, "disable_auto_resume", False)
    effective_eval_interval = args.eval_interval if args.eval_interval is not None else DEFAULT_EVAL_INTERVAL

    # Resolve internet access: CLI flags override per-task setting
    if getattr(args, "disable_internet", False):
        internet = False
    elif getattr(args, "enable_internet", False):
        internet = True
    else:
        internet = task_spec.internet

    print(f"Running agent '{agent.name}' on task: {task_spec.task_id}")
    print(f"  Work image:  {task_spec.work_image_key}")
    print(f"  Run ID:      {run_id}")
    print(f"  Timeout:     {effective_timeout}s")
    if args.model or config.agent_model or agent.default_model:
        print(f"  Model:       {args.model or config.agent_model or agent.default_model}")
    if not task_spec.game_mode:
        eval_status = f"{effective_eval_interval}s" if not disable_auto_eval and effective_eval_interval > 0 else "disabled"
        print(f"  Auto-eval:   {eval_status}")
        hook_status = "disabled" if disable_stop_hook or not agent.stop_hook else agent.stop_hook
        print(f"  Stop hook:   {hook_status}")
        resume_status = "disabled" if disable_auto_resume or not agent.resume_cmd else "enabled"
        print(f"  Auto-resume: {resume_status}")
    print(f"  Internet:    {'enabled' if internet else 'disabled (judge + API only)'}")
    max_subs = getattr(args, "max_submissions", None)
    sub_cooldown = getattr(args, "submission_cooldown", None)
    if max_subs is not None:
        print(f"  Max submissions: {max_subs}")
    if sub_cooldown is not None:
        print(f"  Submit cooldown: {sub_cooldown}s")
    if config.work_cpu_limit:
        print(f"  Work CPU:    {config.work_cpu_limit}")
    if config.work_mem_limit:
        print(f"  Work mem:    {config.work_mem_limit}")
    if config.judge_cpu_limit:
        print(f"  Judge CPU:   {config.judge_cpu_limit}")
    if config.judge_mem_limit:
        print(f"  Judge mem:   {config.judge_mem_limit}")
    print(f"  Judge URL:   {args.judge_url}")
    print(f"  Log dir:     {run_log_dir}")
    print(f"  Agent log:   {run_log_dir / 'run_agent.log'}")
    print(f"  Agent output: {run_log_dir / 'agent_output.txt'}")
    print()

    result = run_agent(
        task_spec=task_spec,
        agent=agent,
        config=config,
        backend=backend,
        run_id=run_id,
        model=args.model,
        timeout=args.timeout,
        judge_url=args.judge_url,
        eval_interval=effective_eval_interval,
        disable_stop_hook=disable_stop_hook,
        disable_auto_eval=disable_auto_eval,
        disable_auto_resume=disable_auto_resume,
        internet=internet,
        verbose=verbose,
        shutdown_event=shutdown_event,
        max_submissions=getattr(args, "max_submissions", None),
        submission_cooldown=getattr(args, "submission_cooldown", None),
    )

    print(f"\nAgent completed in {result.runtime_seconds:.1f}s")
    print(f"  Timed out:        {result.timed_out}")
    if task_spec.game_mode:
        print(f"  Game sessions:    {result.total_rounds}")
        if result.best_score is not None:
            print(f"  Best score:       {result.best_score:.0f}")
    else:
        print(f"  Total rounds:     {result.total_rounds}")
        print(f"  Best pass rate:   {result.best_pass_rate:.2%}")
        if result.best_score is not None:
            print(f"  Best score:       {result.best_score:.0f}")
        print(f"  Best round:       {result.best_round}")
    print(f"  Final archive:    {run_log_dir / 'final_archive.tar.gz'}")

    combined = {
        "agent": agent.name,
        "task": task_spec.task_id,
        "run_id": run_id,
        "model": args.model or config.agent_model or agent.default_model,
        **result.to_dict(),
    }
    (run_log_dir / "final_result.json").write_text(
        json.dumps(combined, indent=2, ensure_ascii=False)
    )
    print()
    print(json.dumps(combined, indent=2, ensure_ascii=False))
    return combined


def _resolve_experiment_tasks(experiment, config: SForgeConfig) -> list[TaskSpec]:
    """Resolve all tasks listed in an experiment config."""
    tasks_dir = config.tasks_dir
    benchmark = load_benchmark(tasks_dir)
    specs: list[TaskSpec] = []
    for task_id in experiment.tasks:
        task_file = tasks_dir / f"{task_id}.json"
        if not task_file.exists():
            print(f"Error: task '{task_id}' (from experiment config) not found at {task_file}")
            sys.exit(1)
        specs.append(make_task_spec(task_file, benchmark))
    return specs


def _apply_experiment_overrides(
    task_spec: TaskSpec,
    args,
    base_config: SForgeConfig,
    experiment,
) -> tuple[SForgeConfig, argparse.Namespace]:
    """Apply experiment overrides for a single task, respecting the priority chain.

    Priority (highest to lowest):
      1. CLI flags (args)
      2. experiment.tasks.<task_id>.* (per-task override)
      3. experiment.defaults.* (experiment-level defaults)
      4. task JSON / env vars / SForgeConfig defaults (base_config, original args)
    """
    if experiment is None:
        return copy.deepcopy(base_config), copy.copy(args)

    from sforge.harness.experiment import resolve_task_overrides

    merged = resolve_task_overrides(experiment, task_spec.task_id)

    task_config = copy.deepcopy(base_config)

    # Apply experiment-level model config (single model per experiment)
    model_cfg = experiment.model
    if model_cfg:
        if model_cfg.api_key is not None:
            task_config.agent_api_key = model_cfg.api_key
        if model_cfg.api_base_url is not None:
            task_config.agent_api_base_url = model_cfg.api_base_url
        if model_cfg.model is not None:
            task_config.agent_model = model_cfg.model

    if merged.extra_env:
        task_config.agent_extra_env.update(merged.extra_env)

    task_args = copy.copy(args)

    if task_args.agent is None and merged.agent is not None:
        task_args.agent = merged.agent

    if task_args.model is None and merged.model is not None:
        task_args.model = merged.model

    if task_args.timeout is None and merged.timeout is not None:
        task_args.timeout = merged.timeout

    if task_args.eval_interval is None and merged.eval_interval is not None:
        task_args.eval_interval = merged.eval_interval

    if not task_args.disable_stop_hook and merged.disable_stop_hook is not None:
        task_args.disable_stop_hook = merged.disable_stop_hook

    if not task_args.disable_auto_eval and merged.disable_auto_eval is not None:
        task_args.disable_auto_eval = merged.disable_auto_eval

    if not getattr(task_args, "disable_auto_resume", False) and getattr(merged, "disable_auto_resume", None) is not None:
        task_args.disable_auto_resume = merged.disable_auto_resume

    if not task_args.disable_internet and not task_args.enable_internet:
        if merged.internet is not None:
            if merged.internet:
                task_args.enable_internet = True
            else:
                task_args.disable_internet = True

    if task_config.work_cpu_limit is None and merged.work_cpu_limit is not None:
        task_config.work_cpu_limit = merged.work_cpu_limit
    if task_config.work_mem_limit is None and merged.work_mem_limit is not None:
        task_config.work_mem_limit = merged.work_mem_limit
    if task_config.judge_cpu_limit is None and merged.judge_cpu_limit is not None:
        task_config.judge_cpu_limit = merged.judge_cpu_limit
    if task_config.judge_mem_limit is None and merged.judge_mem_limit is not None:
        task_config.judge_mem_limit = merged.judge_mem_limit

    if getattr(task_args, "backend", None) is None and merged.backend is not None:
        task_args.backend = merged.backend
        task_config.backend = merged.backend

    if getattr(task_args, "judge_url", None) == "http://host.docker.internal:8080" and merged.judge_url is not None:
        task_args.judge_url = merged.judge_url

    if getattr(task_args, "max_submissions", None) is None and merged.max_submissions is not None:
        task_args.max_submissions = merged.max_submissions

    if getattr(task_args, "submission_cooldown", None) is None and merged.submission_cooldown is not None:
        task_args.submission_cooldown = merged.submission_cooldown

    return task_config, task_args


def _effective_config_dict(
    task_spec: TaskSpec,
    args,
    config: SForgeConfig,
) -> dict:
    """Build a JSON-serializable dict of the effective run parameters for a task.

    Captures the fully-resolved config (after CLI + experiment + defaults merging)
    so runs are reproducible.  API keys are redacted.
    """
    agent_name = args.agent or None
    model = args.model or config.agent_model or None
    timeout = args.timeout or config.agent_timeout or None
    eval_interval = args.eval_interval if args.eval_interval is not None else DEFAULT_EVAL_INTERVAL

    if getattr(args, "disable_internet", False):
        internet = False
    elif getattr(args, "enable_internet", False):
        internet = True
    else:
        internet = task_spec.internet

    d: dict = {
        "task_id": task_spec.task_id,
        "agent": agent_name,
        "model": model,
        "timeout": timeout,
        "eval_interval": eval_interval,
        "disable_stop_hook": getattr(args, "disable_stop_hook", False),
        "disable_auto_eval": getattr(args, "disable_auto_eval", False),
        "disable_auto_resume": getattr(args, "disable_auto_resume", False),
        "internet": internet,
        "judge_url": getattr(args, "judge_url", None),
        "work_cpu_limit": config.work_cpu_limit,
        "work_mem_limit": config.work_mem_limit,
        "judge_cpu_limit": config.judge_cpu_limit,
        "judge_mem_limit": config.judge_mem_limit,
        "max_submissions": getattr(args, "max_submissions", None),
        "submission_cooldown": getattr(args, "submission_cooldown", None),
    }
    if config.agent_api_base_url:
        d["api_base_url"] = config.agent_api_base_url
    if config.agent_api_key:
        d["api_key"] = config.agent_api_key[:8] + "..."
    if config.agent_extra_env:
        d["extra_env"] = config.agent_extra_env
    # Only when opt-in CPU pinning is set, so unpinned run configs are unchanged.
    cpu_affinity = cpu_affinity_identity(config)
    if cpu_affinity:
        d["cpu_affinity"] = cpu_affinity
    return d


def _apply_resource_profile(config: SForgeConfig, task_spec: TaskSpec, roles):
    """Size local Docker containers like the task's PyroMind Sandboxes.

    The task's deployment profile (deploy/pyromind/RESOURCES.md) replaces the
    CPU/memory limits the task scripts pass; SFORGE_RESOURCE_PROFILE=native
    keeps them.  Returns the config and the record of what was replaced."""
    from sforge.harness.resource_profile import apply_local_profile, summary

    config, record = apply_local_profile(config, task_spec, tuple(roles))
    line = summary(record)
    if line:
        print(line, file=sys.stderr)
    return config, record


def _write_resource_profile(path: Path, record) -> None:
    if record:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")


def cmd_run(args):
    """Run a coding agent on one or more tasks, optionally auto-evaluate."""
    import shutil
    import uuid

    # Load experiment config first so env vars are injected before _make_config
    experiment = None
    if args.experiment:
        from sforge.harness.experiment import load_experiment

        experiment = load_experiment(Path(args.experiment).resolve())

    base_config = _make_config(args)

    # Apply experiment-level backend/judge_url defaults before creating backend
    if experiment:
        from sforge.harness.experiment import resolve_task_overrides
        exp_defaults = experiment.defaults
        if exp_defaults.backend and base_config.backend == "docker" and getattr(args, "backend", None) is None:
            base_config.backend = exp_defaults.backend
        if exp_defaults.judge_url and args.judge_url == "http://host.docker.internal:8080":
            args.judge_url = exp_defaults.judge_url

    # Resolve task list
    if args.task:
        task_specs = _resolve_tasks(args, base_config)
    elif experiment:
        task_specs = _resolve_experiment_tasks(experiment, base_config)
    else:
        print("Error: --task or --experiment is required")
        sys.exit(1)

    backend = create_backend_from_config(base_config)

    run_id = args.run_id or uuid.uuid4().hex[:12]
    multi = len(task_specs) > 1
    verbose = not args.silent and not multi

    # --- Resolve per-task overrides upfront and persist configs ---
    run_root = base_config.log_dir / "runs" / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    if experiment and args.experiment:
        shutil.copy2(Path(args.experiment).resolve(), run_root / "experiment.yaml")

    task_runs: list[tuple[TaskSpec, SForgeConfig, argparse.Namespace]] = []
    unified_tasks: dict[str, dict] = {}

    for ts in task_specs:
        task_config, task_args = _apply_experiment_overrides(
            ts, args, base_config, experiment,
        )
        task_config, resource_record = _apply_resource_profile(
            task_config, ts, ("work", "judge"),
        )
        task_runs.append((ts, task_config, task_args))

        cfg_dict = _effective_config_dict(ts, task_args, task_config)
        if resource_record:
            cfg_dict["resource_profile"] = resource_record
        unified_tasks[ts.task_id] = cfg_dict

        task_log_dir = run_root / ts.task_id
        task_log_dir.mkdir(parents=True, exist_ok=True)
        (task_log_dir / "run_config.json").write_text(
            json.dumps(cfg_dict, indent=2, ensure_ascii=False)
        )

    unified = {
        "run_id": run_id,
        "experiment": args.experiment or None,
        "stagger": args.stagger or (experiment.stagger if experiment else None),
        "tasks": unified_tasks,
    }
    (run_root / "run_config.json").write_text(json.dumps(unified, indent=2, ensure_ascii=False))

    # Resolve stagger: CLI flag wins over experiment YAML
    stagger = args.stagger
    if stagger is None and experiment and experiment.stagger:
        stagger = experiment.stagger

    if multi:
        print(f"Multi-task run (verbose output disabled, check log files for details)")
        print(f"  Run ID:  {run_id}")
        if experiment:
            print(f"  Experiment: {args.experiment}")
        n = len(task_runs)
        stagger_delay = stagger / n if stagger and n > 1 else 0
        if stagger:
            print(f"  Tasks:   {', '.join(t.task_id for t in task_specs)} (staggered over {stagger}s, {stagger_delay:.1f}s apart)")
        else:
            print(f"  Tasks:   {', '.join(t.task_id for t in task_specs)} (all in parallel)")
        print()

    shutdown_event = threading.Event()

    def _sigint_handler(signum, frame):
        if not shutdown_event.is_set():
            shutdown_event.set()
            print("\nShutting down — stopping containers (Ctrl+C again has no effect)...")
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    old_sigint = signal.signal(signal.SIGINT, _sigint_handler)

    def _invoke(ts: TaskSpec, task_config: SForgeConfig, task_args):
        try:
            return ts, _run_single_task(
                ts, task_args, task_config, backend, run_id,
                verbose=verbose, shutdown_event=shutdown_event,
            ), None
        except SystemExit:
            raise
        except Exception as e:
            return ts, None, e

    summaries: list[dict] = []

    try:
        if multi:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(task_runs)) as ex:
                futures = []
                for i, (ts, tc, ta) in enumerate(task_runs):
                    if i > 0 and stagger_delay > 0:
                        time.sleep(stagger_delay)
                        if shutdown_event.is_set():
                            break
                    futures.append(ex.submit(_invoke, ts, tc, ta))
                for fut in concurrent.futures.as_completed(futures):
                    ts, summary, err = fut.result()
                    if err is not None:
                        print(f"\n[{ts.task_id}] FAILED: {err}", file=sys.stderr)
                        summaries.append({"task": ts.task_id, "error": str(err)})
                    elif summary is not None:
                        summaries.append(summary)
        else:
            ts, tc, ta = task_runs[0]
            _, summary, err = _invoke(ts, tc, ta)
            if err is not None:
                raise err
            if summary is not None:
                summaries.append(summary)
    except KeyboardInterrupt:
        if not shutdown_event.is_set():
            shutdown_event.set()
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print("\nShutting down — stopping containers...")
    finally:
        signal.signal(signal.SIGINT, old_sigint)

    if multi:
        run_root = base_config.log_dir / "runs" / run_id
        summary_path = run_root / "summary.json"
        summary_path.write_text(json.dumps({
            "run_id": run_id,
            "tasks": summaries,
        }, indent=2, ensure_ascii=False))

        print(f"\n=== Multi-task summary ({len(task_specs)} tasks) ===")
        for s in summaries:
            tid = s.get("task", "?")
            if "error" in s:
                mark = f"ERROR: {s['error']}"
            elif "best_pass_rate" in s:
                mark = f"{s['best_pass_rate']:.2%}"
                if s.get("best_score") is not None:
                    mark += f"  (score {s['best_score']:.0f})"
            elif "eval" in s and s["eval"]:
                mark = f"{s['eval'].get('pass_rate', 0):.2%}"
            else:
                mark = "-"
            print(f"  {tid:<30} {mark}")
        print(f"\nSummary saved: {summary_path}")


def _print_eval_report(report, *, include_json: bool) -> None:
    """Print a local ``EvalReport`` or authenticated coordinator report."""

    value = report if isinstance(report, dict) else report.to_dict()
    print()
    print("Results:")
    print(f"  Total tests: {value['total_tests']}")
    print(f"  Passed:      {value['passed']}")
    print(f"  Failed:      {value['failed']}")
    print(f"  Errors:      {value['errors']}")
    print(f"  Pass rate:   {value['pass_rate']:.2%}")
    if value.get("score_0_100") is not None:
        print(f"  Score 0-100: {value['score_0_100']:.2f}")
    print(f"  Runtime:     {value['runtime_seconds']:.2f}s")
    print(f"  Timed out:   {value['timed_out']}")

    if include_json:
        print()
        print(json.dumps(value, indent=2, ensure_ascii=False))


def _request_coordinator_final_judge(
    *,
    args,
    config: SForgeConfig,
    task_spec: TaskSpec,
    archive_bytes: bytes,
    final_run_id: str,
) -> dict:
    """Authorize final Judge through the live handoff coordinator only.

    The archive bytes are never uploaded on this endpoint.  Their digest is
    only caller-side confirmation; the coordinator re-opens its immutable
    source-run archive and checks it against the retired Work lease.
    """

    source_run_id = (args.handoff_source_run_id or "").strip()
    judge_url = (args.judge_url or "").strip().rstrip("/")
    if not source_run_id:
        raise ValueError(
            "--handoff-source-run-id is required with "
            "--standalone-final-handoff"
        )
    if not judge_url:
        raise ValueError(
            "--judge-url is required with --standalone-final-handoff"
        )
    parsed_judge_url = urlparse(judge_url)
    if (
        parsed_judge_url.scheme != "http"
        or parsed_judge_url.hostname not in {"127.0.0.1", "::1"}
        or parsed_judge_url.port is None
        or parsed_judge_url.username is not None
        or parsed_judge_url.password is not None
        or parsed_judge_url.path not in {"", "/"}
        or parsed_judge_url.params
        or parsed_judge_url.query
        or parsed_judge_url.fragment
    ):
        raise ValueError(
            "--judge-url for standalone final handoff must be an explicit "
            "loopback HTTP origin with a port"
        )
    if not config.single_gpu_handoff:
        raise ValueError(
            "--standalone-final-handoff requires "
            "SFORGE_SINGLE_GPU_HANDOFF=1"
        )
    if task_spec.portable is None:
        raise ValueError(
            "--standalone-final-handoff requires a portable task contract"
        )
    if not config.trusted_lease_id or not config.accelerator_allocation_key:
        raise ValueError(
            "standalone final handoff requires the scheduler-issued lease "
            "and allocation identities"
        )

    final_round_id = (getattr(args, "final_round_id", None) or "").strip()
    payload = {
        "admin_secret": ADMIN_SECRET,
        "source_run_id": source_run_id,
        "final_run_id": final_run_id,
        "task_id": task_spec.task_id,
        "trusted_lease_id": config.trusted_lease_id,
        "allocation_key": config.accelerator_allocation_key,
        "evaluation_contract_sha256": (
            task_spec.portable.evaluation_contract_sha256
        ),
        "archive_sha256": "sha256:" + hashlib.sha256(archive_bytes).hexdigest(),
        "archive_size": len(archive_bytes),
        "timeout": args.timeout,
    }
    if final_round_id:
        # Task verifiers select their evaluation profile from the round ID
        # they see in the Judge context; a task whose final profile is keyed
        # on a "final-" prefix names it here (default stays "manual-1").
        payload["round_id"] = final_round_id
    endpoint = f"{judge_url}/api/v1/handoff/final-judge"
    read_timeout = (args.timeout or task_spec.judge.eval_timeout) + 120
    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=(10, read_timeout),
        )
        response.raise_for_status()
        body = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise RuntimeError(
            f"coordinator final Judge request failed: {exc}"
        ) from exc

    if not isinstance(body, dict):
        raise RuntimeError("coordinator final Judge response is not an object")
    if body.get("state") != "CLOSED":
        raise RuntimeError(
            "coordinator final Judge did not close the exact retired lease"
        )
    if body.get("recovery_policy") != "never_resume":
        raise RuntimeError(
            "coordinator final Judge response lost the never-resume fence"
        )
    epoch = body.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise RuntimeError("coordinator final Judge response has invalid epoch")
    report = body.get("report")
    if not isinstance(report, dict) or report.get("task_id") != task_spec.task_id:
        raise RuntimeError(
            "coordinator final Judge response has the wrong task report"
        )
    expected_portable_contract = {
        "evaluation_contract_sha256": (
            task_spec.portable.evaluation_contract_sha256
        ),
        "selected_ids": task_spec.portable.defaults.to_dict(),
    }
    report_metrics = report.get("metrics")
    if (
        not isinstance(report_metrics, dict)
        or report_metrics.get("portable_contract")
        != expected_portable_contract
    ):
        raise RuntimeError(
            "coordinator final Judge report has the wrong portable contract"
        )
    for key in (
        "total_tests",
        "passed",
        "failed",
        "errors",
        "pass_rate",
        "runtime_seconds",
        "timed_out",
    ):
        if key not in report:
            raise RuntimeError(
                f"coordinator final Judge report is missing {key}"
            )
    return report


def cmd_eval(args):
    """Submit an archive for evaluation."""
    import uuid

    config = _make_config(args)
    task_spec = _resolve_task(args, config)

    # Read archive
    if args.archive == "-":
        archive_bytes = sys.stdin.buffer.read()
    else:
        archive_path = Path(args.archive)
        if not archive_path.exists():
            print(f"Error: archive file not found: {args.archive}")
            sys.exit(1)
        archive_bytes = archive_path.read_bytes()

    run_id = args.run_id or uuid.uuid4().hex[:12]

    if args.handoff_source_run_id and not args.standalone_final_handoff:
        raise ValueError(
            "--handoff-source-run-id requires --standalone-final-handoff"
        )
    if getattr(args, "final_round_id", None) and not args.standalone_final_handoff:
        raise ValueError("--final-round-id requires --standalone-final-handoff")
    if config.single_gpu_handoff and not args.standalone_final_handoff:
        raise ValueError(
            "ordinary local eval is disabled while single GPU handoff is "
            "enabled; use --standalone-final-handoff with the exact source run"
        )
    if args.standalone_final_handoff:
        print(f"Evaluating retired Work archive for task: {task_spec.task_id}")
        print(f"  Archive size: {len(archive_bytes)} bytes")
        print(f"  Source run:   {args.handoff_source_run_id}")
        print(f"  Final run ID: {run_id}")
        report = _request_coordinator_final_judge(
            args=args,
            config=config,
            task_spec=task_spec,
            archive_bytes=archive_bytes,
            final_run_id=run_id,
        )
        _print_eval_report(report, include_json=args.json)
        return

    config, resource_record = _apply_resource_profile(config, task_spec, ("judge",))
    backend = create_backend_from_config(config)
    submissions_root = (
        config.log_dir / "runs" / run_id / task_spec.task_id / "submissions"
    )
    submissions_root.mkdir(parents=True, exist_ok=True)

    # Pick the next available manual-N subdir so repeated runs with the same
    # --run-id don't silently overwrite prior results.
    existing = [p.name for p in submissions_root.iterdir() if p.is_dir()]
    n = 1
    while f"manual-{n}" in existing:
        n += 1
    sub_log_dir = submissions_root / f"manual-{n}"
    _write_resource_profile(sub_log_dir / "resource_profile.json", resource_record)

    print(f"Evaluating archive for task: {task_spec.task_id}")
    print(f"  Archive size: {len(archive_bytes)} bytes")
    print(f"  Run ID:       {run_id}")
    print(f"  Log dir:      {sub_log_dir}")
    print()

    review = _review_eval_archive(
        config, task_spec, backend, archive_bytes, sub_log_dir,
        skip=getattr(args, "skip_integrity_review", False),
    )
    if review is not None and review.blocking:
        report = _integrity_stop_report(
            task_spec, run_id, archive_bytes, review, sub_log_dir
        )
        _print_eval_report(report, include_json=args.json)
        return

    report = judge_submission(
        task_spec=task_spec,
        archive=archive_bytes,
        config=config,
        backend=backend,
        submission_id=run_id,
        timeout=args.timeout,
        log_dir=sub_log_dir,
        verbose=not args.silent,
        run_id=run_id,
        round_id=f"manual-{n}",
    )

    _print_eval_report(report, include_json=args.json)


def _review_eval_archive(
    config, task_spec, backend, archive: bytes, log_dir: Path, *, skip: bool = False
):
    """Review an archive before an ordinary ``sforge eval`` benchmarks it.

    A local eval is how an archive, typically a run's final, gets its
    authoritative score, so it gets the review an agent submission gets
    (kind ``agent``, as for PyroMind budget-finals): the same policy, cache,
    mode and error policy as the Judge server. Returns None when the review
    is off for agent submissions (``SFORGE_LLM_JUDGE_MODE=off`` or ``KINDS``
    without ``agent``) or skipped with ``--skip-integrity-review``.
    """
    from sforge.harness.llm_judge.bundle import DockerWorkImageBaseline, NoBaseline

    llm_config = config.llm_judge
    if skip or not llm_config.covers("agent"):
        print(
            "WARNING: the LLM integrity review is off for this evaluation; "
            "the archive goes to the GPU Judge unreviewed.",
            file=sys.stderr,
        )
        return None
    # Fail closed like `sforge serve`: an enabled review that cannot run
    # stops the evaluation instead of silently admitting the archive.
    from sforge.harness.llm_judge.config import LLMJudgeConfigError

    try:
        llm_config.check_ready()
    except LLMJudgeConfigError as exc:
        print(
            f"Error: the LLM integrity review cannot run ({exc}). Configure it "
            "as for `sforge serve` (docs/LLM_JUDGE.md). To benchmark a trusted "
            "seed, oracle or deliberate control archive without a review, pass "
            "--skip-integrity-review (or set SFORGE_LLM_JUDGE_MODE=off for a "
            "wrapper script that does not forward flags).",
            file=sys.stderr,
        )
        sys.exit(2)
    reviewer, egress = _build_integrity_reviewer(
        config,
        llm_config,
        backend,
        cache_dir=config.log_dir / "integrity_review_cache",
    )
    baseline = (
        DockerWorkImageBaseline(backend.client)
        if backend.backend_name == "docker"
        else NoBaseline(f"the {backend.backend_name} backend cannot read Work images")
    )
    try:
        reviewer.validate_tasks([task_spec])
        review = reviewer.review(
            task_spec=task_spec, archive=archive, baseline=baseline, log_dir=log_dir
        )
    finally:
        if egress is not None:
            egress.close()
    print(
        f"Integrity review: {review.status.upper()} ({review.reason})"
        + (" -- not benchmarked" if review.blocking else "")
    )
    if review.feedback_to_agent:
        print(f"  feedback: {review.feedback_to_agent}")
    for rule in review.rules:
        print(f"  rule: {rule}")
    print(f"  artifacts: {log_dir / 'integrity_review'}")
    print()
    return review


def _integrity_stop_report(task_spec, submission_id: str, archive: bytes, review, log_dir: Path):
    """Terminal report for an eval the review stops, as the Judge server writes it."""
    from sforge.harness.llm_judge.reviewer import (
        STATUS_VIOLATION,
        rejected_report,
        unavailable_report,
    )
    from sforge.harness.run_evaluation import _durable_atomic_write_bytes, _persist_report
    from sforge.harness.score_rescale import rescale_score

    submitted_at = time.time()
    log_dir.mkdir(parents=True, exist_ok=True)
    if archive:
        _durable_atomic_write_bytes(log_dir / "submission.tar.gz", archive)
    build = rejected_report if review.status == STATUS_VIOLATION else unavailable_report
    report = build(task_spec, submission_id, review, submitted_at=submitted_at)
    report.score_0_100 = rescale_score(task_spec.judge.rescale, report.score)
    _persist_report(log_dir, report)
    return report


def _build_integrity_reviewer(config, llm_config, backend, *, cache_dir: Path | None):
    """Reviewer and (for the agent engine) its egress, as `sforge serve` builds them.

    The caller closes the returned egress when it is not None.
    """
    from sforge.harness.llm_judge.agent_engine import AgentReviewEngine
    from sforge.harness.llm_judge.egress import ReviewEgress
    from sforge.harness.llm_judge.reviewer import IntegrityReviewer

    engine = None
    egress = None
    if llm_config.engine == "agent":
        if backend.backend_name != "docker":
            print("Error: the agent review engine requires the Docker backend")
            sys.exit(1)
        egress = ReviewEgress(
            backend.client,
            mode=llm_config.network,
            plan=AgentReviewEngine.egress_plan(llm_config),
            sidecar_image_resolver=lambda image: image,
        )
        from sforge.harness.resource_profile import review_sandbox_limits

        engine = AgentReviewEngine(
            llm_config, backend=backend, egress=egress,
            sandbox_limits=review_sandbox_limits(config),
        )
    reviewer = IntegrityReviewer(
        llm_config,
        tasks_dir=config.tasks_dir,
        cache_dir=cache_dir,
        engine=engine,
    )
    return reviewer, egress


def cmd_llm_judge(args):
    """Run the LLM integrity review on archives offline (no GPU, no Judge).

    Uses the same reviewer, policy, prompt and cache as the Judge server, so
    it can audit final archives, stored candidates or crafted test cases.
    Exit status: 0 when nothing was rejected, 3 when any archive was found in
    violation, 2 when any review produced no verdict.
    """
    from sforge.harness.llm_judge.bundle import (
        DockerWorkImageBaseline,
        HostDirectoryBaseline,
        NoBaseline,
    )
    from sforge.harness.llm_judge.config import load_llm_judge_config
    from sforge.harness.llm_judge.reviewer import STATUS_ERROR, STATUS_VIOLATION

    config = _make_config(args)
    task_spec = _resolve_task(args, config)
    llm_config = load_llm_judge_config(
        engine=args.engine,
        agent=args.agent,
        agent_runtime_dir=Path(args.agent_runtime_dir) if args.agent_runtime_dir else None,
        codex_auth_json=Path(args.codex_auth_json) if args.codex_auth_json else None,
        image=args.image,
        network=args.network,
        api_base_url=args.api_base_url,
        api_key_file=Path(args.api_key_file) if args.api_key_file else None,
        api_key_env=args.api_key_env,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        votes=args.votes,
        # Offline verdicts report what an enforcing server would decide.
        mode="enforce",
        fallbacks={
            "agent_runtime_dir": config.agent_runtime_dir,
            "codex_auth_json": config.codex_auth_json_path,
        },
    )
    llm_config.check_ready()
    backend = create_backend_from_config(config)
    reviewer, egress = _build_integrity_reviewer(
        config,
        llm_config,
        backend,
        cache_dir=None if args.no_cache else config.log_dir / "integrity_review_cache",
    )
    reviewer.validate_tasks([task_spec])
    if args.baseline_dir:
        baseline = HostDirectoryBaseline(
            Path(args.baseline_dir),
            documents_root=Path(args.documents_root) if args.documents_root else None,
        )
    else:
        baseline = (
            DockerWorkImageBaseline(backend.client)
            if backend.backend_name == "docker"
            else NoBaseline(f"the {backend.backend_name} backend cannot read Work images")
        )
    out_root = (
        Path(args.out_dir)
        if args.out_dir
        else config.log_dir / "integrity_reviews" / time.strftime("%Y%m%dT%H%M%S")
    )
    results = []
    try:
        _review_archives(args, reviewer, task_spec, baseline, out_root, results)
    finally:
        if egress is not None:
            egress.close()
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    statuses = {record["status"] for record in results}
    if STATUS_VIOLATION in statuses:
        sys.exit(3)
    if STATUS_ERROR in statuses:
        sys.exit(2)


def _review_archives(args, reviewer, task_spec, baseline, out_root, results) -> None:
    for index, raw_path in enumerate(args.archive, 1):
        archive_path = Path(raw_path)
        archive = archive_path.read_bytes()
        log_dir = out_root / f"{index:03d}-{archive_path.name.split('.')[0]}"
        review = reviewer.review(
            task_spec=task_spec,
            archive=archive,
            baseline=baseline,
            log_dir=log_dir,
            candidate_version=args.candidate_version,
        )
        record = review.to_record()
        record["archive"] = str(archive_path)
        record["artifacts"] = str(log_dir / "integrity_review")
        results.append(record)
        if not args.json:
            print(f"{archive_path}: {review.status.upper()} ({review.reason})")
            if review.feedback_to_agent:
                print(f"  feedback: {review.feedback_to_agent}")
            for rule in review.rules:
                print(f"  rule: {rule}")
            print(f"  artifacts: {log_dir / 'integrity_review'}")


def _selected_replay_candidates(candidates, selectors, first, last, every=None):
    if selectors and (first is not None or last is not None):
        raise candidate_replay.CandidateReplayError(
            "--candidate cannot be combined with --from-v or --to-v"
        )
    if selectors and every is not None:
        raise candidate_replay.CandidateReplayError(
            "--candidate cannot be combined with --every"
        )
    if first is not None and first < 1:
        raise candidate_replay.CandidateReplayError("--from-v must be at least 1")
    if last is not None and last < 1:
        raise candidate_replay.CandidateReplayError("--to-v must be at least 1")
    if first is not None and last is not None and first > last:
        raise candidate_replay.CandidateReplayError("--from-v exceeds --to-v")
    if every is not None and every < 1:
        raise candidate_replay.CandidateReplayError("--every must be at least 1")

    if selectors:
        lookup = {
            key: candidate
            for candidate in candidates
            for key in (f"v{candidate.sequence}", candidate.candidate_id)
        }
        selected = []
        seen = set()
        for selector in selectors:
            candidate = lookup.get(selector)
            if candidate is None:
                raise candidate_replay.CandidateReplayError(
                    f"candidate selector was not found: {selector}"
                )
            if candidate.candidate_id not in seen:
                selected.append(candidate)
                seen.add(candidate.candidate_id)
        return sorted(selected, key=lambda item: item.sequence)

    in_range = [
        candidate
        for candidate in candidates
        if (first is None or candidate.sequence >= first)
        and (last is None or candidate.sequence <= last)
    ]
    if every is None or every == 1 or not in_range:
        return in_range
    # Keep every Nth candidate of the range, and always keep the last one so
    # the sampled curve still ends at the run's final state.
    sampled = in_range[::every]
    if sampled[-1].candidate_id != in_range[-1].candidate_id:
        sampled.append(in_range[-1])
    return sampled


def _candidate_replay_plan(
    *, store, task_spec, candidates, origin_run_id, replay_run_id,
    measurement_set, evaluator,
):
    source_identity = _candidate_replay_source_identity(candidates)
    return {
        "schema_version": candidate_replay.MEASUREMENT_SCHEMA_VERSION,
        "kind": candidate_replay.PLAN_KIND,
        "store_id": store.store_id,
        "task_id": task_spec.task_id,
        "origin_run_id": origin_run_id,
        "replay_run_id": replay_run_id,
        "measurement_set": measurement_set,
        "evaluator": evaluator,
        **source_identity,
    }


def _candidate_replay_source_identity(candidates):
    """Canonical immutable snapshot of one source run's candidates/events."""

    ordered_events = sorted(
        (
            (event, candidate)
            for candidate in candidates
            for event in candidate.events
        ),
        key=lambda item: (
            item[0].event_sequence or 0,
            item[0].version_id,
        ),
    )
    return {
        "candidates": [
            {
                "sequence": candidate.sequence,
                "candidate_id": candidate.candidate_id,
                "content_sha256": candidate.content_sha256,
                "input_version_id": candidate_replay.input_version(candidate).version_id,
                "input_archive_sha256": (
                    candidate_replay.input_version(candidate).archive_sha256
                ),
            }
            for candidate in candidates
        ],
        "events": [
            {
                "event_sequence": event.event_sequence,
                "version_id": event.version_id,
                "candidate_sequence": candidate.sequence,
                "candidate_id": candidate.candidate_id,
                "content_sha256": candidate.content_sha256,
                "archive_sha256": event.archive_sha256,
                "source": event.source,
                "round_id": event.round_id,
                "label": event.label,
                "created_at": event.created_at,
            }
            for event, candidate in ordered_events
        ],
    }


def _validate_replay_origin_closed(candidates) -> None:
    """Require the runner's unique final capture to seal the source timeline."""

    events = sorted(
        (event for candidate in candidates for event in candidate.events),
        key=lambda event: (event.event_sequence or 0, event.version_id),
    )
    final_events = [event for event in events if event.source == "final"]
    if len(final_events) != 1:
        raise candidate_replay.CandidateReplayError(
            "origin run is not closed: expected exactly one final event before replay"
        )
    if not events or final_events[0].version_id != events[-1].version_id:
        raise candidate_replay.CandidateReplayError(
            "origin run is not closed: final event must be the last stored event"
        )


def _candidate_replay_plan_source_identity(plan):
    candidates = plan.get("candidates")
    events = plan.get("events")
    if not isinstance(candidates, list) or not isinstance(events, list):
        raise candidate_replay.CandidateReplayError(
            "replay plan source snapshot is invalid"
        )
    return {"candidates": candidates, "events": events}


def _resolve_replay_judge_image(
    backend: ContainerBackend,
    task_spec: TaskSpec,
    config: SForgeConfig,
) -> ResolvedImage:
    """Build if needed, then pin the exact image used by every replay point."""

    try:
        if backend.backend_name == "docker":
            from sforge.harness.backend.docker_backend import DockerBackend

            if not isinstance(backend, DockerBackend):
                raise candidate_replay.CandidateReplayError(
                    "Docker replay backend has an unexpected implementation"
                )
            build_judge_image(
                task_spec,
                config,
                backend.client,
                force_rebuild=False,
            )
        return backend.resolve_image_for_execution(task_spec.judge_image_key)
    except candidate_replay.CandidateReplayError:
        raise
    except Exception as exc:
        raise candidate_replay.CandidateReplayError(
            "failed to pin immutable Judge image"
        ) from exc


def _reuse_replay_judge_image(
    backend: ContainerBackend,
    plan,
    task_spec: TaskSpec,
    config: SForgeConfig,
) -> ResolvedImage:
    """Verify and reuse the exact image committed by an existing plan."""

    del config
    try:
        resolved = candidate_replay.resolved_image_from_replay_plan(
            plan,
            expected_backend=backend.backend_name,
            expected_requested_ref=task_spec.judge_image_key,
        )
        backend.verify_resolved_image_for_execution(resolved)
        return resolved
    except candidate_replay.CandidateReplayError:
        raise
    except Exception as exc:
        raise candidate_replay.CandidateReplayError(
            "pinned Judge image is no longer executable"
        ) from exc


def _resolve_replay_task(args, config: SForgeConfig) -> TaskSpec:
    """Normalize the legacy task resolver's print/exit API for replay JSON."""

    captured = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured):
            return _resolve_task(args, config)
    except candidate_replay.CandidateReplayError:
        raise
    except SystemExit as exc:
        raise candidate_replay.CandidateReplayError(
            "failed to resolve replay task"
        ) from exc
    except Exception as exc:
        raise candidate_replay.CandidateReplayError(
            "failed to resolve replay task"
        ) from exc


def cmd_replay_candidates(args):
    """Replay each unique saved code candidate and persist an explicit curve."""

    replay_lease = None
    try:
        try:
            config = _make_config(args)
        except Exception as exc:
            raise candidate_replay.CandidateReplayError(
                "failed to load replay configuration"
            ) from exc
        if config.version_store_root is None:
            raise candidate_replay.CandidateReplayError(
                "no version store is configured; set "
                "SFORGE_VERSION_STORE_ROOT or use --version-store-root"
            )
        origin_run_id = candidate_replay.safe_replay_id(
            args.run_id, field="run ID"
        )
        measurement_set = candidate_replay.safe_replay_id(
            args.measurement_set, field="measurement set"
        )
        task_spec = _resolve_replay_task(args, config)
        if config.backend != "docker":
            raise candidate_replay.CandidateReplayError(
                "candidate replay currently requires the Docker backend; "
                "Kubernetes immutable image resolution is not yet supported"
            )
        config, resource_record = _apply_resource_profile(
            config, task_spec, ("judge",)
        )
        replay_run_id = candidate_replay.safe_replay_id(
            args.replay_run_id
            or candidate_replay.default_replay_run_id(
                task_id=task_spec.task_id,
                origin_run_id=origin_run_id,
                measurement_set=measurement_set,
            ),
            field="replay run ID",
        )
        store = VersionStore.open(config.version_store_root)
        all_candidates = store.list_candidates(
            task=task_spec.task_id,
            run_id=origin_run_id,
        )
        if not all_candidates:
            raise candidate_replay.CandidateReplayError(
                "no saved candidates matched the task and origin run"
            )
        _validate_replay_origin_closed(all_candidates)
        source_identity = _candidate_replay_source_identity(all_candidates)
        selected_candidates = _selected_replay_candidates(
            all_candidates,
            args.candidate,
            args.from_v,
            args.to_v,
            args.every,
        )
        if not selected_candidates:
            raise candidate_replay.CandidateReplayError(
                "the requested candidate range is empty"
            )
        if args.timeout is not None and args.timeout <= 0:
            raise candidate_replay.CandidateReplayError(
                "--timeout must be a positive integer"
            )
        repeat = int(args.repeat or 0)
        if repeat < 0:
            raise candidate_replay.CandidateReplayError(
                "--repeat must not be negative"
            )
        runtime_context = candidate_replay.parse_runtime_context(
            args.evaluator_context
        )
        root = candidate_replay.replay_root(
            config.log_dir,
            replay_run_id=replay_run_id,
            task_id=task_spec.task_id,
            origin_run_id=origin_run_id,
            measurement_set=measurement_set,
        )

        plan_versions = [
            event
            for candidate in all_candidates
            for event in candidate.events
        ]
        integrity_errors = store.verify(plan_versions)
        if integrity_errors:
            raise candidate_replay.CandidateReplayError(
                "saved candidate archive failed integrity verification: "
                + integrity_errors[0]
            )

        if args.dry_run:
            evaluator = candidate_replay.evaluator_descriptor(
                task_spec,
                config,
                timeout=args.timeout,
                runtime_context=runtime_context,
            )
            measurements = {}
            curve = candidate_replay.build_curve(
                candidates=all_candidates,
                measurements=measurements,
                origin_run_id=origin_run_id,
                replay_run_id=replay_run_id,
                measurement_set=measurement_set,
                evaluator=evaluator,
                curve_root=root,
            )
            if args.json:
                curve["selected_candidates"] = [
                    f"v{candidate.sequence}" for candidate in selected_candidates
                ]
                print(json.dumps(curve, indent=2, ensure_ascii=False))
            else:
                print(
                    f"Replay plan: {len(all_candidates)} unique candidates; "
                    f"{len(selected_candidates)} selected for "
                    f"{task_spec.task_id}/{origin_run_id}"
                )
                selected_sequences = {
                    candidate.sequence for candidate in selected_candidates
                }
                for point in curve["points"]:
                    if point["sequence"] not in selected_sequences:
                        continue
                    print(
                        f"  {point['version']:<6} {point['status']:<10} "
                        f"{point['candidate_id']}"
                    )
            return

        replay_lease = candidate_replay.acquire_replay_lock(
            root,
            base=config.log_dir,
        )
        root = replay_lease.root
        _write_resource_profile(root / "resource_profile.json", resource_record)

        plan_path = root / "replay_plan.json"
        plan_exists = candidate_replay.replay_file_exists(
            plan_path,
            label="replay plan",
        )
        if args.resume and not plan_exists:
            raise candidate_replay.CandidateReplayError(
                "--resume requires an existing replay plan"
            )
        if plan_exists and not args.resume:
            raise candidate_replay.CandidateReplayError(
                "a replay plan already exists; pass --resume to continue it"
            )
        existing_plan = (
            candidate_replay.read_replay_plan(plan_path)
            if args.resume
            else None
        )
        expected_source_identity = (
            _candidate_replay_plan_source_identity(existing_plan)
            if existing_plan is not None
            else source_identity
        )
        latest_candidates = store.list_candidates(
            task=task_spec.task_id,
            run_id=origin_run_id,
        )
        if not latest_candidates:
            raise candidate_replay.CandidateReplayError(
                "origin run disappeared while creating the replay plan"
            )
        _validate_replay_origin_closed(latest_candidates)
        if (
            _candidate_replay_source_identity(latest_candidates)
            != expected_source_identity
        ):
            raise candidate_replay.CandidateReplayError(
                "existing replay plan changed because the origin run changed"
                if existing_plan is not None
                else "origin run changed while creating the replay plan"
            )
        all_candidates = latest_candidates
        selected_candidates = _selected_replay_candidates(
            all_candidates,
            args.candidate,
            args.from_v,
            args.to_v,
            args.every,
        )
        integrity_errors = store.verify(
            [
                event
                for candidate in all_candidates
                for event in candidate.events
            ]
        )
        if integrity_errors:
            raise candidate_replay.CandidateReplayError(
                "saved candidate archive failed integrity verification: "
                + integrity_errors[0]
            )
        preloaded_measurements = {
            candidate.candidate_id: existing
            for candidate in all_candidates
            if (
                existing := candidate_replay.read_measurement(
                    candidate_replay.candidate_measurement_path(root, candidate)
                )
            )
            is not None
        }

        try:
            backend = create_backend_from_config(config)
        except Exception as exc:
            raise candidate_replay.CandidateReplayError(
                "failed to initialize replay backend"
            ) from exc
        runtime_adapter_sha256 = runtime_context.get(
            "task_runtime_adapter_sha256"
        )
        loaded_snapshot = candidate_replay.loaded_evaluator_snapshot(
            expected_main_sha256=runtime_adapter_sha256,
        )
        resolved_image = (
            _reuse_replay_judge_image(
                backend,
                existing_plan,
                task_spec,
                config,
            )
            if existing_plan is not None
            else _resolve_replay_judge_image(
                backend,
                task_spec,
                config,
            )
        )
        evaluator = candidate_replay.evaluator_descriptor(
            task_spec,
            config,
            timeout=args.timeout,
            runtime_context=runtime_context,
            resolved_image=resolved_image,
            loaded_snapshot=loaded_snapshot,
        )
        plan = _candidate_replay_plan(
            store=store,
            task_spec=task_spec,
            candidates=all_candidates,
            origin_run_id=origin_run_id,
            replay_run_id=replay_run_id,
            measurement_set=measurement_set,
            evaluator=evaluator,
        )
        latest_candidates = store.list_candidates(
            task=task_spec.task_id,
            run_id=origin_run_id,
        )
        _validate_replay_origin_closed(latest_candidates)
        if (
            _candidate_replay_source_identity(latest_candidates)
            != expected_source_identity
        ):
            raise candidate_replay.CandidateReplayError(
                "existing replay plan changed because the origin run changed"
                if existing_plan is not None
                else "origin run changed while creating the replay plan"
            )
        candidate_replay.ensure_replay_plan(root, plan)
        measurements = {}
        for candidate in all_candidates:
            path = candidate_replay.candidate_measurement_path(root, candidate)
            existing = preloaded_measurements.get(candidate.candidate_id)
            if existing is not None:
                candidate_replay.validate_existing_measurement(
                    existing,
                    candidate=candidate,
                    origin_run_id=origin_run_id,
                    replay_run_id=replay_run_id,
                    measurement_set=measurement_set,
                    evaluator=evaluator,
                )
                recovered = candidate_replay.recover_terminal_attempt(
                    path,
                    existing,
                    candidate=candidate,
                    origin_run_id=origin_run_id,
                    replay_run_id=replay_run_id,
                    measurement_set=measurement_set,
                    evaluator=evaluator,
                )
                if recovered is not None:
                    candidate_replay.write_measurement(path, recovered)
                    existing = recovered
                measurements[candidate.candidate_id] = existing
        existing_paths = [
            candidate_replay.candidate_measurement_path(root, candidate)
            for candidate in all_candidates
            if candidate.candidate_id in measurements
        ]
        if existing_paths and not args.resume:
            raise candidate_replay.CandidateReplayError(
                "measurements already exist; pass --resume to reuse completed "
                "points and retry incomplete points"
            )

        def origin_source_is_unchanged() -> bool:
            observed = store.list_candidates(
                task=task_spec.task_id,
                run_id=origin_run_id,
            )
            try:
                _validate_replay_origin_closed(observed)
            except candidate_replay.CandidateReplayError:
                return False
            return (
                _candidate_replay_source_identity(observed)
                == expected_source_identity
            )

        def stop_for_changed_origin() -> None:
            stale_curve = candidate_replay.build_curve(
                candidates=all_candidates,
                measurements=measurements,
                origin_run_id=origin_run_id,
                replay_run_id=replay_run_id,
                measurement_set=measurement_set,
                evaluator=evaluator,
                curve_root=root,
            )
            stale_curve["terminal_error"] = {
                "kind": "source_run_changed",
                "message": (
                    "origin run changed after its replay snapshot; choose a new "
                    "measurement set after the run is closed"
                ),
            }
            candidate_replay.write_curve(root, stale_curve)
            if args.json:
                print(json.dumps(stale_curve, indent=2, ensure_ascii=False))
            print(
                "Error: origin run changed after the replay plan was pinned; "
                "the curve is marked stale",
                file=sys.stderr,
            )
            sys.exit(1)

        def evaluator_source_is_unchanged() -> bool:
            try:
                modules = evaluator.get("loaded_evaluator_modules")
                if not isinstance(modules, list):
                    return False
                observed_modules = candidate_replay._loaded_sforge_module_names()
                if "__main__" in modules:
                    observed_modules = sorted([*observed_modules, "__main__"])
                adapter_matches = (
                    runtime_adapter_sha256 is None
                    or candidate_replay.loaded_main_source_sha256()
                    == runtime_adapter_sha256
                )
                return (
                    candidate_replay._evaluator_source_sha256()
                    == evaluator.get("evaluator_source_sha256")
                    and observed_modules == modules
                    and adapter_matches
                    and candidate_replay._loaded_evaluator_bytecode_sha256(
                        modules
                    )
                    == evaluator.get("loaded_evaluator_bytecode_sha256")
                )
            except candidate_replay.CandidateReplayError:
                return False

        def stop_for_changed_evaluator() -> None:
            stale_curve = candidate_replay.build_curve(
                candidates=all_candidates,
                measurements=measurements,
                origin_run_id=origin_run_id,
                replay_run_id=replay_run_id,
                measurement_set=measurement_set,
                evaluator=evaluator,
                curve_root=root,
            )
            stale_curve["terminal_error"] = {
                "kind": "evaluator_source_changed",
                "message": (
                    "SForge evaluator source changed after the replay plan was "
                    "pinned; do not mix this batch with later points"
                ),
            }
            candidate_replay.write_curve(root, stale_curve)
            if args.json:
                print(json.dumps(stale_curve, indent=2, ensure_ascii=False))
            print(
                "Error: evaluator source changed after the replay plan was pinned; "
                "the curve is marked stale",
                file=sys.stderr,
            )
            sys.exit(1)

        for _measurement_pass in range(1 + repeat):
            for candidate in selected_candidates:
                if not origin_source_is_unchanged():
                    stop_for_changed_origin()
                if not evaluator_source_is_unchanged():
                    stop_for_changed_evaluator()
                existing = measurements.get(candidate.candidate_id)
                if existing is not None and existing.get("status") == "completed":
                    done = existing.get("replicates")
                    completed_samples = (
                        sum(
                            1
                            for sample in done
                            if isinstance(sample, dict)
                            and sample.get("status") == "completed"
                        )
                        if isinstance(done, list)
                        else 1
                    )
                    if completed_samples >= 1 + repeat:
                        continue

                version = candidate_replay.input_version(candidate)
                archive = store.read_archive(version)
                attempt = (
                    int(existing.get("attempt", 0)) + 1
                    if existing is not None
                    else 1
                )
                started_at = candidate_replay.utc_now()
                measurement_path = candidate_replay.candidate_measurement_path(
                    root, candidate
                )
                running = candidate_replay.measurement_record(
                    candidate=candidate,
                    version=version,
                    origin_run_id=origin_run_id,
                    replay_run_id=replay_run_id,
                    measurement_set=measurement_set,
                    evaluator=evaluator,
                    attempt=attempt,
                    status="running",
                    started_at=started_at,
                )
                candidate_replay.write_measurement(measurement_path, running)
                attempt_path = measurement_path.parent / f"attempt-{attempt:03d}"
                with candidate_replay.pin_replay_directory(
                    attempt_path,
                    must_create=True,
                ) as attempt_directory:
                    attempt_dir = attempt_directory.path
                    if not args.json:
                        print(
                            f"Replaying v{candidate.sequence} "
                            f"({candidate.candidate_id[:22]}...) attempt {attempt}"
                        )
                    submission_id = candidate_replay.replay_submission_id(
                        replay_run_id=replay_run_id,
                        measurement_set=measurement_set,
                        candidate_id=candidate.candidate_id,
                        attempt=attempt,
                    )
                    try:
                        report = judge_submission(
                            task_spec=task_spec,
                            archive=archive,
                            config=config,
                            backend=backend,
                            submission_id=submission_id,
                            timeout=args.timeout,
                            log_dir=attempt_dir,
                            verbose=not args.silent and not args.json,
                            run_id=replay_run_id,
                            round_id=(
                                f"candidate-v{candidate.sequence}-attempt-{attempt}"
                            ),
                            judge_image_ref=resolved_image.execution_ref,
                        )
                        report_value = report.to_dict()
                        if not isinstance(report_value, dict):
                            raise TypeError("evaluator report must be a mapping")
                    except Exception:
                        report_value = candidate_replay.invalid_terminal_report(
                            task_id=task_spec.task_id,
                            submission_id=submission_id,
                            reason="evaluator_exception",
                            summary="Evaluator raised an internal exception",
                        )
                    try:
                        if not evaluator_source_is_unchanged():
                            report_value = candidate_replay.invalid_terminal_report(
                                task_id=task_spec.task_id,
                                submission_id=submission_id,
                                reason="evaluator_source_changed",
                                summary=(
                                    "Evaluator source changed during this measurement"
                                ),
                            )
                        status = (
                            "failed"
                            if candidate_replay.report_is_infrastructure_failure(
                                report_value
                            )
                            else "completed"
                        )
                        completed = candidate_replay.measurement_record(
                            candidate=candidate,
                            version=version,
                            origin_run_id=origin_run_id,
                            replay_run_id=replay_run_id,
                            measurement_set=measurement_set,
                            evaluator=evaluator,
                            attempt=attempt,
                            status=status,
                            started_at=started_at,
                            completed_at=candidate_replay.utc_now(),
                            report=report_value,
                        )
                        candidate_replay.validate_existing_measurement(
                            completed,
                            candidate=candidate,
                            origin_run_id=origin_run_id,
                            replay_run_id=replay_run_id,
                            measurement_set=measurement_set,
                            evaluator=evaluator,
                        )
                    except Exception:
                        report_value = candidate_replay.invalid_terminal_report(
                            task_id=task_spec.task_id,
                            submission_id=submission_id,
                        )
                        status = "failed"
                        completed = candidate_replay.measurement_record(
                            candidate=candidate,
                            version=version,
                            origin_run_id=origin_run_id,
                            replay_run_id=replay_run_id,
                            measurement_set=measurement_set,
                            evaluator=evaluator,
                            attempt=attempt,
                            status=status,
                            started_at=started_at,
                            completed_at=candidate_replay.utc_now(),
                            report=report_value,
                        )
                        candidate_replay.validate_existing_measurement(
                            completed,
                            candidate=candidate,
                            origin_run_id=origin_run_id,
                            replay_run_id=replay_run_id,
                            measurement_set=measurement_set,
                            evaluator=evaluator,
                        )
                    candidate_replay.write_measurement(
                        attempt_dir / "measurement.json", completed
                    )
                if repeat:
                    prior = existing.get("replicates") if existing else None
                    samples = [
                        dict(sample)
                        for sample in (prior if isinstance(prior, list) else [])
                        if isinstance(sample, dict)
                    ]
                    completed_report = completed.get("report") or {}
                    samples.append(
                        {
                            "attempt": attempt,
                            "status": status,
                            "score": completed_report.get("score"),
                            "valid": completed_report.get("valid"),
                            "started_at": started_at,
                            "completed_at": completed.get("completed_at"),
                        }
                    )
                    completed = dict(completed)
                    completed["replicates"] = samples
                    candidate_replay.validate_existing_measurement(
                        completed,
                        candidate=candidate,
                        origin_run_id=origin_run_id,
                        replay_run_id=replay_run_id,
                        measurement_set=measurement_set,
                        evaluator=evaluator,
                    )
                candidate_replay.write_measurement(measurement_path, completed)
                measurements[candidate.candidate_id] = completed
                if not origin_source_is_unchanged():
                    stop_for_changed_origin()
                if not evaluator_source_is_unchanged():
                    stop_for_changed_evaluator()
                curve = candidate_replay.build_curve(
                    candidates=all_candidates,
                    measurements=measurements,
                    origin_run_id=origin_run_id,
                    replay_run_id=replay_run_id,
                    measurement_set=measurement_set,
                    evaluator=evaluator,
                    curve_root=root,
                )
                if status == "failed":
                    curve["terminal_error"] = {
                        "kind": "infrastructure_failure",
                        "candidate": f"v{candidate.sequence}",
                        "message": (
                            "replay infrastructure failed; resume after fixing "
                            "the evaluator"
                        ),
                    }
                candidate_replay.write_curve(root, curve)
                if status == "failed":
                    if args.json:
                        print(json.dumps(curve, indent=2, ensure_ascii=False))
                    print(
                        f"Error: replay infrastructure failed for v{candidate.sequence}; "
                        "rerun with --resume after fixing the evaluator",
                        file=sys.stderr,
                    )
                    sys.exit(1)

        if not origin_source_is_unchanged():
            stop_for_changed_origin()
        if not evaluator_source_is_unchanged():
            stop_for_changed_evaluator()
        curve = candidate_replay.build_curve(
            candidates=all_candidates,
            measurements=measurements,
            origin_run_id=origin_run_id,
            replay_run_id=replay_run_id,
            measurement_set=measurement_set,
            evaluator=evaluator,
            curve_root=root,
        )
        curve_path = candidate_replay.write_curve(root, curve)
        if not origin_source_is_unchanged():
            stop_for_changed_origin()
        if not evaluator_source_is_unchanged():
            stop_for_changed_evaluator()
        if args.json:
            print(json.dumps(curve, indent=2, ensure_ascii=False))
        else:
            print(
                f"Completed {curve['completed_count']}/{curve['candidate_count']} "
                f"candidate measurements"
            )
            print(
                "Curve: "
                f"{replay_lease.display_root / curve_path.name}"
            )
    except (candidate_replay.CandidateReplayError, VersionStoreError) as exc:
        if getattr(args, "json", False):
            print(
                json.dumps(
                    {
                        "kind": "sforge-candidate-replay-error",
                        "status": "error",
                        "error": str(exc),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)
    finally:
        if replay_lease is not None:
            replay_lease.close()


def cmd_serve(args):
    """Start the judge HTTP server."""
    config = _make_config(args)

    from sforge.harness.judge_server import create_app
    import uvicorn

    app = create_app(config)
    hosts = _serve_hosts(args.host)
    print(f"Starting SForge judge server on port {args.port}")
    if len(hosts) == 1:
        uvicorn.run(app, host=hosts[0], port=args.port)
        return
    # Several explicit host-local addresses (for example the Docker bridge
    # gateway that task sidecars reach plus loopback for the standalone final
    # handoff client) share one port; each socket is bound before serving.
    server = uvicorn.Server(uvicorn.Config(app, port=args.port))
    sockets = [_bind_serve_socket(host, args.port) for host in hosts]
    server.run(sockets=sockets)


def _serve_hosts(raw: list[str] | str | None) -> list[str]:
    """Deduplicated, non-empty bind addresses in the order given."""
    if raw is None:
        return ["0.0.0.0"]
    values = [raw] if isinstance(raw, str) else list(raw)
    hosts: list[str] = []
    for value in values:
        host = value.strip()
        if not host:
            raise ValueError("--host must not be empty")
        if host not in hosts:
            hosts.append(host)
    if not hosts:
        raise ValueError("--host must name at least one bind address")
    return hosts


def _bind_serve_socket(host: str, port: int):
    """A bound, inheritable listening socket for one explicit address."""
    import socket

    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    sock = socket.socket(family=family, type=socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
    except OSError as exc:
        sock.close()
        raise RuntimeError(f"cannot bind judge server to {host}:{port}: {exc}") from exc
    sock.set_inheritable(True)
    return sock


def cmd_proxy(args):
    """Start a local API reverse proxy (for use with --disable-internet)."""
    config = _make_config(args)

    http_proxy = config.http_proxy
    https_proxy = config.https_proxy
    if not http_proxy and not https_proxy:
        print(
            "ERROR: No proxy configured. Set SFORGE_HTTPS_PROXY (or HTTPS_PROXY) "
            "before starting the proxy.",
            file=sys.stderr,
        )
        sys.exit(1)

    from sforge.harness.api_proxy import APIProxy

    proxy = APIProxy(
        target_url=args.target,
        http_proxy=http_proxy,
        https_proxy=https_proxy,
        host=args.host,
        port=args.port,
    )
    print(f"Starting API proxy on {proxy.local_url}")
    print(f"  Forwarding to: {proxy.target_url}")
    print(f"  Via proxy:     {https_proxy or http_proxy}")
    print()
    print("Usage: in another terminal, run:")
    print(f"  export SFORGE_AGENT_API_BASE_URL=http://host.docker.internal:{proxy.port}")
    print(f"  python -m sforge run --task <TASK> --agent <AGENT> --disable-internet")
    print()
    proxy.run_forever()


def cmd_visualizer(args):
    """Start the run-results visualizer web UI."""
    from sforge.visualizer.server import create_app as create_viz_app
    import uvicorn

    config = _make_config(args)
    runs_dir = Path(args.runs_dir).resolve()
    tasks_dir = Path(args.tasks_dir).resolve() if args.tasks_dir else config.tasks_dir
    if not tasks_dir.is_dir():
        tasks_dir = None
    if not runs_dir.is_dir():
        print(f"WARNING: runs dir does not exist: {runs_dir}")
    else:
        n = sum(1 for p in runs_dir.iterdir() if p.is_dir())
        print(f"Found {n} run folder(s) in {runs_dir}")
    print(f"Reading tasks from: {tasks_dir if tasks_dir else '(none)'}")
    app = create_viz_app(runs_dir, tasks_dir=tasks_dir)
    print(f"Open: http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_list(args):
    """List available tasks."""
    import unicodedata

    def _display_width(s: str) -> int:
        w = 0
        for ch in s:
            w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        return w

    def _pad(s: str, width: int) -> str:
        return s + " " * (width - _display_width(s))

    config = _make_config(args)
    benchmark = load_benchmark(config.tasks_dir)
    tasks = load_all_tasks(config.tasks_dir, benchmark)

    if not tasks:
        print(f"No tasks found in {config.tasks_dir}")
        return

    col_id = max(_display_width("ID"), max(_display_width(t.task_id) for t in tasks)) + 2
    col_name = max(_display_width("Name"), max(_display_width(t.name) for t in tasks)) + 2
    col_base = max(_display_width("Base Image"), max(_display_width(t.base_image) for t in tasks)) + 2
    col_parser = max(_display_width("Parser"), max(_display_width(t.judge.parser) for t in tasks)) + 2

    print(f"{_pad('ID', col_id)} {_pad('Name', col_name)} {_pad('Base Image', col_base)} {_pad('Parser', col_parser)}")
    print("-" * (col_id + col_name + col_base + col_parser + 3))
    for t in tasks:
        print(f"{_pad(t.task_id, col_id)} {_pad(t.name, col_name)} {_pad(t.base_image, col_base)} {_pad(t.judge.parser, col_parser)}")


def _version_json(version) -> dict:
    return {
        "schema_version": 1,
        "version_id": version.version_id,
        "task_id": version.task_id,
        "task_directory": version.task_directory,
        "run_id": version.run_id,
        "event_sequence": version.event_sequence,
        "candidate_sequence": version.candidate_sequence,
        "source": version.source,
        "round_id": version.round_id,
        "label": version.label,
        "created_at": version.created_at,
        "archive_sha256": version.archive_sha256,
        "content_sha256": version.content_sha256,
        "candidate_id": version.candidate_id,
        "archive_size": version.archive_size,
        "member_count": version.member_count,
        "metadata": version.metadata,
        "archive_path": str(version.archive_path.resolve()),
    }


def _candidate_json(candidate) -> dict:
    return {
        "schema_version": 1,
        "candidate_id": candidate.candidate_id,
        "content_sha256": candidate.content_sha256,
        "sequence": candidate.sequence,
        "task_id": candidate.task_id,
        "task_directory": candidate.task_directory,
        "run_id": candidate.run_id,
        "created_at": candidate.created_at,
        "label": candidate.label,
        "archive_path": str(candidate.archive_path.resolve()),
        "events": [_version_json(event) for event in candidate.events],
    }


def _timeline_event_json(item) -> dict:
    return {
        **_version_json(item.version),
        "event_sequence": item.event_sequence,
        "candidate_sequence": item.candidate_sequence,
        "candidate_id": item.candidate_id,
        "previous_candidate_id": item.previous_candidate_id,
        "relation": item.relation,
        "first_seen_event_sequence": item.first_seen_event_sequence,
    }


def _replay_judge_limits(evaluator):
    if isinstance(evaluator, dict) and "judge_cpu_limit" in evaluator:
        return evaluator.get("judge_cpu_limit"), evaluator.get("judge_mem_limit")
    return None


def _require_same_judge_limits(existing, evaluator, path) -> None:
    """A version measurement set is never continued at another Judge size."""
    before = _replay_judge_limits(existing.get("evaluator"))
    now = _replay_judge_limits(evaluator)
    if before is not None and now is not None and before != now:
        raise candidate_replay.CandidateReplayError(
            f"{path} was measured with Judge limits {before[0]} CPU/{before[1]}, "
            f"not the current {now[0]} CPU/{now[1]} "
            "(deploy/pyromind/RESOURCES.md); choose a new --measurement-set, "
            "or set SFORGE_RESOURCE_PROFILE=native to finish it at its "
            "original size"
        )


def cmd_replay_versions(args):
    """Score every retained implementation of a closed run, one Judge each.

    For tasks that declare ``version_enumeration``, the agent keeps one
    directory per implementation and moves a selector file when a version
    wins. Checkpoint replay therefore re-scores whichever implementation the
    selector pointed at and never scores the ones it skipped. This command
    walks the stored events, extracts every distinct (version, content)
    implementation, and judges each exactly once via the host-side
    candidate_version override — the same x-axis as the agent's own public
    measurements, with zero redundant Judge runs.
    """

    replay_lease = None
    try:
        try:
            config = _make_config(args)
        except Exception as exc:
            raise candidate_replay.CandidateReplayError(
                "failed to load replay configuration"
            ) from exc
        if config.version_store_root is None:
            raise candidate_replay.CandidateReplayError(
                "no version store is configured; set "
                "SFORGE_VERSION_STORE_ROOT or use --version-store-root"
            )
        origin_run_id = candidate_replay.safe_replay_id(
            args.run_id, field="run ID"
        )
        measurement_set = candidate_replay.safe_replay_id(
            args.measurement_set, field="measurement set"
        )
        task_spec = _resolve_replay_task(args, config)
        if config.backend != "docker":
            raise candidate_replay.CandidateReplayError(
                "per-version replay currently requires the Docker backend"
            )
        config, resource_record = _apply_resource_profile(
            config, task_spec, ("judge",)
        )
        if args.timeout is not None and args.timeout <= 0:
            raise candidate_replay.CandidateReplayError(
                "--timeout must be a positive integer"
            )
        if int(getattr(args, "repeat", 0) or 0) < 0:
            raise candidate_replay.CandidateReplayError(
                "--repeat must not be negative"
            )
        replay_run_id = candidate_replay.safe_replay_id(
            args.replay_run_id
            or f"{origin_run_id}-versions-{measurement_set}",
            field="replay run ID",
        )
        store = VersionStore.open(config.version_store_root)
        events = store.list_versions(
            task=task_spec.task_id, run_id=origin_run_id
        )
        if sum(1 for event in events if event.source == "final") != 1:
            raise candidate_replay.CandidateReplayError(
                "origin run is not closed: expected exactly one final event"
            )
        integrity_errors = store.verify(events)
        if integrity_errors:
            raise candidate_replay.CandidateReplayError(
                "saved archive failed integrity verification: "
                + integrity_errors[0]
            )
        implementations = candidate_replay.enumerate_version_implementations(
            store=store, task_spec=task_spec, run_id=origin_run_id
        )
        events_by_id = {event.version_id: event for event in events}

        if args.dry_run:
            if args.json:
                print(json.dumps(implementations, indent=2, ensure_ascii=False))
            else:
                print(
                    f"Per-version plan: {len(implementations)} distinct "
                    f"implementations for {task_spec.task_id}/{origin_run_id}"
                )
                for item in implementations:
                    print(
                        f"  {item['sequence']:>3} {item['implementation_id']}"
                        f"  from {item['first_event_round']}"
                        f"  {item['first_event_label'] or '-'}"
                    )
            return

        root = (
            config.log_dir
            / "runs"
            / replay_run_id
            / task_spec.task_id
            / "version-replays"
            / origin_run_id
            / measurement_set
        )
        replay_lease = candidate_replay.acquire_replay_lock(
            root, base=config.log_dir
        )
        root = replay_lease.root
        _write_resource_profile(root / "resource_profile.json", resource_record)
        try:
            backend = create_backend_from_config(config)
        except Exception as exc:
            raise candidate_replay.CandidateReplayError(
                "failed to initialize replay backend"
            ) from exc
        resolved_image = _resolve_replay_judge_image(backend, task_spec, config)
        evaluator = candidate_replay.evaluator_descriptor(
            task_spec,
            config,
            timeout=args.timeout,
            resolved_image=resolved_image,
        )

        repeat = int(getattr(args, "repeat", 0) or 0)
        measurements: dict[str, dict] = {}
        # Benchmark noise is large enough to reorder neighbouring versions, so
        # a repeated pass measures every implementation again and the curve
        # reports the median rather than whichever pass ran fastest.
        for _measurement_pass in range(1 + repeat):
            for item in implementations:
                impl_dir = root / item["implementation_id"]
                measurement_path = impl_dir / "measurement.json"
                existing = measurements.get(
                    item["implementation_id"]
                ) or candidate_replay.read_version_measurement(measurement_path)
                if existing is not None:
                    _require_same_judge_limits(existing, evaluator, measurement_path)
                if existing is not None and existing.get("status") == "completed":
                    done = existing.get("replicates")
                    completed_samples = (
                        sum(
                            1
                            for sample in done
                            if isinstance(sample, dict)
                            and sample.get("status") == "completed"
                        )
                        if isinstance(done, list)
                        else 1
                    )
                    measurements[item["implementation_id"]] = existing
                    if completed_samples >= 1 + repeat:
                        continue
                impl_dir.mkdir(parents=True, exist_ok=True)
                event = events_by_id[item["first_event_id"]]
                # The source event usually predates the agent moving the
                # selector to this implementation, and judges (including
                # their isolated deployment children, which re-derive the
                # selection from the tree on their own) score whatever the
                # selector names. Rewrite it so the tree itself asks for the
                # implementation being measured; candidate_version stays in
                # the context as a cross-check for judges that read it.
                archive = candidate_replay.rewrite_selector_archive(
                    store.read_archive(event),
                    selector_file=str(
                        (task_spec.version_enumeration or {}).get(
                            "selector_file"
                        )
                    ),
                    version_number=item["version_number"],
                )
                if not args.json:
                    print(
                        f"Judging {item['implementation_id']} "
                        f"(version {item['version_number']}, from "
                        f"{item['first_event_round']})"
                    )
                started_at = candidate_replay.utc_now()
                try:
                    report = judge_submission(
                        task_spec=task_spec,
                        archive=archive,
                        config=config,
                        backend=backend,
                        submission_id=(
                            f"{replay_run_id}-{item['implementation_id']}"
                        ),
                        timeout=args.timeout,
                        log_dir=impl_dir / "judge",
                        verbose=not args.json,
                        run_id=replay_run_id,
                        round_id=f"candidate-{item['implementation_id']}",
                        judge_image_ref=resolved_image.execution_ref,
                        candidate_version=item["version_number"],
                    )
                    report_value = report.to_dict()
                    if not isinstance(report_value, dict):
                        raise TypeError("evaluator report must be a mapping")
                except Exception:
                    report_value = candidate_replay.invalid_terminal_report(
                        task_id=task_spec.task_id,
                        submission_id=(
                            f"{replay_run_id}-{item['implementation_id']}"
                        ),
                        reason="evaluator_exception",
                        summary="Evaluator raised an internal exception",
                    )
                status = (
                    "failed"
                    if candidate_replay.report_is_infrastructure_failure(
                        report_value
                    )
                    else "completed"
                )
                measurement = {
                    "kind": candidate_replay.VERSION_MEASUREMENT_KIND,
                    "schema_version": 1,
                    "measurement_set": measurement_set,
                    "origin_run_id": origin_run_id,
                    "replay_run_id": replay_run_id,
                    "implementation": dict(item),
                    "selector_rewritten_to": item["version_number"],
                    "status": status,
                    "started_at": started_at,
                    "completed_at": candidate_replay.utc_now(),
                    "evaluator": dict(evaluator),
                    "report": report_value,
                }
                if repeat:
                    prior = existing.get("replicates") if existing else None
                    samples = [
                        dict(sample)
                        for sample in (prior if isinstance(prior, list) else [])
                        if isinstance(sample, dict)
                    ]
                    samples.append(
                        {
                            "attempt": len(samples) + 1,
                            "status": status,
                            "score": report_value.get("score"),
                            "valid": report_value.get("valid"),
                            "started_at": started_at,
                            "completed_at": measurement["completed_at"],
                        }
                    )
                    measurement["replicates"] = samples
                candidate_replay.write_measurement(measurement_path, measurement)
                measurements[item["implementation_id"]] = measurement
                if status == "failed":
                    print(
                        f"Error: infrastructure failure at "
                        f"{item['implementation_id']}; rerun to retry",
                        file=sys.stderr,
                    )
                    sys.exit(1)

        direction = str(evaluator.get("score_direction", "maximize"))
        best = None
        points = []
        for item in implementations:
            measurement = measurements.get(item["implementation_id"])
            report = (measurement or {}).get("report") or {}
            score = report.get("score")
            valid = report.get("valid")
            statistics_value = candidate_replay.replicate_statistics(measurement)
            if statistics_value is not None:
                # Repeated passes: report the median so benchmark noise cannot
                # masquerade as a better implementation.
                score = statistics_value["score_median"]
            if (
                measurement is not None
                and valid is True
                and isinstance(score, (int, float))
            ):
                if best is None or (
                    score > best if direction == "maximize" else score < best
                ):
                    best = score
            points.append(
                {
                    **item,
                    "status": (
                        measurement.get("status")
                        if measurement is not None
                        else "planned"
                    ),
                    "score": score if measurement is not None else None,
                    "valid": valid if measurement is not None else None,
                    "history_best": best,
                    **({"statistics": statistics_value} if statistics_value else {}),
                }
            )
        curve = {
            "kind": "sforge-version-curve",
            "schema_version": 1,
            "task_id": task_spec.task_id,
            "origin_run_id": origin_run_id,
            "replay_run_id": replay_run_id,
            "measurement_set": measurement_set,
            "implementation_count": len(implementations),
            "completed_count": sum(
                1
                for value in measurements.values()
                if value.get("status") == "completed"
            ),
            "score_direction": direction,
            "evaluator": dict(evaluator),
            "updated_at": candidate_replay.utc_now(),
            "points": points,
        }
        candidate_replay.write_measurement(root / "versions_curve.json", curve)
        if args.json:
            print(json.dumps(curve, indent=2, ensure_ascii=False))
        else:
            print(
                "Per-version curve: "
                f"{replay_lease.display_root / 'versions_curve.json'}"
            )
            for point in points:
                score_text = (
                    "null" if point["score"] is None else f"{point['score']:.4f}"
                )
                print(
                    f"  v{point['version_number']:03d} "
                    f"{point['implementation_id']:<24} {score_text:>12} "
                    f"valid={point['valid']}"
                )
    except candidate_replay.CandidateReplayError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if replay_lease is not None:
            replay_lease.close()


def cmd_versions(args):
    """List and optionally verify host-persisted submission versions."""

    config = _make_config(args)
    if config.version_store_root is None:
        print(
            "Error: no version store is configured; set "
            "SFORGE_VERSION_STORE_ROOT or use --version-store-root",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        store = VersionStore.open(config.version_store_root)
        versions = store.list_versions(
            task=args.task,
            run_id=args.run_id,
            source=args.source,
        )
        candidates = (
            store.list_candidates(
                task=args.task,
                run_id=args.run_id,
                source=args.source,
            )
            if args.candidates
            else []
        )
        timeline = (
            store.list_timeline(
                task=args.task,
                run_id=args.run_id,
                source=args.source,
            )
            if args.timeline
            else []
        )
    except VersionStoreError as exc:
        print(f"Error: invalid version store: {exc}", file=sys.stderr)
        sys.exit(2)

    if args.json:
        print(
            json.dumps(
                (
                    [_candidate_json(candidate) for candidate in candidates]
                    if args.candidates
                    else (
                        [_timeline_event_json(item) for item in timeline]
                        if args.timeline
                        else [_version_json(version) for version in versions]
                    )
                ),
                indent=2,
                ensure_ascii=False,
            )
        )
    elif args.candidates and not candidates:
        print("No saved candidates.")
    elif args.timeline and not timeline:
        print("No saved timeline events.")
    elif not args.candidates and not args.timeline and not versions:
        print("No saved version events.")
    elif args.candidates:
        headers = (
            "CREATED",
            "VERSION",
            "TASK",
            "RUN",
            "TREE_SHA256",
            "EVENTS",
            "ARCHIVE",
        )
        rows = [
            (
                candidate.created_at,
                f"v{candidate.sequence}",
                candidate.task_directory,
                candidate.run_id,
                candidate.content_sha256[:12],
                str(len(candidate.events)),
                str(candidate.archive_path.resolve()),
            )
            for candidate in candidates
        ]
        widths = [
            max(len(headers[index]), *(len(row[index]) for row in rows))
            for index in range(len(headers) - 1)
        ]
        print(
            "  ".join(
                [
                    *(headers[index].ljust(widths[index]) for index in range(len(widths))),
                    headers[-1],
                ]
            )
        )
        for row in rows:
            print(
                "  ".join(
                    [
                        *(row[index].ljust(widths[index]) for index in range(len(widths))),
                        row[-1],
                    ]
                )
            )
    elif args.timeline:
        headers = (
            "EVENT",
            "CREATED",
            "SOURCE",
            "ROUND",
            "TASK",
            "RUN",
            "VERSION_ID",
            "CANDIDATE",
            "RELATION",
        )
        rows = [
            (
                str(item.event_sequence),
                item.version.created_at,
                item.version.source,
                item.version.round_id or "-",
                item.version.task_directory,
                item.version.run_id,
                item.version.version_id,
                f"v{item.candidate_sequence}",
                item.relation,
            )
            for item in timeline
        ]
        widths = [
            max(len(headers[index]), *(len(row[index]) for row in rows))
            for index in range(len(headers) - 1)
        ]
        print(
            "  ".join(
                [
                    *(headers[index].ljust(widths[index]) for index in range(len(widths))),
                    headers[-1],
                ]
            )
        )
        for row in rows:
            print(
                "  ".join(
                    [
                        *(row[index].ljust(widths[index]) for index in range(len(widths))),
                        row[-1],
                    ]
                )
            )
    else:
        headers = (
            "CREATED",
            "SOURCE",
            "ROUND",
            "TASK",
            "RUN",
            "SHA256",
            "ARCHIVE",
        )
        rows = [
            (
                version.created_at,
                version.source,
                version.round_id or "-",
                version.task_directory,
                version.run_id,
                version.archive_sha256[:12],
                str(version.archive_path.resolve()),
            )
            for version in versions
        ]
        widths = [
            max(len(headers[index]), *(len(row[index]) for row in rows))
            for index in range(len(headers) - 1)
        ]
        print(
            "  ".join(
                [
                    *(headers[index].ljust(widths[index]) for index in range(len(widths))),
                    headers[-1],
                ]
            )
        )
        for row in rows:
            print(
                "  ".join(
                    [
                        *(row[index].ljust(widths[index]) for index in range(len(widths))),
                        row[-1],
                    ]
                )
            )

    if args.verify:
        versions_to_verify = (
            [event for candidate in candidates for event in candidate.events]
            if args.candidates
            else (
                [item.version for item in timeline]
                if args.timeline
                else versions
            )
        )
        errors = store.verify(versions_to_verify)
        if errors:
            for error in errors:
                print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)



def cmd_fetch_tasks(args):
    """Download benchmark task definitions from HuggingFace Hub."""
    from huggingface_hub import snapshot_download
    from sforge.harness.constants import BENCHMARK_REGISTRY, DEFAULT_BENCHMARK

    benchmark = args.benchmark or DEFAULT_BENCHMARK
    repo_id = args.repo or BENCHMARK_REGISTRY.get(benchmark)
    if not repo_id:
        available = ", ".join(sorted(BENCHMARK_REGISTRY))
        print(
            f"Error: unknown benchmark '{benchmark}'. "
            f"Available: {available}\n"
            f"Or specify --repo <org/repo> directly.",
            file=sys.stderr,
        )
        sys.exit(1)

    tasks_dir = Path(args.tasks_dir).resolve() if args.tasks_dir else Path("tasks").resolve()
    revision = args.revision or None

    print(f"Fetching tasks from HuggingFace Hub")
    print(f"  Repo:      {repo_id}")
    if revision:
        print(f"  Revision:  {revision}")
    print(f"  Target:    {tasks_dir}")
    print()

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(tasks_dir),
        revision=revision,
    )

    n_tasks = sum(1 for p in tasks_dir.glob("*.json"))
    print(f"\nDone. {n_tasks} task file(s) in {tasks_dir}")


# --- Main parser ---


def main():
    parser = argparse.ArgumentParser(
        prog="sforge",
        description="SForge: Evaluation harness for coding agents",
    )

    # Global flags
    parser.add_argument(
        "--log-dir", "--output-dir", dest="log_dir", type=Path, default=None,
        help="Directory for generated run output (--log-dir is kept for compatibility)",
    )
    parser.add_argument("--tasks-dir", dest="tasks_dir", type=Path, default=None)
    parser.add_argument("--silent", action="store_true", default=False,
                        help="Suppress detailed log output (auto-enabled for multi-task runs)")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # build
    p_build = subparsers.add_parser("build", help="Build work + judge images")
    p_build_group = p_build.add_mutually_exclusive_group(required=True)
    p_build_group.add_argument("--task", nargs="+",
                               help="One or more task IDs (e.g. --task minitorch gitlet)")
    p_build_group.add_argument("--all", action="store_true", default=False,
                               help="Build images for all tasks")
    p_build.add_argument("--force-rebuild", action="store_true", default=False,
                         help="Force rebuild work + judge images (skip base)")
    p_build.add_argument("--force-rebuild-with-base", action="store_true", default=False,
                         help="Force rebuild ALL images including base")
    p_build.set_defaults(func=cmd_build)

    # pull
    p_pull = subparsers.add_parser("pull", help="Pull pre-built images from remote registry")
    p_pull_group = p_pull.add_mutually_exclusive_group(required=True)
    p_pull_group.add_argument("--task", nargs="+",
                              help="One or more task IDs")
    p_pull_group.add_argument("--all", action="store_true", default=False,
                              help="Pull images for all tasks")
    p_pull.add_argument("--registry", dest="registry", default=None,
                        help="Remote container registry (overrides SFORGE_REGISTRY)")
    p_pull.set_defaults(func=cmd_pull)

    # push
    p_push = subparsers.add_parser("push", help="Push built images to remote registry")
    p_push_group = p_push.add_mutually_exclusive_group(required=True)
    p_push_group.add_argument("--task", nargs="+",
                              help="One or more task IDs")
    p_push_group.add_argument("--all", action="store_true", default=False,
                              help="Push images for all tasks")
    p_push.add_argument("--registry", dest="registry", default=None,
                        help="Remote container registry (overrides SFORGE_REGISTRY)")
    p_push.set_defaults(func=cmd_push)

    # run
    p_run = subparsers.add_parser("run", help="Run an agent on one or more tasks")
    p_run.add_argument("--backend", choices=["docker", "k8s"], default=None,
                       help="Container backend (default from SFORGE_BACKEND or 'docker')")
    p_run.add_argument("--task", default=None, nargs="+",
                       help="One or more task IDs (e.g. --task ahc056 ahc057). "
                            "Multiple tasks are run fully in parallel.")
    p_run.add_argument("--experiment", default=None,
                       help="Path to experiment YAML config file (model config + per-task overrides)")
    p_run.add_argument("--agent", default=None, help="Agent name (claude-code, aider, codex)")
    p_run.add_argument("--model", default=None, help="Model override")
    p_run.add_argument("--timeout", type=int, default=None, help="Agent timeout in seconds")
    p_run.add_argument("--eval-interval", type=int, default=None, help=f"Auto-eval interval in seconds (default {DEFAULT_EVAL_INTERVAL})")
    p_run.add_argument("--disable-auto-eval", action="store_true", default=False,
                       dest="disable_auto_eval", help="Disable background auto-evaluation daemon")
    p_run.add_argument("--disable-stop-hook", action="store_true", default=False,
                       dest="disable_stop_hook", help="Disable the agent stop hook (agent can exit normally)")
    p_run.add_argument("--disable-auto-resume", action="store_true", default=False,
                       dest="disable_auto_resume", help="Disable auto-resume on abnormal agent exit")
    p_run.add_argument("--max-submissions", type=int, default=None, dest="max_submissions",
                       help="Maximum number of agent submissions per run (default: unlimited)")
    p_run.add_argument("--submission-cooldown", type=int, default=None, dest="submission_cooldown",
                       help="Minimum seconds between agent submissions (default: no cooldown)")
    p_run.add_argument("--stagger", type=int, default=None, dest="stagger",
                       help="Spread task launches evenly over N seconds (e.g. --stagger 300)")
    p_run.add_argument("--judge-url", default="http://host.docker.internal:8080", help="Judge server URL")
    p_run.add_argument("--run-id", default=None, help="Run ID for tracking")
    net_group = p_run.add_mutually_exclusive_group()
    net_group.add_argument(
        "--disable-internet", action="store_true", default=False,
        dest="disable_internet",
        help="Force all tasks to run without internet (only judge server + API allowed). "
             "Requires sudo iptables access.",
    )
    net_group.add_argument(
        "--enable-internet", action="store_true", default=False,
        dest="enable_internet",
        help="Force all tasks to run with full internet access (overrides per-task setting).",
    )
    p_run.add_argument("--work-cpu-limit", type=int, default=None, dest="work_cpu_limit",
                        help="Number of CPUs for work containers (e.g. 4)")
    p_run.add_argument("--work-mem-limit", default=None, dest="work_mem_limit",
                        help="Memory limit for work containers (e.g. '8g', '4096m')")
    p_run.add_argument("--judge-cpu-limit", type=int, default=None, dest="judge_cpu_limit",
                        help="Number of CPUs for judge containers (e.g. 2)")
    p_run.add_argument("--judge-mem-limit", default=None, dest="judge_mem_limit",
                        help="Memory limit for judge containers (e.g. '4g')")
    p_run.add_argument(
        "--version-store-root",
        type=Path,
        default=None,
        help="Host-only persistent submission version store",
    )
    p_run.set_defaults(func=cmd_run)

    # eval
    p_eval = subparsers.add_parser("eval", help="Evaluate an archive")
    p_eval.add_argument("--backend", choices=["docker", "k8s"], default=None,
                        help="Container backend (default from SFORGE_BACKEND or 'docker')")
    p_eval.add_argument("--task", required=True, help="Task ID")
    p_eval.add_argument("--archive", required=True, help="Path to .tar.gz archive (or - for stdin)")
    p_eval.add_argument("--run-id", default=None, help="Submission/run ID")
    p_eval.add_argument(
        "--standalone-final-handoff",
        action="store_true",
        help=(
            "Ask the live coordinator to Judge its authenticated retired-Work "
            "archive; never starts a local Judge"
        ),
    )
    p_eval.add_argument(
        "--handoff-source-run-id",
        default=None,
        help="Exact source run owning the retired Work lease",
    )
    p_eval.add_argument(
        "--final-round-id",
        default=None,
        help=(
            "Round ID the coordinator presents to the standalone final Judge "
            "(default manual-1); tasks whose verifier keys the final profile "
            "on a 'final-' prefix pass one here"
        ),
    )
    p_eval.add_argument(
        "--judge-url",
        default=None,
        help="Live handoff coordinator URL (required for standalone final handoff)",
    )
    p_eval.add_argument("--timeout", type=int, default=None, help="Eval timeout in seconds")
    p_eval.add_argument("--judge-cpu-limit", type=int, default=None, dest="judge_cpu_limit",
                        help="Number of CPUs for judge container (e.g. 2)")
    p_eval.add_argument("--judge-mem-limit", default=None, dest="judge_mem_limit",
                        help="Memory limit for judge container (e.g. '4g')")
    p_eval.add_argument("--json", action="store_true", help="Also output JSON report")
    p_eval.add_argument(
        "--skip-integrity-review",
        action="store_true",
        help=(
            "Benchmark without the LLM integrity review: trusted seed/oracle "
            "smoke or deliberate control archives only, never an agent's "
            "final archive"
        ),
    )
    p_eval.set_defaults(func=cmd_eval)

    # llm-judge
    p_llm = subparsers.add_parser(
        "llm-judge",
        help="Run the LLM integrity review on archives (offline, no GPU)",
    )
    p_llm.add_argument("--backend", choices=["docker", "k8s"], default=None)
    p_llm.add_argument("--task", required=True, help="Task ID")
    p_llm.add_argument(
        "--archive", required=True, action="append",
        help="Submission .tar.gz to review (repeatable)",
    )
    p_llm.add_argument(
        "--candidate-version", type=int, default=None,
        help="Review this retained version instead of the archive's selector",
    )
    p_llm.add_argument(
        "--baseline-dir", default=None,
        help="Host directory mirroring the Work cwd (default: read the Work image)",
    )
    p_llm.add_argument(
        "--documents-root", default=None,
        help="Host directory mapped to / for agent-visible briefs with --baseline-dir",
    )
    p_llm.add_argument("--out-dir", default=None, help="Where review artifacts go")
    p_llm.add_argument("--engine", choices=["agent", "api"], default=None)
    p_llm.add_argument("--agent", choices=["codex", "claude-code"], default=None)
    p_llm.add_argument("--agent-runtime-dir", default=None)
    p_llm.add_argument("--codex-auth-json", default=None)
    p_llm.add_argument("--image", choices=["judge", "work"], default=None)
    p_llm.add_argument("--network", choices=["isolated", "bridge"], default=None)
    p_llm.add_argument("--api-base-url", default=None)
    p_llm.add_argument("--api-key-file", default=None)
    p_llm.add_argument("--api-key-env", default=None)
    p_llm.add_argument("--model", default=None)
    p_llm.add_argument("--reasoning-effort", default=None)
    p_llm.add_argument("--votes", type=int, default=None)
    p_llm.add_argument("--no-cache", action="store_true", help="Ignore cached verdicts")
    p_llm.add_argument("--json", action="store_true", help="Print JSON records")
    p_llm.set_defaults(func=cmd_llm_judge)

    # replay-candidates
    p_replay = subparsers.add_parser(
        "replay-candidates",
        help="Evaluate each unique saved code candidate and persist a curve",
    )
    p_replay.add_argument(
        "--backend",
        choices=["docker", "k8s"],
        default=None,
        help="Container backend (default from SFORGE_BACKEND or 'docker')",
    )
    p_replay.add_argument("--task", required=True, help="Task ID")
    p_replay.add_argument(
        "--run-id",
        "--source-run-id",
        dest="run_id",
        required=True,
        help=(
            "Closed origin agent run whose unique candidates will be replayed "
            "(requires its persisted final event, including for --dry-run)"
        ),
    )
    p_replay.add_argument(
        "--measurement-set",
        required=True,
        help="Stable name for one evaluator/runtime measurement series",
    )
    p_replay.add_argument(
        "--replay-run-id",
        default=None,
        help="Output run ID (derived deterministically when omitted)",
    )
    p_replay.add_argument(
        "--version-store-root",
        type=Path,
        default=None,
        help="Host-only persistent submission version store",
    )
    p_replay.add_argument(
        "--candidate",
        action="append",
        default=[],
        help="Replay one vN or candidate-ID (repeatable; default: all)",
    )
    p_replay.add_argument(
        "--from-v",
        type=int,
        default=None,
        help="Replay candidates starting at this inclusive vN sequence",
    )
    p_replay.add_argument(
        "--to-v",
        type=int,
        default=None,
        help="Replay candidates through this inclusive vN sequence",
    )
    p_replay.add_argument(
        "--every",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Sample every Nth candidate of the selected range (v1, v1+N, ...); "
            "the range's last candidate is always included so the curve ends "
            "at the final state. Unsampled candidates stay score: null and can "
            "be filled in later with --resume."
        ),
    )
    p_replay.add_argument(
        "--repeat",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Measure every selected candidate N extra times (default 0, one "
            "measurement each). The curve then reports the median score plus "
            "the observed spread, so benchmark run-to-run noise is not read "
            "as a better candidate. Repeats are spread across passes rather "
            "than run back to back."
        ),
    )
    p_replay.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed points and retry interrupted/infrastructure failures",
    )
    p_replay.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify and print the unique-candidate plan without creating a backend",
    )
    p_replay.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Evaluation timeout in seconds",
    )
    p_replay.add_argument(
        "--evaluator-context",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Pin one task-runtime or external-artifact identity in the "
            "evaluator fingerprint (repeatable)"
        ),
    )
    p_replay.add_argument(
        "--judge-cpu-limit",
        type=int,
        default=None,
        dest="judge_cpu_limit",
        help="Number of CPUs for each Judge container",
    )
    p_replay.add_argument(
        "--judge-mem-limit",
        default=None,
        dest="judge_mem_limit",
        help="Memory limit for each Judge container",
    )
    p_replay.add_argument(
        "--json",
        action="store_true",
        help="Print the candidate-keyed curve as JSON",
    )
    p_replay.set_defaults(func=cmd_replay_candidates)

    # serve
    p_rv = subparsers.add_parser(
        "replay-versions",
        help=(
            "Score every retained implementation of a closed run once, via "
            "the task's version_enumeration declaration"
        ),
    )
    p_rv.add_argument("--backend", choices=["docker", "k8s"], default=None)
    p_rv.add_argument("--task", required=True, help="Task ID")
    p_rv.add_argument(
        "--run-id",
        "--source-run-id",
        dest="run_id",
        required=True,
        help="Closed origin agent run whose implementations will be scored",
    )
    p_rv.add_argument(
        "--measurement-set",
        required=True,
        help="Stable name for one evaluator/runtime measurement series",
    )
    p_rv.add_argument("--replay-run-id", default=None)
    p_rv.add_argument("--version-store-root", type=Path, default=None)
    p_rv.add_argument("--timeout", type=int, default=None)
    p_rv.add_argument(
        "--repeat",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Measure every implementation N extra times (default 0, one "
            "measurement each). The curve then reports the median score plus "
            "the observed spread, so benchmark run-to-run noise is not read "
            "as a better implementation. Repeats are spread across passes "
            "rather than run back to back, so drift affects every point "
            "alike."
        ),
    )
    p_rv.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the distinct implementations without creating a backend",
    )
    p_rv.add_argument("--judge-cpu-limit", type=int, default=None, dest="judge_cpu_limit")
    p_rv.add_argument("--judge-mem-limit", default=None, dest="judge_mem_limit")
    p_rv.add_argument("--json", action="store_true")
    p_rv.set_defaults(func=cmd_replay_versions)

    p_serve = subparsers.add_parser("serve", help="Start judge HTTP server")
    p_serve.add_argument(
        "--host",
        action="append",
        default=None,
        help=(
            "bind address (default 0.0.0.0); repeat to listen on several "
            "explicit addresses on the same port"
        ),
    )
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument(
        "--version-store-root",
        type=Path,
        default=None,
        help="Host-only persistent submission version store",
    )
    p_serve.set_defaults(func=cmd_serve)

    # proxy
    p_proxy = subparsers.add_parser(
        "proxy", help="Start local API reverse proxy (for --disable-internet)"
    )
    p_proxy.add_argument(
        "--target", required=True,
        help="Upstream API URL to forward to (e.g. https://api.anthropic.com)",
    )
    p_proxy.add_argument("--host", default="0.0.0.0")
    p_proxy.add_argument("--port", type=int, default=9090)
    p_proxy.set_defaults(func=cmd_proxy)

    # visualizer
    p_viz = subparsers.add_parser("visualizer", help="Start run-results visualizer web UI")
    p_viz.add_argument("--runs-dir", default="logs/runs", help="Directory of run folders")
    p_viz.add_argument("--tasks-dir", dest="tasks_dir", default=None, help="Directory of task JSONs (for score_direction). Defaults to the harness tasks/ dir.")
    p_viz.add_argument("--host", default="127.0.0.1")
    p_viz.add_argument("--port", type=int, default=8000)
    p_viz.set_defaults(func=cmd_visualizer)

    # list
    p_list = subparsers.add_parser("list", help="List available tasks")
    p_list.set_defaults(func=cmd_list)

    # versions
    p_versions = subparsers.add_parser(
        "versions",
        help="List and verify persistent submission versions",
    )
    p_versions.add_argument(
        "--version-store-root",
        type=Path,
        default=None,
        help="Host-only persistent submission version store",
    )
    p_versions.add_argument(
        "--task",
        default=None,
        help="Filter by task ID or task directory",
    )
    p_versions.add_argument("--run-id", default=None, help="Filter by run ID")
    p_versions.add_argument(
        "--source",
        choices=sorted(VERSION_SOURCES),
        default=None,
        help="Filter by version source",
    )
    p_versions.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable version manifests",
    )
    versions_view = p_versions.add_mutually_exclusive_group()
    versions_view.add_argument(
        "--candidates",
        action="store_true",
        help=(
            "Group archive events into ordered logical code versions by "
            "normalized content hash"
        ),
    )
    versions_view.add_argument(
        "--timeline",
        action="store_true",
        help="Show physical events annotated as first-seen, revisit, or unchanged",
    )
    p_versions.add_argument(
        "--verify",
        action="store_true",
        help="Verify archive sizes and SHA-256 digests",
    )
    p_versions.set_defaults(func=cmd_versions)

    # fetch-tasks
    p_fetch = subparsers.add_parser(
        "fetch-tasks",
        help="Download benchmark task definitions from HuggingFace Hub",
    )
    p_fetch.add_argument(
        "benchmark", nargs="?", default=None,
        help="Benchmark name (default: mlsysforge). Use --repo for unlisted repos.",
    )
    p_fetch.add_argument(
        "--repo", default=None,
        help="HuggingFace repo ID (e.g. Infrathon/Infrathon). "
             "Overrides the benchmark name lookup.",
    )
    p_fetch.add_argument(
        "--revision", default=None,
        help="Git revision (branch, tag, or commit hash) to download.",
    )
    p_fetch.set_defaults(func=cmd_fetch_tasks)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
