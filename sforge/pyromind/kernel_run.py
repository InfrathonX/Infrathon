"""Compatibility imports for the original FP8 launcher; lifecycle lives in runner."""
from sforge.pyromind.runner import (
    ArtifactMirror, _drain, _private_installer, _private_module, duration_seconds,
    guard_judge_backend, initialize_task_store, private_runtime_diagnostics,
    public_artifact_paths, run_task, supervisor_source, termination_cleanup,
    validate_no_credentials, verify_shared_version,
)


def run_kernel(args):
    from pathlib import Path
    if not getattr(args, 'deployment', None):
        args.deployment = Path(__file__).resolve().parents[2] / 'deploy/pyromind/kernel/deployment.json'
    return run_task(args)
