"""Small, explicit controller-side setup helpers for the integration demo."""
import json
import os
from pathlib import Path

DEMO = Path(__file__).parent / "demo"


def initialize_store(root: Path, store_id: str):
    root.mkdir(parents=True, exist_ok=False)
    (root / ".sforge-version-store.json").write_text(json.dumps({
        "schema_version": 1, "kind": "sforge-version-store", "store_id": store_id,
    }))
    task_root = root / "tasks/addition"
    (task_root / "runs").mkdir(parents=True)
    (task_root / "task.json").write_text(json.dumps({
        "schema_version": 1, "task_directory": "addition", "task_id": "addition",
    }))


def install_worker_commands():
    from sforge.harness.evolve_scripts import generate_submit_script, generate_save_version_script
    for name, generator in [("sforge-submit", generate_submit_script),
                            ("sforge-save-version", generate_save_version_script)]:
        path = Path("/usr/local/bin") / name
        path.write_text(generator())
        path.chmod(0o755)


if __name__ == "__main__":
    install_worker_commands()
