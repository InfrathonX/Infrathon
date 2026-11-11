"""`python -m sforge.visualizer` — standalone entry for the visualizer."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from sforge.visualizer.server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="sforge-visualizer")
    parser.add_argument("--runs-dir", default="logs/runs", help="Directory of run folders")
    parser.add_argument("--tasks-dir", default="tasks", help="Directory of task JSONs (for score_direction)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    # tasks-dir provides per-task metadata (score_direction). Degrade gracefully
    # to None when the directory is absent so the visualizer still runs.
    tasks_dir = Path(args.tasks_dir)
    if not tasks_dir.is_dir():
        tasks_dir = None
    app = create_app(runs_dir, tasks_dir=tasks_dir)
    print(f"SForge visualizer serving runs from: {runs_dir.resolve()}")
    print(f"SForge visualizer reading tasks from: {tasks_dir.resolve() if tasks_dir else '(none)'}")
    print(f"Open: http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
