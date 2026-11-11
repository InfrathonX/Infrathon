#!/usr/bin/env python3
"""The Infrathon release task list (scripts/infrathon/tasks.json).

    scripts/infrathon/tasks.py table                      markdown support table (README)
    scripts/infrathon/tasks.py check <task directory> <hardware> [--allow-untested]

``check`` exits 0 when the task is released and may be published for that
hardware type: ``qualified``, or ``untested`` with ``--allow-untested``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LIST = Path(__file__).resolve().with_name("tasks.json")
HARDWARE_TYPES = ("h100-94gb-400w", "h100-80gb-700w")
STATUSES = ("qualified", "untested", "pending", "unsupported")


def load() -> dict:
    document = json.loads(LIST.read_text(encoding="utf-8"))
    for task, entry in document["tasks"].items():
        hardware = entry.get("hardware", {})
        if entry.get("release") is not None and set(hardware) != set(HARDWARE_TYPES):
            raise SystemExit(f"{LIST}: {task} must give a status for each of {HARDWARE_TYPES}")
        for hw, status in hardware.items():
            if hw not in HARDWARE_TYPES or status not in STATUSES:
                raise SystemExit(f"{LIST}: {task}: bad hardware status {hw}={status}")
    return document


def table(document: dict) -> str:
    rows = ["| task | " + " | ".join(HARDWARE_TYPES) + " | note |",
            "|---|" + "---|" * len(HARDWARE_TYPES) + "---|"]
    for task, entry in sorted(document["tasks"].items()):
        if entry.get("release") is None:
            cells = ["not released"] * len(HARDWARE_TYPES)
        else:
            cells = [entry["hardware"][hw] for hw in HARDWARE_TYPES]
        rows.append(f"| `{task}` | " + " | ".join(cells) + f" | {entry.get('note', '')} |")
    return "\n".join(rows) + "\n"


def check(document: dict, task: str, hardware: str, allow_untested: bool) -> str | None:
    """None when publishable, else the reason it is not."""
    entry = document["tasks"].get(task)
    if entry is None:
        return f"{task} is not in {LIST.name}"
    if entry.get("release") is None:
        return f"{task} is not part of the release: {entry.get('note', '')}"
    status = entry["hardware"].get(hardware)
    if status == "qualified" or (status == "untested" and allow_untested):
        return None
    hint = " (pass --allow-untested after a smoke run)" if status == "untested" else ""
    return f"{task} is {status} on {hardware}{hint}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("table")
    checker = sub.add_parser("check")
    checker.add_argument("task")
    checker.add_argument("hardware", choices=HARDWARE_TYPES)
    checker.add_argument("--allow-untested", action="store_true")
    args = parser.parse_args()
    document = load()
    if args.command == "table":
        sys.stdout.write(table(document))
        return 0
    reason = check(document, args.task, args.hardware, args.allow_untested)
    if reason:
        print(f"ERROR: {reason}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
