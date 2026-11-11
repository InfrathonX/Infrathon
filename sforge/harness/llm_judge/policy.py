"""Per-task integrity policies for the LLM review.

Every task must ship ``llm_judge/POLICY.md`` in its task directory (next to
``tasks/<task_id>.json``); the Judge server refuses to start a review-enabled
run for a task without a complete one.  The Markdown body tells
the reviewer what the task requires, what counts as circumventing it, and --
just as important -- which unusual-looking techniques are allowed.  An
optional YAML front matter block configures what the reviewer is shown::

    ---
    instruction_files:          # agent-visible briefs inside the Work image
      - /app/workdir/lab/AGENT_TASK.md
    context_files: []           # other agent-visible files worth reading
    review_exclude: ["*.pt"]    # submitted files listed but never inlined
    private_paths: []           # removed from the review image before review
    version_enumeration:        # only when the task JSON declares none
      root: solutions
      directory_pattern: "objective_v([0-9]{3})"
      selector_file: FINAL_VERSION
    ---

A policy must be safe to show to the agent: the reviewer's feedback is built
from it.  Never describe hidden cases, seeds or Judge internals here.

The body follows one of two layouts, sections in order.  The examples
layout: ``## What the task requires``, ``## Examples: reward hacking``,
``## Examples: instruction violations`` and ``## Examples: legitimate work``
-- task-specific cases on both sides of the line, which the general rules'
definitions extend to cases they do not list.  The older rules layout:
``## What the task requires``, ``## Reject`` and ``## Allow -- never reject
for these``.  docs/LLM_JUDGE_POLICY_AUTHORING.md describes how a policy is
written and checked.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

POLICY_DIRECTORY_NAME = "llm_judge"
POLICY_FILE_NAME = "POLICY.md"
POLICY_MAX_BYTES = 256 * 1024
RULES_LAYOUT = (
    "## What the task requires",
    "## Reject",
    "## Allow",
)
EXAMPLES_LAYOUT = (
    "## What the task requires",
    "## Examples: reward hacking",
    "## Examples: instruction violations",
    "## Examples: legitimate work",
)
# The original layout; a body using any "## Examples:" heading is checked
# against EXAMPLES_LAYOUT instead.
REQUIRED_SECTIONS = RULES_LAYOUT
_FRONT_MATTER_KEYS = {
    "instruction_files",
    "context_files",
    "review_exclude",
    "private_paths",
    "version_enumeration",
}


class PolicyError(ValueError):
    """A task integrity policy file is malformed."""


@dataclass(frozen=True)
class TaskIntegrityPolicy:
    task_id: str
    source: str | None
    body: str
    sha256: str
    instruction_files: tuple[str, ...] = ()
    context_files: tuple[str, ...] = ()
    review_exclude: tuple[str, ...] = ()
    # Absolute paths deleted from the review container before the agent
    # starts (hidden data an unprivileged user could still read).
    private_paths: tuple[str, ...] = ()
    # Review-only selector layout for tasks whose JSON declares none, so the
    # reviewer knows which retained version the Judge evaluates.
    version_enumeration: dict[str, str] | None = None

    @property
    def has_task_rules(self) -> bool:
        return bool(self.body.strip())


def _string_list(value: Any, *, key: str, source: Path) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise PolicyError(f"{source}: {key} must be a list of strings")
    return tuple(item.strip() for item in value)


def _absolute_container_paths(
    values: tuple[str, ...], *, key: str, source: Path
) -> tuple[str, ...]:
    for value in values:
        path = PurePosixPath(value)
        if not path.is_absolute() or ".." in path.parts:
            raise PolicyError(
                f"{source}: {key} entries must be absolute container paths"
            )
    return values


def parse_policy_text(task_id: str, text: str, *, source: Path) -> TaskIntegrityPolicy:
    """Parse one policy document (optional YAML front matter + Markdown)."""

    settings: dict[str, Any] = {}
    body = text
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end < 0:
            raise PolicyError(f"{source}: unterminated front matter")
        try:
            loaded = yaml.safe_load(text[4:end]) or {}
        except yaml.YAMLError as exc:
            raise PolicyError(f"{source}: invalid front matter: {exc}") from exc
        if not isinstance(loaded, dict):
            raise PolicyError(f"{source}: front matter must be a mapping")
        if "enabled" in loaded:
            raise PolicyError(
                f"{source}: the integrity review is mandatory; a policy cannot "
                "disable it (remove 'enabled')"
            )
        unknown = set(loaded) - _FRONT_MATTER_KEYS
        if unknown:
            raise PolicyError(
                f"{source}: unknown front matter keys {sorted(unknown)}"
            )
        settings = loaded
        body = text[end + len("\n---\n"):]
    enumeration = settings.get("version_enumeration")
    if enumeration is not None:
        if (
            not isinstance(enumeration, dict)
            or not all(isinstance(value, str) for value in enumeration.values())
            or set(enumeration) - {"root", "selector_file", "directory_pattern", "file_pattern"}
            or not enumeration.get("root")
            or not enumeration.get("selector_file")
            or bool(enumeration.get("directory_pattern")) == bool(enumeration.get("file_pattern"))
        ):
            raise PolicyError(
                f"{source}: version_enumeration needs root, selector_file and "
                "exactly one of directory_pattern / file_pattern"
            )
    return TaskIntegrityPolicy(
        task_id=task_id,
        source=str(source),
        body=body.strip() + "\n" if body.strip() else "",
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        instruction_files=_absolute_container_paths(
            _string_list(
                settings.get("instruction_files"),
                key="instruction_files",
                source=source,
            ),
            key="instruction_files",
            source=source,
        ),
        context_files=_absolute_container_paths(
            _string_list(
                settings.get("context_files"), key="context_files", source=source
            ),
            key="context_files",
            source=source,
        ),
        review_exclude=_string_list(
            settings.get("review_exclude"), key="review_exclude", source=source
        ),
        version_enumeration=dict(enumeration) if enumeration is not None else None,
        private_paths=_absolute_container_paths(
            _string_list(
                settings.get("private_paths"), key="private_paths", source=source
            ),
            key="private_paths",
            source=source,
        ),
    )


def policy_candidates(
    task_id: str,
    *,
    tasks_dir: Path | None,
    policy_dir: Path | None,
) -> list[Path]:
    """Where a task's policy may live, most specific first."""

    candidates: list[Path] = []
    if policy_dir is not None:
        candidates.append(policy_dir / f"{task_id}.md")
    if tasks_dir is not None:
        task_root = tasks_dir.parent / POLICY_DIRECTORY_NAME
        candidates.append(task_root / f"{task_id}.md")
        candidates.append(task_root / POLICY_FILE_NAME)
    return candidates


def load_task_policy(
    task_id: str,
    *,
    tasks_dir: Path | None,
    policy_dir: Path | None = None,
) -> TaskIntegrityPolicy:
    """Load the first policy that exists, or the empty general-rules policy."""

    for path in policy_candidates(task_id, tasks_dir=tasks_dir, policy_dir=policy_dir):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        if len(raw) > POLICY_MAX_BYTES:
            raise PolicyError(f"{path}: policy exceeds {POLICY_MAX_BYTES} bytes")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PolicyError(f"{path}: policy is not UTF-8") from exc
        return parse_policy_text(task_id, text, source=path)
    return TaskIntegrityPolicy(
        task_id=task_id,
        source=None,
        body="",
        sha256=hashlib.sha256(b"").hexdigest(),
    )


def policy_layout(body: str) -> tuple[str, ...]:
    """The section headings a policy body must contain, in order."""

    if any(line.strip().startswith("## Examples:") for line in body.splitlines()):
        return EXAMPLES_LAYOUT
    return RULES_LAYOUT


def policy_problems(policy: TaskIntegrityPolicy) -> list[str]:
    """Why a loaded policy cannot gate its task (empty when it can)."""

    if policy.source is None:
        return [
            f"no {POLICY_DIRECTORY_NAME}/{POLICY_FILE_NAME} (every task must "
            "ship one; see docs/LLM_JUDGE_POLICY_AUTHORING.md)"
        ]
    problems: list[str] = []
    position = -1
    for heading in policy_layout(policy.body):
        found = next(
            (
                index
                for index, line in enumerate(policy.body.splitlines())
                if line.strip().startswith(heading)
            ),
            None,
        )
        if found is None:
            problems.append(f"{policy.source}: missing section {heading!r}")
        elif found < position:
            problems.append(f"{policy.source}: section {heading!r} is out of order")
        else:
            position = found
    return problems
