"""Minimal protocols required by the pinned mini-SWE-agent DefaultAgent."""

from __future__ import annotations

from typing import Any, Protocol

__version__ = "1.14.4"


class Model(Protocol):
    cost: float
    n_calls: int

    def query(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]: ...

    def get_template_vars(self) -> dict[str, Any]: ...


class Environment(Protocol):
    def execute(self, command: str, cwd: str = "") -> dict[str, Any]: ...

    def get_template_vars(self) -> dict[str, Any]: ...
