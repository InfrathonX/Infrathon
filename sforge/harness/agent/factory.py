"""Agent factory and registry."""

from __future__ import annotations

from sforge.harness.agent.base import Agent
from sforge.harness.config import SForgeConfig

from sforge.harness.agent.claude_code import ClaudeCodeAgent
from sforge.harness.agent.codex import CodexAgent

# ---------------------------------------------------------------------------
# Registry: name → agent class
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[Agent]] = {
    "claude-code": ClaudeCodeAgent,
    "codex": CodexAgent,
}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def get_agent_class(name: str) -> type[Agent]:
    """Look up an agent class by name.  Raises :class:`ValueError` if not found."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown agent: {name!r}. Available: {available}")
    return _REGISTRY[name]


def list_agent_classes() -> list[type[Agent]]:
    """Return all registered agent classes."""
    return list(_REGISTRY.values())


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_agent(name: str, config: SForgeConfig) -> Agent:
    """Create an :class:`Agent` instance for the named agent."""
    cls = get_agent_class(name)
    return cls(config)
