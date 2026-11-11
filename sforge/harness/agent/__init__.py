"""Pluggable agent abstraction for SForge.

Provides :class:`Agent` (base class with config as class attributes)
and concrete implementations per coding agent.  Select via ``create_agent()``.
"""

from sforge.harness.agent.base import Agent
from sforge.harness.agent.factory import (
    create_agent,
    get_agent_class,
    list_agent_classes,
)

__all__ = [
    "Agent",
    "create_agent",
    "get_agent_class",
    "list_agent_classes",
]
