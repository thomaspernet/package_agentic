"""Core agent execution components.

This module contains fundamental components for agent execution:
- BaseAgentRunner: Abstract base class for all agent runners
- Capability: Pluggable agent behavior primitive
- TurnBudget: Soft turn budget with self-extension (Capability)
- ToolErrorRecovery: Intelligent tool error tracking (Capability)
"""

from .base_runner import BaseAgentRunner
from .capabilities import Capability
from .errors import structured_tool_error
from .tool_error_recovery import ToolErrorRecovery
from .turn_budget import TurnBudget

__all__ = [
    "BaseAgentRunner",
    "Capability",
    "structured_tool_error",
    "ToolErrorRecovery",
    "TurnBudget",
]
