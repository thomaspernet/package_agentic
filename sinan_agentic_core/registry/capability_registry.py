"""Capability Registry — register and resolve named capability factories.

Mirrors :class:`ToolRegistry` for capabilities. A capability factory is a
callable that takes a ``config`` dict (parsed from YAML) and returns a
:class:`Capability` instance.

The two built-in factories — ``turn_budget`` and ``error_recovery`` — are
registered on import so users can refer to them from ``agents.yaml``
without writing any Python. Custom capabilities register via the
:func:`register_capability` decorator::

    from sinan_agentic_core import register_capability, Capability

    @register_capability("audit_log")
    def build_audit_log(config: dict) -> Capability:
        return MyAuditCapability(**config)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..core.capabilities import Capability

CapabilityFactory = Callable[[dict[str, Any]], Capability]


class CapabilityNotFoundError(KeyError):
    """Raised when a YAML capability reference points to an unregistered name."""


class CapabilityRegistry:
    """Central registry mapping capability names to factory callables."""

    def __init__(self) -> None:
        self._factories: dict[str, CapabilityFactory] = {}

    def register(self, name: str, factory: CapabilityFactory) -> None:
        """Register *factory* under *name*. Re-registration overwrites."""
        self._factories[name] = factory

    def is_registered(self, name: str) -> bool:
        return name in self._factories

    def list_names(self) -> list[str]:
        return sorted(self._factories.keys())

    def get(self, name: str) -> CapabilityFactory:
        """Return the factory for *name* or raise :class:`CapabilityNotFoundError`."""
        if name not in self._factories:
            available = ", ".join(self.list_names()) or "<none>"
            raise CapabilityNotFoundError(
                f"Capability '{name}' is not registered. Available: {available}"
            )
        return self._factories[name]

    def build(
        self, name: str, config: dict[str, Any] | None = None
    ) -> Capability:
        """Resolve *name* through the registry and invoke its factory."""
        factory = self.get(name)
        return factory(dict(config) if config else {})


_global_registry = CapabilityRegistry()


def get_capability_registry() -> CapabilityRegistry:
    """Return the process-wide capability registry."""
    return _global_registry


def register_capability(
    name: str,
) -> Callable[[CapabilityFactory], CapabilityFactory]:
    """Decorator: register *factory* under *name* in the global registry.

    Usage::

        @register_capability("audit_log")
        def build_audit_log(config: dict) -> Capability:
            return AuditLog(**config)
    """

    def decorator(factory: CapabilityFactory) -> CapabilityFactory:
        _global_registry.register(name, factory)
        return factory

    return decorator


# ---------------------------------------------------------------------------
# Built-in capability factories
# ---------------------------------------------------------------------------


@register_capability("turn_budget")
def _build_turn_budget(config: dict[str, Any]) -> Capability:
    """Build a :class:`TurnBudget` from YAML config."""
    from ..core.turn_budget import TurnBudget

    return TurnBudget(**config)


@register_capability("error_recovery")
def _build_error_recovery(config: dict[str, Any]) -> Capability:
    """Build a :class:`ToolErrorRecovery` from YAML config.

    Defaults ``tool_registry`` to the global registry so YAML users do not
    need to plumb it through.
    """
    from ..core.tool_error_recovery import ToolErrorRecovery
    from .tool_registry import get_tool_registry

    cfg = dict(config)
    cfg.setdefault("tool_registry", get_tool_registry())
    return ToolErrorRecovery(**cfg)
