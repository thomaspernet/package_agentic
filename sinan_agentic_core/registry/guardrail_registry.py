"""Guardrail Registry - Centralized definition of all available guardrails."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GuardrailDefinition:
    """Schema for a single guardrail."""

    name: str
    description: str
    function: Callable[..., Any]
    category: str  # "input", "output"


@dataclass
class GuardrailRegistry:
    """Central registry of all guardrails available to agents.

    Dataclass-driven design makes it easy to add new guardrails.
    """

    _guardrails: dict[str, GuardrailDefinition] = field(default_factory=dict)

    def register(self, guardrail_def: GuardrailDefinition) -> None:
        """Register a new guardrail."""
        self._guardrails[guardrail_def.name] = guardrail_def

    def get_guardrail(self, name: str) -> GuardrailDefinition | None:
        """Get a specific guardrail by name."""
        return self._guardrails.get(name)

    def get_guardrails_by_category(self, category: str) -> list[GuardrailDefinition]:
        """Get all guardrails in a category."""
        return [g for g in self._guardrails.values() if g.category == category]

    def get_guardrail_functions(self, guardrail_names: list[str]) -> list[Callable[..., Any]]:
        """Get actual function objects for given guardrail names."""
        return [
            self._guardrails[name].function for name in guardrail_names if name in self._guardrails
        ]

    def get_all_functions(self) -> dict[str, Callable[..., Any]]:
        """Get all registered guardrail functions as a mapping."""
        return {name: gdef.function for name, gdef in self._guardrails.items()}


# Global guardrail registry instance
_global_registry = GuardrailRegistry()


def get_guardrail_registry() -> GuardrailRegistry:
    """Get the global guardrail registry."""
    return _global_registry


def register_guardrail(
    name: str,
    description: str,
    category: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register a guardrail.

    Usage:
        @register_guardrail(
            name="validate_cypher_syntax",
            description="Validate Cypher query syntax",
            category="input"
        )
        @input_guardrail
        async def validate_cypher_syntax(ctx, value):
            ...
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        guardrail_def = GuardrailDefinition(
            name=name,
            description=description,
            function=func,
            category=category,
        )
        _global_registry.register(guardrail_def)
        return func

    return decorator
