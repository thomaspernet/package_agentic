"""Example: declare capabilities directly in agents.yaml.

This example shows two ways to wire capabilities through YAML:

1. **Shorthand keys** — ``turn_budget`` and ``error_recovery`` are
   built-in. Set them at the top level of an agent entry.
2. **Explicit list** — the ``capabilities:`` block references any
   registered capability by name. Custom capabilities register through
   ``@register_capability`` and are referenced exactly like built-ins.

Run::

    python examples/yaml_capabilities/run.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agents import RunContextWrapper

from sinan_agentic_core import (
    AgentDefinition,
    Capability,
    load_agent_catalog,
    register_agent,
    register_capability,
)


# ---------------------------------------------------------------------------
# Custom capability — registered under a name that agents.yaml can reference.
# ---------------------------------------------------------------------------


class AuditLog(Capability):
    """Records every LLM start with the configured label."""

    def __init__(self, label: str = "audit") -> None:
        self.label = label
        self.events: list[str] = []

    def instructions(self, ctx: RunContextWrapper[Any]) -> str | None:
        return f"[{self.label}] active"

    def on_llm_start(
        self,
        ctx: RunContextWrapper[Any],
        agent: Any,
        system_prompt: str | None,
        input_items: Any,
    ) -> None:
        self.events.append(f"llm_start[{self.label}]")


@register_capability("audit_log")
def build_audit_log(config: dict[str, Any]) -> Capability:
    return AuditLog(**config)


# ---------------------------------------------------------------------------
# Wire YAML -> AgentDefinition.
# ---------------------------------------------------------------------------


def main() -> None:
    catalog = load_agent_catalog(Path(__file__).parent / "agents.yaml")
    entry = catalog.get("scribe")

    capabilities = entry.build_capabilities()
    print(f"Loaded {len(capabilities)} capabilities for 'scribe':")
    for cap in capabilities:
        print(f"  - {type(cap).__name__}")

    register_agent(
        AgentDefinition(
            name=entry.description.split(".")[0].strip().lower() or "scribe",
            description=entry.description,
            instructions="You are a scribe. Write a single sentence and stop.",
            model=entry.model,
            tools=[],
            capabilities=capabilities,
        )
    )


if __name__ == "__main__":
    main()
