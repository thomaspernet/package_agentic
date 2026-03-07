"""Agent YAML catalog — load agent definitions from a YAML file.

Static agent config (model, description, tools) lives in YAML.
Dynamic parts (instructions, output_dataclass, hosted_tools) stay in Python.

Usage:
    from agents_core import load_agent_catalog

    catalog = load_agent_catalog("agents.yaml")
    cfg = catalog.get("chatbot_agent")
    # cfg.model -> "reasoning"
    # cfg.description -> "Knowledge base research agent..."
    # cfg.tools -> ["think", "search", ...]
"""

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AgentYamlEntry(BaseModel):
    """Single agent entry from agents.yaml."""

    model: str
    description: str
    tools: list[str] = []


class AgentCatalog(BaseModel):
    """All agents loaded from agents.yaml."""

    agents: dict[str, AgentYamlEntry] = {}

    def get(self, name: str) -> AgentYamlEntry:
        """Get agent config by name. Raises KeyError if not found."""
        if name not in self.agents:
            available = ", ".join(sorted(self.agents.keys()))
            raise KeyError(
                f"Agent '{name}' not found in agents.yaml. "
                f"Available: {available}"
            )
        return self.agents[name]

    def list_agents(self) -> list[str]:
        """List all agent names in the catalog."""
        return list(self.agents.keys())


def load_agent_catalog(path: str | Path) -> AgentCatalog:
    """Load agent catalog from a YAML file.

    Args:
        path: Path to the agents.yaml file.

    Returns:
        AgentCatalog with all agent entries.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for agent catalog loading. "
            "Install it with: pip install pyyaml"
        )

    path = Path(path)
    if not path.exists():
        logger.warning("agents.yaml not found at %s, using empty catalog", path)
        return AgentCatalog()

    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    return AgentCatalog.model_validate(data)
