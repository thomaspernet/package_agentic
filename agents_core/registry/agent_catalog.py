"""Agent YAML catalog — load agent definitions from a YAML file.

Static agent config (model, description, tools) lives in YAML.
Dynamic parts (instructions, output_dataclass, hosted_tools) stay in Python.

Features:
  - tool_groups: reusable named tool sets, referenced via ``group: name``
  - Conditional tools: ``tool: name`` + ``when: dot.path`` (resolved against config)
  - Agent-level conditions: ``when: dot.path`` on the agent entry

Usage::

    from agents_core import load_agent_catalog

    catalog = load_agent_catalog("agents.yaml")

    # Resolve tools (expand groups, evaluate conditions)
    cfg = catalog.get("chatbot_agent", config=my_config)
    cfg.model   # "reasoning"
    cfg.tools   # ["think", "discover", ...] — groups expanded, conditions evaluated

    # Check agent-level condition
    if catalog.is_enabled("web_search_agent", config=my_config):
        ...
"""

import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public resolved type
# ---------------------------------------------------------------------------


class AgentYamlEntry(BaseModel):
    """Resolved agent entry — tools are plain strings."""

    model: str
    description: str
    tools: list[str] = []


# ---------------------------------------------------------------------------
# Config path resolution
# ---------------------------------------------------------------------------


def _resolve_dot_path(obj: object, path: str) -> Any:
    """Navigate a dot-separated attribute path on *obj*.

    Returns ``None`` when any segment is missing.
    """
    current: Any = obj
    for part in path.split("."):
        try:
            current = getattr(current, part)
        except AttributeError:
            return None
    return current


def _check_condition(when: str | None, config: object | None) -> bool:
    """Evaluate a ``when`` condition against *config*."""
    if not when:
        return True
    if config is None:
        return False
    return bool(_resolve_dot_path(config, when))


# ---------------------------------------------------------------------------
# Tool resolution
# ---------------------------------------------------------------------------


def _resolve_tools(
    raw_tools: list[Any],
    tool_groups: dict[str, list[str]],
    config: object | None,
) -> list[str]:
    """Resolve mixed tool entries into a plain string list.

    Supported entry formats:
      - ``"tool_name"`` — included as-is
      - ``{group: "group_name"}`` — expanded from *tool_groups*
      - ``{tool: "tool_name", when: "dot.path"}`` — included if condition is truthy
    """
    resolved: list[str] = []
    for item in raw_tools:
        if isinstance(item, str):
            resolved.append(item)
        elif isinstance(item, dict):
            if "group" in item:
                group_name = item["group"]
                if group_name not in tool_groups:
                    available = ", ".join(sorted(tool_groups.keys()))
                    raise KeyError(
                        f"Tool group '{group_name}' not found. "
                        f"Available: {available}"
                    )
                resolved.extend(tool_groups[group_name])
            elif "tool" in item:
                if _check_condition(item.get("when"), config):
                    resolved.append(item["tool"])
    return resolved


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


class AgentCatalog:
    """Agent catalog loaded from ``agents.yaml``.

    Holds raw YAML data and resolves tool groups / conditions on ``get()``.
    """

    def __init__(
        self,
        tool_groups: dict[str, list[str]],
        raw_agents: dict[str, dict[str, Any]],
    ) -> None:
        self._tool_groups = tool_groups
        self._raw_agents = raw_agents

    def get(
        self,
        name: str,
        config: object | None = None,
    ) -> AgentYamlEntry:
        """Get a resolved agent entry.

        Groups are expanded and conditional tools are evaluated against
        *config*.  If *config* is ``None``, conditional tools are skipped.
        """
        if name not in self._raw_agents:
            available = ", ".join(sorted(self._raw_agents.keys()))
            raise KeyError(
                f"Agent '{name}' not found in agents.yaml. "
                f"Available: {available}"
            )
        raw = self._raw_agents[name]
        return AgentYamlEntry(
            model=raw["model"],
            description=raw["description"],
            tools=_resolve_tools(
                raw.get("tools", []), self._tool_groups, config
            ),
        )

    def is_enabled(
        self,
        name: str,
        config: object | None = None,
    ) -> bool:
        """Check if an agent passes its ``when`` condition.

        Returns ``True`` when the agent has no ``when`` clause.
        Returns ``False`` when the agent is not in the catalog.
        """
        if name not in self._raw_agents:
            return False
        when = self._raw_agents[name].get("when")
        if not when:
            return True
        return _check_condition(when, config)

    def list_agents(self) -> list[str]:
        """List all agent names in the catalog."""
        return list(self._raw_agents.keys())


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


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
        return AgentCatalog(tool_groups={}, raw_agents={})

    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    return AgentCatalog(
        tool_groups=data.get("tool_groups", {}),
        raw_agents=data.get("agents", {}),
    )
