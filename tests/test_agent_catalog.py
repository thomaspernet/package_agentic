"""Tests for AgentCatalog — tool groups, conditional tools, agent-level conditions."""

import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents_core.registry.agent_catalog import (
    AgentCatalog,
    AgentYamlEntry,
    _check_condition,
    _resolve_dot_path,
    _resolve_tools,
    load_agent_catalog,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**kwargs) -> SimpleNamespace:
    """Build a nested SimpleNamespace from dot-path kwargs.

    Example: _make_config(**{"agents.web_search.enabled": True})
    """
    root = SimpleNamespace()
    for path, value in kwargs.items():
        parts = path.split(".")
        current = root
        for part in parts[:-1]:
            if not hasattr(current, part):
                setattr(current, part, SimpleNamespace())
            current = getattr(current, part)
        setattr(current, parts[-1], value)
    return root


# ---------------------------------------------------------------------------
# _resolve_dot_path
# ---------------------------------------------------------------------------


class TestResolveDotPath:
    def test_simple_path(self):
        cfg = _make_config(**{"agents.web_search.enabled": True})
        assert _resolve_dot_path(cfg, "agents.web_search.enabled") is True

    def test_missing_segment_returns_none(self):
        cfg = SimpleNamespace()
        assert _resolve_dot_path(cfg, "agents.web_search.enabled") is None

    def test_single_segment(self):
        cfg = SimpleNamespace(debug=True)
        assert _resolve_dot_path(cfg, "debug") is True

    def test_falsy_value(self):
        cfg = _make_config(**{"feature.enabled": False})
        assert _resolve_dot_path(cfg, "feature.enabled") is False


# ---------------------------------------------------------------------------
# _check_condition
# ---------------------------------------------------------------------------


class TestCheckCondition:
    def test_no_when_returns_true(self):
        assert _check_condition(None, None) is True
        assert _check_condition("", None) is True

    def test_no_config_returns_false(self):
        assert _check_condition("agents.web_search.enabled", None) is False

    def test_truthy_condition(self):
        cfg = _make_config(**{"agents.web_search.enabled": True})
        assert _check_condition("agents.web_search.enabled", cfg) is True

    def test_falsy_condition(self):
        cfg = _make_config(**{"agents.web_search.enabled": False})
        assert _check_condition("agents.web_search.enabled", cfg) is False

    def test_missing_path_is_false(self):
        cfg = SimpleNamespace()
        assert _check_condition("agents.web_search.enabled", cfg) is False


# ---------------------------------------------------------------------------
# _resolve_tools
# ---------------------------------------------------------------------------


class TestResolveTools:
    def test_plain_strings(self):
        assert _resolve_tools(["a", "b", "c"], {}, None) == ["a", "b", "c"]

    def test_group_expansion(self):
        groups = {"nav": ["discover", "search", "read"]}
        result = _resolve_tools([{"group": "nav"}, "think"], groups, None)
        assert result == ["discover", "search", "read", "think"]

    def test_unknown_group_raises(self):
        with pytest.raises(KeyError, match="Tool group 'missing'"):
            _resolve_tools([{"group": "missing"}], {}, None)

    def test_conditional_tool_included(self):
        cfg = _make_config(**{"feature.enabled": True})
        raw = [{"tool": "web_search", "when": "feature.enabled"}]
        assert _resolve_tools(raw, {}, cfg) == ["web_search"]

    def test_conditional_tool_excluded(self):
        cfg = _make_config(**{"feature.enabled": False})
        raw = [{"tool": "web_search", "when": "feature.enabled"}]
        assert _resolve_tools(raw, {}, cfg) == []

    def test_conditional_without_config_excluded(self):
        raw = [{"tool": "web_search", "when": "feature.enabled"}]
        assert _resolve_tools(raw, {}, None) == []

    def test_conditional_without_when_included(self):
        raw = [{"tool": "web_search"}]
        assert _resolve_tools(raw, {}, None) == ["web_search"]

    def test_mixed_entries(self):
        groups = {"reasoning": ["think", "plan"]}
        cfg = _make_config(**{"web.enabled": True, "beta.enabled": False})
        raw = [
            "base_tool",
            {"group": "reasoning"},
            {"tool": "web_search", "when": "web.enabled"},
            {"tool": "beta_tool", "when": "beta.enabled"},
        ]
        result = _resolve_tools(raw, groups, cfg)
        assert result == ["base_tool", "think", "plan", "web_search"]

    def test_empty_list(self):
        assert _resolve_tools([], {}, None) == []


# ---------------------------------------------------------------------------
# AgentCatalog
# ---------------------------------------------------------------------------


class TestAgentCatalog:
    def _make_catalog(self):
        return AgentCatalog(
            tool_groups={"nav": ["discover", "search"]},
            raw_agents={
                "chatbot": {
                    "model": "reasoning",
                    "description": "Main agent",
                    "tools": [
                        {"group": "nav"},
                        "think",
                        {"tool": "web_search", "when": "web.enabled"},
                    ],
                },
                "web_agent": {
                    "model": "fast",
                    "description": "Web search",
                    "tools": [],
                    "when": "web.enabled",
                },
                "always_on": {
                    "model": "default",
                    "description": "Always enabled",
                    "tools": ["tool_a"],
                },
            },
        )

    def test_get_resolves_groups_and_conditions(self):
        catalog = self._make_catalog()
        cfg = _make_config(**{"web.enabled": True})
        entry = catalog.get("chatbot", config=cfg)
        assert entry.model == "reasoning"
        assert entry.tools == ["discover", "search", "think", "web_search"]

    def test_get_without_config_skips_conditionals(self):
        catalog = self._make_catalog()
        entry = catalog.get("chatbot")
        assert entry.tools == ["discover", "search", "think"]

    def test_get_missing_agent_raises(self):
        catalog = self._make_catalog()
        with pytest.raises(KeyError, match="not found"):
            catalog.get("nonexistent")

    def test_is_enabled_true(self):
        catalog = self._make_catalog()
        cfg = _make_config(**{"web.enabled": True})
        assert catalog.is_enabled("web_agent", config=cfg) is True

    def test_is_enabled_false(self):
        catalog = self._make_catalog()
        cfg = _make_config(**{"web.enabled": False})
        assert catalog.is_enabled("web_agent", config=cfg) is False

    def test_is_enabled_no_when_always_true(self):
        catalog = self._make_catalog()
        assert catalog.is_enabled("always_on") is True

    def test_is_enabled_missing_agent_false(self):
        catalog = self._make_catalog()
        assert catalog.is_enabled("nonexistent") is False

    def test_is_enabled_no_config_false(self):
        catalog = self._make_catalog()
        assert catalog.is_enabled("web_agent") is False

    def test_list_agents(self):
        catalog = self._make_catalog()
        assert set(catalog.list_agents()) == {"chatbot", "web_agent", "always_on"}


# ---------------------------------------------------------------------------
# load_agent_catalog
# ---------------------------------------------------------------------------


class TestLoadAgentCatalog:
    def test_load_from_file(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            tool_groups:
              reasoning: [think, plan]

            agents:
              my_agent:
                model: fast
                description: Test agent
                tools:
                  - group: reasoning
                  - custom_tool
                  - tool: optional
                    when: feature.on
        """)
        path = tmp_path / "agents.yaml"
        path.write_text(yaml_content)

        catalog = load_agent_catalog(path)
        assert "my_agent" in catalog.list_agents()

        # Without config — conditional skipped
        entry = catalog.get("my_agent")
        assert entry.tools == ["think", "plan", "custom_tool"]

        # With config — conditional included
        cfg = _make_config(**{"feature.on": True})
        entry = catalog.get("my_agent", config=cfg)
        assert entry.tools == ["think", "plan", "custom_tool", "optional"]

    def test_load_missing_file_returns_empty(self, tmp_path):
        catalog = load_agent_catalog(tmp_path / "nonexistent.yaml")
        assert catalog.list_agents() == []

    def test_load_empty_file(self, tmp_path):
        path = tmp_path / "agents.yaml"
        path.write_text("")
        catalog = load_agent_catalog(path)
        assert catalog.list_agents() == []

    def test_load_no_tool_groups(self, tmp_path):
        yaml_content = textwrap.dedent("""\
            agents:
              simple:
                model: default
                description: No groups
                tools: [a, b]
        """)
        path = tmp_path / "agents.yaml"
        path.write_text(yaml_content)

        catalog = load_agent_catalog(path)
        entry = catalog.get("simple")
        assert entry.tools == ["a", "b"]


# ---------------------------------------------------------------------------
# AgentYamlEntry
# ---------------------------------------------------------------------------


class TestAgentYamlEntry:
    def test_defaults(self):
        entry = AgentYamlEntry(model="fast", description="test")
        assert entry.tools == []

    def test_with_tools(self):
        entry = AgentYamlEntry(model="fast", description="test", tools=["a", "b"])
        assert entry.tools == ["a", "b"]
