"""Tests for :class:`ToolTracer` (core/tool_tracer.py).

Covers:
- Lifecycle hooks fire in order across multiple tool calls
- ``truncate_args`` and ``truncate_result`` honor their limits
- A custom ``sink`` captures every line (no leakage to ``print``)
- Multi-agent runs (one agent invoking another) render readable boundaries
- ``clone()`` produces an independent copy with the same configuration
- Composition with ``ToolErrorRecovery`` and ``TurnBudget`` does not break
  either capability's behavior
"""

from __future__ import annotations

import json
import re
from typing import Any
from unittest.mock import Mock

import pytest
from agents import RunContextWrapper

from sinan_agentic_core.core.tool_error_recovery import ToolErrorRecovery
from sinan_agentic_core.core.tool_tracer import ToolTracer
from sinan_agentic_core.core.turn_budget import TurnBudget


def _ctx() -> RunContextWrapper[Any]:
    return RunContextWrapper(context=None)


def _tool(name: str) -> Any:
    t = Mock()
    t.name = name
    return t


def _agent(name: str) -> Any:
    a = Mock()
    a.name = name
    return a


# ------------------------------------------------------------------ #
# Hook firing order
# ------------------------------------------------------------------ #


class TestHookOrder:
    def test_two_tool_calls_emit_in_order(self) -> None:
        captured: list[str] = []
        tracer = ToolTracer(sink=captured.append)
        ctx = _ctx()
        agent = _agent("main")

        tracer.on_agent_start(ctx, agent)
        tracer.on_tool_start(ctx, _tool("alpha"), '{"x": 1}')
        tracer.on_tool_end(ctx, _tool("alpha"), '{"ok": true}')
        tracer.on_tool_start(ctx, _tool("beta"), '{"y": 2}')
        tracer.on_tool_end(ctx, _tool("beta"), '{"ok": false}')
        tracer.on_agent_end(ctx, agent, "done")

        assert captured == [
            "[agent start] main",
            '[tool start] alpha args={"x": 1}',
            '[tool end] alpha result={"ok": true}',
            '[tool start] beta args={"y": 2}',
            '[tool end] beta result={"ok": false}',
            "[agent end] main",
        ]

    def test_anonymous_tool_falls_back_to_repr(self) -> None:
        captured: list[str] = []
        tracer = ToolTracer(sink=captured.append)

        bare = object()
        tracer.on_tool_start(_ctx(), bare, "")
        # No "name" attribute — line still renders without raising.
        assert any("[tool start]" in line for line in captured)


# ------------------------------------------------------------------ #
# Truncation
# ------------------------------------------------------------------ #


class TestTruncation:
    def test_truncate_args(self) -> None:
        captured: list[str] = []
        tracer = ToolTracer(sink=captured.append, truncate_args=10)
        long_args = "0123456789ABCDEFGHIJ"  # 20 chars

        tracer.on_tool_start(_ctx(), _tool("t"), long_args)

        assert captured == ["[tool start] t args=0123456789..."]

    def test_truncate_result(self) -> None:
        captured: list[str] = []
        tracer = ToolTracer(sink=captured.append, truncate_result=5)

        tracer.on_tool_end(_ctx(), _tool("t"), "abcdefghij")

        assert captured == ["[tool end] t result=abcde..."]

    def test_zero_disables_truncation(self) -> None:
        captured: list[str] = []
        tracer = ToolTracer(sink=captured.append, truncate_args=0, truncate_result=0)
        long = "x" * 1000

        tracer.on_tool_start(_ctx(), _tool("t"), long)
        tracer.on_tool_end(_ctx(), _tool("t"), long)

        assert captured == [f"[tool start] t args={long}", f"[tool end] t result={long}"]

    def test_short_payload_unchanged(self) -> None:
        captured: list[str] = []
        tracer = ToolTracer(sink=captured.append, truncate_args=200, truncate_result=500)

        tracer.on_tool_start(_ctx(), _tool("t"), "tiny")
        assert "..." not in captured[0]
        assert captured[0].endswith("args=tiny")


# ------------------------------------------------------------------ #
# Custom sink isolation (no leak to print)
# ------------------------------------------------------------------ #


class TestSinkIsolation:
    def test_custom_sink_captures_every_line(self, capsys: pytest.CaptureFixture[str]) -> None:
        captured: list[str] = []
        tracer = ToolTracer(sink=captured.append)
        ctx = _ctx()
        agent = _agent("main")

        tracer.on_agent_start(ctx, agent)
        tracer.on_tool_start(ctx, _tool("t"), '{"k": "v"}')
        tracer.on_tool_end(ctx, _tool("t"), '{"ok": true}')
        tracer.on_agent_end(ctx, agent, "done")

        # Every line landed in the custom sink.
        assert len(captured) == 4
        # Nothing printed to stdout/stderr — the default print sink was
        # replaced and never consulted.
        out = capsys.readouterr()
        assert out.out == ""
        assert out.err == ""

    def test_default_sink_is_print(self, capsys: pytest.CaptureFixture[str]) -> None:
        tracer = ToolTracer()
        tracer.on_tool_start(_ctx(), _tool("t"), "{}")
        out = capsys.readouterr()
        assert "[tool start] t args={}" in out.out


# ------------------------------------------------------------------ #
# Timestamps
# ------------------------------------------------------------------ #


class TestTimestamps:
    def test_include_timestamps_prepends_clock(self) -> None:
        captured: list[str] = []
        tracer = ToolTracer(sink=captured.append, include_timestamps=True)
        tracer.on_tool_start(_ctx(), _tool("t"), "{}")

        assert len(captured) == 1
        # HH:MM:SS.mmm prefix.
        assert re.match(r"^\d{2}:\d{2}:\d{2}\.\d{3} \[tool start\] t", captured[0])

    def test_timestamps_off_by_default(self) -> None:
        captured: list[str] = []
        tracer = ToolTracer(sink=captured.append)
        tracer.on_tool_start(_ctx(), _tool("t"), "{}")

        assert captured[0].startswith("[tool start] t")


# ------------------------------------------------------------------ #
# Multi-agent boundaries (agent-as-tool style)
# ------------------------------------------------------------------ #


class TestMultiAgentBoundaries:
    def test_nested_agent_run_emits_readable_lines(self) -> None:
        """An agent invoking another via ``as_tool`` produces nested
        agent_start/agent_end pairs around the inner tool calls."""
        captured: list[str] = []
        tracer = ToolTracer(sink=captured.append)
        ctx = _ctx()
        outer = _agent("planner")
        inner = _agent("retriever")

        tracer.on_agent_start(ctx, outer)
        tracer.on_tool_start(ctx, _tool("retriever"), '{"q": "topic"}')
        # Inner agent runs inside the as_tool call.
        tracer.on_agent_start(ctx, inner)
        tracer.on_tool_start(ctx, _tool("search"), '{"q": "topic"}')
        tracer.on_tool_end(ctx, _tool("search"), '{"hits": 3}')
        tracer.on_agent_end(ctx, inner, "summary")
        tracer.on_tool_end(ctx, _tool("retriever"), "summary")
        tracer.on_agent_end(ctx, outer, "answer")

        assert captured == [
            "[agent start] planner",
            '[tool start] retriever args={"q": "topic"}',
            "[agent start] retriever",
            '[tool start] search args={"q": "topic"}',
            '[tool end] search result={"hits": 3}',
            "[agent end] retriever",
            "[tool end] retriever result=summary",
            "[agent end] planner",
        ]


# ------------------------------------------------------------------ #
# clone() — independent copy with shared config
# ------------------------------------------------------------------ #


class TestClone:
    def test_clone_carries_config(self) -> None:
        sink: list[str] = []
        original = ToolTracer(
            sink=sink.append,
            truncate_args=42,
            truncate_result=84,
            include_timestamps=True,
        )

        copy = original.clone()

        assert copy is not original
        assert copy.sink is original.sink
        assert copy.truncate_args == 42
        assert copy.truncate_result == 84
        assert copy.include_timestamps is True

    def test_clones_share_no_mutable_state(self) -> None:
        # ToolTracer has no mutable per-run state, but the clone must still
        # be a distinct object so future state additions don't leak.
        original = ToolTracer()
        copy = original.clone()
        assert copy is not original
        assert isinstance(copy, ToolTracer)

    def test_clones_emit_independently(self) -> None:
        a_sink: list[str] = []
        b_sink: list[str] = []
        a = ToolTracer(sink=a_sink.append)
        b = a.clone()
        b.sink = b_sink.append  # rebind sink on the clone

        a.on_tool_start(_ctx(), _tool("alpha"), "{}")
        b.on_tool_start(_ctx(), _tool("beta"), "{}")

        assert a_sink == ["[tool start] alpha args={}"]
        assert b_sink == ["[tool start] beta args={}"]


# ------------------------------------------------------------------ #
# Composition with ToolErrorRecovery / TurnBudget
# ------------------------------------------------------------------ #


class TestComposition:
    def test_error_recovery_still_tracks_when_tracer_attached(self) -> None:
        captured: list[str] = []
        tracer = ToolTracer(sink=captured.append)
        recovery = ToolErrorRecovery()
        ctx = _ctx()
        tool = _tool("failing")

        # Both capabilities receive every hook (the runtime fans them out).
        tracer.on_tool_start(ctx, tool, '{"id": 1}')
        recovery.on_tool_start(ctx, tool, '{"id": 1}')
        tracer.on_tool_end(ctx, tool, json.dumps({"error": "boom"}))
        recovery.on_tool_end(ctx, tool, json.dumps({"error": "boom"}))

        # Recovery did its job.
        assert recovery.has_errors
        assert recovery.get_error_summary()["failing"]["error"] == "boom"
        # Tracer also emitted lines for the same call.
        assert any("[tool start] failing" in line for line in captured)
        assert any("[tool end] failing" in line for line in captured)

    def test_turn_budget_still_counts_when_tracer_attached(self) -> None:
        captured: list[str] = []
        tracer = ToolTracer(sink=captured.append)
        budget = TurnBudget(default_turns=3)
        ctx = _ctx()
        agent = _agent("main")

        tracer.on_agent_start(ctx, agent)
        budget.on_llm_start(ctx, agent, None, [])
        tracer.on_tool_start(ctx, _tool("t"), "{}")
        tracer.on_tool_end(ctx, _tool("t"), "{}")
        budget.on_llm_start(ctx, agent, None, [])
        tracer.on_agent_end(ctx, agent, "done")

        assert budget.turns_used == 2
        assert budget.remaining == 1
        # Tracer captured agent + tool boundaries.
        assert captured[0] == "[agent start] main"
        assert captured[-1] == "[agent end] main"


# ------------------------------------------------------------------ #
# Registry wiring
# ------------------------------------------------------------------ #


class TestRegistryFactory:
    def test_tool_tracer_registered_with_defaults(self) -> None:
        from sinan_agentic_core.registry.capability_registry import (
            get_capability_registry,
        )

        reg = get_capability_registry()
        cap = reg.build("tool_tracer")
        assert isinstance(cap, ToolTracer)
        assert cap.truncate_args == 200
        assert cap.truncate_result == 500
        assert cap.include_timestamps is False

    def test_tool_tracer_accepts_yaml_config(self) -> None:
        from sinan_agentic_core.registry.capability_registry import (
            get_capability_registry,
        )

        reg = get_capability_registry()
        cap = reg.build(
            "tool_tracer",
            {
                "truncate_args": 50,
                "truncate_result": 100,
                "include_timestamps": True,
            },
        )
        assert isinstance(cap, ToolTracer)
        assert cap.truncate_args == 50
        assert cap.truncate_result == 100
        assert cap.include_timestamps is True


# ------------------------------------------------------------------ #
# Top-level re-exports
# ------------------------------------------------------------------ #


class TestPublicAPI:
    def test_top_level_export(self) -> None:
        import sinan_agentic_core as pkg

        assert hasattr(pkg, "ToolTracer")
        assert pkg.ToolTracer is ToolTracer
        assert "ToolTracer" in pkg.__all__

    def test_core_export(self) -> None:
        import sinan_agentic_core.core as core

        assert hasattr(core, "ToolTracer")
        assert core.ToolTracer is ToolTracer
        assert "ToolTracer" in core.__all__
