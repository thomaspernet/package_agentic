"""Tool tracer — observation-only Capability for non-streaming runs.

Prints a single line for every tool call (name, arguments, return value) and
agent boundary during a run. Mirrors :class:`ToolErrorRecovery` and
:class:`TurnBudget` in shape so it can be attached the same way::

    tracer = ToolTracer(sink=print)
    agent_def = AgentDefinition(..., capabilities=[tracer])

Designed for non-streaming runs that have no ``on_event`` channel. The
streaming path already emits ``tool_call`` / ``tool_output`` events, so
consumers running ``streaming=True`` should keep using ``on_event``.

The tracer never mutates state used by the agent — it only emits text to
``sink``. That makes it safe to compose with other capabilities like
:class:`ToolErrorRecovery`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from agents import RunContextWrapper, Tool

from .capabilities import Capability


class ToolTracer(Capability):
    """Print every tool call and agent boundary during a run.

    Each tool start/end and agent start/end is rendered as a single line
    handed to ``sink`` (default :func:`print`). ``args`` and ``result`` are
    truncated independently so noisy payloads do not flood the trace.

    Attributes:
        sink: One-arg callable that receives each line. Defaults to
            :func:`print`. Pass a logger method (``logger.info``), a list's
            ``append``, or any other ``Callable[[str], None]`` to redirect
            output.
        truncate_args: Maximum length of the arguments string before
            truncation with ``...``. ``0`` disables truncation.
        truncate_result: Maximum length of the result string before
            truncation with ``...``. ``0`` disables truncation.
        include_timestamps: When ``True``, every emitted line is prefixed
            with a wall-clock ``HH:MM:SS.mmm`` timestamp.
    """

    def __init__(
        self,
        sink: Callable[[str], None] = print,
        truncate_args: int = 200,
        truncate_result: int = 500,
        include_timestamps: bool = False,
    ) -> None:
        self.sink = sink
        self.truncate_args = truncate_args
        self.truncate_result = truncate_result
        self.include_timestamps = include_timestamps

    # ------------------------------------------------------------------ #
    # Capability hooks
    # ------------------------------------------------------------------ #

    def on_agent_start(
        self,
        ctx: RunContextWrapper[Any],
        agent: Any,
    ) -> None:
        name = self._agent_name(agent)
        self._emit(f"[agent start] {name}")

    def on_agent_end(
        self,
        ctx: RunContextWrapper[Any],
        agent: Any,
        output: Any,
    ) -> None:
        name = self._agent_name(agent)
        self._emit(f"[agent end] {name}")

    def on_tool_start(
        self,
        ctx: RunContextWrapper[Any],
        tool: Tool,
        args: str,
    ) -> None:
        name = self._tool_name(tool)
        rendered = self._truncate(args, self.truncate_args)
        self._emit(f"[tool start] {name} args={rendered}")

    def on_tool_end(
        self,
        ctx: RunContextWrapper[Any],
        tool: Tool,
        result: str,
    ) -> None:
        name = self._tool_name(tool)
        rendered = self._truncate(result, self.truncate_result)
        self._emit(f"[tool end] {name} result={rendered}")

    # ------------------------------------------------------------------ #
    # Cloning
    # ------------------------------------------------------------------ #

    def clone(self) -> ToolTracer:
        """Return a fresh tracer carrying the same configuration."""
        return ToolTracer(
            sink=self.sink,
            truncate_args=self.truncate_args,
            truncate_result=self.truncate_result,
            include_timestamps=self.include_timestamps,
        )

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _emit(self, line: str) -> None:
        if self.include_timestamps:
            line = f"{self._timestamp()} {line}"
        self.sink(line)

    @staticmethod
    def _timestamp() -> str:
        now = time.time()
        local = time.localtime(now)
        millis = int((now - int(now)) * 1000)
        return f"{time.strftime('%H:%M:%S', local)}.{millis:03d}"

    @staticmethod
    def _truncate(value: str, limit: int) -> str:
        text = value if isinstance(value, str) else str(value)
        if limit <= 0 or len(text) <= limit:
            return text
        return text[:limit] + "..."

    @staticmethod
    def _tool_name(tool: Any) -> str:
        return str(getattr(tool, "name", None) or tool)

    @staticmethod
    def _agent_name(agent: Any) -> str:
        return str(getattr(agent, "name", None) or agent)
