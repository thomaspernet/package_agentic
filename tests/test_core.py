"""Tests for BaseAgentRunner (core/base_runner.py)."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from agents import Usage

from agents_core.models.context import AgentContext
from agents_core.registry.agent_registry import AgentDefinition, AgentRegistry
from agents_core.registry.guardrail_registry import GuardrailDefinition, GuardrailRegistry
from agents_core.registry.tool_registry import ToolDefinition, ToolRegistry
from agents_core.session.agent_session import AgentSession


@pytest.fixture
def _registries():
    """Build isolated registries with sample data."""
    agent_reg = AgentRegistry()
    tool_reg = ToolRegistry()
    guardrail_reg = GuardrailRegistry()

    tool_fn = lambda: "result"
    tool_reg.register(ToolDefinition("test_tool", "desc", tool_fn, "cat", "p", "r"))

    guardrail_fn = lambda: True
    guardrail_reg.register(GuardrailDefinition("test_guard", "desc", guardrail_fn, "output"))

    agent_reg.register(
        AgentDefinition(
            name="basic_agent",
            description="basic",
            instructions="You are a basic agent",
            tools=["test_tool"],
            guardrails=["test_guard"],
        )
    )

    return agent_reg, tool_reg, guardrail_reg


@pytest.fixture
def runner(_registries):
    """Instantiate BaseAgentRunner with patched registries."""
    agent_reg, tool_reg, guardrail_reg = _registries

    with (
        patch("agents_core.core.base_runner.get_agent_registry", return_value=agent_reg),
        patch("agents_core.core.base_runner.get_tool_registry", return_value=tool_reg),
        patch("agents_core.core.base_runner.get_guardrail_registry", return_value=guardrail_reg),
    ):
        from agents_core.core.base_runner import BaseAgentRunner

        return BaseAgentRunner()


class TestBaseAgentRunnerInit:
    def test_loads_tool_map(self, runner):
        assert "test_tool" in runner.tool_map

    def test_loads_guardrail_map(self, runner):
        assert "test_guard" in runner.guardrail_map


class TestSetupHelpers:
    def test_setup_context(self, runner):
        ctx = runner.setup_context(database_connector=Mock())
        assert isinstance(ctx, AgentContext)
        assert ctx.has_data is False

    def test_setup_session_with_id(self, runner):
        session = runner.setup_session(session_id="my-id")
        assert session.session_id == "my-id"

    def test_setup_session_generates_uuid(self, runner):
        session = runner.setup_session()
        assert len(session.session_id) > 0

    def test_setup_session_with_history(self, runner):
        history = [{"role": "user", "content": "hello"}]
        session = runner.setup_session(session_id="h1", initial_history=history)
        assert session.session_id == "h1"


class TestAggregateUsage:
    def test_single_response(self, runner, mock_run_result):
        usage = runner._aggregate_usage(mock_run_result)
        assert usage["requests"] == 1
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 50
        assert usage["total_tokens"] == 150

    def test_empty_responses(self, runner):
        result = Mock()
        result.raw_responses = []
        usage = runner._aggregate_usage(result)
        assert usage["total_tokens"] == 0


class TestCreateAgent:
    async def test_basic_agent(self, runner):
        ctx = AgentContext(database_connector=Mock())
        agent = await runner.create_agent("basic_agent", ctx)
        assert agent.name == "basic_agent"

    async def test_not_found_raises(self, runner):
        ctx = AgentContext(database_connector=Mock())
        with pytest.raises(ValueError, match="not found"):
            await runner.create_agent("nonexistent", ctx)

    async def test_callable_instructions(self, runner):
        runner.agent_registry.register(
            AgentDefinition(
                name="dynamic_agent",
                description="dynamic",
                instructions=lambda ctx, agent: "dynamic instructions",
            )
        )
        ctx = AgentContext(database_connector=Mock())
        agent = await runner.create_agent("dynamic_agent", ctx)
        assert agent.name == "dynamic_agent"

    async def test_output_dataclass_type(self, runner):
        from pydantic import BaseModel

        class MyOutput(BaseModel):
            answer: str

        runner.agent_registry.register(
            AgentDefinition(
                name="typed_agent",
                description="typed",
                instructions="test",
                output_dataclass=MyOutput,
            )
        )
        ctx = AgentContext(database_connector=Mock())
        agent = await runner.create_agent("typed_agent", ctx)
        assert agent.name == "typed_agent"

    async def test_output_dataclass_string(self, runner):
        runner.agent_registry.register(
            AgentDefinition(
                name="str_typed_agent",
                description="typed by name",
                instructions="test",
                output_dataclass="ChatResponse",
            )
        )
        ctx = AgentContext(database_connector=Mock())
        agent = await runner.create_agent("str_typed_agent", ctx)
        assert agent.name == "str_typed_agent"

    async def test_handoffs(self, runner):
        runner.agent_registry.register(
            AgentDefinition(name="target_agent", description="target", instructions="target")
        )
        runner.agent_registry.register(
            AgentDefinition(
                name="source_agent",
                description="source",
                instructions="source",
                handoffs=["target_agent"],
            )
        )
        ctx = AgentContext(database_connector=Mock())
        agent = await runner.create_agent("source_agent", ctx)
        assert agent.name == "source_agent"

    async def test_agent_as_tool(self, runner):
        runner.agent_registry.register(
            AgentDefinition(name="sub_agent", description="sub desc", instructions="sub")
        )
        runner.agent_registry.register(
            AgentDefinition(
                name="parent_agent",
                description="parent",
                instructions="parent",
                tools=["sub_agent"],
            )
        )
        ctx = AgentContext(database_connector=Mock())
        agent = await runner.create_agent("parent_agent", ctx)
        assert agent.name == "parent_agent"

    async def test_model_settings_fn(self, runner):
        from agents import ModelSettings

        runner.agent_registry.register(
            AgentDefinition(
                name="settings_agent",
                description="with settings",
                instructions="test",
                model_settings_fn=lambda ctx: ModelSettings(temperature=0.5),
            )
        )
        ctx = AgentContext(database_connector=Mock())
        agent = await runner.create_agent("settings_agent", ctx)
        assert agent.name == "settings_agent"


class TestRunAgent:
    async def test_returns_output_and_usage(self, runner, mock_run_result):
        ctx = AgentContext(database_connector=Mock())
        session = AgentSession(session_id="run-test")

        with (
            patch.object(runner, "create_agent", new_callable=AsyncMock, return_value=Mock()),
            patch("agents_core.core.base_runner.Runner") as mock_runner_cls,
        ):
            mock_runner_cls.run = AsyncMock(return_value=mock_run_result)
            result = await runner.run_agent("basic_agent", session, ctx, "hello")

        assert result["output"] == "Test response"
        assert result["usage"]["input_tokens"] == 100
