"""Agents Core - Generic agent orchestration framework.

This package provides a reusable framework for building multi-agent systems
using the OpenAI Agents SDK with code orchestration.

Core Components:
- AgentSession: Manages conversation history
- AgentContext: Stores workflow state and data
- AgentRegistry: Declarative agent definitions
- ToolRegistry: Declarative tool definitions
- AgentOrchestrator: Workflow orchestration logic

Quick Start:
    from agents_core import (
        AgentSession,
        AgentContext,
        AgentOrchestrator,
        register_agent,
        register_tool,
        AgentDefinition
    )
    
    # Define your agent
    my_agent = AgentDefinition(
        name="my_agent",
        description="Does something",
        instructions="You are a helpful assistant...",
        tools=["my_tool"]
    )
    register_agent(my_agent)
    
    # Run orchestrator
    orchestrator = AgentOrchestrator()
    result = await orchestrator.run(
        user_query="Do something",
        context_data={"database_connector": db}
    )
"""

from .session import AgentSession, ConversationHistory
from .models.context import AgentContext
from .registry import (
    AgentDefinition,
    AgentRegistry,
    get_agent_registry,
    register_agent,
    ToolDefinition,
    ToolRegistry,
    get_tool_registry,
    register_tool,
)

from .orchestrator import AgentOrchestrator

__all__ = [
    "AgentSession",
    "ConversationHistory",
    "AgentContext",
    "AgentDefinition",
    "AgentRegistry",
    "get_agent_registry",
    "register_agent",
    "ToolDefinition",
    "ToolRegistry",
    "get_tool_registry",
    "register_tool",
    "AgentOrchestrator",
]
