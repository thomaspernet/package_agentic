# Agents Core

A framework for building AI agents using the OpenAI Agents SDK. Fork this repository to quickly create agent-based applications.

## Features

- **Declarative Agent Definitions** - Define agents as data structures, not code
- **Registry Pattern** - Central registries for agents, tools, and guardrails
- **Agent Factory** - Build Agent instances from registry with one call
- **Chat Service** - Ready-to-use `chat()`, `chat_with_hooks()`, `chat_streamed()` for API endpoints
- **RunHooks** - Track tool calls in real time via `StreamingRunHooks`
- **Session Management** - In-memory and SQLite-backed conversation history
- **Dynamic Context** - Runtime personalization of agent behavior
- **Streaming Events** - Token-level streaming and SSE-compatible event emission
- **Output Models** - Standard `ToolOutput` and `ChatResponse` dataclasses

## Installation

```bash
# Clone and install
git clone https://github.com/thomaspernet/package_agentic.git
cd package_agentic
pip install -e .

# Or install directly from GitHub
pip install git+https://github.com/thomaspernet/package_agentic.git
```

Set your API key:

```bash
export OPENAI_API_KEY="your-key"
```

## Quick Start

Run the example:

```bash
python examples/simple_chat_agent.py
```

---

## Usage

### 1. Define a Tool

```python
from agents import function_tool
from agents_core import register_tool

@register_tool(
    name="get_weather",
    description="Get weather for a city",
    category="api",
    parameters_description="city (str): City name",
    returns_description="Weather data dict",
)
@function_tool
async def get_weather(ctx, city: str) -> dict:
    return {"temperature": 72, "conditions": "sunny"}
```

### 2. Define an Agent

```python
from agents_core import AgentDefinition, register_agent

weather_agent = AgentDefinition(
    name="weather_assistant",
    description="Helps with weather queries",
    instructions="You help users get weather information.",
    tools=["get_weather"],
    model="gpt-4o-mini",
)

register_agent(weather_agent)
```

### 3. Run the Agent

The simplest way is `create_agent_from_registry`, which resolves the definition and its tools into a ready-to-use `Agent`:

```python
from agents import Runner
from agents_core import create_agent_from_registry

agent = create_agent_from_registry("weather_assistant")
result = await Runner.run(agent, "What's the weather in Paris?")
print(result.final_output)
```

---

## Chat Service

For API endpoints, use the chat functions. They handle session history, error handling, and return structured responses.

### Non-streaming chat

```python
from agents_core import chat, AgentSession

session = AgentSession(session_id="user-123")

result = await chat(
    message="What's the weather in Paris?",
    agent_name="weather_assistant",
    session=session,
)

# result = {"success": True, "response": "...", "session_id": "user-123", "tools_called": [...]}
```

### Streaming with tool notifications (RunHooks)

Uses `Runner.run()` with `StreamingRunHooks` to emit events when tools are called. The agent runs to completion, but you get real-time notifications about tool usage. Good for showing progress indicators in a UI.

```python
from agents_core import chat_with_hooks, AgentSession

session = AgentSession(session_id="user-123")

async for event in chat_with_hooks(
    message="What's the weather in Paris?",
    agent_name="weather_assistant",
    session=session,
    tool_friendly_names={"get_weather": "Checking weather"},
):
    # event = {"event": "thinking"|"tool_start"|"tool_end"|"answer"|"error", "data": {...}}
    if event["event"] == "tool_start":
        print(f"Tool: {event['data']['friendly_name']}")
    elif event["event"] == "answer":
        print(event["data"]["response"])
```

### Token-level streaming

Uses `Runner.run_streamed()` to stream text deltas as they are generated. Best for displaying the response character-by-character in a chat UI.

```python
from agents_core import chat_streamed, AgentSession

session = AgentSession(session_id="user-123")

async for event in chat_streamed(
    message="What's the weather in Paris?",
    agent_name="weather_assistant",
    session=session,
):
    # Events: text_delta, tool_call, tool_output, agent_updated, message_output, answer, error
    if event["event"] == "text_delta":
        print(event["data"]["delta"], end="", flush=True)
    elif event["event"] == "tool_call":
        print(f"\n[Calling {event['data']['tool']}...]")
    elif event["event"] == "answer":
        print(f"\n\nDone. Tools used: {event['data']['tools_called']}")
```

---

## Session Persistence

### In-memory (default)

```python
from agents_core import AgentSession

session = AgentSession(session_id="user-123")

await session.add_items([{"role": "user", "content": "Hello!"}])
history = await session.get_items()  # OpenAI-compatible message list
```

### SQLite (persistent)

Store conversations across restarts. Messages are saved with timestamps, metadata, and session archiving.

```python
from agents_core import SQLiteSessionStore

store = SQLiteSessionStore("data/conversations.db")

# Messages
store.add_message("session-123", "user", "What's the weather?")
store.add_message("session-123", "assistant", "It's sunny in Paris.")

# Retrieve in OpenAI format
history = store.get_conversation_history("session-123")
# [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

# Session management
sessions = store.get_active_sessions()    # List active sessions
store.archive_session("session-123")      # Archive (keeps data)
store.clear_session("session-123")        # Delete permanently
```

---

## Output Models

Standard dataclasses for consistent return types across your tools and API.

### ToolOutput

```python
from agents_core import ToolOutput

# Return from your tool functions
output = ToolOutput(
    success=True,
    data={"temperature": 72, "conditions": "sunny"},
    metadata={"source": "weather_api"},
)

output.to_dict()
# {"success": True, "data": {...}, "metadata": {...}}
```

### ChatResponse

```python
from agents_core import ChatResponse

response = ChatResponse(
    success=True,
    response="It's 72F and sunny in Paris.",
    session_id="user-123",
    tools_called=["get_weather"],
)

response.to_dict()
# {"success": True, "response": "...", "session_id": "...", "tools_called": [...]}
```

---

## Dynamic Context

Customize agent behavior at runtime by passing context data.

### Define a Context Type

```python
from dataclasses import dataclass

@dataclass
class UserContext:
    user_id: str
    user_name: str
    role: str = "user"
    language: str = "English"
```

### Create Dynamic Instructions

```python
from agents import Agent, RunContextWrapper

def dynamic_instructions(ctx: RunContextWrapper[UserContext], agent: Agent) -> str:
    c = ctx.context
    return f"""You are a helpful assistant for {c.user_name}.
Respond in {c.language}. User role: {c.role}."""

agent = Agent[UserContext](
    name="personalized_assistant",
    instructions=dynamic_instructions,
    model="gpt-4o-mini",
)
```

### Run with Context

```python
from agents import Runner

context = UserContext(
    user_id="123",
    user_name="Thomas",
    role="developer",
)

result = await Runner.run(
    agent,
    input="How should I structure my code?",
    context=context,
)
```

See `examples/dynamic_context_agent.py` for a complete example.

---

## Project Structure

```
package_agentic/
├── agents_core/
│   ├── __init__.py              # Main exports
│   ├── orchestrator.py          # Multi-agent orchestration
│   ├── registry/
│   │   ├── agent_registry.py    # AgentDefinition + registry
│   │   ├── agent_factory.py     # create_agent_from_registry()
│   │   ├── tool_registry.py     # ToolDefinition + registry
│   │   └── guardrail_registry.py
│   ├── services/
│   │   ├── chat_service.py      # chat(), chat_with_hooks(), chat_streamed()
│   │   ├── run_hooks.py         # StreamingRunHooks (tool call tracking)
│   │   └── streaming_helper.py  # Event emission helpers
│   ├── session/
│   │   ├── agent_session.py     # In-memory session
│   │   └── sqlite_store.py      # SQLite persistence
│   ├── models/
│   │   ├── context.py           # AgentContext
│   │   └── outputs/             # ToolOutput, ChatResponse
│   ├── agents/                  # Your agent definitions
│   ├── tools/                   # Your tool implementations
│   └── guardrails/              # Input validation
├── examples/
│   ├── simple_chat_agent.py
│   └── dynamic_context_agent.py
├── pyproject.toml
└── README.md
```

---

## Core Components

### AgentDefinition

```python
AgentDefinition(
    name="my_agent",
    description="What this agent does",
    instructions="System prompt...",      # String or function
    tools=["tool1", "tool2"],             # Tool names from registry
    guardrails=["guardrail1"],            # Optional validators
    model="gpt-4o-mini",
    output_dataclass=MyOutput,            # Optional structured output
)
```

### AgentSession

```python
from agents_core import AgentSession

session = AgentSession(session_id="user-123")

await session.add_items([
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
])

history = await session.get_items()
```

### AgentContext

```python
from agents_core import AgentContext

context = AgentContext(
    database_connector=your_db,
    schema="Your schema...",
    filters={"user_id": "123"},
)
```

---

## Multi-Agent Orchestration

```python
from agents_core import AgentOrchestrator

class MyOrchestrator(AgentOrchestrator):
    async def execute(self, user_query: str, context_data: dict):
        session = self.setup_session()
        context = self.setup_context(**context_data)
        
        intent = await self.run_agent("intent_analyzer", session, context)
        
        if intent.category == "weather":
            result = await self.run_agent("weather_agent", session, context)
        else:
            result = await self.run_agent("general_agent", session, context)
        
        return {"success": True, "result": result}
```

---

## Development

```bash
pip install -e ".[dev]"
pytest
black agents_core/
ruff check agents_core/
```

---

## License

MIT License
