# Agents Core

A framework for building AI agents using the OpenAI Agents SDK. Fork this repository to quickly create agent-based applications.

## Features

- **Declarative Agent Definitions** - Define agents as data structures, not code
- **Registry Pattern** - Central registries for agents, tools, and guardrails
- **Session Management** - Built-in conversation history compatible with the SDK
- **Dynamic Context** - Runtime personalization of agent behavior
- **Streaming Events** - Real-time event emission for SSE/WebSocket

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

```python
from agents import Agent, Runner
from agents_core import get_agent_registry, get_tool_registry

# Get from registries
agent_def = get_agent_registry().get("weather_assistant")
tools = get_tool_registry().get_tool_functions(agent_def.tools)

# Create and run
agent = Agent(
    name=agent_def.name,
    instructions=agent_def.instructions,
    model=agent_def.model,
    tools=tools,
)

result = await Runner.run(agent, "What's the weather in Paris?")
print(result.final_output)
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
    instructions=dynamic_instructions,  # Pass function, not string
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
│   ├── __init__.py          # Main exports
│   ├── orchestrator.py      # Multi-agent orchestration
│   ├── core/                # Base runner
│   ├── session/             # Conversation history
│   ├── models/              # Context and output models
│   ├── registry/            # Agent/Tool/Guardrail registries
│   ├── services/            # Streaming helpers
│   ├── agents/              # Your agent definitions
│   ├── tools/               # Your tool implementations
│   └── guardrails/          # Input validation
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

### StreamingHelper

```python
from agents_core import StreamingHelper

streaming = StreamingHelper(event_callback=your_callback)
streaming.emit_agent_start("analyzer", iteration=1)
streaming.emit_answer("The result is...", sources=[])
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
