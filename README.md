# 🤖 Agents Core - AI Agent Framework

A clean, state-of-the-art framework for building AI agents using the **OpenAI Agents SDK**.

**Fork this repository** to quickly spin up agent-based applications.

## ✨ Features

- **Declarative Agent Definitions** - Define agents as data, not code
- **Tool Registry** - Register tools with `@register_tool` decorator  
- **Session Management** - Built-in conversation history (SDK-compatible)
- **Streaming Events** - Real-time events for SSE/WebSocket
- **Easy to Fork** - Clean structure, minimal dependencies

## 🚀 Quick Start

### 1. Install

```bash
# Clone and install
git clone https://github.com/thomaspernet/package_agentic.git
cd package_agentic
pip install -e .

# Or install dependencies directly
pip install openai-agents pydantic
```

### 2. Set API Key

```bash
export OPENAI_API_KEY="your-key"
```

### 3. Run Example

```bash
python examples/simple_chat_agent.py
```

---

## 📖 Usage

### Define a Tool

```python
from agents import function_tool
from agents_core import register_tool

@register_tool(
    name="get_weather",
    description="Get weather for a city",
    category="api",
)
@function_tool
async def get_weather(ctx, city: str) -> dict:
    # Your implementation
    return {"temperature": 72, "conditions": "sunny"}
```

### Define an Agent

```python
from agents_core import AgentDefinition, register_agent

weather_agent = AgentDefinition(
    name="weather_assistant",
    description="Helps with weather queries",
    instructions="You help users get weather information. Use the get_weather tool.",
    tools=["get_weather"],
    model="gpt-4o-mini",
)

register_agent(weather_agent)
```

### Run the Agent

```python
from agents import Agent, Runner
from agents_core import get_agent_registry, get_tool_registry, AgentSession

# Get from registries
agent_def = get_agent_registry().get("weather_assistant")
tools = get_tool_registry().get_tool_functions(agent_def.tools)

# Create SDK agent
agent = Agent(
    name=agent_def.name,
    instructions=agent_def.instructions,
    model=agent_def.model,
    tools=tools,
)

# Run
result = await Runner.run(agent, "What's the weather in Paris?")
print(result.final_output)
```

---

## 📁 Project Structure

```
package_agentic/
├── agents_core/           # 🎯 Main package
│   ├── __init__.py       # Clean exports
│   ├── orchestrator.py   # Multi-agent orchestration
│   ├── core/             # Base runner
│   ├── session/          # Conversation history
│   ├── models/           # Context & outputs
│   ├── registry/         # Agent/Tool/Guardrail registries
│   ├── services/         # Streaming & helpers
│   ├── agents/           # Your agent definitions (ADD HERE)
│   ├── tools/            # Your tool implementations (ADD HERE)
│   └── guardrails/       # Input validation (ADD HERE)
├── examples/             # Ready-to-run examples
├── pyproject.toml        # pip install -e .
└── README.md
```

---

## 🔧 Core Components

### AgentDefinition

Define agents declaratively:

```python
AgentDefinition(
    name="my_agent",
    description="What this agent does",
    instructions="System prompt...",  # Or a function for dynamic prompts
    tools=["tool1", "tool2"],         # Tool names from registry
    guardrails=["guardrail1"],        # Optional input validators
    model="gpt-4o-mini",              # Or gpt-4o, gpt-4-turbo, etc.
    output_dataclass=MyOutput,        # Optional structured output
)
```

### AgentSession

Manage conversation history:

```python
from agents_core import AgentSession

session = AgentSession(session_id="user-123")

# Add messages
await session.add_items([
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
])

# Get history for SDK
history = await session.get_items()

# Check if needs summarization
if session.needs_summarization():
    # Trigger your summarization logic
    pass
```

### AgentContext

Share state across agents:

```python
from agents_core import AgentContext

context = AgentContext(
    database_connector=your_db,  # Any database
    schema="Your schema...",
    filters={"user_id": "123"},
)

# Access in tools via ctx.context
```

### StreamingHelper

Emit real-time events for SSE/WebSocket:

```python
from agents_core import StreamingHelper

streaming = StreamingHelper(event_callback=your_sse_callback)

streaming.emit_agent_start("analyzer", iteration=1)
streaming.emit_answer("The weather is sunny!", sources=[])
streaming.emit_error("Something went wrong")
```

---

## 🔀 Multi-Agent Orchestration

For complex workflows with multiple agents:

```python
from agents_core import AgentOrchestrator

class MyOrchestrator(AgentOrchestrator):
    async def execute(self, user_query: str, context_data: dict):
        session = self.setup_session()
        context = self.setup_context(**context_data)
        
        # Step 1: Analyze intent
        intent = await self.run_agent("intent_analyzer", session, context)
        
        # Step 2: Route to specialist
        if intent.category == "weather":
            result = await self.run_agent("weather_agent", session, context)
        else:
            result = await self.run_agent("general_agent", session, context)
        
        return {"success": True, "result": result}
```

---

## 🎯 Best Practices

1. **One tool = One function** - Keep tools focused and single-purpose
2. **Use registries** - Don't hardcode agents/tools, use the registry pattern
3. **Session per conversation** - Store session_id for continuity
4. **Context for shared state** - Pass data between agents via context
5. **Extend, don't modify** - Subclass `AgentContext` for your domain

---

## 📦 Installation Options

```bash
# Development install (editable)
pip install -e ".[dev]"

# From PyPI (when published)
pip install agents-core

# From GitHub
pip install git+https://github.com/thomaspernet/package_agentic.git
```

---

## 🛠️ Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black agents_core/
ruff check agents_core/
```

---

## 📄 License

MIT License - Fork and build amazing things!

---

**Built with ❤️ using [OpenAI Agents SDK](https://github.com/openai/openai-agents-python)**

---

## 📚 Extended Documentation

Below is detailed documentation for advanced usage patterns.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ YOUR APPLICATION LAYER                                       │
│ - API endpoints                                              │
│ - Business logic (services)                                  │
│ - Domain models                                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ ORCHESTRATION LAYER (This Package)                          │
│ ┌──────────────────────┐  ┌──────────────────────────────┐  │
│ │ agents_core          │  │ workflows_core               │  │
│ │ - AgentOrchestrator  │  │ - Prefect Flows              │  │
│ │ - AgentSession       │  │ - Prefect Tasks              │  │
│ │ - AgentContext       │  │ - Orchestration patterns     │  │
│ │ - Registry system    │  │                              │  │
│ └──────────────────────┘  └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ INFRASTRUCTURE LAYER                                         │
│ - Database connectors                                        │
│ - LLM providers                                              │
│ - External APIs                                              │
└─────────────────────────────────────────────────────────────┘
```

**Workflow**: Your API → AgentOrchestrator/Prefect Flows → Your Services → Infrastructure

---

## Agents Core

Multi-agent orchestration system using the **OpenAI Agents SDK** with a **code orchestration pattern**.

### Agents Core Folder Structure

```
agents_core/
├── __init__.py                    # Main exports
├── config.py                      # Configuration dataclasses
├── orchestrator.py                # AgentOrchestrator - main workflow controller
│
├── session/                       # Conversation history management
│   ├── agent_session.py          # AgentSession (implements SessionABC from SDK)
│   └── __init__.py
│
├── models/                        # Data models
│   ├── context.py                # AgentContext - shared state across agents
│   ├── outputs/                  # Agent output dataclasses
│   └── __init__.py
│
├── registry/                      # Declarative registration system
│   ├── agent_registry.py         # AgentDefinition, register_agent()
│   ├── tool_registry.py          # ToolDefinition, register_tool()
│   └── __init__.py
│
├── agents/                        # Your agent definitions (ADD YOUR AGENTS HERE)
│   ├── orchestration/            # Orchestration agents (routing, planning, synthesis)
│   └── __init__.py
│
├── tools/                         # Your tool implementations (ADD YOUR TOOLS HERE)
│   └── __init__.py
│
├── guardrails/                    # Input validation (ADD YOUR GUARDRAILS HERE)
│   └── __init__.py
│
├── services/                      # Helper services (planning, routing, streaming)
│   └── __init__.py
│
├── utils/                         # Utility functions
│   └── __init__.py
│
└── instructions/                  # Reusable instruction templates
    └── shared/
        └── __init__.py
```

### Agents Core Components

#### 1. **`session/` - Conversation History Management**

**Purpose**: Manages conversation history for multi-agent workflows, compatible with OpenAI Agents SDK.

**Key Files**:
- `agent_session.py`: `AgentSession` class implementing `SessionABC`

**What it does**:
- Stores conversation messages (user, assistant, system)
- Integrates with SDK's Runner
- Handles structured output cleanup
- Supports session metadata
- Triggers summarization when history grows too long

**When to use**:
- Every agent workflow needs a session
- Pass to `Runner(session=session, ...)`
- Store session_id for conversation continuity

**Example**:
```python
from agents_core import AgentSession

session = AgentSession(session_id="unique-id", max_items=50)

# Add messages
await session.add_items([
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"}
])

# Get history
history = await session.get_items()

# Check if needs summarization
if session.needs_summarization():
    # Trigger your summarization logic
    pass
```

---

#### 2. **`models/context.py` - AgentContext**

**Purpose**: Stores shared state and data that flows through your agent workflow.

**What it contains**:
- Database connector
- Schema information
- Query results
- User filters/parameters
- Discovered data during workflow

**When to use**:
- Initialize at workflow start
- Pass to `Runner(context=context, ...)`
- Agents access via `ctx.context` in tools
- Accumulate results as agents execute

**Example**:
```python
from agents_core import AgentContext

context = AgentContext(
    database_connector=your_db,
    schema="Your schema definition...",
    filters={"user_id": "123"}
)

# Add data discovered during workflow
context.add_discovered_item("users", user_data)

# Check if data was collected
if context.has_data:
    results = context.query_results
```

**Extend for your use case**:
```python
from dataclasses import dataclass
from agents_core import AgentContext

@dataclass
class MyAppContext(AgentContext):
    workspace_id: str = ""
    user_preferences: dict = field(default_factory=dict)
    custom_metadata: dict = field(default_factory=dict)
```

---

#### 3. **`registry/` - Declarative Registration System**

**Purpose**: Define agents and tools as data structures (not code), enabling discovery and dynamic composition.

**Key Files**:
- `agent_registry.py`: `AgentDefinition`, `register_agent()`
- `tool_registry.py`: `ToolDefinition`, `register_tool()`

**Agent Registry**:

Define agents declaratively:

```python
from agents_core import AgentDefinition, register_agent

my_agent = AgentDefinition(
    name="data_analyzer",
    description="Analyzes database schema and generates queries",
    instructions="You are a data analyst. Analyze schemas and write queries...",
    tools=["execute_query", "inspect_schema"],  # Tool names
    guardrails=["validate_query"],              # Guardrail names
    model="gpt-4o-mini",
    output_dataclass=MyOutputClass  # For structured output
)

register_agent(my_agent)
```

**Dynamic Instructions** (inject schema, filters, etc.):

```python
def dynamic_instructions(context: AgentContext, agent_def: AgentDefinition) -> str:
    return f"""
You are a data analyst.

Database Schema:
{context.schema}

User Filters: {context.filters}

Analyze and generate queries.
"""

my_agent = AgentDefinition(
    name="analyzer",
    description="...",
    instructions=dynamic_instructions,  # Function, not string!
    tools=["execute_query"],
    requires_schema_injection=True
)
```

**Tool Registry**:

Register tools with metadata:

```python
from agents_core.registry import register_tool
from agents import function_tool

@register_tool(
    name="execute_query",
    description="Execute database query",
    category="database",
    parameters_description="query (str): SQL/Cypher query to execute",
    returns_description="Dict with success status and results"
)
@function_tool
async def execute_query(ctx, query: str) -> dict:
    # ctx.context = your AgentContext
    # ctx.session = your AgentSession
    db = ctx.context.database_connector
    result = await db.execute(query)
    return {"success": True, "data": result}
```

---

#### 4. **`orchestrator.py` - AgentOrchestrator**

**Purpose**: Main workflow controller that runs agents in sequence or based on routing logic.

**Pattern**: Code orchestration (NOT SDK handoffs)
- You control the workflow logic
- Call agents explicitly in code
- Accumulate results in context
- Route based on outputs

**Example**:
```python
from agents_core import AgentOrchestrator

orchestrator = AgentOrchestrator()

result = await orchestrator.run(
    user_query="Analyze sales data for Q4",
    context_data={
        "database_connector": db,
        "schema": schema_text,
        "filters": {"year": 2024}
    }
)
```

**Extend for your workflow**:
```python
class MyOrchestrator(AgentOrchestrator):
    async def run(self, user_query: str, context_data: dict):
        session = AgentSession(session_id=str(uuid.uuid4()))
        context = MyAppContext(**context_data)
        
        # Your workflow logic
        # Step 1: Analyze intent
        intent = await self._run_agent("intent_analyzer", session, context)
        
        # Step 2: Route based on intent
        if intent.category == "query":
            result = await self._run_agent("query_generator", session, context)
        elif intent.category == "analysis":
            result = await self._run_agent("data_analyst", session, context)
        
        # Step 3: Synthesize response
        response = await self._run_agent("synthesizer", session, context)
        
        return {"success": True, "response": response}
```

---

#### 5. **`agents/` - Agent Definitions**

**Purpose**: Where you define your specific agents.

**Structure**:
- `/agents/orchestration/`: Routing, planning, synthesis agents
- `/agents/domain_specific/`: Your domain agents (add subfolders as needed)

**Pattern**:
```python
# agents/my_agents.py
from agents_core import AgentDefinition, register_agent

analyzer_agent = AgentDefinition(
    name="analyzer",
    description="Analyzes data patterns",
    instructions="You analyze data and identify patterns...",
    tools=["fetch_data", "statistical_analysis"],
    model="gpt-4o"
)
register_agent(analyzer_agent)

query_agent = AgentDefinition(
    name="query_generator",
    description="Generates database queries",
    instructions="You write optimized database queries...",
    tools=["execute_query", "validate_query"],
    model="gpt-4o-mini"
)
register_agent(query_agent)
```

Import in `agents/__init__.py` to auto-register on startup.

---

#### 6. **`tools/` - Tool Implementations**

**Purpose**: Functions that agents can call (database queries, API calls, computations, etc.).

**Pattern**:
```python
# tools/database_tools.py
from agents import function_tool
from agents_core.registry import register_tool

@register_tool(
    name="execute_query",
    description="Execute database query",
    category="database",
    parameters_description="query (str): Query to execute",
    returns_description="Query results"
)
@function_tool
async def execute_query(ctx, query: str) -> dict:
    db = ctx.context.database_connector
    results = await db.execute(query)
    return {"success": True, "data": results}

@register_tool(
    name="fetch_user_data",
    description="Fetch user profile data",
    category="data",
    parameters_description="user_id (str): User ID",
    returns_description="User profile dict"
)
@function_tool
async def fetch_user_data(ctx, user_id: str) -> dict:
    # Access context and session
    db = ctx.context.database_connector
    filters = ctx.context.filters
    
    user = await db.get_user(user_id, filters=filters)
    return user
```

**Key Rules**:
- Use `@function_tool` decorator from SDK
- Use `@register_tool` decorator for metadata
- Access context via `ctx.context`
- Access session via `ctx.session`
- Return structured dicts
- Tools are automatically executed by SDK when agents call them

---

#### 7. **`guardrails/` - Input Validation**

**Purpose**: Validate agent inputs before execution (prevent malformed queries, injection attacks, etc.).

**Pattern**:
```python
# guardrails/query_guardrails.py

def validate_query(ctx, query: str) -> tuple[bool, str]:
    """Validate SQL/Cypher query.
    
    Returns:
        (is_valid, error_message)
    """
    # Check for dangerous patterns
    if "DROP" in query.upper() or "DELETE" in query.upper():
        return False, "Destructive queries not allowed"
    
    # Check syntax (use your parser)
    if not is_valid_syntax(query):
        return False, "Invalid query syntax"
    
    return True, ""

def validate_user_input(ctx, user_input: str) -> tuple[bool, str]:
    """Validate user input."""
    if len(user_input) > 10000:
        return False, "Input too long"
    
    if contains_malicious_content(user_input):
        return False, "Malicious content detected"
    
    return True, ""
```

Register in orchestrator:
```python
orchestrator.register_guardrail("validate_query", validate_query)
```

---

#### 8. **`services/` - Helper Services**

**Purpose**: Reusable business logic that supports orchestration (planning, routing, streaming, etc.).

**Common Services**:
- **Planning Service**: Breaks down complex queries into steps
- **Routing Helper**: Determines which agent to run next
- **Streaming Helper**: Manages SSE streaming for real-time responses
- **Safety Guard**: Prevents infinite loops, rate limiting

**Example**:
```python
# services/routing_helper.py

class RoutingHelper:
    """Determines which agent to run based on context."""
    
    def get_next_agent(self, context: AgentContext, last_output: Any) -> str:
        """Route to next agent based on workflow state."""
        # Your routing logic
        if context.query_results:
            return "synthesizer"
        elif context.filters:
            return "query_generator"
        else:
            return "intent_analyzer"
```

---

#### 9. **`utils/` - Utility Functions**

**Purpose**: Helper functions for message formatting, context manipulation, output parsing, etc.

**Common Utils**:
- **Message Builder**: Format messages for prompts
- **Context Utils**: Extract data from context
- **Output Formatter**: Parse and format agent outputs
- **Filter Instructions**: Generate filter-based prompt injections

**Example**:
```python
# utils/message_builder.py

def format_schema_for_prompt(schema_data: dict) -> str:
    """Format schema for agent instructions."""
    lines = ["Database Schema:", ""]
    
    for table, columns in schema_data.items():
        lines.append(f"**{table}**:")
        for col in columns:
            lines.append(f"  - {col['name']}: {col['type']}")
        lines.append("")
    
    return "\n".join(lines)
```

---

#### 10. **`instructions/shared/` - Reusable Instruction Templates**

**Purpose**: Store reusable prompt templates and instruction snippets.

**Example**:
```markdown
# instructions/shared/query_generation.md

You are an expert database query generator.

## Your Task
Generate optimized queries based on user requests.

## Guidelines
- Always use parameterized queries
- Optimize for performance
- Validate syntax before returning

## Output Format
Return a JSON object with:
- query: The SQL/Cypher query
- explanation: Why this query solves the problem
- confidence: Your confidence level (0.0-1.0)
```

Load in agent definition:
```python
with open("instructions/shared/query_generation.md") as f:
    instructions = f.read()

query_agent = AgentDefinition(
    name="query_generator",
    instructions=instructions,
    tools=["execute_query"]
)
```

---

## Workflows Core

Background task orchestration using **Prefect 3.x** for multi-step workflows.

### Workflows Core Folder Structure

```
workflows_core/
├── __init__.py                    # Main exports
├── flows/                         # Workflow definitions (@flow)
│   ├── orchestration/            # Master orchestration flows
│   │   └── __init__.py
│   └── __init__.py
└── tasks/                         # Reusable task units (@task)
    └── __init__.py
```

### Workflows Core Patterns

#### 1. **`flows/` - Workflow Definitions**

**Purpose**: Complete business processes orchestrated with Prefect.

**Pattern**: One flow = One complete workflow

**Example**:
```python
# flows/data_processing_flow.py
from prefect import flow
from workflows_core.tasks import fetch_data_task, process_data_task, save_data_task

@flow(name="Data Processing Workflow", log_prints=True)
async def process_data_flow(input_id: str, db_connector):
    """Complete workflow for processing data.
    
    Steps:
    1. Fetch raw data
    2. Process and transform
    3. Save results
    
    Args:
        input_id: Input data identifier
        db_connector: Database connection
        
    Returns:
        Dict with success status and results
    """
    # Step 1: Fetch
    raw_data = await fetch_data_task(input_id, db_connector)
    
    # Step 2: Process
    processed_data = await process_data_task(raw_data)
    
    # Step 3: Save
    await save_data_task(processed_data, db_connector)
    
    return {
        "success": True,
        "processed_count": len(processed_data)
    }
```

**Parallel Execution**:
```python
@flow(name="Parallel Processing", log_prints=True)
async def parallel_flow(items: list, db_connector):
    """Process multiple items in parallel."""
    
    # Submit tasks in parallel
    futures = [
        process_item_task.submit(item, db_connector)
        for item in items
    ]
    
    # Wait for all (synchronous, blocks until done)
    results = [future.result() for future in futures]
    
    return {"success": True, "results": results}
```

**Key Rules**:
- Use `@flow` decorator
- Use `log_prints=True` to see print statements
- Return structured dicts
- Handle errors gracefully
- Keep flows focused (one business process)

---

#### 2. **`flows/orchestration/` - Master Orchestration Flows**

**Purpose**: Compose multiple atomic flows with conditional routing.

**Pattern**: Master orchestrator calls atomic flows based on triggers

**Example**:
```python
# flows/orchestration/master_workflow.py
from prefect import flow
from workflows_core.flows import upload_flow, process_flow, analyze_flow

@flow(name="Master Data Orchestration", log_prints=True)
async def orchestrate_data_workflow(
    data_id: str,
    trigger_source: str = "manual",
    db_connector = None
):
    """Master orchestrator routing to different flows.
    
    Args:
        trigger_source: Controls execution path
            - "upload": Upload → Process → Analyze
            - "api": Process → Analyze (skip upload)
            - "scheduled": Analyze only
    """
    results = {}
    
    # Conditional routing
    if trigger_source == "upload":
        results["upload"] = await upload_flow(data_id, db_connector)
    
    if trigger_source in ["upload", "api"]:
        results["process"] = await process_flow(data_id, db_connector)
    
    # Always analyze
    results["analyze"] = await analyze_flow(data_id, db_connector)
    
    return {"success": True, "results": results}
```

**Use Cases**:
- File upload → Parse → Process → Save
- Conditional workflows (if A then B, else C)
- Background processing (API returns immediately, workflow continues)

---

#### 3. **`tasks/` - Reusable Task Units**

**Purpose**: Thin wrappers around business logic, decorated with `@task`.

**Pattern**: Tasks delegate to services (NO business logic in tasks)

**Example**:
```python
# tasks/database_tasks.py
from prefect import task
from prefect.cache_policies import NONE as NO_CACHE

@task(name="Fetch Data", cache_policy=NO_CACHE)
async def fetch_data_task(data_id: str, db_connector):
    """Fetch data from database.
    
    Thin wrapper - delegates to db_connector.
    """
    result = await db_connector.get_data(data_id)
    return result

@task(name="Save Data", cache_policy=NO_CACHE)
async def save_data_task(data: dict, db_connector):
    """Save data to database."""
    await db_connector.save(data)
    return {"success": True}

@task(name="Transform Data")
async def transform_data_task(raw_data: dict) -> dict:
    """Transform data (no db, default caching OK)."""
    # Call your transformation function
    transformed = your_service.transform(raw_data)
    return transformed
```

**Key Rules**:
1. ✅ Use `cache_policy=NO_CACHE` for tasks with **non-serializable params** (db_connector, file handles)
2. ✅ Tasks are **THIN WRAPPERS** - delegate to services
3. ✅ One task = One focused operation
4. ✅ Tasks can be reused across multiple flows
5. ✅ Group related tasks by domain (create task files as needed)

**When to use `cache_policy=NO_CACHE`**:
```python
# ✅ NEED NO_CACHE (non-serializable param)
@task(cache_policy=NO_CACHE)
async def task_with_db(data, db_connector):  # db_connector not serializable
    pass

# ✅ DEFAULT CACHING OK (all params are serializable)
@task()
async def task_with_primitives(data: dict, count: int):  # dict and int are serializable
    pass
```

---

## Quick Start

### 1. Clone and Adapt

```bash
# Copy package to your project
cp -r package_agentic /path/to/your/project/

# Rename if desired
mv package_agentic orchestration_framework
```

### 2. Define Your First Agent

```python
# your_project/agents_core/agents/my_agents.py
from agents_core import AgentDefinition, register_agent

data_analyzer = AgentDefinition(
    name="data_analyzer",
    description="Analyzes database data",
    instructions="""
You are a data analyst. Analyze the data and provide insights.

Use the available tools to:
1. Execute queries to fetch data
2. Analyze patterns
3. Generate reports
""",
    tools=["execute_query", "generate_report"],
    model="gpt-4o-mini"
)

register_agent(data_analyzer)
```

### 3. Define Your First Tool

```python
# your_project/agents_core/tools/database_tools.py
from agents import function_tool
from agents_core.registry import register_tool

@register_tool(
    name="execute_query",
    description="Execute database query",
    category="database",
    parameters_description="query (str): SQL query",
    returns_description="Query results as list of dicts"
)
@function_tool
async def execute_query(ctx, query: str) -> dict:
    db = ctx.context.database_connector
    results = await db.execute(query)
    return {"success": True, "data": results}
```

### 4. Run Your First Workflow

```python
# your_project/main.py
from agents_core import AgentOrchestrator

async def main():
    orchestrator = AgentOrchestrator()
    
    result = await orchestrator.run(
        user_query="Show me sales data for last quarter",
        context_data={
            "database_connector": your_db_connection,
            "schema": your_schema_text
        }
    )
    
    print(result)

# Run
import asyncio
asyncio.run(main())
```

### 5. Create Your First Prefect Workflow

```python
# your_project/workflows_core/tasks/my_tasks.py
from prefect import task
from prefect.cache_policies import NONE as NO_CACHE

@task(name="Process Data", cache_policy=NO_CACHE)
async def process_data_task(data: dict, db_connector):
    result = await your_service.process(data, db_connector)
    return result

# your_project/workflows_core/flows/my_flows.py
from prefect import flow
from workflows_core.tasks.my_tasks import process_data_task

@flow(name="My Workflow", log_prints=True)
async def my_workflow(input_data: dict, db_connector):
    result = await process_data_task(input_data, db_connector)
    return {"success": True, "result": result}
```

---

## Usage Patterns

### Pattern 1: Simple Agent Workflow

Single agent, single tool, straightforward execution.

```python
from agents_core import AgentOrchestrator, AgentDefinition, register_agent
from agents import function_tool
from agents_core.registry import register_tool

# Define tool
@register_tool(
    name="fetch_weather",
    description="Fetch weather data",
    category="api",
    parameters_description="city (str): City name",
    returns_description="Weather data dict"
)
@function_tool
async def fetch_weather(ctx, city: str) -> dict:
    # Call external API
    weather_data = await your_weather_api.get(city)
    return weather_data

# Define agent
weather_agent = AgentDefinition(
    name="weather_assistant",
    description="Provides weather information",
    instructions="You help users get weather information. Use fetch_weather tool.",
    tools=["fetch_weather"],
    model="gpt-4o-mini"
)
register_agent(weather_agent)

# Run
orchestrator = AgentOrchestrator()
result = await orchestrator.run(
    user_query="What's the weather in Paris?",
    context_data={"database_connector": None}  # Not needed for this example
)
```

---

### Pattern 2: Multi-Agent Sequential Workflow

Multiple agents run in sequence, each building on previous results.

```python
class MyOrchestrator(AgentOrchestrator):
    async def run(self, user_query: str, context_data: dict):
        session = AgentSession(session_id=str(uuid.uuid4()))
        context = AgentContext(**context_data)
        
        # Step 1: Analyze intent
        intent_result = await self._run_agent("intent_analyzer", session, context)
        
        # Step 2: Generate plan
        plan_result = await self._run_agent("planner", session, context)
        
        # Step 3: Execute plan
        execution_result = await self._run_agent("executor", session, context)
        
        # Step 4: Synthesize response
        final_result = await self._run_agent("synthesizer", session, context)
        
        return {"success": True, "response": final_result}
```

---

### Pattern 3: Agent + Workflow Integration

Combine agent intelligence with Prefect workflow orchestration.

```python
# workflows_core/tasks/agent_tasks.py
from prefect import task
from prefect.cache_policies import NONE as NO_CACHE
from agents_core import AgentOrchestrator

@task(name="Run Agent Analysis", cache_policy=NO_CACHE)
async def run_agent_analysis_task(data: dict, db_connector):
    """Run AI agent for analysis."""
    orchestrator = AgentOrchestrator()
    result = await orchestrator.run(
        user_query=f"Analyze this data: {data}",
        context_data={"database_connector": db_connector, "data": data}
    )
    return result

# workflows_core/flows/intelligent_flow.py
@flow(name="Intelligent Processing", log_prints=True)
async def intelligent_processing_flow(data: dict, db_connector):
    # Step 1: Fetch data
    raw_data = await fetch_data_task(data["id"], db_connector)
    
    # Step 2: AI analysis
    ai_result = await run_agent_analysis_task(raw_data, db_connector)
    
    # Step 3: Save insights
    await save_insights_task(ai_result, db_connector)
    
    return {"success": True, "insights": ai_result}
```

---

### Pattern 4: Dynamic Routing Based on Agent Output

Route to different agents based on previous agent's output.

```python
class DynamicOrchestrator(AgentOrchestrator):
    async def run(self, user_query: str, context_data: dict):
        session = AgentSession(session_id=str(uuid.uuid4()))
        context = AgentContext(**context_data)
        
        # Initial routing agent
        routing_result = await self._run_agent("router", session, context)
        
        # Route based on output
        if routing_result.intent == "query":
            result = await self._run_agent("query_specialist", session, context)
        elif routing_result.intent == "analysis":
            result = await self._run_agent("analysis_specialist", session, context)
        elif routing_result.intent == "report":
            result = await self._run_agent("report_generator", session, context)
        else:
            result = await self._run_agent("general_assistant", session, context)
        
        return {"success": True, "result": result}
```

---

## Integration with Your Project

### Step 1: Install Dependencies

```bash
pip install agents prefect openai
```

### Step 2: Set Up Your Project Structure

```
your_project/
├── agents_core/          # Copy from package_agentic
├── workflows_core/       # Copy from package_agentic
├── your_business_logic/  # Your services, models, etc.
│   ├── services/
│   ├── models/
│   └── database/
├── main.py               # Entry point
└── config.yaml           # Configuration
```

### Step 3: Extend AgentContext for Your Domain

```python
# your_project/agents_core/models/custom_context.py
from dataclasses import dataclass, field
from typing import Dict, Any
from agents_core import AgentContext

@dataclass
class YourAppContext(AgentContext):
    """Extended context for your application."""
    user_id: str = ""
    workspace_id: str = ""
    permissions: Dict[str, bool] = field(default_factory=dict)
    custom_data: Dict[str, Any] = field(default_factory=dict)
```

### Step 4: Define Your Agents

```python
# your_project/agents_core/agents/your_agents.py
from agents_core import AgentDefinition, register_agent

# Import and register all your agents
from .domain_agents import data_agent, report_agent, query_agent

# Agents are auto-registered via register_agent() calls in their modules
```

### Step 5: Define Your Tools

```python
# your_project/agents_core/tools/your_tools.py
from agents import function_tool
from agents_core.registry import register_tool
from your_business_logic.services import YourService

@register_tool(
    name="your_tool",
    description="Does something useful",
    category="business",
    parameters_description="param (str): Parameter",
    returns_description="Result dict"
)
@function_tool
async def your_tool(ctx, param: str) -> dict:
    service = YourService(ctx.context.database_connector)
    result = await service.do_something(param)
    return {"success": True, "data": result}
```

### Step 6: Create Your Orchestrator

```python
# your_project/orchestrator.py
from agents_core import AgentOrchestrator
from your_project.agents_core.models.custom_context import YourAppContext

class YourOrchestrator(AgentOrchestrator):
    async def run(self, user_query: str, context_data: dict):
        # Use your custom context
        context = YourAppContext(**context_data)
        
        # Your workflow logic
        # ... (similar to examples above)
```

### Step 7: Integrate with Your API

```python
# your_project/api/endpoints.py
from fastapi import APIRouter, BackgroundTasks
from your_project.orchestrator import YourOrchestrator
from your_project.workflows_core.flows import your_workflow

router = APIRouter()

@router.post("/analyze")
async def analyze_data(request: dict, background_tasks: BackgroundTasks):
    # Option 1: Run agent synchronously
    orchestrator = YourOrchestrator()
    result = await orchestrator.run(
        user_query=request["query"],
        context_data={"database_connector": db, "user_id": request["user_id"]}
    )
    return result

@router.post("/process")
async def process_data(request: dict, background_tasks: BackgroundTasks):
    # Option 2: Run workflow in background
    background_tasks.add_task(
        your_workflow,
        data_id=request["data_id"],
        db_connector=db
    )
    return {"success": True, "message": "Processing started"}
```

---

## Summary

This package provides:

✅ **Generic Agent Orchestration** (`agents_core/`)
- Session management
- Context propagation
- Declarative agent/tool registration
- Code orchestration pattern

✅ **Generic Workflow Orchestration** (`workflows_core/`)
- Prefect flows and tasks
- Parallel execution
- Orchestration patterns

✅ **Production-Ready Patterns**
- Separation of concerns
- Reusable components
- Easy to extend
- No domain-specific logic

### Next Steps

1. Clone this package to your project
2. Extend `AgentContext` for your domain
3. Define your agents in `agents_core/agents/`
4. Define your tools in `agents_core/tools/`
5. Create workflows in `workflows_core/flows/`
6. Integrate with your API/application

**Happy orchestrating! 🚀**
