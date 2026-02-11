"""Chat functions for API endpoints.

Three flavours of chat, from simplest to most granular:

- ``chat()``            — non-streaming, returns a dict
- ``chat_with_hooks()`` — yields SSE-style dicts with tool-call notifications
- ``chat_streamed()``   — yields token-level deltas via ``Runner.run_streamed()``

All three handle session history, error handling, and return structured
events so your API layer stays thin.

Usage:
    from agents_core.services.chat import chat, chat_with_hooks, chat_streamed
    from agents_core import AgentSession

    session = AgentSession(session_id="user-123")

    # Simple
    result = await chat("Hello!", agent_name="my_agent", session=session)

    # Streaming tokens
    async for event in chat_streamed("Hello!", "my_agent", session):
        print(event)
"""

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

from agents import Runner, ItemHelpers
from openai.types.responses import ResponseTextDeltaEvent

from ..registry.agent_factory import create_agent_from_registry
from ..session import AgentSession
from .hooks import StreamingRunHooks

logger = logging.getLogger(__name__)


async def chat(
    message: str,
    agent_name: str,
    session: AgentSession,
    model_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single chat turn (non-streaming).

    Args:
        message: User message text.
        agent_name: Name of a registered agent.
        session: ``AgentSession`` that tracks conversation history.
        model_override: Use a different model than the agent definition.

    Returns:
        ``{"success": True, "response": str, "session_id": str, "tools_called": list}``
        or ``{"success": False, "error": str, "session_id": str}`` on failure.
    """
    try:
        agent = create_agent_from_registry(agent_name, model_override)

        await session.add_items([{"role": "user", "content": message}])
        history = await session.get_items()

        result = await Runner.run(starting_agent=agent, input=history)
        response = result.final_output

        await session.add_items([{"role": "assistant", "content": response}])

        return {
            "success": True,
            "response": response,
            "session_id": session.session_id,
            "tools_called": [],
        }
    except Exception as e:
        logger.error("Chat error: %s", e, exc_info=True)
        return {"success": False, "error": str(e), "session_id": session.session_id}


async def chat_with_hooks(
    message: str,
    agent_name: str,
    session: AgentSession,
    tool_friendly_names: Optional[Dict[str, str]] = None,
    model_override: Optional[str] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Chat with real-time tool-call notifications via ``RunHooks``.

    Uses ``Runner.run()`` internally — the agent runs to completion, but
    ``StreamingRunHooks`` pushes events to an ``asyncio.Queue`` so you can
    stream progress to the client while it executes.

    Args:
        message: User message text.
        agent_name: Name of a registered agent.
        session: ``AgentSession`` for conversation history.
        tool_friendly_names: Optional ``tool_name → display name`` mapping.
        model_override: Use a different model than the agent definition.

    Yields:
        Event dicts with an ``"event"`` key and a ``"data"`` dict::

            {"event": "thinking",   "data": {"message": "..."}}
            {"event": "tool_start", "data": {"tool": "...", ...}}
            {"event": "tool_end",   "data": {"tool": "...", ...}}
            {"event": "finalizing", "data": {"message": "..."}}
            {"event": "answer",     "data": {"response": "...", "tools_called": [...]}}
            {"event": "error",      "data": {"error": "..."}}
    """
    queue: asyncio.Queue = asyncio.Queue()
    hooks = StreamingRunHooks(queue, tool_friendly_names)

    try:
        yield {"event": "thinking", "data": {"message": "Analyzing your question..."}}

        agent = create_agent_from_registry(agent_name, model_override)
        await session.add_items([{"role": "user", "content": message}])
        history = await session.get_items()

        # Run agent with hooks in the background
        async def _run():
            return await Runner.run(starting_agent=agent, input=history, hooks=hooks)

        task = asyncio.create_task(_run())

        # Forward hook events as they arrive
        while not task.done():
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.5)
                yield event
            except asyncio.TimeoutError:
                continue

        # Drain any remaining events
        while not queue.empty():
            yield await queue.get()

        yield {"event": "finalizing", "data": {"message": "Generating response..."}}

        result = await task
        response = result.final_output
        await session.add_items([{"role": "assistant", "content": response}])

        yield {
            "event": "answer",
            "data": {"response": response, "tools_called": hooks.tools_called},
        }
    except Exception as e:
        logger.error("Chat hooks error: %s", e, exc_info=True)
        yield {"event": "error", "data": {"error": str(e)}}


async def chat_streamed(
    message: str,
    agent_name: str,
    session: AgentSession,
    model_override: Optional[str] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Chat with token-level streaming via ``Runner.run_streamed()``.

    Yields events for every text delta, tool invocation, tool output,
    agent handoff, and the final answer.  This is the most granular
    streaming option — ideal for rendering responses character-by-character.

    Args:
        message: User message text.
        agent_name: Name of a registered agent.
        session: ``AgentSession`` for conversation history.
        model_override: Use a different model than the agent definition.

    Yields:
        Event dicts::

            {"event": "text_delta",      "data": {"delta": "..."}}
            {"event": "tool_call",       "data": {"tool": "...", "message": "..."}}
            {"event": "tool_output",     "data": {"output": "..."}}
            {"event": "message_output",  "data": {"text": "..."}}
            {"event": "agent_updated",   "data": {"agent": "..."}}
            {"event": "answer",          "data": {"response": "...", "tools_called": [...]}}
            {"event": "error",           "data": {"error": "..."}}
    """
    try:
        agent = create_agent_from_registry(agent_name, model_override)
        await session.add_items([{"role": "user", "content": message}])
        history = await session.get_items()

        result = Runner.run_streamed(starting_agent=agent, input=history)

        tools_called: List[str] = []

        async for event in result.stream_events():
            # Token-level text deltas
            if event.type == "raw_response_event" and isinstance(
                event.data, ResponseTextDeltaEvent
            ):
                yield {"event": "text_delta", "data": {"delta": event.data.delta}}

            # Higher-level run-item events
            elif event.type == "run_item_stream_event":
                item = event.item

                if item.type == "tool_call_item":
                    name = getattr(item, "name", "unknown")
                    tools_called.append(name)
                    yield {
                        "event": "tool_call",
                        "data": {
                            "tool": name,
                            "message": f"Calling {name.replace('_', ' ')}...",
                        },
                    }
                elif item.type == "tool_call_output_item":
                    yield {
                        "event": "tool_output",
                        "data": {"output": str(item.output)[:500]},
                    }
                elif item.type == "message_output_item":
                    yield {
                        "event": "message_output",
                        "data": {"text": ItemHelpers.text_message_output(item)},
                    }

            # Agent handoff
            elif event.type == "agent_updated_stream_event":
                yield {
                    "event": "agent_updated",
                    "data": {"agent": event.new_agent.name},
                }

        # Final answer
        response = result.final_output
        await session.add_items([{"role": "assistant", "content": response}])

        yield {
            "event": "answer",
            "data": {"response": response, "tools_called": tools_called},
        }
    except Exception as e:
        logger.error("Streaming error: %s", e, exc_info=True)
        yield {"event": "error", "data": {"error": str(e)}}
