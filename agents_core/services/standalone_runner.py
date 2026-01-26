"""Standalone agent runner for single-purpose agents.

This service runs individual agents without the full orchestrator overhead.
Useful for agents that don't require multi-agent routing.
"""

import logging
from typing import Any

from ..core import BaseAgentRunner
from ..session import AgentSession
from ..models.context import AgentContext

logger = logging.getLogger(__name__)


# Create a singleton instance of the base runner
_runner = BaseAgentRunner()


async def run_standalone_agent(
    agent_name: str,
    user_input: str,
    context: AgentContext
) -> Any:
    """Run a single agent without orchestrator.
    
    This is a lightweight runner for single-purpose agents that don't 
    need complex orchestration or multi-agent routing.
    
    Args:
        agent_name: Name of registered agent to run
        user_input: Input message/prompt for the agent
        context: AgentContext with required data (neo4j_connector, etc.)
        
    Returns:
        Agent's structured output (dataclass instance)
        
    Raises:
        ValueError: If agent not found in registry
        Exception: If agent execution fails
        
    Example:
        ```python
        from agents_core.services.standalone_runner import run_standalone_agent
        from agents_core.models.context import AgentContext
        
        context = AgentContext(neo4j_connector=db)
        result = await run_standalone_agent(
            agent_name="chunking_strategy_agent",
            user_input="Analyze this document...",
            context=context
        )
        print(result.strategy, result.reasoning)
        ```
    """
    logger.info(f"🤖 Running standalone agent: {agent_name}")
    
    # Create minimal session using base runner
    session = _runner.setup_session(session_id=None, initial_history=None)
    
    # Add system message
    await session.add_items([{
        "role": "system",
        "content": "You are an intelligent assistant."
    }])
    
    # Add user input
    await session.add_items([{
        "role": "user",
        "content": user_input
    }])
    
    logger.info(f"📝 User input: {user_input[:100]}...")
    
    # Run agent using base runner
    try:
        result = await _runner.run_agent(
            agent_name=agent_name,
            session=session,
            context=context,
            input_message=""  # Empty since message already in session
        )
        
        logger.info(f"✅ Agent '{agent_name}' completed successfully")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Agent '{agent_name}' failed: {e}", exc_info=True)
        raise

