"""Base runner for agent execution.

Provides shared functionality for running agents across different contexts
(orchestrator, standalone, etc.).
"""

import logging
from typing import Optional, Dict, Any
from abc import ABC

from agents import Agent, Runner, RunContextWrapper

from ..session import AgentSession
from ..models.context import AgentContext
from ..models import outputs as output_models
from ..registry import get_agent_registry, get_tool_registry, get_guardrail_registry

logger = logging.getLogger(__name__)


class BaseAgentRunner(ABC):
    """Base class for agent execution with shared setup methods.
    
    Provides reusable methods for:
    - Setting up agent registries and tool/guardrail mappings
    - Creating agent instances with proper configuration
    - Running agents with context and session management
    
    Subclasses can focus on orchestration logic while reusing these core methods.
    """
    
    def __init__(self):
        """Initialize registries and build tool/guardrail mappings."""
        self.agent_registry = get_agent_registry()
        self.tool_registry = get_tool_registry()
        self.guardrail_registry = get_guardrail_registry()
        
        # Build tool and guardrail mappings
        self.tool_map = {
            name: tool_def.function 
            for name, tool_def in self.tool_registry._tools.items()
        }
        
        self.guardrail_map = {
            name: guardrail_def.function 
            for name, guardrail_def in self.guardrail_registry._guardrails.items()
        }
        
        logger.info(f"✅ Loaded {len(self.tool_map)} tools: {list(self.tool_map.keys())}")
        logger.info(f"✅ Loaded {len(self.guardrail_map)} guardrails: {list(self.guardrail_map.keys())}")
    
    def setup_context(self, **context_data) -> AgentContext:
        """Setup context with provided data.
        
        Args:
            **context_data: Arbitrary context data (neo4j_connector, filters, etc.)
            
        Returns:
            Initialized AgentContext
        """
        return AgentContext(**context_data)
    
    def setup_session(
        self,
        session_id: Optional[str] = None,
        initial_history: Optional[list] = None
    ) -> AgentSession:
        """Setup session for agent execution.
        
        Args:
            session_id: Optional session ID for continuity
            initial_history: Optional conversation history
            
        Returns:
            Initialized AgentSession
        """
        if session_id is None:
            import uuid
            session_id = str(uuid.uuid4())
        
        return AgentSession(session_id=session_id, initial_history=initial_history)
    
    async def create_agent(
        self,
        agent_name: str,
        context: AgentContext
    ) -> Agent:
        """Create an agent instance with proper tools and configuration.
        
        Args:
            agent_name: Name of registered agent to create
            context: AgentContext for dynamic instruction generation
            
        Returns:
            Configured Agent instance
            
        Raises:
            ValueError: If agent not found in registry
        """
        agent_def = self.agent_registry.get(agent_name)
        if not agent_def:
            available = self.agent_registry.list_all()
            raise ValueError(
                f"Agent '{agent_name}' not found in registry. "
                f"Available agents: {available}"
            )
        
        # Instructions - call if function, use if string
        instructions = agent_def.instructions
        if callable(instructions):
            ctx_wrapper = RunContextWrapper(context)
            instructions = instructions(ctx_wrapper, agent_def)
        
        # Get tools - handle both regular tools and agents-as-tools
        agent_tools = []
        for tool_name in agent_def.tools:
            if tool_name in self.tool_map:
                # Regular tool
                agent_tools.append(self.tool_map[tool_name])
            elif tool_name in self.agent_registry._agents:
                # Agent as tool - create the agent and convert to tool
                tool_agent = await self.create_agent(
                    agent_name=tool_name,
                    context=context
                )
                agent_tools.append(tool_agent.as_tool(
                    tool_name=tool_name,
                    tool_description=self.agent_registry._agents[tool_name].description
                ))
        
        # Get guardrails
        agent_guardrails = [
            self.guardrail_map[guardrail_name]
            for guardrail_name in agent_def.guardrails
            if guardrail_name in self.guardrail_map
        ]
        
        # Get handoffs - convert agent names to agent instances
        handoffs = []
        for handoff_name in agent_def.handoffs:
            if handoff_name in self.agent_registry._agents:
                handoff_agent = await self.create_agent(
                    agent_name=handoff_name,
                    context=context
                )
                handoffs.append(handoff_agent)
        
        # Get output type
        if agent_def.output_dataclass:
            if isinstance(agent_def.output_dataclass, str):
                output_type = getattr(output_models, agent_def.output_dataclass)
            else:
                output_type = agent_def.output_dataclass
        else:
            output_type = str
        
        # Get model settings
        model_settings = None
        if agent_def.model_settings_fn:
            ctx_wrapper = RunContextWrapper(context)
            model_settings = agent_def.model_settings_fn(ctx_wrapper)
        
        # Create agent
        agent_kwargs = {
            "name": agent_def.name,
            "instructions": instructions,
            "tools": agent_tools,
            "output_guardrails": agent_guardrails if agent_guardrails else [],
            "model": agent_def.model,
            "output_type": output_type
        }
        
        # Add handoffs if any
        if handoffs:
            agent_kwargs["handoffs"] = handoffs
        
        # Only add model_settings if it's not None
        if model_settings is not None:
            agent_kwargs["model_settings"] = model_settings
        
        agent = Agent[AgentContext](**agent_kwargs)
        
        logger.info(f"🤖 Created agent: {agent_name} (model: {agent_def.model})")
        
        return agent
    
    async def run_agent(
        self,
        agent_name: str,
        session: AgentSession,
        context: AgentContext,
        input_message: str = ""
    ) -> Any:
        """Run agent and return structured output.
        
        SDK automatically handles:
        - Tool execution (execute_cypher_query, etc.)
        - Adding tool results to session as Message(role="tool")
        - Agent seeing results and iterating if needed
        
        Args:
            agent_name: Name of agent to run
            session: Session with conversation history
            context: AgentContext with required data
            input_message: Optional input message for the run
            
        Returns:
            Agent's structured output (dataclass or final output)
        """
        # Create the agent with all tools and handoffs properly configured
        agent = await self.create_agent(
            agent_name=agent_name,
            context=context
        )
        
        logger.info(f"▶️  Running agent: {agent_name}")
        
        # Run the agent
        result = await Runner.run(
            starting_agent=agent,
            input=input_message,
            session=session,
            context=context
        )
        
        logger.info(f"✅ Agent '{agent_name}' completed successfully")
        
        # Return final output if available, otherwise raw result
        return result.final_output if hasattr(result, 'final_output') else result
