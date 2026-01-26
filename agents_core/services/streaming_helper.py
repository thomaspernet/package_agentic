"""Streaming helper for event emission.

This module provides a generic streaming event system for agent workflows.
Events can be consumed by SSE endpoints, WebSockets, or other real-time channels.
"""

from dataclasses import dataclass
from typing import Optional, List, Any, Dict


# ============================================================================
# Generic Event Types (no external dependencies)
# ============================================================================

@dataclass
class BaseEvent:
    """Base class for all streaming events."""
    event_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {"event_type": self.event_type}


@dataclass
class AgentStartEvent(BaseEvent):
    """Emitted when an agent starts processing."""
    agent_name: str
    iteration: int
    
    def __init__(self, agent_name: str, iteration: int = 1):
        self.event_type = "agent_start"
        self.agent_name = agent_name
        self.iteration = iteration
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "agent_name": self.agent_name,
            "iteration": self.iteration
        }


@dataclass
class AgentCompleteEvent(BaseEvent):
    """Emitted when an agent completes processing."""
    agent_name: str
    iteration: int
    
    def __init__(self, agent_name: str, iteration: int = 1):
        self.event_type = "agent_complete"
        self.agent_name = agent_name
        self.iteration = iteration
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "agent_name": self.agent_name,
            "iteration": self.iteration
        }


@dataclass
class ThinkingEvent(BaseEvent):
    """Emitted when an agent is thinking/processing."""
    message: str
    agent_name: Optional[str] = None
    
    def __init__(self, message: str, agent_name: Optional[str] = None):
        self.event_type = "thinking"
        self.message = message
        self.agent_name = agent_name
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "message": self.message,
            "agent_name": self.agent_name
        }


@dataclass
class ToolCallEvent(BaseEvent):
    """Emitted when an agent calls a tool."""
    tool_name: str
    arguments: Dict[str, Any]
    agent_name: Optional[str] = None
    
    def __init__(self, tool_name: str, arguments: Dict[str, Any], agent_name: Optional[str] = None):
        self.event_type = "tool_call"
        self.tool_name = tool_name
        self.arguments = arguments
        self.agent_name = agent_name
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "agent_name": self.agent_name
        }


@dataclass
class StreamingTextEvent(BaseEvent):
    """Emitted for streaming text chunks."""
    text: str
    agent_name: Optional[str] = None
    
    def __init__(self, text: str, agent_name: Optional[str] = None):
        self.event_type = "text_delta"
        self.text = text
        self.agent_name = agent_name
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "text": self.text,
            "agent_name": self.agent_name
        }


@dataclass
class AnswerEvent(BaseEvent):
    """Emitted when final answer is ready."""
    answer: str
    sources: List[Any]
    followup_question: Optional[str] = None
    confidence: Optional[float] = None
    
    def __init__(
        self, 
        answer: str, 
        sources: Optional[List[Any]] = None,
        followup_question: Optional[str] = None,
        confidence: Optional[float] = None
    ):
        self.event_type = "answer"
        self.answer = answer
        self.sources = sources or []
        self.followup_question = followup_question
        self.confidence = confidence
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "answer": self.answer,
            "sources": self.sources,
            "followup_question": self.followup_question,
            "confidence": self.confidence
        }


@dataclass
class ErrorEvent(BaseEvent):
    """Emitted when an error occurs."""
    error: str
    agent_name: Optional[str] = None
    
    def __init__(self, error: str, agent_name: Optional[str] = None):
        self.event_type = "error"
        self.error = error
        self.agent_name = agent_name
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "error": self.error,
            "agent_name": self.agent_name
        }


# ============================================================================
# Streaming Helper
# ============================================================================

class StreamingHelper:
    """Helper for emitting streaming events during workflow execution."""
    
    def __init__(self, event_callback: Optional[callable] = None):
        """Initialize streaming helper.
        
        Args:
            event_callback: Optional callback function for streaming events
        """
        self.event_callback = event_callback
    
    def emit_agent_start(self, agent_name: str, iteration: int):
        """Emit agent start event.
        
        Args:
            agent_name: Name of the agent starting
            iteration: Current iteration number
        """
        if self.event_callback:
            self.event_callback(AgentStartEvent(agent_name, iteration))
    
    def emit_agent_complete(self, agent_name: str, iteration: int):
        """Emit agent complete event.
        
        Args:
            agent_name: Name of the agent that completed
            iteration: Current iteration number
        """
        if self.event_callback:
            self.event_callback(AgentCompleteEvent(agent_name, iteration))
    
    def emit_answer(
        self,
        answer: str,
        sources: List[Any],
        followup_question: Optional[str] = None,
        confidence: Optional[float] = None
    ):
        """Emit final answer event.
        
        Args:
            answer: The final answer to the user's question
            sources: List of sources used to generate the answer
            followup_question: Optional suggested follow-up question
            confidence: Optional confidence score (0.0 to 1.0)
        """
        if self.event_callback:
            self.event_callback(AnswerEvent(
                answer=answer,
                followup_question=followup_question,
                confidence=confidence,
                sources=sources
            ))
    
    def emit_error(self, error_msg: str):
        """Emit error event.
        
        Args:
            error_msg: Error message to emit
        """
        if self.event_callback:
            self.event_callback(ErrorEvent(error_msg))
