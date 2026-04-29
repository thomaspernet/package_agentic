"""Standard output models for agent responses.

Provides reusable dataclasses for:
- ToolOutput: Base return type for all tools
- ChatResponse: Standard chat response with tool tracking
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolOutput:
    """Standard output format for tool functions.

    Use as a base class or directly for consistent tool returns.

    Attributes:
        success: Whether the tool executed successfully
        data: The actual data returned
        error: Error message if success is False
        metadata: Additional context
    """

    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"success": self.success}
        if self.data is not None:
            result["data"] = self.data
        if self.error:
            result["error"] = self.error
        if self.metadata:
            result.update(self.metadata)
        return result


@dataclass
class ChatResponse:
    """Standard response format for chat interactions.

    Attributes:
        success: Whether the chat completed successfully
        response: The assistant's text response
        session_id: Session ID for conversation continuity
        tools_called: List of tool names that were invoked
        error: Error message if success is False
        usage: Token usage dict with requests, input_tokens, output_tokens,
            total_tokens, input_tokens_details, output_tokens_details
    """

    success: bool
    response: str = ""
    session_id: str = "default"
    tools_called: list[str] = field(default_factory=list)
    error: str | None = None
    usage: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "success": self.success,
            "response": self.response,
            "session_id": self.session_id,
        }
        if self.tools_called:
            result["tools_called"] = self.tools_called
        if self.error:
            result["error"] = self.error
        if self.usage:
            result["usage"] = self.usage
        return result


__all__ = [
    "ToolOutput",
    "ChatResponse",
]
