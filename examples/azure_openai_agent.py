"""Example: run a simple chat against Azure OpenAI.

Set the following environment variables before running::

    export AZURE_OPENAI_API_KEY="..."
    export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
    export AZURE_OPENAI_API_VERSION="2024-08-01-preview"
    export AZURE_OPENAI_DEPLOYMENT="gpt-4o"

The ``AZURE_OPENAI_DEPLOYMENT`` value is the deployment name in your Azure
resource — it doubles as the ``model`` field on the ``AgentDefinition`` below.
That value is forwarded verbatim as ``model=`` on the underlying SDK call, so
deployments named after the model they back (e.g. deployment ``gpt-4o`` ->
model ``gpt-4o``) let the same ``agents.yaml`` work across providers.

Usage::

    python examples/azure_openai_agent.py
"""

from __future__ import annotations

import asyncio
import os

from agents import Agent, Runner
from pydantic import SecretStr

from sinan_agentic_core import (
    AgentDefinition,
    AgentSession,
    AzureOpenAIProviderConfig,
    configure_llm_provider,
    get_agent_registry,
    register_agent,
)


def _build_provider_config() -> AzureOpenAIProviderConfig:
    return AzureOpenAIProviderConfig(
        api_key=SecretStr(os.environ["AZURE_OPENAI_API_KEY"]),
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    )


def _register_agent(deployment: str) -> None:
    register_agent(
        AgentDefinition(
            name="azure_assistant",
            description="A friendly assistant running on Azure OpenAI",
            instructions=(
                "You are a concise, friendly assistant. Answer in one or two " "sentences."
            ),
            tools=[],
            model=deployment,
        )
    )


async def run() -> None:
    config = _build_provider_config()
    configure_llm_provider(config)

    deployment = os.environ["AZURE_OPENAI_DEPLOYMENT"]
    _register_agent(deployment)

    agent_def = get_agent_registry().get("azure_assistant")
    if agent_def is None:
        raise RuntimeError("azure_assistant was not registered")

    sdk_agent = Agent(
        name=agent_def.name,
        instructions=agent_def.instructions,
        model=agent_def.model,
        tools=[],
    )

    session = AgentSession(session_id="azure-example")
    await session.add_items([{"role": "user", "content": "What is the capital of France?"}])

    result = await Runner.run(
        starting_agent=sdk_agent,
        input=await session.get_items(),
    )
    print("Assistant:", result.final_output)


def main() -> None:
    required = (
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_DEPLOYMENT",
    )
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        print("Missing required environment variables:", ", ".join(missing))
        return
    asyncio.run(run())


if __name__ == "__main__":
    main()
