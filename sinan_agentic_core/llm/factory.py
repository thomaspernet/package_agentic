"""Build an OpenAI client from a provider config and register it as the SDK default.

This is the single entry point that consumers call once at process start. It
hides the SDK setters (``set_default_openai_client``, ``set_default_openai_api``,
``set_tracing_disabled``) so callers never reach into ``agents.*`` internals.
"""

from __future__ import annotations

import logging

from agents import (
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
)
from openai import AsyncAzureOpenAI, AsyncOpenAI

from .config import AzureOpenAIProviderConfig, OpenAIProviderConfig

logger = logging.getLogger(__name__)


def configure_llm_provider(
    config: OpenAIProviderConfig | AzureOpenAIProviderConfig,
) -> AsyncOpenAI:
    """Build the right ``AsyncOpenAI`` / ``AsyncAzureOpenAI`` client and wire it
    into the OpenAI Agents SDK.

    Side effects:
      - Calls :func:`agents.set_default_openai_client` with ``use_for_tracing``
        from *config*.
      - Calls :func:`agents.set_default_openai_api` with ``config.api_mode``.
      - Calls :func:`agents.set_tracing_disabled` when ``config.disable_tracing``
        is true.

    Returns:
        The constructed client. Most callers ignore it; return it so callers
        that need to make raw OpenAI calls (e.g. embeddings) can reuse it.
    """
    client = _build_client(config)
    set_default_openai_client(client, use_for_tracing=config.use_for_tracing)
    set_default_openai_api(config.api_mode)
    if config.disable_tracing:
        set_tracing_disabled(True)
    logger.info(
        "Configured LLM provider %r (api_mode=%s, tracing=%s)",
        config.provider,
        config.api_mode,
        "off" if config.disable_tracing else "on",
    )
    return client


def _build_client(
    config: OpenAIProviderConfig | AzureOpenAIProviderConfig,
) -> AsyncOpenAI:
    if isinstance(config, AzureOpenAIProviderConfig):
        return AsyncAzureOpenAI(
            api_key=config.api_key.get_secret_value(),
            azure_endpoint=config.azure_endpoint,
            api_version=config.api_version,
            azure_deployment=config.azure_deployment,
        )
    return AsyncOpenAI(
        api_key=config.api_key.get_secret_value(),
        base_url=config.base_url,
        organization=config.organization,
        project=config.project,
    )
