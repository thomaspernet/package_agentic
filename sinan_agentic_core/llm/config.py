"""Pydantic provider configs for the LLM layer.

A single discriminated union (:data:`LLMProviderConfig`) covers public OpenAI
and Azure OpenAI deployments. Other OpenAI-compatible endpoints (Together,
OpenRouter, local proxies) reuse :class:`OpenAIProviderConfig` via ``base_url``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, TypeAdapter

ApiMode = Literal["chat_completions", "responses"]


class _BaseProviderConfig(BaseModel):
    """Fields shared by every OpenAI-compatible provider."""

    model_config = ConfigDict(extra="forbid")

    api_key: SecretStr
    api_mode: ApiMode = "responses"
    use_for_tracing: bool = True
    disable_tracing: bool = False


class OpenAIProviderConfig(_BaseProviderConfig):
    """Public OpenAI (or any OpenAI-compatible endpoint via ``base_url``).

    The ``model`` field on each agent in ``agents.yaml`` is forwarded verbatim
    as the ``model=`` argument on the underlying SDK call (e.g. ``gpt-4o``).
    """

    provider: Literal["openai"] = "openai"
    base_url: str | None = None
    organization: str | None = None
    project: str | None = None


class AzureOpenAIProviderConfig(_BaseProviderConfig):
    """Azure OpenAI deployment.

    Azure exposes only the chat-completions API and rejects requests to the
    public OpenAI tracing backend, so the defaults pin ``api_mode`` to
    ``chat_completions`` and disable tracing.

    The ``model`` field on each agent in ``agents.yaml`` is forwarded verbatim
    as the ``model=`` argument on the underlying SDK call. On Azure that value
    is the **deployment name**, not the underlying model id. The same
    ``agents.yaml`` works across providers when Azure deployments are named
    after the model they back (e.g. deployment ``gpt-4o`` -> model ``gpt-4o``).
    """

    provider: Literal["azure_openai"] = "azure_openai"
    azure_endpoint: str
    api_version: str
    azure_deployment: str | None = None
    api_mode: ApiMode = "chat_completions"
    use_for_tracing: bool = False
    disable_tracing: bool = True


LLMProviderConfig = Annotated[
    OpenAIProviderConfig | AzureOpenAIProviderConfig,
    Field(discriminator="provider"),
]
"""Discriminated union of every supported provider config."""


_LLM_PROVIDER_ADAPTER: TypeAdapter[OpenAIProviderConfig | AzureOpenAIProviderConfig] = TypeAdapter(
    LLMProviderConfig
)


def parse_llm_provider_config(
    data: dict[str, Any],
) -> OpenAIProviderConfig | AzureOpenAIProviderConfig:
    """Validate a raw mapping against :data:`LLMProviderConfig`.

    Raises :class:`pydantic.ValidationError` for unknown providers or invalid
    fields (the ``provider`` discriminator rejects anything that is not
    ``openai`` or ``azure_openai``).
    """
    return _LLM_PROVIDER_ADAPTER.validate_python(data)
