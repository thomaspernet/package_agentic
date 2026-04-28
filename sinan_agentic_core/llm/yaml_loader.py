"""YAML loader for LLM provider configs with ``${VAR}`` env interpolation.

Secrets must not live in YAML — instead, write ``${OPENAI_API_KEY}`` and
populate the environment at process start. Missing variables raise
:class:`KeyError` so a typo or a forgotten ``export`` fails loudly rather
than silently falling back to an empty string.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .config import (
    AzureOpenAIProviderConfig,
    OpenAIProviderConfig,
    parse_llm_provider_config,
)

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def load_llm_provider_config(
    path: str | Path,
    section: str = "llm",
) -> OpenAIProviderConfig | AzureOpenAIProviderConfig:
    """Load and validate a provider config from *path*.

    Args:
        path: Path to a YAML file. The file must contain a top-level mapping
            with a *section* key holding the provider config.
        section: Top-level key under which the provider config lives.
            Defaults to ``"llm"``.

    Raises:
        ImportError: If PyYAML is not installed.
        FileNotFoundError: If *path* does not exist.
        KeyError: If the section is missing or an interpolated environment
            variable is not set.
        pydantic.ValidationError: If the config does not match a supported
            provider shape.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required for load_llm_provider_config. "
            "Install it with: pip install pyyaml"
        ) from exc

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"LLM provider config not found at {path}")

    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    if section not in data:
        raise KeyError(
            f"Section '{section}' not found in {path}. "
            f"Available top-level keys: {sorted(data.keys()) or '<none>'}"
        )

    raw = data[section]
    if not isinstance(raw, dict):
        raise TypeError(
            f"Section '{section}' in {path} must be a mapping, got " f"{type(raw).__name__}."
        )

    interpolated = _interpolate_env(raw)
    return parse_llm_provider_config(interpolated)


def _interpolate_env(value: Any) -> Any:
    """Recursively replace ``${VAR}`` placeholders with environment values.

    Strings that consist solely of one placeholder are replaced with the raw
    env value; mixed strings (``"prefix-${X}-suffix"``) get string
    substitution. Missing variables raise :class:`KeyError`.
    """
    if isinstance(value, str):
        return _interpolate_string(value)
    if isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_env(v) for v in value]
    return value


def _interpolate_string(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        var = match.group(1)
        if var not in os.environ:
            raise KeyError(f"Environment variable '{var}' referenced in LLM config " f"is not set.")
        return os.environ[var]

    return _ENV_PATTERN.sub(replace, text)
