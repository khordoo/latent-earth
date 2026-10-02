"""Decision providers for Latent Earth."""

from __future__ import annotations

from latent_earth.config import ProviderConfig, InferenceConfig
from latent_earth.providers.base import DecisionProvider
from latent_earth.providers.mock import MockProvider
from latent_earth.providers.jev import JevProvider
from latent_earth.providers.openai_compatible import OpenAICompatibleProvider


def get_provider(
    provider_config: ProviderConfig,
    inference_config: InferenceConfig,
) -> DecisionProvider:
    """Instantiate a provider according to configuration."""
    name = provider_config.name.lower()
    if name == "mock":
        return MockProvider(
            model=provider_config.model,
            prompt_version=provider_config.prompt_version,
            concurrency=inference_config.concurrency,
        )
    elif name == "jev":
        return JevProvider(
            model=provider_config.model,
            prompt_version=provider_config.prompt_version,
            base_url=provider_config.base_url,
            api_key=provider_config.resolve_api_key(),
            concurrency=inference_config.concurrency,
            max_retries=inference_config.max_retries,
            timeout_seconds=inference_config.timeout_seconds,
            rate_limit_per_second=inference_config.rate_limit_per_second,
        )
    elif name in ("openai", "openai_compatible", "vllm", "ollama", "lmstudio"):
        return OpenAICompatibleProvider(
            model=provider_config.model,
            prompt_version=provider_config.prompt_version,
            base_url=provider_config.base_url,
            api_key=provider_config.resolve_api_key(),
            concurrency=inference_config.concurrency,
            max_retries=inference_config.max_retries,
            timeout_seconds=inference_config.timeout_seconds,
        )
    else:
        raise ValueError(
            f"Unknown provider '{provider_config.name}'. Supported: 'jev', 'mock', 'openai_compatible'."
        )


__all__ = [
    "DecisionProvider",
    "MockProvider",
    "JevProvider",
    "OpenAICompatibleProvider",
    "get_provider",
]
