"""Decision providers for Latent Earth."""

from __future__ import annotations

from latent_earth.config import ProviderConfig, InferenceConfig
from latent_earth.providers.base import DecisionProvider
from latent_earth.providers.mock import MockProvider
from latent_earth.providers.jev import JevProvider
from latent_earth.providers.openai_compatible import OpenAICompatibleProvider
from latent_earth.providers.openrouter import OpenRouterD1Provider, DEFAULT_OPENROUTER_DECISIONS_URL, DEFAULT_OPENROUTER_MODEL


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
    elif name in ("openrouter", "openrouter_d1", "openrouterd1", "liquid", "liquid_d1"):
        model = provider_config.model if provider_config.model and provider_config.model != "jev-latest" else DEFAULT_OPENROUTER_MODEL
        base_url = provider_config.base_url if provider_config.base_url and "typesafe.ai" not in provider_config.base_url else DEFAULT_OPENROUTER_DECISIONS_URL
        return OpenRouterD1Provider(
            model=model,
            prompt_version=provider_config.prompt_version,
            base_url=base_url,
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
            f"Unknown provider '{provider_config.name}'. Supported: 'jev', 'openrouter', 'mock', 'openai_compatible'."
        )


__all__ = [
    "DecisionProvider",
    "MockProvider",
    "JevProvider",
    "OpenRouterD1Provider",
    "OpenAICompatibleProvider",
    "get_provider",
]
