from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Connection settings shared by the agents and optional live runtime."""

    provider: str
    model_name: str
    temperature: float = 0.2
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Return a supported provider name, accepting common aliases and typos."""

    normalized = value.strip().lower().replace("_", "-")
    aliases = {
        "anthorpic": "anthropic",
        "claude": "anthropic",
        "google": "gemini",
        "google-genai": "gemini",
        "open-ai": "openai",
        "open-ai-compatible": "custom",
        "openai-compatible": "custom",
        "local": "ollama",
        "open-router": "openrouter",
    }
    normalized = aliases.get(normalized, normalized)
    supported = {"openai", "custom", "gemini", "anthropic", "ollama", "openrouter"}
    if normalized not in supported:
        raise ValueError(f"Unsupported LLM provider: {value!r}. Expected one of {sorted(supported)}.")
    return normalized


def build_chat_model(config: ProviderConfig):
    """Build a LangChain chat model lazily for the selected provider.

    Offline mode never calls this function, so the lab remains runnable without
    provider packages or API keys. Imports stay local to produce useful errors
    only when live mode is explicitly configured.
    """

    provider = normalize_provider(config.provider)
    common = {"model": config.model_name, "temperature": config.temperature}

    try:
        if provider in {"openai", "custom"}:
            from langchain_openai import ChatOpenAI

            kwargs = dict(common)
            if config.api_key:
                kwargs["api_key"] = config.api_key
            if provider == "custom":
                if not config.base_url:
                    raise ValueError("CUSTOM_BASE_URL is required for the custom provider.")
                kwargs["base_url"] = config.base_url
            return ChatOpenAI(**kwargs)

        if provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            kwargs = dict(common)
            if config.api_key:
                kwargs["google_api_key"] = config.api_key
            return ChatGoogleGenerativeAI(**kwargs)

        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            kwargs = dict(common)
            if config.api_key:
                kwargs["api_key"] = config.api_key
            return ChatAnthropic(**kwargs)

        if provider == "ollama":
            from langchain_ollama import ChatOllama

            kwargs = dict(common)
            if config.base_url:
                kwargs["base_url"] = config.base_url
            return ChatOllama(**kwargs)

        from langchain_openrouter import ChatOpenRouter

        kwargs = dict(common)
        if config.api_key:
            kwargs["api_key"] = config.api_key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenRouter(**kwargs)
    except ImportError as exc:
        raise RuntimeError(
            f"Live provider '{provider}' needs its LangChain integration installed. "
            "Use force_offline=True for the deterministic lab runtime."
        ) from exc
