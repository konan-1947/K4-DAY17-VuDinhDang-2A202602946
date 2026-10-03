from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from model_provider import ProviderConfig, normalize_provider


@dataclass
class LabConfig:
    """Shared paths, compact-memory knobs, and optional live model settings."""

    base_dir: Path
    data_dir: Path
    state_dir: Path
    compact_threshold_tokens: int
    compact_keep_messages: int
    model: ProviderConfig
    judge_model: ProviderConfig


def _load_dotenv() -> None:
    """Load a local .env when python-dotenv is installed, without requiring it."""

    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _provider_config(prefix: str, default_provider: str, default_model: str) -> ProviderConfig:
    provider = normalize_provider(os.getenv(f"{prefix}_PROVIDER", default_provider))
    model_name = os.getenv(f"{prefix}_MODEL", default_model)
    temperature = float(os.getenv(f"{prefix}_TEMPERATURE", "0.2"))
    key_names = {
        "openai": "OPENAI_API_KEY",
        "custom": "CUSTOM_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "ollama": "OLLAMA_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    url_names = {
        "custom": "CUSTOM_BASE_URL",
        "ollama": "OLLAMA_BASE_URL",
        "openrouter": "OPENROUTER_BASE_URL",
    }
    return ProviderConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        api_key=os.getenv(f"{prefix}_API_KEY") or os.getenv(key_names[provider]),
        base_url=os.getenv(f"{prefix}_BASE_URL") or os.getenv(url_names.get(provider, "")),
    )


def load_config(base_dir: Path | None = None) -> LabConfig:
    """Load config from environment and create the ignored runtime state directory."""

    _load_dotenv()
    root = (base_dir or Path(__file__).resolve().parent.parent).resolve()
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    model = _provider_config("LLM", "openai", "gpt-4o-mini")
    judge_model = _provider_config("JUDGE", model.provider, model.model_name)
    return LabConfig(
        base_dir=root,
        data_dir=root / "data",
        state_dir=state_dir,
        compact_threshold_tokens=max(32, int(os.getenv("COMPACT_THRESHOLD_TOKENS", "900"))),
        compact_keep_messages=max(1, int(os.getenv("COMPACT_KEEP_MESSAGES", "4"))),
        model=model,
        judge_model=judge_model,
    )
