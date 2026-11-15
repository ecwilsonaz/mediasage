"""Configuration loading with environment variable priority."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from backend.jellyfin_client import get_jellyfin_client
from backend.models import AppConfig, DefaultsConfig, JellyfinConfig, LLMConfig, PlexConfig
from backend.plex_client import get_plex_client

# Load .env file (if it exists) - env vars take priority
load_dotenv()

# User config file path (for UI-saved settings)
USER_CONFIG_PATH = Path("data/config.user.yaml")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def remove_empty_values(d: dict[str, Any]) -> dict[str, Any]:
    """Remove keys with empty string or None values, recursively."""
    result: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            nested = remove_empty_values(v)
            if nested:  # Only include non-empty dicts
                result[k] = nested
        elif v not in (None, ""):
            result[k] = v
    return result


# Default model mappings per provider
MODEL_DEFAULTS = {
    "anthropic": {
        "analysis": "claude-sonnet-5-5",
        "generation": "claude-haiku-4-5",
    },
    "openai": {
        "analysis": "gpt-6.1-sol",
        "generation": "gpt-6-luna",
    },
    "gemini": {
        "analysis": "gemini-3.5-flash-lite",
        "generation": "gemini-3.5-flash-lite",
    },
    "ollama": {
        "analysis": "",  # Populated from Ollama API
        "generation": "",
    },
    "custom": {
        "analysis": "",  # User-specified
        "generation": "",
    },
}


def load_yaml_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path("config.yaml")

    if not config_path.exists():
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def load_user_yaml_config() -> dict[str, Any]:
    """Load user configuration from config.user.yaml."""
    if not USER_CONFIG_PATH.exists():
        return {}
    with open(USER_CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


class ConfigSaveError(Exception):
    """Raised when configuration cannot be saved."""
    pass


def save_user_config(updates: dict[str, Any]) -> None:
    """Save user configuration to config.user.yaml.

    Only saves non-empty values. Preserves existing user config.

    Raises:
        ConfigSaveError: If file cannot be written (permissions, disk full, etc.)
    """
    existing = load_user_yaml_config()
    merged = deep_merge(existing, updates)
    cleaned = remove_empty_values(merged)

    try:
        with open(USER_CONFIG_PATH, "w") as f:
            yaml.dump(cleaned, f, default_flow_style=False)
    except PermissionError:
        raise ConfigSaveError(
            f"Permission denied writing to {USER_CONFIG_PATH}. "
            "Check that the data directory is writable. "
            "For Docker, ensure the volume is mounted with correct permissions "
            "(e.g., user directive or chown to UID 1000)."
        )
    except OSError as e:
        raise ConfigSaveError(
            f"Failed to save configuration to {USER_CONFIG_PATH}: {e}. "
            "Check disk space and directory permissions."
        )


def get_env_or_yaml(
    env_key: str, yaml_value: Any, default: Any = None
) -> Any:
    """Get value from environment variable or fall back to YAML value.

    An empty variable counts as unset, so compose lines like PLEX_URL=${PLEX_URL:-}
    don't override what was saved in Settings.
    """
    env_value = os.environ.get(env_key)
    if env_value:
        return env_value
    if yaml_value is not None:
        return yaml_value
    return default


# Settings fields that an environment variable overrides, keyed by UpdateConfigRequest field
_ENV_OVERRIDABLE = {
    "media_server": "MEDIA_SERVER",
    "plex_url": "PLEX_URL",
    "plex_token": "PLEX_TOKEN",
    "music_library": "PLEX_MUSIC_LIBRARY",
    "jellyfin_url": "JELLYFIN_URL",
    "jellyfin_token": "JELLYFIN_TOKEN",
    "jellyfin_music_library": "JELLYFIN_MUSIC_LIBRARY",
    "llm_provider": "LLM_PROVIDER",
    "model_analysis": "LLM_MODEL_ANALYSIS",
    "model_generation": "LLM_MODEL_GENERATION",
    "ollama_url": "OLLAMA_URL",
    "custom_url": "CUSTOM_LLM_URL",
    "request_timeout": "LLM_TIMEOUT",
    "library_sync_hours": "LIBRARY_SYNC_HOURS",
}

_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "custom": "CUSTOM_LLM_API_KEY",
}


def saved_api_keys(llm_yaml: dict[str, Any]) -> dict[str, str]:
    """API keys saved in Settings, by provider.

    Older versions saved a single api_key, which belongs to the provider saved alongside it.
    """
    keys = {p: k for p, k in (llm_yaml.get("api_keys") or {}).items() if k}
    legacy_key, legacy_provider = llm_yaml.get("api_key"), llm_yaml.get("provider")
    if legacy_key and legacy_provider and legacy_provider not in keys:
        keys[legacy_provider] = legacy_key
    return keys


def provider_api_key(provider: str, saved_keys: dict[str, str]) -> str:
    """The provider's key: its environment variable, else the key saved for it in Settings."""
    env_var = _API_KEY_ENV.get(provider)
    return (env_var and os.environ.get(env_var)) or saved_keys.get(provider, "")


def env_overrides(provider: str) -> dict[str, str]:
    """Settings fields currently set by environment variables, mapped to the variable name."""
    overrides = {field: var for field, var in _ENV_OVERRIDABLE.items() if os.environ.get(var)}
    key_var = _API_KEY_ENV.get(provider)
    if key_var and os.environ.get(key_var):
        overrides["llm_api_key"] = key_var
    if provider == "custom" and os.environ.get("CUSTOM_LLM_MODEL"):
        overrides.setdefault("model_analysis", "CUSTOM_LLM_MODEL")
        overrides.setdefault("model_generation", "CUSTOM_LLM_MODEL")
    return overrides


def load_config(config_path: Path | None = None) -> AppConfig:
    """Load configuration with priority chain.

    Priority order:
    1. Environment variables (highest)
    2. config.user.yaml (UI-saved settings)
    3. config.yaml file
    4. Default values (lowest)
    """
    yaml_config = load_yaml_config(config_path)
    user_config = load_user_yaml_config()

    # Merge: user config overrides base yaml config
    yaml_config = deep_merge(yaml_config, user_config)

    # Extract nested config sections
    plex_yaml = yaml_config.get("plex", {})
    jellyfin_yaml = yaml_config.get("jellyfin", {})
    llm_yaml = yaml_config.get("llm", {})
    defaults_yaml = yaml_config.get("defaults", {})

    yaml_provider = llm_yaml.get("provider")

    # Determine LLM provider - explicit setting or auto-detect from API keys
    explicit_provider = get_env_or_yaml(
        "LLM_PROVIDER", llm_yaml.get("provider"), None
    )

    # Each provider has its own key, so switching providers never sends another provider's key
    api_keys = saved_api_keys(llm_yaml)
    provider = explicit_provider or next(
        (p for p in ("gemini", "openai", "anthropic") if provider_api_key(p, api_keys)),
        "gemini",  # Default
    )
    api_key = provider_api_key(provider, api_keys)

    # Get model defaults for the provider
    provider_defaults = MODEL_DEFAULTS.get(provider, MODEL_DEFAULTS["gemini"])

    # Build configuration
    plex_config = PlexConfig(
        url=get_env_or_yaml("PLEX_URL", plex_yaml.get("url"), ""),
        token=get_env_or_yaml("PLEX_TOKEN", plex_yaml.get("token"), ""),
        music_library=get_env_or_yaml(
            "PLEX_MUSIC_LIBRARY", plex_yaml.get("music_library"), "Music"
        ),
    )

    jellyfin_config = JellyfinConfig(
        url=get_env_or_yaml("JELLYFIN_URL", jellyfin_yaml.get("url"), ""),
        token=get_env_or_yaml("JELLYFIN_TOKEN", jellyfin_yaml.get("token"), ""),
        music_library=get_env_or_yaml(
            "JELLYFIN_MUSIC_LIBRARY", jellyfin_yaml.get("music_library"), "Music"
        ),
    )

    # MEDIA_SERVER env > YAML > Jellyfin if it's the only server configured > Plex
    media_server = os.environ.get("MEDIA_SERVER") or yaml_config.get("media_server")
    if not media_server:
        media_server = "jellyfin" if jellyfin_config.url and not plex_config.url else "plex"

    # Get local provider settings
    ollama_url = get_env_or_yaml(
        "OLLAMA_URL", llm_yaml.get("ollama_url"), "http://localhost:11434"
    )
    ollama_context_window_str = get_env_or_yaml(
        "OLLAMA_CONTEXT_WINDOW", llm_yaml.get("ollama_context_window"), 32768
    )
    ollama_context_window = int(ollama_context_window_str) if isinstance(
        ollama_context_window_str, str
    ) else ollama_context_window_str
    custom_url = get_env_or_yaml(
        "CUSTOM_LLM_URL", llm_yaml.get("custom_url"), ""
    )
    custom_context_window_str = get_env_or_yaml(
        "CUSTOM_CONTEXT_WINDOW", llm_yaml.get("custom_context_window"), 32768
    )
    # Handle string from env var
    custom_context_window = int(custom_context_window_str) if isinstance(
        custom_context_window_str, str
    ) else custom_context_window_str

    # Determine model names with proper fallback chain
    # When env var overrides to a DIFFERENT provider, use that provider's defaults
    # (prevents using custom provider's model names with gemini provider, etc.)
    # A YAML file without a provider is not a switch: when LLM_PROVIDER locks the provider, the Settings UI
    # never writes llm.provider, so the models saved there belong to the env provider.
    env_provider = os.environ.get("LLM_PROVIDER")
    provider_changed_by_env = bool(env_provider and yaml_provider and env_provider != yaml_provider)

    if provider_changed_by_env:
        # Env var switched to different provider - use new provider's defaults
        # (unless model env vars are also explicitly set)
        model_analysis = os.environ.get("LLM_MODEL_ANALYSIS") or provider_defaults["analysis"]
        model_generation = os.environ.get("LLM_MODEL_GENERATION") or provider_defaults["generation"]
    else:
        # Same provider or no env override - YAML models take precedence
        model_analysis = get_env_or_yaml(
            "LLM_MODEL_ANALYSIS",
            llm_yaml.get("model_analysis"),
            provider_defaults["analysis"],
        )
        model_generation = get_env_or_yaml(
            "LLM_MODEL_GENERATION",
            llm_yaml.get("model_generation"),
            provider_defaults["generation"],
        )

    # CUSTOM_LLM_MODEL sets both models for a custom endpoint; LLM_MODEL_* still win
    custom_model = os.environ.get("CUSTOM_LLM_MODEL")
    if provider == "custom" and custom_model:
        model_analysis = os.environ.get("LLM_MODEL_ANALYSIS") or custom_model
        model_generation = os.environ.get("LLM_MODEL_GENERATION") or custom_model

    request_timeout = int(get_env_or_yaml("LLM_TIMEOUT", llm_yaml.get("request_timeout"), 600))

    llm_config = LLMConfig(
        provider=provider,
        api_key=api_key,
        api_keys=api_keys,
        model_analysis=model_analysis,
        model_generation=model_generation,
        smart_generation=llm_yaml.get("smart_generation", False),
        ollama_url=ollama_url,
        ollama_context_window=ollama_context_window,
        custom_url=custom_url,
        custom_context_window=custom_context_window,
        request_timeout=request_timeout,
    )

    defaults_config = DefaultsConfig(
        track_count=defaults_yaml.get("track_count", 25)
    )

    return AppConfig(
        media_server=media_server,
        plex=plex_config,
        jellyfin=jellyfin_config,
        llm=llm_config,
        defaults=defaults_config,
        library_sync_hours=float(
            get_env_or_yaml("LIBRARY_SYNC_HOURS", yaml_config.get("library_sync_hours"), 24)
        ),
    )


# Global config instance (loaded on import, can be refreshed)
_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Get the current configuration, loading if necessary."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def refresh_config(config_path: Path | None = None) -> AppConfig:
    """Reload configuration from file and environment."""
    global _config
    _config = load_config(config_path)
    return _config


def update_config_values(updates: dict[str, Any]) -> AppConfig:
    """Update configuration values and persist to config.user.yaml.

    Changes are saved to config.user.yaml so they survive server restarts.
    Environment variables still take priority over saved settings.
    """
    global _config
    if _config is None:
        _config = load_config()

    # Create updated config by merging updates
    plex_updates = {}
    jellyfin_updates = {}
    llm_updates = {}
    media_server = updates.get("media_server")

    if "plex_url" in updates and updates["plex_url"]:
        plex_updates["url"] = updates["plex_url"]
    if "plex_token" in updates and updates["plex_token"]:
        plex_updates["token"] = updates["plex_token"]
    if "music_library" in updates and updates["music_library"]:
        plex_updates["music_library"] = updates["music_library"]

    if "jellyfin_url" in updates and updates["jellyfin_url"]:
        jellyfin_updates["url"] = updates["jellyfin_url"]
    if "jellyfin_token" in updates and updates["jellyfin_token"]:
        jellyfin_updates["token"] = updates["jellyfin_token"]
    if "jellyfin_music_library" in updates and updates["jellyfin_music_library"]:
        jellyfin_updates["music_library"] = updates["jellyfin_music_library"]

    if "llm_provider" in updates and updates["llm_provider"]:
        new_provider = updates["llm_provider"]
        llm_updates["provider"] = new_provider

        # Default models only on an actual provider change, so re-saving the same
        # provider (Settings, setup wizard) keeps models the user picked
        if new_provider != _config.llm.provider and new_provider in MODEL_DEFAULTS:
            defaults = MODEL_DEFAULTS[new_provider]
            if not updates.get("model_analysis"):
                llm_updates["model_analysis"] = defaults["analysis"]
            if not updates.get("model_generation"):
                llm_updates["model_generation"] = defaults["generation"]

    # Keys are kept per provider. An entered key belongs to the provider being saved; keys
    # from environment variables are used but never written to the file.
    api_keys = dict(_config.llm.api_keys)
    provider = llm_updates.get("provider", _config.llm.provider)
    if updates.get("llm_api_key"):
        api_keys[provider] = updates["llm_api_key"]
    if api_keys != _config.llm.api_keys or "provider" in llm_updates:
        # Replaces the single api_key older versions saved
        llm_updates["api_keys"] = api_keys
        llm_updates["api_key"] = ""
    if "model_analysis" in updates and updates["model_analysis"]:
        llm_updates["model_analysis"] = updates["model_analysis"]
    if "model_generation" in updates and updates["model_generation"]:
        llm_updates["model_generation"] = updates["model_generation"]

    # Local provider settings
    if "ollama_url" in updates and updates["ollama_url"]:
        llm_updates["ollama_url"] = updates["ollama_url"]
    if "ollama_context_window" in updates and updates["ollama_context_window"]:
        llm_updates["ollama_context_window"] = updates["ollama_context_window"]
    if "custom_url" in updates and updates["custom_url"]:
        llm_updates["custom_url"] = updates["custom_url"]
    if "custom_context_window" in updates and updates["custom_context_window"]:
        llm_updates["custom_context_window"] = updates["custom_context_window"]
    if updates.get("request_timeout"):
        llm_updates["request_timeout"] = updates["request_timeout"]

    # Create new config with updates
    new_plex = _config.plex.model_copy(update=plex_updates)
    new_jellyfin = _config.jellyfin.model_copy(update=jellyfin_updates)
    new_llm = _config.llm.model_copy(
        update={**llm_updates, "api_keys": api_keys, "api_key": provider_api_key(provider, api_keys)}
    )

    library_sync_hours = updates.get("library_sync_hours")  # 0 (off) is a real value
    _config = AppConfig(
        media_server=media_server or _config.media_server,
        plex=new_plex,
        jellyfin=new_jellyfin,
        llm=new_llm,
        defaults=_config.defaults,
        library_sync_hours=_config.library_sync_hours if library_sync_hours is None else library_sync_hours,
    )

    # Persist to user config file
    user_updates: dict[str, Any] = {}
    if media_server:
        user_updates["media_server"] = media_server
    if library_sync_hours is not None:
        user_updates["library_sync_hours"] = library_sync_hours
    if plex_updates:
        user_updates["plex"] = plex_updates
    if jellyfin_updates:
        user_updates["jellyfin"] = jellyfin_updates
    if llm_updates:
        user_updates["llm"] = llm_updates

    if user_updates:
        save_user_config(user_updates)

    return _config


def get_current_media_client():
    """Return the Plex or Jellyfin client for the configured media server, or None."""
    if get_config().media_server == "jellyfin":
        return get_jellyfin_client()
    return get_plex_client()
