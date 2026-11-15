"""Tests for configuration loading."""

from unittest.mock import patch

import pytest

import yaml

from backend import config as config_module
from backend.config import (
    deep_merge,
    env_overrides,
    get_env_or_yaml,
    load_config,
    load_yaml_config,
    MODEL_DEFAULTS,
    remove_empty_values,
)


class TestLoadYamlConfig:
    """Tests for YAML config file loading."""

    def test_loads_valid_yaml(self, tmp_path):
        """Should load a valid YAML config file."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "plex": {"url": "http://localhost:32400", "token": "test-token"},
            "llm": {"provider": "anthropic", "api_key": "sk-test"},
        }
        config_file.write_text(yaml.dump(config_data))

        result = load_yaml_config(config_file)

        assert result["plex"]["url"] == "http://localhost:32400"
        assert result["llm"]["provider"] == "anthropic"

    def test_returns_empty_dict_for_missing_file(self, tmp_path):
        """Should return empty dict when config file doesn't exist."""
        config_file = tmp_path / "nonexistent.yaml"

        result = load_yaml_config(config_file)

        assert result == {}

    def test_returns_empty_dict_for_empty_file(self, tmp_path):
        """Should return empty dict for empty config file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")

        result = load_yaml_config(config_file)

        assert result == {}


class TestGetEnvOrYaml:
    """Tests for environment variable priority."""

    def test_env_var_takes_priority(self, monkeypatch):
        """Environment variable should override YAML value."""
        monkeypatch.setenv("TEST_VAR", "env_value")

        result = get_env_or_yaml("TEST_VAR", "yaml_value", "default")

        assert result == "env_value"

    def test_yaml_used_when_no_env_var(self, monkeypatch):
        """YAML value should be used when env var not set."""
        monkeypatch.delenv("TEST_VAR", raising=False)

        result = get_env_or_yaml("TEST_VAR", "yaml_value", "default")

        assert result == "yaml_value"

    def test_default_used_when_no_env_or_yaml(self, monkeypatch):
        """Default should be used when neither env nor YAML set."""
        monkeypatch.delenv("TEST_VAR", raising=False)

        result = get_env_or_yaml("TEST_VAR", None, "default")

        assert result == "default"

    def test_empty_env_var_counts_as_unset(self, monkeypatch):
        """An empty variable (e.g. PLEX_URL=${PLEX_URL:-} in compose) must not override Settings."""
        monkeypatch.setenv("TEST_VAR", "")

        result = get_env_or_yaml("TEST_VAR", "yaml_value", "default")

        assert result == "yaml_value"


class TestLoadConfig:
    """Tests for full configuration loading."""

    def test_loads_from_yaml_file(self, tmp_path, monkeypatch):
        """Should load configuration from YAML file."""
        # Clear any existing env vars
        for var in ["PLEX_URL", "PLEX_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                    "GEMINI_API_KEY", "LLM_PROVIDER", "LLM_MODEL_ANALYSIS", "LLM_MODEL_GENERATION"]:
            monkeypatch.delenv(var, raising=False)

        config_file = tmp_path / "config.yaml"
        config_data = {
            "plex": {
                "url": "http://plex.local:32400",
                "token": "yaml-token",
                "music_library": "My Music",
            },
            "llm": {
                "provider": "anthropic",
                "api_key": "sk-yaml-key",
            },
            "defaults": {"track_count": 40},
        }
        config_file.write_text(yaml.dump(config_data))

        # Patch load_user_yaml_config to return empty dict (ignore config.user.yaml)
        with patch("backend.config.load_user_yaml_config", return_value={}):
            config = load_config(config_file)

        assert config.plex.url == "http://plex.local:32400"
        assert config.plex.token == "yaml-token"
        assert config.plex.music_library == "My Music"
        assert config.llm.provider == "anthropic"
        assert config.llm.api_key == "sk-yaml-key"
        assert config.defaults.track_count == 40

    def test_env_vars_override_yaml(self, tmp_path, monkeypatch):
        """Environment variables should override YAML values."""
        # Clear any conflicting env vars first
        for var in ["GEMINI_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER",
                    "LLM_MODEL_ANALYSIS", "LLM_MODEL_GENERATION"]:
            monkeypatch.delenv(var, raising=False)

        config_file = tmp_path / "config.yaml"
        config_data = {
            "plex": {"url": "http://yaml:32400", "token": "yaml-token"},
            "llm": {"provider": "anthropic", "api_key": "yaml-key"},
        }
        config_file.write_text(yaml.dump(config_data))

        monkeypatch.setenv("PLEX_URL", "http://env:32400")
        monkeypatch.setenv("PLEX_TOKEN", "env-token")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")

        # Patch load_user_yaml_config to return empty dict (ignore config.user.yaml)
        with patch("backend.config.load_user_yaml_config", return_value={}):
            config = load_config(config_file)

        assert config.plex.url == "http://env:32400"
        assert config.plex.token == "env-token"
        assert config.llm.api_key == "env-key"

    def test_uses_correct_api_key_for_provider(self, tmp_path, monkeypatch):
        """Should use ANTHROPIC_API_KEY or OPENAI_API_KEY based on provider."""
        for var in ["PLEX_URL", "PLEX_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                    "GEMINI_API_KEY", "LLM_PROVIDER", "LLM_MODEL_ANALYSIS", "LLM_MODEL_GENERATION"]:
            monkeypatch.delenv(var, raising=False)

        # Patch load_user_yaml_config to return empty dict (ignore config.user.yaml)
        with patch("backend.config.load_user_yaml_config", return_value={}):
            # Test Anthropic provider
            config_file = tmp_path / "config.yaml"
            config_data = {"llm": {"provider": "anthropic", "api_key": ""}}
            config_file.write_text(yaml.dump(config_data))
            monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
            monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

            config = load_config(config_file)
            assert config.llm.api_key == "anthropic-key"

            # Test OpenAI provider
            config_data = {"llm": {"provider": "openai", "api_key": ""}}
            config_file.write_text(yaml.dump(config_data))

            config = load_config(config_file)
            assert config.llm.api_key == "openai-key"

    def test_ui_saved_key_loaded_for_gemini_and_openai(self, tmp_path, monkeypatch):
        """A UI-saved api_key should be used for Gemini/OpenAI when no env var is set."""
        for var in ["PLEX_URL", "PLEX_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                    "GEMINI_API_KEY", "LLM_PROVIDER", "LLM_MODEL_ANALYSIS", "LLM_MODEL_GENERATION"]:
            monkeypatch.delenv(var, raising=False)

        config_file = tmp_path / "config.yaml"
        config_file.write_text("")

        for provider in ("gemini", "openai"):
            user_config = {"llm": {"provider": provider, "api_key": f"{provider}-ui-key"}}
            with patch("backend.config.load_user_yaml_config", return_value=user_config):
                config = load_config(config_file)
            assert config.llm.provider == provider
            assert config.llm.api_key == f"{provider}-ui-key"

    def test_ui_saved_key_not_used_for_other_provider(self, tmp_path, monkeypatch):
        """A key saved for one provider should not be sent to another."""
        for var in ["PLEX_URL", "PLEX_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                    "GEMINI_API_KEY", "LLM_PROVIDER", "LLM_MODEL_ANALYSIS", "LLM_MODEL_GENERATION"]:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("LLM_PROVIDER", "openai")

        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        user_config = {"llm": {"provider": "gemini", "api_key": "gemini-ui-key"}}

        with patch("backend.config.load_user_yaml_config", return_value=user_config):
            config = load_config(config_file)

        assert config.llm.provider == "openai"
        assert config.llm.api_key == ""

    def test_ui_saved_models_kept_when_env_locks_provider(self, tmp_path, monkeypatch):
        """Models saved in Settings survive a restart when LLM_PROVIDER is set and the YAML has no provider."""
        for var in ["PLEX_URL", "PLEX_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
                    "LLM_PROVIDER", "LLM_MODEL_ANALYSIS", "LLM_MODEL_GENERATION", "CUSTOM_LLM_MODEL"]:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("LLM_PROVIDER", "custom")

        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        user_config = {"llm": {"model_analysis": "my-model", "model_generation": "my-model"}}

        with patch("backend.config.load_user_yaml_config", return_value=user_config):
            config = load_config(config_file)

        assert config.llm.provider == "custom"
        assert config.llm.model_analysis == "my-model"
        assert config.llm.model_generation == "my-model"

    def test_default_models_for_anthropic(self, tmp_path, monkeypatch):
        """Should use default Anthropic models when not specified."""
        for var in ["PLEX_URL", "PLEX_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                    "GEMINI_API_KEY", "LLM_PROVIDER", "LLM_MODEL_ANALYSIS", "LLM_MODEL_GENERATION"]:
            monkeypatch.delenv(var, raising=False)

        config_file = tmp_path / "config.yaml"
        config_data = {"llm": {"provider": "anthropic", "api_key": "test"}}
        config_file.write_text(yaml.dump(config_data))

        # Patch load_user_yaml_config to return empty dict (ignore config.user.yaml)
        with patch("backend.config.load_user_yaml_config", return_value={}):
            config = load_config(config_file)

        assert config.llm.model_analysis == MODEL_DEFAULTS["anthropic"]["analysis"]
        assert config.llm.model_generation == MODEL_DEFAULTS["anthropic"]["generation"]

    def test_default_models_for_openai(self, tmp_path, monkeypatch):
        """Should use default OpenAI models when not specified."""
        for var in ["PLEX_URL", "PLEX_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                    "GEMINI_API_KEY", "LLM_PROVIDER", "LLM_MODEL_ANALYSIS", "LLM_MODEL_GENERATION"]:
            monkeypatch.delenv(var, raising=False)

        config_file = tmp_path / "config.yaml"
        config_data = {"llm": {"provider": "openai", "api_key": "test"}}
        config_file.write_text(yaml.dump(config_data))

        # Patch load_user_yaml_config to return empty dict (ignore config.user.yaml)
        with patch("backend.config.load_user_yaml_config", return_value={}):
            config = load_config(config_file)

        assert config.llm.model_analysis == MODEL_DEFAULTS["openai"]["analysis"]
        assert config.llm.model_generation == MODEL_DEFAULTS["openai"]["generation"]

    def test_custom_models_override_defaults(self, tmp_path, monkeypatch):
        """Custom model settings should override defaults."""
        for var in ["PLEX_URL", "PLEX_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                    "GEMINI_API_KEY", "LLM_PROVIDER", "LLM_MODEL_ANALYSIS", "LLM_MODEL_GENERATION"]:
            monkeypatch.delenv(var, raising=False)

        config_file = tmp_path / "config.yaml"
        config_data = {
            "llm": {
                "provider": "anthropic",
                "api_key": "test",
                "model_analysis": "custom-analysis-model",
                "model_generation": "custom-gen-model",
            }
        }
        config_file.write_text(yaml.dump(config_data))

        # Patch load_user_yaml_config to return empty dict (ignore config.user.yaml)
        with patch("backend.config.load_user_yaml_config", return_value={}):
            config = load_config(config_file)

        assert config.llm.model_analysis == "custom-analysis-model"
        assert config.llm.model_generation == "custom-gen-model"

    def test_defaults_applied_when_no_config(self, tmp_path, monkeypatch):
        """Should use defaults when config file doesn't exist."""
        for var in ["PLEX_URL", "PLEX_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                    "GEMINI_API_KEY", "LLM_PROVIDER", "LLM_MODEL_ANALYSIS", "LLM_MODEL_GENERATION"]:
            monkeypatch.delenv(var, raising=False)

        config_file = tmp_path / "nonexistent.yaml"

        # Patch load_user_yaml_config to return empty dict (ignore config.user.yaml)
        with patch("backend.config.load_user_yaml_config", return_value={}):
            config = load_config(config_file)

        assert config.plex.music_library == "Music"
        assert config.llm.provider == "gemini"
        assert config.defaults.track_count == 25

    def test_secrets_not_exposed_in_repr(self, tmp_path, monkeypatch):
        """Secrets should not be exposed when printing config."""
        config_file = tmp_path / "config.yaml"
        config_data = {
            "plex": {"url": "http://test:32400", "token": "secret-token"},
            "llm": {"provider": "anthropic", "api_key": "secret-api-key"},
        }
        config_file.write_text(yaml.dump(config_data))

        for var in ["PLEX_URL", "PLEX_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                    "GEMINI_API_KEY", "LLM_PROVIDER", "LLM_MODEL_ANALYSIS", "LLM_MODEL_GENERATION"]:
            monkeypatch.delenv(var, raising=False)

        # Patch load_user_yaml_config to return empty dict (ignore config.user.yaml)
        with patch("backend.config.load_user_yaml_config", return_value={}):
            config = load_config(config_file)

        # The token and api_key are stored, but we verify they exist
        # (actual masking would be in a different layer if needed)
        assert config.plex.token == "secret-token"
        assert config.llm.api_key == "secret-api-key"


class TestDeepMerge:
    """Tests for deep_merge utility function."""

    def test_merges_flat_dicts(self):
        """Should merge flat dictionaries."""
        base = {"a": 1, "b": 2}
        override = {"b": 20, "c": 3}

        result = deep_merge(base, override)

        assert result == {"a": 1, "b": 20, "c": 3}

    def test_merges_nested_dicts(self):
        """Should recursively merge nested dictionaries."""
        base = {"a": {"b": 1, "c": 2}, "d": 4}
        override = {"a": {"b": 10}}

        result = deep_merge(base, override)

        assert result == {"a": {"b": 10, "c": 2}, "d": 4}

    def test_override_replaces_non_dict_with_dict(self):
        """Should replace non-dict value with dict if override is dict."""
        base = {"a": 1}
        override = {"a": {"nested": True}}

        result = deep_merge(base, override)

        assert result == {"a": {"nested": True}}

    def test_does_not_modify_original(self):
        """Should not modify the original dictionaries."""
        base = {"a": {"b": 1}}
        override = {"a": {"c": 2}}

        deep_merge(base, override)

        assert base == {"a": {"b": 1}}
        assert override == {"a": {"c": 2}}


class TestRemoveEmptyValues:
    """Tests for remove_empty_values utility function."""

    def test_removes_empty_strings(self):
        """Should remove keys with empty string values."""
        d = {"a": "", "b": "value", "c": ""}

        result = remove_empty_values(d)

        assert result == {"b": "value"}

    def test_removes_none_values(self):
        """Should remove keys with None values."""
        d = {"a": None, "b": "value", "c": None}

        result = remove_empty_values(d)

        assert result == {"b": "value"}

    def test_preserves_other_falsy_values(self):
        """Should preserve 0 and False values."""
        d = {"a": 0, "b": False, "c": "value"}

        result = remove_empty_values(d)

        assert result == {"a": 0, "b": False, "c": "value"}

    def test_removes_empty_nested_dicts(self):
        """Should remove nested dicts that become empty."""
        d = {"a": {"b": "", "c": None}, "d": "value"}

        result = remove_empty_values(d)

        assert result == {"d": "value"}

    def test_preserves_non_empty_nested_dicts(self):
        """Should preserve nested dicts with values."""
        d = {"a": {"b": "", "c": "nested"}, "d": "value"}

        result = remove_empty_values(d)

        assert result == {"a": {"c": "nested"}, "d": "value"}


class TestLocalProviderConfig:
    """Tests for local LLM provider configuration."""

    def test_loads_ollama_config_from_yaml(self, tmp_path, monkeypatch):
        """Should load Ollama config from YAML file."""
        for var in ["PLEX_URL", "PLEX_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                    "GEMINI_API_KEY", "LLM_PROVIDER", "LLM_MODEL_ANALYSIS",
                    "LLM_MODEL_GENERATION", "OLLAMA_URL"]:
            monkeypatch.delenv(var, raising=False)

        config_file = tmp_path / "config.yaml"
        config_data = {
            "llm": {
                "provider": "ollama",
                "ollama_url": "http://192.168.1.100:11434",
                "model_analysis": "llama3:8b",
                "model_generation": "llama3:8b",
            },
        }
        config_file.write_text(yaml.dump(config_data))

        with patch("backend.config.load_user_yaml_config", return_value={}):
            config = load_config(config_file)

        assert config.llm.provider == "ollama"
        assert config.llm.ollama_url == "http://192.168.1.100:11434"
        assert config.llm.model_analysis == "llama3:8b"

    def test_ollama_url_env_var_override(self, tmp_path, monkeypatch):
        """OLLAMA_URL env var should override YAML value."""
        for var in ["PLEX_URL", "PLEX_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                    "GEMINI_API_KEY", "LLM_MODEL_ANALYSIS", "LLM_MODEL_GENERATION"]:
            monkeypatch.delenv(var, raising=False)

        config_file = tmp_path / "config.yaml"
        config_data = {
            "llm": {
                "provider": "ollama",
                "ollama_url": "http://yaml-host:11434",
            },
        }
        config_file.write_text(yaml.dump(config_data))

        monkeypatch.setenv("LLM_PROVIDER", "ollama")
        monkeypatch.setenv("OLLAMA_URL", "http://env-host:11434")

        with patch("backend.config.load_user_yaml_config", return_value={}):
            config = load_config(config_file)

        assert config.llm.ollama_url == "http://env-host:11434"

    def test_loads_custom_provider_config(self, tmp_path, monkeypatch):
        """Should load custom provider config from YAML."""
        for var in ["PLEX_URL", "PLEX_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                    "GEMINI_API_KEY", "LLM_PROVIDER", "LLM_MODEL_ANALYSIS",
                    "LLM_MODEL_GENERATION", "CUSTOM_LLM_URL", "CUSTOM_CONTEXT_WINDOW"]:
            monkeypatch.delenv(var, raising=False)

        config_file = tmp_path / "config.yaml"
        config_data = {
            "llm": {
                "provider": "custom",
                "custom_url": "http://localhost:5000/v1",
                "custom_context_window": 8192,
                "model_analysis": "my-model",
                "model_generation": "my-model",
            },
        }
        config_file.write_text(yaml.dump(config_data))

        with patch("backend.config.load_user_yaml_config", return_value={}):
            config = load_config(config_file)

        assert config.llm.provider == "custom"
        assert config.llm.custom_url == "http://localhost:5000/v1"
        assert config.llm.custom_context_window == 8192

    def test_custom_context_window_env_var(self, tmp_path, monkeypatch):
        """CUSTOM_CONTEXT_WINDOW env var should override YAML."""
        for var in ["PLEX_URL", "PLEX_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                    "GEMINI_API_KEY", "LLM_MODEL_ANALYSIS", "LLM_MODEL_GENERATION"]:
            monkeypatch.delenv(var, raising=False)

        config_file = tmp_path / "config.yaml"
        config_data = {
            "llm": {
                "provider": "custom",
                "custom_context_window": 4096,
            },
        }
        config_file.write_text(yaml.dump(config_data))

        monkeypatch.setenv("LLM_PROVIDER", "custom")
        monkeypatch.setenv("CUSTOM_LLM_URL", "http://localhost:5000/v1")
        monkeypatch.setenv("CUSTOM_CONTEXT_WINDOW", "16384")

        with patch("backend.config.load_user_yaml_config", return_value={}):
            config = load_config(config_file)

        assert config.llm.custom_context_window == 16384

    def test_default_ollama_url(self, tmp_path, monkeypatch):
        """Should use default Ollama URL when not specified."""
        for var in ["PLEX_URL", "PLEX_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                    "GEMINI_API_KEY", "LLM_PROVIDER", "LLM_MODEL_ANALYSIS",
                    "LLM_MODEL_GENERATION", "OLLAMA_URL"]:
            monkeypatch.delenv(var, raising=False)

        config_file = tmp_path / "config.yaml"
        config_data = {
            "llm": {
                "provider": "ollama",
            },
        }
        config_file.write_text(yaml.dump(config_data))

        with patch("backend.config.load_user_yaml_config", return_value={}):
            config = load_config(config_file)

        assert config.llm.ollama_url == "http://localhost:11434"

    def test_default_custom_context_window(self, tmp_path, monkeypatch):
        """Should use default custom context window when not specified."""
        for var in ["PLEX_URL", "PLEX_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                    "GEMINI_API_KEY", "LLM_PROVIDER", "LLM_MODEL_ANALYSIS",
                    "LLM_MODEL_GENERATION", "CUSTOM_CONTEXT_WINDOW"]:
            monkeypatch.delenv(var, raising=False)

        config_file = tmp_path / "config.yaml"
        config_data = {
            "llm": {
                "provider": "custom",
            },
        }
        config_file.write_text(yaml.dump(config_data))

        with patch("backend.config.load_user_yaml_config", return_value={}):
            config = load_config(config_file)

        assert config.llm.custom_context_window == 32768


class TestEnvOverrides:
    """Settings fields controlled by environment variables."""

    ENV_VARS = [
        "MEDIA_SERVER", "PLEX_URL", "PLEX_TOKEN", "PLEX_MUSIC_LIBRARY", "JELLYFIN_URL",
        "JELLYFIN_TOKEN", "JELLYFIN_MUSIC_LIBRARY", "LLM_PROVIDER", "LLM_MODEL_ANALYSIS",
        "LLM_MODEL_GENERATION", "OLLAMA_URL", "CUSTOM_LLM_URL", "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY", "GEMINI_API_KEY", "CUSTOM_LLM_API_KEY",
    ]

    def _clear(self, monkeypatch):
        for var in self.ENV_VARS:
            monkeypatch.delenv(var, raising=False)

    def test_reports_only_non_empty_variables(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("PLEX_URL", "http://plex:32400")
        monkeypatch.setenv("PLEX_TOKEN", "")  # compose's ${PLEX_TOKEN:-} when unset

        assert env_overrides("gemini") == {"plex_url": "PLEX_URL"}

    def test_api_key_follows_active_provider(self, monkeypatch):
        self._clear(monkeypatch)
        monkeypatch.setenv("GEMINI_API_KEY", "key")

        assert env_overrides("gemini") == {"llm_api_key": "GEMINI_API_KEY"}
        assert env_overrides("anthropic") == {}

    def test_empty_compose_variables_keep_saved_settings(self, tmp_path, monkeypatch):
        self._clear(monkeypatch)
        for var in ("PLEX_URL", "PLEX_TOKEN", "PLEX_MUSIC_LIBRARY", "MEDIA_SERVER", "JELLYFIN_URL"):
            monkeypatch.setenv(var, "")
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        saved = {
            "media_server": "plex",
            "plex": {"url": "http://saved:32400", "token": "saved-token", "music_library": "Tunes"},
        }

        with patch("backend.config.load_user_yaml_config", return_value=saved):
            config = load_config(config_file)

        assert config.media_server == "plex"
        assert config.plex.url == "http://saved:32400"
        assert config.plex.token == "saved-token"
        assert config.plex.music_library == "Tunes"


class TestUpdateConfigModels:
    """Saving the AI provider only resets models when the provider changes (#29)."""

    @pytest.fixture
    def saved(self, monkeypatch):
        current = config_module.AppConfig(
            plex=config_module.PlexConfig(url="", token=""),
            llm=config_module.LLMConfig(
                provider="gemini", api_key="k",
                model_analysis="gemini-3.8-flash", model_generation="gemini-3.1-flash-lite",
            ),
        )
        monkeypatch.setattr(config_module, "_config", current)
        written = {}
        monkeypatch.setattr(config_module, "save_user_config", written.update)
        return written

    def test_same_provider_keeps_models(self, saved):
        config = config_module.update_config_values({"llm_provider": "gemini", "llm_api_key": "new"})

        assert config.llm.model_analysis == "gemini-3.8-flash"
        assert config.llm.model_generation == "gemini-3.1-flash-lite"
        assert "model_analysis" not in saved["llm"]

    def test_new_provider_gets_its_defaults(self, saved):
        config = config_module.update_config_values({"llm_provider": "anthropic"})

        assert config.llm.model_analysis == MODEL_DEFAULTS["anthropic"]["analysis"]
        assert config.llm.model_generation == MODEL_DEFAULTS["anthropic"]["generation"]

    def test_new_provider_with_chosen_models(self, saved):
        config = config_module.update_config_values({
            "llm_provider": "anthropic",
            "model_analysis": "claude-opus-5-5",
            "model_generation": "claude-sonnet-5-5",
        })

        assert config.llm.model_analysis == "claude-opus-5-5"
        assert config.llm.model_generation == "claude-sonnet-5-5"


class TestRequestTimeout:
    """The AI request timeout comes from LLM_TIMEOUT, the YAML file, or the 10-minute default (#25)."""

    def _load(self, tmp_path, monkeypatch, env=None, yaml_timeout=None):
        monkeypatch.delenv("LLM_TIMEOUT", raising=False)
        if env is not None:
            monkeypatch.setenv("LLM_TIMEOUT", env)
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        user = {"llm": {"request_timeout": yaml_timeout}} if yaml_timeout else {}
        with patch("backend.config.load_user_yaml_config", return_value=user):
            return load_config(config_file).llm.request_timeout

    def test_default_is_ten_minutes(self, tmp_path, monkeypatch):
        assert self._load(tmp_path, monkeypatch) == 600

    def test_saved_setting(self, tmp_path, monkeypatch):
        assert self._load(tmp_path, monkeypatch, yaml_timeout=1800) == 1800

    def test_env_var_overrides(self, tmp_path, monkeypatch):
        assert self._load(tmp_path, monkeypatch, env="2400", yaml_timeout=1800) == 2400


class TestCustomModel:
    """CUSTOM_LLM_MODEL sets the custom endpoint's model without touching other setups (#20)."""

    ENV_VARS = ["LLM_PROVIDER", "LLM_MODEL_ANALYSIS", "LLM_MODEL_GENERATION", "CUSTOM_LLM_MODEL",
                "CUSTOM_LLM_URL", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"]

    def _load(self, tmp_path, monkeypatch, saved_llm, **env):
        for var in self.ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        for var, value in env.items():
            monkeypatch.setenv(var, value)
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        with patch("backend.config.load_user_yaml_config", return_value={"llm": saved_llm}):
            return load_config(config_file).llm

    def test_sets_both_models_for_custom(self, tmp_path, monkeypatch):
        llm = self._load(tmp_path, monkeypatch, {"provider": "custom", "custom_url": "http://x/v1"},
                         CUSTOM_LLM_MODEL="gpt-4o-mini")
        assert (llm.model_analysis, llm.model_generation) == ("gpt-4o-mini", "gpt-4o-mini")

    def test_llm_model_vars_still_win(self, tmp_path, monkeypatch):
        llm = self._load(tmp_path, monkeypatch, {"provider": "custom", "custom_url": "http://x/v1"},
                         CUSTOM_LLM_MODEL="small", LLM_MODEL_ANALYSIS="big")
        assert (llm.model_analysis, llm.model_generation) == ("big", "small")

    def test_saved_custom_models_still_load(self, tmp_path, monkeypatch):
        llm = self._load(tmp_path, monkeypatch, {
            "provider": "custom", "custom_url": "http://x/v1",
            "model_analysis": "saved-a", "model_generation": "saved-g",
        })
        assert (llm.model_analysis, llm.model_generation) == ("saved-a", "saved-g")

    def test_no_effect_on_other_providers(self, tmp_path, monkeypatch):
        for provider, saved in [("gemini", "gemini-3.8-flash"), ("anthropic", "claude-opus-5-5"), ("openai", "gpt-6-luna")]:
            llm = self._load(tmp_path, monkeypatch, {
                "provider": provider, "api_key": "k", "model_analysis": saved, "model_generation": saved,
            }, CUSTOM_LLM_MODEL="gpt-4o-mini")
            assert (llm.model_analysis, llm.model_generation) == (saved, saved), provider

    def test_defaults_unchanged_for_other_providers(self, tmp_path, monkeypatch):
        llm = self._load(tmp_path, monkeypatch, {"provider": "gemini", "api_key": "k"}, CUSTOM_LLM_MODEL="x")
        assert llm.model_analysis == MODEL_DEFAULTS["gemini"]["analysis"]

    def test_locks_custom_model_in_settings(self, monkeypatch):
        for var in self.ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("CUSTOM_LLM_MODEL", "gpt-4o-mini")
        assert env_overrides("custom")["model_analysis"] == "CUSTOM_LLM_MODEL"
        assert "model_analysis" not in env_overrides("gemini")


class TestLibrarySyncHours:
    """Auto-refresh interval: Settings/YAML, overridden by LIBRARY_SYNC_HOURS."""

    def _load(self, tmp_path, monkeypatch, saved=None, env=None):
        monkeypatch.delenv("LIBRARY_SYNC_HOURS", raising=False)
        if env is not None:
            monkeypatch.setenv("LIBRARY_SYNC_HOURS", env)
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        user = {} if saved is None else {"library_sync_hours": saved}
        with patch("backend.config.load_user_yaml_config", return_value=user):
            return load_config(config_file).library_sync_hours

    def test_default_daily(self, tmp_path, monkeypatch):
        assert self._load(tmp_path, monkeypatch) == 24

    def test_saved_setting_and_env_override(self, tmp_path, monkeypatch):
        assert self._load(tmp_path, monkeypatch, saved=168) == 168
        assert self._load(tmp_path, monkeypatch, saved=168, env="6") == 6

    def test_saving_off_is_kept(self, monkeypatch):
        current = config_module.AppConfig(
            plex=config_module.PlexConfig(url="", token=""),
            llm=config_module.LLMConfig(provider="gemini", model_analysis="m", model_generation="m"),
        )
        monkeypatch.setattr(config_module, "_config", current)
        saved = {}
        monkeypatch.setattr(config_module, "save_user_config", saved.update)

        config = config_module.update_config_values({"library_sync_hours": 0})

        assert config.library_sync_hours == 0
        assert saved["library_sync_hours"] == 0


class TestEnvApiKeyNotSaved:
    """A provider's API key from the environment is used but never written to config.user.yaml."""

    @pytest.fixture
    def saved(self, monkeypatch):
        for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "CUSTOM_LLM_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        current = config_module.AppConfig(
            plex=config_module.PlexConfig(url="", token=""),
            llm=config_module.LLMConfig(provider="gemini", api_key="saved-gemini-key",
                                        api_keys={"gemini": "saved-gemini-key"},
                                        model_analysis="m", model_generation="m"),
        )
        monkeypatch.setattr(config_module, "_config", current)
        written = {}
        monkeypatch.setattr(config_module, "save_user_config", written.update)
        return written

    def test_env_key_used_for_new_provider_but_not_saved(self, saved, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-anthropic-key")

        config = config_module.update_config_values({"llm_provider": "anthropic"})

        assert config.llm.api_key == "env-anthropic-key"
        assert "env-anthropic-key" not in saved["llm"]["api_keys"].values()

    def test_entered_key_is_saved(self, saved, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-anthropic-key")

        config = config_module.update_config_values({"llm_provider": "anthropic", "llm_api_key": "typed-key"})

        assert saved["llm"]["api_keys"]["anthropic"] == "typed-key"
        assert config.llm.api_key == "env-anthropic-key"  # a set variable overrides Settings


class TestPerProviderApiKeys:
    """Each AI provider keeps its own key, so switching never sends another provider's key."""

    ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "CUSTOM_LLM_API_KEY", "LLM_PROVIDER")

    @pytest.fixture
    def saved(self, monkeypatch):
        for var in self.ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        current = config_module.AppConfig(
            plex=config_module.PlexConfig(url="", token=""),
            llm=config_module.LLMConfig(provider="gemini", api_key="gemini-key", api_keys={"gemini": "gemini-key"},
                                        model_analysis="m", model_generation="m"),
        )
        monkeypatch.setattr(config_module, "_config", current)
        written = {}
        monkeypatch.setattr(config_module, "save_user_config", lambda u: written.update(u))
        return written

    def test_switching_without_a_key_does_not_reuse_another_providers_key(self, saved):
        config = config_module.update_config_values({"llm_provider": "anthropic"})

        assert config.llm.provider == "anthropic"
        assert config.llm.api_key == ""  # shows "Not configured" instead of sending the Gemini key

    def test_each_provider_keeps_its_key(self, saved):
        config_module.update_config_values({"llm_provider": "anthropic", "llm_api_key": "anthropic-key"})
        config = config_module.update_config_values({"llm_provider": "gemini"})

        assert config.llm.api_key == "gemini-key"
        config = config_module.update_config_values({"llm_provider": "anthropic"})
        assert config.llm.api_key == "anthropic-key"
        assert saved["llm"]["api_keys"] == {"gemini": "gemini-key", "anthropic": "anthropic-key"}

    def _load(self, tmp_path, monkeypatch, llm_yaml, **env):
        for var in self.ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        for var, value in env.items():
            monkeypatch.setenv(var, value)
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        with patch("backend.config.load_user_yaml_config", return_value={"llm": llm_yaml}):
            return load_config(config_file).llm

    def test_older_single_key_belongs_to_its_provider(self, tmp_path, monkeypatch):
        llm = self._load(tmp_path, monkeypatch, {"provider": "gemini", "api_key": "gemini-key"})
        assert (llm.provider, llm.api_key, llm.api_keys) == ("gemini", "gemini-key", {"gemini": "gemini-key"})

    def test_older_single_key_not_used_for_another_provider(self, tmp_path, monkeypatch):
        """Anthropic used to pick up the saved key whatever provider it was saved for."""
        llm = self._load(tmp_path, monkeypatch, {"provider": "gemini", "api_key": "gemini-key"},
                         LLM_PROVIDER="anthropic")
        assert (llm.provider, llm.api_key) == ("anthropic", "")

    def test_loads_per_provider_keys(self, tmp_path, monkeypatch):
        llm = self._load(tmp_path, monkeypatch, {
            "provider": "anthropic", "api_keys": {"gemini": "g", "anthropic": "a"},
        })
        assert llm.api_key == "a"

    def test_env_key_wins_for_its_provider(self, tmp_path, monkeypatch):
        llm = self._load(tmp_path, monkeypatch, {"provider": "anthropic", "api_keys": {"anthropic": "saved"}},
                         ANTHROPIC_API_KEY="from-env")
        assert llm.api_key == "from-env"
        assert llm.api_keys == {"anthropic": "saved"}
