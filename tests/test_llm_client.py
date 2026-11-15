"""Tests for LLM client."""

from unittest.mock import MagicMock, patch

import pytest


class TestLLMClientInitialization:
    """Tests for LLM client initialization."""

    def test_anthropic_client_init(self, mocker):
        """Should initialize Anthropic client correctly."""
        from backend.llm_client import LLMClient
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="anthropic",
            api_key="sk-ant-test-key",
            model_analysis="claude-sonnet-4-5-latest",
            model_generation="claude-haiku-4-5-latest",
        )

        with patch("backend.llm_client.anthropic") as mock_anthropic:
            client = LLMClient(config)
            mock_anthropic.Anthropic.assert_called_once_with(api_key="sk-ant-test-key", timeout=600)
            assert client.provider == "anthropic"

    def test_openai_client_init(self, mocker):
        """Should initialize OpenAI client correctly."""
        from backend.llm_client import LLMClient
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="openai",
            api_key="sk-test-key",
            model_analysis="gpt-4.1",
            model_generation="gpt-4.1-mini",
        )

        with patch("backend.llm_client.openai") as mock_openai:
            client = LLMClient(config)
            mock_openai.OpenAI.assert_called_once_with(api_key="sk-test-key", timeout=600)
            assert client.provider == "openai"

    def test_invalid_api_key_anthropic(self, mocker):
        """Should handle invalid Anthropic API key."""
        from backend.llm_client import LLMClient
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="anthropic",
            api_key="invalid-key",
            model_analysis="claude-sonnet-4-5-latest",
            model_generation="claude-haiku-4-5-latest",
        )

        with patch("backend.llm_client.anthropic") as mock_anthropic:
            mock_anthropic.Anthropic.return_value = MagicMock()
            client = LLMClient(config)
            # Client should be created; validation happens on first API call
            assert client.provider == "anthropic"

    def test_invalid_api_key_openai(self, mocker):
        """Should handle invalid OpenAI API key."""
        from backend.llm_client import LLMClient
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="openai",
            api_key="invalid-key",
            model_analysis="gpt-4.1",
            model_generation="gpt-4.1-mini",
        )

        with patch("backend.llm_client.openai") as mock_openai:
            mock_openai.OpenAI.return_value = MagicMock()
            client = LLMClient(config)
            # Client should be created; validation happens on first API call
            assert client.provider == "openai"


class TestLLMClientAnalyze:
    """Tests for LLM analysis calls."""

    def test_analyze_uses_analysis_model(self, mocker):
        """Should use analysis model for analyze calls."""
        from backend.llm_client import LLMClient
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="anthropic",
            api_key="test-key",
            model_analysis="claude-sonnet-4-5-latest",
            model_generation="claude-haiku-4-5-latest",
        )

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"result": "test"}')]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        with patch("backend.llm_client.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.Anthropic.return_value = mock_client

            client = LLMClient(config)
            client.analyze("test prompt", "system prompt")

            # Verify the analysis model was used
            call_args = mock_client.messages.create.call_args
            assert call_args.kwargs["model"] == "claude-sonnet-4-5-latest"


class TestLLMClientGenerate:
    """Tests for LLM generation calls."""

    def test_generate_uses_generation_model(self, mocker):
        """Should use generation model for generate calls."""
        from backend.llm_client import LLMClient
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="anthropic",
            api_key="test-key",
            model_analysis="claude-sonnet-4-5-latest",
            model_generation="claude-haiku-4-5-latest",
            smart_generation=False,
        )

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='[{"artist": "Test", "title": "Song"}]')]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        with patch("backend.llm_client.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.Anthropic.return_value = mock_client

            client = LLMClient(config)
            client.generate("test prompt", "system prompt")

            # Verify the generation model was used
            call_args = mock_client.messages.create.call_args
            assert call_args.kwargs["model"] == "claude-haiku-4-5-latest"

    def test_smart_generation_uses_analysis_model(self, mocker):
        """Should use analysis model when smart_generation is enabled."""
        from backend.llm_client import LLMClient
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="anthropic",
            api_key="test-key",
            model_analysis="claude-sonnet-4-5-latest",
            model_generation="claude-haiku-4-5-latest",
            smart_generation=True,
        )

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='[{"artist": "Test", "title": "Song"}]')]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50

        with patch("backend.llm_client.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.Anthropic.return_value = mock_client

            client = LLMClient(config)
            client.generate("test prompt", "system prompt")

            # Verify the analysis model was used for generation
            call_args = mock_client.messages.create.call_args
            assert call_args.kwargs["model"] == "claude-sonnet-4-5-latest"


class TestLLMClientTokenTracking:
    """Tests for token and cost tracking."""

    def test_tracks_tokens_anthropic(self, mocker):
        """Should track tokens for Anthropic calls."""
        from backend.llm_client import LLMClient
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="anthropic",
            api_key="test-key",
            model_analysis="claude-sonnet-4-5-latest",
            model_generation="claude-haiku-4-5-latest",
        )

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"result": "test"}')]
        mock_response.usage.input_tokens = 150
        mock_response.usage.output_tokens = 75

        with patch("backend.llm_client.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = mock_response
            mock_anthropic.Anthropic.return_value = mock_client

            client = LLMClient(config)
            result = client.analyze("test prompt", "system prompt")

            assert result.input_tokens == 150
            assert result.output_tokens == 75
            assert result.total_tokens == 225

    def test_tracks_tokens_openai(self, mocker):
        """Should track tokens for OpenAI calls."""
        from backend.llm_client import LLMClient
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="openai",
            api_key="test-key",
            model_analysis="gpt-4.1",
            model_generation="gpt-4.1-mini",
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"result": "test"}'))]
        mock_response.usage.prompt_tokens = 150
        mock_response.usage.completion_tokens = 75

        with patch("backend.llm_client.openai") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_client

            client = LLMClient(config)
            result = client.analyze("test prompt", "system prompt")

            assert result.input_tokens == 150
            assert result.output_tokens == 75
            assert result.total_tokens == 225


class TestOllamaProvider:
    """Tests for Ollama provider."""

    def test_ollama_client_init_no_client_created(self):
        """Ollama provider should not create a persistent client."""
        from backend.llm_client import LLMClient
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="ollama",
            api_key="",
            model_analysis="llama3:8b",
            model_generation="llama3:8b",
            ollama_url="http://localhost:11434",
        )

        client = LLMClient(config)
        assert client.provider == "ollama"
        assert client._client is None  # Ollama uses httpx directly

    def test_complete_ollama_success(self, mocker):
        """Should make completion request to Ollama API."""
        from backend.llm_client import LLMClient
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="ollama",
            api_key="",
            model_analysis="llama3:8b",
            model_generation="llama3:8b",
            ollama_url="http://localhost:11434",
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": '{"result": "test"}',
            "prompt_eval_count": 100,
            "eval_count": 50,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("backend.llm_client.httpx.Client") as mock_httpx:
            mock_client_instance = MagicMock()
            mock_client_instance.post.return_value = mock_response
            mock_client_instance.__enter__ = MagicMock(return_value=mock_client_instance)
            mock_client_instance.__exit__ = MagicMock(return_value=False)
            mock_httpx.return_value = mock_client_instance

            client = LLMClient(config)
            result = client._complete_ollama("test prompt", "system prompt", "llama3:8b")

            assert result.content == '{"result": "test"}'
            assert result.input_tokens == 100
            assert result.output_tokens == 50
            assert result.model == "llama3:8b"

            # Verify correct endpoint was called
            mock_client_instance.post.assert_called_once()
            call_args = mock_client_instance.post.call_args
            assert "/api/generate" in call_args[0][0]

    def _ollama_call(self, response_text, context_window=40960):
        from backend.llm_client import LLMClient
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="ollama", model_analysis="qwen3:8b", model_generation="qwen3:8b",
            ollama_context_window=context_window,
        )
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": response_text, "prompt_eval_count": 1, "eval_count": 1}
        with patch("backend.llm_client.httpx.Client") as mock_httpx:
            http = mock_httpx.return_value.__enter__.return_value
            http.post.return_value = mock_response
            result = LLMClient(config)._complete_ollama("prompt", "system", "qwen3:8b")
        return result, http.post.call_args.kwargs["json"]

    def test_ollama_uses_configured_timeout(self):
        """The Ollama call waits as long as the configured request timeout (#25)."""
        from backend.llm_client import LLMClient
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="ollama", model_analysis="m", model_generation="m", request_timeout=1800,
        )
        with patch("backend.llm_client.httpx.Client") as mock_httpx:
            http = mock_httpx.return_value.__enter__.return_value
            http.post.return_value.json.return_value = {"response": "[]"}
            LLMClient(config)._complete_ollama("prompt", "system", "m")

        assert mock_httpx.call_args.kwargs["timeout"] == 1800

    def test_timeout_explains_the_setting(self):
        import httpx
        from backend.llm_client import LLMClient
        from backend.models import LLMConfig

        for timeout, waited in [(90, "90 seconds"), (900, "15 minutes")]:
            config = LLMConfig(provider="ollama", model_analysis="m", model_generation="m", request_timeout=timeout)
            client = LLMClient(config)
            with patch.object(client, "_complete_ollama", side_effect=httpx.ReadTimeout("timed out")):
                with pytest.raises(RuntimeError, match=f"within {waited}. Raise Request Timeout"):
                    client.analyze("prompt", "system")

    def test_empty_model_gives_clear_error(self):
        """An empty model name used to reach the provider as model= (#20)."""
        from backend.llm_client import LLMClient
        from backend.models import LLMConfig

        config = LLMConfig(provider="custom", custom_url="http://x/v1", model_analysis="", model_generation="")
        with patch("backend.llm_client.openai"):
            client = LLMClient(config)
            with pytest.raises(RuntimeError, match="No AI model is set"):
                client.generate("prompt", "system")
            client._client.chat.completions.create.assert_not_called()

    def test_missing_cloud_key_gives_clear_error(self):
        from backend.llm_client import LLMClient
        from backend.models import LLMConfig

        config = LLMConfig(provider="anthropic", api_key="", model_analysis="claude-sonnet-5-5",
                           model_generation="claude-haiku-4-5")
        with patch("backend.llm_client.anthropic"):
            client = LLMClient(config)
            with pytest.raises(RuntimeError, match="No API key is set for Anthropic. Add one in Settings."):
                client.analyze("prompt", "system")
            client._client.messages.create.assert_not_called()

    def test_cloud_clients_use_configured_timeout(self):
        from backend.llm_client import LLMClient
        from backend.models import LLMConfig

        def config(provider):
            return LLMConfig(provider=provider, api_key="k", model_analysis="m",
                             model_generation="m", request_timeout=900)

        with patch("backend.llm_client.anthropic") as mock_anthropic:
            LLMClient(config("anthropic"))
        assert mock_anthropic.Anthropic.call_args.kwargs["timeout"] == 900

        with patch("backend.llm_client.genai") as mock_genai:
            LLMClient(config("gemini"))
        assert mock_genai.Client.call_args.kwargs["http_options"].timeout == 900_000

    def test_local_responses_cost_nothing(self):
        """Ollama and custom endpoints showed the placeholder price for unknown models."""
        from backend.llm_client import LLMClient, LLMResponse
        from backend.models import LLMConfig

        reply = LLMResponse(content="[]", input_tokens=100_000, output_tokens=10_000, model="qwen3:8b")
        for provider in ("ollama", "custom"):
            config = LLMConfig(provider=provider, model_analysis="qwen3:8b", model_generation="qwen3:8b")
            client = LLMClient(config)
            method = "_complete_ollama" if provider == "ollama" else "_complete_openai"
            with patch.object(client, method, return_value=reply):
                assert client.generate("prompt", "system").estimated_cost() == 0.0, provider

    def test_cloud_responses_still_priced(self):
        from backend.llm_client import LLMClient, LLMResponse
        from backend.models import LLMConfig

        config = LLMConfig(provider="gemini", api_key="k", model_analysis="gemini-3.5-flash-lite",
                           model_generation="gemini-3.5-flash-lite")
        reply = LLMResponse(content="[]", input_tokens=1_000_000, output_tokens=0, model="gemini-3.5-flash-lite")
        with patch("backend.llm_client.genai"):
            client = LLMClient(config)
        with patch.object(client, "_complete_gemini", return_value=reply):
            assert client.generate("prompt", "system").estimated_cost() == 0.30

    def test_ollama_sends_context_window(self):
        """Without num_ctx Ollama truncates prompts sized for the full context (#24)."""
        _, body = self._ollama_call("[]", context_window=40960)
        assert body["options"] == {"num_ctx": 40960}

    def test_ollama_strips_inline_thinking(self):
        """Older Ollama versions return a reasoning model's thinking inline."""
        result, _ = self._ollama_call('<think>\nPicking tracks...\n</think>\n[{"title": "A"}]')
        assert result.content == '[{"title": "A"}]'

    def test_complete_dispatch_routes_to_ollama(self, mocker):
        """Should route 'ollama' provider to _complete_ollama method."""
        from backend.llm_client import LLMClient
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="ollama",
            api_key="",
            model_analysis="llama3:8b",
            model_generation="llama3:8b",
            ollama_url="http://localhost:11434",
        )

        client = LLMClient(config)

        # Mock the _complete_ollama method
        mock_ollama = mocker.patch.object(client, "_complete_ollama")
        mock_ollama.return_value = MagicMock(content="test")

        client._complete("test prompt", "system prompt", "llama3:8b")

        mock_ollama.assert_called_once_with("test prompt", "system prompt", "llama3:8b")


class TestCustomProvider:
    """Tests for custom OpenAI-compatible provider."""

    def test_custom_client_init_creates_openai_client(self):
        """Custom provider should create OpenAI client with custom base_url."""
        from backend.llm_client import LLMClient
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="custom",
            api_key="",
            model_analysis="my-model",
            model_generation="my-model",
            custom_url="http://localhost:5000/v1",
            custom_context_window=8192,
        )

        with patch("backend.llm_client.openai") as mock_openai:
            client = LLMClient(config)
            mock_openai.OpenAI.assert_called_once_with(
                api_key="not-needed",
                base_url="http://localhost:5000/v1",
                timeout=600,
            )
            assert client.provider == "custom"

    def test_complete_custom_success(self, mocker):
        """Should make completion request to custom endpoint via _complete_openai."""
        from backend.llm_client import LLMClient
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="custom",
            api_key="",
            model_analysis="my-model",
            model_generation="my-model",
            custom_url="http://localhost:5000/v1",
            custom_context_window=8192,
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"result": "test"}'))]
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50

        with patch("backend.llm_client.openai") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.OpenAI.return_value = mock_client

            client = LLMClient(config)
            result = client._complete_openai("test prompt", "system prompt", "my-model")

            assert result.content == '{"result": "test"}'
            assert result.input_tokens == 100
            assert result.output_tokens == 50
            assert result.model == "my-model"

    def test_complete_dispatch_routes_to_custom(self, mocker):
        """Should route 'custom' provider to _complete_openai method."""
        from backend.llm_client import LLMClient
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="custom",
            api_key="",
            model_analysis="my-model",
            model_generation="my-model",
            custom_url="http://localhost:5000/v1",
        )

        with patch("backend.llm_client.openai"):
            client = LLMClient(config)

        mock_openai_method = mocker.patch.object(client, "_complete_openai")
        mock_openai_method.return_value = MagicMock(content="test")

        client._complete("test prompt", "system prompt", "my-model")

        mock_openai_method.assert_called_once_with("test prompt", "system prompt", "my-model")


class TestLocalProviderCosts:
    """Tests for local provider cost calculations."""

    def test_ollama_cost_is_zero(self):
        """Ollama provider should have zero cost."""
        from backend.llm_client import get_model_cost
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="ollama",
            api_key="",
            model_analysis="llama3:8b",
            model_generation="llama3:8b",
        )

        costs = get_model_cost("llama3:8b", config)
        assert costs["input"] == 0.0
        assert costs["output"] == 0.0

    def test_custom_cost_is_zero(self):
        """Custom provider should have zero cost."""
        from backend.llm_client import get_model_cost
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="custom",
            api_key="",
            model_analysis="my-model",
            model_generation="my-model",
            custom_url="http://localhost:5000/v1",
        )

        costs = get_model_cost("my-model", config)
        assert costs["input"] == 0.0
        assert costs["output"] == 0.0

    def test_estimate_cost_is_zero_for_local(self):
        """estimate_cost_for_model should return 0 for local providers."""
        from backend.llm_client import estimate_cost_for_model
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="ollama",
            api_key="",
            model_analysis="llama3:8b",
            model_generation="llama3:8b",
        )

        cost = estimate_cost_for_model("llama3:8b", 10000, 5000, config)
        assert cost == 0.0


class TestLocalProviderContextLimits:
    """Tests for local provider context limit lookups."""

    def test_custom_context_from_config(self):
        """Custom provider should use context window from config."""
        from backend.llm_client import get_model_context_limit
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="custom",
            api_key="",
            model_analysis="my-model",
            model_generation="my-model",
            custom_url="http://localhost:5000/v1",
            custom_context_window=16384,
        )

        limit = get_model_context_limit("my-model", config)
        assert limit == 16384

    def test_ollama_default_context(self):
        """Ollama provider should use default 32768 context."""
        from backend.llm_client import get_model_context_limit
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="ollama",
            api_key="",
            model_analysis="llama3:8b",
            model_generation="llama3:8b",
        )

        limit = get_model_context_limit("llama3:8b", config)
        assert limit == 32768

    def test_ollama_context_from_config(self):
        """Ollama provider should use ollama_context_window from config."""
        from backend.llm_client import get_model_context_limit
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="ollama",
            api_key="",
            model_analysis="qwen3:8b",
            model_generation="qwen3:8b",
            ollama_context_window=40960,  # Detected from model info
        )

        limit = get_model_context_limit("qwen3:8b", config)
        assert limit == 40960

    def test_max_tracks_for_custom_model(self):
        """Should calculate max tracks based on custom context window."""
        from backend.llm_client import get_max_tracks_for_model
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="custom",
            api_key="",
            model_analysis="my-model",
            model_generation="my-model",
            custom_url="http://localhost:5000/v1",
            custom_context_window=16384,
        )

        max_tracks = get_max_tracks_for_model("my-model", config=config)
        # (16384 * 0.9 - 1000) / 50 = ~274 tracks
        assert max_tracks > 200  # Should be reasonable number based on 16k context


class TestOllamaModelInfoParsing:
    """Tests for Ollama model info context window parsing."""

    def test_context_from_model_info(self):
        """Should extract context_length from model_info field."""
        from unittest.mock import MagicMock
        from backend.llm_client import get_ollama_model_info

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model_info": {
                "general.architecture": "qwen3",
                "qwen3.context_length": 40960,
            },
            "details": {"parameter_size": "8B"},
            "parameters": "",
            "modelfile": "",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("backend.llm_client.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            result = get_ollama_model_info("http://localhost:11434", "qwen3:8b")

        assert result is not None
        assert result.context_window == 40960
        assert result.parameter_size == "8B"

    def test_num_ctx_overrides_model_info(self):
        """Explicit num_ctx in parameters should override model_info."""
        from unittest.mock import MagicMock
        from backend.llm_client import get_ollama_model_info

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model_info": {
                "general.architecture": "llama",
                "llama.context_length": 8192,
            },
            "details": {},
            "parameters": "num_ctx 4096",  # User-configured override
            "modelfile": "",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("backend.llm_client.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            result = get_ollama_model_info("http://localhost:11434", "llama3:8b")

        assert result is not None
        assert result.context_window == 4096  # num_ctx takes precedence

    def test_fallback_to_default_when_no_context_info(self):
        """Should use 32768 default when no context info available."""
        from unittest.mock import MagicMock
        from backend.llm_client import get_ollama_model_info

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model_info": {},
            "details": {},
            "parameters": "",
            "modelfile": "",
        }
        mock_response.raise_for_status = MagicMock()

        with patch("backend.llm_client.httpx.Client") as mock_client:
            mock_client.return_value.__enter__.return_value.post.return_value = mock_response
            result = get_ollama_model_info("http://localhost:11434", "unknown-model")

        assert result is not None
        assert result.context_window == 32768


class TestJsonParsing:
    """Tests for JSON parsing from LLM responses."""

    def test_parse_json_with_extra_text(self):
        """Should handle LLM responses with extra text after JSON array."""
        from backend.llm_client import LLMClient, LLMResponse
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="anthropic",
            api_key="test",
            model_analysis="test",
            model_generation="test",
        )

        with patch("backend.llm_client.anthropic"):
            client = LLMClient(config)

        # Simulate LLM adding explanation after JSON
        response = LLMResponse(
            content='[{"artist": "Test", "title": "Song"}]\n\nThis is a great selection because...',
            input_tokens=100,
            output_tokens=50,
            model="test",
        )

        result = client.parse_json_response(response)
        assert result == [{"artist": "Test", "title": "Song"}]

    def test_parse_json_with_nested_objects(self):
        """Should handle nested JSON objects with extra text."""
        from backend.llm_client import LLMClient, LLMResponse
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="anthropic",
            api_key="test",
            model_analysis="test",
            model_generation="test",
        )

        with patch("backend.llm_client.anthropic"):
            client = LLMClient(config)

        response = LLMResponse(
            content='{"title": "Test", "tracks": [{"name": "Song"}]} Extra text here',
            input_tokens=100,
            output_tokens=50,
            model="test",
        )

        result = client.parse_json_response(response)
        assert result == {"title": "Test", "tracks": [{"name": "Song"}]}

    def test_extract_json_bounds_with_strings_containing_brackets(self):
        """Should handle JSON with brackets inside strings."""
        from backend.llm_client import LLMClient
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="anthropic",
            api_key="test",
            model_analysis="test",
            model_generation="test",
        )

        with patch("backend.llm_client.anthropic"):
            client = LLMClient(config)

        content = '[{"reason": "This track [live] is great"}] Some explanation'
        result = client._extract_json_bounds(content)
        assert result == '[{"reason": "This track [live] is great"}]'

    def test_repair_unescaped_quotes_in_string(self):
        """Should repair unescaped double quotes inside string values."""
        from backend.llm_client import LLMClient, LLMResponse
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="anthropic",
            api_key="test",
            model_analysis="test",
            model_generation="test",
        )

        with patch("backend.llm_client.anthropic"):
            client = LLMClient(config)

        # LLM put quotes around song name inside reason field
        response = LLMResponse(
            content='[{"artist": "Phoenix", "title": "Fences", "reason": "The song "Fences" is great"}]',
            input_tokens=100,
            output_tokens=50,
            model="test",
        )

        result = client.parse_json_response(response)
        assert result[0]["artist"] == "Phoenix"
        assert result[0]["title"] == "Fences"
        assert "Fences" in result[0]["reason"]

    def test_repair_multiple_unescaped_quotes(self):
        """Should repair multiple unescaped quotes in one string."""
        from backend.llm_client import LLMClient, LLMResponse
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="anthropic",
            api_key="test",
            model_analysis="test",
            model_generation="test",
        )

        with patch("backend.llm_client.anthropic"):
            client = LLMClient(config)

        response = LLMResponse(
            content='[{"reason": "Both "Song A" and "Song B" are perfect"}]',
            input_tokens=100,
            output_tokens=50,
            model="test",
        )

        result = client.parse_json_response(response)
        assert "Song A" in result[0]["reason"]
        assert "Song B" in result[0]["reason"]

    def test_repair_json_with_newlines_in_strings(self):
        """Should handle newlines inside string values."""
        from backend.llm_client import LLMClient, LLMResponse
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="anthropic",
            api_key="test",
            model_analysis="test",
            model_generation="test",
        )

        with patch("backend.llm_client.anthropic"):
            client = LLMClient(config)

        # Note: The actual newline character in the string
        response = LLMResponse(
            content='[{"reason": "Line one\nLine two"}]',
            input_tokens=100,
            output_tokens=50,
            model="test",
        )

        result = client.parse_json_response(response)
        assert "Line one" in result[0]["reason"]


class TestModelCatalog:
    """Tests for the cloud model catalog."""

    def test_defaults_are_in_catalog(self):
        """Every cloud provider default should have a known price and context window."""
        from backend.config import MODEL_DEFAULTS
        from backend.model_catalog import CATALOG_BY_ID

        for provider in ("anthropic", "openai", "gemini"):
            for role in ("analysis", "generation"):
                model = MODEL_DEFAULTS[provider][role]
                assert model in CATALOG_BY_ID, model
                assert CATALOG_BY_ID[model].provider == provider

    def test_catalog_yaml_types(self):
        """YAML values should load as the declared types."""
        from backend.model_catalog import MODEL_CATALOG

        for m in MODEL_CATALOG:
            assert isinstance(m.context, int), m.id
            assert isinstance(m.input_cost, (int, float)), m.id
            assert isinstance(m.output_cost, (int, float)), m.id

    def test_catalog_rejects_unknown_fields(self, tmp_path):
        """A typo in a field name should fail loudly, not be ignored."""
        from backend.model_catalog import load_catalog

        path = tmp_path / "catalog.yaml"
        path.write_text(
            "gemini:\n  m:\n    label: M\n    context: 1000\n"
            "    input_cost: 1\n    output_cost: 2\n    inptu_cost: 3\n"
        )
        with pytest.raises(TypeError):
            load_catalog(path)

    def test_catalog_ids_unique(self):
        """Model IDs should not repeat."""
        from backend.model_catalog import CATALOG_BY_ID, MODEL_CATALOG

        assert len(CATALOG_BY_ID) == len(MODEL_CATALOG)

    def test_new_models_get_full_context(self):
        """1M-context models should fit far more tracks than the 128K fallback."""
        from backend.llm_client import get_max_tracks_for_model

        assert get_max_tracks_for_model("gemini-3.5-flash-lite") > 20_000
        assert get_max_tracks_for_model("claude-sonnet-5-5") > 20_000
        assert get_max_tracks_for_model("claude-haiku-4-5") < 5_000

    def test_long_context_pricing(self):
        """OpenAI requests above 272K input tokens use the long-context rate."""
        from backend.llm_client import estimate_cost_for_model

        short = estimate_cost_for_model("gpt-6-luna", 200_000, 0)
        long = estimate_cost_for_model("gpt-6-luna", 400_000, 0)
        assert short == 200_000 / 1_000_000 * 0.10
        assert long == 400_000 / 1_000_000 * 0.20

    def test_flat_pricing_without_threshold(self):
        """Models without a long-context tier keep their base rate."""
        from backend.llm_client import estimate_cost_for_model

        cost = estimate_cost_for_model("gemini-3.5-flash-lite", 480_000, 0)
        assert cost == 480_000 / 1_000_000 * 0.30


class TestProviderRequestParams:
    """Tests for provider-specific request parameters on newer models."""

    def _anthropic_response(self, blocks, stop_reason="end_turn"):
        response = MagicMock()
        response.content = blocks
        response.stop_reason = stop_reason
        response.usage.input_tokens = 10
        response.usage.output_tokens = 5
        return response

    def _block(self, block_type, text=""):
        block = MagicMock()
        block.type = block_type
        block.text = text
        return block

    def test_anthropic_skips_thinking_blocks(self):
        """Text should come from text blocks even when a thinking block is first."""
        from backend.llm_client import LLMClient
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="anthropic", api_key="k",
            model_analysis="claude-sonnet-5-5", model_generation="claude-haiku-4-5",
        )
        response = self._anthropic_response(
            [self._block("thinking"), self._block("text", '{"ok": true}')]
        )

        with patch("backend.llm_client.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = response
            mock_anthropic.Anthropic.return_value = mock_client

            result = LLMClient(config).analyze("prompt", "system")

        assert result.content == '{"ok": true}'
        kwargs = mock_client.messages.create.call_args.kwargs
        assert kwargs["output_config"] == {"effort": "medium"}

    def test_anthropic_haiku_sends_no_effort(self):
        """Haiku 4.5 rejects effort, so it must not be sent."""
        from backend.llm_client import LLMClient
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="anthropic", api_key="k",
            model_analysis="claude-haiku-4-5", model_generation="claude-haiku-4-5",
        )

        with patch("backend.llm_client.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = self._anthropic_response(
                [self._block("text", "{}")]
            )
            mock_anthropic.Anthropic.return_value = mock_client

            LLMClient(config).analyze("prompt", "system")

        assert "output_config" not in mock_client.messages.create.call_args.kwargs

    def test_anthropic_refusal_raises(self):
        """A refusal should surface as an error instead of empty content."""
        from backend.llm_client import LLMClient
        from backend.models import LLMConfig

        config = LLMConfig(
            provider="anthropic", api_key="k",
            model_analysis="claude-sonnet-5-5", model_generation="claude-haiku-4-5",
        )

        with patch("backend.llm_client.anthropic") as mock_anthropic:
            mock_client = MagicMock()
            mock_client.messages.create.return_value = self._anthropic_response([], "refusal")
            mock_anthropic.Anthropic.return_value = mock_client

            with pytest.raises(RuntimeError, match="declined"):
                LLMClient(config).analyze("prompt", "system")

    def _openai_call_kwargs(self, provider, model):
        from backend.llm_client import LLMClient
        from backend.models import LLMConfig

        config = LLMConfig(
            provider=provider, api_key="k", model_analysis=model, model_generation=model,
            custom_url="http://localhost:5000/v1",
        )
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="{}"))]

        with patch("backend.llm_client.openai") as mock_openai:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = response
            mock_openai.OpenAI.return_value = mock_client

            LLMClient(config).analyze("prompt", "system")

        return mock_client.chat.completions.create.call_args.kwargs

    def test_openai_reasoning_model_params(self):
        """OpenAI reasoning models use max_completion_tokens and reasoning_effort."""
        kwargs = self._openai_call_kwargs("openai", "gpt-6-luna")

        assert kwargs["max_completion_tokens"] == 16000
        assert kwargs["reasoning_effort"] == "low"
        assert "max_tokens" not in kwargs

    def test_openai_legacy_model_has_no_effort(self):
        """Non-reasoning OpenAI models don't get reasoning_effort."""
        kwargs = self._openai_call_kwargs("openai", "gpt-4.1-mini")

        assert kwargs["max_completion_tokens"] == 16000
        assert "reasoning_effort" not in kwargs

    def test_custom_provider_keeps_max_tokens(self):
        """Custom OpenAI-compatible servers still get max_tokens."""
        kwargs = self._openai_call_kwargs("custom", "local-model")

        assert kwargs["max_tokens"] == 8192
        assert "max_completion_tokens" not in kwargs
