"""Tests for msgflux.models.providers.google module."""

import json
from unittest.mock import MagicMock, Mock, patch

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator


# ============================================================
# 1. Import & Registration
# ============================================================


class TestGoogleProviderImport:
    """Test Google provider import and registration."""

    def test_google_import_available(self):
        """Test that Google provider imports correctly when deps are available."""
        try:
            from msgflux.models.providers.google import GoogleChatCompletion

            assert GoogleChatCompletion is not None
        except ImportError:
            pytest.skip("google-genai not installed")

    def test_google_models_registered(self):
        """Test that Google models are registered with @register_model."""
        pytest.importorskip("google.genai", reason="google-genai not installed")

        from msgflux.models.registry import model_registry

        if "chat_completion" in model_registry:
            assert "google" in model_registry.get("chat_completion", {})


# ============================================================
# 2. Initialization
# ============================================================


class TestGoogleChatCompletionInit:
    """Test suite for GoogleChatCompletion initialization."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key-12345")

    @pytest.fixture
    def mock_genai_client(self):
        with patch("msgflux.models.providers.google.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            yield mock_genai, mock_client

    def test_basic_initialization(self, mock_genai_client):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import GoogleChatCompletion

        model = GoogleChatCompletion(model_id="gemini-2.0-flash")

        assert model.model_id == "gemini-2.0-flash"
        assert model.provider == "google"
        assert model.model_type == "chat_completion"

    def test_initialization_with_parameters(self, mock_genai_client):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import GoogleChatCompletion

        model = GoogleChatCompletion(
            model_id="gemini-2.0-flash",
            max_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            stop=["STOP"],
        )

        assert model.sampling_run_params["max_output_tokens"] == 2048
        assert model.sampling_run_params["temperature"] == 0.7
        assert model.sampling_run_params["top_p"] == 0.9
        assert model.sampling_run_params["top_k"] == 40
        assert model.sampling_run_params["stop_sequences"] == ["STOP"]

    def test_initialization_with_thinking_params(self, mock_genai_client):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import GoogleChatCompletion

        model = GoogleChatCompletion(
            model_id="gemini-2.0-flash-thinking",
            enable_thinking=True,
            thinking_budget=8192,
            return_reasoning=True,
            reasoning_in_tool_call=True,
        )

        assert model.enable_thinking is True
        assert model.thinking_budget == 8192
        assert model.return_reasoning is True
        assert model.reasoning_in_tool_call is True

    def test_missing_api_key(self, monkeypatch):
        pytest.importorskip("google.genai")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

        from msgflux.models.providers.google import GoogleChatCompletion

        with pytest.raises(ValueError, match="Google API key is not available"):
            GoogleChatCompletion(model_id="gemini-2.0-flash")

    def test_api_key_from_google_env(self, monkeypatch, mock_genai_client):
        pytest.importorskip("google.genai")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "fallback-key")

        from msgflux.models.providers.google import GoogleChatCompletion

        model = GoogleChatCompletion(model_id="gemini-2.0-flash")
        assert model.model_id == "gemini-2.0-flash"

    def test_empty_sampling_params(self, mock_genai_client):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import GoogleChatCompletion

        model = GoogleChatCompletion(model_id="gemini-2.0-flash")

        assert model.sampling_run_params == {}


# ============================================================
# 3. Helper Functions
# ============================================================


class TestOpenAIToolsToGenAI:
    """Test OpenAI → GenAI tool schema conversion."""

    def test_single_tool_conversion(self):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import _openai_tools_to_gemini

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"}
                        },
                        "required": ["city"],
                    },
                },
            }
        ]

        result = _openai_tools_to_gemini(tools)

        assert len(result) == 1
        assert "function_declarations" in result[0]
        decls = result[0]["function_declarations"]
        assert len(decls) == 1
        assert decls[0]["name"] == "get_weather"
        assert decls[0]["description"] == "Get weather for a city"
        assert decls[0]["parameters"]["type"] == "OBJECT"
        assert decls[0]["parameters"]["properties"]["city"]["type"] == "STRING"
        assert decls[0]["parameters"]["required"] == ["city"]

    def test_multiple_tools_conversion(self):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import _openai_tools_to_gemini

        tools = [
            {"type": "function", "function": {"name": "tool_a", "parameters": {}}},
            {"type": "function", "function": {"name": "tool_b", "parameters": {}}},
        ]

        result = _openai_tools_to_gemini(tools)
        decls = result[0]["function_declarations"]
        assert len(decls) == 2
        assert decls[0]["name"] == "tool_a"
        assert decls[1]["name"] == "tool_b"

    def test_tool_without_description(self):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import _openai_tools_to_gemini

        tools = [
            {
                "type": "function",
                "function": {"name": "no_desc", "parameters": {}},
            }
        ]

        result = _openai_tools_to_gemini(tools)
        decl = result[0]["function_declarations"][0]
        assert "description" not in decl

    def test_nested_schema_conversion(self):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import _openai_tools_to_gemini

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "create_order",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "qty": {"type": "integer"},
                                    },
                                },
                            }
                        },
                    },
                },
            }
        ]

        result = _openai_tools_to_gemini(tools)
        params = result[0]["function_declarations"][0]["parameters"]
        assert params["properties"]["items"]["type"] == "ARRAY"
        assert params["properties"]["items"]["items"]["type"] == "OBJECT"


class TestConvertJsonSchemaToGenAI:
    """Test JSON Schema → GenAI schema conversion."""

    def test_simple_object(self):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import _convert_json_schema_to_gemini

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }

        result = _convert_json_schema_to_gemini(schema)
        assert result["type"] == "OBJECT"
        assert result["properties"]["name"]["type"] == "STRING"
        assert result["required"] == ["name"]

    def test_additional_properties_stripped(self):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import _convert_json_schema_to_gemini

        schema = {"type": "object", "additionalProperties": False}
        result = _convert_json_schema_to_gemini(schema)
        assert "additionalProperties" not in result

    def test_enum_preserved(self):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import _convert_json_schema_to_gemini

        schema = {"type": "string", "enum": ["a", "b", "c"]}
        result = _convert_json_schema_to_gemini(schema)
        assert result["enum"] == ["a", "b", "c"]


class TestGenAIToolChoice:
    """Test tool_choice conversion to GenAI FunctionCallingConfig."""

    def test_auto(self):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import _genai_tool_choice

        result = _genai_tool_choice("auto")
        assert result == {"function_calling_config": {"mode": "AUTO"}}

    def test_none_string(self):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import _genai_tool_choice

        result = _genai_tool_choice("none")
        assert result == {"function_calling_config": {"mode": "NONE"}}

    def test_required(self):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import _genai_tool_choice

        result = _genai_tool_choice("required")
        assert result == {"function_calling_config": {"mode": "ANY"}}

    def test_none_value(self):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import _genai_tool_choice

        result = _genai_tool_choice(None)
        assert result == {"function_calling_config": {"mode": "AUTO"}}

    def test_forced_tool_dict(self):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import _genai_tool_choice

        result = _genai_tool_choice(
            {"type": "function", "function": {"name": "get_weather"}}
        )
        assert result["function_calling_config"]["mode"] == "ANY"
        assert result["function_calling_config"]["allowed_function_names"] == [
            "get_weather"
        ]

    def test_forced_tool_string(self):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import _genai_tool_choice

        result = _genai_tool_choice("get_weather")
        assert result["function_calling_config"]["mode"] == "ANY"
        assert result["function_calling_config"]["allowed_function_names"] == [
            "get_weather"
        ]


# ============================================================
# 4. Build Config
# ============================================================


class TestBuildConfig:
    """Test _build_config method."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    @pytest.fixture
    def mock_genai_client(self):
        with patch("msgflux.models.providers.google.genai") as mock_genai:
            mock_genai.Client.return_value = MagicMock()
            yield mock_genai

    def test_basic_config(self, mock_genai_client):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import GoogleChatCompletion

        model = GoogleChatCompletion(model_id="gemini-2.0-flash", temperature=0.5)
        config = model._build_config()

        assert config["temperature"] == 0.5
        assert "system_instruction" not in config

    def test_config_with_system_instruction(self, mock_genai_client):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import GoogleChatCompletion

        model = GoogleChatCompletion(model_id="gemini-2.0-flash")
        config = model._build_config(system_instruction="You are helpful.")

        assert config["system_instruction"] == "You are helpful."

    def test_config_with_thinking(self, mock_genai_client):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import GoogleChatCompletion

        model = GoogleChatCompletion(
            model_id="gemini-2.0-flash",
            enable_thinking=True,
            thinking_budget=4096,
        )
        config = model._build_config()

        assert "thinking_config" in config
        assert config["thinking_config"]["include_thoughts"] is True
        assert config["thinking_config"]["thinking_budget"] == 4096

    def test_config_with_tools(self, mock_genai_client):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import GoogleChatCompletion

        model = GoogleChatCompletion(model_id="gemini-2.0-flash")
        tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                },
            }
        ]

        config = model._build_config(tool_schemas=tool_schemas, tool_choice="auto")

        assert "tools" in config
        assert "tool_config" in config
        assert config["tool_config"]["function_calling_config"]["mode"] == "AUTO"

    def test_config_with_generation_schema(self, mock_genai_client):
        pytest.importorskip("google.genai")

        import msgspec

        from msgflux.models.providers.google import GoogleChatCompletion

        class MyOutput(msgspec.Struct):
            answer: str
            confidence: float

        model = GoogleChatCompletion(model_id="gemini-2.0-flash")
        config = model._build_config(generation_schema=MyOutput)

        assert config["response_mime_type"] == "application/json"
        assert "response_json_schema" in config
        schema = config["response_json_schema"]
        assert "answer" in schema["properties"]
        assert "confidence" in schema["properties"]


# ============================================================
# 5. Process Model Output
# ============================================================


class TestProcessModelOutput:
    """Test _process_model_output method."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    @pytest.fixture
    def mock_genai_client(self):
        with patch("msgflux.models.providers.google.genai") as mock_genai:
            mock_genai.Client.return_value = MagicMock()
            yield mock_genai

    def _make_model_output(self, parts, usage=None):
        """Create a mock Gemini model output."""
        mock_content = Mock()
        mock_content.parts = parts

        mock_candidate = Mock()
        mock_candidate.content = mock_content

        mock_usage = Mock()
        if usage:
            mock_usage.prompt_token_count = usage.get("prompt_tokens", 0)
            mock_usage.candidates_token_count = usage.get("completion_tokens", 0)
            mock_usage.total_token_count = usage.get("total_tokens", 0)
            mock_usage.thoughts_token_count = usage.get("reasoning_tokens", None)
        else:
            mock_usage.prompt_token_count = 10
            mock_usage.candidates_token_count = 20
            mock_usage.total_token_count = 30
            mock_usage.thoughts_token_count = None

        mock_output = Mock()
        mock_output.candidates = [mock_candidate]
        mock_output.usage_metadata = mock_usage
        return mock_output

    def _make_text_part(self, text, thought=False):
        part = Mock()
        part.text = text
        part.thought = thought
        part.function_call = None
        return part

    def _make_function_call_part(self, name, args, call_id=None, thought_sig=None):
        fc = Mock()
        fc.name = name
        fc.args = args
        fc.id = call_id

        part = Mock()
        part.text = None
        part.thought = False
        part.function_call = fc
        part.thought_signature = thought_sig
        return part

    def test_text_response(self, mock_genai_client):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import GoogleChatCompletion

        model = GoogleChatCompletion(model_id="gemini-2.0-flash")
        parts = [self._make_text_part("Hello, world!")]
        output = self._make_model_output(parts)

        response = model._process_model_output(output)

        assert response.response_type == "text_generation"
        assert response.data == "Hello, world!"

    def test_text_with_reasoning(self, mock_genai_client):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import GoogleChatCompletion

        model = GoogleChatCompletion(
            model_id="gemini-2.0-flash",
            enable_thinking=True,
            return_reasoning=True,
        )
        parts = [
            self._make_text_part("Let me think...", thought=True),
            self._make_text_part("The answer is 42."),
        ]
        output = self._make_model_output(parts)

        response = model._process_model_output(output)

        assert response.response_type == "reasoning_text_generation"
        assert response.data.answer == "The answer is 42."
        assert response.data.think == "Let me think..."

    def test_tool_call_response(self, mock_genai_client):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import GoogleChatCompletion

        model = GoogleChatCompletion(model_id="gemini-2.0-flash")
        parts = [
            self._make_function_call_part(
                "get_weather", {"city": "NYC"}, call_id="call_1", thought_sig="sig_abc"
            )
        ]
        output = self._make_model_output(parts)

        response = model._process_model_output(output)

        assert response.response_type == "tool_call"
        assert isinstance(response.data, ToolCallAggregator)
        calls = response.data.get_calls()
        assert len(calls) == 1
        assert calls[0][0] == "call_1"
        assert calls[0][1] == "get_weather"
        assert calls[0][2] == {"city": "NYC"}

    def test_tool_call_preserves_thought_signature(self, mock_genai_client):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import GoogleChatCompletion

        model = GoogleChatCompletion(model_id="gemini-2.0-flash")
        parts = [
            self._make_function_call_part(
                "search", {"q": "test"}, call_id="call_1", thought_sig="my_sig"
            )
        ]
        output = self._make_model_output(parts)

        response = model._process_model_output(output)

        agg = response.data
        assert agg.tool_calls[0]["thought_signature"] == "my_sig"

    def test_tool_call_with_reasoning(self, mock_genai_client):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import GoogleChatCompletion

        model = GoogleChatCompletion(
            model_id="gemini-2.0-flash",
            enable_thinking=True,
            reasoning_in_tool_call=True,
        )
        parts = [
            self._make_text_part("I need to search...", thought=True),
            self._make_function_call_part(
                "search", {"q": "test"}, call_id="call_1"
            ),
        ]
        output = self._make_model_output(parts)

        response = model._process_model_output(output)

        assert response.response_type == "tool_call"
        agg = response.data
        assert agg.reasoning == "I need to search..."

    def test_usage_metadata(self, mock_genai_client):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import GoogleChatCompletion

        model = GoogleChatCompletion(model_id="gemini-2.0-flash")
        parts = [self._make_text_part("hello")]
        output = self._make_model_output(
            parts,
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "reasoning_tokens": 25,
            },
        )

        response = model._process_model_output(output)

        assert response.metadata.usage["prompt_tokens"] == 100
        assert response.metadata.usage["completion_tokens"] == 50
        assert response.metadata.usage["total_tokens"] == 150
        assert response.metadata.usage["reasoning_tokens"] == 25

    def test_structured_output(self, mock_genai_client):
        pytest.importorskip("google.genai")

        import msgspec

        from msgflux.models.providers.google import GoogleChatCompletion

        class MyOutput(msgspec.Struct):
            answer: str
            score: int

        model = GoogleChatCompletion(model_id="gemini-2.0-flash")
        parts = [self._make_text_part('{"answer": "hello", "score": 42}')]
        output = self._make_model_output(parts)

        response = model._process_model_output(output, generation_schema=MyOutput)

        assert response.response_type == "structured"
        assert response.data.answer == "hello"
        assert response.data.score == 42

    def test_empty_response(self, mock_genai_client):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import GoogleChatCompletion

        model = GoogleChatCompletion(model_id="gemini-2.0-flash")
        mock_content = Mock()
        mock_content.parts = []
        mock_candidate = Mock()
        mock_candidate.content = mock_content
        mock_usage = Mock()
        mock_usage.prompt_token_count = 5
        mock_usage.candidates_token_count = 0
        mock_usage.total_token_count = 5
        mock_usage.thoughts_token_count = None
        mock_output = Mock()
        mock_output.candidates = [mock_candidate]
        mock_output.usage_metadata = mock_usage

        response = model._process_model_output(mock_output)
        assert response.response_type == "text_generation"
        assert response.data == ""

    def test_multiple_function_calls(self, mock_genai_client):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import GoogleChatCompletion

        model = GoogleChatCompletion(model_id="gemini-2.0-flash")
        parts = [
            self._make_function_call_part(
                "get_weather", {"city": "NYC"}, call_id="call_1"
            ),
            self._make_function_call_part(
                "get_time", {"tz": "UTC"}, call_id="call_2"
            ),
        ]
        output = self._make_model_output(parts)

        response = model._process_model_output(output)

        assert response.response_type == "tool_call"
        calls = response.data.get_calls()
        assert len(calls) == 2
        assert calls[0][1] == "get_weather"
        assert calls[1][1] == "get_time"

    def test_function_call_no_id_fallback(self, mock_genai_client):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import GoogleChatCompletion

        model = GoogleChatCompletion(model_id="gemini-2.0-flash")
        parts = [
            self._make_function_call_part("search", {"q": "test"}, call_id=None)
        ]
        output = self._make_model_output(parts)

        response = model._process_model_output(output)

        calls = response.data.get_calls()
        assert calls[0][0] == "call_0"


# ============================================================
# 6. __call__ and acall interface
# ============================================================


class TestCallInterface:
    """Test __call__ and acall public interface."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    @pytest.fixture
    def mock_genai_client(self):
        with patch("msgflux.models.providers.google.genai") as mock_genai:
            mock_genai.Client.return_value = MagicMock()
            yield mock_genai

    def test_call_with_string_message(self, mock_genai_client):
        """Test that string input is converted to user message."""
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import GoogleChatCompletion

        model = GoogleChatCompletion(model_id="gemini-2.0-flash")

        # Mock _generate to capture args
        captured = {}

        def mock_generate(**kwargs):
            captured.update(kwargs)
            resp = ModelResponse()
            resp.set_response_type("text_generation")
            resp.add("mocked")
            return resp

        model._generate = mock_generate

        model("Hello!")

        assert isinstance(captured["messages"], list)
        assert captured["messages"][0]["role"] == "user"
        assert captured["messages"][0]["content"] == "Hello!"

    def test_call_with_chat_messages(self, mock_genai_client):
        """Test that ChatMessages is converted to chatml."""
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import GoogleChatCompletion

        model = GoogleChatCompletion(model_id="gemini-2.0-flash")

        captured = {}

        def mock_generate(**kwargs):
            captured.update(kwargs)
            resp = ModelResponse()
            resp.set_response_type("text_generation")
            resp.add("mocked")
            return resp

        model._generate = mock_generate

        cm = ChatMessages([{"role": "user", "content": "hi"}])
        model(cm)

        assert isinstance(captured["messages"], list)

    def test_call_passes_system_prompt(self, mock_genai_client):
        """Test that system_prompt is passed as system_instruction."""
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import GoogleChatCompletion

        model = GoogleChatCompletion(model_id="gemini-2.0-flash")

        captured = {}

        def mock_generate(**kwargs):
            captured.update(kwargs)
            resp = ModelResponse()
            resp.set_response_type("text_generation")
            resp.add("mocked")
            return resp

        model._generate = mock_generate

        model("hi", system_prompt="Be helpful")

        assert captured["system_instruction"] == "Be helpful"

    def test_call_passes_tool_schemas(self, mock_genai_client):
        """Test that tool_schemas and tool_choice are forwarded."""
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import GoogleChatCompletion

        model = GoogleChatCompletion(model_id="gemini-2.0-flash")

        captured = {}

        def mock_generate(**kwargs):
            captured.update(kwargs)
            resp = ModelResponse()
            resp.set_response_type("text_generation")
            resp.add("mocked")
            return resp

        model._generate = mock_generate

        tools = [
            {
                "type": "function",
                "function": {"name": "test_tool", "parameters": {}},
            }
        ]
        model("hi", tool_schemas=tools, tool_choice="auto")

        assert captured["tool_schemas"] == tools
        assert captured["tool_choice"] == "auto"

    def test_call_tool_choice_string_to_dict(self, mock_genai_client):
        """Test that non-standard tool_choice string is converted to dict."""
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import GoogleChatCompletion

        model = GoogleChatCompletion(model_id="gemini-2.0-flash")

        captured = {}

        def mock_generate(**kwargs):
            captured.update(kwargs)
            resp = ModelResponse()
            resp.set_response_type("text_generation")
            resp.add("mocked")
            return resp

        model._generate = mock_generate

        model("hi", tool_schemas=[{"type": "function", "function": {"name": "x", "parameters": {}}}], tool_choice="get_weather")

        assert captured["tool_choice"] == {
            "type": "function",
            "function": {"name": "get_weather"},
        }

    def test_stream_rejects_typed_parser(self, mock_genai_client):
        """Test that stream=True with typed_parser raises ValueError."""
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import GoogleChatCompletion

        model = GoogleChatCompletion(model_id="gemini-2.0-flash")

        with pytest.raises(ValueError, match="typed_parser"):
            model("hi", stream=True, typed_parser="xml")


# ============================================================
# 7. Ensure ChatMessages conversion
# ============================================================


class TestEnsureChatMessages:
    """Test _ensure_chat_messages helper."""

    @pytest.fixture(autouse=True)
    def setup_env(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    @pytest.fixture
    def mock_genai_client(self):
        with patch("msgflux.models.providers.google.genai") as mock_genai:
            mock_genai.Client.return_value = MagicMock()
            yield mock_genai

    def test_from_chat_messages(self, mock_genai_client):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import GoogleChatCompletion

        model = GoogleChatCompletion(model_id="gemini-2.0-flash")
        cm = ChatMessages([{"role": "user", "content": "hi"}])

        result = model._ensure_chat_messages(cm)
        assert isinstance(result, ChatMessages)
        assert len(result) == 1
        # Should be a copy
        cm.append({"role": "assistant", "content": "bye"})
        assert len(result) == 1

    def test_from_list(self, mock_genai_client):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import GoogleChatCompletion

        model = GoogleChatCompletion(model_id="gemini-2.0-flash")
        messages = [{"role": "user", "content": "hi"}]

        result = model._ensure_chat_messages(messages)
        assert isinstance(result, ChatMessages)
        assert len(result) == 1

    def test_from_none(self, mock_genai_client):
        pytest.importorskip("google.genai")
        from msgflux.models.providers.google import GoogleChatCompletion

        model = GoogleChatCompletion(model_id="gemini-2.0-flash")

        result = model._ensure_chat_messages(None)
        assert isinstance(result, ChatMessages)
        assert len(result) == 0
