"""Tests for msgflux.nn.modules.predictor module."""

import pytest
from unittest.mock import Mock, AsyncMock
from msgflux.nn.modules.predictor import Predictor
from msgflux.message import Message
from msgflux.models.base import BaseModel
from msgflux.models.response import ModelResponse


class MockModel(BaseModel):
    """Mock model for testing Predictor."""

    model_type = "test_predictor"
    provider = "mock"

    def __init__(self):
        self.model_id = "test-model"
        self._api_key = None
        self.model = None
        self.processor = None
        self.client = None

    def _initialize(self):
        pass

    def __call__(self, data, **kwargs):
        response = ModelResponse()
        response.set_response_type("audio_generation")
        response.add({"audio": f"generated from {data}"})
        return response

    async def acall(self, data, **kwargs):
        response = ModelResponse()
        response.set_response_type("audio_generation")
        response.add({"audio": f"async generated from {data}"})
        return response


class TestPredictor:
    """Test suite for Predictor class."""

    def test_predictor_initialization(self):
        """Test predictor initialization."""
        model = MockModel()
        predictor = Predictor(
            name="test_predictor",
            model=model,
            task_inputs="content",
            response_mode="outputs.prediction"
        )

        assert predictor.name == "test_predictor"
        assert predictor.model == model
        assert predictor.task_inputs == "content"
        assert predictor.response_mode == "outputs.prediction"

    def test_predictor_with_execution_kwargs(self):
        """Test predictor with execution kwargs."""
        model = MockModel()
        predictor = Predictor(
            name="predictor",
            model=model,
            execution_kwargs={"temperature": 0.7, "max_tokens": 100}
        )

        assert predictor.execution_kwargs["temperature"] == 0.7
        assert predictor.execution_kwargs["max_tokens"] == 100

    def test_set_model_invalid_type(self):
        """Test that setting invalid model type raises TypeError."""
        with pytest.raises(TypeError):
            Predictor(name="test", model="not a model")

    def test_prepare_task_with_message(self):
        """Test _prepare_task with Message input."""
        model = MockModel()
        predictor = Predictor(
            name="test",
            model=model,
            task_inputs="content"
        )

        message = Message(content="test input")
        inputs = predictor._prepare_task(message)

        assert inputs.data == "test input"

    def test_prepare_task_with_plain_data(self):
        """Test _prepare_task with plain data."""
        model = MockModel()
        predictor = Predictor(
            name="test",
            model=model
        )

        inputs = predictor._prepare_task("plain data")

        assert inputs.data == "plain data"

    def test_prepare_task_with_model_preference(self):
        """Test _prepare_task with model preference."""
        model = MockModel()
        predictor = Predictor(
            name="test",
            model=model,
            model_preference="context.preferred_model"
        )

        message = Message(content="test")
        message.context["preferred_model"] = "gpt-4"

        inputs = predictor._prepare_task(message)

        assert inputs.model_preference == "gpt-4"

    def test_prepare_model_execution(self):
        """Test _prepare_model_execution."""
        model = MockModel()
        predictor = Predictor(
            name="test",
            model=model,
            execution_kwargs={"temperature": 0.5}
        )

        params = predictor._prepare_model_execution("test data")

        assert params.data == "test data"
        assert params.temperature == 0.5

    def test_prepare_model_execution_with_preference(self):
        """Test _prepare_model_execution with model preference."""
        model = MockModel()
        predictor = Predictor(
            name="test",
            model=model
        )

        params = predictor._prepare_model_execution("test data", model_preference="gpt-4")

        assert params.data == "test data"
        assert params.model_preference == "gpt-4"

    def test_execute_model(self):
        """Test _execute_model."""
        model = MockModel()
        predictor = Predictor(
            name="test",
            model=model
        )

        response = predictor._execute_model("test input")

        assert isinstance(response, ModelResponse)
        assert response.response_type == "audio_generation"

    @pytest.mark.asyncio
    async def test_aexecute_model(self):
        """Test async _aexecute_model."""
        model = MockModel()
        predictor = Predictor(
            name="test",
            model=model
        )

        response = await predictor._aexecute_model("test input")

        assert isinstance(response, ModelResponse)
        assert response.response_type == "audio_generation"

    def test_forward_with_message(self):
        """Test forward with Message."""
        model = MockModel()
        predictor = Predictor(
            name="test",
            model=model,
            task_inputs="content",
            response_mode="outputs.result"
        )

        message = Message(content="test content")
        result = predictor(message)

        assert isinstance(result, Message)
        assert "result" in result.outputs

    def test_forward_with_plain_data(self):
        """Test forward with plain data and plain_response mode."""
        model = MockModel()
        predictor = Predictor(
            name="test",
            model=model,
            response_mode="plain_response"
        )

        result = predictor("plain input")

        # With plain_response mode, should return the processed response
        assert result is not None

    @pytest.mark.asyncio
    async def test_aforward_with_message(self):
        """Test async forward with Message."""
        model = MockModel()
        predictor = Predictor(
            name="test",
            model=model,
            task_inputs="content",
            response_mode="outputs.result"
        )

        message = Message(content="async test")
        result = await predictor.acall(message)

        assert isinstance(result, Message)
        assert "result" in result.outputs

    def test_inspect_model_execution_params(self):
        """Test inspect_model_execution_params for debugging."""
        model = MockModel()
        predictor = Predictor(
            name="test",
            model=model,
            task_inputs="content",
            execution_kwargs={"temperature": 0.7}
        )

        message = Message(content="test")
        params = predictor.inspect_model_execution_params(message)

        assert params.data == "test"
        assert params.temperature == 0.7

    def test_process_model_response_unsupported_type(self):
        """Test _process_model_response with unsupported response type."""
        model = MockModel()
        predictor = Predictor(
            name="test",
            model=model
        )

        invalid_response = Mock(spec=ModelResponse)
        invalid_response.response_type = "unsupported_type"

        with pytest.raises(ValueError, match="Unsupported model response type"):
            predictor._process_model_response(invalid_response, "message")

    def test_response_template(self):
        """Test predictor with response template."""
        model = MockModel()
        predictor = Predictor(
            name="test",
            model=model,
            response_template="Result: {}"
        )

        assert predictor.response_template == "Result: {}"

    def test_name_setting(self):
        """Test that name is properly set."""
        model = MockModel()
        predictor = Predictor(
            name="my_custom_predictor",
            model=model
        )

        assert predictor.name == "my_custom_predictor"
        assert predictor.get_module_name() == "my_custom_predictor"
