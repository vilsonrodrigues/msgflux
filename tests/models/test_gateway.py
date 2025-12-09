"""Tests for msgflux.models.gateway module."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, time, timezone

from msgflux.models.gateway import ModelGateway
from msgflux.exceptions import ModelRouterError
from msgflux.models.base import BaseModel
from msgflux.models.response import ModelResponse


class MockModel(BaseModel):
    """Mock model for testing gateway."""

    def __init__(self, model_id: str, model_type: str = "chat_completion",
                 provider: str = "mock", should_fail: bool = False):
        self.model_id = model_id
        self.model_type = model_type
        self.provider = provider
        self.should_fail = should_fail
        self.call_count = 0

    def _initialize(self):
        pass

    def __call__(self, **kwargs):
        self.call_count += 1
        if self.should_fail:
            raise RuntimeError(f"Mock failure for {self.model_id}")
        response = ModelResponse()
        response.add(f"Response from {self.model_id}")
        return response

    async def acall(self, **kwargs):
        self.call_count += 1
        if self.should_fail:
            raise RuntimeError(f"Mock async failure for {self.model_id}")
        response = ModelResponse()
        response.add(f"Async response from {self.model_id}")
        return response

    def serialize(self):
        return {
            "model_id": self.model_id,
            "model_type": self.model_type,
            "provider": self.provider,
        }


class TestModelGatewayInitialization:
    """Test suite for ModelGateway initialization."""

    def test_gateway_initialization_basic(self):
        """Test basic ModelGateway initialization."""
        models = [
            MockModel("model-1"),
            MockModel("model-2"),
        ]

        gateway = ModelGateway(models=models)

        assert len(gateway.models) == 2
        assert gateway.model_type == "chat_completion"
        assert gateway.current_model_index == 0

    def test_gateway_initialization_with_time_constraints(self):
        """Test ModelGateway with time constraints."""
        models = [
            MockModel("model-1"),
            MockModel("model-2"),
        ]

        time_constraints = {
            "model-1": [("22:00", "06:00")],
        }

        gateway = ModelGateway(models=models, time_constraints=time_constraints)

        assert gateway.raw_time_constraints == time_constraints
        assert "model-1" in gateway.parsed_time_constraints

    def test_gateway_empty_models_list(self):
        """Test ModelGateway raises error with empty models list."""
        with pytest.raises(TypeError, match="`models` must be a non-empty list"):
            ModelGateway(models=[])

    def test_gateway_invalid_models_type(self):
        """Test ModelGateway raises error with invalid models type."""
        with pytest.raises(TypeError, match="`models` must be a non-empty list"):
            ModelGateway(models="not-a-list")

    def test_gateway_non_basemodel_instances(self):
        """Test ModelGateway raises error when models don't inherit from BaseModel."""
        with pytest.raises(TypeError, match="inheriting from `BaseModel`"):
            ModelGateway(models=[MockModel("model-1"), "not-a-model"])

    def test_gateway_single_model_warning(self):
        """Test ModelGateway with only one model (should warn but not fail)."""
        models = [MockModel("model-1")]

        # Should not raise, but log a warning
        gateway = ModelGateway(models=models)
        assert len(gateway.models) == 1

    def test_gateway_mixed_model_types(self):
        """Test ModelGateway raises error with different model types."""
        models = [
            MockModel("model-1", model_type="chat_completion"),
            MockModel("model-2", model_type="text_embedder"),
        ]

        with pytest.raises(TypeError, match="must be of the same `model_type`"):
            ModelGateway(models=models)

    def test_gateway_duplicate_model_ids(self):
        """Test ModelGateway raises error with duplicate model IDs."""
        models = [
            MockModel("model-1"),
            MockModel("model-1"),
        ]

        with pytest.raises(ValueError, match="Duplicate model ID"):
            ModelGateway(models=models)

    def test_gateway_model_without_model_type(self):
        """Test ModelGateway raises error when model lacks model_type."""
        model = MockModel("model-1")
        del model.model_type

        with pytest.raises(AttributeError, match="does not have a valid `model_type`"):
            ModelGateway(models=[model, MockModel("model-2")])

    def test_gateway_model_without_model_id(self):
        """Test ModelGateway raises error when model lacks model_id."""
        model = MockModel("model-1")
        del model.model_id

        with pytest.raises(AttributeError, match="does not have a valid `model_id`"):
            ModelGateway(models=[model, MockModel("model-2")])

    def test_gateway_model_without_provider(self):
        """Test ModelGateway raises error when model lacks provider."""
        model = MockModel("model-1")
        del model.provider

        with pytest.raises(AttributeError, match="does not have a valid `provider`"):
            ModelGateway(models=[model, MockModel("model-2")])


class TestTimeConstraintParsing:
    """Test suite for time constraint parsing."""

    def test_parse_time_constraints_valid(self):
        """Test parsing valid time constraints."""
        models = [MockModel("model-1"), MockModel("model-2")]
        time_constraints = {
            "model-1": [("09:00", "17:00")],
            "model-2": [("22:00", "06:00")],
        }

        gateway = ModelGateway(models=models, time_constraints=time_constraints)

        assert "model-1" in gateway.parsed_time_constraints
        assert "model-2" in gateway.parsed_time_constraints
        assert len(gateway.parsed_time_constraints["model-1"]) == 1

    def test_parse_time_constraints_multiple_intervals(self):
        """Test parsing multiple time intervals for a model."""
        models = [MockModel("model-1"), MockModel("model-2")]
        time_constraints = {
            "model-1": [("09:00", "12:00"), ("13:00", "17:00")],
        }

        gateway = ModelGateway(models=models, time_constraints=time_constraints)

        assert len(gateway.parsed_time_constraints["model-1"]) == 2

    def test_parse_time_constraints_invalid_format(self):
        """Test parsing time constraints with invalid time format."""
        models = [MockModel("model-1"), MockModel("model-2")]
        time_constraints = {
            "model-1": [("25:00", "06:00")],  # Invalid hour
        }

        with pytest.raises(ValueError, match="Invalid time format"):
            ModelGateway(models=models, time_constraints=time_constraints)

    def test_parse_time_constraints_not_list(self):
        """Test parsing time constraints when intervals is not a list."""
        models = [MockModel("model-1"), MockModel("model-2")]
        time_constraints = {
            "model-1": "not-a-list",
        }

        with pytest.raises(TypeError, match="must be a list of tuples"):
            ModelGateway(models=models, time_constraints=time_constraints)

    def test_parse_time_constraints_invalid_interval_type(self):
        """Test parsing time constraints with invalid interval type."""
        models = [MockModel("model-1"), MockModel("model-2")]
        time_constraints = {
            "model-1": ["not-a-tuple"],
        }

        with pytest.raises(TypeError, match="must be a tuple/list of two strings"):
            ModelGateway(models=models, time_constraints=time_constraints)

    def test_parse_time_constraints_non_string_times(self):
        """Test parsing time constraints with non-string times."""
        models = [MockModel("model-1"), MockModel("model-2")]
        time_constraints = {
            "model-1": [(9, 17)],  # Numbers instead of strings
        }

        with pytest.raises(TypeError, match="must be strings"):
            ModelGateway(models=models, time_constraints=time_constraints)

    def test_parse_time_constraints_nonexistent_model(self):
        """Test warning when time constraint references nonexistent model."""
        models = [MockModel("model-1"), MockModel("model-2")]
        time_constraints = {
            "nonexistent-model": [("09:00", "17:00")],
        }

        # Should not raise, but log a warning
        gateway = ModelGateway(models=models, time_constraints=time_constraints)
        assert "nonexistent-model" in gateway.parsed_time_constraints


class TestTimeRestriction:
    """Test suite for time restriction checking."""

    def test_is_time_restricted_no_constraints(self):
        """Test model with no time constraints is not restricted."""
        models = [MockModel("model-1"), MockModel("model-2")]
        gateway = ModelGateway(models=models)

        assert not gateway._is_time_restricted("model-1")

    def test_is_time_restricted_within_range(self):
        """Test model is restricted when current time is within range."""
        models = [MockModel("model-1"), MockModel("model-2")]

        # Mock current time to be 10:00
        with patch("msgflux.models.gateway.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(
                2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc
            )
            mock_datetime.strptime = datetime.strptime

            time_constraints = {
                "model-1": [("09:00", "17:00")],
            }

            gateway = ModelGateway(models=models, time_constraints=time_constraints)
            assert gateway._is_time_restricted("model-1")

    def test_is_time_restricted_outside_range(self):
        """Test model is not restricted when current time is outside range."""
        models = [MockModel("model-1"), MockModel("model-2")]

        # Mock current time to be 08:00
        with patch("msgflux.models.gateway.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(
                2025, 1, 1, 8, 0, 0, tzinfo=timezone.utc
            )
            mock_datetime.strptime = datetime.strptime

            time_constraints = {
                "model-1": [("09:00", "17:00")],
            }

            gateway = ModelGateway(models=models, time_constraints=time_constraints)
            assert not gateway._is_time_restricted("model-1")

    def test_is_time_restricted_midnight_crossover(self):
        """Test time restriction crossing midnight (e.g., 22:00 to 06:00)."""
        models = [MockModel("model-1"), MockModel("model-2")]

        # Mock current time to be 23:00 (restricted)
        with patch("msgflux.models.gateway.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(
                2025, 1, 1, 23, 0, 0, tzinfo=timezone.utc
            )
            mock_datetime.strptime = datetime.strptime

            time_constraints = {
                "model-1": [("22:00", "06:00")],
            }

            gateway = ModelGateway(models=models, time_constraints=time_constraints)
            assert gateway._is_time_restricted("model-1")

        # Mock current time to be 03:00 (also restricted)
        with patch("msgflux.models.gateway.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(
                2025, 1, 1, 3, 0, 0, tzinfo=timezone.utc
            )
            mock_datetime.strptime = datetime.strptime

            gateway = ModelGateway(models=models, time_constraints=time_constraints)
            assert gateway._is_time_restricted("model-1")


class TestModelExecution:
    """Test suite for model execution."""

    def test_execute_model_basic(self):
        """Test basic model execution."""
        models = [MockModel("model-1"), MockModel("model-2")]
        gateway = ModelGateway(models=models)

        response = gateway()

        assert response is not None
        assert models[0].call_count == 1
        assert "model-1" in response.data

    def test_execute_model_with_fallback(self):
        """Test fallback when first model fails."""
        models = [
            MockModel("model-1", should_fail=True),
            MockModel("model-2"),
        ]
        gateway = ModelGateway(models=models)

        response = gateway()

        assert response is not None
        assert models[0].call_count == 1
        assert models[1].call_count == 1
        assert "model-2" in response.data

    def test_execute_model_all_fail(self):
        """Test error when all models fail."""
        models = [
            MockModel("model-1", should_fail=True),
            MockModel("model-2", should_fail=True),
        ]
        gateway = ModelGateway(models=models)

        with pytest.raises(ModelRouterError, match="All .* available models failed"):
            gateway()

    def test_execute_model_with_preference(self):
        """Test model preference is respected."""
        model1 = MockModel("model-1")
        model2 = MockModel("model-2")
        models = [model1, model2]
        gateway = ModelGateway(models=models)

        response = gateway(model_preference="model-2")

        assert response is not None
        assert model2.call_count == 1
        assert model1.call_count == 0
        assert "model-2" in response.data

    def test_execute_model_with_preference_fallback(self):
        """Test fallback when preferred model fails."""
        model1 = MockModel("model-1")
        model2 = MockModel("model-2", should_fail=True)
        models = [model1, model2]
        gateway = ModelGateway(models=models)

        response = gateway(model_preference="model-2")

        assert response is not None
        assert model2.call_count == 1
        assert model1.call_count == 1
        assert "model-1" in response.data

    def test_execute_model_with_kwargs(self):
        """Test passing kwargs to model."""
        model = MockModel("model-1")
        gateway = ModelGateway(models=[model, MockModel("model-2")])

        response = gateway(temperature=0.7, max_tokens=100)

        assert response is not None
        assert model.call_count == 1

    def test_execute_model_time_restricted(self):
        """Test execution skips time-restricted models."""
        models = [MockModel("model-1"), MockModel("model-2")]

        with patch("msgflux.models.gateway.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(
                2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc
            )
            mock_datetime.strptime = datetime.strptime

            time_constraints = {
                "model-1": [("09:00", "17:00")],
            }

            gateway = ModelGateway(models=models, time_constraints=time_constraints)
            response = gateway()

            # model-1 is restricted, so model-2 should be used
            assert models[0].call_count == 0
            assert models[1].call_count == 1
            assert "model-2" in response.data

    def test_execute_model_all_restricted(self):
        """Test error when all models are time-restricted."""
        models = [MockModel("model-1"), MockModel("model-2")]

        with patch("msgflux.models.gateway.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(
                2025, 1, 1, 10, 0, 0, tzinfo=timezone.utc
            )
            mock_datetime.strptime = datetime.strptime

            time_constraints = {
                "model-1": [("09:00", "17:00")],
                "model-2": [("09:00", "17:00")],
            }

            gateway = ModelGateway(models=models, time_constraints=time_constraints)

            with pytest.raises(
                ModelRouterError, match="No model available due to time constraints"
            ):
                gateway()

    @pytest.mark.asyncio
    async def test_aexecute_model_basic(self):
        """Test basic async model execution."""
        models = [MockModel("model-1"), MockModel("model-2")]
        gateway = ModelGateway(models=models)

        response = await gateway.acall()

        assert response is not None
        assert models[0].call_count == 1
        assert "model-1" in response.data

    @pytest.mark.asyncio
    async def test_aexecute_model_with_fallback(self):
        """Test async fallback when first model fails."""
        models = [
            MockModel("model-1", should_fail=True),
            MockModel("model-2"),
        ]
        gateway = ModelGateway(models=models)

        response = await gateway.acall()

        assert response is not None
        assert models[0].call_count == 1
        assert models[1].call_count == 1
        assert "model-2" in response.data

    @pytest.mark.asyncio
    async def test_aexecute_model_all_fail(self):
        """Test async error when all models fail."""
        models = [
            MockModel("model-1", should_fail=True),
            MockModel("model-2", should_fail=True),
        ]
        gateway = ModelGateway(models=models)

        with pytest.raises(ModelRouterError, match="All .* available models failed"):
            await gateway.acall()


class TestGatewaySerialization:
    """Test suite for gateway serialization."""

    def test_serialize_basic(self):
        """Test basic gateway serialization."""
        models = [MockModel("model-1"), MockModel("model-2")]
        gateway = ModelGateway(models=models)

        serialized = gateway.serialize()

        assert "msgflux_type" in serialized
        assert serialized["msgflux_type"] == "model_gateway"
        assert "state" in serialized
        assert "models" in serialized["state"]
        assert len(serialized["state"]["models"]) == 2

    def test_serialize_with_time_constraints(self):
        """Test serialization preserves time constraints."""
        models = [MockModel("model-1"), MockModel("model-2")]
        time_constraints = {
            "model-1": [("09:00", "17:00")],
        }
        gateway = ModelGateway(models=models, time_constraints=time_constraints)

        serialized = gateway.serialize()

        assert "time_constraints" in serialized["state"]
        assert serialized["state"]["time_constraints"] == time_constraints

    @patch("msgflux.models.model.Model.from_serialized")
    def test_from_serialized_basic(self, mock_from_serialized):
        """Test deserializing gateway from data."""
        mock_from_serialized.side_effect = [
            MockModel("model-1"),
            MockModel("model-2"),
        ]

        data = {
            "msgflux_type": "model_gateway",
            "state": {
                "models": [
                    {"model_id": "model-1"},
                    {"model_id": "model-2"},
                ],
            },
        }

        gateway = ModelGateway.from_serialized(data)

        assert len(gateway.models) == 2
        assert mock_from_serialized.call_count == 2

    @patch("msgflux.models.model.Model.from_serialized")
    def test_from_serialized_with_time_constraints(self, mock_from_serialized):
        """Test deserializing gateway with time constraints."""
        mock_from_serialized.side_effect = [
            MockModel("model-1"),
            MockModel("model-2"),
        ]

        time_constraints = {"model-1": [("09:00", "17:00")]}
        data = {
            "msgflux_type": "model_gateway",
            "state": {
                "models": [
                    {"model_id": "model-1"},
                    {"model_id": "model-2"},
                ],
                "time_constraints": time_constraints,
            },
        }

        gateway = ModelGateway.from_serialized(data)

        assert gateway.raw_time_constraints == time_constraints

    def test_from_serialized_invalid_type(self):
        """Test error when deserializing with wrong msgflux_type."""
        data = {
            "msgflux_type": "wrong_type",
            "state": {},
        }

        with pytest.raises(ValueError, match="Incorrect msgflux type"):
            ModelGateway.from_serialized(data)

    def test_from_serialized_no_models(self):
        """Test error when deserializing without models."""
        data = {
            "msgflux_type": "model_gateway",
            "state": {
                "models": [],
            },
        }

        with pytest.raises(ValueError, match="does not contain templates"):
            ModelGateway.from_serialized(data)
