"""Tests for msgflux.models.response module."""

import pytest

from msgflux.models.response import ModelResponse, ModelStreamResponse


class TestModelResponse:
    """Test suite for ModelResponse."""

    def test_model_response_initialization(self):
        """Test ModelResponse initialization."""
        response = ModelResponse()
        assert response.data is None
        assert response.metadata is None
        assert response.response_type is None

    def test_model_response_add_data(self):
        """Test adding data to ModelResponse."""
        response = ModelResponse()
        test_data = "Test response content"
        response.add(test_data)
        assert response.data == test_data

    def test_model_response_consume(self):
        """Test consuming data from ModelResponse."""
        response = ModelResponse()
        test_data = {"result": "success", "value": 42}
        response.add(test_data)
        consumed = response.consume()
        assert consumed == test_data

    def test_model_response_set_metadata(self):
        """Test setting metadata on ModelResponse."""
        response = ModelResponse()
        metadata = {"tokens": 100, "model": "gpt-4"}
        response.set_metadata(metadata)
        assert response.metadata == metadata

    def test_model_response_with_none_data(self):
        """Test ModelResponse with None data."""
        response = ModelResponse()
        response.add(None)
        assert response.consume() is None


class TestModelStreamResponse:
    """Test suite for ModelStreamResponse."""

    def test_model_stream_response_initialization(self):
        """Test ModelStreamResponse initialization."""
        response = ModelStreamResponse()
        assert response.data is None
        assert response.metadata is None
        assert response.response_type is None

    @pytest.mark.asyncio
    async def test_model_stream_response_add_and_consume(self):
        """Test adding data and consuming from ModelStreamResponse."""
        response = ModelStreamResponse()
        chunks = ["Hello", " ", "world", "!"]

        for chunk in chunks:
            response.add(chunk)
        response.add(None)

        consumed_chunks = []
        async for chunk in response.consume():
            consumed_chunks.append(chunk)

        assert consumed_chunks == chunks

    @pytest.mark.asyncio
    async def test_model_stream_response_empty_stream(self):
        """Test consuming from empty ModelStreamResponse."""
        response = ModelStreamResponse()
        response.add(None)

        consumed_chunks = []
        async for chunk in response.consume():
            consumed_chunks.append(chunk)

        assert consumed_chunks == []
