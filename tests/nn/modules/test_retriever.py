"""Tests for msgflux.nn.modules.retriever module."""

import pytest
from unittest.mock import Mock, MagicMock

from msgflux.nn.modules.retriever import Retriever
from msgflux.nn.modules.embedder import Embedder
from msgflux.message import Message
from msgflux.models.base import BaseModel
from msgflux.models.response import ModelResponse
from msgflux.data.retrievers.types import LexicalRetriever, SemanticRetriever


class MockEmbedderModel(BaseModel):
    """Mock embedder model for testing."""

    model_type = "text_embedder"
    provider = "mock"
    batch_support = True

    def __init__(self):
        self.model_id = "test-embedder"
        self._api_key = None
        self.model = None
        self.processor = None
        self.client = None

    def _initialize(self):
        pass

    def __call__(self, **kwargs):
        data = kwargs.get("data", [])
        if isinstance(data, list):
            embeddings = [[0.1, 0.2, 0.3] for _ in data]
        else:
            embeddings = [[0.1, 0.2, 0.3]]
        response = ModelResponse()
        response.set_response_type("embeddings")
        response.add(embeddings)
        return response

    async def acall(self, **kwargs):
        return self(**kwargs)


class MockLexicalRetriever(LexicalRetriever):
    """Mock lexical retriever for testing."""

    def __call__(self, queries, **kwargs):
        """Return mock results for each query."""
        results = []
        for query in queries:
            results.append([
                {"data": f"Result 1 for {query}", "score": 0.95},
                {"data": f"Result 2 for {query}", "score": 0.85}
            ])
        return results


class MockSemanticRetriever(SemanticRetriever):
    """Mock semantic retriever for testing."""

    def __call__(self, queries, **kwargs):
        """Return mock results for embeddings."""
        results = []
        for _ in queries:
            results.append([
                {"data": "Semantic result 1", "score": 0.92},
                {"data": "Semantic result 2", "score": 0.88}
            ])
        return results


class TestRetriever:
    """Test suite for Retriever module."""

    def test_retriever_initialization(self):
        """Test Retriever basic initialization."""
        mock_retriever = MockLexicalRetriever()
        ret = Retriever(retriever=mock_retriever)

        assert ret.retriever is mock_retriever
        assert ret.embedder is None

    def test_retriever_initialization_with_config(self):
        """Test Retriever initialization with configuration."""
        mock_retriever = MockLexicalRetriever()
        config = {"top_k": 5, "threshold": 0.7}
        ret = Retriever(retriever=mock_retriever, config=config)

        assert ret.retriever is mock_retriever
        assert ret._buffers["config"]["top_k"] == 5
        assert ret._buffers["config"]["threshold"] == 0.7

    def test_retriever_inheritance_from_module(self):
        """Test that Retriever inherits from Module."""
        from msgflux.nn.modules.module import Module
        assert issubclass(Retriever, Module)

    def test_retriever_with_top_k(self):
        """Test Retriever with top_k configuration."""
        mock_retriever = MockLexicalRetriever()
        config = {"top_k": 3}
        ret = Retriever(retriever=mock_retriever, config=config)

        assert ret._buffers["config"]["top_k"] == 3

    def test_retriever_with_embedder_model(self):
        """Test Retriever with embedder model."""
        mock_retriever = MockSemanticRetriever()
        mock_model = MockEmbedderModel()
        ret = Retriever(retriever=mock_retriever, model=mock_model)

        assert ret.embedder is not None
        assert isinstance(ret.embedder, Embedder)
        assert ret.model is mock_model

    def test_retriever_with_embedder_instance(self):
        """Test Retriever with Embedder instance."""
        mock_retriever = MockSemanticRetriever()
        mock_model = MockEmbedderModel()
        embedder = Embedder(model=mock_model)
        ret = Retriever(retriever=mock_retriever, model=embedder)

        assert ret.embedder is embedder

    def test_retriever_forward_single_string(self):
        """Test Retriever forward with single string query."""
        mock_retriever = MockLexicalRetriever()
        ret = Retriever(retriever=mock_retriever)

        result = ret("test query")

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["query"] == "test query"
        assert len(result[0]["results"]) == 2

    def test_retriever_forward_list_of_strings(self):
        """Test Retriever forward with list of string queries."""
        mock_retriever = MockLexicalRetriever()
        ret = Retriever(retriever=mock_retriever)

        result = ret(["query1", "query2"])

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["query"] == "query1"
        assert result[1]["query"] == "query2"

    def test_retriever_with_embedder(self):
        """Test Retriever with embedder for semantic search."""
        mock_retriever = MockSemanticRetriever()
        mock_model = MockEmbedderModel()
        ret = Retriever(retriever=mock_retriever, model=mock_model)

        result = ret("semantic query")

        assert isinstance(result, list)
        assert len(result) == 1
        assert len(result[0]["results"]) == 2

    def test_retriever_forward_with_message(self):
        """Test Retriever forward with Message object."""
        mock_retriever = MockLexicalRetriever()
        ret = Retriever(
            retriever=mock_retriever,
            message_fields={"task_inputs": "query"}
        )

        msg = Message()
        msg.query = "message query"
        result = ret(msg)

        assert isinstance(result, list)

    def test_retriever_forward_list_of_dicts(self):
        """Test Retriever forward with list of dictionaries."""
        mock_retriever = MockLexicalRetriever()
        config = {"dict_key": "text"}
        ret = Retriever(retriever=mock_retriever, config=config)

        queries = [{"text": "query1"}, {"text": "query2"}]
        result = ret(queries)

        assert len(result) == 2

    def test_retriever_list_of_dicts_missing_dict_key(self):
        """Test Retriever raises error when dict_key is missing."""
        mock_retriever = MockLexicalRetriever()
        ret = Retriever(retriever=mock_retriever)

        queries = [{"text": "query1"}]
        with pytest.raises(AttributeError, match="require a `dict_key`"):
            ret(queries)

    @pytest.mark.asyncio
    async def test_retriever_aforward_single_string(self):
        """Test Retriever async forward with single string."""
        mock_retriever = MockLexicalRetriever()
        ret = Retriever(retriever=mock_retriever)

        result = await ret.aforward("async query")

        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_retriever_aforward_with_embedder(self):
        """Test Retriever async forward with embedder."""
        mock_retriever = MockSemanticRetriever()
        mock_model = MockEmbedderModel()
        ret = Retriever(retriever=mock_retriever, model=mock_model)

        result = await ret.aforward("async semantic query")

        assert isinstance(result, list)
        assert len(result) == 1

    def test_retriever_config_top_k(self):
        """Test Retriever with top_k configuration."""
        mock_retriever = MockLexicalRetriever()
        config = {"top_k": 10}
        ret = Retriever(retriever=mock_retriever, config=config)

        params = ret._prepare_retriever_execution(["test"])
        assert params["top_k"] == 10

    def test_retriever_config_return_score(self):
        """Test Retriever with return_score configuration."""
        mock_retriever = MockLexicalRetriever()
        config = {"return_score": True}
        ret = Retriever(retriever=mock_retriever, config=config)

        params = ret._prepare_retriever_execution(["test"])
        assert params["return_score"] is True

    def test_retriever_config_threshold(self):
        """Test Retriever with threshold configuration."""
        mock_retriever = MockLexicalRetriever()
        config = {"threshold": 0.8}
        ret = Retriever(retriever=mock_retriever, config=config)

        params = ret._prepare_retriever_execution(["test"])
        assert params["threshold"] == 0.8

    def test_retriever_config_no_threshold(self):
        """Test Retriever without threshold configuration."""
        mock_retriever = MockLexicalRetriever()
        ret = Retriever(retriever=mock_retriever)

        params = ret._prepare_retriever_execution(["test"])
        assert "threshold" not in params

    def test_retriever_config_invalid_type(self):
        """Test Retriever raises TypeError for invalid config type."""
        mock_retriever = MockLexicalRetriever()

        with pytest.raises(TypeError, match="`config` must be a dict or None"):
            Retriever(retriever=mock_retriever, config="invalid")

    def test_retriever_invalid_retriever_type(self):
        """Test Retriever raises TypeError for invalid retriever type."""
        mock_retriever = Mock()

        with pytest.raises(TypeError, match="`retriever` requires"):
            Retriever(retriever=mock_retriever)

    def test_retriever_model_property_getter(self):
        """Test Retriever model property getter."""
        mock_retriever = MockSemanticRetriever()
        mock_model = MockEmbedderModel()
        ret = Retriever(retriever=mock_retriever, model=mock_model)

        assert ret.model is mock_model

    def test_retriever_model_property_getter_no_embedder(self):
        """Test Retriever model property getter with no embedder."""
        mock_retriever = MockLexicalRetriever()
        ret = Retriever(retriever=mock_retriever)

        assert ret.model is None

    def test_retriever_model_property_setter(self):
        """Test Retriever model property setter."""
        mock_retriever = MockSemanticRetriever()
        ret = Retriever(retriever=mock_retriever)

        mock_model = MockEmbedderModel()
        ret.model = mock_model

        assert ret.model is mock_model
        assert isinstance(ret.embedder, Embedder)

    def test_retriever_with_templates(self):
        """Test Retriever with templates."""
        mock_retriever = MockLexicalRetriever()
        templates = {"response": "Results: {{ content }}"}
        ret = Retriever(retriever=mock_retriever, templates=templates)

        assert ret.templates == templates

    def test_retriever_with_response_mode(self):
        """Test Retriever with custom response_mode."""
        mock_retriever = MockLexicalRetriever()
        ret = Retriever(retriever=mock_retriever, response_mode="custom.field")

        assert ret.response_mode == "custom.field"

    def test_retriever_inspect_embedder_params(self):
        """Test inspect_embedder_params method."""
        mock_retriever = MockSemanticRetriever()
        mock_model = MockEmbedderModel()
        ret = Retriever(retriever=mock_retriever, model=mock_model)

        params = ret.inspect_embedder_params("test query")

        assert params["queries"] == ["test query"]

    def test_retriever_inspect_embedder_params_no_embedder(self):
        """Test inspect_embedder_params returns empty dict without embedder."""
        mock_retriever = MockLexicalRetriever()
        ret = Retriever(retriever=mock_retriever)

        params = ret.inspect_embedder_params("test query")

        assert params == {}

    def test_retriever_result_formatting(self):
        """Test Retriever formats results correctly."""
        mock_retriever = MockLexicalRetriever()
        ret = Retriever(retriever=mock_retriever)

        result = ret("test")

        assert "results" in result[0]
        assert "query" in result[0]
        assert "data" in result[0]["results"][0]
        assert "score" in result[0]["results"][0]

    def test_retriever_with_name(self):
        """Test Retriever initialization with name."""
        mock_retriever = MockLexicalRetriever()
        ret = Retriever(retriever=mock_retriever, name="my_retriever")

        assert ret.name == "my_retriever"

    def test_retriever_semantic_type(self):
        """Test Retriever with SemanticRetriever type."""
        mock_retriever = MockSemanticRetriever()
        ret = Retriever(retriever=mock_retriever)

        assert isinstance(ret.retriever, SemanticRetriever)

    def test_retriever_lexical_type(self):
        """Test Retriever with LexicalRetriever type."""
        mock_retriever = MockLexicalRetriever()
        ret = Retriever(retriever=mock_retriever)

        assert isinstance(ret.retriever, LexicalRetriever)
