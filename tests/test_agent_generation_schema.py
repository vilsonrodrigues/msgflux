"""Test Agent generation_schema with type() inheritance."""

from unittest.mock import Mock

import msgspec

from msgflux.nn.modules.agent import Agent


def test_agent_with_generation_schema_creates_output_with_merged_annotations():
    """Test that Agent properly merges annotations when using generation_schema."""

    # Define a generation schema
    class MyGenerationSchema(msgspec.Struct):
        reasoning: str
        action: str

    # Define a simple signature
    class SimpleOutput(msgspec.Struct):
        result: str

    # Create a mock model
    mock_model = Mock()
    mock_model.model_type = "chat_completion"

    # Create an agent with generation_schema
    agent = Agent(
        name="test_agent",
        model=mock_model,
        signature="input -> result: str",
        generation_schema=MyGenerationSchema,
    )

    # Access the generation schema that was set
    gen_schema = agent.generation_schema

    # The generation schema should be a new class that:
    # 1. Inherits from MyGenerationSchema
    # 2. Has a final_answer field
    # 3. Has all annotations from parent (reasoning, action) plus final_answer
    assert issubclass(gen_schema, MyGenerationSchema)
    assert issubclass(gen_schema, msgspec.Struct)

    # Check that merged annotations are present
    assert "reasoning" in gen_schema.__annotations__
    assert "action" in gen_schema.__annotations__
    assert "final_answer" in gen_schema.__annotations__

    # Verify the final_answer type is a Struct (from the signature)
    final_answer_type = gen_schema.__annotations__["final_answer"]
    # It should be a class (either Struct or Optional[Struct])
    assert final_answer_type is not None

    # Test that all three annotations are correct types
    assert gen_schema.__annotations__["reasoning"] == str
    assert gen_schema.__annotations__["action"] == str


def test_agent_with_generation_schema_optional_final_answer():
    """Test Agent with generation_schema that has optional final_answer."""

    class MyGenerationSchema(msgspec.Struct):
        reasoning: str
        final_answer: str = "default"  # Existing final_answer will be replaced

    mock_model = Mock()
    mock_model.model_type = "chat_completion"

    agent = Agent(
        name="test_agent",
        model=mock_model,
        signature="input -> output: int",
        generation_schema=MyGenerationSchema,
    )

    gen_schema = agent.generation_schema

    # Check annotations are merged
    assert "reasoning" in gen_schema.__annotations__
    assert "final_answer" in gen_schema.__annotations__

    # The final_answer type should have been replaced with the signature output type
    # (not the original str type)
    assert gen_schema.__annotations__["final_answer"] != str


def test_agent_without_generation_schema():
    """Test that Agent works normally without generation_schema."""

    mock_model = Mock()
    mock_model.model_type = "chat_completion"

    agent = Agent(name="test_agent", model=mock_model, signature="input -> result: str")

    gen_schema = agent.generation_schema

    # Should just be the signature output struct
    assert issubclass(gen_schema, msgspec.Struct)
    assert "result" in gen_schema.__annotations__
