"""
Test for Agent used as tool with signature.

This test validates that when an Agent with a signature is used as a tool,
the tool schema correctly includes parameters from the signature inputs.
"""

from typing import Literal
from unittest.mock import MagicMock

import msgflux as mf
from msgflux.nn import Agent


def create_mock_model():
    """Create a mock chat completion model."""
    model = MagicMock()
    model.model_type = "chat_completion"

    # Mock the model response
    mock_response = MagicMock()
    mock_response.response_type = "text_generation"
    mock_response.consume.return_value = "Mocked response"
    model.return_value = mock_response

    return model


def test_agent_as_tool_with_signature():
    """Test that agent with signature generates correct tool schema."""
    mock_model = create_mock_model()

    # Create signature
    class Classify(mf.Signature):
        """Classify sentiment of a given sentence."""

        sentence: str = mf.InputField()
        sentiment: Literal["positive", "negative", "neutral"] = mf.OutputField()
        confidence: float = mf.OutputField()

    # Define agent with signature (using AutoParams class attributes)
    class SentimentClassifier(Agent):
        """An agent specializing in sentiment analysis."""

        name = "SentimentClassifier"
        model = mock_model
        signature = Classify

    # Apply tool_config
    SentimentClassifier = mf.tool_config(return_direct=True)(SentimentClassifier)

    # Create assistant with the agent as tool
    assistant = Agent(
        name="Assistant",
        model=mock_model,
        tools=[SentimentClassifier],
    )

    # Check tool schemas
    schemas = assistant.tool_library.get_tool_json_schemas()

    # Validate schema
    assert len(schemas) == 1
    schema = schemas[0]

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "SentimentClassifier"
    assert (
        schema["function"]["description"]
        == "An agent specializing in sentiment analysis."
    )

    # Check parameters - should include 'sentence' from signature
    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    assert "sentence" in params["properties"]
    assert params["properties"]["sentence"]["type"] == "string"
    assert "sentence" in params["required"]

    print("✓ Test passed: Agent as tool with signature generates correct schema")


def test_agent_as_tool_with_signature_and_multimodal():
    """Test that multimodal types are excluded from tool schema."""
    mock_model = create_mock_model()

    # Create signature with multimodal type
    class ImageAnalysis(mf.Signature):
        """Analyze an image."""

        image: mf.Image = mf.InputField()
        text: str = mf.InputField()
        analysis: str = mf.OutputField()

    # Define agent with signature (using AutoParams class attributes)
    class ImageAnalyzer(Agent):
        """An agent for image analysis."""

        name = "ImageAnalyzer"
        model = mock_model
        signature = ImageAnalysis

    # Create assistant with the agent as tool
    assistant = Agent(
        name="Assistant",
        model=mock_model,
        tools=[ImageAnalyzer],
    )

    # Check tool schemas
    schemas = assistant.tool_library.get_tool_json_schemas()

    # Validate schema
    assert len(schemas) == 1
    params = schemas[0]["function"]["parameters"]

    # Should only have 'text' parameter (image excluded as multimodal)
    assert "text" in params["properties"]
    assert "image" not in params["properties"]
    assert params["properties"]["text"]["type"] == "string"

    print("✓ Test passed: Multimodal types excluded from agent tool schema")


def test_agent_as_tool_with_string_signature():
    """Test that agent with string signature generates correct tool schema."""
    mock_model = create_mock_model()

    # Define agent with string signature (using AutoParams class attributes)
    class Summarizer(Agent):
        """An agent for text summarization."""

        name = "Summarizer"
        model = mock_model
        signature = "text: str, max_length: int -> summary: str"

    # Create assistant with the agent as tool
    assistant = Agent(
        name="Assistant",
        model=mock_model,
        tools=[Summarizer],
    )

    # Check tool schemas
    schemas = assistant.tool_library.get_tool_json_schemas()

    # Validate schema
    assert len(schemas) == 1
    params = schemas[0]["function"]["parameters"]

    # Should have both parameters from string signature
    assert "text" in params["properties"]
    assert "max_length" in params["properties"]
    assert params["properties"]["text"]["type"] == "string"
    assert params["properties"]["max_length"]["type"] == "integer"

    print("✓ Test passed: Agent with string signature generates correct schema")


if __name__ == "__main__":
    test_agent_as_tool_with_signature()
    test_agent_as_tool_with_signature_and_multimodal()
    test_agent_as_tool_with_string_signature()

    print("\n✅ All tests passed! Agent as tool with signature validated.")
    print("\nFeature summary:")
    print("  1. Agents with signatures generate correct tool schemas")
    print("  2. Signature inputs are properly included as tool parameters")
    print("  3. Multimodal types are correctly excluded from tool schemas")
    print("  4. Works with both class-based and string signatures")
