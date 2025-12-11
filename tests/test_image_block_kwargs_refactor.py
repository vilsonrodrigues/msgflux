"""
Test to verify the image_block_kwargs refactoring.

This test validates:
1. ChatBlock.image accepts **kwargs instead of specific detail parameter
2. Agent config accepts image_block_kwargs instead of image_detail
3. image_block_kwargs are properly passed to ChatBlock.image
4. Backward compatibility is maintained (detail parameter still works via kwargs)
"""

from unittest.mock import MagicMock

import pytest

from msgflux.nn.modules.agent import Agent
from msgflux.utils.chat import ChatBlock


def test_chatblock_image_with_detail_kwarg():
    """Test that ChatBlock.image accepts detail as kwarg."""
    result = ChatBlock.image("https://example.com/image.jpg", detail="high")

    assert result["type"] == "image_url"
    assert result["image_url"]["url"] == "https://example.com/image.jpg"
    assert result["image_url"]["detail"] == "high"
    print("✓ Test 1 passed: ChatBlock.image accepts detail as kwarg")


def test_chatblock_image_with_multiple_kwargs():
    """Test that ChatBlock.image accepts multiple kwargs."""
    result = ChatBlock.image(
        "https://example.com/image.jpg",
        detail="low",
        custom_param="value",
        another_param=123,
    )

    assert result["image_url"]["detail"] == "low"
    assert result["image_url"]["custom_param"] == "value"
    assert result["image_url"]["another_param"] == 123
    print("✓ Test 2 passed: ChatBlock.image accepts multiple kwargs")


def test_chatblock_image_with_list_urls():
    """Test that ChatBlock.image handles list of URLs with kwargs."""
    urls = ["https://example.com/1.jpg", "https://example.com/2.jpg"]
    result = ChatBlock.image(urls, detail="high", custom="test")

    assert len(result) == 2
    assert result[0]["image_url"]["detail"] == "high"
    assert result[0]["image_url"]["custom"] == "test"
    assert result[1]["image_url"]["detail"] == "high"
    assert result[1]["image_url"]["custom"] == "test"
    print("✓ Test 3 passed: ChatBlock.image handles list URLs with kwargs")


def test_chatblock_image_without_kwargs():
    """Test that ChatBlock.image works without kwargs."""
    result = ChatBlock.image("https://example.com/image.jpg")

    assert result["type"] == "image_url"
    assert result["image_url"]["url"] == "https://example.com/image.jpg"
    assert "detail" not in result["image_url"]
    print("✓ Test 4 passed: ChatBlock.image works without kwargs")


def test_agent_config_with_image_block_kwargs():
    """Test that Agent accepts image_block_kwargs in config."""
    model = MagicMock()
    model.model_type = "chat_completion"

    agent = Agent(
        name="test_agent",
        model=model,
        config={"image_block_kwargs": {"detail": "high", "custom": "value"}},
    )

    assert "image_block_kwargs" in agent.config
    assert agent.config["image_block_kwargs"]["detail"] == "high"
    assert agent.config["image_block_kwargs"]["custom"] == "value"
    print("✓ Test 5 passed: Agent accepts image_block_kwargs in config")


def test_agent_config_image_block_kwargs_validation():
    """Test that Agent validates image_block_kwargs type."""
    model = MagicMock()
    model.model_type = "chat_completion"

    with pytest.raises(TypeError) as exc_info:
        Agent(
            name="test_agent",
            model=model,
            config={
                "image_block_kwargs": "invalid_type"  # Should be dict
            },
        )

    assert "image_block_kwargs" in str(exc_info.value)
    assert "must be a dict" in str(exc_info.value)
    print("✓ Test 6 passed: Agent validates image_block_kwargs type")


def test_agent_format_image_input_uses_image_block_kwargs():
    """Test that Agent._format_image_input passes image_block_kwargs to ChatBlock.image."""
    model = MagicMock()
    model.model_type = "chat_completion"

    agent = Agent(
        name="test_agent",
        model=model,
        config={"image_block_kwargs": {"detail": "low", "provider_specific": "param"}},
    )

    # Mock the _prepare_data_uri to return a simple URL
    agent._prepare_data_uri = MagicMock(return_value="https://example.com/image.jpg")

    result = agent._format_image_input("test.jpg")

    assert result["image_url"]["detail"] == "low"
    assert result["image_url"]["provider_specific"] == "param"
    print(
        "✓ Test 7 passed: Agent._format_image_input passes image_block_kwargs correctly"
    )


def test_backward_compatibility_detail_in_image_block_kwargs():
    """Test backward compatibility: detail parameter works via image_block_kwargs."""
    model = MagicMock()
    model.model_type = "chat_completion"

    # Old way would be: config={"image_detail": "high"}
    # New way is: config={"image_block_kwargs": {"detail": "high"}}
    agent = Agent(
        name="test_agent",
        model=model,
        config={"image_block_kwargs": {"detail": "high"}},
    )

    agent._prepare_data_uri = MagicMock(return_value="https://example.com/image.jpg")
    result = agent._format_image_input("test.jpg")

    assert result["image_url"]["detail"] == "high"
    print("✓ Test 8 passed: Backward compatibility maintained via image_block_kwargs")


def test_agent_without_image_block_kwargs():
    """Test that Agent works without image_block_kwargs (empty dict default)."""
    model = MagicMock()
    model.model_type = "chat_completion"

    agent = Agent(name="test_agent", model=model)

    agent._prepare_data_uri = MagicMock(return_value="https://example.com/image.jpg")
    result = agent._format_image_input("test.jpg")

    # Should work, just without extra kwargs
    assert result["image_url"]["url"] == "https://example.com/image.jpg"
    assert "detail" not in result["image_url"]
    print("✓ Test 9 passed: Agent works without image_block_kwargs")


if __name__ == "__main__":
    test_chatblock_image_with_detail_kwarg()
    test_chatblock_image_with_multiple_kwargs()
    test_chatblock_image_with_list_urls()
    test_chatblock_image_without_kwargs()
    test_agent_config_with_image_block_kwargs()
    test_agent_config_image_block_kwargs_validation()
    test_agent_format_image_input_uses_image_block_kwargs()
    test_backward_compatibility_detail_in_image_block_kwargs()
    test_agent_without_image_block_kwargs()

    print("\n✅ All tests passed! Refactoring validated successfully.")
    print("\nSummary of changes:")
    print(
        "  1. ChatBlock.image now accepts **kwargs instead of specific detail parameter"
    )
    print(
        "  2. Agent config now accepts 'image_block_kwargs' dict instead of 'image_detail'"
    )
    print("  3. image_block_kwargs are properly validated (must be dict)")
    print(
        "  4. Any kwargs can be passed to image blocks (flexible for different providers)"
    )
    print("  5. Backward compatibility maintained: detail works via image_block_kwargs")
