"""
Test to verify the video_block_kwargs implementation.

This test validates:
1. ChatBlock.video accepts **kwargs
2. Agent config accepts video_block_kwargs
3. video_block_kwargs are properly passed to ChatBlock.video
4. Multiple provider-specific parameters can be passed
"""

from unittest.mock import MagicMock

import pytest

from msgflux.nn.modules.agent import Agent
from msgflux.utils.chat import ChatBlock


def test_chatblock_video_with_kwargs():
    """Test that ChatBlock.video accepts kwargs."""
    result = ChatBlock.video(
        "https://example.com/video.mp4", format="mp4", quality="high"
    )

    assert result["type"] == "video_url"
    assert result["video_url"]["url"] == "https://example.com/video.mp4"
    assert result["video_url"]["format"] == "mp4"
    assert result["video_url"]["quality"] == "high"
    print("✓ Test 1 passed: ChatBlock.video accepts kwargs")


def test_chatblock_video_with_multiple_kwargs():
    """Test that ChatBlock.video accepts multiple kwargs."""
    result = ChatBlock.video(
        "https://example.com/video.mp4",
        format="mp4",
        resolution="1920x1080",
        provider_param="value",
        custom_option=123,
    )

    assert result["video_url"]["format"] == "mp4"
    assert result["video_url"]["resolution"] == "1920x1080"
    assert result["video_url"]["provider_param"] == "value"
    assert result["video_url"]["custom_option"] == 123
    print("✓ Test 2 passed: ChatBlock.video accepts multiple kwargs")


def test_chatblock_video_with_list_urls():
    """Test that ChatBlock.video handles list of URLs with kwargs."""
    urls = ["https://example.com/1.mp4", "https://example.com/2.mp4"]
    result = ChatBlock.video(urls, format="mp4", quality="high")

    assert len(result) == 2
    assert result[0]["video_url"]["format"] == "mp4"
    assert result[0]["video_url"]["quality"] == "high"
    assert result[1]["video_url"]["format"] == "mp4"
    assert result[1]["video_url"]["quality"] == "high"
    print("✓ Test 3 passed: ChatBlock.video handles list URLs with kwargs")


def test_chatblock_video_without_kwargs():
    """Test that ChatBlock.video works without kwargs."""
    result = ChatBlock.video("https://example.com/video.mp4")

    assert result["type"] == "video_url"
    assert result["video_url"]["url"] == "https://example.com/video.mp4"
    assert len(result["video_url"]) == 1  # Only url
    print("✓ Test 4 passed: ChatBlock.video works without kwargs")


def test_agent_config_with_video_block_kwargs():
    """Test that Agent accepts video_block_kwargs in config."""
    model = MagicMock()
    model.model_type = "chat_completion"

    agent = Agent(
        name="test_agent",
        model=model,
        config={"video_block_kwargs": {"format": "mp4", "quality": "high"}},
    )

    assert "video_block_kwargs" in agent.config
    assert agent.config["video_block_kwargs"]["format"] == "mp4"
    assert agent.config["video_block_kwargs"]["quality"] == "high"
    print("✓ Test 5 passed: Agent accepts video_block_kwargs in config")


def test_agent_config_video_block_kwargs_validation():
    """Test that Agent validates video_block_kwargs type."""
    model = MagicMock()
    model.model_type = "chat_completion"

    with pytest.raises(TypeError) as exc_info:
        Agent(
            name="test_agent",
            model=model,
            config={
                "video_block_kwargs": "invalid_type"  # Should be dict
            },
        )

    assert "video_block_kwargs" in str(exc_info.value)
    assert "must be a dict" in str(exc_info.value)
    print("✓ Test 6 passed: Agent validates video_block_kwargs type")


def test_agent_format_video_input_uses_video_block_kwargs():
    """Test that Agent._format_video_input passes video_block_kwargs to ChatBlock.video."""
    model = MagicMock()
    model.model_type = "chat_completion"

    agent = Agent(
        name="test_agent",
        model=model,
        config={"video_block_kwargs": {"format": "mp4", "provider_specific": "param"}},
    )

    # Test with URL (first case)
    result = agent._format_video_input("https://example.com/video.mp4")
    assert result["video_url"]["format"] == "mp4"
    assert result["video_url"]["provider_specific"] == "param"
    print("✓ Test 7a passed: Agent._format_video_input (URL) passes video_block_kwargs")

    # Test with file path (second case - requires mocking _prepare_data_uri)
    agent._prepare_data_uri = MagicMock(return_value="base64encodedvideo")
    result = agent._format_video_input("test.mp4")
    assert result["video_url"]["format"] == "mp4"
    assert result["video_url"]["provider_specific"] == "param"
    print(
        "✓ Test 7b passed: Agent._format_video_input (file) passes video_block_kwargs"
    )


def test_agent_without_video_block_kwargs():
    """Test that Agent works without video_block_kwargs (empty dict default)."""
    model = MagicMock()
    model.model_type = "chat_completion"

    agent = Agent(name="test_agent", model=model)

    result = agent._format_video_input("https://example.com/video.mp4")
    assert result["video_url"]["url"] == "https://example.com/video.mp4"
    assert len(result["video_url"]) == 1  # Only url
    print("✓ Test 8 passed: Agent works without video_block_kwargs")


def test_agent_with_both_image_and_video_kwargs():
    """Test that Agent can have both image_block_kwargs and video_block_kwargs."""
    model = MagicMock()
    model.model_type = "chat_completion"

    agent = Agent(
        name="test_agent",
        model=model,
        config={
            "image_block_kwargs": {"detail": "high"},
            "video_block_kwargs": {"format": "mp4", "quality": "1080p"},
        },
    )

    assert agent.config["image_block_kwargs"]["detail"] == "high"
    assert agent.config["video_block_kwargs"]["format"] == "mp4"
    assert agent.config["video_block_kwargs"]["quality"] == "1080p"
    print(
        "✓ Test 9 passed: Agent supports both image_block_kwargs and video_block_kwargs"
    )


if __name__ == "__main__":
    test_chatblock_video_with_kwargs()
    test_chatblock_video_with_multiple_kwargs()
    test_chatblock_video_with_list_urls()
    test_chatblock_video_without_kwargs()
    test_agent_config_with_video_block_kwargs()
    test_agent_config_video_block_kwargs_validation()
    test_agent_format_video_input_uses_video_block_kwargs()
    test_agent_without_video_block_kwargs()
    test_agent_with_both_image_and_video_kwargs()

    print("\n✅ All tests passed! video_block_kwargs implemented successfully.")
    print("\nSummary:")
    print("  1. ChatBlock.video now accepts **kwargs")
    print("  2. Agent config accepts 'video_block_kwargs' dict")
    print("  3. video_block_kwargs are properly validated (must be dict)")
    print("  4. Any kwargs can be passed to video blocks (flexible for providers)")
    print("  5. Compatible with image_block_kwargs (both can be used together)")
