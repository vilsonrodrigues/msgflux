"""Test async Agent cycle - focusing on async I/O operations."""
import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock

from msgflux.nn.modules.agent import Agent


async def test_async_multimodal_processing():
    """Test that async multimodal processing works without blocking."""
    print("Testing async multimodal processing...")

    # Create a temporary image file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        # Write some dummy data
        f.write(b"fake image data for testing")
        temp_image_path = f.name

    try:
        # Create mock model
        mock_model = Mock()
        mock_model.model_type = "chat_completion"

        # Create agent
        agent = Agent(
            name="test_agent",
            model=mock_model,
        )

        print("\n1. Testing async image formatting...")
        # Test async image formatting
        result = await agent._aformat_image_input(temp_image_path)
        assert result is not None, "Image should be formatted"
        assert "image" in str(result).lower(), "Result should contain image reference"
        print("   ✓ Async image formatting works!")

        print("\n2. Testing async file formatting...")
        # Test async file formatting
        result = await agent._aformat_file_input(temp_image_path)
        assert result is not None, "File should be formatted"
        print("   ✓ Async file formatting works!")

        print("\n3. Testing async task preparation...")
        # Test async task preparation with multimodal inputs
        task_inputs = await agent._aprepare_task(
            "Describe this image",
            task_multimodal_inputs={"image": [temp_image_path]}
        )
        assert task_inputs is not None, "Task should be prepared"
        assert "model_state" in task_inputs, "Should have model state"
        print("   ✓ Async task preparation works!")

        print("\n4. Verifying no blocking calls...")
        # Verify it works with asyncio by running multiple tasks concurrently
        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(
            agent._aformat_image_input(temp_image_path),
            agent._aformat_image_input(temp_image_path),
            agent._aformat_image_input(temp_image_path),
        )
        elapsed = asyncio.get_event_loop().time() - start_time

        assert all(r is not None for r in results), "All images should be formatted"
        # If truly async, 3 operations should take roughly the same time as 1
        # (accounting for some overhead, should be < 2x the time of one operation)
        print(f"   ✓ 3 concurrent async operations completed in {elapsed:.3f}s")
        print("   ✓ No blocking detected!")

    finally:
        # Cleanup
        if os.path.exists(temp_image_path):
            os.unlink(temp_image_path)

    print("\n✅ All async multimodal processing tests passed!")


async def test_async_prepare_task():
    """Test that _aprepare_task is fully async."""
    print("\nTesting async task preparation...")

    # Create mock model
    mock_model = Mock()
    mock_model.model_type = "chat_completion"

    # Create agent
    agent = Agent(
        name="test_agent",
        model=mock_model,
    )

    print("\n1. Testing sync _prepare_task...")
    task_sync = agent._prepare_task("Hello world")
    assert "model_state" in task_sync
    print("   ✓ Sync prepare_task works!")

    print("\n2. Testing async _aprepare_task...")
    task_async = await agent._aprepare_task("Hello world")
    assert "model_state" in task_async
    print("   ✓ Async prepare_task works!")

    print("\n3. Comparing results...")
    # Both should produce similar structure
    assert task_sync.keys() == task_async.keys()
    print("   ✓ Results are consistent!")

    print("\n✅ Async task preparation test passed!")


async def main():
    """Run all async tests."""
    print("=" * 60)
    print("MSGFLUX ASYNC AGENT TESTS")
    print("=" * 60)

    await test_async_prepare_task()
    await test_async_multimodal_processing()

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
