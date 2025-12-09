"""Test async Agent cycle."""
import asyncio

from msgflux.nn.modules.agent import Agent


class MockModel:
    """Mock model for testing."""

    def __call__(self, **kwargs):
        """Sync call."""
        return type(
            "MockResponse",
            (),
            {
                "response_type": "text_generation",
                "data": "Hello from sync model!",
                "consume": lambda: "Hello from sync model!",
            },
        )()

    async def acall(self, **kwargs):
        """Async call."""
        await asyncio.sleep(0.1)  # Simulate async work
        return type(
            "MockResponse",
            (),
            {
                "response_type": "text_generation",
                "data": "Hello from async model!",
                "consume": lambda: "Hello from async model!",
            },
        )()


async def test_async_agent():
    """Test async agent flow."""
    print("Testing async Agent cycle...")

    # Create agent with mock model
    agent = Agent(
        name="test_agent",
        model=MockModel(),
    )

    # Test async forward
    print("\n1. Testing aforward with simple text...")
    response = await agent.aforward("What is 2+2?")
    print(f"   Response: {response}")
    assert response == "Hello from async model!"
    print("   ✓ Passed!")

    # Test sync forward for comparison
    print("\n2. Testing forward (sync) for comparison...")
    response_sync = agent.forward("What is 2+2?")
    print(f"   Response: {response_sync}")
    assert response_sync == "Hello from sync model!"
    print("   ✓ Passed!")

    print("\n✅ All async agent tests passed!")


if __name__ == "__main__":
    asyncio.run(test_async_agent())
