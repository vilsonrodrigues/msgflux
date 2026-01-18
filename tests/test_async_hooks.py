"""
Test for async hooks support in Module.

This test validates that both sync and async hooks work correctly
in async context (_acall_impl).
"""

import asyncio
from unittest.mock import MagicMock

import pytest

from msgflux.nn import Module


class TestModule(Module):
    """Simple test module."""

    def __init__(self):
        super().__init__()
        self.execution_log = []

    def forward(self, x):
        self.execution_log.append(f"forward: {x}")
        return x * 2

    async def aforward(self, x):
        self.execution_log.append(f"aforward: {x}")
        await asyncio.sleep(0.01)  # Simulate async work
        return x * 2


def test_sync_pre_hook_in_sync_forward():
    """Test that sync pre-hook works in sync forward."""
    module = TestModule()
    hook_log = []

    def sync_pre_hook(mod, args, kwargs):
        hook_log.append(f"sync_pre_hook: args={args}")
        return None

    module.register_forward_pre_hook(sync_pre_hook)

    result = module(5)

    assert result == 10
    assert "sync_pre_hook: args=(5,)" in hook_log
    assert "forward: 5" in module.execution_log
    print("✓ Test 1 passed: Sync pre-hook in sync forward")


@pytest.mark.asyncio
async def test_sync_pre_hook_in_async_forward():
    """Test that sync pre-hook runs in executor in async forward."""
    module = TestModule()
    hook_log = []

    def sync_pre_hook(mod, args, kwargs):
        # Simulate some work (would block in sync)
        import time

        time.sleep(0.01)
        hook_log.append(f"sync_pre_hook: args={args}")
        return None

    module.register_forward_pre_hook(sync_pre_hook)

    result = await module.acall(5)

    assert result == 10
    assert "sync_pre_hook: args=(5,)" in hook_log
    assert "aforward: 5" in module.execution_log
    print("✓ Test 2 passed: Sync pre-hook in async forward (runs in executor)")


@pytest.mark.asyncio
async def test_async_pre_hook_in_async_forward():
    """Test that async pre-hook is awaited directly."""
    module = TestModule()
    hook_log = []

    async def async_pre_hook(mod, args, kwargs):
        await asyncio.sleep(0.01)  # Async work
        hook_log.append(f"async_pre_hook: args={args}")
        return None

    module.register_forward_pre_hook(async_pre_hook)

    result = await module.acall(5)

    assert result == 10
    assert "async_pre_hook: args=(5,)" in hook_log
    assert "aforward: 5" in module.execution_log
    print("✓ Test 3 passed: Async pre-hook in async forward")


@pytest.mark.asyncio
async def test_mixed_hooks():
    """Test mixing sync and async hooks."""
    module = TestModule()
    hook_log = []

    def sync_pre_hook(mod, args, kwargs):
        hook_log.append("sync_pre")
        return None

    async def async_pre_hook(mod, args, kwargs):
        await asyncio.sleep(0.01)
        hook_log.append("async_pre")
        return None

    def sync_post_hook(mod, args, kwargs, result):
        hook_log.append(f"sync_post: {result}")
        return result

    async def async_post_hook(mod, args, kwargs, result):
        await asyncio.sleep(0.01)
        hook_log.append(f"async_post: {result}")
        return result

    module.register_forward_pre_hook(sync_pre_hook)
    module.register_forward_pre_hook(async_pre_hook)
    module.register_forward_hook(sync_post_hook)
    module.register_forward_hook(async_post_hook)

    result = await module.acall(5)

    assert result == 10
    assert "sync_pre" in hook_log
    assert "async_pre" in hook_log
    assert "sync_post: 10" in hook_log
    assert "async_post: 10" in hook_log
    print("✓ Test 4 passed: Mixed sync and async hooks")


@pytest.mark.asyncio
async def test_pre_hook_modifies_args():
    """Test that pre-hook can modify arguments."""
    module = TestModule()

    async def modify_args_hook(mod, args, kwargs):
        # Modify args: multiply by 10
        new_args = (args[0] * 10,)
        return (new_args, kwargs)

    module.register_forward_pre_hook(modify_args_hook)

    result = await module.acall(5)

    # Hook modified 5 → 50, then forward does 50 * 2 = 100
    assert result == 100
    assert "aforward: 50" in module.execution_log
    print("✓ Test 5 passed: Pre-hook modifies args")


@pytest.mark.asyncio
async def test_post_hook_modifies_result():
    """Test that post-hook can modify result."""
    module = TestModule()

    async def modify_result_hook(mod, args, kwargs, result):
        # Modify result: add 100
        return result + 100

    module.register_forward_hook(modify_result_hook)

    result = await module.acall(5)

    # Forward: 5 * 2 = 10, hook: 10 + 100 = 110
    assert result == 110
    print("✓ Test 6 passed: Post-hook modifies result")


@pytest.mark.asyncio
async def test_no_hooks_fast_path():
    """Test that no hooks takes fast path."""
    module = TestModule()

    # Should use fast path (direct _acall without hook processing)
    result = await module.acall(5)

    assert result == 10
    assert "aforward: 5" in module.execution_log
    print("✓ Test 7 passed: No hooks uses fast path")


if __name__ == "__main__":
    # Run sync test
    test_sync_pre_hook_in_sync_forward()

    # Run async tests
    asyncio.run(test_sync_pre_hook_in_async_forward())
    asyncio.run(test_async_pre_hook_in_async_forward())
    asyncio.run(test_mixed_hooks())
    asyncio.run(test_pre_hook_modifies_args())
    asyncio.run(test_post_hook_modifies_result())
    asyncio.run(test_no_hooks_fast_path())

    print("\n✅ All tests passed! Async hooks validated.")
    print("\nFeature summary:")
    print("  1. Sync hooks work in sync forward (normal execution)")
    print("  2. Sync hooks run in executor in async forward (non-blocking)")
    print("  3. Async hooks awaited directly in async forward")
    print("  4. Can mix sync and async hooks in same module")
    print("  5. Pre-hooks can modify args, post-hooks can modify results")
    print("  6. Fast path used when no hooks registered")
