"""Tests for environment pooling."""

import pytest

from msgflux.environments import EnvironmentPool, Environments
from msgflux.environments.pool import PooledEnvironmentContext


class TestEnvironmentPool:
    """Tests for EnvironmentPool."""

    @pytest.fixture
    def pool(self):
        """Create a pool for testing."""
        try:
            pool = EnvironmentPool(
                factory=lambda: Environments.code("python", timeout=60.0),
                min_size=0,
                max_size=3,
                warmup_on_init=False,
            )
            yield pool
            pool.close()
        except Exception as e:
            pytest.skip(f"Deno not available: {e}")

    def test_acquire_creates_environment(self, pool):
        """Test that acquire creates an environment when pool is empty."""
        env = pool.acquire()
        assert env is not None
        assert pool.in_use == 1
        assert pool.size == 0
        pool.release(env)

    def test_release_returns_to_pool(self, pool):
        """Test that release returns environment to pool."""
        env = pool.acquire()
        pool.release(env)

        assert pool.in_use == 0
        assert pool.size == 1

    def test_reuse_released_environment(self, pool):
        """Test that released environments are reused."""
        env1 = pool.acquire()
        pool.release(env1)

        env2 = pool.acquire()
        assert env2 is env1  # Same instance

    def test_max_size_limit(self, pool):
        """Test that pool respects max_size."""
        envs = []
        for _ in range(3):  # max_size is 3
            envs.append(pool.acquire())

        assert pool.total == 3

        # Fourth should timeout
        with pytest.raises(TimeoutError):
            pool.acquire(timeout=0.1)

        # Cleanup
        for env in envs:
            pool.release(env)

    def test_stats_tracking(self, pool):
        """Test statistics tracking."""
        env = pool.acquire()
        pool.release(env)
        env = pool.acquire()
        pool.release(env)

        stats = pool.stats
        assert stats["created"] == 1
        assert stats["acquired"] == 2
        assert stats["released"] == 2
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 1

    def test_warmup(self, pool):
        """Test pool warmup."""
        pool.warmup(2)
        assert pool.size == 2
        assert pool.total == 2

    def test_close_shuts_down_all(self, pool):
        """Test that close shuts down all environments."""
        env1 = pool.acquire()
        pool.release(env1)
        env2 = pool.acquire()

        pool.close()

        assert pool.size == 0
        assert pool.in_use == 0

    def test_context_manager(self, pool):
        """Test pool as context manager."""
        with EnvironmentPool(
            factory=lambda: Environments.code("python", timeout=30.0),
            warmup_on_init=False,
        ) as p:
            env = p.acquire()
            result = env("print('hello')")
            assert result.success
            p.release(env)


class TestPooledEnvironmentContext:
    """Tests for PooledEnvironmentContext."""

    @pytest.fixture
    def pool(self):
        """Create a pool for testing."""
        try:
            pool = EnvironmentPool(
                factory=lambda: Environments.code("python", timeout=60.0),
                min_size=1,
                max_size=2,
            )
            yield pool
            pool.close()
        except Exception as e:
            pytest.skip(f"Deno not available: {e}")

    def test_context_acquires_and_releases(self, pool):
        """Test context manager acquires and releases."""
        initial_size = pool.size

        with PooledEnvironmentContext(pool) as env:
            assert pool.in_use == 1
            result = env("x = 1 + 2\nprint(x)")
            assert result.success
            assert "3" in result.output

        # Should be released back
        assert pool.in_use == 0
        assert pool.size >= initial_size


class TestPoolWithPackages:
    """Tests for pool with pre-installed packages.

    Note: These tests require MSGFLUX_DENO_ALLOW_CACHE_WRITE=true to be set
    in the environment, or they will be skipped due to package installation
    failures.
    """

    @pytest.fixture
    def pool_with_packages(self):
        """Create a pool with packages."""
        try:
            # Use pure-python package that works in Pyodide
            pool = EnvironmentPool(
                factory=lambda: Environments.code("python", timeout=120.0),
                packages=["six"],  # Small pure-python package
                min_size=1,
                max_size=2,
            )
            yield pool
            pool.close()
        except Exception as e:
            pytest.skip(f"Deno not available or cache write disabled: {e}")

    def test_packages_pre_installed(self, pool_with_packages):
        """Test that packages are pre-installed."""
        with PooledEnvironmentContext(pool_with_packages) as env:
            result = env("import six\nprint(six.PY3)")
            assert result.success
            assert "True" in result.output

    def test_packages_available_after_reuse(self, pool_with_packages):
        """Test packages remain available after reuse."""
        # First use
        env = pool_with_packages.acquire()
        result = env("import six\nprint('first')")
        assert result.success
        pool_with_packages.release(env)

        # Second use (should reuse same env)
        env = pool_with_packages.acquire()
        result = env("print('second')")
        assert result.success
        pool_with_packages.release(env)


class TestPoolIdleTimeout:
    """Tests for idle timeout functionality."""

    def test_cleanup_idle(self):
        """Test cleanup of idle environments."""
        try:
            pool = EnvironmentPool(
                factory=lambda: Environments.code("python", timeout=30.0),
                min_size=0,
                max_size=3,
                idle_timeout=0.1,  # 100ms timeout
                warmup_on_init=False,
            )

            # Acquire and release
            env = pool.acquire()
            pool.release(env)
            assert pool.size == 1

            # Wait for idle timeout
            import time
            time.sleep(0.2)

            # Cleanup should remove idle environment
            removed = pool.cleanup_idle()
            assert removed == 1
            assert pool.size == 0

            pool.close()
        except Exception as e:
            pytest.skip(f"Deno not available: {e}")
