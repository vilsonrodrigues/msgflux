"""Environment pooling for efficient resource reuse.

This module provides a pool of pre-initialized environments that can be
reused across multiple executions, avoiding the overhead of creating
new environments for each task.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, TypeVar

from msgflux.environments.base import BaseEnvironment
from msgflux.environments.code.base import BaseCodeEnvironment
from msgflux.logger import logger

T = TypeVar("T", bound=BaseEnvironment)


@dataclass
class PooledEnvironment:
    """Wrapper for a pooled environment with metadata."""

    environment: BaseCodeEnvironment
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    use_count: int = 0
    packages: List[str] = field(default_factory=list)

    def mark_used(self) -> None:
        """Mark this environment as used."""
        self.last_used_at = time.time()
        self.use_count += 1


class EnvironmentPool:
    """Pool of reusable code execution environments.

    The pool maintains a collection of pre-initialized environments that can
    be acquired for use and released back to the pool. This avoids the
    overhead of creating new environments for each execution.

    Features:
    - Pre-warming: Create environments with packages installed ahead of time
    - Automatic cleanup: Reset environments when released
    - Size limits: Control maximum pool size
    - Idle timeout: Optionally remove idle environments

    Example:
        >>> from msgflux.environments import Environments
        >>> from msgflux.environments.pool import EnvironmentPool
        >>>
        >>> # Create a pool with pre-installed packages
        >>> pool = EnvironmentPool(
        ...     factory=lambda: Environments.code("python/deno_pyodide"),
        ...     packages=["numpy", "pandas"],
        ...     min_size=2,
        ...     max_size=5,
        ... )
        >>>
        >>> # Acquire an environment from the pool
        >>> env = pool.acquire()
        >>> result = env("import numpy; print(numpy.__version__)")
        >>>
        >>> # Release back to pool
        >>> pool.release(env)
        >>>
        >>> # Or use as context manager
        >>> with pool.acquire() as env:
        ...     result = env("print('hello')")
    """

    def __init__(
        self,
        factory: Callable[[], BaseCodeEnvironment],
        *,
        packages: Optional[List[str]] = None,
        min_size: int = 0,
        max_size: int = 10,
        idle_timeout: Optional[float] = None,
        warmup_on_init: bool = True,
    ):
        """Initialize the environment pool.

        Args:
            factory:
                Callable that creates a new environment instance.
            packages:
                List of packages to pre-install in pooled environments.
            min_size:
                Minimum number of environments to keep in the pool.
                These are created during warmup.
            max_size:
                Maximum number of environments in the pool.
            idle_timeout:
                Optional timeout in seconds after which idle environments
                are removed (except min_size). None means no timeout.
            warmup_on_init:
                Whether to warm up the pool during initialization.
        """
        self._factory = factory
        self._packages = packages or []
        self._min_size = min_size
        self._max_size = max_size
        self._idle_timeout = idle_timeout

        self._pool: deque[PooledEnvironment] = deque()
        self._in_use: Dict[int, PooledEnvironment] = {}
        self._lock = threading.RLock()
        self._closed = False

        # Statistics
        self._stats = {
            "created": 0,
            "acquired": 0,
            "released": 0,
            "destroyed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        if warmup_on_init and min_size > 0:
            self._warmup(min_size)

    def _warmup(self, count: int) -> None:
        """Create and warm up environments.

        Args:
            count: Number of environments to create.
        """
        logger.info(f"Warming up {count} environments...")
        for i in range(count):
            try:
                pooled = self._create_pooled_environment()
                self._pool.append(pooled)
                logger.debug(f"Warmed up environment {i + 1}/{count}")
            except Exception as e:
                logger.error(f"Failed to warm up environment {i + 1}: {e}")
                break
        logger.info(f"Pool warmup complete: {len(self._pool)} environments ready")

    def _create_pooled_environment(self) -> PooledEnvironment:
        """Create a new pooled environment with packages installed."""
        env = self._factory()
        self._stats["created"] += 1

        # Install packages
        installed = []
        for package in self._packages:
            try:
                if env.install_package(package):
                    installed.append(package)
                    logger.debug(f"Pre-installed package: {package}")
            except Exception as e:
                logger.warning(f"Failed to install {package}: {e}")

        return PooledEnvironment(environment=env, packages=installed)

    def acquire(
        self,
        timeout: Optional[float] = None,
    ) -> BaseCodeEnvironment:
        """Acquire an environment from the pool.

        Args:
            timeout:
                Maximum time to wait for an available environment.
                None means wait indefinitely.

        Returns:
            An environment ready for use.

        Raises:
            RuntimeError: If pool is closed.
            TimeoutError: If no environment available within timeout.
        """
        if self._closed:
            raise RuntimeError("Pool is closed")

        start_time = time.time()

        with self._lock:
            # Try to get from pool
            while self._pool:
                pooled = self._pool.popleft()

                # Check idle timeout
                if self._idle_timeout is not None:
                    idle_time = time.time() - pooled.last_used_at
                    if idle_time > self._idle_timeout:
                        self._destroy_environment(pooled)
                        continue

                # Found a valid environment
                pooled.mark_used()
                self._in_use[id(pooled.environment)] = pooled
                self._stats["acquired"] += 1
                self._stats["cache_hits"] += 1
                logger.debug(f"Acquired pooled environment (uses: {pooled.use_count})")
                return pooled.environment

            # No available environment - check if we can create a new one
            total = len(self._pool) + len(self._in_use)
            if total < self._max_size:
                pooled = self._create_pooled_environment()
                pooled.mark_used()
                self._in_use[id(pooled.environment)] = pooled
                self._stats["acquired"] += 1
                self._stats["cache_misses"] += 1
                logger.debug("Created new environment for pool")
                return pooled.environment

        # Pool is full and all environments are in use
        # Wait for one to be released
        while True:
            elapsed = time.time() - start_time
            if timeout is not None and elapsed >= timeout:
                raise TimeoutError("No environment available within timeout")

            time.sleep(0.1)

            with self._lock:
                if self._pool:
                    pooled = self._pool.popleft()
                    pooled.mark_used()
                    self._in_use[id(pooled.environment)] = pooled
                    self._stats["acquired"] += 1
                    self._stats["cache_hits"] += 1
                    return pooled.environment

    def release(self, env: BaseCodeEnvironment) -> None:
        """Release an environment back to the pool.

        The environment is reset before being returned to the pool.

        Args:
            env: The environment to release.
        """
        if self._closed:
            try:
                env.shutdown()
            except Exception:  # noqa: S110
                pass  # Ignore shutdown errors when pool is closed
            return

        with self._lock:
            env_id = id(env)
            if env_id not in self._in_use:
                logger.warning("Attempted to release unknown environment")
                return

            pooled = self._in_use.pop(env_id)
            self._stats["released"] += 1

            # Reset the environment for reuse
            try:
                pooled.environment.reset()
            except Exception as e:
                logger.warning(f"Failed to reset environment: {e}")
                self._destroy_environment(pooled)
                return

            # Check if we should keep it (respect max_size)
            if len(self._pool) < self._max_size:
                self._pool.append(pooled)
                pool_size = len(self._pool)
                logger.debug(f"Released environment to pool (size: {pool_size})")
            else:
                self._destroy_environment(pooled)

    def _destroy_environment(self, pooled: PooledEnvironment) -> None:
        """Destroy a pooled environment."""
        try:
            pooled.environment.shutdown()
        except Exception as e:
            logger.debug(f"Error shutting down environment: {e}")
        self._stats["destroyed"] += 1

    @property
    def size(self) -> int:
        """Current number of available environments in the pool."""
        with self._lock:
            return len(self._pool)

    @property
    def in_use(self) -> int:
        """Number of environments currently in use."""
        with self._lock:
            return len(self._in_use)

    @property
    def total(self) -> int:
        """Total number of environments (available + in use)."""
        with self._lock:
            return len(self._pool) + len(self._in_use)

    @property
    def stats(self) -> Dict[str, int]:
        """Pool statistics."""
        with self._lock:
            return {
                **self._stats,
                "pool_size": len(self._pool),
                "in_use": len(self._in_use),
            }

    def warmup(self, count: Optional[int] = None) -> None:
        """Warm up additional environments.

        Args:
            count: Number of environments to create. Defaults to min_size.
        """
        if self._closed:
            raise RuntimeError("Pool is closed")

        count = count or self._min_size
        with self._lock:
            current = len(self._pool) + len(self._in_use)
            to_create = min(count, self._max_size - current)
            if to_create > 0:
                self._warmup(to_create)

    def cleanup_idle(self) -> int:
        """Remove idle environments beyond min_size.

        Returns:
            Number of environments removed.
        """
        if self._idle_timeout is None:
            return 0

        removed = 0
        current_time = time.time()

        with self._lock:
            # Keep at least min_size
            while len(self._pool) > self._min_size:
                # Check oldest (leftmost)
                if not self._pool:
                    break

                pooled = self._pool[0]
                idle_time = current_time - pooled.last_used_at

                if idle_time > self._idle_timeout:
                    self._pool.popleft()
                    self._destroy_environment(pooled)
                    removed += 1
                else:
                    # Environments are ordered by last use, so if this one
                    # isn't idle, none of the others will be either
                    break

        if removed > 0:
            logger.debug(f"Cleaned up {removed} idle environments")

        return removed

    def close(self) -> None:
        """Close the pool and shutdown all environments."""
        with self._lock:
            self._closed = True

            # Shutdown all pooled environments
            while self._pool:
                pooled = self._pool.popleft()
                self._destroy_environment(pooled)

            # Shutdown all in-use environments
            for pooled in self._in_use.values():
                self._destroy_environment(pooled)
            self._in_use.clear()

        logger.info("Environment pool closed")

    def __enter__(self) -> "EnvironmentPool":
        """Context manager entry."""
        return self

    def __exit__(self, *_) -> None:
        """Context manager exit with cleanup."""
        self.close()


class PooledEnvironmentContext:
    """Context manager for acquiring an environment from a pool.

    This provides a convenient way to acquire and automatically release
    an environment using the `with` statement.

    Example:
        >>> with pool.context() as env:
        ...     result = env("print('hello')")
    """

    def __init__(self, pool: EnvironmentPool, timeout: Optional[float] = None):
        self._pool = pool
        self._timeout = timeout
        self._env: Optional[BaseCodeEnvironment] = None

    def __enter__(self) -> BaseCodeEnvironment:
        self._env = self._pool.acquire(timeout=self._timeout)
        return self._env

    def __exit__(self, *_) -> None:
        if self._env is not None:
            self._pool.release(self._env)
            self._env = None
