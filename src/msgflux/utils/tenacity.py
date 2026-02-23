from __future__ import annotations

from tenacity import retry
from tenacity import stop_after_attempt as _stop_after_attempt
from tenacity import stop_after_delay as _stop_after_delay

from msgflux.envs import envs


def _noop_decorator(fn):
    """No-op decorator when retry is disabled."""
    return fn


def build_tool_retry(
    *,
    enabled: bool = True,
    attempts: int | None = None,
    delay: int | None = None,
):
    """Build a retry decorator for tools.

    Args:
        enabled: Whether retry is enabled. If False, returns a no-op decorator.
        attempts: Max retry attempts. Defaults to envs.tool_stop_after_attempt.
        delay: Max retry delay in seconds. Defaults to envs.tool_stop_after_delay.

    Returns:
        A tenacity retry decorator or a no-op decorator.
    """
    if not enabled:
        return _noop_decorator

    _attempts = attempts if attempts is not None else envs.tool_stop_after_attempt
    _delay = delay if delay is not None else envs.tool_stop_after_delay

    return _build_retry(attempts=_attempts, delay=_delay)


def build_model_retry(
    *,
    enabled: bool = True,
    attempts: int | None = None,
    delay: int | None = None,
):
    """Build a retry decorator for models.

    Args:
        enabled: Whether retry is enabled. If False, returns a no-op decorator.
        attempts: Max retry attempts. Defaults to envs.model_stop_after_attempt.
        delay: Max retry delay in seconds. Defaults to envs.model_stop_after_delay.

    Returns:
        A tenacity retry decorator or a no-op decorator.
    """
    if not enabled:
        return _noop_decorator

    _attempts = attempts if attempts is not None else envs.model_stop_after_attempt
    _delay = delay if delay is not None else envs.model_stop_after_delay

    return _build_retry(attempts=_attempts, delay=_delay)


def _build_retry(*, attempts: int, delay: int):
    """Build a tenacity retry decorator with the given stop conditions."""
    stop_conditions = []
    if attempts > 0:
        stop_conditions.append(_stop_after_attempt(attempts))
    if delay > 0:
        stop_conditions.append(_stop_after_delay(delay))

    if not stop_conditions:
        return _noop_decorator

    stop = stop_conditions[0]
    for condition in stop_conditions[1:]:
        stop = stop | condition

    return retry(reraise=True, stop=stop)


# Static decorators for backward compatibility (used by model providers)
model_retry = build_model_retry()
tool_retry = build_tool_retry()
