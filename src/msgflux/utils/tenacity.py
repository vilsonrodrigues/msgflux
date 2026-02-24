from tenacity import retry, stop_after_attempt, stop_after_delay

from msgflux.envs import envs

# Default retry decorator for tools
default_tool_retry = retry(
    reraise=True,
    stop=(
        stop_after_delay(envs.tool_stop_after_delay)
        | stop_after_attempt(envs.tool_stop_after_attempt)
    ),
)

# Default retry decorator for models
default_model_retry = retry(
    reraise=True,
    stop=(
        stop_after_delay(envs.model_stop_after_delay)
        | stop_after_attempt(envs.model_stop_after_attempt)
    ),
)


def apply_retry(method, retry_config=None, *, default):
    """Apply a retry decorator to a method.

    Args:
        method: The method to wrap with retry.
        retry_config: A tenacity retry decorator, False to disable, or None for default.
        default: The default retry decorator to use when retry_config is None.

    Returns:
        The method wrapped with retry, or the original method if disabled.
    """
    if retry_config is False:
        return method
    dec = retry_config if retry_config is not None else default
    return dec(method)
