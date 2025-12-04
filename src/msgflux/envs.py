import os
from typing import Any, Literal

from msgspec_ext import BaseSettings, SettingsConfigDict


def set_envs(**kwargs: Any):
    """Sets environment variables based on named arguments.

    Args:
        kwargs:
            Named arguments where name is the key of the
            environment variable and value is the value
            to be assigned.

    !!! example
        ```python
        set_envs(VERBOSE=0, LOCAL_RANK=0)
        ```

    """
    for key, value in kwargs.items():
        os.environ[key.upper()] = str(value)


class EnvironmentVariables(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".msgflux_env",
        env_prefix="msgflux_",
    )

    # Max objects in cache to functions
    # max_lru_cache: int = 16

    # If set to False, msgflux will not print logs
    # If set to True, msgflux will print logs
    verbose: bool = False

    # Logging configuration
    # If set to False, msgflux will not configure logging
    # If set to True, msgflux will configure logging using
    #    the default configuration or the configuration
    #    file specified by msgflux_LOGGING_CONFIG_PATH
    configure_logging: bool = True
    logging_config_path: str = None

    # Timeout in secounds to a tool execution
    # default is None, the functions has not Timeout
    tool_timeout: int = None

    # This is used for configuring the default logging level
    logging_level: str = "INFO"

    # if set, MSGFLUX_LOGGING_PREFIX will be prepended to all log messages
    logging_prefix: str = "MSGFLUX_"

    # Trace function calls
    # If set to True, msgflux will trace function calls. Useful for debugging
    trace_function: bool = False

    # Telemetry configuration (msgtrace-sdk)
    # if set, msgflux will track executions using msgtrace-sdk
    telemetry_enabled: bool = False

    # OTLP endpoint for msgtrace
    telemetry_otlp_endpoint: str = "http://localhost:8000/api/v1/traces/export"

    # Exporter type: "otlp" or "console"
    telemetry_exporter: Literal["console", "otlp"] = "otlp"

    # Service name for telemetry
    telemetry_service_name: str = "msgflux"

    # Capture platform details (CPU, OS, Python version)
    telemetry_capture_platform: bool = False

    # Capture tool call responses
    telemetry_capture_tool_call_responses: bool = True

    # Capture agent state, system prompt and tool schemas
    telemetry_capture_agent_prepare_model_execution: bool = False

    # Legacy support - maps to telemetry_enabled
    @property
    def telemetry_requires_trace(self) -> bool:
        """Legacy property for backward compatibility."""
        return self.telemetry_enabled

    # State checkpoint, if True, if a module output is in message, skip process
    # if False, reprocess
    state_checkpoint: bool = False

    # Max attemps to model clients
    model_stop_after_attempt: int = 5

    # Model retry delay
    model_stop_after_delay: int = 0

    # Max attemps to tool call
    tool_stop_after_attempt: int = 5

    # Tool retry delay
    tool_stop_after_delay: int = 0

    # Num threads to async pool executor
    executor_num_threads: int = 2

    # Num async workers. Each worker has an own eventloop
    executor_num_async_workers: int = 1

    # HTTPX max retries
    httpx_max_retries: int = 5


envs = EnvironmentVariables()


def configure_msgtrace_env():
    """Configure msgtrace-sdk environment variables from msgflux settings.

    This function maps msgflux telemetry settings to msgtrace-sdk
    environment variables, ensuring proper integration.
    """
    if envs.telemetry_enabled:
        os.environ["MSGTRACE_TELEMETRY_ENABLED"] = "true"
    else:
        os.environ["MSGTRACE_TELEMETRY_ENABLED"] = "false"

    os.environ["MSGTRACE_OTLP_ENDPOINT"] = envs.telemetry_otlp_endpoint
    os.environ["MSGTRACE_EXPORTER"] = envs.telemetry_exporter
    os.environ["MSGTRACE_SERVICE_NAME"] = envs.telemetry_service_name

    if envs.telemetry_capture_platform:
        os.environ["MSGTRACE_CAPTURE_PLATFORM"] = "true"
    else:
        os.environ["MSGTRACE_CAPTURE_PLATFORM"] = "false"


# Configure msgtrace on module import
configure_msgtrace_env()
