"""Code environment providers."""

from msgflux.environments.code.providers.deno_pyodide import DenoPyodideSandbox


def __getattr__(name: str):
    """Lazy import for optional providers.

    AgentSandboxPython and AgentSandboxShell require the agent-sandbox
    package which is an optional dependency.
    """
    if name == "AgentSandboxPython":
        from msgflux.environments.code.providers.agent_sandbox import (
            AgentSandboxPython,
        )

        return AgentSandboxPython
    if name == "AgentSandboxShell":
        from msgflux.environments.code.providers.agent_sandbox import AgentSandboxShell

        return AgentSandboxShell
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DenoPyodideSandbox", "AgentSandboxPython", "AgentSandboxShell"]
