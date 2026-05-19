# /// script
# dependencies = ["just-bash"]
# ///
# ruff: noqa: T201

import asyncio
import tempfile
from pathlib import Path

import msgflux as mf
from msgflux import nn
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.utils.msgspec import msgspec_dumps


def tool_call_response(
    tool_name: str,
    parameters: dict[str, str],
    *,
    call_id: str,
) -> ModelResponse:
    response = ModelResponse()
    response.set_response_type("tool_call")
    aggregator = ToolCallAggregator()
    aggregator.process(0, call_id, tool_name, msgspec_dumps(parameters))
    response.add(aggregator)
    response.reasoning = None
    response.metadata = {}
    return response


def text_response(text: str) -> ModelResponse:
    response = ModelResponse()
    response.set_response_type("text_generation")
    response.add(text)
    response.reasoning = None
    response.metadata = {}
    return response


class ScriptedModel:
    model_type = "chat_completion"

    def __init__(self) -> None:
        self._called_tool = False

    def __call__(self, **kwargs: object) -> ModelResponse:  # noqa: ARG002
        if not self._called_tool:
            self._called_tool = True
            return tool_call_response(
                "shell",
                {"command": 'grep -R "session_id" /workspace/project | wc -l'},
                call_id="call_shell",
            )
        return text_response("The shell sandbox inspected the mounted project.")

    async def acall(self, **kwargs: object) -> ModelResponse:
        return self(**kwargs)


def build_agent(project_dir: Path) -> nn.Agent:
    shell = mf.Sandbox.shell(
        "just-bash",
        mounts={"/workspace/project": project_dir},
        cwd="/workspace",
    )
    return nn.Agent(
        name="shell_agent",
        model=ScriptedModel(),
        tools=[shell],
        config={"verbose": True},
        instructions=(
            "Use the shell tool to inspect files mounted in /workspace/project. "
            "Keep command output concise."
        ),
    )


async def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        project_dir = Path(temp_dir) / "project"
        project_dir.mkdir()
        (project_dir / "README.md").write_text(
            "session_id identifies the durable conversation scope.\n"
            "run_id identifies the resumable execution attempt.\n",
            encoding="utf-8",
        )
        agent = build_agent(project_dir)
        response = await agent.acall(
            "Count references to session_id in the mounted project."
        )
        print(response)


if __name__ == "__main__":
    asyncio.run(main())
