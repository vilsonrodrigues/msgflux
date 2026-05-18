# /// script
# dependencies = []
# ///
# ruff: noqa: T201

import argparse
import asyncio
import tempfile
from pathlib import Path

import msgflux as mf
from msgflux import nn
from msgflux.models.response import ModelResponse
from msgflux.models.tool_call_agg import ToolCallAggregator
from msgflux.utils.msgspec import msgspec_dumps

DEFAULT_ARTIFACT = """msgFlux runtime notes
session_id identifies the durable conversation scope.
run_id identifies the resumable execution attempt inside the session.
namespace identifies which module or agent writes runtime state.
Artifacts should be read in bounded slices instead of loading the whole file.
"""


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


class ArtifactReaderModel:
    """Deterministic model used to make the example runnable offline."""

    model_type = "chat_completion"

    def __init__(self) -> None:
        self._called_tool = False

    def __call__(self, **kwargs: object) -> ModelResponse:
        if not self._called_tool:
            self._called_tool = True
            code = """
info = artifacts.info("runtime_notes")
chunk = artifacts.read("runtime_notes", offset=0, limit=500)
print(f"artifact={info['name']} size={info['size']} {info['unit']}")
result = chunk
""".strip()
            return tool_call_response(
                "python_interpreter",
                {"code": code},
                call_id="call_python_interpreter",
            )

        messages = kwargs.get("messages")
        tool_output = _last_tool_output(messages)
        return text_response(
            "The artifact was mounted through Agent.acall(..., artifacts=...).\n\n"
            f"Code interpreter output:\n{tool_output}"
        )

    async def acall(self, **kwargs: object) -> ModelResponse:
        return self(**kwargs)


def _last_tool_output(messages: object) -> str:
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") == "tool":
            content = message.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""


def build_agent() -> nn.Agent:
    return nn.Agent(
        name="artifact_reader_agent",
        model=ArtifactReaderModel(),
        code_interpreter=mf.Sandbox.python("local"),
        config={
            "verbose": True,
            "code_interpreter": {
                "ptc": True,
                "artifacts": True,
                "ptc_tools": {"allow": "*"},
            },
        },
        instructions=(
            "Use python_interpreter to inspect mounted artifacts through the "
            "artifacts namespace. Always use bounded reads with limit."
        ),
    )


async def run_with_artifact(artifact_path: Path) -> None:
    agent = build_agent()
    response = await agent.acall(
        (
            "Read the uploaded runtime notes artifact using the local Python "
            "sandbox. Report the artifact size and the first bounded slice."
        ),
        artifacts={"runtime_notes": artifact_path},
    )
    print(response)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-path",
        type=Path,
        help="Optional local file to mount as the runtime_notes artifact.",
    )
    args = parser.parse_args()

    if args.artifact_path is not None:
        await run_with_artifact(args.artifact_path)
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_path = Path(temp_dir) / "runtime_notes.txt"
        artifact_path.write_text(DEFAULT_ARTIFACT, encoding="utf-8")
        await run_with_artifact(artifact_path)


if __name__ == "__main__":
    asyncio.run(main())
