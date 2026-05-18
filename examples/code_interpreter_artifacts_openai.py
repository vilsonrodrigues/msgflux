# /// script
# dependencies = ["openai"]
# ///
# ruff: noqa: T201

import argparse
import asyncio
import tempfile
from pathlib import Path

import msgflux as mf
from msgflux import nn

mf.load_dotenv()

DEFAULT_ARTIFACT = """msgFlux runtime notes
session_id identifies the durable conversation scope.
run_id identifies the resumable execution attempt inside the session.
namespace identifies which module or agent writes runtime state.
Artifacts are mounted into the code interpreter as lazy file references.
Use artifacts.info(name) to inspect metadata.
Use artifacts.read(name, offset=0, limit=...) to read bounded slices.
Do not load whole large files into vars unless explicitly necessary.
"""


def build_agent(model_name: str) -> nn.Agent:
    return nn.Agent(
        name="openai_artifact_reader",
        model=mf.Model.chat_completion(model_name),
        code_interpreter=mf.Sandbox.python("local"),
        config={
            "verbose": True,
            "max_tool_turns": 6,
            "code_interpreter": {
                "ptc": True,
                "artifacts": True,
                "ptc_tools": {"allow": "*"},
            },
        },
        instructions=(
            "You have a local Python code interpreter tool. A file is mounted "
            "as artifact 'runtime_notes'. Use python_interpreter to inspect it. "
            "Inside Python, artifacts.info('runtime_notes') returns a dict, so "
            "read the size with info['size']. artifacts.read requires keyword "
            "arguments: artifacts.read('runtime_notes', offset=0, limit=800), "
            "and returns a str. The interpreter only returns stdout and the "
            "result variable; bare expressions are not returned. Set result to "
            "a concise string containing the size and a summary. Do not call "
            "read with positional offset/limit. Do not call decode(). Do not "
            "use attribute access on the info dict. Do not answer without "
            "calling python_interpreter first."
        ),
    )


async def run_with_artifact(model_name: str, artifact_path: Path) -> None:
    agent = build_agent(model_name)
    response = await agent.acall(
        (
            "Read the uploaded runtime_notes artifact with the local sandbox. "
            "Report the size, mention that bounded artifact reads were used, "
            "and summarize the notes. Use this exact Python pattern inside "
            "python_interpreter:\n"
            "info = artifacts.info('runtime_notes')\n"
            "chunk = artifacts.read('runtime_notes', offset=0, limit=800)\n"
            "print(f\"artifact=runtime_notes size={info['size']} {info['unit']}\")\n"
            "result = f\"size={info['size']} {info['unit']}\\n{chunk}\""
        ),
        artifacts={"runtime_notes": artifact_path},
    )
    print(response)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-4.1-mini")
    parser.add_argument(
        "--artifact-path",
        type=Path,
        help="Optional local file to mount as the runtime_notes artifact.",
    )
    args = parser.parse_args()

    if args.artifact_path is not None:
        await run_with_artifact(args.model, args.artifact_path)
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_path = Path(temp_dir) / "runtime_notes.txt"
        artifact_path.write_text(DEFAULT_ARTIFACT, encoding="utf-8")
        await run_with_artifact(args.model, artifact_path)


if __name__ == "__main__":
    asyncio.run(main())
