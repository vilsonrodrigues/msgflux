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
llm_query asks a focused worker agent about a bounded context slice.
For debug, print the slice or metadata instead of retaining large text in vars.
Do not load whole large files into variables unless explicitly necessary.
"""


def build_llm_query(model_name: str):
    query_agent = nn.Agent(
        name="artifact_llm_query_worker",
        model=mf.Model.chat_completion(model_name),
        instructions=(
            "Answer only from <context>. If the answer is absent, say "
            "'not found in context'. Keep the answer concise."
        ),
        templates={
            "task": (
                "Question:\n{{ task }}\n\n"
                "<context>\n{{ context }}\n</context>"
            )
        },
    )

    async def llm_query(task: str, context: str) -> str:
        """Ask a focused LLM question about a supplied context slice."""
        return await query_agent.acall(task={"task": task}, vars={"context": context})

    return llm_query


def build_agent(model_name: str) -> nn.Agent:
    return nn.Agent(
        name="openai_artifact_reader",
        model=mf.Model.chat_completion(model_name),
        tools=[build_llm_query(model_name)],
        code_interpreter=mf.Sandbox.python("local"),
        config={
            "verbose": True,
            "max_tool_turns": 6,
            "code_interpreter": {
                "ptc": True,
                "artifacts": True,
                "ptc_tools": {"allow": ["llm_query"]},
            },
        },
        instructions=(
            "You have a local Python code interpreter tool. A file is mounted "
            "as artifact 'runtime_notes'. Use python_interpreter to inspect it. "
            "Inside Python, artifacts.info('runtime_notes') returns a dict, so "
            "read the size with info['size']. artifacts.read requires keyword "
            "arguments: artifacts.read('runtime_notes', offset=0, limit=800), "
            "and returns a str. The interpreter only returns stdout and the "
            "result variable; bare expressions are not returned. For debug, "
            "prefer print(...) over storing large text in variables. Pass the "
            "bounded artifact slice directly into await tools.llm_query(..., "
            "context=artifacts.read(...)) instead of assigning the slice to a "
            "long-lived variable. Set result to a concise string containing the "
            "size and llm_query answer. Do not call read with positional "
            "offset/limit. Do not call decode(). Do not use attribute access on "
            "the info dict. Do not answer without calling python_interpreter "
            "first."
        ),
    )


async def run_with_artifact(model_name: str, artifact_path: Path) -> None:
    agent = build_agent(model_name)
    response = await agent.acall(
        (
            "Read the uploaded runtime_notes artifact with the local sandbox. "
            "Report the size, mention that bounded artifact reads were used, "
            "and use llm_query to answer how session_id, run_id, namespace and "
            "artifacts relate to durable execution. Use this exact Python "
            "pattern inside "
            "python_interpreter:\n"
            "info = artifacts.info('runtime_notes')\n"
            "print(f\"artifact=runtime_notes size={info['size']} {info['unit']}\")\n"
            "print(artifacts.read('runtime_notes', offset=0, limit=300))\n"
            "answer = await tools.llm_query(\n"
            "    task='How do session_id, run_id, namespace and artifacts relate "
            "to durable execution?',\n"
            "    context=artifacts.read('runtime_notes', offset=0, limit=800),\n"
            ")\n"
            "result = f\"size={info['size']} {info['unit']}\\n{answer}\""
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
