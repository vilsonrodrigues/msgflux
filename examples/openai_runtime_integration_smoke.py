# /// script
# dependencies = ["openai", "just-bash"]
# ///
# ruff: noqa: T201

import argparse
import asyncio
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import msgflux as mf
from msgflux import nn
from msgflux.runtime import PermissionManager, TodoManager
from msgflux.runtime.interactions import AskUserManager
from msgflux.tools.builtin import (
    AskUser,
    FileEdit,
    FileRead,
    SendUserMessage,
    TodoWrite,
)

mf.load_dotenv()


RUNTIME_NOTES = """msgFlux runtime smoke notes
session_id identifies the durable conversation scope.
run_id identifies the resumable execution attempt inside the session.
namespace identifies which module or agent writes runtime state.
AgentInbox can inject incoming user messages during execution.
Artifacts are mounted into the code interpreter as lazy file references.
Use await artifacts["info"](name) to inspect artifact metadata.
Use await artifacts["read"](name, offset=0, limit=...) for bounded reads.
Programmatic tool calls in python_interpreter use await tools["name"](...).
"""


def _compact(value: Any, *, limit: int = 240) -> str:
    if isinstance(value, Mapping):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def print_event(event) -> None:
    attrs = event.attributes
    detail = {
        key: attrs.get(key)
        for key in (
            "agent_name",
            "response_type",
            "tool_name",
            "tool_call_id",
            "caller_name",
            "caller_namespace",
            "caller_session_id",
            "caller_run_id",
            "action",
            "policy",
            "status",
            "result",
            "error",
        )
        if attrs.get(key) is not None
    }
    print(f"{event.name}: {_compact(detail or attrs)}")


def build_llm_query(model_name: str):
    query_agent = nn.Agent(
        name="openai_runtime_query_worker",
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

    async def llm_query(query: str, context: str) -> str:
        """Ask a focused LLM question about a supplied context slice."""
        return await query_agent.acall(task={"task": query}, vars={"context": context})

    return llm_query


async def auto_answer_user_questions(
    manager: AskUserManager,
    stop_event: asyncio.Event,
):
    answered: set[str] = set()
    while not stop_event.is_set():
        for request in manager.list_pending():
            if request.request_id in answered:
                continue
            manager.answer(
                request.request_id,
                {
                    question.question: question.options[0].label
                    for question in request.questions
                },
                annotations={"source": "openai_runtime_smoke"},
            )
            answered.add(request.request_id)
        await asyncio.sleep(0.05)


def build_agent(
    *,
    model_name: str,
    ask_user_manager: AskUserManager,
    permission_manager: PermissionManager,
    todo_manager: TodoManager,
    workspace: Path,
) -> nn.Agent:
    return nn.Agent(
        name="openai_runtime_smoke_agent",
        model=mf.Model.chat_completion(model_name),
        tools=[
            TodoWrite(todo_manager),
            SendUserMessage(),
            AskUser(ask_user_manager),
            FileRead(),
            FileEdit(),
            mf.Sandbox.shell("just-bash", fs_mode="overlay", workspace=workspace),
            build_llm_query(model_name),
        ],
        code_interpreter=mf.Sandbox.python("local"),
        permission_manager=permission_manager,
        checkpointer=mf.InMemoryCheckpointStore(),
        config={
            "verbose": True,
            "max_tool_turns": 14,
            "code_interpreter": {
                "ptc": True,
                "artifacts": True,
                "ptc_tools": {"allow": ["llm_query"]},
            },
        },
        instructions=(
            "You are running a msgFlux runtime integration smoke test. Follow "
            "this exact tool sequence as closely as possible: "
            "1) call todo_write with one in_progress todo; "
            "2) call SendUserMessage with status='progress'; "
            "3) call ask_user once to confirm proceeding, using header='Confirm' "
            "because headers must be 12 characters or fewer; "
            "4) call Read on the provided notes file with offset=1, never "
            "offset=0, because Read uses 1-based line numbers; "
            "5) call edit to replace 'status=todo' with 'status=done'; "
            "6) call shell with command exactly 'grep status=done notes.txt'. "
            "The shell sandbox is mounted at /workspace and starts there, so "
            "do not use the host absolute path inside shell commands; "
            "7) call python_interpreter and inside it use "
            "await artifacts['read']('runtime_notes', offset=0, limit=600), "
            "then await tools['llm_query'](query='...', context=...) with that "
            "bounded context. "
            "After the tools complete, return a concise summary of what worked."
        ),
    )


async def run(model_name: str) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir) / "workspace"
        workspace.mkdir()
        notes_path = workspace / "notes.txt"
        notes_path.write_text("alpha\nstatus=todo\n", encoding="utf-8")
        artifact_path = Path(temp_dir) / "runtime_notes.txt"
        artifact_path.write_text(RUNTIME_NOTES, encoding="utf-8")

        ask_user_manager = AskUserManager(timeout=30)
        permission_manager = PermissionManager(default_mode="bypass")
        todo_manager = TodoManager()
        inbox = mf.AgentInbox()
        inbox.user_message(
            "During the smoke test, prefer bounded artifact reads and concise output."
        )

        agent = build_agent(
            model_name=model_name,
            ask_user_manager=ask_user_manager,
            permission_manager=permission_manager,
            todo_manager=todo_manager,
            workspace=workspace,
        )
        agent.set_agent_inbox(inbox)

        scope = mf.ExecutionScope(
            session_id="openai-runtime-smoke",
            run_id="runtime-smoke-001",
            permission_mode="bypass",
        )
        prompt = (
            "Run the integration smoke test. The notes file is "
            f"{notes_path}. The workspace is {workspace}. Use the "
            "runtime_notes artifact in python_interpreter. Important: Read "
            "uses 1-based line offsets, so call Read with offset=1. Important: "
            "inside the shell sandbox, use relative path notes.txt, not the "
            "host absolute path."
        )

        stop_answers = asyncio.Event()
        answer_task = asyncio.create_task(
            auto_answer_user_questions(ask_user_manager, stop_answers)
        )
        try:
            async for event in agent.astream_events(
                prompt,
                scope=scope,
                artifacts={"runtime_notes": artifact_path},
            ):
                print_event(event)
        finally:
            stop_answers.set()
            await answer_task

        print("\nFinal workspace files:")
        for path in sorted(workspace.iterdir()):
            print(f"- {path.name}: {_compact(path.read_text(encoding='utf-8'))}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-5.4-mini")
    args = parser.parse_args()
    asyncio.run(run(args.model))


if __name__ == "__main__":
    main()
