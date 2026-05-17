# /// script
# dependencies = []
# ///
# ruff: noqa: T201

import argparse

import msgflux as mf
from msgflux import nn

mf.load_dotenv()


BIG_CONTEXT = "\n".join(
    [
        "MSGFLUX Runtime Notes",
        "session_id identifies the durable conversation scope.",
        "run_id identifies a resumable execution attempt within a session.",
        "namespace identifies which module or agent writes runtime state.",
        "AgentInbox stores incoming user messages and system notes.",
        "ExecutionScope carries session, run, namespace, and runtime policies.",
        "The code interpreter receives user variables through vars.",
        "Programmatic tool calls use tools.<name>(...) in the interpreter.",
        "llm_query lets interpreter code ask an agent about a context slice.",
        "Large context should be sliced before querying to avoid unnecessary tokens.",
        "Workflow: slice big_context, print the slice, call llm_query.",
        "Future shell environments can route tools into isolated sandboxes.",
    ]
    * 18
)


def build_llm_query(model_name: str):
    query_agent = nn.Agent(
        name="llm_query_worker",
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
        name="context_repl_agent",
        model=mf.Model.chat_completion(model_name),
        tools=[build_llm_query(model_name)],
        code_interpreter=mf.Sandbox.python("local"),
        config={
            "code_interpreter": {
                "ptc": True,
                "ptc_tools": {"allow": ["llm_query"]},
                "inject_vars": True,
                "notify_vars": True,
            },
            "max_tool_turns": 6,
        },
        instructions=(
            "Use python_interpreter to inspect vars['big_context']. "
            "Start with chunk = vars['big_context'][:1000]. Print the slice "
            "range you inspect, then call await tools.llm_query(task=..., "
            "context=chunk). Store the worker answer in result together with "
            "the printed slice range. Return the final answer after the tool "
            "result."
        ),
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-4.1-mini")
    args = parser.parse_args()

    agent = build_agent(args.model)
    result = await agent.acall(
        (
            "Use the code interpreter to inspect the first context slice and "
            "ask llm_query how session_id, run_id, and namespace relate to "
            "durable execution. Include the inspected slice range."
        ),
        vars={"big_context": BIG_CONTEXT},
    )
    print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
