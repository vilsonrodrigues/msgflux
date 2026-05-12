from __future__ import annotations

from typing import Any, Mapping

from msgflux.runtime.interactions import AskUserManager


class AskUser:
    """Ask the user questions during async agent execution."""

    name = "ask_user"
    display_name = "Ask User"
    description = (
        "Ask the user one or more questions to clarify ambiguity, gather "
        "preferences, or request a decision during execution."
    )
    requires_user_interaction = True
    read_only = True
    concurrency_safe = True

    def __init__(
        self,
        manager: AskUserManager,
        *,
        timeout: float | None = None,
    ):
        self.manager = manager
        self.timeout = timeout

    def __call__(
        self,
        questions: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ask the user questions.

        Args:
            questions: List of question objects. Each object requires
                `question`, `header`, and `options`; `multi_select` is optional.
            metadata: Optional metadata for adapters.
        """
        _ = (questions, metadata)
        raise RuntimeError("AskUser is async-only. Use `agent.acall(...)`.")

    async def acall(
        self,
        questions: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        answer = await self.manager.request(
            questions,
            tool_name=self.name,
            metadata=metadata,
            timeout=self.timeout,
        )
        return {
            "answers": dict(answer.answers),
            "annotations": dict(answer.annotations),
            "message": self._format_tool_result(answer.answers),
        }

    def _format_tool_result(self, answers: Mapping[str, Any]) -> str:
        answers_text = ", ".join(
            f'"{question}"="{answer}"' for question, answer in answers.items()
        )
        return (
            f"User answered your questions: {answers_text}. "
            "Continue with the user's answers in mind."
        )
