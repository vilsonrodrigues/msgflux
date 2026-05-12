from __future__ import annotations

import asyncio
from asyncio import AbstractEventLoop
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from time import time
from typing import Any, Mapping
from uuid import uuid4

from msgflux.runtime.events import (
    emit_user_interaction_answered,
    emit_user_interaction_cancelled,
    emit_user_interaction_requested,
)


class UserInteractionRuntimeError(RuntimeError):
    """Base error for user interaction requests."""


class UserInteractionCancelledError(UserInteractionRuntimeError):
    """Raised when a user interaction is cancelled or rejected."""


class UserInteractionTimeoutError(UserInteractionRuntimeError):
    """Raised when a user interaction is not answered in time."""


@dataclass(frozen=True)
class UserQuestionOption:
    label: str
    description: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class UserQuestion:
    question: str
    header: str
    options: list[UserQuestionOption]
    multi_select: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "header": self.header,
            "options": [option.to_dict() for option in self.options],
            "multi_select": self.multi_select,
        }


@dataclass(frozen=True)
class UserInteractionRequest:
    questions: list[UserQuestion]
    request_id: str = field(default_factory=lambda: f"user_{uuid4().hex[:12]}")
    tool_name: str | None = None
    caller_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time)

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "request_id": self.request_id,
                "questions": [question.to_dict() for question in self.questions],
                "tool_name": self.tool_name,
                "caller_name": self.caller_name,
                "metadata": dict(self.metadata),
                "created_at": self.created_at,
            }.items()
            if value is not None and value != {}
        }


@dataclass(frozen=True)
class UserInteractionAnswer:
    request_id: str
    answers: Mapping[str, Any]
    annotations: Mapping[str, Any] = field(default_factory=dict)
    answered_at: float = field(default_factory=time)

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None and value != {}
        }


@dataclass(frozen=True)
class _PendingUserInteraction:
    future: asyncio.Future[UserInteractionAnswer]
    loop: AbstractEventLoop


class AskUserManager:
    """Async-first coordinator for user questions during runtime execution."""

    def __init__(self, *, timeout: float | None = None) -> None:
        self.timeout = timeout
        self._pending: dict[str, _PendingUserInteraction] = {}
        self._requests: dict[str, UserInteractionRequest] = {}

    async def request(
        self,
        questions: list[UserQuestion | Mapping[str, Any]],
        *,
        tool_name: str | None = None,
        caller_name: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        timeout: float | None = None,
    ) -> UserInteractionAnswer:
        return await self.request_user(
            UserInteractionRequest(
                questions=normalize_questions(questions),
                tool_name=tool_name,
                caller_name=caller_name,
                metadata=deepcopy(dict(metadata or {})),
            ),
            timeout=timeout,
        )

    async def request_user(
        self,
        request: UserInteractionRequest,
        *,
        timeout: float | None = None,
    ) -> UserInteractionAnswer:
        self._requests[request.request_id] = request
        loop = asyncio.get_running_loop()
        future: asyncio.Future[UserInteractionAnswer] = loop.create_future()
        self._pending[request.request_id] = _PendingUserInteraction(
            future=future,
            loop=loop,
        )
        emit_user_interaction_requested(request.to_dict())
        try:
            return await asyncio.wait_for(
                future,
                timeout=self.timeout if timeout is None else timeout,
            )
        except TimeoutError as exc:
            self._pending.pop(request.request_id, None)
            emit_user_interaction_cancelled(
                {
                    "request_id": request.request_id,
                    "reason": "user interaction timed out",
                }
            )
            raise UserInteractionTimeoutError("User interaction timed out.") from exc

    def answer(
        self,
        request_id: str,
        answers: Mapping[str, Any],
        *,
        annotations: Mapping[str, Any] | None = None,
    ) -> UserInteractionAnswer:
        if not isinstance(answers, Mapping) or not answers:
            raise ValueError("`answers` must be a non-empty mapping.")
        return self._resolve(
            UserInteractionAnswer(
                request_id=request_id,
                answers=deepcopy(dict(answers)),
                annotations=deepcopy(dict(annotations or {})),
            )
        )

    def cancel(self, request_id: str, *, reason: str | None = None) -> None:
        pending = self._pending.pop(request_id, None)
        if pending is None:
            raise KeyError(f"User interaction request `{request_id}` is not pending.")

        resolved_reason = reason or "User interaction cancelled."

        def _set_exception() -> None:
            if not pending.future.done():
                pending.future.set_exception(
                    UserInteractionCancelledError(resolved_reason)
                )

        try:
            pending.loop.call_soon_threadsafe(_set_exception)
        except RuntimeError as exc:
            self._pending[request_id] = pending
            raise UserInteractionRuntimeError(
                f"User interaction request `{request_id}` cannot be cancelled "
                "because its event loop is not accepting callbacks."
            ) from exc
        emit_user_interaction_cancelled(
            {"request_id": request_id, "reason": resolved_reason}
        )

    def _resolve(self, answer: UserInteractionAnswer) -> UserInteractionAnswer:
        pending = self._pending.pop(answer.request_id, None)
        if pending is None:
            raise KeyError(
                f"User interaction request `{answer.request_id}` is not pending."
            )

        def _set_result() -> None:
            if not pending.future.done():
                pending.future.set_result(answer)

        try:
            pending.loop.call_soon_threadsafe(_set_result)
        except RuntimeError as exc:
            self._pending[answer.request_id] = pending
            raise UserInteractionRuntimeError(
                f"User interaction request `{answer.request_id}` cannot be answered "
                "because its event loop is not accepting callbacks."
            ) from exc
        emit_user_interaction_answered(answer.to_dict())
        return answer

    def get_request(self, request_id: str) -> UserInteractionRequest | None:
        return self._requests.get(request_id)

    def list_pending(self) -> list[UserInteractionRequest]:
        return [
            request
            for request_id, request in self._requests.items()
            if request_id in self._pending
        ]


def normalize_questions(
    questions: list[UserQuestion | Mapping[str, Any]],
) -> list[UserQuestion]:
    if not isinstance(questions, list) or not 1 <= len(questions) <= 4:
        raise ValueError("`questions` must contain 1 to 4 question objects.")

    seen_questions = set()
    normalized = []
    for question in questions:
        if isinstance(question, UserQuestion):
            normalized_question = question
        elif isinstance(question, Mapping):
            normalized_question = _normalize_question_mapping(question)
        else:
            raise TypeError("Each question must be a UserQuestion or mapping.")

        if normalized_question.question in seen_questions:
            raise ValueError("Question texts must be unique.")
        seen_questions.add(normalized_question.question)
        normalized.append(normalized_question)
    return normalized


def _normalize_question_mapping(question: Mapping[str, Any]) -> UserQuestion:
    text = question.get("question")
    header = question.get("header")
    options = question.get("options")
    multi_select = question.get("multi_select", question.get("multiSelect", False))

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Each question requires a non-empty `question`.")
    if not isinstance(header, str) or not header.strip():
        raise ValueError("Each question requires a non-empty `header`.")
    if len(header.strip()) > 12:
        raise ValueError("Question `header` must be 12 characters or fewer.")
    if not isinstance(options, list) or not 2 <= len(options) <= 4:
        raise ValueError("Each question requires 2 to 4 `options`.")
    if not isinstance(multi_select, bool):
        raise TypeError("`multi_select` must be a bool when provided.")

    return UserQuestion(
        question=text.strip(),
        header=header.strip(),
        options=_normalize_options(options),
        multi_select=multi_select,
    )


def _normalize_options(options: list[Mapping[str, Any]]) -> list[UserQuestionOption]:
    seen_labels = set()
    normalized = []
    for option in options:
        if not isinstance(option, Mapping):
            raise TypeError("Each option must be a mapping.")
        label = option.get("label")
        description = option.get("description")
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Each option requires a non-empty `label`.")
        if label in seen_labels:
            raise ValueError("Option labels must be unique within each question.")
        seen_labels.add(label)
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Each option requires a non-empty `description`.")
        normalized.append(
            UserQuestionOption(
                label=label.strip(),
                description=description.strip(),
            )
        )
    return normalized
