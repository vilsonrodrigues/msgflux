"""Inline — first-class pipeline orchestrator.

Composes modules into a pipeline defined by a DSL expression.
Supports sequential, parallel, conditional, and while-loop execution
with optional checkpoint-per-step durability.
"""

import uuid
from typing import TYPE_CHECKING, Callable, Mapping, Optional

from msgflux.chat_messages import ChatMessages
from msgflux.dotdict import dotdict
from msgflux.dsl.inline.core import AsyncInlineDSL, InlineDSL
from msgflux.dsl.inline.runtime import AsyncDurableInlineDSL, DurableInlineDSL

if TYPE_CHECKING:
    from msgflux.data.stores.base import CheckpointStore


class Inline:
    """First-class pipeline orchestrator.

    Composes callables into a pipeline using a lightweight DSL.

    Args:
        expression:
            DSL string describing the execution flow.

            Supported constructs:

            - **Sequential**: ``"a -> b -> c"``
            - **Parallel**: ``"a -> [b, c] -> d"``
            - **Conditional**: ``"{score > 0.5 ? good, bad}"``
            - **While loop**: ``"@{counter < 5}: increment;"``
        modules:
            Mapping of step names to callables.  Each callable receives
            a :class:`~msgflux.dotdict` message and may return a ``dict``
            delta (merged automatically) or mutate in-place.
        max_iterations:
            Safety limit for while loops.

    Examples:
        >>> import msgflux as mf
        >>>
        >>> inline = mf.Inline(
        ...     "extract -> enrich -> summarize",
        ...     modules={
        ...         "extract": extract,
        ...         "enrich": enrich,
        ...         "summarize": summarize,
        ...     },
        ... )
        >>>
        >>> # Non-durable
        >>> result = inline(mf.dotdict({"question": "Why is the sky blue?"}))
        >>>
        >>> # Durable with checkpoint store
        >>> store = mf.InMemoryCheckpointStore()
        >>> result = inline(
        ...     mf.dotdict({"question": "..."}),
        ...     store=store,
        ...     session_id="user_42",
        ...     run_id="run_1",
        ... )
    """

    def __init__(
        self,
        expression: str,
        modules: Mapping[str, Callable],
        *,
        max_iterations: int = 1000,
    ):
        self._expression = expression
        self._step_modules = dict(modules)
        self._max_iterations = max_iterations
        # Validate expression eagerly
        InlineDSL().parse(expression)

    def __call__(
        self,
        message: dotdict,
        *,
        store: Optional["CheckpointStore"] = None,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        namespace: str = "inline",
        max_retries: int = 0,
        retry_delay: float = 1.0,
    ) -> dotdict:
        """Execute the pipeline synchronously.

        Args:
            message: Input :class:`~msgflux.dotdict` to process.
            store: Checkpoint store for durability.  ``None`` disables
                checkpointing.
            session_id: Session identifier — propagated via
                :class:`~msgflux.chat_messages.ChatMessages` context
                and used as checkpoint store key.
            run_id: Unique execution identifier.  Auto-generated when
                omitted.
            namespace: Checkpoint namespace.
            max_retries: Per-step retry limit (durable mode only).
            retry_delay: Delay between retries in seconds.

        Returns:
            The message after all steps have executed.
        """
        resolved_session = session_id or "default"

        with ChatMessages.session_context(
            session_id=resolved_session,
            namespace=namespace,
        ):
            if store is not None:
                dsl = DurableInlineDSL(
                    store,
                    namespace=namespace,
                    session_id=resolved_session,
                    run_id=run_id or str(uuid.uuid4()),
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    max_iterations=self._max_iterations,
                )
                return dsl(self._expression, self._step_modules, message)

            dsl = InlineDSL(max_iterations=self._max_iterations)
            return dsl(self._expression, self._step_modules, message)

    async def acall(
        self,
        message: dotdict,
        *,
        store: Optional["CheckpointStore"] = None,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        namespace: str = "inline",
        max_retries: int = 0,
        retry_delay: float = 1.0,
    ) -> dotdict:
        """Execute the pipeline asynchronously.

        Same parameters as :meth:`__call__`.
        """
        resolved_session = session_id or "default"

        with ChatMessages.session_context(
            session_id=resolved_session,
            namespace=namespace,
        ):
            if store is not None:
                dsl = AsyncDurableInlineDSL(
                    store,
                    namespace=namespace,
                    session_id=resolved_session,
                    run_id=run_id or str(uuid.uuid4()),
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    max_iterations=self._max_iterations,
                )
                return await dsl(
                    self._expression, self._step_modules, message,
                )

            dsl = AsyncInlineDSL(max_iterations=self._max_iterations)
            return await dsl(self._expression, self._step_modules, message)

    def __repr__(self) -> str:
        modules_str = ", ".join(self._step_modules.keys())
        return (
            f"Inline(expression={self._expression!r}, "
            f"modules=[{modules_str}])"
        )
