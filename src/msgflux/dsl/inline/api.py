"""Public API for inline DSL execution with optional durable execution."""

from typing import Any, Callable, Mapping, Optional

from msgflux.dotdict import dotdict
from msgflux.dsl.inline.parser import ainline as _ainline_impl
from msgflux.dsl.inline.parser import inline as _inline_impl


def inline(
    expression: str,
    modules: Mapping[str, Callable],
    message: dotdict,
    *,
    run_id: Optional[str] = None,
    store: Optional[Any] = None,
) -> dotdict:
    """Execute a workflow defined in DSL expression over a given ``message``.

    When *run_id* is provided, enables checkpoint/resume: if execution
    crashes, re-calling with the same *run_id* resumes from the last
    successful step.

    Args:
        expression: DSL pipeline string.
        modules: Mapping of module names to callables.
        message: Input ``dotdict`` message.
        run_id: Optional run identifier for durable execution.
        store: Optional ``CheckpointStore`` instance.  Defaults to
            ``MemoryCheckpointStore`` when *run_id* is provided.

    Returns:
        The resulting ``dotdict`` message after executing the pipeline.
    """
    if run_id is not None:
        from msgflux.context import _current_run_id, set_run_id  # noqa: PLC0415
        from msgflux.dsl.inline.runtime import DurableInlineDSL  # noqa: PLC0415

        if not isinstance(expression, str):
            raise TypeError("`expression` must be a str")
        if not isinstance(message, dotdict):
            raise TypeError("`message` must be an instance of `msgflux.dotdict`")
        if not isinstance(modules, Mapping):
            raise TypeError("`modules` must be a `Mapping`")

        token = set_run_id(run_id)
        try:
            dsl = DurableInlineDSL(store=store)
            return dsl(expression, modules, message)
        finally:
            _current_run_id.reset(token)

    return _inline_impl(expression, modules, message)


async def ainline(
    expression: str,
    modules: Mapping[str, Callable],
    message: dotdict,
    *,
    run_id: Optional[str] = None,
    store: Optional[Any] = None,
) -> dotdict:
    """Async version of :func:`inline`.

    Args:
        expression: DSL pipeline string.
        modules: Mapping of module names to callables.
        message: Input ``dotdict`` message.
        run_id: Optional run identifier for durable execution.
        store: Optional ``CheckpointStore`` instance.

    Returns:
        The resulting ``dotdict`` message after executing the pipeline.
    """
    if run_id is not None:
        from msgflux.context import _current_run_id, set_run_id  # noqa: PLC0415
        from msgflux.dsl.inline.runtime import AsyncDurableInlineDSL  # noqa: PLC0415

        if not isinstance(expression, str):
            raise TypeError("`expression` must be a str")
        if not isinstance(message, dotdict):
            raise TypeError("`message` must be an instance of `msgflux.dotdict`")
        if not isinstance(modules, Mapping):
            raise TypeError("`modules` must be a `Mapping`")

        token = set_run_id(run_id)
        try:
            dsl = AsyncDurableInlineDSL(store=store)
            return await dsl(expression, modules, message)
        finally:
            _current_run_id.reset(token)

    return await _ainline_impl(expression, modules, message)
