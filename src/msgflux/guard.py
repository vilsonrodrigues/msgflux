import asyncio
import inspect
from typing import Any, Callable, Dict


class _GuardInterrupt(Exception):  # noqa: N818
    """Internal exception for short-circuit when policy='message'."""

    def __init__(self, response: str):
        self.response = response


class Guard:
    """Validates input/output of a Module with a configurable reaction policy.

    Args:
        validator: Callable that receives ``data=...`` and returns a dict
            with ``"safe"`` (bool) and optionally ``"message"`` (str).
        on: ``"input"`` or ``"output"``.
        policy: ``"raise"`` (raises exception) or ``"message"``
            (returns message as response, short-circuiting the pipeline).
    """

    _VALID_ON = {"input", "output"}
    _VALID_POLICY = {"raise", "message"}

    def __init__(
        self,
        validator: Callable[..., Dict[str, Any]],
        *,
        on: str = "input",
        policy: str = "raise",
    ):
        if not callable(validator):
            raise TypeError(f"`validator` must be callable, given `{type(validator)}`")
        if on not in self._VALID_ON:
            raise ValueError(f"`on` must be one of {self._VALID_ON}, given `{on!r}`")
        if policy not in self._VALID_POLICY:
            raise ValueError(
                f"`policy` must be one of {self._VALID_POLICY}, given `{policy!r}`"
            )

        self.validator = validator
        self.on = on
        self.policy = policy

    def __call__(self, **kwargs: Any) -> Dict[str, Any]:
        return self.validator(**kwargs)

    async def acall(self, **kwargs: Any) -> Dict[str, Any]:
        """Async call: delegates to validator.acall(), awaits coroutine, or
        runs sync validator in an executor.
        """
        if hasattr(self.validator, "acall"):
            return await self.validator.acall(**kwargs)
        elif inspect.iscoroutinefunction(self.validator):
            return await self.validator(**kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self.validator(**kwargs))

    def __repr__(self) -> str:
        return f"Guard(on={self.on!r}, policy={self.policy!r})"
