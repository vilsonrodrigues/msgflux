"""Tests for Guard class and guard integration."""

import asyncio

import pytest

from msgflux.exceptions import UnsafeModelResponseError, UnsafeUserInputError
from msgflux.guard import Guard, _GuardInterrupt


# ---------------------------------------------------------------------------
# Guard instantiation
# ---------------------------------------------------------------------------


class TestGuardInit:
    def test_valid_guard(self):
        guard = Guard(validator=lambda data: {"safe": True}, on="input", policy="raise")
        assert guard.on == "input"
        assert guard.policy == "raise"

    def test_defaults(self):
        guard = Guard(validator=lambda data: {"safe": True})
        assert guard.on == "input"
        assert guard.policy == "raise"

    def test_invalid_on(self):
        with pytest.raises(ValueError, match="`on` must be one of"):
            Guard(validator=lambda data: {"safe": True}, on="middle")

    def test_invalid_policy(self):
        with pytest.raises(ValueError, match="`policy` must be one of"):
            Guard(validator=lambda data: {"safe": True}, policy="ignore")

    def test_non_callable_validator(self):
        with pytest.raises(TypeError, match="`validator` must be callable"):
            Guard(validator="not_callable")

    def test_repr(self):
        guard = Guard(
            validator=lambda data: {"safe": True}, on="output", policy="message"
        )
        assert repr(guard) == "Guard(on='output', policy='message')"


# ---------------------------------------------------------------------------
# Guard __call__ / acall
# ---------------------------------------------------------------------------


class TestGuardCall:
    def test_sync_call(self):
        guard = Guard(validator=lambda data: {"safe": data != "bad"})
        assert guard(data="good") == {"safe": True}
        assert guard(data="bad") == {"safe": False}

    @pytest.mark.asyncio
    async def test_acall_with_sync_validator(self):
        guard = Guard(validator=lambda data: {"safe": True})
        result = await guard.acall(data="hello")
        assert result == {"safe": True}

    @pytest.mark.asyncio
    async def test_acall_with_coroutine_validator(self):
        async def async_validator(data):
            return {"safe": data != "bad", "message": "blocked"}

        guard = Guard(validator=async_validator)
        result = await guard.acall(data="good")
        assert result["safe"] is True

        result = await guard.acall(data="bad")
        assert result["safe"] is False

    @pytest.mark.asyncio
    async def test_acall_with_acall_method(self):
        class ValidatorWithAcall:
            def __call__(self, data):
                return {"safe": True}

            async def acall(self, data):
                return {"safe": False, "message": "async check"}

        guard = Guard(validator=ValidatorWithAcall())
        result = await guard.acall(data="anything")
        assert result["safe"] is False
        assert result["message"] == "async check"


# ---------------------------------------------------------------------------
# _GuardInterrupt
# ---------------------------------------------------------------------------


class TestGuardInterrupt:
    def test_stores_response(self):
        exc = _GuardInterrupt("blocked message")
        assert exc.response == "blocked message"

    def test_is_exception(self):
        assert issubclass(_GuardInterrupt, Exception)


# ---------------------------------------------------------------------------
# Policy behaviour (unit-level, no Agent)
# ---------------------------------------------------------------------------


class TestGuardPolicy:
    """Test the guard execution logic extracted from Module."""

    def _run_guards(self, guards, data):
        """Simulate what Module._execute_input_guards does."""
        for guard in guards:
            result = guard(data=data)
            if not result.get("safe", True):
                message = result.get("message")
                if guard.policy == "message":
                    raise _GuardInterrupt(message or "Blocked by guard.")
                raise UnsafeUserInputError(message)

    def test_policy_raise(self):
        guard = Guard(
            validator=lambda data: {"safe": False, "message": "bad input"},
            on="input",
            policy="raise",
        )
        with pytest.raises(UnsafeUserInputError, match="bad input"):
            self._run_guards([guard], "test")

    def test_policy_message(self):
        guard = Guard(
            validator=lambda data: {"safe": False, "message": "Sorry, blocked."},
            on="input",
            policy="message",
        )
        with pytest.raises(_GuardInterrupt) as exc_info:
            self._run_guards([guard], "test")
        assert exc_info.value.response == "Sorry, blocked."

    def test_policy_message_no_message(self):
        guard = Guard(
            validator=lambda data: {"safe": False},
            on="input",
            policy="message",
        )
        with pytest.raises(_GuardInterrupt) as exc_info:
            self._run_guards([guard], "test")
        assert exc_info.value.response == "Blocked by guard."

    def test_safe_passes(self):
        guard = Guard(
            validator=lambda data: {"safe": True},
            on="input",
            policy="raise",
        )
        self._run_guards([guard], "test")  # Should not raise

    def test_multiple_guards_first_blocks(self):
        g1 = Guard(
            validator=lambda data: {"safe": False, "message": "g1 blocked"},
            on="input",
            policy="raise",
        )
        g2 = Guard(
            validator=lambda data: {"safe": True},
            on="input",
            policy="raise",
        )
        with pytest.raises(UnsafeUserInputError, match="g1 blocked"):
            self._run_guards([g1, g2], "test")

    def test_multiple_guards_second_blocks(self):
        g1 = Guard(
            validator=lambda data: {"safe": True},
            on="input",
            policy="raise",
        )
        g2 = Guard(
            validator=lambda data: {"safe": False, "message": "g2 blocked"},
            on="input",
            policy="message",
        )
        with pytest.raises(_GuardInterrupt) as exc_info:
            self._run_guards([g1, g2], "test")
        assert exc_info.value.response == "g2 blocked"


# ---------------------------------------------------------------------------
# Exception messages
# ---------------------------------------------------------------------------


class TestExceptionMessages:
    def test_unsafe_user_input_default(self):
        exc = UnsafeUserInputError()
        assert str(exc) == "Unsafe user input detected"

    def test_unsafe_user_input_custom(self):
        exc = UnsafeUserInputError("custom message")
        assert str(exc) == "custom message"

    def test_unsafe_model_response_default(self):
        exc = UnsafeModelResponseError()
        assert str(exc) == "Unsafe model response detected"

    def test_unsafe_model_response_custom(self):
        exc = UnsafeModelResponseError("toxic content")
        assert str(exc) == "toxic content"
