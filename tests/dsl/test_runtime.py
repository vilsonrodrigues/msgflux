"""Tests for durable inline DSL execution."""

import pytest

from msgflux.context import _current_run_id, set_run_id
from msgflux.data.stores import MemoryCheckpointStore, StepStatus
from msgflux.dotdict import dotdict
from msgflux.dsl.inline.runtime import (
    DurableInlineDSL,
    MaxRetriesExceededError,
    _step_name,
)


class TestStepName:
    def test_module_step(self):
        step = {"type": "module", "module": "prep"}
        assert _step_name(step, 0) == "module:prep#0"

    def test_parallel_step(self):
        step = {"type": "parallel", "modules": ["a", "b"]}
        assert _step_name(step, 1) == "parallel:[a,b]#1"

    def test_conditional_step(self):
        step = {"type": "conditional", "condition": "x > 5"}
        assert _step_name(step, 2) == "conditional:x > 5#2"

    def test_while_step(self):
        step = {"type": "while", "condition": "counter < 10"}
        assert _step_name(step, 0) == "while:counter < 10#0"

    def test_duplicate_module_different_index(self):
        step = {"type": "module", "module": "prep"}
        assert _step_name(step, 0) != _step_name(step, 1)


class TestDurableInlineDSLFallback:
    def test_fallback_without_run_id(self):
        """Without run_id in context, DurableInlineDSL behaves like InlineDSL."""

        def prep(msg):
            msg["prepared"] = True
            return msg

        dsl = DurableInlineDSL()
        msg = dotdict()
        result = dsl("prep", {"prep": prep}, msg)
        assert result.prepared is True


class TestDurableInlineDSLCheckpoint:
    def test_checkpoint_and_skip_on_resume(self):
        """Steps that completed should be skipped on resume."""
        call_count = {"prep": 0, "final": 0}

        def prep(msg):
            call_count["prep"] += 1
            msg["prep_done"] = True
            return msg

        def final(msg):
            call_count["final"] += 1
            msg["final_done"] = True
            return msg

        store = MemoryCheckpointStore()
        modules = {"prep": prep, "final": final}

        # First run
        token = set_run_id("run-1")
        try:
            dsl = DurableInlineDSL(store=store)
            result = dsl("prep -> final", modules, dotdict())
            assert result.prep_done is True
            assert result.final_done is True
            assert call_count == {"prep": 1, "final": 1}
        finally:
            _current_run_id.reset(token)

        # Resume with same run_id — both steps should be skipped
        call_count = {"prep": 0, "final": 0}
        token = set_run_id("run-1")
        try:
            dsl = DurableInlineDSL(store=store)
            result = dsl("prep -> final", modules, dotdict())
            assert result.final_done is True
            assert call_count == {"prep": 0, "final": 0}
        finally:
            _current_run_id.reset(token)

    def test_resume_after_failure(self):
        """Prep succeeds, final fails. On resume, prep is skipped, final retries."""
        attempt = {"count": 0}

        def prep(msg):
            msg["prepared"] = True
            return msg

        def final(msg):
            attempt["count"] += 1
            if attempt["count"] == 1:
                raise RuntimeError("transient error")
            msg["done"] = True
            return msg

        store = MemoryCheckpointStore()
        modules = {"prep": prep, "final": final}

        # First run — fails at final
        token = set_run_id("run-2")
        try:
            dsl = DurableInlineDSL(store=store)
            with pytest.raises(RuntimeError, match="transient error"):
                dsl("prep -> final", modules, dotdict())
        finally:
            _current_run_id.reset(token)

        # Resume — prep skipped, final retried
        token = set_run_id("run-2")
        try:
            dsl = DurableInlineDSL(store=store)
            result = dsl("prep -> final", modules, dotdict())
            assert result.done is True
            assert result.prepared is True  # restored from snapshot
        finally:
            _current_run_id.reset(token)

    def test_different_run_id_is_fresh(self):
        """Different run_id should execute from scratch."""
        call_count = {"prep": 0}

        def prep(msg):
            call_count["prep"] += 1
            msg["x"] = call_count["prep"]
            return msg

        store = MemoryCheckpointStore()
        modules = {"prep": prep}

        # First run
        token = set_run_id("run-a")
        try:
            DurableInlineDSL(store=store)("prep", modules, dotdict())
        finally:
            _current_run_id.reset(token)

        # Different run_id — should execute fresh
        token = set_run_id("run-b")
        try:
            result = DurableInlineDSL(store=store)("prep", modules, dotdict())
            assert result.x == 2  # prep called again
        finally:
            _current_run_id.reset(token)

    def test_snapshot_restores_data(self):
        """Snapshot should restore the message state correctly."""

        def step_a(msg):
            msg["a"] = 1
            msg["nested"] = {"key": "value"}
            return msg

        def step_b(msg):
            msg["b"] = msg.a + 1
            return msg

        store = MemoryCheckpointStore()
        modules = {"step_a": step_a, "step_b": step_b}

        # First run
        token = set_run_id("run-snap")
        try:
            result = DurableInlineDSL(store=store)(
                "step_a -> step_b", modules, dotdict()
            )
            assert result.a == 1
            assert result.b == 2
            assert result.nested.key == "value"
        finally:
            _current_run_id.reset(token)

        # Resume — both skipped, last snapshot restored
        token = set_run_id("run-snap")
        try:
            result = DurableInlineDSL(store=store)(
                "step_a -> step_b", modules, dotdict()
            )
            assert result.a == 1
            assert result.b == 2
            assert result.nested["key"] == "value"
        finally:
            _current_run_id.reset(token)


class TestDurableInlineDSLMaxRetries:
    def test_max_retries_exceeded(self):
        """Step that always fails should raise MaxRetriesExceededError."""

        def always_fail(msg):
            raise RuntimeError("always fails")

        store = MemoryCheckpointStore()
        modules = {"fail": always_fail}

        for i in range(3):
            token = set_run_id("run-retry")
            try:
                dsl = DurableInlineDSL(store=store, max_retries=3)
                with pytest.raises(RuntimeError, match="always fails"):
                    dsl("fail", modules, dotdict())
            finally:
                _current_run_id.reset(token)

        # 4th attempt should raise MaxRetriesExceededError
        token = set_run_id("run-retry")
        try:
            dsl = DurableInlineDSL(store=store, max_retries=3)
            with pytest.raises(MaxRetriesExceededError, match="failed after 3 retries"):
                dsl("fail", modules, dotdict())
        finally:
            _current_run_id.reset(token)

    def test_custom_max_retries(self):
        """Custom max_retries should be respected."""

        def always_fail(msg):
            raise RuntimeError("fail")

        store = MemoryCheckpointStore()
        modules = {"fail": always_fail}

        # Fail once
        token = set_run_id("run-custom-retry")
        try:
            dsl = DurableInlineDSL(store=store, max_retries=1)
            with pytest.raises(RuntimeError, match="fail"):
                dsl("fail", modules, dotdict())
        finally:
            _current_run_id.reset(token)

        # 2nd attempt should raise MaxRetriesExceededError
        token = set_run_id("run-custom-retry")
        try:
            dsl = DurableInlineDSL(store=store, max_retries=1)
            with pytest.raises(MaxRetriesExceededError):
                dsl("fail", modules, dotdict())
        finally:
            _current_run_id.reset(token)


class TestDurableInlineDSLWhileLoop:
    def test_while_loop_with_checkpoint(self):
        """While loop iterations should be checkpointed individually."""
        call_count = {"increment": 0}

        def prep(msg):
            msg["counter"] = 0
            return msg

        def increment(msg):
            call_count["increment"] += 1
            msg["counter"] = msg.get("counter", 0) + 1
            return msg

        def final(msg):
            msg["done"] = True
            return msg

        store = MemoryCheckpointStore()
        modules = {"prep": prep, "increment": increment, "final": final}

        token = set_run_id("run-while")
        try:
            dsl = DurableInlineDSL(store=store)
            result = dsl(
                "prep -> @{counter < 3}: increment; -> final",
                modules,
                dotdict(),
            )
            assert result.counter == 3
            assert result.done is True
            assert call_count["increment"] == 3
        finally:
            _current_run_id.reset(token)

        # Resume — everything completed, should skip all
        call_count["increment"] = 0
        token = set_run_id("run-while")
        try:
            dsl = DurableInlineDSL(store=store)
            result = dsl(
                "prep -> @{counter < 3}: increment; -> final",
                modules,
                dotdict(),
            )
            assert result.counter == 3
            assert result.done is True
            assert call_count["increment"] == 0
        finally:
            _current_run_id.reset(token)

    def test_while_loop_resume_mid_iteration(self):
        """If crash occurs mid-iteration, resume from that iteration."""
        attempt = {"count": 0}

        def prep(msg):
            msg["counter"] = 0
            return msg

        def increment(msg):
            attempt["count"] += 1
            msg["counter"] = msg.get("counter", 0) + 1
            # Fail on 3rd call (iteration 2, counter going from 2 to 3)
            if attempt["count"] == 3:
                raise RuntimeError("mid-iteration crash")
            return msg

        store = MemoryCheckpointStore()
        modules = {"prep": prep, "increment": increment}

        # First run — crashes on iteration 2
        token = set_run_id("run-while-crash")
        try:
            dsl = DurableInlineDSL(store=store)
            with pytest.raises(RuntimeError, match="mid-iteration crash"):
                dsl(
                    "prep -> @{counter < 5}: increment;",
                    modules,
                    dotdict(),
                )
        finally:
            _current_run_id.reset(token)

        # Resume — prep + iterations 0,1 completed, iteration 2 retries
        token = set_run_id("run-while-crash")
        try:
            dsl = DurableInlineDSL(store=store)
            result = dsl(
                "prep -> @{counter < 5}: increment;",
                modules,
                dotdict(),
            )
            assert result.counter == 5
        finally:
            _current_run_id.reset(token)


class TestInlineDurable:
    def test_inline_with_run_id(self):
        """inline() with run_id uses DurableInlineDSL."""
        from msgflux.dsl.inline import inline

        def prep(msg):
            msg["x"] = 1
            return msg

        result = inline("prep", {"prep": prep}, dotdict(), run_id="test-run")
        assert result.x == 1

    def test_inline_without_run_id(self):
        """inline() without run_id uses regular InlineDSL."""
        from msgflux.dsl.inline import inline

        def prep(msg):
            msg["x"] = 1
            return msg

        result = inline("prep", {"prep": prep}, dotdict())
        assert result.x == 1

    def test_inline_resume(self):
        """inline() with same run_id should resume."""
        from msgflux.dsl.inline import inline

        call_count = {"prep": 0}

        def prep(msg):
            call_count["prep"] += 1
            msg["x"] = 1
            return msg

        store = MemoryCheckpointStore()

        # First run
        inline("prep", {"prep": prep}, dotdict(), run_id="resume-test", store=store)
        assert call_count["prep"] == 1

        # Resume — should skip
        inline("prep", {"prep": prep}, dotdict(), run_id="resume-test", store=store)
        assert call_count["prep"] == 1  # not called again

    def test_inline_run_id_context_cleanup(self):
        """run_id context should be cleaned up after inline()."""
        from msgflux.context import get_run_id
        from msgflux.dsl.inline import inline

        def prep(msg):
            msg["x"] = 1
            return msg

        inline("prep", {"prep": prep}, dotdict(), run_id="cleanup-test")
        assert get_run_id() is None

    def test_inline_run_id_context_cleanup_on_error(self):
        """run_id context should be cleaned up even on error."""
        from msgflux.context import get_run_id
        from msgflux.dsl.inline import inline

        def fail(msg):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            inline("fail", {"fail": fail}, dotdict(), run_id="cleanup-err")
        assert get_run_id() is None
