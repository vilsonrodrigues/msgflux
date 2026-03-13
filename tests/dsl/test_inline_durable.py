"""Tests for durable inline DSL — checkpoint per step, resume, and delta pattern."""

from unittest.mock import patch

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.context import get_session_context
from msgflux.data.stores import InMemoryCheckpointStore
from msgflux.dotdict import DELETE, dotdict
from msgflux.dsl.inline import DurableInlineDSL, Inline, InlineDSL
from msgflux.dsl.inline.runtime import AsyncDurableInlineDSL


# ── Helpers ──────────────────────────────────────────────────────────────────


def _prep(msg):
    msg["output"] = {"agent": "xpto", "score": 10}
    msg["counter"] = 0
    return msg


def _increment(msg):
    msg["counter"] = msg.get("counter", 0) + 1
    return msg


def _feat_a(msg):
    msg["feat_a"] = "result_a"
    return msg


def _feat_b(msg):
    msg["feat_b"] = "result_b"
    return msg


def _final(msg):
    msg["final"] = "done"
    return msg


MODULES = {
    "prep": _prep,
    "increment": _increment,
    "feat_a": _feat_a,
    "feat_b": _feat_b,
    "final": _final,
}


# ── Delta-returning modules ─────────────────────────────────────────────────


def _delta_prep(msg):
    """Returns a dict delta instead of mutating."""
    return {"output": {"agent": "xpto", "score": 10}, "counter": 0}


def _delta_increment(msg):
    return {"counter": msg.get("counter", 0) + 1}


def _delta_feat_a(msg):
    return {"feat_a": "result_a"}


def _delta_feat_b(msg):
    return {"feat_b": "result_b"}


def _delta_final(msg):
    return {"final": "done"}


DELTA_MODULES = {
    "prep": _delta_prep,
    "increment": _delta_increment,
    "feat_a": _delta_feat_a,
    "feat_b": _delta_feat_b,
    "final": _delta_final,
}


# ── dotdict.apply ────────────────────────────────────────────────────────────


class TestDotdictApply:
    def test_apply_dict(self):
        msg = dotdict({"x": 1})
        msg.apply({"y": 2, "z": 3})
        assert msg["x"] == 1
        assert msg["y"] == 2
        assert msg["z"] == 3

    def test_apply_none_noop(self):
        msg = dotdict({"x": 1})
        result = msg.apply(None)
        assert result is msg
        assert msg["x"] == 1

    def test_apply_overwrite(self):
        msg = dotdict({"x": 1})
        msg.apply({"x": 99})
        assert msg["x"] == 99

    def test_apply_delete(self):
        msg = dotdict({"x": 1, "y": 2})
        msg.apply({"x": DELETE})
        assert "x" not in msg
        assert msg["y"] == 2

    def test_apply_deep_merge(self):
        msg = dotdict({"nested": {"a": 1, "b": 2}})
        msg.apply({"nested": {"b": 99, "c": 3}})
        assert msg["nested"]["a"] == 1
        assert msg["nested"]["b"] == 99
        assert msg["nested"]["c"] == 3

    def test_apply_returns_self(self):
        msg = dotdict()
        result = msg.apply({"key": "val"})
        assert result is msg

    def test_apply_rejects_non_dict(self):
        msg = dotdict()
        with pytest.raises(TypeError, match="must be a dict"):
            msg.apply("not a dict")


# ── Delta pattern in InlineDSL ──────────────────────────────────────────────


class TestDeltaPattern:
    def test_delta_sequential(self):
        """Modules returning dicts get applied correctly."""
        msg = dotdict()
        result = Inline("prep -> final", DELTA_MODULES)(msg)
        assert result["counter"] == 0
        assert result["final"] == "done"

    def test_delta_parallel(self):
        msg = dotdict()
        result = Inline("prep -> [feat_a, feat_b] -> final", DELTA_MODULES)(msg)
        assert result["feat_a"] == "result_a"
        assert result["feat_b"] == "result_b"
        assert result["final"] == "done"

    def test_delta_while_loop(self):
        msg = dotdict()
        result = Inline(
            "prep -> @{counter < 3}: increment; -> final",
            DELTA_MODULES,
        )(msg)
        assert result["counter"] == 3
        assert result["final"] == "done"

    def test_delta_conditional(self):
        msg = dotdict()
        result = Inline(
            "prep -> {output.agent == 'xpto'?feat_a,feat_b} -> final",
            DELTA_MODULES,
        )(msg)
        assert result["feat_a"] == "result_a"
        assert "feat_b" not in result
        assert result["final"] == "done"

    def test_mixed_legacy_and_delta(self):
        """Mix legacy (in-place) and delta modules."""
        mixed = {
            "prep": _prep,           # legacy
            "feat_a": _delta_feat_a,  # delta
            "final": _final,         # legacy
        }
        msg = dotdict()
        result = Inline("prep -> feat_a -> final", mixed)(msg)
        assert result["counter"] == 0
        assert result["feat_a"] == "result_a"
        assert result["final"] == "done"


# ── DurableInlineDSL — checkpoint and resume ─────────────────────────────────


class TestDurableInlineCheckpoint:
    def test_checkpoint_saved_per_step(self):
        store = InMemoryCheckpointStore()
        dsl = DurableInlineDSL(
            store, namespace="test", session_id="s1", run_id="r1",
        )
        msg = dotdict()
        dsl("prep -> feat_a -> final", DELTA_MODULES, msg)

        state = store.load_state("test", "s1", "r1")
        assert state is not None
        assert state["status"] == "completed"
        assert state["message_snapshot"]["final"] == "done"

    def test_resume_from_crash(self):
        store = InMemoryCheckpointStore()

        # Simulate: step 0 (prep) completed, step 1 (fail) crashed
        store.save_state("test", "s1", "r1", {
            "run_id": "r1",
            "expression": "prep -> feat_a -> final",
            "status": "running",
            "cursor": {"step_index": 1, "frames": []},
            "message_snapshot": {"output": {"agent": "xpto", "score": 10}, "counter": 0},
        })

        # Track which modules actually ran
        ran = []

        def tracking_feat_a(msg):
            ran.append("feat_a")
            return {"feat_a": "result_a"}

        def tracking_final(msg):
            ran.append("final")
            return {"final": "done"}

        modules = {
            "prep": lambda msg: (_ for _ in ()).throw(RuntimeError("should not run")),
            "feat_a": tracking_feat_a,
            "final": tracking_final,
        }

        dsl = DurableInlineDSL(
            store, namespace="test", session_id="s1", run_id="r1",
        )
        msg = dotdict()
        result = dsl("prep -> feat_a -> final", modules, msg)

        assert result["counter"] == 0  # from snapshot
        assert result["feat_a"] == "result_a"
        assert result["final"] == "done"
        assert "prep" not in ran  # prep was skipped
        assert ran == ["feat_a", "final"]

    def test_resume_skips_completed_expression_mismatch(self):
        """If expression changed, don't resume — start fresh."""
        store = InMemoryCheckpointStore()

        store.save_state("test", "s1", "r1", {
            "run_id": "r1",
            "expression": "old_expr",
            "status": "running",
            "cursor": {"step_index": 2, "frames": []},
            "message_snapshot": {"old": True},
        })

        dsl = DurableInlineDSL(
            store, namespace="test", session_id="s1", run_id="r1",
        )
        msg = dotdict()
        result = dsl("prep -> final", DELTA_MODULES, msg)

        assert result["final"] == "done"
        assert "old" not in result

    def test_completed_run_starts_fresh(self):
        """A completed run should not be resumed."""
        store = InMemoryCheckpointStore()

        store.save_state("test", "s1", "r1", {
            "run_id": "r1",
            "expression": "prep -> final",
            "status": "completed",
            "cursor": {"step_index": 2, "frames": []},
            "message_snapshot": {"stale": True},
        })

        dsl = DurableInlineDSL(
            store, namespace="test", session_id="s1", run_id="r1",
        )
        msg = dotdict()
        result = dsl("prep -> final", DELTA_MODULES, msg)

        assert result["final"] == "done"
        assert "stale" not in result

    def test_error_saves_failed_state(self):
        store = InMemoryCheckpointStore()

        def crasher(msg):
            raise ValueError("boom")

        modules = {"prep": _delta_prep, "crash": crasher, "final": _delta_final}

        dsl = DurableInlineDSL(
            store, namespace="test", session_id="s1", run_id="r1",
        )
        msg = dotdict()

        with pytest.raises(ValueError, match="boom"):
            dsl("prep -> crash -> final", modules, msg)

        state = store.load_state("test", "s1", "r1")
        assert state["status"] == "failed"
        assert "boom" in state["error"]


# ── While loop resume ────────────────────────────────────────────────────────


class TestDurableWhileResume:
    def test_while_checkpoint_per_iteration(self):
        store = InMemoryCheckpointStore()

        dsl = DurableInlineDSL(
            store, namespace="test", session_id="s1", run_id="r1",
        )
        msg = dotdict()
        result = dsl(
            "prep -> @{counter < 3}: increment; -> final",
            DELTA_MODULES,
            msg,
        )
        assert result["counter"] == 3
        assert result["final"] == "done"

        state = store.load_state("test", "s1", "r1")
        assert state["status"] == "completed"

    def test_while_resume_mid_loop(self):
        store = InMemoryCheckpointStore()

        # Simulate crash after 2 iterations of while loop (step_index=1 is the while)
        store.save_state("test", "s1", "r1", {
            "run_id": "r1",
            "expression": "prep -> @{counter < 5}: increment; -> final",
            "status": "running",
            "cursor": {
                "step_index": 1,
                "frames": [{
                    "type": "while",
                    "condition": "counter < 5",
                    "iteration": 2,
                    "inner_step": 0,
                }],
            },
            "message_snapshot": {"output": {"agent": "xpto", "score": 10}, "counter": 2},
        })

        dsl = DurableInlineDSL(
            store, namespace="test", session_id="s1", run_id="r1",
        )
        msg = dotdict()
        result = dsl(
            "prep -> @{counter < 5}: increment; -> final",
            DELTA_MODULES,
            msg,
        )

        assert result["counter"] == 5
        assert result["final"] == "done"


# ── Async durable ────────────────────────────────────────────────────────────


class TestAsyncDurableInline:
    @pytest.mark.asyncio
    async def test_async_checkpoint_and_resume(self):
        store = InMemoryCheckpointStore()

        async def async_prep(msg):
            return {"output": {"agent": "xpto"}, "counter": 0}

        async def async_feat_a(msg):
            return {"feat_a": "result_a"}

        async def async_final(msg):
            return {"final": "done"}

        modules = {
            "prep": async_prep,
            "feat_a": async_feat_a,
            "final": async_final,
        }

        dsl = AsyncDurableInlineDSL(
            store, namespace="async_test", session_id="s1", run_id="r1",
        )
        msg = dotdict()
        result = await dsl("prep -> feat_a -> final", modules, msg)

        assert result["feat_a"] == "result_a"
        assert result["final"] == "done"

        state = store.load_state("async_test", "s1", "r1")
        assert state["status"] == "completed"

    @pytest.mark.asyncio
    async def test_async_resume_from_crash(self):
        store = InMemoryCheckpointStore()

        store.save_state("async_test", "s1", "r1", {
            "run_id": "r1",
            "expression": "prep -> feat_a -> final",
            "status": "running",
            "cursor": {"step_index": 1, "frames": []},
            "message_snapshot": {"counter": 0},
        })

        ran = []

        async def async_feat_a(msg):
            ran.append("feat_a")
            return {"feat_a": "ok"}

        async def async_final(msg):
            ran.append("final")
            return {"final": "done"}

        modules = {
            "prep": lambda msg: (_ for _ in ()).throw(RuntimeError("skip")),
            "feat_a": async_feat_a,
            "final": async_final,
        }

        dsl = AsyncDurableInlineDSL(
            store, namespace="async_test", session_id="s1", run_id="r1",
        )
        msg = dotdict()
        result = await dsl("prep -> feat_a -> final", modules, msg)

        assert result["counter"] == 0  # from snapshot
        assert result["feat_a"] == "ok"
        assert result["final"] == "done"
        assert ran == ["feat_a", "final"]



# ── Session context propagation ───────────────────────────────────────────────


class TestSessionContextPropagation:
    def test_durable_inline_propagates_session_id(self):
        """DurableInlineDSL sets ChatMessages.session_context during execution."""
        store = InMemoryCheckpointStore()
        captured = {}

        def capture_session(msg):
            ctx = get_session_context()
            captured["session_id"] = ctx["session_id"]
            captured["namespace"] = ctx["namespace"]
            return {"captured": True}

        modules = {"capture": capture_session}
        dsl = DurableInlineDSL(
            store, namespace="my_pipeline", session_id="user_42", run_id="r1",
        )
        dsl("capture", modules, dotdict())

        assert captured["session_id"] == "user_42"
        assert captured["namespace"] == "my_pipeline"

    def test_chatmessages_inherits_session_from_durable_inline(self):
        """ChatMessages created inside DurableInlineDSL inherit session_id."""
        store = InMemoryCheckpointStore()
        captured_sessions = []

        def create_chat(msg):
            chat = ChatMessages()
            captured_sessions.append(chat.session_id)
            return {"chat_created": True}

        modules = {"create_chat": create_chat}
        dsl = DurableInlineDSL(
            store, namespace="test", session_id="sess_abc", run_id="r1",
        )
        dsl("create_chat", modules, dotdict())

        assert captured_sessions == ["sess_abc"]

    def test_no_session_leak_after_durable_inline(self):
        """Session context is cleaned up after DurableInlineDSL completes."""
        store = InMemoryCheckpointStore()

        def noop(msg):
            return {"done": True}

        modules = {"noop": noop}
        dsl = DurableInlineDSL(
            store, namespace="test", session_id="temp_sess", run_id="r1",
        )
        dsl("noop", modules, dotdict())

        ctx = get_session_context()
        assert ctx["session_id"] is None

    @pytest.mark.asyncio
    async def test_async_durable_propagates_session_id(self):
        """AsyncDurableInlineDSL propagates session_id via ContextVar."""
        store = InMemoryCheckpointStore()
        captured = {}

        async def capture_session(msg):
            ctx = get_session_context()
            captured["session_id"] = ctx["session_id"]
            return {"captured": True}

        modules = {"capture": capture_session}
        dsl = AsyncDurableInlineDSL(
            store, namespace="async_pipe", session_id="async_user", run_id="r1",
        )
        await dsl("capture", modules, dotdict())

        assert captured["session_id"] == "async_user"


# ── Inline first-class object ────────────────────────────────────────────────


class TestInlineClass:
    def test_basic_call(self):
        """Inline() works as a first-class callable."""
        pipeline = Inline("prep -> final", DELTA_MODULES)
        result = pipeline(dotdict())

        assert result["counter"] == 0
        assert result["final"] == "done"

    def test_call_with_store(self):
        """Inline() supports durable mode via store kwarg."""
        store = InMemoryCheckpointStore()
        pipeline = Inline("prep -> feat_a -> final", DELTA_MODULES)

        result = pipeline(
            dotdict(),
            store=store,
            session_id="s1",
            run_id="r1",
            namespace="test",
        )

        assert result["feat_a"] == "result_a"
        assert result["final"] == "done"

        state = store.load_state("test", "s1", "r1")
        assert state["status"] == "completed"

    def test_session_propagation(self):
        """Inline() propagates session_id to ChatMessages context."""
        captured = {}

        def capture(msg):
            ctx = get_session_context()
            captured["session_id"] = ctx["session_id"]
            return {"captured": True}

        pipeline = Inline("capture", {"capture": capture})
        pipeline(dotdict(), session_id="user_99")

        assert captured["session_id"] == "user_99"

    def test_repr(self):
        pipeline = Inline("a -> b", {"a": lambda m: m, "b": lambda m: m})
        assert "Inline(" in repr(pipeline)
        assert "a -> b" in repr(pipeline)

    def test_reusable(self):
        """Same Inline instance can be called multiple times."""
        pipeline = Inline("prep -> increment -> final", DELTA_MODULES)

        r1 = pipeline(dotdict())
        r2 = pipeline(dotdict())

        assert r1["counter"] == 1
        assert r2["counter"] == 1
        assert r1 is not r2

    @pytest.mark.asyncio
    async def test_acall(self):
        """Inline.acall() works asynchronously."""
        async def async_prep(msg):
            return {"counter": 0}

        async def async_final(msg):
            return {"final": "done"}

        modules = {"prep": async_prep, "final": async_final}
        pipeline = Inline("prep -> final", modules)

        result = await pipeline.acall(dotdict())
        assert result["final"] == "done"

    @pytest.mark.asyncio
    async def test_acall_with_store(self):
        """Inline.acall() supports durable mode."""
        store = InMemoryCheckpointStore()

        async def async_prep(msg):
            return {"counter": 0}

        async def async_final(msg):
            return {"final": "done"}

        modules = {"prep": async_prep, "final": async_final}
        pipeline = Inline("prep -> final", modules)

        result = await pipeline.acall(
            dotdict(), store=store, session_id="s1", run_id="r1",
        )
        assert result["final"] == "done"

        state = store.load_state("inline", "s1", "r1")
        assert state["status"] == "completed"

    def test_import_from_msgflux(self):
        """Inline is importable from msgflux root."""
        import msgflux
        assert hasattr(msgflux, "Inline")
        assert msgflux.Inline is Inline


# ── Retry logic ─────────────────────────────────────────────────────────────


class TestRetry:
    def test_retry_succeeds_on_second_attempt(self):
        """Step fails once then succeeds — no error raised."""
        store = InMemoryCheckpointStore()
        call_count = {"n": 0}

        def flaky(msg):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("transient")
            return {"ok": True}

        modules = {"flaky": flaky}
        dsl = DurableInlineDSL(
            store, namespace="t", session_id="s", run_id="r",
            max_retries=2, retry_delay=0,
        )
        result = dsl("flaky", modules, dotdict())

        assert result["ok"] is True
        assert call_count["n"] == 2

        state = store.load_state("t", "s", "r")
        assert state["status"] == "completed"

    def test_retry_exhausted_raises(self):
        """All retry attempts fail — saves failed state and raises."""
        store = InMemoryCheckpointStore()
        call_count = {"n": 0}

        def always_fail(msg):
            call_count["n"] += 1
            raise ValueError("permanent")

        modules = {"fail": always_fail}
        dsl = DurableInlineDSL(
            store, namespace="t", session_id="s", run_id="r",
            max_retries=2, retry_delay=0,
        )

        with pytest.raises(ValueError, match="permanent"):
            dsl("fail", modules, dotdict())

        # 1 initial + 2 retries = 3 attempts
        assert call_count["n"] == 3

        state = store.load_state("t", "s", "r")
        assert state["status"] == "failed"
        assert "permanent" in state["error"]

    def test_retry_zero_means_no_retry(self):
        """max_retries=0 means single attempt, no retry."""
        store = InMemoryCheckpointStore()
        call_count = {"n": 0}

        def fail_once(msg):
            call_count["n"] += 1
            raise RuntimeError("fail")

        modules = {"fail": fail_once}
        dsl = DurableInlineDSL(
            store, namespace="t", session_id="s", run_id="r",
            max_retries=0, retry_delay=0,
        )

        with pytest.raises(RuntimeError):
            dsl("fail", modules, dotdict())

        assert call_count["n"] == 1

    @patch("msgflux.dsl.inline.runtime.time.sleep")
    def test_retry_delay_is_respected(self, mock_sleep):
        """Retry delay is called between attempts."""
        store = InMemoryCheckpointStore()
        call_count = {"n": 0}

        def flaky(msg):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise RuntimeError("transient")
            return {"ok": True}

        modules = {"flaky": flaky}
        dsl = DurableInlineDSL(
            store, namespace="t", session_id="s", run_id="r",
            max_retries=3, retry_delay=2.5,
        )
        dsl("flaky", modules, dotdict())

        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(2.5)

    def test_retry_preserves_message_state(self):
        """Message state from earlier steps is preserved across retries."""
        store = InMemoryCheckpointStore()
        call_count = {"n": 0}

        def flaky_final(msg):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("transient")
            return {"final": "done"}

        modules = {
            "prep": _delta_prep,
            "flaky": flaky_final,
        }
        dsl = DurableInlineDSL(
            store, namespace="t", session_id="s", run_id="r",
            max_retries=1, retry_delay=0,
        )
        result = dsl("prep -> flaky", modules, dotdict())

        assert result["counter"] == 0  # prep ran before retry
        assert result["final"] == "done"

    @pytest.mark.asyncio
    async def test_async_retry_succeeds_on_second_attempt(self):
        """Async: step fails once then succeeds."""
        store = InMemoryCheckpointStore()
        call_count = {"n": 0}

        async def flaky(msg):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("transient")
            return {"ok": True}

        modules = {"flaky": flaky}
        dsl = AsyncDurableInlineDSL(
            store, namespace="t", session_id="s", run_id="r",
            max_retries=2, retry_delay=0,
        )
        result = await dsl("flaky", modules, dotdict())

        assert result["ok"] is True
        assert call_count["n"] == 2

    @pytest.mark.asyncio
    async def test_async_retry_exhausted_raises(self):
        """Async: all retries fail — raises and saves failed state."""
        store = InMemoryCheckpointStore()
        call_count = {"n": 0}

        async def always_fail(msg):
            call_count["n"] += 1
            raise ValueError("permanent")

        modules = {"fail": always_fail}
        dsl = AsyncDurableInlineDSL(
            store, namespace="t", session_id="s", run_id="r",
            max_retries=1, retry_delay=0,
        )

        with pytest.raises(ValueError, match="permanent"):
            await dsl("fail", modules, dotdict())

        assert call_count["n"] == 2

        state = store.load_state("t", "s", "r")
        assert state["status"] == "failed"


# ── Event emission ──────────────────────────────────────────────────────────


class TestEventEmission:
    def test_events_emitted_per_step(self):
        """Each completed step emits a step_completed event."""
        store = InMemoryCheckpointStore()
        dsl = DurableInlineDSL(
            store, namespace="t", session_id="s", run_id="r",
        )
        dsl("prep -> feat_a -> final", DELTA_MODULES, dotdict())

        events = store.load_events("t", "s", "r")
        types = [e["type"] for e in events]

        assert types[0] == "run_started"
        assert types[-1] == "run_completed"
        # 3 steps = 3 step_completed events
        step_events = [e for e in events if e["type"] == "step_completed"]
        assert len(step_events) == 3
        assert step_events[0]["step_name"] == "prep"
        assert step_events[1]["step_name"] == "feat_a"
        assert step_events[2]["step_name"] == "final"

    def test_events_contain_timestamp(self):
        """All events have a timestamp field."""
        store = InMemoryCheckpointStore()
        dsl = DurableInlineDSL(
            store, namespace="t", session_id="s", run_id="r",
        )
        dsl("prep -> final", DELTA_MODULES, dotdict())

        events = store.load_events("t", "s", "r")
        for event in events:
            assert "timestamp" in event
            assert isinstance(event["timestamp"], float)

    def test_step_failed_event_on_error(self):
        """A failed step emits a step_failed event with error info."""
        store = InMemoryCheckpointStore()

        def crasher(msg):
            raise ValueError("boom")

        modules = {"prep": _delta_prep, "crash": crasher}
        dsl = DurableInlineDSL(
            store, namespace="t", session_id="s", run_id="r",
        )

        with pytest.raises(ValueError, match="boom"):
            dsl("prep -> crash", modules, dotdict())

        events = store.load_events("t", "s", "r")
        failed = [e for e in events if e["type"] == "step_failed"]
        assert len(failed) == 1
        assert failed[0]["step_name"] == "crash"
        assert "boom" in failed[0]["error"]

    def test_run_resumed_event(self):
        """Resuming emits run_resumed instead of run_started."""
        store = InMemoryCheckpointStore()

        store.save_state("t", "s", "r", {
            "run_id": "r",
            "expression": "prep -> final",
            "status": "running",
            "cursor": {"step_index": 1, "frames": []},
            "message_snapshot": {"counter": 0},
        })

        dsl = DurableInlineDSL(
            store, namespace="t", session_id="s", run_id="r",
        )
        dsl("prep -> final", DELTA_MODULES, dotdict())

        events = store.load_events("t", "s", "r")
        assert events[0]["type"] == "run_resumed"

    def test_parallel_step_event_name(self):
        """Parallel steps include module names in step_name."""
        store = InMemoryCheckpointStore()
        dsl = DurableInlineDSL(
            store, namespace="t", session_id="s", run_id="r",
        )
        dsl("prep -> [feat_a, feat_b] -> final", DELTA_MODULES, dotdict())

        events = store.load_events("t", "s", "r")
        step_events = [e for e in events if e["type"] == "step_completed"]
        names = [e["step_name"] for e in step_events]
        assert "[feat_a, feat_b]" in names

    @pytest.mark.asyncio
    async def test_async_events_emitted(self):
        """Async durable also emits events."""
        store = InMemoryCheckpointStore()

        async def async_prep(msg):
            return {"counter": 0}

        async def async_final(msg):
            return {"final": "done"}

        modules = {"prep": async_prep, "final": async_final}
        dsl = AsyncDurableInlineDSL(
            store, namespace="t", session_id="s", run_id="r",
        )
        await dsl("prep -> final", modules, dotdict())

        events = store.load_events("t", "s", "r")
        types = [e["type"] for e in events]
        assert "run_started" in types
        assert "run_completed" in types
        assert types.count("step_completed") == 2
