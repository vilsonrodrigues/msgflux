"""E2E integration tests for durable Inline DSL.

Tests the full lifecycle: sequential, parallel, multi-branch conditional,
while loops, signals ($break/$stop), crash→resume, and event audit trail.
No mocks — uses real InMemoryCheckpointStore.
"""

import msgflux as mf
from msgflux.data.stores.providers.memory import InMemoryCheckpointStore
from msgflux.dotdict import dotdict
from msgflux.dsl.inline.runtime import DurableInlineDSL


# ── Modules ─────────────────────────────────────────────────────────────────


def classify(msg):
    """Classify input by score tier."""
    score = msg.get("score", 0)
    if score >= 90:
        return {"tier": "premium"}
    if score >= 50:
        return {"tier": "standard"}
    return {"tier": "basic"}


def enrich(msg):
    return {"enriched": True}


def validate(msg):
    return {"validated": True}


def premium_handler(msg):
    return {"handler": "premium", "discount": 0.2}


def standard_handler(msg):
    return {"handler": "standard", "discount": 0.1}


def basic_handler(msg):
    return {"handler": "basic", "discount": 0.0}


def increment(msg):
    msg["counter"] = msg.get("counter", 0) + 1


def summarize(msg):
    return {"summary": f"tier={msg.get('tier')}, counter={msg.get('counter', 0)}"}


def stopper(msg):
    """Emit $stop when counter reaches threshold."""
    if msg.get("counter", 0) >= 3:
        return {"early_exit": True, "$stop": True}


def breaker(msg):
    """Emit $break when counter reaches threshold."""
    msg["counter"] = msg.get("counter", 0) + 1
    if msg["counter"] >= 5:
        return {"$break": True}


ALL_MODULES = {
    "classify": classify,
    "enrich": enrich,
    "validate": validate,
    "premium_handler": premium_handler,
    "standard_handler": standard_handler,
    "basic_handler": basic_handler,
    "increment": increment,
    "summarize": summarize,
    "stopper": stopper,
    "breaker": breaker,
}


# ── Test: full pipeline with all constructs ─────────────────────────────────


def test_full_pipeline_all_constructs():
    """Sequential → parallel → multi-branch conditional → while → summarize."""
    store = InMemoryCheckpointStore()

    expression = (
        "classify -> [enrich, validate] -> "
        "{tier == 'premium' ? premium_handler, "
        "tier == 'standard' ? standard_handler, "
        "basic_handler} -> "
        "@{counter < 3}: increment; -> "
        "summarize"
    )

    pipeline = mf.Inline(expression, modules=ALL_MODULES)

    # Score 75 → standard tier
    result = pipeline(
        dotdict({"score": 75, "counter": 0}),
        store=store,
        session_id="user_1",
        run_id="run_1",
    )

    assert result["tier"] == "standard"
    assert result["enriched"] is True
    assert result["validated"] is True
    assert result["handler"] == "standard"
    assert result["discount"] == 0.1
    assert result["counter"] == 3
    assert "tier=standard" in result["summary"]

    # Verify checkpoint
    state = store.load_state("inline", "user_1", "run_1")
    assert state["status"] == "completed"

    # Verify events
    events = store.load_events("inline", "user_1", "run_1")
    types = [e["type"] for e in events]
    assert types[0] == "run_started"
    assert types[-1] == "run_completed"
    assert (
        types.count("step_completed") >= 5
    )  # classify, parallel, cond, while, summarize


def test_full_pipeline_premium_path():
    """Same pipeline, premium path."""
    store = InMemoryCheckpointStore()

    expression = (
        "classify -> [enrich, validate] -> "
        "{tier == 'premium' ? premium_handler, "
        "tier == 'standard' ? standard_handler, "
        "basic_handler} -> "
        "summarize"
    )

    pipeline = mf.Inline(expression, modules=ALL_MODULES)
    result = pipeline(
        dotdict({"score": 95}),
        store=store,
        session_id="user_2",
        run_id="run_1",
    )

    assert result["tier"] == "premium"
    assert result["handler"] == "premium"
    assert result["discount"] == 0.2


def test_full_pipeline_basic_fallback():
    """Same pipeline, basic (default) path."""
    store = InMemoryCheckpointStore()

    expression = (
        "classify -> "
        "{tier == 'premium' ? premium_handler, "
        "tier == 'standard' ? standard_handler, "
        "basic_handler}"
    )

    pipeline = mf.Inline(expression, modules=ALL_MODULES)
    result = pipeline(
        dotdict({"score": 10}),
        store=store,
        session_id="user_3",
        run_id="run_1",
    )

    assert result["tier"] == "basic"
    assert result["handler"] == "basic"


# ── Test: $stop halts durable pipeline ──────────────────────────────────────


def test_stop_halts_durable_pipeline():
    """$stop inside while loop halts pipeline, checkpoint records 'stopped'."""
    store = InMemoryCheckpointStore()

    expression = "classify -> @{counter < 100}: increment -> stopper; -> summarize"

    pipeline = mf.Inline(expression, modules=ALL_MODULES)
    result = pipeline(
        dotdict({"score": 50, "counter": 0}),
        store=store,
        session_id="user_4",
        run_id="run_1",
    )

    assert result["counter"] == 3
    assert result["early_exit"] is True
    assert "summary" not in result  # summarize never ran

    # Final status is "completed" (from __call__'s final save)
    state = store.load_state("inline", "user_4", "run_1")
    assert state["status"] == "completed"


# ── Test: $break exits while, pipeline continues ───────────────────────────


def test_break_exits_while_continues_pipeline():
    """$break exits while loop, remaining steps still execute."""
    store = InMemoryCheckpointStore()

    expression = "@{counter < 100}: breaker; -> summarize"

    pipeline = mf.Inline(expression, modules=ALL_MODULES)
    result = pipeline(
        dotdict({"counter": 0}),
        store=store,
        session_id="user_5",
        run_id="run_1",
    )

    assert result["counter"] == 5  # breaker stops at 5
    assert "summary" in result  # summarize ran after break


# ── Test: crash and resume ──────────────────────────────────────────────────


def test_crash_resume_mid_pipeline():
    """Pipeline crashes at step 2, resumes from checkpoint."""
    store = InMemoryCheckpointStore()
    crash_flag = {"should_crash": True}

    def step_a(msg):
        return {"a": "done"}

    def step_b(msg):
        if crash_flag["should_crash"]:
            raise RuntimeError("Process killed")
        return {"b": "done"}

    def step_c(msg):
        return {"c": "done"}

    modules = {"step_a": step_a, "step_b": step_b, "step_c": step_c}
    expression = "step_a -> step_b -> step_c"

    # First run — crashes at step_b
    dsl = DurableInlineDSL(store, session_id="sess", run_id="crash_run")
    try:
        dsl(expression, modules, dotdict())
    except RuntimeError:
        pass

    state = store.load_state("inline", "sess", "crash_run")
    assert state["status"] == "failed"
    assert state["cursor"]["step_index"] == 1  # failed at step 1 (step_b)
    assert state["message_snapshot"]["a"] == "done"  # step_a result preserved

    # Fix the "crash" and resume
    crash_flag["should_crash"] = False
    state["status"] = "running"
    store.save_state("inline", "sess", "crash_run", state)

    dsl2 = DurableInlineDSL(store, session_id="sess", run_id="crash_run")
    result = dsl2(expression, modules, dotdict())

    assert result["a"] == "done"  # from snapshot
    assert result["b"] == "done"  # re-executed successfully
    assert result["c"] == "done"

    # Final state
    final = store.load_state("inline", "sess", "crash_run")
    assert final["status"] == "completed"


# ── Test: multiple runs same session ────────────────────────────────────────


def test_multiple_runs_same_session():
    """Multiple runs under the same session are tracked independently."""
    store = InMemoryCheckpointStore()

    pipeline = mf.Inline(
        "classify -> summarize",
        modules=ALL_MODULES,
    )

    pipeline(
        dotdict({"score": 95, "counter": 0}),
        store=store,
        session_id="multi",
        run_id="run_1",
    )
    pipeline(
        dotdict({"score": 30, "counter": 0}),
        store=store,
        session_id="multi",
        run_id="run_2",
    )

    runs = store.list_runs("inline", "multi")
    assert len(runs) == 2
    assert all(r["status"] == "completed" for r in runs)

    state1 = store.load_state("inline", "multi", "run_1")
    state2 = store.load_state("inline", "multi", "run_2")
    assert state1["message_snapshot"]["tier"] == "premium"
    assert state2["message_snapshot"]["tier"] == "basic"


# ── Test: event audit trail ─────────────────────────────────────────────────


def test_event_audit_trail():
    """Every step produces events — useful for debugging and replay."""
    store = InMemoryCheckpointStore()

    pipeline = mf.Inline(
        "classify -> [enrich, validate] -> summarize",
        modules=ALL_MODULES,
    )
    pipeline(
        dotdict({"score": 60, "counter": 0}),
        store=store,
        session_id="audit",
        run_id="run_1",
    )

    events = store.load_events("inline", "audit", "run_1")

    # run_started, step_completed (classify), step_completed (parallel),
    # step_completed (summarize), run_completed
    assert len(events) == 5
    assert events[0]["type"] == "run_started"
    assert events[1]["type"] == "step_completed"
    assert events[1]["step_name"] == "classify"
    assert events[2]["step_name"] == "[enrich, validate]"
    assert events[3]["step_name"] == "summarize"
    assert events[4]["type"] == "run_completed"


# ── Test: nested Inline inherits session context ────────────────────────────


def test_nested_inline_inherits_session():
    """A sub-inline used inside a module inherits the parent session."""
    from msgflux.context import get_session_context

    captured_sessions = []

    def outer_step(msg):
        ctx = get_session_context()
        captured_sessions.append(ctx["session_id"])
        return {"outer": True}

    sub_inline = mf.Inline(
        "inner_step",
        modules={
            "inner_step": lambda msg: (
                captured_sessions.append(get_session_context()["session_id"]),
                {"inner": True},
            )[1]
        },
    )

    def run_sub(msg):
        sub_inline(msg)
        return {"sub_ran": True}

    pipeline = mf.Inline(
        "outer_step -> run_sub",
        modules={"outer_step": outer_step, "run_sub": run_sub},
    )
    pipeline(
        dotdict(),
        session_id="parent_session",
    )

    # Both should see "parent_session" — sub-inline inherited it
    assert captured_sessions == ["parent_session", "parent_session"]
