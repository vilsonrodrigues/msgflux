"""E2E integration tests for Agent durable execution with real API calls."""

import msgflux as mf
from msgflux import nn


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Weather in {city}: 28°C, sunny"


def get_population(city: str) -> str:
    """Get the population of a city."""
    return f"Population of {city}: 900,000"


class WeatherAgent(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-nano")
    instructions = "Answer the user question using the available tools."
    tools = [get_weather, get_population]


def test_durable_normal_flow_with_tool_calls():
    """Agent completes with tool calls and checkpoints are saved."""
    store = mf.InMemoryCheckpointStore()
    agent = WeatherAgent(checkpointer=store)

    chat = mf.ChatMessages(session_id="user_1")
    response = agent("What is the weather in Natal?", messages=chat)

    assert isinstance(response, str)
    assert "28" in response or "natal" in response.lower() or "sunny" in response.lower()

    # Checkpoint should be completed
    runs = store.list_runs("WeatherAgent", "user_1")
    assert len(runs) >= 1
    assert runs[0]["status"] == "completed"

    # State should contain full conversation
    state = store.load_state("WeatherAgent", "user_1", runs[0]["run_id"])
    assert state is not None
    restored = mf.ChatMessages()
    restored._hydrate_state(state["messages"])
    chatml = restored.to_chatml()
    # At minimum: user + assistant(tool_call) + tool + assistant(final)
    assert len(chatml) >= 3


def test_durable_resume_after_simulated_crash():
    """Simulate crash mid-tool-loop and resume from checkpoint."""
    store = mf.InMemoryCheckpointStore()

    # Step 1: Create initial agent, start task, save running checkpoint
    agent1 = WeatherAgent(checkpointer=store)
    chat1 = mf.ChatMessages(session_id="crash_test")
    chat1.begin_turn(inputs="What is the weather and population of Natal?")
    chat1.add_user("What is the weather and population of Natal?")

    # Manually save a "running" checkpoint (as if agent crashed after 1st tool call)
    chat1.extend([
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_sim_1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city":"Natal"}'}}
        ]},
        {"role": "tool", "tool_call_id": "call_sim_1",
         "content": "Weather in Natal: 28°C, sunny"},
    ])
    agent1._checkpoint_save(chat1, {}, status="running")

    # Step 2: New agent (simulates process restart) should resume
    agent2 = WeatherAgent(checkpointer=store)
    fresh_chat = mf.ChatMessages(session_id="crash_test")
    response = agent2("This is ignored on resume", messages=fresh_chat)

    assert isinstance(response, str)
    # The model should use the tool result from the checkpoint
    assert "28" in response or "natal" in response.lower() or "weather" in response.lower()

    # Checkpoint should now be completed
    runs = store.list_runs("WeatherAgent", "crash_test")
    completed = [r for r in runs if r["status"] == "completed"]
    assert len(completed) >= 1


def test_durable_sqlite_persistence():
    """Full cycle with SQLite: run -> crash -> restart -> resume."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "durable_e2e.sqlite3")

        # Step 1: First process — partial execution
        store1 = mf.SQLiteCheckpointStore(path=db_path)
        agent1 = WeatherAgent(checkpointer=store1)

        chat = mf.ChatMessages(session_id="sqlite_test")
        chat.begin_turn(inputs="What is the weather in Natal?")
        chat.add_user("What is the weather in Natal?")
        agent1._checkpoint_save(chat, {"source": "e2e"}, status="running")
        store1.close()

        # Step 2: Second process — resume
        store2 = mf.SQLiteCheckpointStore(path=db_path)
        agent2 = WeatherAgent(checkpointer=store2)

        fresh = mf.ChatMessages(session_id="sqlite_test")
        response = agent2("ignored", messages=fresh)
        assert isinstance(response, str)

        runs = store2.list_runs("WeatherAgent", "sqlite_test")
        completed = [r for r in runs if r["status"] == "completed"]
        assert len(completed) >= 1
        store2.close()


def test_durable_no_resume_after_completed():
    """After a completed run, next call should NOT resume (starts fresh)."""
    store = mf.InMemoryCheckpointStore()
    agent = WeatherAgent(checkpointer=store)

    # First call - completes
    chat1 = mf.ChatMessages(session_id="multi")
    r1 = agent("What is the weather in Natal?", messages=chat1)
    assert isinstance(r1, str)

    # Second call - should be a fresh execution, not resume
    chat2 = mf.ChatMessages(session_id="multi")
    r2 = agent("What is the population of Natal?", messages=chat2)
    assert isinstance(r2, str)

    # Final state should be completed (UPSERT overwrites same turn slot)
    runs = store.list_runs("WeatherAgent", "multi")
    assert len(runs) >= 1
    assert runs[0]["status"] == "completed"


if __name__ == "__main__":
    print("=== Test 1: Normal flow with tool calls ===")
    test_durable_normal_flow_with_tool_calls()
    print("PASSED\n")

    print("=== Test 2: Resume after simulated crash ===")
    test_durable_resume_after_simulated_crash()
    print("PASSED\n")

    print("=== Test 3: SQLite persistence ===")
    test_durable_sqlite_persistence()
    print("PASSED\n")

    print("=== Test 4: No resume after completed ===")
    test_durable_no_resume_after_completed()
    print("PASSED\n")

    print("All e2e durable tests passed!")
