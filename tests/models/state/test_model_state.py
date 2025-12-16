"""Tests for ModelState."""

from msgflux.models.state import (
    ModelState,
    LifecycleType,
    Policy,
    Role,
    ToolCall,
)


class TestModelStateBasics:
    """Basic ModelState functionality tests."""

    def test_create_empty_model_state(self):
        model_state = ModelState()
        assert model_state.message_count == 0
        assert model_state.adapter_name == "openai-chat"
        assert model_state.current_branch == "main"

    def test_add_user_message(self):
        model_state = ModelState()
        msg = model_state.add_user("Hello!")

        assert model_state.message_count == 1
        assert msg.role == Role.USER
        assert msg.text == "Hello!"
        assert msg.index == 0

    def test_add_assistant_message(self):
        model_state = ModelState()
        model_state.add_user("Hello!")
        msg = model_state.add_assistant("Hi there!")

        assert model_state.message_count == 2
        assert msg.role == Role.ASSISTANT
        assert msg.text == "Hi there!"
        assert msg.index == 1

    def test_add_tool_result(self):
        model_state = ModelState()
        model_state.add_user("Calculate 2+2")
        model_state.add_assistant(
            tool_calls=[ToolCall(id="call_123", name="calculator", arguments={"expr": "2+2"})]
        )
        msg = model_state.add_tool_result("call_123", "4")

        assert model_state.message_count == 3
        assert msg.role == Role.TOOL
        assert msg.tool_result.content == "4"
        assert msg.tool_result.call_id == "call_123"

    def test_get_by_role(self):
        model_state = ModelState()
        model_state.add_user("Hello")
        model_state.add_assistant("Hi")
        model_state.add_user("How are you?")
        model_state.add_assistant("I'm good!")

        users = model_state.get_by_role(Role.USER)
        assistants = model_state.get_by_role(Role.ASSISTANT)

        assert len(users) == 2
        assert len(assistants) == 2

    def test_get_last_messages(self):
        model_state = ModelState()
        model_state.add_user("1")
        model_state.add_assistant("2")
        model_state.add_user("3")
        model_state.add_assistant("4")

        last_two = model_state.get_last(2)
        assert len(last_two) == 2
        assert last_two[0].text == "3"
        assert last_two[1].text == "4"

    def test_clear_messages(self):
        model_state = ModelState()
        model_state.add_user("Hello")
        model_state.add_assistant("Hi")

        assert model_state.message_count == 2
        model_state.clear()
        assert model_state.message_count == 0


class TestLifecycleManagement:
    """Tests for message lifecycle management."""

    def test_ephemeral_turns_ttl(self):
        model_state = ModelState()

        # Add permanent message
        model_state.add_user("Permanent")

        # Add ephemeral message with 2 turns TTL
        model_state.add_user(
            "Temporary",
            lifecycle=LifecycleType.EPHEMERAL_TURNS,
            ttl_turns=2,
        )

        assert model_state.message_count == 2

        # Advance turns
        model_state.advance_turn()
        ephemeral = model_state.get_by_lifecycle(LifecycleType.EPHEMERAL_TURNS)
        assert len(ephemeral) == 1
        assert ephemeral[0].metadata.ttl_turns == 1

        model_state.advance_turn()
        ephemeral = model_state.get_by_lifecycle(LifecycleType.EPHEMERAL_TURNS)
        assert len(ephemeral) == 1
        assert ephemeral[0].metadata.ttl_turns == 0

    def test_scope_context_manager(self):
        model_state = ModelState()

        model_state.add_user("Before scope")

        with model_state.scope("tool_loop"):
            assert model_state.current_scope == "tool_loop"
            model_state.add_assistant("In scope")
            msg = model_state.messages[-1]
            assert msg.metadata.scope_id == "tool_loop"

        assert model_state.current_scope is None
        model_state.add_user("After scope")
        msg = model_state.messages[-1]
        assert msg.metadata.scope_id is None

    def test_nested_scopes(self):
        model_state = ModelState()

        with model_state.scope("outer"):
            assert model_state.current_scope == "outer"
            model_state.add_user("In outer")

            with model_state.scope("inner"):
                assert model_state.current_scope == "inner"
                model_state.add_user("In inner")

        assert model_state.current_scope is None


class TestVersionControl:
    """Tests for Git-like version control."""

    def test_commit(self):
        model_state = ModelState()
        model_state.add_user("Hello")
        model_state.add_assistant("Hi")

        commit = model_state.commit("First checkpoint")
        assert commit.commit_message == "First checkpoint"
        assert len(commit.message_hashes) == 2

    def test_branch_and_checkout(self):
        model_state = ModelState()
        model_state.add_user("Main message")
        model_state.commit("Main commit")

        # Create branch
        branch = model_state.branch("experiment")
        assert branch.name == "experiment"
        assert "experiment" in model_state.branches

        # Checkout branch
        model_state.checkout("experiment")
        assert model_state.current_branch == "experiment"

        # Add to branch
        model_state.add_user("Experiment message")
        model_state.commit("Experiment commit")
        assert model_state.message_count == 2

        # Go back to main
        model_state.checkout("main")
        assert model_state.current_branch == "main"
        assert model_state.message_count == 1

    def test_create_branch_on_checkout(self):
        model_state = ModelState()
        model_state.add_user("Hello")
        model_state.commit("Initial")

        model_state.checkout("feature", create=True)
        assert model_state.current_branch == "feature"
        assert "feature" in model_state.branches

    def test_merge_branches(self):
        model_state = ModelState()
        model_state.add_user("Main 1")
        model_state.commit("Main commit 1")

        model_state.branch("feature")
        model_state.checkout("feature")
        model_state.add_user("Feature 1")
        model_state.commit("Feature commit")

        model_state.checkout("main")
        merge_commit = model_state.merge("feature")

        # Main had 1, feature had 2 (Main 1 + Feature 1), concat = 3
        # But dedupe would give 2
        assert model_state.message_count == 3
        assert "Merge" in merge_commit.commit_message

    def test_revert(self):
        model_state = ModelState()
        model_state.add_user("Message 1")
        model_state.commit("Commit 1")

        model_state.add_user("Message 2")
        model_state.commit("Commit 2")

        assert model_state.message_count == 2

        model_state.revert(1)
        assert model_state.message_count == 1

    def test_log(self):
        model_state = ModelState()
        model_state.add_user("1")
        model_state.commit("First")
        model_state.add_user("2")
        model_state.commit("Second")
        model_state.add_user("3")
        model_state.commit("Third")

        history = model_state.log(limit=5)
        assert len(history) >= 3
        assert history[0].commit_message == "Third"


class TestPolicyAndCompaction:
    """Tests for compaction policies."""

    def test_create_with_policy(self):
        model_state = ModelState(
            policy=Policy(type="sliding_window", max_messages=10)
        )
        assert model_state._policy is not None

    def test_needs_compaction(self):
        model_state = ModelState(
            policy=Policy(type="sliding_window", max_messages=5)
        )

        for i in range(3):
            model_state.add_user(f"Message {i}")

        assert not model_state.needs_compaction()

        for i in range(3):
            model_state.add_user(f"Message {i+3}")

        assert model_state.needs_compaction()

    def test_compact_sliding_window(self):
        model_state = ModelState(
            policy=Policy(type="sliding_window", max_messages=3)
        )

        for i in range(6):
            model_state.add_user(f"Message {i}")

        result = model_state.compact()
        assert result.stats.get("compacted") is True
        assert model_state.message_count <= 3


class TestProviderFormat:
    """Tests for provider format conversion."""

    def test_to_provider_openai(self):
        model_state = ModelState(adapter="openai-chat")
        model_state.add_user("Hello!")
        model_state.add_assistant("Hi there!")

        result = model_state.to_provider(system_prompt="You are helpful")

        assert "messages" in result
        messages = result["messages"]

        # System prompt should be first
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful"

        # Then user and assistant
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Hello!"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "Hi there!"

    def test_to_provider_with_tool_calls(self):
        model_state = ModelState()
        model_state.add_user("Calculate")
        model_state.add_assistant(
            tool_calls=[ToolCall(id="call_1", name="calc", arguments={"x": 1})]
        )
        model_state.add_tool_result("call_1", "42")

        result = model_state.to_provider()
        messages = result["messages"]

        assert len(messages) == 3
        assert "tool_calls" in messages[1]
        assert messages[2]["role"] == "tool"
        assert messages[2]["tool_call_id"] == "call_1"


class TestSerialization:
    """Tests for serialization and persistence."""

    def test_serialize_deserialize_msgpack(self):
        model_state = ModelState()
        model_state.add_user("Hello")
        model_state.add_assistant("Hi")
        model_state.commit("Checkpoint")

        data = model_state.serialize()
        restored = ModelState.deserialize(data)

        assert restored.message_count == 2
        assert restored.messages[0].text == "Hello"
        assert restored.messages[1].text == "Hi"

    def test_serialize_deserialize_json(self):
        model_state = ModelState()
        model_state.add_user("Hello")
        model_state.add_assistant("Hi")
        model_state.commit("Checkpoint")

        json_str = model_state.to_json()
        restored = ModelState.from_json(json_str)

        assert restored.message_count == 2
        assert restored.messages[0].text == "Hello"

    def test_serialize_with_policy(self):
        model_state = ModelState(
            policy={"type": "sliding_window", "max_messages": 10}
        )
        model_state.add_user("Test")
        model_state.commit("Test")

        data = model_state.serialize()
        restored = ModelState.deserialize(data)

        assert restored._policy_config is not None


class TestStats:
    """Tests for statistics and utilities."""

    def test_stats(self):
        model_state = ModelState()
        model_state.add_user("Hello")
        model_state.add_assistant("Hi")
        model_state.add_user(
            "Temp",
            lifecycle=LifecycleType.EPHEMERAL_TURNS,
            ttl_turns=2,
        )

        stats = model_state.stats()

        assert stats["message_count"] == 3
        assert stats["messages_by_role"]["user"] == 2
        assert stats["messages_by_role"]["assistant"] == 1
        assert stats["messages_by_lifecycle"]["permanent"] == 2
        assert stats["messages_by_lifecycle"]["ephemeral_turns"] == 1

    def test_gc(self):
        model_state = ModelState()
        model_state.add_user("1")
        model_state.commit("C1")
        model_state.add_user("2")
        model_state.commit("C2")

        # Create orphan branch and delete it
        model_state.branch("temp")
        model_state.checkout("temp")
        model_state.add_user("Temp")
        model_state.commit("Temp")
        model_state.checkout("main")
        model_state._history.delete_branch("temp")

        # GC should clean up
        removed = model_state.gc()
        assert removed >= 0

    def test_repr(self):
        model_state = ModelState()
        model_state.add_user("Hello")

        repr_str = repr(model_state)
        assert "ModelState" in repr_str
        assert "openai-chat" in repr_str

    def test_len_and_iter(self):
        model_state = ModelState()
        model_state.add_user("1")
        model_state.add_user("2")
        model_state.add_user("3")

        assert len(model_state) == 3

        messages = list(model_state)
        assert len(messages) == 3
