"""Tests for ChatMessages container."""

import pytest

from msgflux.chat_messages import ChatMessages
from msgflux.examples import Example


# ============================================================
# 1. Construction
# ============================================================


class TestConstruction:
    def test_empty(self):
        cm = ChatMessages()
        assert len(cm) == 0
        assert cm.session_id is None
        assert cm.namespace is None

    def test_from_list(self):
        items = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        cm = ChatMessages(items)
        assert len(cm) == 2

    def test_from_chatml_classmethod(self):
        items = [{"role": "user", "content": "hi"}]
        cm = ChatMessages.from_chatml(items)
        assert len(cm) == 1

    def test_session_id_and_namespace(self):
        cm = ChatMessages(session_id="s1", namespace="ns1")
        assert cm.session_id == "s1"
        assert cm.namespace == "ns1"

    def test_items_are_deepcopied(self):
        original = {"role": "user", "content": "hi"}
        cm = ChatMessages([original])
        original["content"] = "changed"
        assert cm[0]["content"] == "hi"

    def test_non_mapping_raises(self):
        with pytest.raises(TypeError, match="Mapping"):
            ChatMessages(["not a dict"])


# ============================================================
# 2. List-like operations
# ============================================================


class TestListOps:
    def test_append(self):
        cm = ChatMessages()
        cm.append({"role": "user", "content": "hello"})
        assert len(cm) == 1
        assert cm[0]["role"] == "user"

    def test_append_non_mapping_raises(self):
        cm = ChatMessages()
        with pytest.raises(TypeError, match="Mapping"):
            cm.append("not a dict")

    def test_insert(self):
        cm = ChatMessages(
            [
                {"role": "user", "content": "first"},
                {"role": "user", "content": "third"},
            ]
        )
        cm.insert(1, {"role": "user", "content": "second"})
        assert len(cm) == 3
        assert cm[1]["content"] == "second"

    def test_extend(self):
        cm = ChatMessages([{"role": "user", "content": "a"}])
        cm.extend(
            [
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
            ]
        )
        assert len(cm) == 3

    def test_extend_with_chat_messages(self):
        cm1 = ChatMessages([{"role": "user", "content": "a"}])
        cm2 = ChatMessages([{"role": "assistant", "content": "b"}])
        cm1.extend(cm2)
        assert len(cm1) == 2

    def test_getitem(self):
        cm = ChatMessages([{"role": "user", "content": "hello"}])
        assert cm[0]["content"] == "hello"

    def test_getitem_slice(self):
        items = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
        cm = ChatMessages(items)
        sliced = cm[0:2]
        assert len(sliced) == 2

    def test_setitem(self):
        cm = ChatMessages([{"role": "user", "content": "old"}])
        cm[0] = {"role": "user", "content": "new"}
        assert cm[0]["content"] == "new"

    def test_setitem_non_mapping_raises(self):
        cm = ChatMessages([{"role": "user", "content": "x"}])
        with pytest.raises(TypeError, match="Mapping"):
            cm[0] = "not a dict"

    def test_bool_empty(self):
        assert not ChatMessages()

    def test_bool_non_empty(self):
        assert ChatMessages([{"role": "user", "content": "hi"}])

    def test_iter(self):
        items = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        cm = ChatMessages(items)
        roles = [item["role"] for item in cm]
        assert roles == ["user", "assistant"]

    def test_repr(self):
        cm = ChatMessages([{"role": "user", "content": "hi"}], session_id="s1")
        r = repr(cm)
        assert "ChatMessages" in r
        assert "size=1" in r
        assert "s1" in r

    def test_copy(self):
        cm = ChatMessages(
            [{"role": "user", "content": "hi"}],
            session_id="s1",
            namespace="ns1",
        )
        cm.metadata = {"key": "val"}
        cm.reasoning_content = "think"
        cm.response_id = "resp_1"
        copied = cm.copy()
        assert len(copied) == 1
        assert copied.session_id == "s1"
        assert copied.namespace == "ns1"
        assert copied.metadata == {"key": "val"}
        assert copied.reasoning_content == "think"
        assert copied.response_id == "resp_1"
        # Ensure independence
        cm.append({"role": "assistant", "content": "bye"})
        assert len(copied) == 1


# ============================================================
# 3. Turn lifecycle
# ============================================================


class TestTurnLifecycle:
    def test_begin_end_turn(self):
        cm = ChatMessages()
        turn_id = cm.begin_turn(inputs={"q": "test"})
        assert turn_id == "turn_1"
        assert cm.get_active_turn() is not None
        result = cm.end_turn(assistant_output="answer", status="completed")
        assert result is not None
        assert result["status"] == "completed"
        assert result["assistant_output"] == "answer"
        assert cm.get_active_turn() is None

    def test_turns_property(self):
        cm = ChatMessages()
        cm.begin_turn(inputs="q1")
        cm.end_turn(assistant_output="a1")
        cm.begin_turn(inputs="q2")
        cm.end_turn(assistant_output="a2")
        turns = cm.turns
        assert len(turns) == 2

    def test_begin_turn_interrupts_active(self):
        cm = ChatMessages()
        cm.begin_turn(inputs="q1")
        # Start another without ending first
        cm.begin_turn(inputs="q2")
        turns = cm.turns
        assert turns[0]["status"] == "interrupted"

    def test_end_turn_no_active(self):
        cm = ChatMessages()
        result = cm.end_turn(assistant_output="x")
        assert result is None

    def test_turn_markers_in_items(self):
        cm = ChatMessages()
        cm.begin_turn()
        cm.add_user("hello")
        cm.end_turn()
        types = [item.get("type") or item.get("event") for item in cm]
        # turn_marker items have type="turn_marker"
        assert any(item.get("type") == "turn_marker" for item in cm)

    def test_custom_turn_id(self):
        cm = ChatMessages()
        tid = cm.begin_turn(turn_id="my_turn")
        assert tid == "my_turn"

    def test_turn_session_id(self):
        cm = ChatMessages(session_id="s1")
        cm.begin_turn()
        turn = cm.get_active_turn()
        assert turn["session_id"] == "s1"

    def test_turn_with_vars_and_metadata(self):
        cm = ChatMessages()
        cm.begin_turn(vars={"temperature": 0.5}, metadata={"source": "test"})
        turn = cm.get_active_turn()
        assert turn["vars"] == {"temperature": 0.5}
        assert turn["metadata"] == {"source": "test"}


# ============================================================
# 4. Content adders
# ============================================================


class TestContentAdders:
    def test_add_user(self):
        cm = ChatMessages()
        cm.add_user("hello")
        assert cm[0] == {"role": "user", "content": "hello"}

    def test_add_assistant(self):
        cm = ChatMessages()
        cm.add_assistant("response")
        assert cm[0] == {"role": "assistant", "content": "response"}

    def test_add_system(self):
        cm = ChatMessages()
        cm.add_system("system prompt")
        assert cm[0] == {"role": "system", "content": "system prompt"}

    def test_add_tool(self):
        cm = ChatMessages()
        cm.add_tool("call_123", "result data")
        assert cm[0]["role"] == "tool"
        assert cm[0]["tool_call_id"] == "call_123"

    def test_add_message(self):
        cm = ChatMessages()
        cm.add_message("developer", "dev msg")
        assert cm[0]["role"] == "developer"
        assert cm[0]["content"] == "dev msg"

    def test_add_message_invalid_role_type(self):
        cm = ChatMessages()
        with pytest.raises(TypeError, match="str"):
            cm.add_message(123, "content")

    def test_add_reasoning(self):
        cm = ChatMessages()
        cm.add_reasoning("thinking step by step")
        assert cm[0]["type"] == "reasoning"
        assert cm[0]["reasoning_content"] == "thinking step by step"
        assert cm[0]["role"] == "assistant"

    def test_add_assistant_response_with_reasoning(self):
        cm = ChatMessages()
        cm.add_assistant_response("answer", reasoning_content="thinking")
        assert len(cm) == 2
        assert cm[0]["type"] == "reasoning"
        assert cm[1]["role"] == "assistant"

    def test_add_assistant_response_no_reasoning(self):
        cm = ChatMessages()
        cm.add_assistant_response("answer")
        assert len(cm) == 1
        assert cm[0]["role"] == "assistant"

    def test_add_chatml(self):
        cm = ChatMessages()
        cm.add_chatml(
            [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
            ]
        )
        assert len(cm) == 2

    def test_add_response_items(self):
        cm = ChatMessages()
        cm.add_response_items(
            [
                {"type": "message", "role": "user", "content": "hello"},
            ]
        )
        assert len(cm) == 1

    def test_update_metadata(self):
        cm = ChatMessages()
        cm.update_metadata({"model": "gpt-4", "tokens": 100})
        assert cm.metadata["model"] == "gpt-4"
        assert cm.metadata["tokens"] == 100

    def test_update_metadata_invalid_type(self):
        cm = ChatMessages()
        with pytest.raises(TypeError, match="Mapping"):
            cm.update_metadata("not a dict")

    def test_set_response_id(self):
        cm = ChatMessages()
        cm.set_response_id("resp_abc")
        assert cm.response_id == "resp_abc"

    def test_set_response_id_none(self):
        cm = ChatMessages()
        cm.set_response_id("resp_abc")
        cm.set_response_id(None)
        assert cm.response_id is None


# ============================================================
# 5. Normalization
# ============================================================


class TestNormalization:
    def test_reasoning_content_extracted(self):
        cm = ChatMessages()
        cm.append(
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "my thinking",
            }
        )
        # Should be split into reasoning + assistant items
        assert len(cm) == 2
        assert cm[0]["type"] == "reasoning"
        assert cm[0]["reasoning_content"] == "my thinking"
        assert cm[1]["role"] == "assistant"

    def test_reasoning_text_extracted(self):
        cm = ChatMessages()
        cm.append(
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_text": "my thinking",
            }
        )
        assert len(cm) == 2
        assert cm[0]["type"] == "reasoning"

    def test_think_field_extracted(self):
        cm = ChatMessages()
        cm.append(
            {
                "role": "assistant",
                "content": "answer",
                "think": "my thinking",
            }
        )
        assert len(cm) == 2
        assert cm[0]["type"] == "reasoning"

    def test_reasoning_type_item(self):
        cm = ChatMessages()
        cm.append(
            {
                "type": "reasoning",
                "reasoning_content": "thinking",
            }
        )
        assert len(cm) == 1
        assert cm[0]["type"] == "reasoning"
        assert cm[0]["role"] == "assistant"

    def test_reasoning_with_content_list(self):
        cm = ChatMessages()
        cm.append(
            {
                "type": "reasoning",
                "content": [{"text": "part1"}, {"text": "part2"}],
            }
        )
        assert cm[0]["reasoning_content"] == "part1part2"

    def test_empty_reasoning_skipped(self):
        cm = ChatMessages()
        cm.append(
            {
                "type": "reasoning",
                "reasoning_content": "",
            }
        )
        assert len(cm) == 0

    def test_user_message_passes_through(self):
        cm = ChatMessages()
        cm.append({"role": "user", "content": "hi"})
        assert cm[0] == {"role": "user", "content": "hi"}


# ============================================================
# 6. Format conversion — to_chatml
# ============================================================


class TestToChatML:
    def test_simple_messages(self):
        cm = ChatMessages(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        )
        chatml = cm.to_chatml()
        assert len(chatml) == 2
        assert chatml[0] == {"role": "user", "content": "hi"}
        assert chatml[1] == {"role": "assistant", "content": "hello"}

    def test_turn_markers_filtered(self):
        cm = ChatMessages()
        cm.begin_turn()
        cm.add_user("hello")
        cm.add_assistant("world")
        cm.end_turn()
        chatml = cm.to_chatml()
        # Only user + assistant, no turn_markers
        assert all(item.get("type") != "turn_marker" for item in chatml)
        roles = [m["role"] for m in chatml]
        assert "user" in roles
        assert "assistant" in roles

    def test_function_call_converted(self):
        cm = ChatMessages()
        cm.append(
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city": "SP"}',
            }
        )
        chatml = cm.to_chatml()
        assert len(chatml) == 1
        msg = chatml[0]
        assert msg["role"] == "assistant"
        assert msg["tool_calls"][0]["id"] == "call_1"
        assert msg["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_function_call_output_converted(self):
        cm = ChatMessages()
        cm.append(
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "sunny",
            }
        )
        chatml = cm.to_chatml()
        assert chatml[0]["role"] == "tool"
        assert chatml[0]["tool_call_id"] == "call_1"
        assert chatml[0]["content"] == "sunny"

    def test_response_message_converted(self):
        cm = ChatMessages()
        cm.append(
            {
                "type": "message",
                "role": "user",
                "content": "hello",
            }
        )
        chatml = cm.to_chatml()
        assert chatml[0] == {"role": "user", "content": "hello"}

    def test_response_message_with_parts(self):
        cm = ChatMessages()
        cm.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            }
        )
        chatml = cm.to_chatml()
        # Single text part should be simplified to string
        assert chatml[0] == {"role": "assistant", "content": "hello"}

    def test_reasoning_converted_to_chatml(self):
        cm = ChatMessages()
        cm.add_reasoning("my thinking")
        chatml = cm.to_chatml()
        assert chatml[0]["role"] == "assistant"
        assert chatml[0]["content"] == "my thinking"


# ============================================================
# 7. Format conversion — to_responses_input
# ============================================================


class TestToResponsesInput:
    def test_simple_messages(self):
        cm = ChatMessages(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        )
        resp = cm.to_responses_input()
        assert len(resp) == 2
        assert resp[0]["type"] == "message"
        assert resp[0]["role"] == "user"
        assert resp[1]["type"] == "message"
        assert resp[1]["role"] == "assistant"

    def test_chatml_tool_call_converted(self):
        cm = ChatMessages(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "SP"}',
                            },
                        }
                    ],
                }
            ]
        )
        resp = cm.to_responses_input()
        assert resp[0]["type"] == "function_call"
        assert resp[0]["call_id"] == "call_1"

    def test_chatml_tool_result_converted(self):
        cm = ChatMessages(
            [
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "sunny",
                }
            ]
        )
        resp = cm.to_responses_input()
        assert resp[0]["type"] == "function_call_output"
        assert resp[0]["call_id"] == "call_1"
        assert resp[0]["output"] == "sunny"

    def test_function_call_passthrough(self):
        cm = ChatMessages()
        cm.append(
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city": "SP"}',
            }
        )
        resp = cm.to_responses_input()
        assert resp[0]["type"] == "function_call"
        assert resp[0]["call_id"] == "call_1"

    def test_function_call_output_passthrough(self):
        cm = ChatMessages()
        cm.append(
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "result",
            }
        )
        resp = cm.to_responses_input()
        assert resp[0]["type"] == "function_call_output"

    def test_turn_markers_filtered(self):
        cm = ChatMessages()
        cm.begin_turn()
        cm.add_user("hello")
        cm.end_turn()
        resp = cm.to_responses_input()
        assert all(item.get("type") != "turn_marker" for item in resp)


# ============================================================
# 8. Session management
# ============================================================


class TestSessionManagement:
    def test_session_context_sets_session_id(self):
        with ChatMessages.session_context(session_id="s1"):
            cm = ChatMessages()
            assert cm.session_id == "s1"
        # Outside context, should be None
        cm2 = ChatMessages()
        assert cm2.session_id is None

    def test_session_context_sets_namespace(self):
        with ChatMessages.session_context(namespace="ns1"):
            cm = ChatMessages()
            assert cm.namespace == "ns1"

    def test_session_context_auto_generates_id(self):
        with ChatMessages.session_context():
            cm = ChatMessages()
            assert cm.session_id is not None
            assert cm.session_id.startswith("sess_")

    def test_nested_session_contexts(self):
        with ChatMessages.session_context(session_id="outer"):
            cm1 = ChatMessages()
            assert cm1.session_id == "outer"
            with ChatMessages.session_context(session_id="inner"):
                cm2 = ChatMessages()
                assert cm2.session_id == "inner"
            cm3 = ChatMessages()
            assert cm3.session_id == "outer"

    def test_get_session_context(self):
        with ChatMessages.session_context(session_id="s1", namespace="ns1"):
            ctx = ChatMessages.get_session_context()
            assert ctx["session_id"] == "s1"
            assert ctx["namespace"] == "ns1"

    def test_configure_session(self):
        cm = ChatMessages()
        cm.configure_session(session_id="new_s", namespace="new_ns")
        assert cm.session_id == "new_s"
        assert cm.namespace == "new_ns"

    def test_configure_session_from_context(self):
        with ChatMessages.session_context(session_id="ctx_s"):
            cm = ChatMessages()
            cm.session_id = None  # Reset
            cm.configure_session()
            assert cm.session_id == "ctx_s"


# ============================================================
# 9. Serialization
# ============================================================


class TestSerialization:
    def test_to_state_roundtrip(self):
        cm = ChatMessages(
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            session_id="s1",
            namespace="ns1",
        )
        cm.metadata = {"model": "gpt-4"}
        cm.reasoning_content = "thinking"
        cm.response_id = "resp_1"

        state = cm._to_state()

        cm2 = ChatMessages()
        cm2._hydrate_state(state)

        assert len(cm2) == 2
        assert cm2.session_id == "s1"
        assert cm2.namespace == "ns1"
        assert cm2.metadata == {"model": "gpt-4"}
        assert cm2.reasoning_content == "thinking"
        assert cm2.response_id == "resp_1"

    def test_to_state_with_turns(self):
        cm = ChatMessages()
        cm.begin_turn(inputs="q1")
        cm.add_user("hello")
        cm.add_assistant("world")
        cm.end_turn(assistant_output="world")

        state = cm._to_state()

        cm2 = ChatMessages()
        cm2._hydrate_state(state)

        assert len(cm2._turns) == 1
        assert cm2._turns[0]["assistant_output"] == "world"

    def test_hydrate_invalid_state(self):
        cm = ChatMessages()
        cm._hydrate_state("not a mapping")  # should not raise
        assert len(cm) == 0

    def test_hydrate_partial_state(self):
        cm = ChatMessages()
        cm._hydrate_state({"items": [{"role": "user", "content": "hi"}]})
        assert len(cm) == 1

    def test_to_items(self):
        cm = ChatMessages([{"role": "user", "content": "hi"}])
        items = cm.to_items()
        assert items == [{"role": "user", "content": "hi"}]
        # Ensure it's a copy
        items.append({"role": "assistant", "content": "bye"})
        assert len(cm) == 1


# ============================================================
# 10. Fork
# ============================================================


class TestFork:
    def test_fork_full_copy(self):
        cm = ChatMessages(
            [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
            ]
        )
        forked = cm.fork()
        assert len(forked) == 2
        cm.append({"role": "user", "content": "c"})
        assert len(forked) == 2

    def test_fork_upto_turn(self):
        cm = ChatMessages(session_id="s1")
        cm.begin_turn(inputs="q1")
        cm.add_user("hello")
        cm.add_assistant("world")
        cm.end_turn(assistant_output="world")
        cm.begin_turn(inputs="q2")
        cm.add_user("foo")
        cm.add_assistant("bar")
        cm.end_turn(assistant_output="bar")

        forked = cm.fork(upto_turn=0)
        # Should only have items up to end of turn 0
        assert len(forked._turns) == 1
        assert forked._turns[0]["assistant_output"] == "world"
        # Items from turn 1 should not be present
        assert len(forked) < len(cm)

    def test_fork_negative(self):
        cm = ChatMessages([{"role": "user", "content": "hi"}], session_id="s1")
        cm.metadata = {"key": "val"}
        forked = cm.fork(upto_turn=-1)
        assert len(forked) == 0
        assert forked.session_id == "s1"
        assert forked.metadata == {"key": "val"}

    def test_fork_beyond_turns(self):
        cm = ChatMessages([{"role": "user", "content": "hi"}])
        forked = cm.fork(upto_turn=100)
        assert len(forked) == len(cm)

    def test_fork_invalid_type(self):
        cm = ChatMessages()
        with pytest.raises(TypeError, match="int or None"):
            cm.fork(upto_turn="bad")


# ============================================================
# 11. to_examples
# ============================================================


class TestToExamples:
    def test_basic_example(self):
        cm = ChatMessages()
        cm.begin_turn(inputs={"question": "What is 2+2?"})
        cm.add_user("What is 2+2?")
        cm.add_assistant("4")
        cm.end_turn(assistant_output="4")

        examples = cm.to_examples()
        assert len(examples) == 1
        assert isinstance(examples[0], Example)
        assert examples[0].labels == {"response": "4"}

    def test_examples_with_reasoning(self):
        cm = ChatMessages()
        cm.begin_turn(inputs={"q": "test"})
        cm.add_reasoning("let me think")
        cm.add_assistant("answer")
        cm.end_turn(assistant_output="answer")

        examples = cm.to_examples()
        assert examples[0].reasoning == "let me think"

    def test_examples_without_output_skipped(self):
        cm = ChatMessages()
        cm.begin_turn(inputs="q1")
        cm.add_user("hello")
        cm.end_turn()  # no assistant_output

        examples = cm.to_examples()
        assert len(examples) == 0

    def test_examples_with_vars(self):
        cm = ChatMessages()
        cm.begin_turn(inputs={"q": "test"}, vars={"temperature": 0.5})
        cm.add_assistant("result")
        cm.end_turn(assistant_output="result")

        examples = cm.to_examples()
        assert examples[0].inputs["vars"] == {"temperature": 0.5}

    def test_examples_without_history(self):
        cm = ChatMessages()
        cm.begin_turn(inputs={"q": "test"})
        cm.add_assistant("result")
        cm.end_turn(assistant_output="result")

        examples = cm.to_examples(include_history=False)
        assert "history" not in examples[0].inputs

    def test_examples_custom_keys(self):
        cm = ChatMessages()
        cm.begin_turn(inputs={"q": "test"})
        cm.add_assistant("result")
        cm.end_turn(assistant_output="result")

        examples = cm.to_examples(history_key="ctx", output_key="answer")
        assert "ctx" in examples[0].inputs
        assert "answer" in examples[0].labels


# ============================================================
# 12. Multimodal content building
# ============================================================


class TestMultimodal:
    def test_build_multimodal_text_only(self):
        content = ChatMessages.build_multimodal_content(text="hello")
        assert len(content) == 1
        assert content[0] == {"type": "text", "text": "hello"}

    def test_build_multimodal_no_input(self):
        content = ChatMessages.build_multimodal_content()
        assert content == []

    def test_build_multimodal_media_dict_passthrough(self):
        media_dict = {"type": "input_image", "image_url": "http://example.com/img.png"}
        content = ChatMessages.build_multimodal_content(
            text="look at this",
            media={"image": media_dict},
        )
        assert len(content) == 2
        assert content[0] == media_dict
        assert content[1] == {"type": "text", "text": "look at this"}

    def test_build_multimodal_invalid_media_type(self):
        with pytest.raises(TypeError, match="Mapping"):
            ChatMessages.build_multimodal_content(media="not a dict")

    def test_add_user_multimodal_no_media(self):
        cm = ChatMessages()
        cm.add_user_multimodal(text="hello")
        assert len(cm) == 1
        assert cm[0]["content"] == "hello"

    def test_add_user_multimodal_no_text_no_media(self):
        cm = ChatMessages()
        cm.add_user_multimodal()
        assert len(cm) == 1
        assert cm[0]["content"] == ""


# ============================================================
# 13. Audio URL conversion
# ============================================================


class TestAudioUrlConversion:
    def test_base64_audio_converted(self):
        cm = ChatMessages()
        result = cm._audio_url_to_input_audio(
            {
                "type": "audio_url",
                "audio_url": {"url": "data:audio/mp3;base64,AAAA"},
            }
        )
        assert result is not None
        assert result["type"] == "input_audio"
        assert result["input_audio"]["data"] == "AAAA"
        assert result["input_audio"]["format"] == "mp3"

    def test_non_base64_returns_none(self):
        cm = ChatMessages()
        result = cm._audio_url_to_input_audio(
            {
                "type": "audio_url",
                "audio_url": {"url": "https://example.com/audio.mp3"},
            }
        )
        assert result is None

    def test_no_audio_url_mapping(self):
        cm = ChatMessages()
        result = cm._audio_url_to_input_audio({"type": "audio_url", "audio_url": "str"})
        assert result is None


# ============================================================
# 14. Response part conversion (to_chatml helpers)
# ============================================================


class TestResponsePartConversion:
    def test_output_text_to_chatml(self):
        cm = ChatMessages()
        result = cm._response_part_to_chatml({"type": "output_text", "text": "hello"})
        assert result == {"type": "text", "text": "hello"}

    def test_input_text_to_chatml(self):
        cm = ChatMessages()
        result = cm._response_part_to_chatml({"type": "input_text", "text": "hello"})
        assert result == {"type": "text", "text": "hello"}

    def test_input_image_to_chatml(self):
        cm = ChatMessages()
        result = cm._response_part_to_chatml(
            {
                "type": "input_image",
                "image_url": "http://example.com/img.png",
                "detail": "high",
            }
        )
        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == "http://example.com/img.png"
        assert result["image_url"]["detail"] == "high"

    def test_input_file_to_chatml(self):
        cm = ChatMessages()
        result = cm._response_part_to_chatml(
            {
                "type": "input_file",
                "file_id": "file_123",
                "filename": "doc.pdf",
            }
        )
        assert result["type"] == "file"
        assert result["file"]["file_id"] == "file_123"

    def test_input_audio_to_chatml(self):
        cm = ChatMessages()
        result = cm._response_part_to_chatml(
            {
                "type": "input_audio",
                "input_audio": {"data": "AAAA", "format": "mp3"},
            }
        )
        assert result["type"] == "input_audio"
        assert result["input_audio"]["data"] == "AAAA"

    def test_unknown_type_returns_none(self):
        cm = ChatMessages()
        result = cm._response_part_to_chatml({"type": "unknown_type"})
        assert result is None


# ============================================================
# 15. Normalize tool output
# ============================================================


class TestNormalizeToolOutput:
    def test_string_passthrough(self):
        cm = ChatMessages()
        assert cm._normalize_tool_output("hello") == "hello"

    def test_none_returns_empty(self):
        cm = ChatMessages()
        assert cm._normalize_tool_output(None) == ""

    def test_mapping_with_text_type(self):
        cm = ChatMessages()
        result = cm._normalize_tool_output({"type": "text", "text": "hello"})
        assert result == [{"type": "input_text", "text": "hello"}]

    def test_list_of_parts(self):
        cm = ChatMessages()
        result = cm._normalize_tool_output(
            [
                {"type": "text", "text": "hello"},
                {"type": "input_text", "text": "world"},
            ]
        )
        assert len(result) == 2
        assert result[0] == {"type": "input_text", "text": "hello"}
        assert result[1] == {"type": "input_text", "text": "world"}


# ============================================================
# 16. Edge cases
# ============================================================


class TestEdgeCases:
    def test_response_message_invalid_role_filtered(self):
        cm = ChatMessages()
        result = cm._response_message_to_chatml({"role": "unknown", "content": "x"})
        assert result is None

    def test_response_message_string_content(self):
        cm = ChatMessages()
        result = cm._response_message_to_chatml({"role": "user", "content": "hello"})
        assert result == {"role": "user", "content": "hello"}

    def test_response_message_non_list_content(self):
        cm = ChatMessages()
        result = cm._response_message_to_chatml({"role": "user", "content": 42})
        assert result == {"role": "user", "content": "42"}

    def test_response_message_empty_parts(self):
        cm = ChatMessages()
        result = cm._response_message_to_chatml({"role": "user", "content": []})
        assert result == {"role": "user", "content": ""}

    def test_safe_copy_fallback(self):
        # Objects that can't be deepcopied should fallback to str()
        import threading

        lock = threading.Lock()
        result = ChatMessages._safe_copy(lock)
        assert isinstance(result, str)

    def test_extend_non_mapping_raises(self):
        cm = ChatMessages()
        with pytest.raises(TypeError, match="Mapping"):
            cm.extend([{"role": "user", "content": "ok"}, "not a dict"])
