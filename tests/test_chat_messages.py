from msgflux.chat_messages import ChatMessages
from msgflux.context import session_context


def test_chat_messages_inherit_session_context():
    with session_context(session_id="user_42", namespace="support"):
        chat = ChatMessages()

    assert chat.session_id == "user_42"
    assert chat.namespace == "support"


def test_chat_messages_state_roundtrip():
    chat = ChatMessages(session_id="session_1", namespace="agent:test")
    turn_id = chat.begin_turn(inputs="What is 2+2?", vars={"temperature": 0.2})
    chat.add_user("What is 2+2?")
    chat.add_assistant_response("4", reasoning_content="Simple arithmetic.")
    chat.end_turn(assistant_output="4", response_type="text_generation")

    state = chat._to_state()

    restored = ChatMessages()
    restored._hydrate_state(state)

    assert restored.session_id == "session_1"
    assert restored.namespace == "agent:test"
    assert restored.turns[0]["turn_id"] == turn_id
    assert restored.to_chatml()[0]["content"] == "What is 2+2?"
    assert restored.to_chatml()[-1]["content"] == "4"
