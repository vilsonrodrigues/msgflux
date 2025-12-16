"""Versioning Usage Examples - Real-world patterns for context management.

These tests demonstrate practical usage patterns for the versioning system.
"""

from msgflux.models.state import (
    ModelState,
    LifecycleType,
    Policy,
    ToolCall,
)


class TestExampleMultiAgentConversation:
    """Example: Managing context across multiple agent conversations.

    Scenario: A user interacts with different specialized agents.
    Each agent needs its own context branch while sharing common history.
    """

    def test_multi_agent_branching(self):
        # Main conversation model state
        model_state = ModelState()

        # User starts conversation
        model_state.add_user("I need help with my project")
        model_state.add_assistant("I can help! What kind of project?")
        model_state.add_user("A Python web application")
        model_state.commit("Initial conversation")

        # User needs code help - branch to code agent
        model_state.branch("code_agent")
        model_state.checkout("code_agent")
        model_state.add_assistant(
            "[Code Agent] I can help with Python web development. "
            "What framework are you using?"
        )
        model_state.add_user("FastAPI")
        model_state.add_assistant("[Code Agent] Here's a basic FastAPI structure...")
        model_state.commit("Code agent conversation")

        # User needs deployment help - branch from main
        model_state.checkout("main")
        model_state.branch("devops_agent")
        model_state.checkout("devops_agent")
        model_state.add_assistant(
            "[DevOps Agent] I can help with deployment. "
            "Where do you want to deploy?"
        )
        model_state.add_user("AWS")
        model_state.add_assistant("[DevOps Agent] Here's how to deploy FastAPI to AWS...")
        model_state.commit("DevOps agent conversation")

        # Merge all contexts back to main
        model_state.checkout("main")
        model_state.merge("code_agent", strategy="dedupe")
        model_state.merge("devops_agent", strategy="dedupe")
        model_state.commit("Merged all agent conversations")

        # Now main has the full conversation history
        assert model_state.message_count > 3
        assert "Code Agent" in str([m.text for m in model_state.messages])
        assert "DevOps Agent" in str([m.text for m in model_state.messages])


class TestExampleExperimentTracking:
    """Example: A/B testing different prompt strategies.

    Scenario: Testing different prompting approaches and comparing results.
    """

    def test_prompt_experiment_branches(self):
        model_state = ModelState()

        # Base conversation
        model_state.add_user("Explain quantum computing")
        model_state.commit("Base question")

        # Strategy A: Simple explanation
        model_state.branch("strategy_simple")
        model_state.checkout("strategy_simple")
        model_state.add_assistant(
            "Quantum computing uses quantum bits (qubits) that can be "
            "both 0 and 1 at the same time, unlike regular bits."
        )
        commit_a = model_state.commit("Simple explanation")

        # Strategy B: Detailed explanation
        model_state.checkout("main")
        model_state.branch("strategy_detailed")
        model_state.checkout("strategy_detailed")
        model_state.add_assistant(
            "Quantum computing leverages quantum mechanical phenomena like "
            "superposition and entanglement. In superposition, qubits exist in "
            "multiple states simultaneously, enabling parallel computation. "
            "Entanglement creates correlations between qubits..."
        )
        commit_b = model_state.commit("Detailed explanation")

        # Compare strategies
        diff = model_state._history.diff(commit_a.hash, commit_b.hash)

        # Each strategy added one message
        assert diff["ref1_count"] == 2  # base + simple
        assert diff["ref2_count"] == 2  # base + detailed

        # Can easily switch between strategies
        model_state.checkout("strategy_simple")
        assert "Simple" in model_state.messages[-1].text or "regular bits" in model_state.messages[-1].text


class TestExampleToolLoopRecovery:
    """Example: Recovering from failed tool executions.

    Scenario: Agent makes tool calls, some fail, need to retry from checkpoint.
    """

    def test_tool_loop_with_checkpoints(self):
        model_state = ModelState()

        model_state.add_user("Calculate the total revenue from Q1-Q4 reports")
        model_state.commit("User request")

        # First tool call attempt
        model_state.add_assistant(
            tool_calls=[
                ToolCall(id="call_q1", name="read_report", arguments={"quarter": "Q1"})
            ]
        )
        model_state.add_tool_result("call_q1", "Q1 Revenue: $1,000,000")
        model_state.commit("Q1 read successfully")

        # Q2 fails
        model_state.add_assistant(
            tool_calls=[
                ToolCall(id="call_q2", name="read_report", arguments={"quarter": "Q2"})
            ]
        )
        model_state.add_tool_result("call_q2", "ERROR: File not found", is_error=True)

        # Don't commit the error - revert instead
        model_state.revert(1)  # Go back to Q1 checkpoint

        # Try alternative approach
        model_state.add_assistant(
            tool_calls=[
                ToolCall(
                    id="call_q2_alt",
                    name="search_documents",
                    arguments={"query": "Q2 revenue"},
                )
            ]
        )
        model_state.add_tool_result("call_q2_alt", "Q2 Revenue: $1,200,000")
        model_state.commit("Q2 found via search")

        # Continue with Q3, Q4...
        assert model_state.message_count >= 4
        # Error was reverted, so no error messages in final context
        error_msgs = [m for m in model_state.messages if m.role.value == "tool" and m.tool_result.is_error]
        assert len(error_msgs) == 0


class TestExampleContextCompaction:
    """Example: Managing long-running conversations with compaction.

    Scenario: A support conversation that spans many turns needs compression.
    """

    def test_long_conversation_compaction(self):
        def simple_summarizer(messages):
            """Simple summarizer that concatenates key points."""
            texts = [m.text for m in messages if m.text]
            return f"[Summary of {len(messages)} messages]: Discussion covered {', '.join(texts[:2])}..."

        model_state = ModelState(
            policy=Policy(
                type="sliding_window",
                max_messages=5,
                summarize_threshold=2,
            ),
            summarizer=simple_summarizer,
        )

        # Simulate long conversation
        messages = [
            ("user", "I can't login"),
            ("assistant", "Have you tried resetting your password?"),
            ("user", "Yes, still doesn't work"),
            ("assistant", "Let me check your account status"),
            ("user", "Thanks"),
            ("assistant", "I see the issue - your account was locked"),
            ("user", "Can you unlock it?"),
            ("assistant", "Done! Try logging in now"),
            ("user", "It works! Thank you!"),
            ("assistant", "You're welcome! Anything else?"),
        ]

        for role, content in messages:
            if role == "user":
                model_state.add_user(content)
            else:
                model_state.add_assistant(content)

        model_state.commit("Full conversation")

        # Conversation exceeds limit
        assert model_state.needs_compaction()

        # Compact
        result = model_state.compact()

        # After compaction, we have fewer messages than before
        # (summary message + preserved messages)
        assert model_state.message_count < 10
        assert result.stats.get("compacted") is True


class TestExampleEphemeralToolContext:
    """Example: Using ephemeral messages for tool execution context.

    Scenario: Tool results are only needed during execution, not for long-term context.
    """

    def test_ephemeral_tool_execution(self):
        model_state = ModelState()

        model_state.add_user("What's the weather in Paris and Tokyo?")

        # Tool calls with ephemeral results
        with model_state.scope("weather_check"):
            model_state.add_assistant(
                tool_calls=[
                    ToolCall(id="call_paris", name="get_weather", arguments={"city": "Paris"}),
                    ToolCall(id="call_tokyo", name="get_weather", arguments={"city": "Tokyo"}),
                ],
                lifecycle=LifecycleType.EPHEMERAL_SCOPE,
            )

            model_state.add_tool_result(
                "call_paris",
                "Paris: 22C, Sunny",
                lifecycle=LifecycleType.EPHEMERAL_SCOPE,
            )
            model_state.add_tool_result(
                "call_tokyo",
                "Tokyo: 28C, Cloudy",
                lifecycle=LifecycleType.EPHEMERAL_SCOPE,
            )

        # Assistant summarizes (permanent)
        model_state.add_assistant(
            "The weather is:\n- Paris: 22C and sunny\n- Tokyo: 28C and cloudy"
        )

        # Check scope messages
        scope_msgs = [m for m in model_state.messages if m.metadata.scope_id == "weather_check"]
        assert len(scope_msgs) == 3  # 1 assistant + 2 tool results

        # Apply lifecycle policy to clean up
        model_state.set_policy("lifecycle")
        result = model_state.compact()

        # After compaction, ephemeral scope messages removed
        # (since we're no longer in that scope)
        ephemeral = model_state.get_by_lifecycle(LifecycleType.EPHEMERAL_SCOPE)
        assert len(ephemeral) == 0


class TestExampleTimeTravelDebug:
    """Example: Debugging by examining conversation history.

    Scenario: Something went wrong, need to understand the conversation flow.
    """

    def test_time_travel_debugging(self):
        model_state = ModelState()

        # Build up conversation with commits
        model_state.add_user("Create a function to sort a list")
        model_state.commit("User request")

        model_state.add_assistant("```python\ndef sort_list(lst):\n    return sorted(lst)```")
        model_state.commit("First implementation")

        model_state.add_user("It should sort in descending order")
        model_state.commit("User clarification")

        model_state.add_assistant("```python\ndef sort_list(lst):\n    return sorted(lst, reverse=True)```")
        model_state.commit("Fixed implementation")

        model_state.add_user("Actually, I need custom comparator support")
        model_state.commit("Another change request")

        # Debug: What was the conversation state after first implementation?
        history = model_state.log(limit=10)

        # Find the commit after "First implementation"
        first_impl_commit = None
        for commit in history:
            if commit.commit_message == "First implementation":
                first_impl_commit = commit
                break

        assert first_impl_commit is not None

        # Look at the state at that point
        messages_at_that_point = model_state._history.get_messages(first_impl_commit)
        assert len(messages_at_that_point) == 2
        assert "sorted(lst)" in messages_at_that_point[1].text

        # Compare current state vs first implementation
        diff = model_state.diff("HEAD", first_impl_commit.hash)
        assert diff["ref1_count"] > diff["ref2_count"]


class TestExampleConversationFork:
    """Example: Exploring different conversation paths.

    Scenario: User wants to explore "what if" scenarios.
    """

    def test_conversation_what_if(self):
        model_state = ModelState()

        model_state.add_user("Should I learn Python or JavaScript?")
        model_state.add_assistant("Both are great! What's your goal?")
        model_state.add_user("I want to build web applications")
        model_state.commit("Base decision point")

        # Path A: Python
        model_state.branch("python_path")
        model_state.checkout("python_path")
        model_state.add_assistant(
            "For Python web apps, I recommend learning Django or FastAPI. "
            "Python is great for backend development and data processing."
        )
        model_state.add_user("What about the frontend?")
        model_state.add_assistant(
            "You can use templates or connect to a JavaScript frontend. "
            "Many Python developers use React or Vue for the frontend."
        )
        model_state.commit("Python path explored")

        # Path B: JavaScript
        model_state.checkout("main")
        model_state.branch("javascript_path")
        model_state.checkout("javascript_path")
        model_state.add_assistant(
            "For JavaScript web apps, you can use frameworks like React, "
            "Next.js, or Node.js for full-stack development."
        )
        model_state.add_user("Can I do everything in JavaScript?")
        model_state.add_assistant(
            "Yes! Full-stack JavaScript means using Node.js for backend "
            "and React/Vue for frontend. Same language everywhere!"
        )
        model_state.commit("JavaScript path explored")

        # Compare conversation lengths
        model_state.checkout("python_path")
        python_msgs = model_state.message_count

        model_state.checkout("javascript_path")
        js_msgs = model_state.message_count

        assert python_msgs == js_msgs  # Both explored equally

        # User can continue from either path
        model_state.add_user("I think I'll go with JavaScript!")
        assert model_state.message_count == js_msgs + 1
