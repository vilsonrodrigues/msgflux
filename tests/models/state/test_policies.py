"""Tests for compaction policies."""

import pytest

from msgflux.models.state.policies.importance import ImportancePolicy
from msgflux.models.state.policies.lifecycle import LifecyclePolicy
from msgflux.models.state.policies.position import PositionBasedPolicy
from msgflux.models.state.policies.types import Policy
from msgflux.models.state.types import (
    ChatMessage,
    LifecycleType,
    Role,
    TextContent,
    assistant_message,
    system_message,
    user_message,
)


# Helper functions for creating test messages


def create_messages(count: int, role: Role = Role.USER) -> list[ChatMessage]:
    """Create a list of test messages."""
    messages = []
    for i in range(count):
        if role == Role.USER:
            msg = user_message(f"User message {i}", index=i)
        elif role == Role.ASSISTANT:
            msg = assistant_message(f"Assistant message {i}", index=i)
        else:
            msg = system_message(f"System message {i}", index=i)
        messages.append(msg)
    return messages


def create_messages_with_importance(
    importances: list[float],
) -> list[ChatMessage]:
    """Create messages with specific importance scores."""
    messages = []
    for i, importance in enumerate(importances):
        msg = user_message(f"Message {i}", index=i, importance=importance)
        messages.append(msg)
    return messages


def create_messages_with_lifecycle(
    lifecycles: list[LifecycleType],
) -> list[ChatMessage]:
    """Create messages with specific lifecycle types."""
    messages = []
    for i, lifecycle in enumerate(lifecycles):
        msg = user_message(f"Message {i}", index=i, lifecycle=lifecycle)
        messages.append(msg)
    return messages


class TestImportancePolicy:
    """Tests for ImportancePolicy."""

    def test_no_max_messages_no_compaction(self):
        """Test that policy doesn't compact when max_messages is not set."""
        policy = ImportancePolicy(config=Policy(max_messages=None))
        messages = create_messages(10)

        result = policy.apply(messages)

        assert len(result.messages) == 10
        assert result.stats["compacted"] is False
        assert len(result.removed) == 0

    def test_under_limit_no_compaction(self):
        """Test that policy doesn't compact when under the limit."""
        policy = ImportancePolicy(config=Policy(max_messages=10))
        messages = create_messages(5)

        result = policy.apply(messages)

        assert len(result.messages) == 5
        assert result.stats["compacted"] is False
        assert len(result.removed) == 0

    def test_at_limit_no_compaction(self):
        """Test that policy doesn't compact when exactly at the limit."""
        policy = ImportancePolicy(config=Policy(max_messages=5))
        messages = create_messages(5)

        result = policy.apply(messages)

        assert len(result.messages) == 5
        assert result.stats["compacted"] is False
        assert len(result.removed) == 0

    def test_compact_by_importance(self):
        """Test compacting messages based on importance scores."""
        # Set min_importance very high (1.5) so no messages are auto-protected by importance
        policy = ImportancePolicy(
            config=Policy(max_messages=3, min_importance=1.5, preserve_last_n=0)
        )
        # Create messages with varying importance
        messages = create_messages_with_importance([0.1, 0.5, 0.9, 0.3, 0.7])

        result = policy.apply(messages)

        assert len(result.messages) == 3
        assert result.stats["compacted"] is True
        assert result.stats["removed_count"] == 2
        # Messages with lowest importance (0.1, 0.3) should be removed
        remaining_texts = [msg.text for msg in result.messages]
        assert "Message 0" not in remaining_texts  # importance 0.1
        assert "Message 3" not in remaining_texts  # importance 0.3

    def test_preserve_system_messages(self):
        """Test that system messages are preserved."""
        policy = ImportancePolicy(
            config=Policy(
                max_messages=3,
                preserve_system=True,
                min_importance=1.5,  # High value so importance doesn't protect messages
                preserve_last_n=0,
            )
        )
        messages = [
            system_message("System", index=0, importance=0.1),
            user_message("User 1", index=1, importance=0.2),
            user_message("User 2", index=2, importance=0.3),
            user_message("User 3", index=3, importance=0.4),
        ]

        result = policy.apply(messages)

        assert len(result.messages) == 3
        assert result.messages[0].role == Role.SYSTEM
        assert result.messages[0].text == "System"

    def test_preserve_first_n_messages(self):
        """Test preserving the first N messages."""
        policy = ImportancePolicy(
            config=Policy(
                max_messages=5,
                preserve_first_n=2,
                min_importance=1.5,  # High value so importance doesn't protect messages
                preserve_last_n=0,
            )
        )
        messages = create_messages_with_importance([0.1, 0.2, 0.9, 0.8, 0.7, 0.6, 0.5])

        result = policy.apply(messages)

        assert len(result.messages) == 5
        # First 2 messages should be preserved regardless of importance
        assert result.messages[0].text == "Message 0"  # importance 0.1
        assert result.messages[1].text == "Message 1"  # importance 0.2

    def test_preserve_last_n_messages(self):
        """Test preserving the last N messages."""
        policy = ImportancePolicy(
            config=Policy(max_messages=5, preserve_last_n=2)
        )
        messages = create_messages_with_importance([0.9, 0.8, 0.7, 0.6, 0.1, 0.2])

        result = policy.apply(messages)

        assert len(result.messages) == 5
        # Last 2 messages should be preserved regardless of importance
        result_texts = [msg.text for msg in result.messages]
        assert "Message 4" in result_texts  # importance 0.1
        assert "Message 5" in result_texts  # importance 0.2

    def test_preserve_high_importance_messages(self):
        """Test preserving messages above min_importance threshold."""
        policy = ImportancePolicy(
            config=Policy(max_messages=3, min_importance=0.9, preserve_last_n=0)
        )
        messages = create_messages_with_importance([0.5, 0.95, 0.3, 0.92, 0.1])

        result = policy.apply(messages)

        # Messages 1 (0.95) and 3 (0.92) should be protected by min_importance
        # So we'll keep those 2 + 1 more = 3 total
        assert len(result.messages) == 3
        result_texts = [msg.text for msg in result.messages]
        assert "Message 1" in result_texts
        assert "Message 3" in result_texts

    def test_needs_compaction_by_message_count(self):
        """Test needs_compaction based on message count."""
        policy = ImportancePolicy(config=Policy(max_messages=5))
        messages = create_messages(3)

        assert not policy.needs_compaction(messages, token_count=0)

        messages = create_messages(7)
        assert policy.needs_compaction(messages, token_count=0)

    def test_needs_compaction_by_token_count(self):
        """Test needs_compaction based on token count."""
        policy = ImportancePolicy(config=Policy(max_tokens=1000))
        messages = create_messages(5)

        assert not policy.needs_compaction(messages, token_count=500)
        assert policy.needs_compaction(messages, token_count=1500)

    def test_message_indices_updated(self):
        """Test that message indices are properly updated after compaction."""
        policy = ImportancePolicy(config=Policy(max_messages=3))
        messages = create_messages_with_importance([0.1, 0.5, 0.9, 0.3, 0.7])

        result = policy.apply(messages)

        # Indices should be sequential starting from 0
        for i, msg in enumerate(result.messages):
            assert msg.index == i

    def test_empty_messages(self):
        """Test handling of empty message list."""
        policy = ImportancePolicy(config=Policy(max_messages=5))
        messages = []

        result = policy.apply(messages)

        assert len(result.messages) == 0
        assert result.stats["compacted"] is False

    def test_protect_all_candidates(self):
        """Test when all candidates are protected and we still need to compact."""
        policy = ImportancePolicy(
            config=Policy(max_messages=2, preserve_first_n=5)
        )
        messages = create_messages(5)

        result = policy.apply(messages)

        # All messages are protected, so no removal even though we exceeded max
        assert len(result.messages) == 5
        assert result.stats["compacted"] is False

    def test_stats_include_protected_count(self):
        """Test that stats include protected message count."""
        policy = ImportancePolicy(
            config=Policy(max_messages=5, preserve_first_n=2, preserve_last_n=2)
        )
        messages = create_messages(10)

        result = policy.apply(messages)

        assert "protected_count" in result.stats
        assert result.stats["protected_count"] >= 4  # At least first 2 + last 2

    def test_preserve_first_percentage_protection(self):
        """Test that preserve_first_pct protects messages."""
        policy = ImportancePolicy(
            config=Policy(
                max_messages=3,
                preserve_first_pct=0.4,  # First 40% protected (4 out of 10)
                min_importance=1.5,
                preserve_last_n=0,
            )
        )
        messages = create_messages_with_importance([0.1] * 10)

        result = policy.apply(messages)

        # First 4 messages (40% of 10) should be protected
        # So we keep those 4, but max is 3, so all are protected and no compaction
        assert len(result.messages) >= 3

    def test_preserve_last_percentage_protection(self):
        """Test that preserve_last_pct protects messages."""
        policy = ImportancePolicy(
            config=Policy(
                max_messages=3,
                preserve_last_pct=0.3,  # Last 30% protected (3 out of 10)
                min_importance=1.5,
                preserve_first_n=0,
            )
        )
        messages = create_messages_with_importance([0.1] * 10)

        result = policy.apply(messages)

        # Last 3 messages (30% of 10) should be protected
        # With max_messages=3, we should keep exactly 3 (the last ones)
        assert len(result.messages) == 3
        # Verify the last messages are kept
        assert result.messages[-1].text == "Message 9"
        assert result.messages[-2].text == "Message 8"
        assert result.messages[-3].text == "Message 7"

    def test_preserve_both_percentages(self):
        """Test using both first and last percentage protections."""
        policy = ImportancePolicy(
            config=Policy(
                max_messages=5,
                preserve_first_pct=0.2,  # First 20% (2 out of 10)
                preserve_last_pct=0.2,   # Last 20% (2 out of 10)
                min_importance=1.5,
            )
        )
        messages = create_messages_with_importance([0.1] * 10)

        result = policy.apply(messages)

        # First 2 + last 2 = 4 protected, need to keep 5 total
        assert len(result.messages) == 5


class TestPositionBasedPolicy:
    """Tests for PositionBasedPolicy."""

    def test_empty_messages(self):
        """Test handling of empty message list."""
        policy = PositionBasedPolicy(config=Policy())
        messages = []

        result = policy.apply(messages)

        assert len(result.messages) == 0
        assert result.messages == []

    def test_preserve_first_percentage(self):
        """Test preserving the first X% of messages."""
        policy = PositionBasedPolicy(
            config=Policy(preserve_first_pct=0.2, preserve_last_pct=0.0)
        )
        messages = create_messages(10)

        result = policy.apply(messages)

        # First 20% (2 messages) should be in first region
        assert result.stats["first_kept"] == 2

    def test_preserve_last_percentage(self):
        """Test preserving the last X% of messages."""
        policy = PositionBasedPolicy(
            config=Policy(preserve_first_pct=0.0, preserve_last_pct=0.3)
        )
        messages = create_messages(10)

        result = policy.apply(messages)

        # Last 30% (3 messages) should be in last region
        assert result.stats["last_kept"] == 3

    def test_preserve_first_n_overrides_percentage(self):
        """Test that preserve_first_n takes precedence over percentage."""
        policy = PositionBasedPolicy(
            config=Policy(preserve_first_n=5, preserve_first_pct=0.1)
        )
        messages = create_messages(10)

        result = policy.apply(messages)

        # preserve_first_n=5 should override preserve_first_pct=0.1 (which would be 1)
        assert result.stats["first_kept"] >= 5

    def test_preserve_last_n_overrides_percentage(self):
        """Test that preserve_last_n takes precedence over percentage."""
        policy = PositionBasedPolicy(
            config=Policy(preserve_last_n=4, preserve_last_pct=0.1)
        )
        messages = create_messages(10)

        result = policy.apply(messages)

        # preserve_last_n=4 should override preserve_last_pct=0.1 (which would be 1)
        assert result.stats["last_kept"] >= 4

    def test_no_middle_region_when_boundaries_overlap(self):
        """Test when first and last regions overlap."""
        policy = PositionBasedPolicy(
            config=Policy(preserve_first_pct=0.6, preserve_last_pct=0.6)
        )
        messages = create_messages(10)

        result = policy.apply(messages)

        # Boundaries overlap, so no compaction
        assert len(result.messages) == 10
        assert result.stats["compacted"] is False

    def test_summarize_middle_region(self):
        """Test summarizing the middle region."""

        def simple_summarizer(msgs: list[ChatMessage]) -> str:
            return f"Summary of {len(msgs)} messages"

        policy = PositionBasedPolicy(
            config=Policy(
                preserve_first_pct=0.2,
                preserve_last_pct=0.2,
                summarize_threshold=3,
            ),
            summarizer=simple_summarizer,
        )
        messages = create_messages(10)

        result = policy.apply(messages)

        # Should have first region + summary message + last region
        assert result.stats["compacted"] is True
        assert len(result.summarized) > 0
        assert result.summary == f"Summary of {len(result.summarized)} messages"

        # Check that summary message was inserted
        summary_found = False
        for msg in result.messages:
            if msg.role == Role.ASSISTANT and "[Context Summary]" in (msg.text or ""):
                summary_found = True
                break
        assert summary_found

    def test_no_summarization_below_threshold(self):
        """Test that middle region is not summarized below threshold."""

        def simple_summarizer(msgs: list[ChatMessage]) -> str:
            return f"Summary of {len(msgs)} messages"

        policy = PositionBasedPolicy(
            config=Policy(
                preserve_first_pct=0.3,
                preserve_last_pct=0.3,
                summarize_threshold=10,  # High threshold
            ),
            summarizer=simple_summarizer,
        )
        messages = create_messages(10)

        result = policy.apply(messages)

        # Middle region too small to summarize
        assert result.stats["compacted"] is False
        assert len(result.summarized) == 0
        assert result.summary is None

    def test_no_summarization_without_summarizer(self):
        """Test that middle region is not summarized without a summarizer."""
        policy = PositionBasedPolicy(
            config=Policy(
                preserve_first_pct=0.2,
                preserve_last_pct=0.2,
                summarize_threshold=1,
            ),
            summarizer=None,
        )
        messages = create_messages(10)

        result = policy.apply(messages)

        # No summarizer provided, so no summarization
        assert len(result.summarized) == 0
        assert result.summary is None

    def test_message_indices_updated(self):
        """Test that message indices are properly updated."""
        policy = PositionBasedPolicy(
            config=Policy(preserve_first_pct=0.2, preserve_last_pct=0.2)
        )
        messages = create_messages(10)

        result = policy.apply(messages)

        # Indices should be sequential
        for i, msg in enumerate(result.messages):
            assert msg.index == i

    def test_single_message(self):
        """Test handling of a single message."""
        policy = PositionBasedPolicy(
            config=Policy(preserve_first_pct=0.5, preserve_last_pct=0.5)
        )
        messages = create_messages(1)

        result = policy.apply(messages)

        assert len(result.messages) == 1
        assert result.stats["compacted"] is False

    def test_all_messages_in_first_region(self):
        """Test when all messages fall into first region."""
        policy = PositionBasedPolicy(
            config=Policy(preserve_first_pct=1.0, preserve_last_pct=0.0, preserve_last_n=0)
        )
        messages = create_messages(10)

        result = policy.apply(messages)

        assert len(result.messages) == 10
        assert result.stats["compacted"] is False

    def test_all_messages_in_last_region(self):
        """Test when all messages fall into last region."""
        policy = PositionBasedPolicy(
            config=Policy(preserve_first_pct=0.0, preserve_last_pct=1.0, preserve_first_n=0)
        )
        messages = create_messages(10)

        result = policy.apply(messages)

        assert len(result.messages) == 10
        assert result.stats["compacted"] is False

    def test_stats_accuracy(self):
        """Test that stats accurately reflect regions."""
        policy = PositionBasedPolicy(
            config=Policy(preserve_first_n=2, preserve_last_n=3)
        )
        messages = create_messages(10)

        result = policy.apply(messages)

        first_kept = result.stats["first_kept"]
        middle = result.stats["middle_summarized"]
        last_kept = result.stats["last_kept"]

        # Account for possible summary message
        expected_total = first_kept + last_kept
        if middle > 0:
            expected_total += 1  # Summary message
        else:
            expected_total += (10 - 2 - 3)  # Middle messages kept as-is

        assert len(result.messages) == expected_total


class TestLifecyclePolicy:
    """Tests for LifecyclePolicy."""

    def test_permanent_messages_always_kept(self):
        """Test that permanent messages are always kept."""
        policy = LifecyclePolicy(config=Policy())
        messages = [
            user_message("Msg 1", index=0, lifecycle=LifecycleType.PERMANENT),
            user_message("Msg 2", index=1, lifecycle=LifecycleType.PERMANENT),
            user_message("Msg 3", index=2, lifecycle=LifecycleType.PERMANENT),
        ]

        result = policy.apply(messages)

        assert len(result.messages) == 3
        assert len(result.removed) == 0

    def test_remove_expired_ephemeral_turns(self):
        """Test removing ephemeral messages with expired TTL."""
        policy = LifecyclePolicy(config=Policy(), current_turn=5)
        messages = [
            user_message("Permanent", index=0, lifecycle=LifecycleType.PERMANENT),
            user_message(
                "Ephemeral expired",
                index=1,
                lifecycle=LifecycleType.EPHEMERAL_TURNS,
                ttl_turns=0,
            ),
            user_message(
                "Ephemeral active",
                index=2,
                lifecycle=LifecycleType.EPHEMERAL_TURNS,
                ttl_turns=3,
            ),
        ]

        result = policy.apply(messages)

        assert len(result.messages) == 2
        assert len(result.removed) == 1
        assert result.removed[0].text == "Ephemeral expired"

    def test_keep_active_ephemeral_turns(self):
        """Test keeping ephemeral messages with active TTL."""
        policy = LifecyclePolicy(config=Policy())
        messages = [
            user_message(
                "Ephemeral 1",
                index=0,
                lifecycle=LifecycleType.EPHEMERAL_TURNS,
                ttl_turns=5,
            ),
            user_message(
                "Ephemeral 2",
                index=1,
                lifecycle=LifecycleType.EPHEMERAL_TURNS,
                ttl_turns=1,
            ),
        ]

        result = policy.apply(messages)

        assert len(result.messages) == 2
        assert len(result.removed) == 0

    def test_remove_out_of_scope_messages(self):
        """Test removing ephemeral_scope messages outside current scope."""
        policy = LifecyclePolicy(config=Policy(), current_scope="scope_A")
        messages = [
            user_message(
                "In scope",
                index=0,
                lifecycle=LifecycleType.EPHEMERAL_SCOPE,
                scope_id="scope_A",
            ),
            user_message(
                "Out of scope",
                index=1,
                lifecycle=LifecycleType.EPHEMERAL_SCOPE,
                scope_id="scope_B",
            ),
            user_message("Permanent", index=2, lifecycle=LifecycleType.PERMANENT),
        ]

        result = policy.apply(messages)

        assert len(result.messages) == 2
        assert len(result.removed) == 1
        assert result.removed[0].text == "Out of scope"

    def test_keep_in_scope_messages(self):
        """Test keeping ephemeral_scope messages within current scope."""
        policy = LifecyclePolicy(config=Policy(), current_scope="scope_A")
        messages = [
            user_message(
                "In scope 1",
                index=0,
                lifecycle=LifecycleType.EPHEMERAL_SCOPE,
                scope_id="scope_A",
            ),
            user_message(
                "In scope 2",
                index=1,
                lifecycle=LifecycleType.EPHEMERAL_SCOPE,
                scope_id="scope_A",
            ),
        ]

        result = policy.apply(messages)

        assert len(result.messages) == 2
        assert len(result.removed) == 0

    def test_summarizable_messages(self):
        """Test summarizing messages marked as summarizable."""

        def simple_summarizer(msgs: list[ChatMessage]) -> str:
            return f"Summary of {len(msgs)} summarizable messages"

        policy = LifecyclePolicy(
            config=Policy(summarize_threshold=2), summarizer=simple_summarizer
        )
        messages = [
            user_message("Permanent", index=0, lifecycle=LifecycleType.PERMANENT),
            user_message("Sum 1", index=1, lifecycle=LifecycleType.SUMMARIZABLE),
            user_message("Sum 2", index=2, lifecycle=LifecycleType.SUMMARIZABLE),
            user_message("Sum 3", index=3, lifecycle=LifecycleType.SUMMARIZABLE),
        ]

        result = policy.apply(messages)

        # 1 permanent + 1 summary message = 2
        assert len(result.messages) == 2
        assert len(result.summarized) == 3
        assert result.summary == "Summary of 3 summarizable messages"

        # Check that summary message was inserted
        summary_found = False
        for msg in result.messages:
            if msg.role == Role.ASSISTANT and "[Summary]" in (msg.text or ""):
                summary_found = True
                break
        assert summary_found

    def test_no_summarization_below_threshold(self):
        """Test that summarizable messages are not summarized below threshold."""

        def simple_summarizer(msgs: list[ChatMessage]) -> str:
            return f"Summary of {len(msgs)} summarizable messages"

        policy = LifecyclePolicy(
            config=Policy(summarize_threshold=5), summarizer=simple_summarizer
        )
        messages = [
            user_message("Sum 1", index=0, lifecycle=LifecycleType.SUMMARIZABLE),
            user_message("Sum 2", index=1, lifecycle=LifecycleType.SUMMARIZABLE),
        ]

        result = policy.apply(messages)

        # Below threshold, so messages are kept as-is
        assert len(result.messages) == 2
        assert len(result.summarized) == 0
        assert result.summary is None

    def test_no_summarization_without_summarizer(self):
        """Test that summarizable messages are kept without a summarizer."""
        policy = LifecyclePolicy(config=Policy(summarize_threshold=1), summarizer=None)
        messages = [
            user_message("Sum 1", index=0, lifecycle=LifecycleType.SUMMARIZABLE),
            user_message("Sum 2", index=1, lifecycle=LifecycleType.SUMMARIZABLE),
        ]

        result = policy.apply(messages)

        # No summarizer, so messages are kept as-is
        assert len(result.messages) == 2
        assert len(result.summarized) == 0

    def test_advance_turn(self):
        """Test advancing the turn counter."""
        policy = LifecyclePolicy(current_turn=0)

        assert policy.current_turn == 0
        policy.advance_turn()
        assert policy.current_turn == 1
        policy.advance_turn()
        assert policy.current_turn == 2

    def test_set_scope(self):
        """Test setting the current scope."""
        policy = LifecyclePolicy(current_scope=None)

        assert policy.current_scope is None
        policy.set_scope("scope_A")
        assert policy.current_scope == "scope_A"
        policy.set_scope("scope_B")
        assert policy.current_scope == "scope_B"
        policy.set_scope(None)
        assert policy.current_scope is None

    def test_decrement_ttls(self):
        """Test decrementing TTL on ephemeral_turns messages."""
        policy = LifecyclePolicy()
        messages = [
            user_message(
                "Ephemeral 1",
                index=0,
                lifecycle=LifecycleType.EPHEMERAL_TURNS,
                ttl_turns=5,
            ),
            user_message(
                "Ephemeral 2",
                index=1,
                lifecycle=LifecycleType.EPHEMERAL_TURNS,
                ttl_turns=1,
            ),
            user_message("Permanent", index=2, lifecycle=LifecycleType.PERMANENT),
        ]

        result = policy.decrement_ttls(messages)

        assert result[0].metadata.ttl_turns == 4
        assert result[1].metadata.ttl_turns == 0
        assert result[2].metadata.ttl_turns is None

    def test_decrement_ttls_does_not_go_negative(self):
        """Test that TTL doesn't go below 0."""
        policy = LifecyclePolicy()
        messages = [
            user_message(
                "Ephemeral",
                index=0,
                lifecycle=LifecycleType.EPHEMERAL_TURNS,
                ttl_turns=0,
            ),
        ]

        result = policy.decrement_ttls(messages)

        # TTL should stay at 0, not go negative
        assert result[0].metadata.ttl_turns == 0

    def test_message_indices_updated(self):
        """Test that message indices are properly updated."""
        policy = LifecyclePolicy(config=Policy())
        messages = [
            user_message("Msg 1", index=0, lifecycle=LifecycleType.PERMANENT),
            user_message(
                "Expired", index=1, lifecycle=LifecycleType.EPHEMERAL_TURNS, ttl_turns=0
            ),
            user_message("Msg 2", index=2, lifecycle=LifecycleType.PERMANENT),
        ]

        result = policy.apply(messages)

        # Indices should be sequential after removal
        for i, msg in enumerate(result.messages):
            assert msg.index == i

    def test_stats_accuracy(self):
        """Test that stats accurately reflect policy results."""
        policy = LifecyclePolicy(config=Policy(), current_scope="scope_A")
        messages = [
            user_message("Permanent 1", index=0, lifecycle=LifecycleType.PERMANENT),
            user_message(
                "Expired", index=1, lifecycle=LifecycleType.EPHEMERAL_TURNS, ttl_turns=0
            ),
            user_message(
                "Out of scope",
                index=2,
                lifecycle=LifecycleType.EPHEMERAL_SCOPE,
                scope_id="scope_B",
            ),
            user_message("Permanent 2", index=3, lifecycle=LifecycleType.PERMANENT),
        ]

        result = policy.apply(messages)

        assert result.stats["removed_count"] == 2
        assert result.stats["kept_count"] == 2
        assert result.stats["summarized_count"] == 0

    def test_mixed_lifecycle_types(self):
        """Test handling messages with mixed lifecycle types."""
        policy = LifecyclePolicy(config=Policy(), current_turn=10, current_scope="A")
        messages = [
            user_message("Permanent", index=0, lifecycle=LifecycleType.PERMANENT),
            user_message(
                "Ephemeral turns OK",
                index=1,
                lifecycle=LifecycleType.EPHEMERAL_TURNS,
                ttl_turns=5,
            ),
            user_message(
                "Ephemeral turns expired",
                index=2,
                lifecycle=LifecycleType.EPHEMERAL_TURNS,
                ttl_turns=0,
            ),
            user_message(
                "Ephemeral scope OK",
                index=3,
                lifecycle=LifecycleType.EPHEMERAL_SCOPE,
                scope_id="A",
            ),
            user_message(
                "Ephemeral scope wrong",
                index=4,
                lifecycle=LifecycleType.EPHEMERAL_SCOPE,
                scope_id="B",
            ),
            user_message("Summarizable", index=5, lifecycle=LifecycleType.SUMMARIZABLE),
        ]

        result = policy.apply(messages)

        # Should remove: expired turns (index 2), wrong scope (index 4)
        # Should keep: permanent, ephemeral turns OK, ephemeral scope OK, summarizable
        assert len(result.messages) == 4
        assert len(result.removed) == 2

        removed_texts = [msg.text for msg in result.removed]
        assert "Ephemeral turns expired" in removed_texts
        assert "Ephemeral scope wrong" in removed_texts

    def test_empty_messages(self):
        """Test handling of empty message list."""
        policy = LifecyclePolicy(config=Policy())
        messages = []

        result = policy.apply(messages)

        assert len(result.messages) == 0
        assert len(result.removed) == 0
        assert result.stats["removed_count"] == 0
        assert result.stats["kept_count"] == 0

    def test_none_scope_id_handling(self):
        """Test handling of messages with None scope_id."""
        policy = LifecyclePolicy(config=Policy(), current_scope="scope_A")
        messages = [
            user_message(
                "No scope",
                index=0,
                lifecycle=LifecycleType.EPHEMERAL_SCOPE,
                scope_id=None,
            ),
        ]

        result = policy.apply(messages)

        # Message with None scope_id should be removed when current_scope is "scope_A"
        assert len(result.removed) == 1

    def test_none_ttl_handling(self):
        """Test handling of ephemeral_turns messages with None TTL."""
        policy = LifecyclePolicy(config=Policy())
        messages = [
            user_message(
                "No TTL",
                index=0,
                lifecycle=LifecycleType.EPHEMERAL_TURNS,
                ttl_turns=None,
            ),
        ]

        result = policy.apply(messages)

        # Message with None ttl_turns should not be removed
        assert len(result.messages) == 1
        assert len(result.removed) == 0


class TestPolicyIntegration:
    """Integration tests across multiple policies."""

    def test_importance_policy_with_mixed_roles(self):
        """Test ImportancePolicy with different message roles."""
        policy = ImportancePolicy(
            config=Policy(
                max_messages=3,
                preserve_system=True,
                min_importance=1.5,  # High value so importance doesn't protect messages
                preserve_last_n=0,
            )
        )
        messages = [
            system_message("System", index=0),
            user_message("User 1", index=1, importance=0.3),
            assistant_message("Assistant 1", index=2, importance=0.8),
            user_message("User 2", index=3, importance=0.5),
        ]

        result = policy.apply(messages)

        assert len(result.messages) == 3
        assert result.messages[0].role == Role.SYSTEM

    def test_position_policy_preserves_order(self):
        """Test that PositionBasedPolicy preserves message order."""
        policy = PositionBasedPolicy(
            config=Policy(preserve_first_n=2, preserve_last_n=2)
        )
        messages = [
            user_message("First", index=0),
            assistant_message("Second", index=1),
            user_message("Middle 1", index=2),
            assistant_message("Middle 2", index=3),
            user_message("Second Last", index=4),
            assistant_message("Last", index=5),
        ]

        result = policy.apply(messages)

        # Order should be maintained
        assert result.messages[0].text == "First"
        assert result.messages[1].text == "Second"
        # Middle messages may or may not be present depending on summarization
        # But last messages should be at the end
        assert result.messages[-2].text == "Second Last"
        assert result.messages[-1].text == "Last"

    def test_lifecycle_policy_complex_scenario(self):
        """Test LifecyclePolicy with a complex real-world scenario."""

        def summarizer(msgs: list[ChatMessage]) -> str:
            return f"Summarized {len(msgs)} tool interactions"

        policy = LifecyclePolicy(
            config=Policy(summarize_threshold=2),
            summarizer=summarizer,
            current_turn=5,
            current_scope="main_loop",
        )

        messages = [
            system_message("You are a helpful assistant", index=0),
            user_message("What's the weather?", index=1),
            assistant_message(
                "Checking...",
                index=2,
                lifecycle=LifecycleType.SUMMARIZABLE,
            ),
            assistant_message(
                "Tool call result",
                index=3,
                lifecycle=LifecycleType.SUMMARIZABLE,
            ),
            assistant_message(
                "Tool call result 2",
                index=4,
                lifecycle=LifecycleType.SUMMARIZABLE,
            ),
            assistant_message("The weather is sunny!", index=5),
            user_message(
                "Temp context",
                index=6,
                lifecycle=LifecycleType.EPHEMERAL_TURNS,
                ttl_turns=0,
            ),
        ]

        result = policy.apply(messages)

        # Should have: system + user + final assistant + summary message
        # Temp context should be removed
        # Summarizable messages should be summarized
        assert len(result.removed) == 1
        assert len(result.summarized) == 3
        assert result.summary is not None
        assert "tool interactions" in result.summary
