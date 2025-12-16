"""Policy factory for creating policy instances."""

from collections.abc import Callable
from typing import Any

from msgflux.models.state.policies.base import CompactionPolicy
from msgflux.models.state.policies.importance import ImportancePolicy
from msgflux.models.state.policies.lifecycle import LifecyclePolicy
from msgflux.models.state.policies.position import PositionBasedPolicy
from msgflux.models.state.policies.sliding_window import SlidingWindowPolicy
from msgflux.models.state.policies.types import Policy
from msgflux.models.state.types import ChatMessage


def create_policy(
    policy: Policy | dict[str, Any] | str,
    summarizer: Callable[[list[ChatMessage]], str] | None = None,
    **kwargs,
) -> CompactionPolicy:
    """Create a policy from configuration.

    Args:
        policy: Policy config, dict, or type string.
        summarizer: Function to summarize messages.
        **kwargs: Additional config options.

    Returns:
        Configured CompactionPolicy instance.

    Example:
        # From Policy object
        p = create_policy(Policy(type="sliding_window", max_messages=50))

        # From dict
        p = create_policy({"type": "position", "preserve_first_pct": 0.1})

        # From string
        p = create_policy("lifecycle")
    """
    if isinstance(policy, str):
        config = Policy(type=policy, **kwargs)
    elif isinstance(policy, dict):
        config = Policy(**{**policy, **kwargs})
    else:
        config = policy

    policy_type = config.type.lower()

    if policy_type == "sliding_window":
        return SlidingWindowPolicy(config=config, summarizer=summarizer)
    if policy_type == "position":
        return PositionBasedPolicy(config=config, summarizer=summarizer)
    if policy_type == "lifecycle":
        return LifecyclePolicy(config=config, summarizer=summarizer)
    if policy_type == "importance":
        return ImportancePolicy(config=config, summarizer=summarizer)

    raise ValueError(f"Unknown policy type: {policy_type}")
