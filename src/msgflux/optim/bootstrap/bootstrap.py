"""BootstrapFewShot — generate demonstrations from successful teacher executions.

Executes a teacher module on the training set, captures traces from successful
executions (where metric > threshold), and assigns the best demonstrations
per agent.
"""

import logging
import random
from typing import Any, Callable, List, Optional

from msgflux.context import trace_context
from msgflux.examples import Example
from msgflux.nn.modules.module import Module
from msgflux.optim.labeled_fewshot import LabeledFewShot
from msgflux.optim.teleprompter import Teleprompter

logger = logging.getLogger(__name__)


class BootstrapFewShot(Teleprompter):
    """Bootstrap demonstrations from a teacher's successful executions.

    Args:
        metric: ``metric(example, prediction) -> float | bool``
        max_bootstrapped_demos: Maximum bootstrapped demos per agent.
        max_labeled_demos: Maximum raw labeled demos per agent.
        max_rounds: Retry rounds per example.
        max_errors: Maximum errors before stopping.
        metric_threshold: Minimum metric for a trace to qualify.

    Example::

        optimizer = BootstrapFewShot(
            metric=exact_match,
            max_bootstrapped_demos=4,
            max_labeled_demos=4,
        )
        compiled = optimizer.compile(student, trainset=examples)
    """

    def __init__(
        self,
        metric: Callable,
        *,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 16,
        max_rounds: int = 1,
        max_errors: Optional[int] = None,
        metric_threshold: Optional[float] = None,
        teacher_settings: Optional[dict] = None,
    ):
        self.metric = metric
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.max_rounds = max_rounds
        self.max_errors = max_errors
        self.metric_threshold = metric_threshold
        self.teacher_settings = teacher_settings or {}

    def compile(
        self,
        student: Module,
        *,
        trainset: List[Example],
        teacher: Optional[Module] = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> Module:
        self.trainset = trainset
        self._prepare_student_and_teacher(student, teacher)
        self._prepare_agent_mappings()
        self._bootstrap()
        self._train()
        self.student.compile_(
            optimizer="BootstrapFewShot",
            score=None,
        )
        return self.student

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _prepare_student_and_teacher(
        self,
        student: Module,
        teacher: Optional[Module],
    ) -> None:
        self.student = student.reset_copy()
        if teacher is not None:
            self.teacher = teacher.deepcopy()
        else:
            self.teacher = student.deepcopy()

        if not getattr(self.teacher, "_compiled", False):
            tp = LabeledFewShot(k=self.max_labeled_demos)
            self.teacher = tp.compile(
                self.teacher, trainset=self.trainset
            )

    def _prepare_agent_mappings(self) -> None:
        student_agents = list(self.student.named_agents())
        teacher_agents = list(self.teacher.named_agents())

        if len(student_agents) != len(teacher_agents):
            raise ValueError(
                f"Student has {len(student_agents)} agents but teacher has "
                f"{len(teacher_agents)} agents."
            )

        self.name2agent: dict[str, Any] = {}
        self.agent2name: dict[int, str] = {}
        for (s_name, s_agent), (_t_name, t_agent) in zip(
            student_agents, teacher_agents
        ):
            self.name2agent[s_name] = {
                "student": s_agent,
                "teacher": t_agent,
            }
            self.agent2name[id(t_agent)] = s_name

        self.name2traces: dict[str, list] = {
            name: [] for name in self.name2agent
        }

    def _bootstrap(self) -> None:
        error_count = 0
        max_errors = self.max_errors or len(self.trainset) * 10

        for example in self.trainset:
            if error_count >= max_errors:
                break

            for round_idx in range(self.max_rounds):
                try:
                    success = self._bootstrap_one_example(
                        example, round_idx
                    )
                    if success:
                        break
                except Exception:
                    error_count += 1
                    logger.error(
                        "Error bootstrapping example", exc_info=True
                    )
                    break

    def _bootstrap_one_example(
        self, example: Example, round_idx: int = 0,  # noqa: ARG002
    ) -> bool:
        teacher = self.teacher

        # Anti-leak: remove current example from teacher demos
        for _name, agent_pair in self.name2agent.items():
            t_agent = agent_pair["teacher"]
            if hasattr(t_agent, "optimized_examples"):
                t_agent.optimized_examples = [
                    d for d in t_agent.optimized_examples if d is not example
                ]

        # Execute teacher inside a trace context
        with trace_context() as trace:
            inputs = example.inputs if hasattr(example, "inputs") else example
            prediction = teacher(inputs)

        # Evaluate with metric
        score = self.metric(example, prediction)
        if isinstance(score, bool):
            score = float(score)
        if self.metric_threshold is not None and score < self.metric_threshold:
            return False
        if score <= 0:
            return False

        # Extract demos from trace: each entry is (agent, turn_record)
        for agent, turn_record in trace:
            agent_name = self.agent2name.get(id(agent))
            if agent_name is None:
                continue

            demo = Example(
                inputs=turn_record.get("inputs"),
                labels=turn_record.get("assistant_output"),
            )
            self.name2traces[agent_name].append(demo)

        return True

    def _train(self) -> None:
        rng = random.Random(0)  # noqa: S311

        for name, agent_pair in self.name2agent.items():
            s_agent = agent_pair["student"]
            bootstrapped = self.name2traces.get(name, [])

            # Cap bootstrapped demos
            if len(bootstrapped) > self.max_bootstrapped_demos:
                bootstrapped = rng.sample(
                    bootstrapped, self.max_bootstrapped_demos
                )

            # Fill remaining slots with raw labeled demos
            remaining = self.max_labeled_demos - len(bootstrapped)
            labeled = []
            if remaining > 0:
                k = min(remaining, len(self.trainset))
                labeled = rng.sample(self.trainset, k)

            s_agent.optimized_examples = bootstrapped + labeled
