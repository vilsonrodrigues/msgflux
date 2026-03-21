"""BootstrapFewShot - generate demonstrations from successful teacher traces."""

import hashlib
import inspect
import logging
import random
from typing import Any, Callable

from msgflux.context import trace_context
from msgflux.examples import Example
from msgflux.nn.modules.module import Module
from msgflux.optim.labeled_fewshot import LabeledFewShot
from msgflux.optim.teleprompter import Teleprompter

logger = logging.getLogger(__name__)


class BootstrapFewShot(Teleprompter):
    """Bootstrap demonstrations from successful teacher executions."""

    def __init__(
        self,
        metric: Callable | None = None,
        *,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 16,
        max_rounds: int = 1,
        max_errors: int | None = None,
        metric_threshold: float | None = None,
        teacher_settings: dict[str, Any] | None = None,
    ):
        self.metric = metric
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.max_rounds = max_rounds
        self.max_errors = max_errors
        self.metric_threshold = metric_threshold
        self.teacher_settings = teacher_settings or {}
        self.error_count = 0
        self.metric_accepts_trace = False

        if metric is not None:
            try:
                inspect.signature(metric).bind(None, None, None)
            except TypeError:
                self.metric_accepts_trace = False
            else:
                self.metric_accepts_trace = True

    def compile(
        self,
        student: Module,
        *,
        trainset: list[Example],
        teacher: Module | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> Module:
        self.trainset = trainset
        self._prepare_student_and_teacher(student, teacher)
        self._prepare_agent_mappings()
        self._bootstrap()
        self.student = self._train()
        self.student.compile_(
            optimizer="BootstrapFewShot",
            score=None,
        )
        return self.student

    def _prepare_student_and_teacher(
        self,
        student: Module,
        teacher: Module | None,
    ) -> None:
        self.student = student.reset_copy()
        if getattr(self.student, "_compiled", False):
            raise ValueError("Student must be uncompiled.")

        self.teacher = teacher.deepcopy() if teacher is not None else student.deepcopy()

        if (
            self.max_labeled_demos
            and not getattr(self.teacher, "_compiled", False)
        ):
            tp = LabeledFewShot(k=self.max_labeled_demos)
            self.teacher = tp.compile(
                self.teacher.reset_copy(),
                trainset=self.trainset,
            )

    def _prepare_agent_mappings(self) -> None:
        student_agents = list(self.student.named_agents())
        teacher_agents = list(self.teacher.named_agents())

        if len(student_agents) != len(teacher_agents):
            raise ValueError(
                f"Student has {len(student_agents)} agents but teacher has "
                f"{len(teacher_agents)} agents."
            )

        self.name2agent: dict[str, dict[str, Any]] = {}
        self.agent2name: dict[int, str] = {}

        for (s_name, s_agent), (t_name, t_agent) in zip(
            student_agents,
            teacher_agents,
            strict=False,
        ):
            if s_name != t_name:
                raise ValueError(
                    "Student and teacher must have the same program structure."
                )
            if id(s_agent) == id(t_agent):
                raise ValueError("Student and teacher must be different objects.")

            self.name2agent[s_name] = {
                "student": s_agent,
                "teacher": t_agent,
            }
            self.agent2name[id(s_agent)] = s_name
            self.agent2name[id(t_agent)] = t_name

        self.name2traces: dict[str, list[Example]] = {
            name: [] for name in self.name2agent
        }

    def _bootstrap(self, *, max_bootstraps: int | None = None) -> None:
        max_bootstraps = max_bootstraps or self.max_bootstrapped_demos
        bootstrap_attempts = 0
        bootstrapped: dict[int, bool] = {}
        self.name2traces = {name: [] for name in self.name2agent}

        for example_idx, example in enumerate(self.trainset):
            if len(bootstrapped) >= max_bootstraps:
                break

            for round_idx in range(self.max_rounds):
                bootstrap_attempts += 1

                if self._bootstrap_one_example(example, round_idx):
                    bootstrapped[example_idx] = True
                    break

        logger.info(
            "Bootstrapped %s full traces after %s attempts.",
            len(bootstrapped),
            bootstrap_attempts,
        )

        self.validation = [
            example
            for idx, example in enumerate(self.trainset)
            if idx not in bootstrapped
        ]
        random.Random(0).shuffle(self.validation)  # noqa: S311

    def _bootstrap_one_example(
        self,
        example: Example,
        round_idx: int = 0,
    ) -> bool:
        trace: list[tuple[Any, dict[str, Any]]] = []
        teacher_demo_cache, model_settings_cache = self._prepare_teacher_round(
            example,
            round_idx,
        )

        try:
            with trace_context() as trace:
                prediction = self._run_teacher(example)

            success = self._metric_succeeds(example, prediction, trace)
        except Exception:
            success = False
            self.error_count += 1
            logger.error("Error bootstrapping example", exc_info=True)
            max_errors = self.max_errors or len(self.trainset) * 10
            if self.error_count >= max_errors:
                raise
        finally:
            self._restore_teacher_round(teacher_demo_cache, model_settings_cache)

        if success:
            self._record_successful_trace(trace)

        return success

    def _train(self) -> Module:
        rng = random.Random(0)  # noqa: S311
        raw_demos = list(getattr(self, "validation", []))

        for name, agent_pair in self.name2agent.items():
            student_agent = agent_pair["student"]
            augmented_demos = self.name2traces[name][: self.max_bootstrapped_demos]

            sample_size = min(
                self.max_labeled_demos - len(augmented_demos),
                len(raw_demos),
            )
            sample_size = max(0, sample_size)
            raw_demos = rng.sample(raw_demos, sample_size) if sample_size else []
            student_agent.optimized_examples = augmented_demos + list(raw_demos)

        return self.student

    def _metric_succeeds(
        self,
        example: Example,
        prediction: Any,
        trace: list[tuple[Any, dict[str, Any]]],
    ) -> bool:
        if self.metric is None:
            return True

        if self.metric_accepts_trace:
            score = self.metric(example, prediction, trace)
        else:
            score = self.metric(example, prediction)

        numeric_score = float(score) if isinstance(score, bool) else score
        if self.metric_threshold is not None:
            return float(numeric_score) >= self.metric_threshold
        return bool(numeric_score)

    def _prepare_teacher_round(
        self,
        example: Example,
        round_idx: int,
    ) -> tuple[dict[str, list[Example]], dict[int, dict[str, Any]]]:
        teacher_demo_cache: dict[str, list[Example]] = {}
        model_settings_cache: dict[int, dict[str, Any]] = {}

        for name, agent_pair in self.name2agent.items():
            teacher_agent = agent_pair["teacher"]
            current_demos = list(getattr(teacher_agent, "optimized_examples", []))
            teacher_demo_cache[name] = current_demos
            teacher_agent.optimized_examples = [
                demo for demo in current_demos if demo != example
            ]

            model = getattr(teacher_agent, "model", None)
            sampling_run_params = getattr(model, "sampling_run_params", None)
            if not isinstance(sampling_run_params, dict):
                continue

            model_id = id(model)
            if model_id not in model_settings_cache:
                model_settings_cache[model_id] = dict(sampling_run_params)

            updated_params = dict(sampling_run_params)
            updated_params.update(self.teacher_settings)
            if round_idx > 0:
                updated_params["temperature"] = 1.0
            model.sampling_run_params = updated_params

        return teacher_demo_cache, model_settings_cache

    def _restore_teacher_round(
        self,
        teacher_demo_cache: dict[str, list[Example]],
        model_settings_cache: dict[int, dict[str, Any]],
    ) -> None:
        for name, agent_pair in self.name2agent.items():
            agent_pair["teacher"].optimized_examples = teacher_demo_cache.get(
                name, []
            )

        for _name, agent_pair in self.name2agent.items():
            model = getattr(agent_pair["teacher"], "model", None)
            model_id = id(model)
            if model_id in model_settings_cache:
                model.sampling_run_params = model_settings_cache[model_id]

    def _record_successful_trace(
        self,
        trace: list[tuple[Any, dict[str, Any]]],
    ) -> None:
        name2traces: dict[str, list[Example]] = {}
        for agent, turn_record in trace:
            agent_name = self.agent2name.get(id(agent))
            if agent_name is None:
                continue

            demo = Example(
                inputs=turn_record.get("inputs"),
                labels=turn_record.get("assistant_output"),
            )
            name2traces.setdefault(agent_name, []).append(demo)

        for name, demos in name2traces.items():
            selected_demos = demos
            if len(demos) > 1:
                selected_demos = [self._select_demo_from_trace(demos)]
            self.name2traces[name].extend(selected_demos)

    @staticmethod
    def _run_module(example: Example, program: Module | None = None) -> Any:
        if program is None:
            raise ValueError("Teacher module is required.")
        inputs = example.inputs if hasattr(example, "inputs") else example
        if isinstance(inputs, dict):
            try:
                return program(inputs)
            except TypeError as original_exc:
                try:
                    return program(**inputs)
                except TypeError:
                    raise original_exc from None
        return program(inputs)

    def _run_teacher(self, example: Example) -> Any:  # type: ignore[override]
        return self._run_module(example, self.teacher)

    @staticmethod
    def _select_demo_from_trace(demos: list[Example]) -> Example:
        if len(demos) == 1:
            return demos[0]

        payload = repr(
            [(demo.inputs, demo.labels, demo.reasoning) for demo in demos]
        ).encode("utf-8")
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        rng = random.Random(seed)  # noqa: S311
        return rng.choice(demos[:-1]) if rng.random() < 0.5 else demos[-1]
