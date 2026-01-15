"""Integration tests for optimizers with Agent module.

Note: These tests work around a known issue with ModuleDict not being hashable
by directly accessing Agent's Parameter attributes instead of using parameters().
"""

from typing import Any, List, Mapping, Optional, Union
from unittest.mock import MagicMock, patch

import pytest

from msgflux.evaluate.evaluator import EvaluationResult, Evaluator
from msgflux.evaluate.metrics import exact_match
from msgflux.examples import Example
from msgflux.generation.templates import PromptSpec
from msgflux.models.response import ModelResponse
from msgflux.nn.modules.agent import Agent
from msgflux.nn.modules.module import Module
from msgflux.nn.parameter import Parameter
from msgflux.optim.bootstrap import BootstrapFewShot
from msgflux.optim.labeled_fewshot import LabeledFewShot
from msgflux.optim.optimizer import Optimizer
from msgflux.trainer.trainer import Trainer, TrainerConfig


class MockChatCompletionModel:
    """Mock chat completion model for testing.

    This mock has the required model_type attribute to pass validation.
    """

    model_type = "chat_completion"

    def __init__(self, responses: Optional[Mapping[str, str]] = None):
        self.responses = responses or {}
        self.call_count = 0
        self.last_messages = None
        self.last_system_prompt = None

    def _create_response(self, data: str) -> ModelResponse:
        """Create a ModelResponse with the given data."""
        response = ModelResponse()
        response.add(data)
        response.set_response_type("text_generation")
        return response

    def __call__(self, **kwargs) -> ModelResponse:
        self.call_count += 1
        self.last_messages = kwargs.get("messages", [])
        self.last_system_prompt = kwargs.get("system_prompt", "")

        # Extract the user message to determine response
        user_content = ""
        for msg in self.last_messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    user_content = content
                elif isinstance(content, list):
                    # Handle multimodal content
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            user_content = item.get("text", "")
                            break
                break

        # Find matching response
        for key, response in self.responses.items():
            if key in user_content:
                return self._create_response(response)

        return self._create_response("default response")

    async def acall(self, **kwargs) -> ModelResponse:
        return self.__call__(**kwargs)


def get_agent_parameters(agent: Agent) -> List[Parameter]:
    """Extract parameters directly from Agent's attributes.

    This is a workaround for the ModuleDict hashability issue.
    """
    params = []
    # Known Parameter attributes in Agent
    for attr_name in ["system_message", "instructions", "expected_output", "examples"]:
        if hasattr(agent, attr_name):
            param = getattr(agent, attr_name)
            if isinstance(param, Parameter):
                params.append(param)
    return params


@pytest.fixture
def trainset():
    """Create sample training set."""
    return [
        Example(inputs="What is 2+2?", labels="4"),
        Example(inputs="What is 3+3?", labels="6"),
        Example(inputs="What is 4+4?", labels="8"),
        Example(inputs="What is 5+5?", labels="10"),
        Example(inputs="What is 6+6?", labels="12"),
        Example(inputs="What is 7+7?", labels="14"),
        Example(inputs="What is 8+8?", labels="16"),
        Example(inputs="What is 9+9?", labels="18"),
    ]


@pytest.fixture
def valset():
    """Create sample validation set."""
    return [
        Example(inputs="What is 10+10?", labels="20"),
        Example(inputs="What is 11+11?", labels="22"),
    ]


@pytest.fixture
def mock_model():
    """Create mock model with correct responses."""
    return MockChatCompletionModel(
        responses={
            "2+2": "4",
            "3+3": "6",
            "4+4": "8",
            "5+5": "10",
            "6+6": "12",
            "7+7": "14",
            "8+8": "16",
            "9+9": "18",
            "10+10": "20",
            "11+11": "22",
        }
    )


@pytest.fixture
def agent(mock_model):
    """Create Agent instance with mock model."""
    agent = Agent(
        name="calculator",
        model=mock_model,
        system_message="You are a calculator.",
        instructions="Calculate the answer to the math question.",
    )
    return agent


class TestAgentParameters:
    """Test Agent parameter integration."""

    def test_agent_has_parameters(self, agent):
        """Test that Agent has optimizable parameters."""
        params = get_agent_parameters(agent)

        assert len(params) > 0
        assert all(isinstance(p, Parameter) for p in params)

    def test_agent_has_examples_parameter(self, agent):
        """Test that Agent has examples parameter."""
        assert hasattr(agent, "examples")
        assert isinstance(agent.examples, Parameter)
        assert agent.examples.spec == PromptSpec.EXAMPLES

    def test_agent_has_system_message_parameter(self, agent):
        """Test that Agent has system_message parameter."""
        assert hasattr(agent, "system_message")
        assert isinstance(agent.system_message, Parameter)
        assert agent.system_message.spec == PromptSpec.SYSTEM_MESSAGE

    def test_agent_has_instructions_parameter(self, agent):
        """Test that Agent has instructions parameter."""
        assert hasattr(agent, "instructions")
        assert isinstance(agent.instructions, Parameter)
        assert agent.instructions.spec == PromptSpec.INSTRUCTIONS

    def test_agent_parameter_data_accessible(self, agent):
        """Test that parameter data is accessible."""
        assert agent.system_message.data == "You are a calculator."
        assert agent.instructions.data == "Calculate the answer to the math question."

    def test_agent_train_eval_modes(self, agent):
        """Test Agent train/eval mode switching."""
        agent.train()
        assert agent.training is True

        agent.eval()
        assert agent.training is False


class TestLabeledFewShotWithAgent:
    """Test LabeledFewShot optimizer with Agent."""

    def test_optimizer_with_agent_parameters(self, agent, trainset):
        """Test creating optimizer with Agent parameters."""
        params = get_agent_parameters(agent)
        optimizer = LabeledFewShot(params, trainset=trainset, k=4, seed=42)

        assert optimizer is not None
        assert optimizer.k == 4

    def test_step_updates_agent_examples(self, agent, trainset):
        """Test that step updates Agent's examples parameter."""
        # Get initial examples parameter
        initial_examples = agent.examples.data

        params = get_agent_parameters(agent)
        optimizer = LabeledFewShot(params, trainset=trainset, k=4, seed=42)
        optimizer.step()

        # Examples should be updated
        assert agent.examples.data != initial_examples
        assert agent.examples.data is not None
        assert len(agent.examples.data) > 0

    def test_optimizer_respects_requires_grad(self, agent, trainset):
        """Test that optimizer respects requires_grad flag."""
        # Freeze examples parameter
        agent.examples.requires_grad = False
        original_examples = agent.examples.data

        params = get_agent_parameters(agent)
        optimizer = LabeledFewShot(params, trainset=trainset, k=4, seed=42)
        optimizer.step()

        # Examples should not be updated
        assert agent.examples.data == original_examples

    def test_multiple_agents_optimization(self, mock_model, trainset):
        """Test optimizing multiple agents."""
        agent1 = Agent(
            name="calculator1",
            model=mock_model,
            system_message="Calculator 1",
        )
        agent2 = Agent(
            name="calculator2",
            model=mock_model,
            system_message="Calculator 2",
        )

        # Optimize both agents' examples
        all_params = get_agent_parameters(agent1) + get_agent_parameters(agent2)

        optimizer = LabeledFewShot(all_params, trainset=trainset, k=4, seed=42)
        optimizer.step()

        # Both should have examples updated
        assert agent1.examples.data is not None
        assert agent2.examples.data is not None

    def test_step_increments_count(self, agent, trainset):
        """Test that step increments step count."""
        params = get_agent_parameters(agent)
        optimizer = LabeledFewShot(params, trainset=trainset, k=4, seed=42)

        optimizer.step()
        assert optimizer._step_count == 1

        optimizer.step()
        assert optimizer._step_count == 2


class TestBootstrapFewShotWithAgent:
    """Test BootstrapFewShot optimizer with Agent."""

    @pytest.fixture
    def metric(self):
        """Create exact match metric."""
        return exact_match

    def test_optimizer_with_agent_parameters(self, agent, trainset, metric):
        """Test creating BootstrapFewShot with Agent parameters."""
        params = get_agent_parameters(agent)
        optimizer = BootstrapFewShot(
            params,
            metric=metric,
            max_bootstrapped_demos=2,
            max_labeled_demos=4,
            seed=42,
        )

        assert optimizer is not None
        assert optimizer.max_bootstrapped_demos == 2

    def test_step_without_teacher(self, agent, trainset, metric):
        """Test step without teacher (only labeled demos)."""
        params = get_agent_parameters(agent)
        optimizer = BootstrapFewShot(
            params,
            metric=metric,
            max_bootstrapped_demos=2,
            max_labeled_demos=4,
            seed=42,
        )

        optimizer.step(trainset, teacher=None)

        assert optimizer._step_count == 1


class TestEvaluatorWithAgent:
    """Test Evaluator with Agent."""

    @pytest.fixture
    def metric(self):
        """Create exact match metric."""
        return exact_match

    @pytest.fixture
    def evaluator(self, metric):
        """Create Evaluator."""
        return Evaluator(metric=metric)

    def test_evaluate_agent(self, evaluator, agent, valset):
        """Test evaluating an Agent."""
        result = evaluator(agent, valset)

        assert isinstance(result, EvaluationResult)
        assert result.score >= 0.0
        assert result.score <= 100.0

    def test_evaluate_sets_eval_mode(self, evaluator, agent, valset):
        """Test that evaluator sets Agent to eval mode."""
        agent.train()
        assert agent.training is True

        evaluator(agent, valset)

        # Should be restored to training mode
        assert agent.training is True

    def test_evaluate_returns_predictions(self, evaluator, agent, valset):
        """Test that evaluator returns predictions."""
        result = evaluator(agent, valset, return_predictions=True)

        assert len(result.results) == len(valset)

    def test_evaluate_single_example(self, evaluator, agent):
        """Test evaluating a single example."""
        example = Example(inputs="What is 2+2?", labels="4")

        prediction, score = evaluator.evaluate_single(agent, example)

        assert prediction is not None
        assert isinstance(score, float)


class TestTrainerWithAgent:
    """Test Trainer with Agent."""

    @pytest.fixture
    def metric(self):
        """Create exact match metric."""
        return exact_match

    @pytest.fixture
    def evaluator(self, metric):
        """Create Evaluator."""
        return Evaluator(metric=metric)

    @pytest.fixture
    def optimizer(self, agent, trainset):
        """Create optimizer for training."""
        params = get_agent_parameters(agent)
        return LabeledFewShot(params, trainset=trainset, k=4, seed=42)

    def test_trainer_with_agent(self, agent, optimizer, evaluator, trainset, valset):
        """Test training an Agent."""
        trainer = Trainer(
            module=agent,
            optimizer=optimizer,
            evaluator=evaluator,
            config=TrainerConfig(max_epochs=3, verbose=False),
        )

        state = trainer.fit(trainset=trainset, valset=valset)

        assert state.epoch == 2  # 0-indexed
        assert state.best_score >= 0.0

    def test_trainer_tracks_history(
        self, agent, optimizer, evaluator, trainset, valset
    ):
        """Test that trainer tracks history."""
        trainer = Trainer(
            module=agent,
            optimizer=optimizer,
            evaluator=evaluator,
            config=TrainerConfig(max_epochs=3, verbose=False),
        )

        trainer.fit(trainset=trainset, valset=valset)

        assert len(trainer.state.history) == 3
        for entry in trainer.state.history:
            assert "epoch" in entry
            assert "val_score" in entry


class TestEndToEndOptimization:
    """End-to-end optimization tests."""

    @pytest.fixture
    def metric(self):
        """Create exact match metric."""
        return exact_match

    def test_full_optimization_pipeline(self, mock_model, trainset, valset, metric):
        """Test complete optimization pipeline."""
        # Create agent
        agent = Agent(
            name="calculator",
            model=mock_model,
            system_message="You are a calculator.",
            instructions="Calculate the answer.",
        )

        # Create optimizer
        params = get_agent_parameters(agent)
        optimizer = LabeledFewShot(params, trainset=trainset, k=4, seed=42)

        # Create evaluator
        evaluator = Evaluator(metric=metric)

        # Create trainer
        trainer = Trainer(
            module=agent,
            optimizer=optimizer,
            evaluator=evaluator,
            config=TrainerConfig(max_epochs=2, verbose=False),
        )

        # Train
        state = trainer.fit(trainset=trainset, valset=valset)

        # Verify results
        assert state.best_score >= 0.0
        assert agent.examples.data is not None

    def test_state_persistence(self, mock_model, trainset, valset, metric):
        """Test that optimizer state can be saved and loaded."""
        # Create and train agent
        agent1 = Agent(
            name="calculator1",
            model=mock_model,
            system_message="Calculator 1.",
        )

        params1 = get_agent_parameters(agent1)
        optimizer1 = LabeledFewShot(params1, trainset=trainset, k=4, seed=42)
        optimizer1.step()
        state = optimizer1.state_dict()

        # Create new optimizer and load state
        agent2 = Agent(
            name="calculator2",
            model=mock_model,
            system_message="Calculator 2.",
        )

        params2 = get_agent_parameters(agent2)
        optimizer2 = LabeledFewShot(params2, trainset=trainset, k=2, seed=1)
        optimizer2.load_state_dict(state)

        # Verify state was restored
        assert optimizer2.seed == 42

    def test_parameter_update_persists(self, mock_model, trainset):
        """Test that parameter updates persist across forward calls."""
        agent = Agent(
            name="calculator",
            model=mock_model,
            system_message="Calculator.",
        )

        # Initial examples should be None or empty
        initial_examples = agent.examples.data

        # Optimize
        params = get_agent_parameters(agent)
        optimizer = LabeledFewShot(params, trainset=trainset, k=4, seed=42)
        optimizer.step()

        # Examples should be updated
        updated_examples = agent.examples.data
        assert updated_examples != initial_examples

        # Call agent - examples should still be updated
        agent("What is 2+2?")
        assert agent.examples.data == updated_examples


class TestCompilationState:
    """Test module compilation state tracking."""

    @pytest.fixture
    def metric(self):
        """Create exact match metric."""
        return exact_match

    def test_module_not_compiled_initially(self, agent):
        """Test that modules are not compiled initially."""
        assert agent.compiled is False
        assert agent.get_compile_info() == {}

    def test_compile_marks_module(self, agent):
        """Test that compile_() marks module as compiled."""
        agent.compile_(optimizer="TestOptimizer", score=0.95)

        assert agent.compiled is True
        info = agent.get_compile_info()
        assert info["optimizer"] == "TestOptimizer"
        assert info["score"] == 0.95

    def test_decompile_resets_state(self, agent):
        """Test that decompile_() resets compilation state."""
        agent.compile_(optimizer="TestOptimizer", score=0.95)
        assert agent.compiled is True

        agent.decompile_()
        assert agent.compiled is False
        assert agent.get_compile_info() == {}

    def test_trainer_marks_compiled(
        self, mock_model, trainset, valset, metric
    ):
        """Test that Trainer marks module as compiled after training."""
        agent = Agent(
            name="calculator",
            model=mock_model,
            system_message="Calculator.",
        )

        assert agent.compiled is False

        params = get_agent_parameters(agent)
        optimizer = LabeledFewShot(params, trainset=trainset, k=4, seed=42)
        evaluator = Evaluator(metric=metric)

        trainer = Trainer(
            module=agent,
            optimizer=optimizer,
            evaluator=evaluator,
            config=TrainerConfig(max_epochs=2, verbose=False),
        )

        trainer.fit(trainset=trainset, valset=valset)

        assert agent.compiled is True
        info = agent.get_compile_info()
        assert info["optimizer"] == "LabeledFewShot"
        assert "score" in info
        assert "epochs" in info

    def test_compile_state_persisted_in_state_dict(self, agent):
        """Test that compilation state is saved in state_dict."""
        agent.compile_(optimizer="TestOptimizer", score=0.95, custom_key="value")

        state = agent.state_dict()

        assert "_compiled" in state
        assert state["_compiled"] is True
        assert "_compile_info" in state
        assert state["_compile_info"]["optimizer"] == "TestOptimizer"

    def test_compile_state_loaded_from_state_dict(self, mock_model):
        """Test that compilation state is loaded from state_dict."""
        # Create and compile first agent
        agent1 = Agent(
            name="agent1",
            model=mock_model,
            system_message="Agent 1.",
        )
        agent1.compile_(optimizer="TestOptimizer", score=0.95)
        state = agent1.state_dict()

        # Create new agent and load state
        agent2 = Agent(
            name="agent2",
            model=mock_model,
            system_message="Agent 2.",
        )
        assert agent2.compiled is False

        agent2.load_state_dict(state)

        assert agent2.compiled is True
        info = agent2.get_compile_info()
        assert info["optimizer"] == "TestOptimizer"
        assert info["score"] == 0.95
