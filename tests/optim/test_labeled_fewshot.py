"""Tests for LabeledFewShot optimizer with compile() interface."""

import pytest
from unittest.mock import Mock

from msgflux.examples import Example
from msgflux.nn.modules.agent import Agent
from msgflux.optim.labeled_fewshot import LabeledFewShot


@pytest.fixture
def mock_model():
    model = Mock()
    model.model_type = "chat_completion"
    return model


@pytest.fixture
def trainset():
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


class TestLabeledFewShot:
    def test_compile_returns_compiled_module(self, mock_model, trainset):
        student = Agent(name="qa", model=mock_model)
        opt = LabeledFewShot(k=4)
        compiled = opt.compile(student, trainset=trainset)

        assert compiled._compiled is True
        assert compiled is not student  # deep copy

    def test_compile_assigns_demos(self, mock_model, trainset):
        student = Agent(name="qa", model=mock_model)
        opt = LabeledFewShot(k=4)
        compiled = opt.compile(student, trainset=trainset)

        assert len(compiled.optimized_examples) == 4
        for demo in compiled.optimized_examples:
            assert isinstance(demo, Example)

    def test_k_clamped_to_trainset_size(self, mock_model, trainset):
        student = Agent(name="qa", model=mock_model)
        opt = LabeledFewShot(k=100)
        compiled = opt.compile(student, trainset=trainset)

        assert len(compiled.optimized_examples) == len(trainset)

    def test_sample_false_takes_first_k(self, mock_model, trainset):
        student = Agent(name="qa", model=mock_model)
        opt = LabeledFewShot(k=3)
        compiled = opt.compile(student, trainset=trainset, sample=False)

        assert compiled.optimized_examples == trainset[:3]

    def test_reproducible_with_same_seed(self, mock_model, trainset):
        student = Agent(name="qa", model=mock_model)
        opt = LabeledFewShot(k=4)
        compiled1 = opt.compile(student, trainset=trainset, seed=42)
        compiled2 = opt.compile(student, trainset=trainset, seed=42)

        assert compiled1.optimized_examples == compiled2.optimized_examples

    def test_different_seed_different_demos(self, mock_model, trainset):
        student = Agent(name="qa", model=mock_model)
        opt = LabeledFewShot(k=4)
        compiled1 = opt.compile(student, trainset=trainset, seed=0)
        compiled2 = opt.compile(student, trainset=trainset, seed=99)

        assert compiled1.optimized_examples != compiled2.optimized_examples

    def test_original_student_unchanged(self, mock_model, trainset):
        student = Agent(name="qa", model=mock_model)
        original_demos = student.optimized_examples.copy()
        opt = LabeledFewShot(k=4)
        opt.compile(student, trainset=trainset)

        assert student.optimized_examples == original_demos
        assert student._compiled is False

    def test_compile_info(self, mock_model, trainset):
        student = Agent(name="qa", model=mock_model)
        opt = LabeledFewShot(k=4)
        compiled = opt.compile(student, trainset=trainset)

        info = compiled.get_compile_info()
        assert info["optimizer"] == "LabeledFewShot"
