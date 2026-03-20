"""Tutorial: GEPA Optimizer for AIME Math (inspired by DSPy GEPA AIME tutorial).

This tutorial demonstrates how to use the GEPA (Guided Evolution for Prompt
Adaptation) optimizer to improve an Agent's ability to solve AIME-style
math competition problems.

We load a subset of the real AIME dataset from HuggingFace, define a
Chain-of-Thought agent, and use GEPA to iteratively improve its
instructions via reflective feedback.

Reference: https://dspy.ai/tutorials/gepa_aime/
"""

import logging
import random

import msgflux as mf
from msgflux.examples import Example
from msgflux.generation.reasoning.cot import ChainOfThought
from msgflux.nn.modules.agent import Agent
from msgflux.nn.modules.module import Module
from msgflux.optim import GEPA, Evaluate

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

# ── 0. Load environment (expects OPENAI_API_KEY in .env) ────────────
mf.load_dotenv()

MODEL_ID = "openai/gpt-4.1-mini"


# ── 1. Dataset (AIME problems) ─────────────────────────────────────
# Real AIME problems from HuggingFace.
# Small subset to keep costs low (6 train, 4 val, 4 test).

def load_aime_dataset(
    train_size: int = 6,
    val_size: int = 4,
    test_size: int = 4,
):
    """Load AIME dataset from HuggingFace and split into train/val/test."""
    from datasets import load_dataset

    raw = load_dataset("AI-MO/aimo-validation-aime")["train"]

    examples = [
        Example(
            inputs=row["problem"],
            labels=str(row["answer"]),
        )
        for row in raw
    ]
    random.Random(0).shuffle(examples)

    total_needed = train_size + val_size + test_size
    examples = examples[:total_needed]

    train_set = examples[:train_size]
    val_set = examples[train_size : train_size + val_size]
    test_set = examples[train_size + val_size :]

    return train_set, val_set, test_set


print("Loading AIME dataset...")
train_set, val_set, test_set = load_aime_dataset()
print(f"Dataset: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")
print(f"\nSample problem:\n{train_set[0].inputs[:200]}...")
print(f"Answer: {train_set[0].labels}")


# ── 2. Define the program ──────────────────────────────────────────
# A simple Agent with Chain-of-Thought reasoning, analogous to
# dspy.ChainOfThought(GenerateResponse) in the DSPy tutorial.


class MathSolver(Module):
    """Pipeline with a single CoT agent that solves AIME math problems."""

    def __init__(self, model_id: str = MODEL_ID):
        super().__init__()
        self.solver = Agent(
            name="math_solver",
            model=mf.Model.chat_completion(
                model_id, max_tokens=32000, temperature=1.0,
            ),
            system_message="You are an expert mathematician.",
            instructions=(
                "Solve the given math problem step by step. "
                "Your final answer MUST be a single integer with no "
                "additional text, formatting, or explanation after it."
            ),
            expected_output="A single integer as the final answer.",
            generation_schema=ChainOfThought,
        )

    def forward(self, message=None, **kwargs):
        result = self.solver(message, **kwargs)
        return result


program = MathSolver()


# ── 3. Metric helpers ──────────────────────────────────────────────


def _extract_answer(prediction) -> str:
    """Extract the answer string from a prediction.

    With generation_schema=ChainOfThought, the Agent returns a dict like:
        {'reasoning': '...', 'final_answer': '42'}
    """
    if prediction is None:
        return ""
    # Dict from structured generation (ChainOfThought)
    if isinstance(prediction, dict) and "final_answer" in prediction:
        return str(prediction["final_answer"]).strip()
    # msgspec Struct with .final_answer
    if hasattr(prediction, "final_answer"):
        return str(prediction.final_answer).strip()
    # Plain string
    if isinstance(prediction, str):
        return prediction.strip()
    return str(prediction).strip()


# ── 4. Simple metric (for evaluation) ──────────────────────────────


def metric(example: Example, prediction, **kwargs) -> int:
    """Return 1 if the predicted integer matches the expected answer."""
    expected = str(example.labels).strip()
    pred_str = _extract_answer(prediction)
    try:
        return int(int(pred_str) == int(expected))
    except (ValueError, TypeError):
        return 0


# ── 5. Metric with feedback (for GEPA reflective optimization) ─────
# GEPA benefits from richer feedback: when the answer is wrong, we tell
# the optimizer *what* the correct answer was so it can reflect on errors.


def metric_with_feedback(example: Example, prediction, **kwargs) -> dict:
    """Return a dict with ``score`` and ``feedback`` for GEPA."""
    expected = str(example.labels).strip()
    pred_str = _extract_answer(prediction)

    if not pred_str:
        return {
            "score": 0,
            "feedback": (
                f"No prediction was generated. "
                f"The correct answer is {expected}."
            ),
        }

    try:
        pred_int = int(pred_str)
        expected_int = int(expected)
        score = int(pred_int == expected_int)
    except (ValueError, TypeError):
        return {
            "score": 0,
            "feedback": (
                f"Could not parse '{pred_str}' as integer. "
                f"The correct answer is {expected}. "
                "Ensure the final answer is a single integer."
            ),
        }

    if score == 1:
        feedback = f"Correct! The answer is {expected}."
    else:
        feedback = (
            f"Incorrect. You answered {pred_str}, "
            f"but the correct answer is {expected}. "
            f"Review the problem carefully and ensure each step is correct."
        )

    return {"score": score, "feedback": feedback}


# ── 6. Evaluate unoptimized program ────────────────────────────────
print("\n" + "=" * 60)
print("Evaluating UNOPTIMIZED program on test set...")
print("=" * 60)

evaluate = Evaluate(
    devset=test_set,
    metric=metric,
    num_threads=1,
    return_all_scores=True,
)

baseline_result = evaluate(program)
print(f"\nBaseline score: {baseline_result.score:.2%}")
for ex, pred, score in baseline_result.results:
    status = "PASS" if score >= 1.0 else "FAIL"
    pred_str = _extract_answer(pred) or "None"
    print(
        f"  [{status}] ...{ex.inputs[-50:]:50s} "
        f"-> {pred_str:>8s} (expected {ex.labels})"
    )


# ── 7. Create a reflection LM for GEPA ─────────────────────────────
# GEPA needs a callable that takes a prompt string and returns a string.
# We use a simple Agent for this purpose.

reflection_agent = Agent(
    name="reflection_lm",
    model=mf.Model.chat_completion(MODEL_ID, max_tokens=16000, temperature=0.7),
    system_message="You are a prompt optimization expert.",
    instructions=(
        "Given the current instruction and examples of failures, "
        "propose an improved instruction that will help solve "
        "AIME math competition problems more accurately."
    ),
)


def reflection_lm(prompt: str) -> str:
    return reflection_agent(prompt)


# ── 8. Optimize with GEPA ──────────────────────────────────────────
print("\n" + "=" * 60)
print("Running GEPA optimization (3 iterations)...")
print("=" * 60)

optimizer = GEPA(
    metric=metric_with_feedback,
    reflection_lm=reflection_lm,
    max_iterations=3,
    num_candidates=2,
    num_threads=1,
)

optimized_program = optimizer.compile(
    program,
    trainset=train_set,
    valset=val_set,
)

print(f"\nOptimization complete!")
print(f"Compiled: {optimized_program.compiled}")
print(f"Compile info: {optimized_program.get_compile_info()}")


# ── 9. Inspect optimized prompts ───────────────────────────────────
print("\n" + "=" * 60)
print("Optimized Agent Prompts")
print("=" * 60)

for name, agent in optimized_program.named_agents():
    print(f"\n--- Agent: {name} ---")
    opt_prompt = agent.optimized_system_prompt.data
    if opt_prompt:
        print(f"Optimized system prompt:\n{opt_prompt[:800]}")
    else:
        print("(no optimized prompt)")
    if agent.demos:
        print(f"Demos: {len(agent.demos)} examples")


# ── 10. Evaluate optimized program ─────────────────────────────────
print("\n" + "=" * 60)
print("Evaluating OPTIMIZED program on test set...")
print("=" * 60)

optimized_result = evaluate(optimized_program)
print(f"\nOptimized score: {optimized_result.score:.2%}")
for ex, pred, score in optimized_result.results:
    status = "PASS" if score >= 1.0 else "FAIL"
    pred_str = _extract_answer(pred) or "None"
    print(
        f"  [{status}] ...{ex.inputs[-50:]:50s} "
        f"-> {pred_str:>8s} (expected {ex.labels})"
    )


# ── 11. Comparison ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESULTS COMPARISON")
print("=" * 60)
print(f"  Baseline (unoptimized): {baseline_result.score:.2%}")
print(f"  GEPA optimized:        {optimized_result.score:.2%}")
improvement = optimized_result.score - baseline_result.score
if improvement > 0:
    print(f"  Improvement:           +{improvement:.2%}")
elif improvement == 0:
    print("  No change in score.")
else:
    print(f"  Regression:            {improvement:.2%}")
