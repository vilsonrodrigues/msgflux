"""Tutorial: GEPA Optimizer for Classification (Banking77).

Inspired by DSPy's classification_finetuning tutorial, adapted for
msgflux's GEPA prompt optimizer. Instead of finetuning a local model,
we optimise the system prompt of an Agent to improve classification
accuracy on the Banking77 intent dataset.

Reference: https://dspy.ai/tutorials/classification_finetuning/
"""

import logging
import random

import msgflux as mf
from msgflux.examples import Example
from msgflux.nn.modules.agent import Agent
from msgflux.nn.modules.module import Module
from msgflux.optim import GEPA, Evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ── 0. Load environment ────────────────────────────────────────────
mf.load_dotenv()

MODEL_ID = "openai/gpt-4.1-mini"


# ── 1. Load Banking77 dataset ──────────────────────────────────────
print("Loading Banking77 dataset...")
from datasets import load_dataset

ds = load_dataset("legacy-datasets/banking77", split="train")
CLASSES = ds.features["label"].names
print(f"{len(CLASSES)} intent classes")

# Build examples: inputs = customer text, labels = intent label name
raw_data = [
    Example(inputs=row["text"], labels=CLASSES[row["label"]])
    for row in ds
]
random.Random(0).shuffle(raw_data)

# Small subsets for fast iteration
trainset = raw_data[:50]
valset = raw_data[50:80]
testset = raw_data[80:110]

print(f"Splits: train={len(trainset)}, val={len(valset)}, test={len(testset)}")
print(f"\nSample: {trainset[0].inputs!r}")
print(f"Label:  {trainset[0].labels!r}")

# Format the class list for the prompt
CLASSES_STR = ", ".join(CLASSES)


# ── 2. Define the classifier program ───────────────────────────────

class IntentClassifier(Module):
    """Single-agent intent classification pipeline."""

    def __init__(self, model_id: str = MODEL_ID):
        super().__init__()
        self.classifier = Agent(
            name="intent_classifier",
            model=mf.Model.chat_completion(
                model_id, max_tokens=200, temperature=0.0,
            ),
            system_message="You are a banking customer intent classifier.",
            instructions=(
                "Classify the customer message into one of these intents:\n"
                f"{CLASSES_STR}\n\n"
                "Respond with ONLY the intent label, nothing else."
            ),
            expected_output="A single intent label from the list above.",
        )

    def forward(self, message=None, **kwargs):
        return self.classifier(message, **kwargs)


program = IntentClassifier()


# ── 3. Metric ───────────────────────────────────────────────────────

def metric(example: Example, prediction, **kwargs) -> int:
    """Exact match on intent label."""
    if prediction is None:
        return 0
    pred_str = str(prediction).strip().lower()
    expected = str(example.labels).strip().lower()
    return int(pred_str == expected)


def metric_with_feedback(example: Example, prediction, **kwargs) -> dict:
    """Metric with feedback for GEPA reflective optimisation."""
    expected = str(example.labels).strip()
    if prediction is None:
        return {
            "score": 0,
            "feedback": f"No prediction. Correct: '{expected}'.",
        }

    pred_str = str(prediction).strip()
    score = int(pred_str.lower() == expected.lower())

    if score == 1:
        feedback = f"Correct! Intent is '{expected}'."
    else:
        feedback = (
            f"Wrong. Predicted '{pred_str}', correct is '{expected}'. "
            f"Message: \"{example.inputs[:100]}\""
        )
    return {"score": score, "feedback": feedback}


# ── 4. Evaluate unoptimised baseline ───────────────────────────────
print("\n" + "=" * 60)
print("Evaluating UNOPTIMISED baseline on test set...")
print("=" * 60)

evaluate = Evaluate(
    devset=testset,
    metric=metric,
    num_threads=4,
    return_all_scores=True,
)

baseline = evaluate(program)
print(f"\nBaseline accuracy: {baseline.score:.2%}")

# Show some examples
correct = sum(1 for _, _, s in baseline.results if s >= 1.0)
wrong = [(ex, pred, s) for ex, pred, s in baseline.results if s < 1.0]
print(f"  Correct: {correct}/{len(baseline.results)}")
if wrong:
    print("  Sample errors:")
    for ex, pred, _ in wrong[:5]:
        print(f"    '{ex.inputs[:50]}...' -> '{pred}' (expected '{ex.labels}')")


# ── 5. Reflection LM ───────────────────────────────────────────────
reflection_agent = Agent(
    name="reflection_lm",
    model=mf.Model.chat_completion(MODEL_ID, max_tokens=2000, temperature=0.7),
    system_message="You are a prompt optimisation expert.",
    instructions=(
        "Given the current instruction and misclassification examples, "
        "propose an improved instruction for a banking intent classifier."
    ),
)


def reflection_lm(prompt: str) -> str:
    return reflection_agent(prompt)


# ── 6. Optimise with GEPA ──────────────────────────────────────────
print("\n" + "=" * 60)
print("Running GEPA optimisation (3 iterations)...")
print("=" * 60)

optimizer = GEPA(
    metric=metric_with_feedback,
    reflection_lm=reflection_lm,
    max_iterations=3,
    num_candidates=2,
    num_threads=4,
)

optimised = optimizer.compile(
    program,
    trainset=trainset,
    valset=valset,
)

print(f"\nOptimisation complete!")
print(f"Compile info: {optimised.get_compile_info()}")


# ── 7. Inspect optimised prompt ────────────────────────────────────
print("\n" + "=" * 60)
print("Optimised Prompt")
print("=" * 60)

for name, agent in optimised.named_agents():
    opt = agent.optimized_system_prompt.data
    if opt:
        print(f"\n[{name}] optimised system prompt:\n{opt[:1000]}")
    else:
        print(f"\n[{name}] (no optimised prompt)")


# ── 8. Evaluate optimised program ──────────────────────────────────
print("\n" + "=" * 60)
print("Evaluating OPTIMISED program on test set...")
print("=" * 60)

optimised_result = evaluate(optimised)
print(f"\nOptimised accuracy: {optimised_result.score:.2%}")

correct = sum(1 for _, _, s in optimised_result.results if s >= 1.0)
wrong = [(ex, pred, s) for ex, pred, s in optimised_result.results if s < 1.0]
print(f"  Correct: {correct}/{len(optimised_result.results)}")
if wrong:
    print("  Sample errors:")
    for ex, pred, _ in wrong[:5]:
        print(f"    '{ex.inputs[:50]}...' -> '{pred}' (expected '{ex.labels}')")


# ── 9. Comparison ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"  Baseline:  {baseline.score:.2%}")
print(f"  GEPA:      {optimised_result.score:.2%}")
diff = optimised_result.score - baseline.score
if diff > 0:
    print(f"  Improvement: +{diff:.2%}")
elif diff == 0:
    print("  No change.")
else:
    print(f"  Regression: {diff:.2%}")
