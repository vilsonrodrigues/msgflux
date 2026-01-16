"""Utility functions for the SIMBA optimizer.

This module provides helper functions for the SIMBA (Stochastic Introspective
Mini-Batch Ascent) optimizer, including strategies for appending demos and rules,
program wrapping, and model preparation.
"""

import inspect
import json
import logging
import textwrap
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from msgflux.examples import Example
from msgflux.nn.modules.module import Module

logger = logging.getLogger(__name__)


@dataclass
class TrajectoryStep:
    """Represents a single step in an execution trajectory."""

    module_name: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]


@dataclass
class ExecutionResult:
    """Result of executing a wrapped program."""

    prediction: Any
    trace: List[TrajectoryStep]
    score: float
    example: Example
    output_metadata: Dict[str, Any]


def wrap_program(
    program: Module,
    metric: Callable[[Example, Any], float],
) -> Callable[[Example], ExecutionResult]:
    """Wrap a program to capture execution traces and compute metrics.

    This wrapper executes the program, captures the execution trace,
    and computes the metric score for the prediction.

    Args:
        program: The module to wrap.
        metric: Function that takes (example, prediction) and returns a score.

    Returns:
        A wrapped function that returns ExecutionResult.
    """

    def wrapped_program(example: Example) -> ExecutionResult:
        trace: List[TrajectoryStep] = []
        prediction = None
        score = 0.0
        output_metadata: Dict[str, Any] = {}

        # Create hook to capture trace
        def trace_hook(module, args, kwargs, output):
            module_name = getattr(module, "_name", None) or module.__class__.__name__
            inputs = {}
            if args:
                inputs["input"] = args[0] if len(args) == 1 else args
            inputs.update(kwargs)

            if isinstance(output, dict):
                outputs = output
            elif isinstance(output, str):
                outputs = {"output": output}
            else:
                outputs = {"output": str(output)}

            trace.append(TrajectoryStep(
                module_name=module_name,
                inputs=inputs,
                outputs=outputs,
            ))

        # Register hooks
        handles = []
        try:
            for name, mod in program.named_modules():
                if hasattr(mod, "register_forward_hook"):
                    try:
                        h = mod.register_forward_hook(trace_hook)
                        handles.append(h)
                    except Exception:
                        pass

            # Execute program
            try:
                prediction = program(example.inputs)
            except Exception as e:
                logger.warning(f"Program execution error: {e}")

        finally:
            # Remove hooks
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass

        # Compute metric
        try:
            output = metric(example, prediction)
            if isinstance(output, (int, float)):
                score = float(output)
            elif isinstance(output, dict):
                score = output.get("score", 0.0)
                output_metadata = {k: v for k, v in output.items() if k != "score"}
            else:
                score = float(output) if output else 0.0
        except Exception as e:
            logger.warning(f"Metric computation error: {e}")

        return ExecutionResult(
            prediction=prediction,
            trace=trace,
            score=score,
            example=example,
            output_metadata=output_metadata,
        )

    return wrapped_program


def prepare_models_for_resampling(
    program: Module,
    n: int,
    teacher_settings: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """Prepare multiple model configurations for trajectory sampling.

    Creates n different model configurations with varying rollout IDs
    and temperatures for diverse sampling.

    Args:
        program: The base program to get the model from.
        n: Number of model configurations to create.
        teacher_settings: Optional settings for teacher model.

    Returns:
        List of model configurations.
    """
    # Get the base model from the program
    model = None
    if hasattr(program, "model"):
        model = program.model
    elif hasattr(program, "get_model"):
        model = program.get_model()

    if model is None:
        # Return a list of None to indicate using default model
        return [None] * n

    models = []
    start_idx = 0

    # If teacher settings provided, use as first model
    if teacher_settings:
        teacher_model = teacher_settings.get("model", model)
        models.append(teacher_model)
        start_idx = 1

    # Create copies with different configurations
    for i in range(start_idx, n):
        try:
            model_copy = deepcopy(model)
            # Set different temperature for diversity
            if hasattr(model_copy, "temperature"):
                model_copy.temperature = 1.0
            elif hasattr(model_copy, "kwargs"):
                model_copy.kwargs["temperature"] = 1.0
            models.append(model_copy)
        except Exception:
            models.append(model)

    return models


def append_a_demo(demo_input_field_maxlen: int = 100_000) -> Callable:
    """Create a strategy function that appends successful demos.

    This strategy takes a successful execution trace and adds it
    as a demonstration to the predictors.

    Args:
        demo_input_field_maxlen: Maximum characters for input fields.

    Returns:
        A strategy function that appends demos.
    """

    def append_a_demo_(
        bucket: List[ExecutionResult],
        system: Module,
        *,
        predictor2name: Dict[int, str],
        name2predictor: Dict[str, Any],
        batch_10p_score: float,
        batch_90p_score: float,
        prompt_model: Optional[Any] = None,
        **kwargs,
    ) -> bool:
        """Append a successful execution as a demonstration.

        Args:
            bucket: List of execution results sorted by score (best first).
            system: The program/module to modify.
            predictor2name: Mapping from predictor id to name.
            name2predictor: Mapping from name to predictor.
            batch_10p_score: 10th percentile score of the batch.
            batch_90p_score: 90th percentile score of the batch.
            prompt_model: Model for generating prompts (unused in this strategy).

        Returns:
            True if demo was successfully appended, False otherwise.
        """
        good = bucket[0]
        trace = good.trace

        if good.score <= batch_10p_score:
            logger.info(
                f"Skipping appending demo: good score {good.score} "
                f"is at or below 10th percentile {batch_10p_score}"
            )
            return False

        name2demo: Dict[str, Example] = {}

        for step in trace:
            # Truncate long inputs
            inputs = {}
            for k, v in step.inputs.items():
                v_str = str(v)
                if len(v_str) > demo_input_field_maxlen:
                    v_str = f"{v_str[:demo_input_field_maxlen]}\n... <TRUNCATED>"
                inputs[k] = v_str

            # Create demo example
            demo = Example(
                inputs=inputs.get("input", inputs),
                labels=step.outputs.get("output", step.outputs),
            )
            name2demo[step.module_name] = demo

        # Append demos to predictors
        for name, demo in name2demo.items():
            if name in name2predictor:
                predictor = name2predictor[name]
                if hasattr(predictor, "demos"):
                    predictor.demos.append(demo)
                elif hasattr(predictor, "_demos"):
                    predictor._demos.append(demo)

        logger.info(f"Added {len(name2demo)} demos across predictors.")
        return True

    append_a_demo_.__name__ = "append_a_demo"
    return append_a_demo_


def append_a_rule(
    bucket: List[ExecutionResult],
    system: Module,
    *,
    predictor2name: Dict[int, str],
    name2predictor: Dict[str, Any],
    batch_10p_score: float,
    batch_90p_score: float,
    prompt_model: Optional[Any] = None,
    **kwargs,
) -> bool:
    """Append a reflective rule based on trajectory comparison.

    This strategy analyzes the difference between successful and
    unsuccessful executions and generates improvement rules.

    Args:
        bucket: List of execution results sorted by score (best first).
        system: The program/module to modify.
        predictor2name: Mapping from predictor id to name.
        name2predictor: Mapping from name to predictor.
        batch_10p_score: 10th percentile score of the batch.
        batch_90p_score: 90th percentile score of the batch.
        prompt_model: Model for generating reflective rules.

    Returns:
        True if rule was successfully appended, False otherwise.
    """
    if prompt_model is None:
        logger.warning("No prompt_model provided for append_a_rule strategy")
        return False

    good, bad = bucket[0], bucket[-1]
    example = good.example

    # Skip if scores don't show enough contrast
    if good.score <= batch_10p_score or bad.score >= batch_90p_score:
        logger.info(
            f"Skipping rule generation: good score {good.score} at/below 10th pct "
            f"or bad score {bad.score} at/above 90th pct"
        )
        return False

    if good.score <= bad.score:
        # Handle edge case where good isn't actually better
        if good.score > batch_90p_score:
            bad = ExecutionResult(
                prediction=None,
                trace=[],
                score=0.0,
                example=example,
                output_metadata={},
            )
        else:
            good = ExecutionResult(
                prediction=None,
                trace=[],
                score=0.0,
                example=example,
                output_metadata={},
            )

    # Format trajectories
    better_trajectory = _format_trajectory(good.trace, predictor2name)
    worse_trajectory = _format_trajectory(bad.trace, predictor2name)

    # Get module names
    module_names = list(name2predictor.keys())

    # Build reflection prompt
    prompt = _build_reflection_prompt(
        system=system,
        example=example,
        better_trajectory=better_trajectory,
        worse_trajectory=worse_trajectory,
        better_outputs=good.prediction,
        worse_outputs=bad.prediction,
        better_score=good.score,
        worse_score=bad.score,
        better_metadata=good.output_metadata,
        worse_metadata=bad.output_metadata,
        module_names=module_names,
    )

    # Generate advice
    try:
        advice = _generate_advice(prompt_model, prompt, module_names)

        # Apply advice to predictors
        for name, predictor in name2predictor.items():
            if name in advice:
                logger.info(f"Advice for {name}: {advice[name]}")
                _apply_advice(predictor, advice[name])

        return True
    except Exception as e:
        logger.error(f"Rule generation failed: {e}")
        return False


append_a_rule.__name__ = "append_a_rule"


def _format_trajectory(
    trace: List[TrajectoryStep],
    predictor2name: Dict[int, str],
) -> List[Dict[str, Any]]:
    """Format trajectory for reflection."""
    return [
        {
            "module_name": step.module_name,
            "inputs": step.inputs,
            "outputs": step.outputs,
        }
        for step in trace
    ]


def _build_reflection_prompt(
    system: Module,
    example: Example,
    better_trajectory: List[Dict[str, Any]],
    worse_trajectory: List[Dict[str, Any]],
    better_outputs: Any,
    worse_outputs: Any,
    better_score: float,
    worse_score: float,
    better_metadata: Dict[str, Any],
    worse_metadata: Dict[str, Any],
    module_names: List[str],
) -> str:
    """Build the reflection prompt for rule generation."""
    # Get program source if available
    try:
        program_code = inspect.getsource(system.__class__)
    except Exception:
        program_code = f"class {system.__class__.__name__}: ..."

    # Format module definitions
    modules_defn = _inspect_modules(system)

    prompt = f"""You are analyzing two executions of an LLM program to help improve its performance.

## Program Code
{program_code}

## Module Definitions
{modules_defn}

## Input
{_safe_serialize(example.inputs)}

## Expected Output/Labels
{_safe_serialize(example.labels)}

## Better Execution (Score: {better_score})
Trajectory:
{_safe_serialize(better_trajectory)}

Output:
{_safe_serialize(better_outputs)}

Metadata:
{_safe_serialize(better_metadata)}

## Worse Execution (Score: {worse_score})
Trajectory:
{_safe_serialize(worse_trajectory)}

Output:
{_safe_serialize(worse_outputs)}

Metadata:
{_safe_serialize(worse_metadata)}

## Task
For each module ({', '.join(module_names)}), provide concrete advice on how it should behave
to achieve results more like the better execution. Be specific and actionable.

Format your response as JSON:
{{
  "module_name": "If the module receives [pattern], then it should [action/strategy].",
  ...
}}
"""
    return prompt


def _inspect_modules(program: Module) -> str:
    """Inspect module definitions for documentation."""
    lines = ["-" * 80]

    for name, mod in program.named_modules():
        if mod is program:
            continue

        lines.append(f"Module: {name}")
        lines.append(f"  Type: {mod.__class__.__name__}")

        # Get instructions if available
        if hasattr(mod, "instructions"):
            instr = mod.instructions
            if instr:
                lines.append(f"  Instructions: {textwrap.shorten(instr, width=200)}")

        # Get signature info if available
        if hasattr(mod, "signature"):
            sig = mod.signature
            if hasattr(sig, "input_fields"):
                lines.append(f"  Input Fields: {list(sig.input_fields.keys())}")
            if hasattr(sig, "output_fields"):
                lines.append(f"  Output Fields: {list(sig.output_fields.keys())}")

        lines.append("-" * 80)

    return "\n".join(lines)


def _safe_serialize(obj: Any) -> str:
    """Safely serialize an object to JSON string."""
    try:
        return json.dumps(obj, indent=2, default=str)
    except Exception:
        return str(obj)


def _generate_advice(
    prompt_model: Any,
    prompt: str,
    module_names: List[str],
) -> Dict[str, str]:
    """Generate advice using the prompt model."""
    # Call the model
    if hasattr(prompt_model, "__call__"):
        response = prompt_model(prompt)
    elif hasattr(prompt_model, "generate"):
        response = prompt_model.generate(prompt)
    else:
        raise ValueError("prompt_model must be callable or have a generate method")

    # Extract response text
    if hasattr(response, "data"):
        text = response.data
    elif hasattr(response, "text"):
        text = response.text
    elif isinstance(response, str):
        text = response
    else:
        text = str(response)

    # Parse JSON from response
    try:
        # Try to find JSON in the response
        import re

        json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if json_match:
            advice = json.loads(json_match.group())
            return {k: v for k, v in advice.items() if isinstance(v, str)}
    except Exception as e:
        logger.warning(f"Failed to parse advice JSON: {e}")

    # Return empty advice if parsing fails
    return {}


def _apply_advice(predictor: Any, advice: str) -> None:
    """Apply advice to a predictor by updating its instructions."""
    if hasattr(predictor, "signature"):
        sig = predictor.signature
        if hasattr(sig, "instructions"):
            current = sig.instructions or ""
            new_instructions = f"{current}\n\n{advice}".strip()
            if hasattr(sig, "with_instructions"):
                predictor.signature = sig.with_instructions(new_instructions)
            else:
                sig.instructions = new_instructions
    elif hasattr(predictor, "instructions"):
        current = predictor.instructions or ""
        predictor.instructions = f"{current}\n\n{advice}".strip()


def recursive_mask(obj: Any) -> Any:
    """Recursively mask non-serializable objects."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        pass

    if isinstance(obj, dict):
        return {k: recursive_mask(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_mask(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_mask(v) for v in obj)
    else:
        return f"<non-serializable: {type(obj).__name__}>"
