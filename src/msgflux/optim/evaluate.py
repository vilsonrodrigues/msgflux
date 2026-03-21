"""Evaluate - run a program against a dataset and aggregate scores."""

import csv
import importlib
import inspect
import json
import logging
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, List, Tuple

import tqdm

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional for display_table
    pd = None  # type: ignore[assignment]

try:
    from IPython import get_ipython
    from IPython.display import display
except ImportError:  # pragma: no cover - notebook-only nicety
    get_ipython = None  # type: ignore[assignment]
    display = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Container for evaluation results."""

    score: float
    results: List[Tuple[Any, Any, float]] = field(default_factory=list)
    num_errors: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"EvalResult(score={self.score:.2f}%, "
            f"n={len(self.results)}, errors={self.num_errors})"
        )

    def get_successes(self) -> List[Tuple[Any, Any, float]]:
        """Return examples where the score was positive."""
        return [(ex, pred, score) for ex, pred, score in self.results if score > 0]

    def get_failures(self) -> List[Tuple[Any, Any, float]]:
        """Return examples where the score was zero or negative."""
        return [(ex, pred, score) for ex, pred, score in self.results if score <= 0]

    def to_dict(self) -> dict[str, Any]:
        """Convert result metadata into a dictionary."""
        return {
            "score": self.score,
            "num_examples": len(self.results),
            "num_errors": self.num_errors,
            "num_successes": len(self.get_successes()),
            "num_failures": len(self.get_failures()),
            "metadata": self.metadata,
        }


class Evaluate:
    """Evaluate a program on a dataset using a metric function.

    Example::

        evaluator = Evaluate(devset=examples, metric=exact_match)
        result = evaluator(program)
        print(result.score)

    Args:
        devset: Default evaluation dataset.
        metric: ``metric(example, prediction) -> float | bool``.
        num_threads: Number of threads for parallel evaluation.
        display_progress: Show a tqdm progress bar.
        display_table: Show a pandas table of the results.
        max_errors: Maximum errors before aborting.
        failure_score: Score assigned when a program raises an exception.
        save_as_csv: Optional CSV path to save results.
        save_as_json: Optional JSON path to save results.
        return_all_scores: Deprecated. Results are always populated.
    """

    def __init__(
        self,
        *,
        devset: List[Any],
        metric: Callable,
        num_threads: int | None = None,
        display_progress: bool = False,
        display_table: bool | int = False,
        max_errors: int | None = None,
        failure_score: float = 0.0,
        save_as_csv: str | None = None,
        save_as_json: str | None = None,
        return_all_scores: bool = False,
    ):
        self.devset = devset
        self.metric = metric
        self.num_threads = num_threads or 1
        self.display_progress = display_progress
        self.display_table = display_table
        self.max_errors = max_errors
        self.failure_score = failure_score
        self.save_as_csv = save_as_csv
        self.save_as_json = save_as_json
        self.return_all_scores = return_all_scores

    def __call__(
        self,
        program: Any,
        *,
        metric: Callable | None = None,
        devset: List[Any] | None = None,
        num_threads: int | None = None,
        display_progress: bool | None = None,
        display_table: bool | int | None = None,
        save_as_csv: str | None = None,
        save_as_json: str | None = None,
    ) -> EvalResult:
        """Run *program* on every example in *devset* and return the result."""
        metric = metric if metric is not None else self.metric
        devset = devset if devset is not None else self.devset
        num_threads = num_threads if num_threads is not None else self.num_threads
        display_progress = (
            display_progress
            if display_progress is not None
            else self.display_progress
        )
        display_table = (
            display_table if display_table is not None else self.display_table
        )
        save_as_csv = save_as_csv if save_as_csv is not None else self.save_as_csv
        save_as_json = (
            save_as_json if save_as_json is not None else self.save_as_json
        )

        if not devset:
            return EvalResult(score=0.0, results=[], num_errors=0, metadata={})

        was_training = _capture_training_mode(program)
        _set_eval_mode(program)

        try:
            if num_threads <= 1:
                results = self._run_sequential(
                    program=program,
                    devset=devset,
                    metric=metric,
                    display_progress=display_progress,
                )
            else:
                results = self._run_parallel(
                    program=program,
                    devset=devset,
                    metric=metric,
                    num_threads=num_threads,
                    display_progress=display_progress,
                )
        finally:
            _restore_training_mode(program, was_training=was_training)

        total_score = sum(result[2] for result in results)
        avg_score = (100 * total_score / len(results)) if results else 0.0
        metric_name = _metric_name(metric)

        logger.info(
            "Average Metric: %s / %s (%s%%)",
            total_score,
            len(results),
            round(avg_score, 1),
        )

        if display_table:
            if importlib.util.find_spec("pandas") is not None:
                result_df = self._construct_result_table(results, metric_name)
                self._display_result_table(
                    result_df,
                    display_table=display_table,
                )
            else:
                logger.warning(
                    "Skipping table display since `pandas` is not installed."
                )

        if save_as_csv:
            self._write_csv(save_as_csv, results, metric_name)

        if save_as_json:
            self._write_json(save_as_json, results, metric_name)

        num_errors = sum(1 for _, prediction, _ in results if prediction is None)

        return EvalResult(
            score=round(avg_score, 2),
            results=results,
            num_errors=num_errors,
            metadata={
                "num_examples": len(devset),
                "num_evaluated": len(results),
                "metric_name": metric_name,
            },
        )

    def _evaluate_one(
        self,
        program: Any,
        example: Any,
        metric: Callable,
    ) -> Tuple[Any, Any, float]:
        try:
            prediction = _invoke_program(program, example)
            score = _call_metric(metric, example, prediction)
            if not isinstance(score, (int, float)):
                score = float(score)
            return (example, prediction, score)
        except Exception:
            logger.error("Error evaluating example", exc_info=True)
            return (example, None, self.failure_score)

    def _run_sequential(
        self,
        program: Any,
        devset: List[Any],
        metric: Callable,
        *,
        display_progress: bool,
    ) -> List[Tuple[Any, Any, float]]:
        results: List[Tuple[Any, Any, float]] = []
        error_count = 0
        iterator = (
            tqdm.tqdm(devset, disable=not display_progress)
            if display_progress
            else devset
        )
        for example in iterator:
            result = self._evaluate_one(program, example, metric)
            results.append(result)
            if result[1] is None:
                error_count += 1
            if self.max_errors and error_count >= self.max_errors:
                break
        return results

    def _run_parallel(
        self,
        program: Any,
        devset: List[Any],
        metric: Callable,
        num_threads: int,
        *,
        display_progress: bool,
    ) -> List[Tuple[Any, Any, float]]:
        results_by_index: dict[int, Tuple[Any, Any, float]] = {}
        error_count = 0

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {
                executor.submit(self._evaluate_one, program, example, metric): idx
                for idx, example in enumerate(devset)
            }
            pending = set(futures)
            progress = tqdm.tqdm(total=len(futures), disable=not display_progress)
            try:
                while pending:
                    done, pending = wait(
                        pending,
                        return_when=FIRST_COMPLETED,
                    )
                    for future in done:
                        idx = futures[future]
                        result = future.result()
                        results_by_index[idx] = result
                        progress.update(1)
                        if result[1] is None:
                            error_count += 1
                        if self.max_errors and error_count >= self.max_errors:
                            for pending_future in pending:
                                pending_future.cancel()
                            pending.clear()
                            break
            finally:
                progress.close()

        return [results_by_index[idx] for idx in sorted(results_by_index)]

    @staticmethod
    def _prepare_results_output(
        results: List[Tuple[Any, Any, float]],
        metric_name: str,
    ) -> List[dict[str, Any]]:
        return [
            (
                _merge_dicts(example, prediction) | {metric_name: score}
                if _is_dictlike(prediction)
                else _object_to_dict(example)
                | {
                    "prediction": _serialize_scalar(prediction),
                    metric_name: score,
                }
            )
            for example, prediction, score in results
        ]

    def _construct_result_table(
        self,
        results: List[Tuple[Any, Any, float]],
        metric_name: str,
    ) -> pd.DataFrame:
        if pd is None:  # pragma: no cover - guarded by find_spec/checks
            raise RuntimeError("pandas is required to construct result tables.")
        data = self._prepare_results_output(results, metric_name)
        result_df = pd.DataFrame(data)
        if hasattr(result_df, "map"):
            result_df = result_df.map(_truncate_cell)
        else:
            result_df = result_df.applymap(_truncate_cell)
        return result_df

    def _display_result_table(
        self,
        result_df: pd.DataFrame,
        *,
        display_table: bool | int,
    ) -> None:
        if isinstance(display_table, bool):
            df_to_display = result_df.copy()
        else:
            df_to_display = result_df.head(display_table).copy()
        _display_dataframe(df_to_display)

    def _write_csv(
        self,
        path: str,
        results: List[Tuple[Any, Any, float]],
        metric_name: str,
    ) -> None:
        data = self._prepare_results_output(results, metric_name)
        with open(path, "w", newline="", encoding="utf-8") as csvfile:
            if not data:
                csvfile.write("")
                return
            writer = csv.DictWriter(csvfile, fieldnames=list(data[0].keys()))
            writer.writeheader()
            for row in data:
                writer.writerow(row)

    def _write_json(
        self,
        path: str,
        results: List[Tuple[Any, Any, float]],
        metric_name: str,
    ) -> None:
        data = self._prepare_results_output(results, metric_name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, default=str)


def _invoke_program(program: Any, example: Any) -> Any:
    inputs = example.inputs if hasattr(example, "inputs") else example
    if isinstance(inputs, Mapping):
        return program(**inputs)
    return program(inputs)


def _call_metric(metric: Callable, example: Any, prediction: Any) -> Any:
    if _metric_accepts_trace(metric):
        return metric(example, prediction, None)
    return metric(example, prediction)


def _metric_accepts_trace(metric: Callable) -> bool:
    try:
        signature = inspect.signature(metric)
    except (TypeError, ValueError):
        return False

    if any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    ):
        return True

    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    return len(positional) >= 3


def _metric_name(metric: Callable) -> str:
    return getattr(metric, "__name__", metric.__class__.__name__)


def _is_dictlike(value: Any) -> bool:
    return hasattr(value, "items") and callable(value.items)


def _object_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): _serialize_scalar(val) for key, val in value.items()}
    if is_dataclass(value):
        return {
            key: _serialize_scalar(val)
            for key, val in asdict(value).items()
            if not key.startswith("_")
        }
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return {
            str(key): _serialize_scalar(val)
            for key, val in value.model_dump().items()
        }
    if hasattr(value, "dict") and callable(value.dict):
        return {
            str(key): _serialize_scalar(val)
            for key, val in value.dict().items()
        }
    if hasattr(value, "toDict") and callable(value.toDict):
        return {
            str(key): _serialize_scalar(val)
            for key, val in value.toDict().items()
        }
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return {
            str(key): _serialize_scalar(val)
            for key, val in value.to_dict().items()
        }
    if hasattr(value, "__dict__"):
        return {
            str(key): _serialize_scalar(val)
            for key, val in vars(value).items()
            if not key.startswith("_")
        }
    return {"value": _serialize_scalar(value)}


def _merge_dicts(example: Any, prediction: Any) -> dict[str, Any]:
    example_dict = _object_to_dict(example)
    prediction_dict = _object_to_dict(prediction)

    merged: dict[str, Any] = {}
    for key, value in example_dict.items():
        if key in prediction_dict:
            merged[f"example_{key}"] = value
        else:
            merged[key] = value

    for key, value in prediction_dict.items():
        if key in example_dict:
            merged[f"pred_{key}"] = value
        else:
            merged[key] = value

    return merged


def _serialize_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_serialize_scalar(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize_scalar(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _serialize_scalar(val) for key, val in value.items()}
    if is_dataclass(value) or hasattr(value, "__dict__"):
        return _object_to_dict(value)
    return str(value)


def _truncate_cell(content: Any) -> Any:
    words = str(content).split()
    if len(words) > 25:
        return " ".join(words[:25]) + "..."
    return content


def _display_dataframe(df: pd.DataFrame) -> None:
    if pd is None:  # pragma: no cover - guarded by find_spec/checks
        raise RuntimeError("pandas is required to display result tables.")
    if _is_in_ipython_notebook_environment():
        if display is not None:
            display(_configure_dataframe_for_ipython_notebook_display(df))
        return

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        logger.info("\n%s", df.to_string())


def _configure_dataframe_for_ipython_notebook_display(
    df: pd.DataFrame,
) -> pd.DataFrame:
    if pd is None:  # pragma: no cover - guarded by find_spec/checks
        raise RuntimeError("pandas is required to configure result tables.")
    pd.options.display.max_colwidth = 70
    return df


def _is_in_ipython_notebook_environment() -> bool:
    if get_ipython is None:
        return False
    return "IPKernelApp" in getattr(get_ipython(), "config", {})


def _capture_training_mode(program: Any) -> bool | None:
    if hasattr(program, "training"):
        return bool(program.training)
    return None


def _set_eval_mode(program: Any) -> None:
    if hasattr(program, "eval") and callable(program.eval):
        program.eval()


def _restore_training_mode(
    program: Any,
    *,
    was_training: bool | None,
) -> None:
    if was_training and hasattr(program, "train") and callable(program.train):
        program.train()
