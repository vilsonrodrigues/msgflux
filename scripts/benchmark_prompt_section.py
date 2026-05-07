"""Toy benchmark for free-text prompts vs PromptSection prompts.

This is intentionally small and informal. It runs semantically equivalent agents
with free-text prompt fields and PromptSection prompt fields, then writes JSONL
results with simple deterministic scores.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import msgspec
from msgspec_ext import load_dotenv

import msgflux as mf
from msgflux import nn

PromptMode = Literal["free_text", "prompt_section"]
OutputMode = Literal["text", "structured"]


class TriageOutput(msgspec.Struct):
    category: str
    priority: str
    reply: str


@dataclass(frozen=True)
class Case:
    name: str
    task: str
    expected_category: str
    expected_priority: str
    required_terms: tuple[str, ...]


CASES = [
    Case(
        name="billing_after_renewal",
        task=(
            "Customer says: 'I renewed my annual subscription yesterday, but my "
            "account still says payment required and I cannot export invoices.'"
        ),
        expected_category="billing",
        expected_priority="medium",
        required_terms=("subscription", "invoice"),
    ),
    Case(
        name="production_outage",
        task=(
            "Customer says: 'Our checkout API has returned 502 for all users for "
            "18 minutes. We are losing orders right now.'"
        ),
        expected_category="technical",
        expected_priority="high",
        required_terms=("checkout", "502"),
    ),
    Case(
        name="feature_request",
        task=(
            "Customer says: 'Could you add CSV export to the analytics dashboard? "
            "This is not blocking us, but it would help our weekly reporting.'"
        ),
        expected_category="product",
        expected_priority="low",
        required_terms=("CSV", "analytics"),
    ),
]


def build_agent(
    *,
    model_name: str,
    prompt_mode: PromptMode,
    output_mode: OutputMode,
):
    model = mf.Model.chat_completion(model_name, temperature=0)

    if prompt_mode == "free_text":
        system_message = (
            "You are a customer support triage specialist. Be concise, factual, "
            "and operationally useful."
        )
        instructions = (
            "Classify the customer message into one category: billing, technical, "
            "product, account, or other. Set priority to low, medium, or high. "
            "Write a short reply that mentions the relevant customer details."
        )
        expected_output = (
            "For text output, return exactly three lines: Category, Priority, "
            "Reply. For structured output, fill all schema fields."
        )
    else:
        system_message = mf.PromptSection(
            role="customer support triage specialist",
            style="concise, factual, operational",
        )
        instructions = mf.PromptSection(
            task="classify customer message",
            categories=["billing", "technical", "product", "account", "other"],
            priority=["low", "medium", "high"],
            reply="short; mention relevant customer details",
        )
        expected_output = mf.PromptSection(
            text="exactly three lines: Category, Priority, Reply",
            structured="fill all schema fields",
        )

    class TriageAgent(nn.Agent):
        pass

    kwargs: dict[str, Any] = {
        "name": f"triage_{prompt_mode}_{output_mode}",
        "model": model,
        "system_message": system_message,
        "instructions": instructions,
        "expected_output": expected_output,
    }
    if output_mode == "structured":
        kwargs["generation_schema"] = TriageOutput
    return TriageAgent(**kwargs)


def normalize_output(output: Any) -> dict[str, str]:
    if isinstance(output, TriageOutput):
        return {
            "category": output.category,
            "priority": output.priority,
            "reply": output.reply,
        }
    text = str(output)
    lowered = text.lower()
    category = ""
    priority = ""
    for line in text.splitlines():
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == "category":
            category = value
        elif key == "priority":
            priority = value
    return {
        "category": category or lowered,
        "priority": priority or lowered,
        "reply": text,
    }


def score(case: Case, output: Any) -> dict[str, Any]:
    normalized = normalize_output(output)
    reply = normalized["reply"].lower()
    category_ok = case.expected_category in normalized["category"].lower()
    priority_ok = case.expected_priority in normalized["priority"].lower()
    terms_found = [term for term in case.required_terms if term.lower() in reply]
    return {
        "category_ok": category_ok,
        "priority_ok": priority_ok,
        "required_terms_found": terms_found,
        "required_terms_total": len(case.required_terms),
        "score": int(category_ok)
        + int(priority_ok)
        + len(terms_found) / len(case.required_terms),
    }


def metadata_to_dict(metadata: Any) -> Any:
    if metadata is None:
        return None
    if hasattr(metadata, "to_dict"):
        return metadata.to_dict()
    if isinstance(metadata, dict):
        return metadata
    return repr(metadata)


def run_case(
    *,
    model_name: str,
    prompt_mode: PromptMode,
    output_mode: OutputMode,
    case: Case,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = {
        "model": model_name,
        "prompt_mode": prompt_mode,
        "output_mode": output_mode,
        "case": case.name,
    }
    try:
        agent = build_agent(
            model_name=model_name,
            prompt_mode=prompt_mode,
            output_mode=output_mode,
        )
        response = agent(case.task)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        output = response.consume() if hasattr(response, "consume") else response
        result.update(
            {
                "latency_ms": elapsed_ms,
                "output": msgspec.to_builtins(output),
                "metrics": score(case, output),
                "metadata": metadata_to_dict(getattr(response, "metadata", None)),
                "error": None,
            }
        )
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        result.update(
            {
                "latency_ms": elapsed_ms,
                "output": None,
                "metrics": {
                    "category_ok": False,
                    "priority_ok": False,
                    "required_terms_found": [],
                    "required_terms_total": len(case.required_terms),
                    "score": 0,
                },
                "metadata": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["openai/gpt-4.1-mini", "groq/openai/gpt-oss-20b"],
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tmp/prompt_section_benchmark.jsonl"),
    )
    args = parser.parse_args()

    load_dotenv()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with args.output.open("w", encoding="utf-8") as f:
        for run_idx in range(args.runs):
            for model_name in args.models:
                for prompt_mode in ("free_text", "prompt_section"):
                    for output_mode in ("text", "structured"):
                        for case in CASES:
                            result = run_case(
                                model_name=model_name,
                                prompt_mode=prompt_mode,
                                output_mode=output_mode,
                                case=case,
                            )
                            result["run"] = run_idx
                            results.append(result)
                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                            f.flush()
                            status = "error" if result["error"] else result["metrics"]
                            print(  # noqa: T201
                                model_name, prompt_mode, output_mode, case.name, status
                            )

    summary: dict[str, list[float]] = {}
    for result in results:
        key = "|".join([result["model"], result["prompt_mode"], result["output_mode"]])
        summary.setdefault(key, []).append(result["metrics"]["score"])

    print("\nSummary")  # noqa: T201
    for key, scores in sorted(summary.items()):
        print(  # noqa: T201
            f"{key}: {sum(scores) / len(scores):.3f} avg over {len(scores)} cases"
        )
    print(f"\nWrote {args.output}")  # noqa: T201


if __name__ == "__main__":
    main()
