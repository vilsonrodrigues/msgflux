"""Compare three tool compositions on an independently graded coding task.

The offline preflight makes no provider calls. Live runs execute generated code
with local workspace tools and consume the configured provider account.
"""

# Embedded prompts and fixture source are byte-preserved for protocol hashes.
# ruff: noqa: E501

import argparse
import asyncio
import hashlib
import inspect
import json
import random
import subprocess
import sys
import time
from collections import Counter
from copy import deepcopy
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from pathlib import Path
from uuid import uuid4

import msgspec

import msgflux as mf
from msgflux.nn import Agent, ToolTurnLimitExtension
from msgflux.nn.hooks import Hook
from msgflux.runtime import ExecutionScope
from msgflux.runtime.workspace.api import AgentWorkspace
from msgflux.tools.builtin import (
    BashTool,
    EditTool,
    GlobTool,
    GrepTool,
    LsTool,
    MontyTool,
    MontyToolBucket,
    ReadFileTool,
    WriteTool,
)

ROOT: Path
MODEL = "opencode-go/deepseek-v4.1-flash"
TASK = """Fix summarize_orders(events) in reconcile.py using only the Python standard library.

Contract:
* events is a list. Ignore non-dict rows and invalid envelopes. An envelope requires
  id: a nonempty string after stripping; revision: a nonnegative int (bool is invalid);
  status: exactly "paid" or "cancelled". Group by the stripped id.
* Select exactly one winner per id: highest revision; ties use the last occurrence.
  Select BEFORE validating the paid payload. A malformed winner suppresses earlier rows.
* A cancelled winner contributes nothing and needs no paid payload.
* A paid winner needs a nonempty stripped customer string; quantity must be a positive
  int (bool and float invalid); unit_price must be a string accepted by Decimal,
  finite and nonnegative. Invalid payloads are ignored, with no fallback to older rows.
* Round each quantity * unit_price to cents with ROUND_HALF_UP BEFORE summing by customer.
* Return a list sorted by customer ascending, each entry exactly:
  {"customer": str, "order_count": int, "total": str with exactly two decimal places}.
  Zero-price paid orders count. Ignore malformed rows rather than raising.
* Input magnitudes are bounded: absolute integer values <= 1000, finite unit_price
  magnitude <= 1000000, and at most six decimal places. Arbitrarily large exponents
  and integers are outside this task; do not implement support for them.
* Do not mutate the input. The function must be deterministic and perform no I/O.

Inspect the existing implementation and tests, fix it, and run the tests. You may add
regression tests; keep existing tests unchanged. Modify only reconcile.py and new files
under tests/. Work only inside this workspace, do not inspect directories outside it.
Use whichever available tools are most effective. Finish with a concise change and
validation summary. Tests used for final scoring are not present in this workspace.
"""
SYSTEM = """You are a coding agent completing a bounded repository repair task.
Inspect files before changing them. Implement the full supplied contract and validate
with tests. Use tools as needed; avoid printing whole large results. Do not ask questions,
install packages, access network, or edit files outside the workspace. All execution
modes receive this same instruction and task. Once the contract is implemented and
tests pass, finish; avoid speculative optimizations or cases outside the input bounds."""
BUGGY = '''def summarize_orders(events):
    """Aggregate the latest active paid orders by customer."""
    totals = {}
    for event in events:
        if event.get("status") != "paid":
            continue
        customer = event["customer"]
        total = float(event["unit_price"]) * event["quantity"]
        row = totals.setdefault(customer, {"customer": customer, "order_count": 0, "total": 0.0})
        row["order_count"] += 1
        row["total"] += total
    for row in totals.values():
        row["total"] = f"{row['total']:.2f}"
    return list(totals.values())
'''
PUBLIC = """import unittest
from reconcile import summarize_orders

class PublicTests(unittest.TestCase):
    def test_single_order(self):
        row = {"id":"a","revision":0,"status":"paid","customer":"Ana","quantity":2,"unit_price":"17.50"}
        self.assertEqual(summarize_orders([row]), [{"customer":"Ana","order_count":1,"total":"35.00"}])
    def test_cancellation(self):
        rows = [{"id":"a","revision":0,"status":"paid","customer":"Ana","quantity":1,"unit_price":"10.00"},
                {"id":"a","revision":1,"status":"cancelled"}]
        self.assertEqual(summarize_orders(rows), [])
    def test_per_order_rounding(self):
        rows = [{"id":i,"revision":0,"status":"paid","customer":"Ana","quantity":1,"unit_price":"0.005"} for i in ["a","b"]]
        self.assertEqual(summarize_orders(rows), [{"customer":"Ana","order_count":2,"total":"0.02"}])
"""


def reference(events):
    winners = {}
    for row in events:
        if not isinstance(row, dict):
            continue
        key, rev, status = row.get("id"), row.get("revision"), row.get("status")
        if (
            not isinstance(key, str)
            or not key.strip()
            or type(rev) is not int
            or rev < 0
            or status not in ("paid", "cancelled")
        ):
            continue
        key = key.strip()
        if key not in winners or rev >= winners[key]["revision"]:
            winners[key] = row
    totals = {}
    for row in winners.values():
        if row["status"] == "cancelled":
            continue
        customer, quantity, price = (
            row.get("customer"),
            row.get("quantity"),
            row.get("unit_price"),
        )
        if (
            not isinstance(customer, str)
            or not customer.strip()
            or type(quantity) is not int
            or quantity <= 0
            or not isinstance(price, str)
        ):
            continue
        try:
            amount = Decimal(price)
            if not amount.is_finite() or amount < 0:
                continue
            amount = (amount * quantity).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        except (InvalidOperation, ValueError, OverflowError):
            continue
        count, total = totals.get(customer.strip(), (0, Decimal("0")))
        totals[customer.strip()] = (count + 1, total + amount)
    return [
        {"customer": customer, "order_count": count, "total": format(total, ".2f")}
        for customer, (count, total) in sorted(totals.items())
    ]


def cases():
    paid = {
        "id": "a",
        "revision": 0,
        "status": "paid",
        "customer": "Ana",
        "quantity": 1,
        "unit_price": "10.00",
    }
    rows = [
        ("empty", []),
        ("single", [paid]),
        ("invalid_rows", [None, 3, "bad", [], {}]),
    ]
    for key, values in {
        "id": ["", " ", None, 1, " a "],
        "revision": [-1, True, 1.0, None, 2],
        "status": ["unknown", None, "cancelled"],
        "customer": ["", " ", None, " Ana "],
        "quantity": [0, -1, True, 1.0, None, 3],
        "unit_price": [
            -1,
            None,
            "-1",
            "NaN",
            "Infinity",
            "bad",
            "0",
            "0.005",
            "0.004",
            "1e2",
        ],
    }.items():
        for value in values:
            rows.append((f"{key}={value!r}", [{**paid, key: value}]))
    rows += [
        ("revision_out_of_order", [{**paid, "revision": 2, "unit_price": "20"}, paid]),
        ("tie_last_paid", [paid, {**paid, "unit_price": "15"}]),
        (
            "tie_last_cancelled",
            [paid, {"id": "a", "revision": 0, "status": "cancelled"}],
        ),
        (
            "invalid_latest_suppresses",
            [paid, {**paid, "revision": 2, "quantity": True}],
        ),
        (
            "invalid_envelope_does_not_suppress",
            [paid, {**paid, "revision": True, "status": "cancelled"}],
        ),
        (
            "round_before_sum",
            [
                {**paid, "unit_price": "0.005"},
                {**paid, "id": "b", "unit_price": "0.005"},
            ],
        ),
        (
            "sort_customers",
            [{**paid, "customer": "Zoe"}, {**paid, "id": "b", "customer": "Ana"}],
        ),
        ("trim_ids", [paid, {"id": " a ", "revision": 1, "status": "cancelled"}]),
    ]
    rng = random.Random(20261117)  # noqa: S311 - reproducible fixture generation
    for i in range(128):
        events = []
        for _ in range(rng.randrange(1, 25)):
            row = {
                "id": f"o{rng.randrange(6)}",
                "revision": rng.choice([0, 1, 2, 3, True, -1]),
                "status": rng.choice(["paid", "paid", "cancelled", "unknown"]),
                "customer": rng.choice(["Ana", "Zoe", " Bia ", "", None]),
                "quantity": rng.choice([1, 2, 3, 0, -1, True, 1.5]),
                "unit_price": rng.choice(
                    ["1.005", "0.005", "0", "19.99", "-1", "NaN", "bad", 2]
                ),
            }
            events.append(row)
        rows.append((f"seeded_{i:03}", events))
    return [
        {"name": name, "events": deepcopy(events), "expected": reference(events)}
        for name, events in rows
    ]


EVALUATOR = """import ast, copy, importlib.util, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
all_cases = json.loads(pathlib.Path(sys.argv[2]).read_text())
failures = []; passed = 0
try:
    spec = importlib.util.spec_from_file_location("candidate", root / "reconcile.py")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; sys.path.insert(0, str(root)); spec.loader.exec_module(module)
    for case in all_cases:
        before = copy.deepcopy(case["events"])
        data = copy.deepcopy(before)
        try:
            actual = module.summarize_orders(data)
            assert actual == case["expected"], (actual, case["expected"])
            assert data == before, "input mutated"
            passed += 1
        except Exception as error:
            failures.append({"case":case["name"],"error":str(error)[:1500]})
except Exception as error:
    failures.append({"case":"import", "error":repr(error)})
print(json.dumps({"passed":passed,"total":len(all_cases),"failures":failures[:20]}))
"""


def grade(workspace):
    result = subprocess.run(  # noqa: S603 - deliberate candidate evaluation
        [
            sys.executable,
            str(ROOT / "evaluate.py"),
            str(workspace),
            str(ROOT / "golden.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


async def run_one(mode, repetition):
    run_dir = ROOT / "runs" / f"r{repetition}-{mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir = run_dir / "workspace"
    (workspace_dir / "tests").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "reconcile.py").write_text(BUGGY)
    (workspace_dir / "tests/test_public.py").write_text(PUBLIC)
    (workspace_dir / "README.md").write_text(TASK)
    metrics, calls, outcomes = [], [], []
    model = mf.Model.chat_completion(MODEL, temperature=0, max_tokens=16384)
    original = model.acall

    async def recorded(*args, **kwargs):
        start = time.monotonic()
        response = await original(*args, **kwargs)
        metadata = response.metadata or {}
        usage = metadata.get("usage", {})
        metrics.append(
            {"seconds": time.monotonic() - start, "usage": msgspec.to_builtins(usage)}
        )
        (run_dir / "usage.json").write_text(json.dumps(metrics, indent=2))
        return response

    model.acall = recorded
    tools = [
        ReadFileTool(),
        WriteTool(),
        EditTool(),
        BashTool(),
        LsTool(),
        GrepTool(),
        GlobTool(),
    ]
    if mode != "direct":
        cls = MontyTool if mode == "additive" else MontyToolBucket
        tools.insert(0, cls(state_directory=run_dir / "state"))
    workspace = AgentWorkspace.local(workspace_dir)
    agent = Agent(
        name="benchmark",
        model=model,
        tools=tools,
        workspace=workspace,
        system_prompt=SYSTEM,
        extensions=[ToolTurnLimitExtension(18)],
    )
    agent.tool_library.apply_default_usage_guidance()

    def before(event):
        calls.append(
            {
                "name": event.tool_name,
                "id": event.tool_call_id,
                "arguments": msgspec.to_builtins(event.arguments),
            }
        )
        (run_dir / "calls.json").write_text(json.dumps(calls, indent=2))
        return event

    def after(event):
        outcomes.append(
            {
                "name": event.tool_name,
                "error": str(getattr(event, "error", None))
                if getattr(event, "error", None)
                else None,
            }
        )
        return event

    Hook(event="before_tool", handler=before).register(agent.tool_library)
    Hook(event="after_tool", handler=after).register(agent.tool_library)
    start = time.monotonic()
    error = None
    stop_reason = None
    try:
        response = await asyncio.wait_for(
            agent.acall(TASK, scope=ExecutionScope(thread_id="quality-" + uuid4().hex)),
            timeout=600,
        )
        if isinstance(response, dict):
            stop_reason = response.get("stop_reason")
        (run_dir / "answer.txt").write_text(str(response))
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
        (run_dir / "answer.txt").write_text(error)
    elapsed = time.monotonic() - start
    score = grade(workspace_dir)
    public_path = workspace_dir / "tests/test_public.py"
    unchanged_public = public_path.exists() and public_path.read_text() == PUBLIC
    extra_files = sorted(
        str(p.relative_to(workspace_dir))
        for p in workspace_dir.rglob("*")
        if p.is_file()
        and "__pycache__" not in str(p)
        and p.name not in ["README.md", "reconcile.py", "test_public.py"]
    )
    usages = [row["usage"] for row in metrics]
    totals = {
        key: sum(u.get(key, 0) or 0 for u in usages)
        for key in ["input_tokens", "output_tokens", "total_tokens"]
    }
    totals["cached_input_tokens"] = sum(
        (u.get("input_tokens_details") or {}).get("cached_tokens", 0) or 0
        for u in usages
    )
    totals["reasoning_tokens"] = sum(
        (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0) or 0
        for u in usages
    )
    costs = [u.get("cost") for u in usages]
    counts = dict(Counter(row["name"] for row in calls))
    result = {
        "mode": mode,
        "repetition": repetition,
        "model": MODEL,
        "elapsed_seconds": round(elapsed, 2),
        "grade": score,
        "public_tests_unchanged": unchanged_public,
        "extra_files": extra_files,
        "tokens": totals,
        "provider_cost": sum(costs)
        if costs and all(isinstance(v, (int, float)) for v in costs)
        else None,
        "scope_ok": (workspace_dir / "README.md").read_text() == TASK
        and all(name.startswith("tests/") for name in extra_files),
        "model_requests": len(metrics),
        "tool_calls": counts,
        "tool_errors": [row for row in outcomes if row["error"]],
        "error": error,
        "stop_reason": stop_reason,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)  # noqa: T201 - CLI JSON result
    close = getattr(model, "aclose", None)
    if close:
        await close()
    return result


def preflight():
    (ROOT / "prompt.txt").write_text(TASK)
    (ROOT / "system.txt").write_text(SYSTEM)
    (ROOT / "golden.json").write_text(json.dumps(cases(), indent=2))
    (ROOT / "evaluate.py").write_text(EVALUATOR)
    preflight = ROOT / "preflight"
    preflight.mkdir(exist_ok=True)
    (preflight / "reconcile.py").write_text(BUGGY)
    broken = grade(preflight)
    if broken["passed"] >= broken["total"] / 2:
        raise RuntimeError(f"Starter unexpectedly passed preflight: {broken}")
    (ROOT / "preflight.json").write_text(json.dumps(broken, indent=2))
    (preflight / "reconcile.py").write_text(
        "from decimal import Decimal, InvalidOperation, ROUND_HALF_UP\n"
        + inspect.getsource(reference)
        + "\nsummarize_orders = reference\n"
    )
    correct = grade(preflight)
    if correct["passed"] != correct["total"]:
        raise RuntimeError(f"Reference failed preflight: {correct}")
    (ROOT / "preflight-reference.json").write_text(json.dumps(correct, indent=2))
    return broken, correct


async def main():
    global ROOT  # noqa: PLW0603 - one output root for this standalone experiment
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--offline-only", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    ROOT = args.output_dir.expanduser().resolve()
    ROOT.mkdir(parents=True, exist_ok=True)
    previous = ROOT / "manifest.json"
    if previous.exists():
        recorded = json.loads(previous.read_text())
        if (
            recorded.get("task_sha256") != hashlib.sha256(TASK.encode()).hexdigest()
            or recorded.get("system_sha256")
            != hashlib.sha256(SYSTEM.encode()).hexdigest()
        ):
            raise ValueError("Output directory contains a different benchmark protocol")
    if args.env_file and not args.offline_only:
        mf.load_dotenv(str(args.env_file))
    broken, correct = preflight()
    manifest = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens_per_response": 16384,
        "task_sha256": hashlib.sha256(TASK.encode()).hexdigest(),
        "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
        "hidden_cases": len(cases()),
        "design": "3 repetitions per mode, identical task/system, Latin-square order; additive use is optional",
        "python": sys.version,
        "timeout_seconds": 600,
        "protocol_version": 2,
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if args.offline_only:
        print(  # noqa: T201 - offline CLI result
            json.dumps({"preflight": correct, "starter_passed": broken["passed"]})
        )
        return
    order = [
        ["direct", "additive", "bucket"],
        ["additive", "bucket", "direct"],
        ["bucket", "direct", "additive"],
    ]
    results = []
    for repetition, modes in enumerate(order, 1):
        for mode in modes:
            target = ROOT / "runs" / f"r{repetition}-{mode}" / "result.json"
            if target.exists():
                results.append(json.loads(target.read_text()))
                continue
            print(  # noqa: T201 - CLI progress
                f"START repetition={repetition} mode={mode}", flush=True
            )
            results.append(await run_one(mode, repetition))
            (ROOT / "results.json").write_text(json.dumps(results, indent=2))
            if args.pilot:
                return


if __name__ == "__main__":
    asyncio.run(main())
