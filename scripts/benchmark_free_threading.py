"""Offline free-threading and event-loop experiment; emits one JSON record.

Run each configuration in a fresh process. No credentials or provider calls
are used. The CPU workload is deliberately Python code, not a native hash
that may already release the GIL. Agent mode exercises a real LocalTool and
two scripted model responses, optionally with shared SQLite checkpoints.
"""

from __future__ import annotations

import argparse
import asyncio
import contextvars
import gc
import importlib
import importlib.metadata
import json
import math
import resource
import statistics
import sys
import sysconfig
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

TASK_ID = contextvars.ContextVar("benchmark_task_id", default=None)
THERMAL_LIMIT = contextvars.ContextVar("benchmark_thermal_limit", default=70.0)
CANCEL_WORK = threading.Event()


def cpu_kernel(iterations: int, thermal_limit: float | None) -> int:
    value = 123456789
    for start in range(0, iterations, 16384):
        if CANCEL_WORK.is_set():
            raise RuntimeError("CPU workload cancelled during cleanup")
        if thermal_limit is not None:
            check_temperature(thermal_limit)
        for index in range(start, min(start + 16384, iterations)):
            value = ((value * 1664525 + index + 1013904223) & 0xFFFFFFFF) ^ (value >> 7)
    return value


def cpu_work(iterations: int) -> str:
    """Compute a deterministic Python checksum for the requested work size."""
    if TASK_ID.get() is None:
        raise AssertionError("Task context was lost while dispatching CPU work")
    return str(cpu_kernel(iterations, THERMAL_LIMIT.get()))


def cpu_temperatures() -> list[float]:
    temperatures = []
    for directory in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            if (directory / "name").read_text().strip() not in {
                "coretemp",
                "k10temp",
                "zenpower",
            }:
                continue
            temperatures.extend(
                float(path.read_text()) / 1000 for path in directory.glob("temp*_input")
            )
        except OSError:
            continue
    return temperatures


def check_temperature(maximum: float | None) -> list[float]:
    if maximum is None:
        return []
    temperatures = cpu_temperatures()
    if temperatures and max(temperatures) >= maximum:
        raise RuntimeError(
            f"Experiment refused: CPU is {max(temperatures):.1f} C "
            f"(limit {maximum:.1f} C)"
        )
    return temperatures


def rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("This experiment currently measures Linux /proc RSS")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)] if values else 0.0


def select_loop(name: str):
    if name == "asyncio":
        return asyncio.SelectorEventLoop
    return importlib.import_module(name).new_event_loop


def versions() -> dict:
    result = {}
    for name in ("msgspec", "httpx2", "uvloop", "rloop", "zuvloop"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def make_local_tool():
    from msgflux.nn.modules.tool import LocalTool  # noqa: PLC0415

    return LocalTool(
        name="cpu_work",
        description="Compute a checksum using Python arithmetic.",
        annotations={"iterations": int, "return": str},
        tool_config={},
        impl=cpu_work,
    )


async def agent_job(index: int, iterations: int, expected: str, store) -> dict:
    from msgflux.models.response import ModelResponse  # noqa: PLC0415
    from msgflux.models.tool_call_agg import ToolCallAggregator  # noqa: PLC0415
    from msgflux.nn import Agent  # noqa: PLC0415
    from msgflux.runtime.context import ExecutionScope  # noqa: PLC0415

    class ScriptedModel:
        model_type = "chat_completion"

        def __call__(self, **_kwargs):
            raise AssertionError("A benchmark must not invoke a provider")

    calls = 0
    tool_results = []

    def checked_cpu_work(iterations: int) -> str:
        """Compute the actual tool result for independent verification."""
        result = cpu_work(iterations)
        tool_results.append(result)
        return result

    async def scripted_response(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.003)
        response = ModelResponse()
        response.reasoning = None
        if calls == 1:
            tool_calls = ToolCallAggregator()
            tool_calls.process(
                0,
                f"call_{index}",
                "checked_cpu_work",
                json.dumps({"iterations": iterations}),
            )
            response.set_response_type("tool_call")
            response.add(tool_calls)
        elif calls == 2:
            response.set_response_type("text_generation")
            response.add(expected)
        else:
            raise AssertionError("Unexpected model retry in deterministic workload")
        return response

    agent = Agent(
        name="free_threading_agent",
        model=ScriptedModel(),
        tools=[checked_cpu_work],
        checkpoint_store=store,
    )
    agent.generator.aforward = scripted_response
    scope = ExecutionScope(
        namespace="free_threading_agent",
        thread_id=f"thread-{index}",
        run_id=f"run-{index}",
    )
    try:
        result = await agent.acall("Compute the checksum.", scope=scope)
        if result != expected or calls != 2 or tool_results != [expected]:
            raise AssertionError(f"Unexpected Agent result: {result!r}, calls={calls}")
    finally:
        await agent.aclose()
    return {"index": index, "result": result}


async def check_loop_contracts() -> dict:
    """Small correctness probes, separate from normal throughput runs."""
    loop = asyncio.get_running_loop()
    delivered = loop.create_future()
    owner_thread = threading.get_ident()
    token = TASK_ID.set("callback-context")
    callback_context = contextvars.copy_context()
    TASK_ID.reset(token)

    def receive():
        delivered.set_result((threading.get_ident(), TASK_ID.get()))

    def send():
        loop.call_soon_threadsafe(receive, context=callback_context)

    thread = threading.Thread(target=contextvars.Context().run, args=(send,))
    thread.start()
    if await asyncio.wait_for(delivered, 2) != (owner_thread, "callback-context"):
        raise AssertionError("Callback ran on the wrong thread or lost its context")
    thread.join(timeout=2)
    if thread.is_alive():
        raise AssertionError("Cross-thread sender did not exit")
    cleaned_up = []

    async def cancellable():
        try:
            await asyncio.sleep(60)
        finally:
            cleaned_up.append(True)

    pending = asyncio.create_task(cancellable())
    await asyncio.sleep(0)
    pending.cancel()
    try:
        await pending
    except asyncio.CancelledError:
        pass
    if cleaned_up != [True]:
        raise AssertionError("Cancellation did not run cleanup")
    token = TASK_ID.set("context-check")
    try:
        context = contextvars.copy_context()
        inherited = await loop.run_in_executor(None, context.run, TASK_ID.get)
        if inherited != "context-check":
            raise AssertionError("Explicit task context propagation failed")
    finally:
        TASK_ID.reset(token)
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "print(42)",
        stdout=asyncio.subprocess.PIPE,
    )
    stdout, _stderr = await asyncio.wait_for(process.communicate(), 5)
    if stdout.strip() != b"42" or process.returncode != 0:
        raise AssertionError("Subprocess output or exit status changed")
    return {
        "threadsafe_handoff": True,
        "cancellation_cleanup": True,
        "context_propagation": True,
        "subprocess": True,
    }


async def run_batch(  # noqa: C901 - workload and loop cleanup stay together
    args, indices: list[int], expected: str, store, submitted: float
) -> dict:
    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(
        max_workers=args.workers if args.topology != "loops" else 1
    )
    loop.set_default_executor(pool)
    delays: list[float] = []
    finished = asyncio.Event()

    async def heartbeat():
        interval = 0.005
        deadline = loop.time() + interval
        while not finished.is_set():
            await asyncio.sleep(max(0.0, deadline - loop.time()))
            now = loop.time()
            delays.append(max(0.0, now - deadline))
            deadline = now + interval

    pulse = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)
    tool = make_local_tool() if args.workload == "tool" else None
    contracts = await check_loop_contracts() if args.check_contracts else None
    server = None
    connections = set()
    payload = b"msgflux-stream-record\n" * 400

    async def echo(reader, writer):
        current = asyncio.current_task()
        connections.add(current)
        try:
            while data := await reader.read(16384):
                writer.write(data)
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            connections.discard(current)

    if args.workload == "io":
        server = await asyncio.start_server(echo, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]

    async def job(index):
        token = TASK_ID.set(index)
        thermal_token = THERMAL_LIMIT.set(args.max_cpu_celsius)
        try:
            check_temperature(args.max_cpu_celsius)
            if args.workload == "io":
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                try:
                    for _ in range(args.io_rounds):
                        writer.write(payload)
                        await writer.drain()
                        if await reader.readexactly(len(payload)) != payload:
                            raise AssertionError("Stream contents changed")
                finally:
                    writer.close()
                    await writer.wait_closed()
                result = expected
            elif args.workload == "agent":
                result = (await agent_job(index, args.iterations, expected, store))[
                    "result"
                ]
            elif args.workload == "tool":
                result = await tool.acall(iterations=args.iterations)
            elif args.topology == "pool":
                context = contextvars.copy_context()
                result = await loop.run_in_executor(
                    None, context.run, cpu_work, args.iterations
                )
            else:
                result = cpu_work(args.iterations)
            if result != expected:
                raise AssertionError(f"Checksum mismatch for task {index}")
            return {"index": index, "latency": time.perf_counter() - submitted}
        finally:
            THERMAL_LIMIT.reset(thermal_token)
            TASK_ID.reset(token)

    try:
        results = await asyncio.gather(*(job(index) for index in indices))
    finally:
        if server is not None:
            server.close()
            await server.wait_closed()
            if connections:
                await asyncio.wait_for(asyncio.gather(*list(connections)), 5)
        finished.set()
        await pulse
    pending = [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task() and not task.done()
    ]
    return {
        "results": results,
        "heartbeat": delays,
        "pending_tasks": len(pending),
        "contracts": contracts,
    }


def measure(args) -> dict:  # noqa: C901 - experiment lifecycle kept together
    CANCEL_WORK.clear()
    factory = select_loop(args.event_loop)
    store = None
    temporary = tempfile.TemporaryDirectory(prefix="msgflux-free-threading-")
    if args.workload in {"tool", "agent"}:
        importlib.import_module("msgflux.nn.modules.agent")
    if args.checkpoint:
        from msgflux.data.stores import SQLiteCheckpointStore  # noqa: PLC0415

        store = SQLiteCheckpointStore(str(Path(temporary.name) / "checkpoints.sqlite3"))
    gil_before = sys._is_gil_enabled()
    expected = (
        "echo-validated"
        if args.workload == "io"
        else str(cpu_kernel(args.iterations, args.max_cpu_celsius))
    )
    gc.collect()
    baseline_rss = rss_bytes()
    initial_threads = len(threading.enumerate())
    started = time.perf_counter()
    cpu_started = time.process_time()
    batches = []
    workers = []
    futures = []

    def run(indices, future):
        try:
            with asyncio.Runner(loop_factory=factory) as runner:
                batch = runner.run(run_batch(args, indices, expected, store, started))
            future.set_result(batch)
        except BaseException as error:
            future.set_exception(error)

    try:
        if args.topology == "loops":
            for worker_id in range(args.workers):
                future = Future()
                futures.append(future)
                indices = list(range(worker_id, args.tasks, args.workers))
                thread = threading.Thread(
                    target=contextvars.Context().run,
                    args=(run, indices, future),
                    name=f"benchmark-loop-{worker_id}",
                )
                workers.append(thread)
                thread.start()
            batches = [future.result(timeout=45) for future in futures]
        else:
            with asyncio.Runner(loop_factory=factory) as runner:
                batches = [
                    runner.run(
                        run_batch(
                            args, list(range(args.tasks)), expected, store, started
                        )
                    )
                ]
        for thread in workers:
            thread.join(timeout=5)
            if thread.is_alive():
                raise AssertionError("An event-loop worker did not shut down")
        elapsed = time.perf_counter() - started
        cpu_elapsed = time.process_time() - cpu_started
        results = [result for batch in batches for result in batch["results"]]
        if sorted(result["index"] for result in results) != list(range(args.tasks)):
            raise AssertionError("Missing or duplicate task results")
        if any(batch["pending_tasks"] for batch in batches):
            raise AssertionError("Pending asyncio tasks after workload")
        if store is not None:
            for index in range(args.tasks):
                state = store.load_state(
                    "free_threading_agent", f"thread-{index}", f"run-{index}"
                )
                if state is None or state["status"] != "completed":
                    raise AssertionError("Checkpoint did not recover completed state")
    finally:
        CANCEL_WORK.set()
        # Runner closes its default executor. On failure, join every owning
        # loop before closing resources it may still use. The documented
        # external process timeout bounds native-loop hangs.
        for thread in workers:
            thread.join()
        if store is not None:
            store.close()
        temporary.cleanup()
    gc.collect()
    heartbeats = [delay for batch in batches for delay in batch["heartbeat"]]
    latencies = [result["latency"] for result in results]
    return {
        "python": sys.version,
        "free_threaded_build": bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        "gil_before": gil_before,
        "gil_after": sys._is_gil_enabled(),
        "versions": versions(),
        "config": vars(args),
        "completed": len(results),
        "checksum": expected,
        "wall_seconds": elapsed,
        "cpu_seconds": cpu_elapsed,
        "effective_cpu_cores": cpu_elapsed / elapsed,
        "tasks_per_second": args.tasks / elapsed,
        "completion_p50_ms": statistics.median(latencies) * 1000,
        "completion_p95_ms": percentile(latencies, 0.95) * 1000,
        "heartbeat_p95_ms": percentile(heartbeats, 0.95) * 1000,
        "heartbeat_max_ms": max(heartbeats, default=0) * 1000,
        "heartbeat_sample_count": len(heartbeats),
        "heartbeat_missed_intervals": sum(int(delay / 0.005) for delay in heartbeats),
        "rss_baseline_bytes": baseline_rss,
        "rss_process_peak_bytes": max(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            baseline_rss,
            rss_bytes(),
        ),
        "rss_after_bytes": rss_bytes(),
        "thread_delta_after": len(threading.enumerate()) - initial_threads,
        "contracts": [batch["contracts"] for batch in batches],
        "cpu_temperatures_after": cpu_temperatures(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload", choices=("cpu", "tool", "agent", "io"), default="cpu"
    )
    parser.add_argument(
        "--topology", choices=("inline", "pool", "loops"), default="pool"
    )
    parser.add_argument(
        "--event-loop",
        choices=("asyncio", "uvloop", "rloop", "zuvloop"),
        default="asyncio",
    )
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--tasks", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=400000)
    parser.add_argument("--io-rounds", type=int, default=32)
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--check-contracts", action="store_true")
    thermal = parser.add_mutually_exclusive_group()
    thermal.add_argument("--max-cpu-celsius", type=float)
    thermal.add_argument(
        "--no-temperature-limit",
        dest="max_cpu_celsius",
        action="store_const",
        const=None,
    )
    parser.set_defaults(max_cpu_celsius=70.0)
    args = parser.parse_args()
    if (
        args.tasks < args.workers
        or min(args.iterations, args.io_rounds) <= 0
        or (args.max_cpu_celsius is not None and args.max_cpu_celsius <= 0)
    ):
        parser.error(
            "tasks must be >= workers and work sizes/temperature limit must be positive"
        )
    if args.checkpoint and args.workload != "agent":
        parser.error("--checkpoint requires --workload agent")
    if args.topology == "inline" and args.workload != "cpu":
        parser.error("inline is the deliberately blocking CPU control")
    try:
        temperatures = cpu_temperatures()
        check_temperature(args.max_cpu_celsius)
    except RuntimeError as error:
        parser.exit(2, f"{error}\n")
    result = measure(args)
    result["cpu_temperatures_before"] = temperatures
    sys.stdout.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
