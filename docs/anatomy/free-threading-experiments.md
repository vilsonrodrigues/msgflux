# Free-Threading Experiments

## Status and Scope

This investigation starts from `49d50879` and does not change the production
executor, AgentService, dependency declarations, or Python support matrix.
It compares conventional CPython 3.15 with CPython 3.15t with the GIL enabled
and disabled. Alternative loop packages are installed only in experimental
environments.

The notebook initially overheated and rebooted, so sustained comparisons were
first deferred. The user subsequently authorized running without a temperature
restriction. All timing cases below used `--no-temperature-limit`; temperatures
were still recorded. CPU frequencies and thermal state varied substantially,
so the results establish parallel execution and observed compatibility rather
than a trustworthy ranking of small speed differences.

## Verified So Far

| Check | Observation |
| --- | --- |
| Interpreter | uv installed CPython 3.15.0 free-threading without manually compiling CPython. |
| Runtime dependencies | The existing local lockfile installed after building PyYAML 6.0.3, MarkupSafe 3.0.3, msgspec 0.21.1 and uvloop 0.22.1 from source. |
| Development dependencies | NumPy 2.4.6 had no compatible binary in the locked installation attempt. It was excluded by using `--no-dev`. |
| Runtime imports | msgspec, PyYAML, uvloop, httpx2, msgFlux Agent, and the service HTTP module imported with the GIL remaining disabled. |
| Alternative imports | rloop 0.5.0 and zuvloop 0.0.17 imported without enabling the GIL. |
| Agent smoke test | Four Agents completed a scripted tool/final-response cycle with SQLite checkpoint verification; GIL stayed disabled and the reported post-run thread delta was zero. |
| Thermal refusal | The benchmark refused to start its workload at 75.6 C with its default 70 C ceiling. |
| Static checks | Script syntax, Ruff lint and formatting passed. |
| Runtime tests | 208 focused executor/tool/backend and offline durability tests passed on 3.15t with the GIL disabled. |

Import success does not establish thread safety. In particular, the upstream
[msgspec free-threading issue](https://github.com/msgspec/msgspec/issues/1249)
reports data races. Our shared encoder/decoder instances require a deliberate
concurrency audit before recommending GIL-free production operation.

NumPy is not used by the measured Agent path. It is used elsewhere in the
repository by embedding pooling, optional BM25 retrievers, and their tests;
it has not been removed from the project.

## Measured Results

The initial matrix contained 184 fresh-process runs. Another 24 runs exercised
Agents and shared SQLite directly on uvloop, rloop, and zuvloop with the GIL
disabled. **206 of 208 completed successfully**. Two rloop four-loop TCP cases
failed during loop shutdown, one on conventional Python and one without the
GIL. Successful cases verified task identities, arithmetic/tool results, and
completed checkpoint reads where applicable.

Median process-CPU/wall ratios for the four-worker, single-loop/thread-pool
configuration are below. This ratio is the number of logical-CPU equivalents
actually consumed; it does not identify which physical cores ran each thread.
RSS is process high-water including imports, in MiB. Each cell summarizes two
runs of 16 independent jobs/Agents, not concurrent turns on one Agent.

| Workload | Conventional 3.15 CPU equivalents | 3.15t GIL off CPU equivalents | Peak RSS, conventional / GIL off (MiB) |
| --- | ---: | ---: | ---: |
| Pure Python CPU | 1.02 | 3.88 | 27.0 / 35.6 |
| LocalTool | 1.02 | 3.92 | 62.9 / 78.1 |
| Agent + tool | 1.02 | 3.37 | 64.8 / 82.0 |
| Agent + tool + SQLite | 1.02 | 3.33 | 66.5 / 84.4 |

The GIL-on free-threaded binary remained near one CPU equivalent on CPU-heavy
work, while the same binary with the GIL off used multiple CPUs. This isolates
the GIL effect from merely selecting a different Python release.

In the GIL-off pool cases, median elapsed time was about 3.25 s versus 1.01 s
for CPU work at one versus four workers, and 3.35 s versus 1.35 s for the Agent
workload. These are observed medians, **not portable speedup claims**: other
nominally equivalent repeats had large outliers. Temperature samples ranged
from 58.25 to 105.625 C and reported CPU frequencies from roughly 398 to 3992
MHz. Thermal/frequency variation and possible contention were not separated by
a controlled profiler experiment.

The pure CPU workload on independent loop threads blocked each loop's
heartbeat for the batch, whereas a bounded thread pool kept its coordinating
loop responsive. Full Agent setup still produced noticeable heartbeat stalls;
free-threading does not remove synchronous preparation from the loop thread.

### Other Event Loops and Protocols

All four loops passed the initial one-worker checks for context propagation,
callback delivery, cancellation cleanup, subprocesses, and TCP streams in the
three interpreter/GIL modes. I/O measurements used one/two/four independent
loops on conventional and GIL-off interpreters. The additional native-loop
Agent/SQLite runs all passed; their four-worker CPU ratios were approximately
3.1–3.6. The I/O workload is too short/noisy to establish a winning loop.

A separate real HTTPS/chunked-SSE probe used a temporary locally trusted
certificate and httpx2, verifying all three streamed events. It did not invoke
a provider or exercise the AgentService HTTP contract.

| Loop | HTTPS/SSE: conventional / FT GIL on / FT GIL off | Four-thread lifecycle isolation |
| --- | --- | --- |
| asyncio | All passed | All three modes passed |
| uvloop 0.22.1 | All passed | All three modes passed |
| rloop 0.5.0 | All returned `NotImplementedError` for SSL | Failed on conventional and GIL-off; GIL-on timed out |
| zuvloop 0.0.17 | All passed | All three modes passed |

The lifecycle reproducer creates/runs/closes 20 loops on each of four threads
and uses only asyncio and the chosen loop package. It contains no msgFlux
imports. Rloop produced bad-file-descriptor/self-socket cleanup failures and
masked cleanup exceptions. This establishes an observed rloop 0.5.0 problem in
this setup, not a general free-threading or msgFlux failure; no upstream fix
or root-cause proof is claimed here.

Three zuvloop contract records reported one extra Python thread after using
subprocesses. Installed `_process.py` starts one process-wide daemon named
`zuvloop-process-reaper`; the observation matches that bounded design and is
not evidence of one new leaked thread per Agent. Thread counts here come from
`threading.enumerate`, not a complete native-thread census. Uvloop emitted
`asyncio.iscoroutinefunction` deprecation warnings for Python 3.16.

## Event Loop Candidates

| Loop | Experimental role | Current qualification |
| --- | --- | --- |
| `asyncio` | Standard-library reference | Establish correctness and overhead before comparing native loops. |
| `uvloop` 0.22.1 | Existing msgFlux dependency | Passed this limited matrix and HTTPS/SSE probes; deprecation warning noted. Full service conformance remains pending. |
| `rloop` 0.5.0 | Rust selector-loop comparison | Installed source raises `NotImplementedError` for SSL connections; upstream also documents missing SSL/Unix sockets and labels it unsuitable for production. |
| `zuvloop` 0.0.17 | Zig/libuv comparison | Passed this limited matrix, HTTPS/SSE, and isolated lifecycle probes; still needs full service conformance. |

See the [rloop repository](https://github.com/gi0baro/rloop) and
[zuvloop repository](https://github.com/Kludex/zuvloop). Their own benchmark
results are not measurements of msgFlux.

These libraries replace the event-loop implementation. Uvicorn remains a
separate ASGI-server choice. The experiment creates loops with explicit
factories, without changing the production process-wide event-loop policy.

## Workloads and Measurements

`scripts/benchmark_free_threading.py` emits one JSON result per fresh process.
It uses deterministic Python arithmetic, not a native function that might
already release the GIL. Available workloads are:

- `cpu`: direct Python work, comparing an intentionally blocking loop,
  one loop with a bounded thread pool, and independent loops on threads.
- `tool`: the actual msgFlux `LocalTool` async adapter dispatching a synchronous
  CPU function and verifying explicit task-context propagation.
- `agent`: an actual Agent with a tool call and final response from a scripted
  generator. No credentials, network providers, or subscription usage occur.
  `--checkpoint` adds a shared SQLite store and verifies completed states.
- `io`: real loopback TCP streams with exact content verification. This is a
  transport probe, not a full HTTP/SSE acceptance benchmark.

JSON includes actual GIL state after imports and execution, package versions,
completed task identities/checksums, wall/process CPU time, effective CPU cores,
throughput, completion p50/p95, heartbeat delays, baseline/peak/final RSS,
remaining threads, and CPU temperature readings. Completion latency is measured
from batch admission and includes queueing, loop startup, and setup inside the
workload. Process imports are excluded from timing. The process RSS high-water
includes imports; it is not an isolated workload allocation peak. Heartbeat
p95 is over observed wakeups, not weighted by missed intervals, so maximum
delay, sample count, and missed-interval count must be considered together.
GIL state is sampled before/after, not continuously.

`--check-contracts` adds cross-thread callback delivery, context propagation,
cancellation cleanup, and subprocess checks. Keep those runs separate from
throughput comparisons. Every loop thread starts with an explicit empty
`contextvars.Context`; each Agent and async model belongs to its executing
loop. Only the thread-safe SQLite store is shared.

## Reproducing the Experiment

The existing local `uv.lock` was copied into the experimental worktree. It is
not versioned in this repository. Preserve that file and record its digest
when reproducing results. Install each build in a distinct environment with
`uv sync --frozen --no-dev --extra service --no-install-project`; explicitly
select `--python 3.15` or `--python 3.15t` and an independent
`UV_PROJECT_ENVIRONMENT`. The existing environments were reused for measurement; source compilation was
not part of the timed workloads.

Run from the worktree, with the environment paths selected for this machine:

```bash
export PYTHONPATH="$PWD/src"
export MSGFLUX_CONFIGURE_LOGGING=false
export MSGFLUX_MALLOC_TRIM=0
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Same free-threaded binary, two GIL modes. Begin with short one-worker cases.
nice -n 10 timeout 30 .venv/bin/python -X gil=1 \
  scripts/benchmark_free_threading.py --workload cpu --topology pool \
  --workers 1 --tasks 4 --iterations 100000

nice -n 10 timeout 30 .venv/bin/python -X gil=0 \
  scripts/benchmark_free_threading.py --workload cpu --topology pool \
  --workers 1 --tasks 4 --iterations 100000

# Check native loop behavior before increasing concurrency.
nice -n 10 timeout 30 .venv/bin/python \
  scripts/benchmark_free_threading.py --workload io --event-loop zuvloop \
  --topology loops --workers 1 --tasks 4 --io-rounds 4 --check-contracts
```

The default ceiling is 70 C, checked before workload setup, at task admission,
and every 16,384 iterations inside CPU work (including reference computation).
A running CPU function stops cooperatively at its next chunk boundary; there
is no forced thread termination. These thermal checks are included in timing.
Keep tasks short. On failure the owner waits for all loop workers before
closing shared stores; always retain the external process timeout to bound a
native event-loop hang. Begin with one worker, then two only after stable cooling.
To reproduce the user-authorized unrestricted-temperature runs, explicitly
add `--no-temperature-limit`; this disables preflight/chunk limits while
preserving temperature readings. The matrix ran cases sequentially with short
pauses and an external 30-second process timeout. Repeat measurements rather
than interpreting a single short run. Keep `MSGFLUX_MALLOC_TRIM` identical across comparisons; the initial
matrix disables the optional global trim to reduce allocator-side noise.

## Proposed Ownership Model

Start by keeping service coordination and network I/O on their existing loop,
with bounded CPU offload for independent tool work. A multi-loop extension
would assign each conversation an owning worker and route all of its turns,
steering, cancellation, and completion notices through that owner. HTTP
clients, asyncio locks, futures, Agent state, and model streams stay on their
owning loop; thread-safe scheduling bridges the boundary.

The current executor's round-robin coroutine dispatch is not a substitute for
conversation affinity. More async workers alone do not establish ownership
of persistent clients and per-conversation state. Any future change must also
audit shared caches/codecs, context inheritance, admission/recovery races, and
shutdown before increasing the default worker count.

No server, executor, or loop-default change is proposed for merge based on
the observations collected so far.


## Static Review

An explorer launched through the msgFlux CLI with
`opencode-go/deepseek-v4.1-flash` performed a read-only review. The experiment
was refined to check the callback's actual executing thread and explicit
context, enforce the temperature ceiling inside CPU jobs, join loop owners
before shared-resource cleanup on failure, independently validate the actual
tool result, and clarify metric boundaries.

The review also suggested that default executors were not shut down and that
sharing the synchronous checkpoint store was inherently unsupported. Those
claims were not adopted: `asyncio.Runner` closes its default executor, and the
shared store is deliberately exercised through its lock-protected synchronous
interface. This does not establish that all native codecs or the surrounding
runtime are safe without the GIL. The revised paths were subsequently exercised by the user-authorized matrix.
The runtime gate passed 208 tests; the complete repository suite was not run
in this experimental environment, which intentionally excludes the full dev
dependency group.


## Recommendation and Remaining Work

The strongest observed benefit is CPU offload through the existing LocalTool
path. Prefer first evaluating a bounded shared CPU pool with one coordinating
service loop. Multiple event loops also worked in the limited tests, but add
ownership and shutdown obligations and did not demonstrate a decisive benefit
here. Rloop 0.5.0 is not a default candidate given SSL and lifecycle failures.
Zuvloop is a candidate for an opt-in loop factory, with service-level SSE,
reconnect/recovery, cancellation, real provider/client affinity, and long-run
memory tests still required. No production default has changed.

The runtime test command used Python 3.15t with `-X gil=0`, the virtualenv on
PATH, `MSGFLUX_CONFIGURE_LOGGING=false`, and `MSGFLUX_MALLOC_TRIM=0`, covering
executor/rejected-coroutine/local-tool errors, shared backend, checkpoint,
approval and offline durability conformance/process tests. Result: **208 passed,
130 warnings, 28.53 seconds**. This is not certification of all third-party
native dependencies as free-thread safe.
