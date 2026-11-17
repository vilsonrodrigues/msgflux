# Monty tools: DeepSeek Flash coding comparison

Model: `opencode-go/deepseek-v4.1-flash`, temperature 0, max output 16384 per response, 600-second safeguard, 18 tool rounds.
Same system prompt, task, initial files and seven underlying workspace tools: read, write, edit, bash, ls, grep, glob.
Three repetitions per mode, fresh workspace/thread, Latin-square order. Optional additive use is not forced.

## Quality and usage

| Mode | Perfect/completed | Median input | Median output | Median total | Total range | Median requests | Median seconds | Used Monty |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | 3/3 perfect; 3/3 completed | 54,256 | 10,102 | 64,358 | 43,662–167,266 | 10 | 126.2 | 0/3 |
| additive | 3/3 perfect; 3/3 completed | 35,725 | 7,900 | 41,889 | 41,628–120,381 | 8 | 109.3 | 0/3 |
| bucket | 2/3 perfect; 2/3 completed | 77,956 | 13,020 | 90,412 | 81,182–106,799 | 17 | 199.5 | 3/3 |

Each perfect run passes 172 external cases, including seeded combinations and input immutability, preserves the original public tests, and respects allowed file changes.
The buggy starter passes 21/172. The reference implementation passes 172/172. Hidden results are never fed back to the model.

## Cache and reasoning

| Mode | Median cached input | Cache hit, weighted | Median reasoning | Reported charged cost, sum | Reported list cost, sum | Tool errors |
|---|---:|---:|---:|---:|---:|---:|
| direct | 44,672 | 80.5% | 4,912 | 0 | 0.02955466 | 0 |
| additive | 27,392 | 84.4% | 5,382 | 0 | 0.02037805 | 1 |
| bucket | 67,584 | 81.3% | 7,897 | 0 | 0.03181145 | 5 |

Column medians are computed independently and need not add up to the median total.
Token counts come from provider usage for every completed response. Input includes cache hits. Reasoning is included in output, not added to total again.
GO reports zero charged cost under the subscription; list_cost is provider-reported reference consumption, not a new cash charge. Cache was observed, not flushed or controlled.

## Limits

- Independent post-run tests and scope checks are in the archived independent-tests.json; standard test caches are generated artifacts, not forbidden source changes. Raw trial scope_ok may flag those caches; the final scope check classifies them separately.
- One task and three trials per mode are exploratory, not a general benchmark or statistical proof.
- Bash is available in all modes and can batch work/run Python itself. This intentionally keeps the baseline realistic.
- An additive agent can ignore Monty; usage is reported separately. Such a run measures optional availability and schema overhead, not mandatory execution through Monty.
- Timing includes remote provider variance and local tools. Cache warming and provider reasoning vary; token ranges matter.
- One bucket trial reached the shared 18-tool-round ceiling, passed 154/172 and left a cancelled-winner bug. Its final fuzz command wrote /tmp/fuzz.py outside the requested workspace and failed to import reconcile. This is reported as an incomplete run and a trace-level scope deviation, not as a successful final answer.
- A preliminary direct pilot passed 172/172 but hit 300 seconds while pursuing unbounded numeric edge cases. It is excluded. Protocol v2 explicitly bounds inputs and uses a uniform 600-second safeguard.
- Preflight found structured BashTool results were not bridged to Monty; conversion was fixed and regression-tested before measured trials. The runtime was frozen during all measured trials.

## Interpretation

The optional-additive agent never used Monty, so its lower median is not evidence of a gain from Monty execution. The bucket median was 40.5% above direct tools; its mean was only 1.1% above because the direct runs varied widely. One bucket run was incomplete and failed 18 external cases. This sample supports improving interface guidance and return discovery, not a general performance ranking. Proposed changes are listed below.

## Future work

Reuse existing return annotations and a separate `Returns:` docstring section
for output semantics. Expose output schemas and return descriptions through
`tools.describe()` without duplicating declared return types in another init
parameter. Tighten generic guidance around host I/O, keyword arguments, printing
unfamiliar results and batching related operations. Review recovery independently
so a harmless read-result error can retain committed variables when no external
effect is uncertain.

Measure each change separately against this baseline. A forced-additive trial
must be labeled separately: optional-additive agents ignored Monty in all three
measured trials.

## Preserved artifacts

The archive contains the [nine trial rows](experiments/monty-tools-20261117/runs.csv),
[aggregate data](experiments/monty-tools-20261117/summary.json),
[normalized usage](experiments/monty-tools-20261117/request-usage.csv) for all
111 settled model requests, independent test results,
the environment manifest, and the exact task and system prompts. Total measured
usage: **757,577 tokens**, including cache hits. Provider-reported list cost
sums to 0.08174416; the reported charged cost is zero under Go.

Full local traces, generated candidate workspaces and the excluded pilot remain
outside the repository. The compact archive intentionally excludes credentials,
model reasoning text, caches and generated source files.

## Reproduction

Install the optional dependencies and run the offline preflight first:

```bash
uv sync --group dev --extra coding --extra monty
uv run python scripts/benchmark_monty_tools.py \
  --output-dir /tmp/msgflux-monty-preflight --offline-only
```

The preflight validates the 172 golden cases without making provider calls.
For a new live sample, set `OPENCODE_GO_API_KEY` in the environment or supply
an env file explicitly:

```bash
uv run python scripts/benchmark_monty_tools.py \
  --output-dir /tmp/msgflux-monty-sample \
  --env-file /path/to/provider.env
```

The live command runs the three conditions three times. It executes generated
code through local workspace tools, uses the provider account, and may consume
plan allowance. Generated source is model-authored and must be inspected; the
workspace prompt is not an operating-system sandbox. Existing completed trials
are reused only from a directory whose task and system prompt hashes match.
Use a fresh output directory to collect an independent sample.

## Validation

- 61 real-interpreter Monty tests passed after the structured Bash bridge fix.
- 7,842 offline tests passed before that isolated bridge correction.
- Ruff and MkDocs passed.
- The nine measured trials and 111 usage rows agree with the archived aggregates.
- Future instruction and return-discovery changes were not applied during the
  frozen measurement runs.
