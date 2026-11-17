# Monty tools implementation plan

Branch: `feat/monty-tools`, based on upstream main `49d50879`.

## Files and implementation order

1. Inspect tools/buckets, canonical definitions, scoped handles, execution,
   approval/permission pipeline and Monty 1.6 API.
2. Optional `monty` dependency in `pyproject.toml`; update local `uv.lock` with uv
   (the repository ignores lock files). Builtin exports remain dependency-lazy.
3. `tools/builtin/monty.py`: shared execution for `MontyTool` and
   `MontyToolBucket`. Read-only discovery from canonical metadata; calls through
   handles, no raw executor exposed. Helper collisions use `tools.call`.
4. `tools/handles.py`: preserve runtime inputs for ordinary operator calls and
   add copied handles with unique nested invocation identities.
5. `tools/builtin/_monty_state.py`: latest-state files, bounded reads/writes,
   private permissions, atomic replacement and cross-process session locks.
   Persist all active JSON-compatible variables after each successful execution;
   overwrite changed names and never accumulate old values or source. Recreate
   the interpreter for each call. Export/import/delete are explicit host APIs.
6. `tools/types.py`: include non-deferred bucket members when discovering
   background sources, preserving task controls for captured background tools.
7. `tests/tools/test_monty.py`: real interpreter, both compositions, visibility,
   canonical schema/guidance, runtime inputs/hooks, permissions/filter, recursion,
   background dispatch, unique IDs, all active variables, bounded storage,
   restart in an independent process, lock contention, invalid state, output and
   time limits, approval fail-closed and control/cancellation/process cleanup.
8. `.github/workflows/ci.yml`: install the optional Monty extra in the Python
   test matrix so the real-interpreter tests run in CI rather than skip.
9. `docs/learn/nn/agent/tools/monty.md`, builtin/bucket/dependency docs and
   `mkdocs.yml`: examples, discovery, state lifecycle, limits and boundaries.
   Ruff, focused tools/runtime suites, durability gate, offline suite and MkDocs.

## Decisions and risks

- User selected data-only persistence. Native dumps grew from 1 KB to 10 KB over
  200 assignments to the same name, so they are not used. Functions/classes and
  non-JSON data are explicitly reported as not persisted. Every call restores
  all previous values, preserving names it does not modify.
- Interrupted Python stacks cannot be resumed with data-only persistence. Mark
  execution uncertain before starting; propagate control exceptions and require
  reconciliation/reset when an external call may have happened. Errors before
  admitting a host tool retain the previous data and allow corrected code.
  No automatic snippet replay.
- State files and Agent checkpoints are independent transactions. Persist the
  latest completed result to deduplicate a retry of that same invocation.
- SDK tools must not depend on AgentService. Existing service file helpers import
  the service package and require POSIX; account helpers are account-specific.
  The small private file adapter uses the same fsync/replace pattern and handles
  POSIX and Windows advisory locks. This is intentional local duplication.
- Other buckets cannot be captured by current generic bucket rules. Ordinary
  captured tools preserve optional background dispatch; model-loop feedback
  such as handoff/return-direct is rejected.
- Existing approval guards remain authoritative; nested calls without approved
  bindings fail closed. Interactive per-nested-call approval is future work.
- No changes to coding defaults, AgentTool API, provider configuration or
  release version. The user requested commits and a remote research branch after
  reviewing the results; PR creation and merge are outside this archival step.

## Archiving the exploration

Keep two focused commits: the Monty implementation with its tests and user docs,
then a research report, reproducible benchmark script and compact measurements.
Archive the nine per-run results and the 111 per-response usage rows under
`docs/anatomy/experiments/monty-tools-20261117/`. Do not version credentials,
local paths to credential files, generated candidate workspaces, test caches or
full model traces. Existing complete local artifacts remain outside the repo.
The benchmark script must accept its output directory and optional env-file via
CLI, and provide an offline preflight that makes no provider calls.

Validation already completed: 7,842 offline tests before the structured-shell
bridge fix, 61 Monty tests after that fix, Ruff, MkDocs, and nine measured live
DeepSeek Flash runs. Archival script changes require Ruff, offline preflight and
an aggregate-data consistency check; documentation is checked with MkDocs.
