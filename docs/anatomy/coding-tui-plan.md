# Coding TUI (Vulcano): implementation plan

Status: resume and slash-command increment implemented and validated on
`feat/coding-resume`, based on main at `ed581deb` (2026-10-01). The initial
`feat/coding-tui` worktree remains preserved. Vulcano remains outside the v1
release scope.

The first increment now provides `CodingSession`, an optional Textual CLI,
checkpoint-backed history, one visible sidebar and a reserved second sidebar,
extension panel/command registration, workspace read tools, and reviewed edits
to existing files. A follow-up increment adds user profiles, named API-key
accounts and per-thread storage. Local execution and provider-aware tools are implemented in the current
increment. Session discovery and builtin slash commands are implemented in the resume
increment. OAuth login and broader extension types remain future increments. RPC is not required for these increments.

## Configuration and persistence

The coding host uses `~/.msgflux/config.toml` for `default_model`,
`reasoning_effort`, `default_profile`, named profiles, subagent definitions and
the explicit `active_accounts` mapping. Profiles use `tools.active` and
`tools.deferred`; logical groups such as `workspace`, `process` and `agents`
are resolved against the selected model, available executor and live
permissions. A selected account is never silently replaced after an error.
The CLI accepts `--config`, `--profile`, `--model`, `--reasoning-effort`,
`--account` and dotted TOML `-c` overrides.

Named API-key accounts live in `~/.msgflux/accounts/<provider>/<alias>.json`.
They retain aliases and selected model history. The file containing the key
has private permissions; profile TOML and thread metadata do not contain keys.
OAuth login remains separate. The merged `openai-codex` provider reads an
existing configurable OAuth file; the harness does not create or refresh it.

New conversations use `~/.msgflux/threads/<thread_id>/` with checkpoint and
approval SQLite files plus minimal metadata. Checkpoints remain the only
conversation-history authority. Existing shared-store threads can still be
opened by explicit thread ID. A full import into per-thread files is a
separate migration because all runs, commits, events and message items must
be preserved. Prompt defaults stay versioned in the package; future prompt
files are explicit replacements or additions, never generated copies.

For configured subagents, `ModelGateway` is built with `fallback=False`.
The parent may select one advertised model on a call, but an unavailable
model or account does not trigger a silent switch. Before profile hot reload
is introduced, durable resume will need a manifest of the effective profile
and tool registry; edits to a user profile must not silently alter a pending
run.

## Decision and scope

- Use `msgflux.coding` for the Python package and `vulcano` for the application
  name and command. Use the optional dependency extra `coding`. Keep Textual
  imports inside the coding package; importing `msgflux` must not require it.
- Make the existing msgFlux `Agent`, execution scope, workspace, checkpoint,
  approval and event contracts the source of truth. The TUI is a host and a
  projection of those contracts. It must not maintain a second agent history.
- Start with one visible left sidebar for workspace and sessions. Define left
  and right sidebar regions from the first UI increment, with the right region
  initially hidden. The center holds transcript, tool activity and composer;
  header/status and bottom notices are separate regions.
- Deliver after v1 in small dependent PRs. The first usable release must support
  a local workspace, durable reopen, streaming, tool activity and approvals.
  Advanced customization can then grow without replacing the UI shell.

## Evidence and existing dependencies

The old `feat/vulcano-tui` branch (`d78c60c`, 2026-09-25) adds an independent
runtime, session store, `MsgfluxAgentAdapter`, Textual UI and an extension
manager. Its adapter anticipates agent event streaming while the current main
already exposes it. The branch predates the current durable runtime and differs
from main by about 15,900 lines. Review its widgets, commands, gallery,
extension lifecycle and tests as design references; do not merge it wholesale.

Current main offers:

| Contract | Current location | Coding consumer |
| --- | --- | --- |
| Run events | `Module.stream_events`, `ExecutionEvent`, `EventType` | Live transcript and activity |
| Live reconnect | `EventHub`, `ThreadWatcher`, `ThreadSnapshot` | Reattach to an active thread |
| Durable reconnect | `Agent.watch_commits`, `CheckpointStore` | Restore committed state and cursor |
| Identity | `ExecutionScope`, `AgentRun` | Thread/run lineage and cancellation |
| Files | `WorkspaceBackend`, `WorkspaceBinding`, `WorkspaceFilesystem` | File tree and tool environment |
| Edits | `WorkspaceEditor`, `PreparedFileChange` | Review diff and apply through host policy |
| Approvals | `AgentApprovals`, `ApprovalStore` | Pending decision view |
| Background work | `TaskStore`, `AgentInbox` | Task status and delivered notifications |
| Extensibility | `AgentExtension`, hooks, tool library | Agent behavior separate from UI extensions |

`stream_events()` owns the execution it starts; closing its consumer can abort
that run. A TUI detaching from a run must therefore use a host-owned run task
and `ThreadWatcher`/durable observation, or explicitly decide to cancel. Specify
and test this ownership before implementing reconnect. Durable commit events
are coarser than live token deltas; on reconnect, reconstruct from checkpoint
state and then follow live events. A snapshot and subscription must have a
defined handoff so events are neither lost nor displayed twice.

The local POSIX workspace advertises `cooperative_compare`, whereas
`WorkspaceEditor` defaults to `atomic_compare`. A coding host must select a
supported write guarantee explicitly, present the backend's actual guarantees,
and use existing grants and approvals. It must not imply OS isolation for
`LocalWorkspace` or grant authority based on restored UI state.

Pi's current main describes a durable, single-writer harness with a snapshot
plus event watch pattern; its `pico` branch is an older experimental snapshot
relative to that main. Tau demonstrates a Python/Textual boundary between
agent loop, coding session and TUI, including host-owned extension regions.
These are references for interface behavior, not dependencies or APIs to copy.

## Architecture

```text
vulcano CLI / app factory
  -> CodingHost (configuration, model, project trust, scope, stores, workspace)
  -> CodingSession (thread selection, prompt, cancel, resume, approvals)
  -> Agent + existing runtime and durable stores
  -> CodingProjection (snapshot + typed events -> immutable view state)
  -> Textual app (widgets, layout, input, theme)
```

`CodingHost` owns long-lived resources and closes them on exit. Each run has
an explicit scope (`namespace`, `thread_id`, `run_id`, principal, permissions,
environment, abort signal). `CodingSession` serializes prompts for one thread,
keeps the run alive independently of the screen, and exposes async commands and
observations. The projection knows msgFlux event and checkpoint formats; widgets
only know typed presentation state and intents. No widget writes checkpoint
data, approves a call, or opens a workspace directly.

The session catalog is a small host index of thread IDs, labels, project IDs
and last activity. Checkpoint state remains the conversation authority. The
current `CheckpointStore` exposes `list_runs` for a known thread but no general
thread enumeration. Confirm the required catalog semantics in PR 1, then add
the smallest store-neutral catalog contract in a separate PR.
Do not copy `vulcano.sessions.SessionStore` to persist message history. Keep
display preferences (pane widths, expanded sections, theme) outside agent
checkpoints, keyed by workspace/user and versioned independently.

For event application, use `(thread_id, run_id, source_path)` to distinguish
nested agents and tools. Maintain stable IDs for transcript rows, and separate
transient deltas from committed messages. On reconnect, replace the projection
from an authoritative snapshot, then apply events after its cursor. On an event
buffer overflow or stale cursor, resnapshot. Bound displayed tool output and
retain artifact references instead of putting entire outputs in widgets.

## UI shell and extensibility contract

The shell owns five stable regions: `left_sidebar`, `main`, `right_sidebar`,
`composer_accessories`, and `status`. Left and right have independent widths,
visibility and scroll containers. At narrow terminal widths, collapse the
right sidebar first, then the left; keep the composer usable. A second sidebar
can be enabled without restructuring the center area.

Offer a coding-specific extension API, separate from `AgentExtension`:

- `register_command`: name, help, completion and async handler; resolve name
  collisions deterministically and make unregistering explicit.
- `register_panel`: stable ID, title, region (`left`/`right`), order, widget
  factory and optional visibility predicate. The host owns panel framing,
  sizing, scroll and mount/unmount lifecycle.
- `register_renderer`: typed event/block target, priority and render function;
  fallback renderer always remains available.
- `register_composer_action`, `register_status_item`, `register_view` and
  `register_keybinding`: each returns a removal handle. Key collisions and
  reserved safety actions are validated at registration.
- `register_theme` and declarative configuration schema: plugins may supply
  tokens and defaults; the host enforces readability and can switch themes
  without rebuilding agent state.

Extension registration should be transactional: validate all contributions,
then publish one generation; on failure or reload, remove its handles and
mounted widgets. Plugins receive a narrow, capability-scoped `CodingContext`
for reading projections and dispatching host commands. They must not receive
raw checkpoint stores, approval stores or workspace bindings by default.
Distinguish trusted Python UI plugins from untrusted project content. Begin
with explicit Python registration and built-in components; consider entry-point
discovery and hot reload only after lifecycle and compatibility tests pass.
Version the extension API and document which types are stable.

The built-in left sidebar contains a workspace tree and session list. The
center renders user/assistant messages, reasoning summary (when available),
tool cards, progress, diff previews and errors. The composer supports multiline
input, history, slash command selection and clear send/cancel controls. The
right region is initially hidden but can host a file preview, task list or
extension panel. Approvals open a focused review view with the exact tool call,
resources, expiry and prepared diff when present; the decision is submitted
through the existing approval store and the run is resumed explicitly.

## Incremental PRs and implementation order

| PR | Planned files | Work and acceptance gate |
| --- | --- | --- |
| 1. Contract inventory | `docs/anatomy/coding-tui-plan.md`; focused tests under `tests/coding/` only if a missing runtime behavior is found | Confirm lifecycle, snapshot/event handoff, session enumeration, approval resume and workspace grants against main. Record any required core API changes as separate small PRs. |
| 2. Headless coding host | `src/msgflux/coding/{__init__,config,host,session,projection,types}.py`; `tests/coding/test_session.py`, `test_projection.py` | Open/reopen threads, start and cancel runs, stream and resnapshot, show tool/task state, recover from stale cursor. No Textual dependency in this layer. |
| 3. Minimal Textual app | `src/msgflux/coding/tui/{app,layout,widgets,theme}.py`, packaged `.tcss`; `src/msgflux/coding/cli.py`; `pyproject.toml`, `uv.lock`; `tests/coding/test_tui.py` | Optional `coding` extra and `vulcano` command; left sidebar, reserved right sidebar, transcript, multiline composer, status, keyboard and resize behavior. |
| 4. Files and approvals | `src/msgflux/coding/{workspace,approval}.py`; TUI tree, diff and approval widgets; integration tests | Open a configured workspace with live grants; browse safely; show prepared changes; approve/deny/expire and resume without executing a changed request. |
| 5. Extension surface | `src/msgflux/coding/extensions/{api,registry}.py`; panel and command widgets; examples and lifecycle tests | Transactional registration, teardown, collisions, plugin panel in either sidebar, theme switch, command/key handling. |
| 6. Release documentation and hardening | `docs/learn/coding.md`, `docs/learn/coding-extensions.md`, `mkdocs.yml`, `examples/coding_*.py`; remaining integration tests | Working setup and customization examples, documented limits and migration note from experimental Vulcano; cross-platform and terminal smoke checks. |

Each PR should be reviewable independently and state its dependency on the
previous PR. If PR 1 finds a missing core contract, land that contract with
its own tests before PR 2. No version bump belongs in these PRs.

## Validation and release gates

- Run focused `uv run pytest -v tests/coding` and existing agent, checkpoint,
  approval, workspace and event tests affected by each PR.
- Run `uv run ruff format --check`, `uv run ruff check`, and the relevant full
  `uv run pytest -v` gate from `CONTRIBUTING.md` before release. After docs
  changes, `uv sync --group doc` and `uv run mkdocs build`.
- Use Textual `run_test()`/`Pilot` for real interactions at normal and narrow
  terminal sizes, including a second registered sidebar panel. Test keyboard
  navigation and focus after mount/unmount.
- Integration scenarios: streamed model output; nested tool and background
  task; approval across process restart; diff mismatch/expiry; disconnect and
  reattach during an active run; stale cursor/event overflow; cancellation;
  terminal resize; absent optional extra; plugin registration failure and
  reload; workspace identity change and permission denial.
- A release candidate should run manually on supported terminals/OSes. The
  POSIX local workspace is unavailable on Windows; either supply another
  supported backend there or make the CLI report that limitation clearly.

## Main risks and decisions to settle at PR 1

1. **Run ownership:** avoid accidental cancellation when a Textual widget or
   event consumer disappears; make host shutdown and explicit cancel distinct.
2. **History authority:** determine thread enumeration and transcript projection
   from checkpoints before designing session navigation.
3. **Approval authority:** preserve policy version, resource identity, expiry
   and principal across prompt, pause and resume; never infer them from UI state.
4. **Workspace guarantees:** choose a supported compare mode per backend and
   explain cooperative comparison honestly in the review UI.
5. **Extension stability:** freeze only the small registration/context contract
   initially; keep Textual widget internals private until their lifecycle is
   proven.
6. **Packaging and OS support:** pin a Textual range compatible with Python
   3.11-3.14 after testing; ensure the optional extra and command fail with a
   helpful installation message when Textual is absent.

References: [experimental Vulcano branch](https://github.com/msgflux/msgflux/tree/feat/vulcano-tui),
[Pi Harness v2](https://github.com/earendil-works/pi/blob/main/packages/agent/docs/harness-v2.md),
[Pi Pico experiment](https://github.com/earendil-works/pi/blob/pico/packages/agent/docs/pico-v5.md),
[Tau architecture](https://github.com/huggingface/tau/blob/main/src/tau_coding/data/docs/architecture.md),
[Tau extension API](https://github.com/huggingface/tau/blob/main/src/tau_coding/extensions/api.py),
[Textual layout](https://textual.textualize.io/styles/dock/),
[Textual testing](https://textual.textualize.io/guide/testing/).

## Increment: provider workspace and local execution (2026-09-29)

Base updated by fast-forward to upstream/main 49cf941f. Implement in order:

1. Add `coding/workspace.py`: a host filesystem adapter and local subprocess
   executor, reusing workspace authorization, bounded reads and process cleanup.
   No OS sandbox is claimed. Broad filesystem capabilities allow new paths;
   read-only sessions exclude both mutation tools and process execution.
2. Update `coding/tools.py`, `config.py`, and a package-owned prompt module:
   OpenAI/OpenAI Codex use read, bash, apply_patch; other providers use edit/write.
   Bash replaces listing/search helpers. Commentary capability determines whether
   interactive sessions need send_user_message. Noninteractive `-p` forbids
   questions/progress messages. Main agent is named main; reasoning defaults to
   medium where supported.
3. Update `coding/cli.py` and session workspace discovery to consume local
   execution, default editing, explicit read-only, and one-shot print mode.
4. Add a task bucket using existing capture/dispatch and automatic task controls;
   retain controls internally, include wait/message when available. Evaluate a
   discriminated union schema without changing global task lifecycle semantics.
5. Render commentary in `coding/tui/app.py`; update `docs/learn/coding.md`.

Required tests: provider selection with Codex function transport, readonly and
noninteractive exclusions, new-file edits, executor output/cancellation limits,
task bucket routing/schema, CLI one-shot events, and commentary rendering.
Run focused coding/runtime tests, Ruff and MkDocs. A live Codex subscription
smoke test may use gpt-6-luna with medium reasoning in a disposable directory.

Risks: unsandboxed Bash inherits host access; broad grants must not be described
as isolation. Read-only subagents initially rely on tool selection, not an OS
sandbox. Changing main's name changes the prototype checkpoint namespace; check
resume compatibility. Strict provider schema normalization must preserve union
alternatives. OAuth login, new sandbox policies and global Task API redesign
remain separate increments.


### Increment validation and compatibility

The TaskTool bucket is exported from `tools.builtin`; the background-task guide
now documents its action union. A small change in `tools/types.py` reconciles
captured task controls when background sources disappear. Bash's docstring now
reflects configured executor policy instead of claiming mandatory isolation.
The existing prototype namespace is retained when reopening a coding_agent
thread; new threads use main. Existing exact-resource runtime backends are
unchanged; broad grants belong only to the coding host adapter.

Live validation: openai-codex/gpt-6-luna with medium reasoning read a temporary
file, changed it using apply_patch, verified contents with Bash, and persisted
the thread. No credentials were copied into harness state.

### Fix: Codex background notification request

Reproduced HTTP 400 after background Bash: the Codex endpoint rejects system
messages inside input. Update `models/providers/openai_codex.py` to map remaining
system-role history items to developer at the transport boundary, preserving
order, contents and the stable instructions prefix. Canonical history and other
providers remain unchanged. Add regression coverage to the existing Codex tests
and describe the conversion in the model guide. Re-run a real background Bash +
TaskTool wait using gpt-6-luna and the existing OAuth file.

The print-mode reproduction also exposed late async-generator cleanup across
execution contexts. Close nested session generators explicitly in the consumer
context (`coding/session.py`, `coding/cli.py`) and test early stream closure.

The TUI also receives an F4 copy action (`coding/tui/app.py`) to copy selected
text or the displayed conversation, with a focused clipboard test.

## Resume and slash commands increment (2026-10-01)

Continue in `/tmp/msgflux-coding-resume`, branch `feat/coding-resume`, based on
main after #202–#204. Preserve the earlier dirty TUI worktree. Import its coding
host/UI as the starting point and adapt to AgentWorkspace; do not copy old
provider/runtime patches that are already merged.

### Implementation order and affected files

1. `coding/workspace.py`, `tools.py`, `session.py`, `cli.py`: use the current
   injected AgentWorkspace and tool APIs, retain host cwd and permission policy.
2. `coding/storage.py`, new `coding/host.py`: discover valid per-thread stores,
   remember project location, reopen existing threads without creating phantom
   sessions, and own/close resources on successful session changes. Checkpoints
   remain the sole history authority. Legacy shared stores remain explicit.
3. `coding/session.py`: expose saved runs and restore a chosen run snapshot;
   require explicit continuation of unfinished runs using existing Agent guards.
4. `coding/tui/app.py`, a session picker: `/help`, `/resume [thread-id]`, `/new`,
   `/session`, `/runs`, `/continue [run-id]`, `/sidebar`, `/quit`, plus existing
   extension commands. Provide suggestions and deterministic collision rules.
   Restore messages, tool calls/results and pending approvals on reopen.
5. Update `docs/learn/coding.md`, optional extra/entry point/navigation and tests.

### Risks and required verification

- Never turn a missing thread ID into a new conversation during resume.
- Session commands never reach the model or its history; reject switches while
  a run owns the stream, and close old resources only after a new session opens.
- Failed history/model/workspace restoration remains visible; resource drift
  and uncertain command receipts must keep the runtime fail-closed behavior.
- Do not bypass approvals or repeat tool calls while rendering checkpoints.
- Test real SQLite close/reopen and native tool history, saved-run ordering,
  interrupted-run continuation, paused approvals, malformed discovery entries,
  unknown slash commands, selector cancellation and repeated session switching.
- Use Textual Pilot for interactions; run focused coding/runtime checks, Ruff,
  the durability gate and MkDocs. Optional live-model smoke tests must not
  replace deterministic restart tests.

Pi and Tau reference implementations both distinguish `/resume` selection,
`/new` and `/session`. Keep that interaction style. Arbitrary revision rewind,
branching trees, OAuth UI, hot profile reload and background task ownership UI
are subsequent increments, rather than unsafe partial implementations here.

The import also carries three required prerequisites: task bucket cleanup in
`tools/types.py`, visible background command arguments in `runtime/background.py`,
and honoring explicit credential resolvers without demanding an environment key
in `models/openai_compatible.py`. These are retained behavior from the previous
TUI increment or a defect exposed by migration, not new provider features.

A spawned-process test exposed the initial-request durability gap: the ordinary
Agent does not checkpoint the initial prompt before generation. A coding-only
`CodingCheckpointExtension` uses the existing `transform_context` hook and Agent
checkpoint builder before each provider request. No default Agent contract is
changed. The first generation can now recover its original user turn after an
abrupt controller exit. Interrupted runs are terminal according to the runtime;
failed/running/paused runs require explicit continuation. Even terminal states
with unresolved command evidence cannot start a new coding turn.

### Validation of the resume increment

- 86 coding tests passed, including actual SQLite close/reopen, native tool
  history, restored approval review/resume, searchable session/command pickers,
  actual Tab completion and safe failed session replacement.
- A separate spawned worker exits abruptly during the first provider generation;
  the new host resumes the same run ID and original single user turn.
- An unresolved command receipt in a terminal checkpoint blocks a new turn
  before the model is called. Active background tasks block session replacement.
- Read-only reopening preserves resource identity while reducing live grants.
- Full offline suite: 3,826 passed, 33 skipped, two pre-existing warnings.
- Required durability gate: 148 passed. Ruff lint/format and strict MkDocs passed.
- No paid model calls were used; real local filesystem/process and SQLite
  integration complement deterministic model responses and Textual Pilot.
- Deferred: background task recovery controls and arbitrary checkpoint rewind/
  branching. `/resume` reopens saved threads; `/runs` lists their saved turns.

## Terminal interaction review (2026-10-02)

Baseline committed as `d85939d8`. Address the review in a separate increment.

1. Move extension records/registry out of `extensions/__init__.py`; introduce
   declarative builtin command registration in `coding/tui/commands.py`.
2. Update `tui/app.py` composer: Enter submits, Shift/Alt+Enter inserts a line,
   remove Send button and filesystem sidebar, place selectable slash suggestions
   above the composer, navigate all entries with arrows, preserve picker keys.
3. Add a clipboard module: prefer native platform clipboard when available,
   terminal OSC52 fallback with honest feedback, explicit `/copy --file PATH`
   export, selected text/right-click and keyboard copy.
4. Normalize structured tool results before `tool.end` publication in the shared
   ToolLibrary pipeline without changing tool returns. Use existing msgspec
   conversion for dataclasses and Structs. Test sync and async paths.
5. Render tool lifecycle in one card keyed by run/source/call ID, pair durable
   history calls/results, and render shell stdout/stderr/status as readable text.
6. Update coding/event documentation and tests: real key events, all command
   options including quit, selection/right-click, native clipboard failure and
   file fallback, paired/interleaved tools, structured event results, and resume.

Risks: terminal clipboard protocols cannot acknowledge copy; do not claim native
clipboard success after a failed helper. Enter must not hijack modal inputs or
approval buttons. A selected slash command must never reach the model. Tool
results must retain 0/false/empty values, not be selected by truthiness. Output
conversion must not replace the original result supplied to hooks or callers.
Run focused Coding/runtime tests, Ruff/MkDocs and offline regression after source
is frozen. Whole-user-turn collapsing remains a subsequent UI enhancement.

### Review: timestamps and unused conversations

- Reuse `utils.time.utc_now_isoformat` in `coding/storage.py`.
- Add an in-memory draft session (`coding/draft.py`), selected by the CLI host.
  Opening the TUI, inspecting history and running local slash commands do not
  instantiate models, workspaces or thread databases for a new conversation.
- Activate the existing durable session factory on the first prompt; retain
  immediate validation when reopening an existing thread.
- Verify unopened drafts, draft replacement, first-prompt persistence and approval
  controller activation using SQLite and a fake model. Document lazy creation.

### Review validation

- Coding/UI plus structured tool events: 119 tests passed.
- Offline regression: 3,859 passed, 33 skipped, two existing warnings.
- Mandatory checkpoint/approval/durability gate: 148 passed.
- Ruff lint and format check passed (648 Python files); strict MkDocs passed.
- A final command metadata adjustment returns `/copy` to Ready after export;
  its keyboard-driven regression is included in the focused UI tests.
- No provider requests or paid model calls were needed. Real SQLite stores,
  local Bash execution, subprocess crash recovery and Textual Pilot cover the
  relevant integration boundaries. Native clipboard acknowledgement is mocked;
  OSC52 delivery still depends on the user's terminal, with explicit file export.

### Next review topic: tool extensions and CLI selection

Proposal, pending the ongoing coding review:

- Document custom class tools supplied through `AgentExtension.tools()` and
  preserve the extension's existing lifecycle, hooks and registration ownership.
- Extend `CodingExtensions` with a tool catalog, so optional tools can be
  discovered and selected by profile rather than requiring edits to the builtin
  resolver's concrete map. Load extension declarations before resolving tools.
- Keep `profiles.<name>.tools.active` and `deferred` authoritative for selection
  and loading mode. Registering a tool makes it available for selection; it does
  not automatically enable it. Builtins and extension tools share name validation.
- Consider `--tools workspace,my_tool` and
  `--deferred-tools web_fetch,web_search` as per-run selection overrides. Define
  precedence and overlap behavior explicitly before implementation; do not mutate
  the user's TOML file. Read-only and executor requirements still apply.
- Include executable examples for plain AgentExtension use, CodingExtensions
  registration, TOML selection and CLI selection. Test discovery, duplicate names,
  selected loading mode, profile/CLI precedence, lifecycle and permission checks.

These extension and CLI APIs were proposed here and implemented below.

### Estudo de extensões e seleção pela CLI

O estudo foi concluído em `coding-extensions-study.md`, com referências oficiais
pinadas de Pi v1 e Tau e plano incremental de registro por `CodingExtensions` e
seleção por flags. Pi é a referência principal de API e UX; Tau serve de apoio
para decisões de implementação em Python. A implementação segue na seção abaixo.

### Implementação: tools por extensão e seleção CLI

- `CodingExtensions.register_tool` registra classes/factories sem instanciar.
- `--tools` e `--deferred-tools` selecionam builtins e contribuições de extensão
  usando o mesmo resolver; flags substituem a seleção do perfil nesta execução.
- Metadados são carregados em TUI e print, e lote parcial não é publicado.
- Instâncias selecionadas são por sessão; configuração da classe não é mutada.
- Recursos retornados por factories entram no fechamento sync/async da sessão,
  incluindo erro parcial. Factory que falha antes de retornar cuida dos próprios
  recursos. Erro de cleanup não substitui o erro original de construção.
- Apenas active/deferred; queued inbox, codemode e novos backends ficam para depois.
- Testes focados cobrem registro, CLI, SQLite, workspace injetado, recursos,
  schemas e sessões distintas.
- Validação: suíte offline com 3.888 testes passando (33 skips e dois warnings
  existentes), 148 testes de durabilidade e MkDocs strict. Após o ajuste final
  dos handles de registro, os 137 testes de coding passaram novamente. Ruff
  check/format e git diff --check passaram.
