# Experimental Remote Vulcano

This frontend lives on the experimental `feat/coding-service-tui` branch.
It connects to the native Agent service through `AgentSessionClient`; it does
not construct Agents, read checkpoint databases, or own model executions.
The TUI is not part of the v1 main delivery.

## Try It

From this branch's checkout:

```bash
uv sync --group dev --extra service --extra coding
uv run vulcano --workspace /absolute/project --state-dir ~/.msgflux-vulcano-test
```

The separate state directory is useful for testing without sharing configuration
with an existing local service. The default remains `~/.msgflux` when omitted.
Local discovery starts the trusted coding backend when needed and reuses the
same healthy backend from other project directories. The default backend uses
`openai-codex/gpt-6-luna` with medium reasoning when no model is configured, and
the provider reads its existing default `~/.codex/auth.json`. No login flow or
automatic provider/model fallback is added by the TUI.

The left sidebar shows the thread ID and saved sessions. Reopen one with:

```bash
uv run vulcano --thread THREAD_ID --state-dir ~/.msgflux-vulcano-test
```

The saved thread selects its workspace and profile even when the frontend starts
elsewhere. An explicit `--profile` or `--read-only` that conflicts with the saved
thread is rejected; start a new conversation to choose a different mode.

## Backend Configuration

The backend reads `<state-dir>/config.toml` when it starts. For example:

```toml
default_model = "openai-codex/gpt-6-luna"
default_profile = "lite"
reasoning_effort = "medium"

[profiles.lite.tools]
active = ["workspace"]
deferred = ["web_fetch"]

# Optional server-owned approval rules. Use concrete tool names.
[profiles.review]
approvals = ["apply_patch"]

[profiles.review.tools]
active = ["workspace"]
```

Workspace tools are resolved using the provider's existing capabilities. Bash
background execution adds the task bucket. Provider credentials and permission
grants remain in the backend. Its default local workspace uses direct host paths;
it is not an OS sandbox. Use `--read-only` to select the published profile that
denies editing/process permissions and omits Bash.

Configuration changes do not reconfigure a running daemon or rewrite an existing
thread's profile. Apply them when starting a backend. The separate test state
directory also permits another backend configuration without replacing a live
owner. The original experimental branch retains the earlier CLI extension,
account, and subagent work; this remote entry point does not yet expose all of
those controls. Profiles requesting configured subagents require a host factory
that supplies them.

`approvals` is an optional list of concrete tool names owned by the backend
configuration. Each name must resolve to an available executable foreground
tool in the selected profile. For example, the `review` profile above pauses
before `apply_patch` runs and stores its approval journal at
`threads/<thread-id>/approvals.sqlite3`. The read-only profile keeps its
workspace permission ceiling and does not enable the profile's approval policy.
The local service acts as the trusted reviewer under the stable `local-user`
principal; no frontend flag chooses that identity. Approval review shows the
prepared change and tool call through the service API; it does not expose raw
tool arguments. After every pending request has a decision, explicitly resume
the run. Recording a decision does not resume execution automatically.

With no `approvals` entry, tools keep their existing execution behavior and the
backend does not create an approval journal.

Start the review profile with `uv run vulcano --profile review`. Restart the
backend after changing its configuration; an existing daemon keeps the policy
it loaded at startup.

Applications can supply another trusted service factory:

```bash
uv run vulcano --factory my_backend:create_service --profile main \
  --workspace /absolute/project --state-dir ~/.my-coding-service
```

The `--profile` selects a published Agent ID. The factory module must be
importable from a stable installed location or the initial launch directory.

## Terminal Interaction

- **Enter** sends. **Alt/Shift+Enter** inserts a newline.
- Slash hints appear above the composer; use arrows and Tab to navigate/complete.
- `/help`, `/session`, `/runs`, `/resume`, `/new`, `/continue`, `/copy`, and `/quit`
  use registered command declarations.
- Typing while a foreground run is active sends a steering message through its
  AgentInbox. **Escape** explicitly interrupts the selected root run.
- Selection/right click and F4 request clipboard copy. `/copy --file PATH`
  exports locally when the desktop/terminal clipboard mechanism is unavailable.
- Tool calls and results remain grouped during live observation. On reload, the
  transcript restores user messages, commentary, and final answers from the
  snapshot; it does not replay tool calls or reasoning.

New windows bind a thread lazily. An untouched conversation creates no checkpoint
directory or model instance. Observation attaches immediately when reopening a
saved conversation, or before admitting the first prompt in a new one.

## Disconnect And Resume

Observation uses `watch()` independently of prompt admission. Closing the TUI
detaches its observer and HTTP client; the shared service and admitted run keep
running. Reconnects replace the view with a fresh snapshot and do not replay
missed event deltas. If a local daemon returns at a new endpoint, the host
rediscovers it without resending the prompt. An uncertain admission displays its
request ID for reconciliation rather than automatically retrying the POST.

`/continue` requests the service's existing recovery path. It does not bypass
worker-quiescence or command receipt checks. Approval cards show the server's
verified diff and offer **Approve** and **Deny**. **Resume run** becomes available
after every request in the batch is resolved. Reopening a paused conversation
fetches its reviews again from the service. The frontend never reads local
approval or checkpoint stores to make a decision.

## Integration Checks

```bash
uv run pytest -q tests/coding/test_remote_host.py \
  tests/coding/test_remote_backend.py tests/coding/test_remote_tui.py \
  tests/coding/test_remote_socket.py tests/coding/test_remote_approvals.py \
  tests/coding/test_remote_approval_socket.py
MSGFLUX_LIVE_CODING=1 uv run pytest -q tests/coding/test_remote_live.py
```

The opt-in smoke uses the existing Codex subscription, the actual local daemon,
Textual Pilot, `gpt-6-luna`, background Bash, and the task bucket. Offline socket
tests use real HTTP/SSE and tools with deterministic model responses, including
frontend detachment during a run and history restoration in another window.
