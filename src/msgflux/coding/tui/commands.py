"""Builtin slash commands, registered with the same contract as extensions."""

from functools import partial

from msgflux.coding.extensions import CommandSpec


async def help_command(app, _args):
    return (
        "\n".join(f"/{spec.id} — {spec.description}" for spec in app.command_specs())
        + "\nEnter send · Alt/Shift+Enter newline · ↑/↓ commands"
        + " · Tab complete · Esc cancel"
    )


async def resume_command(app, args):
    if args:
        await app._switch_session(args)
    else:
        return await app._show_session_picker()


async def new_command(app, _args):
    await app._switch_session(None)


async def session_command(app, _args):
    latest = getattr(app.coding_session, "latest_run", lambda: None)()
    return (
        f"Thread: {app.coding_session.thread_id}\nWorkspace: {app.workspace}\n"
        f"Latest checkpoint: {latest or 'No runs yet'}"
    )


async def runs_command(app, _args):
    runs = getattr(app.coding_session, "runs", lambda: ())()
    return (
        "\n".join(f"{item['run_id']} · {item['status']}" for item in runs)
        or "No saved checkpoints"
    )


async def continue_command(app, args):
    latest = app.coding_session.latest_run()
    run_id = args or (latest["run_id"] if latest else None)
    if run_id is None:
        raise ValueError("No saved run to continue")
    await app._consume_events(app.coding_session.resume(run_id))


async def sidebar_command(app, _args):
    app.action_toggle_left()


async def copy_command(app, args):
    return await app.copy_transcript(args)


async def quit_command(app, _args):
    app.exit()


# Metadata, completion and dispatch all consume this single declaration list.
_DEFINITIONS = (
    ("help", help_command, "List all commands and shortcuts", False, False),
    (
        "resume",
        resume_command,
        "Reopen a saved session: /resume [thread-id]",
        True,
        True,
    ),
    ("new", new_command, "Start a new conversation", False, True),
    (
        "session",
        session_command,
        "Show current thread and saved run status",
        False,
        False,
    ),
    ("runs", runs_command, "List saved checkpoints", False, False),
    (
        "continue",
        continue_command,
        "Continue unfinished run: /continue [run-id]",
        True,
        True,
    ),
    ("sidebar", sidebar_command, "Toggle sidebar", False, False),
    (
        "copy",
        copy_command,
        "Copy selection/conversation: /copy [all|last] [--file PATH]",
        True,
        False,
    ),
    ("quit", quit_command, "Exit Vulcano", False, True),
)


def builtin_commands(app) -> tuple[CommandSpec, ...]:
    return tuple(
        CommandSpec(
            name, partial(handler, app), description, accepts_args, preserve_status
        )
        for name, handler, description, accepts_args, preserve_status in _DEFINITIONS
    )
