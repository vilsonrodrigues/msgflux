"""Exercise the real Monty worker and canonical library execution pipeline."""

import asyncio

import msgspec
import pytest

from msgflux.exceptions import TaskPauseRequestedError
from msgflux.nn import ToolLibrary
from msgflux.runtime.context import ExecutionScope, execution_context
from msgflux.runtime.permissions import PermissionSet
from msgflux.tools.builtin import MontyTool, MontyToolBucket
from msgflux.tools.config import tool_config
from msgflux.tools.handles import ToolLibraryHandle
from msgflux.tools.policy import ToolFilterPolicy, bind_tool_filter_policy

monty = pytest.importorskip("pydantic_monty")


@pytest.fixture(autouse=True)
def state_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "msgflux.tools.builtin.monty._default_state_directory",
        lambda: tmp_path / "monty",
    )
    return tmp_path / "monty"


def add(a: int, b: int = 1) -> int:
    """Add two integers."""
    return a + b


add.usage_guidance = "Use for exact integer arithmetic."


def invoke(library, code, call_id="m1", **kwargs):
    response = library(
        [(call_id, "monty", {"code": "import tools\n" + code, **kwargs})]
    )
    call = response.tool_calls[0]
    assert call.error is None, call.error
    return call.result


@pytest.fixture(params=[MontyTool, MontyToolBucket], ids=["additive", "bucket"])
def composition(request):
    tool = request.param()
    library = ToolLibrary("python", [tool, add])
    return tool, library


def handle(library, call_id="host"):
    return ToolLibraryHandle(library).for_tool(tool_name="monty", tool_call_id=call_id)


def test_visibility_discovery_and_execution(composition):
    tool, library = composition
    assert library.get_tool_names() == (
        ["monty"] if isinstance(tool, MontyToolBucket) else ["monty", "add"]
    )
    assert invoke(library, "import tools\ntools.list()") == {
        "output": ["add"],
        "stdout": "",
    }
    description = invoke(library, 'tools.describe("add")', "m2")["output"]
    assert description["description"] == "Add two integers."
    assert description["usage_guidance"] == add.usage_guidance
    assert description["input_schema"] == dict(
        library.get_tool_definition("add").input_schema
    )
    assert invoke(library, "x=tools.add(a=20, b=22)\nprint(x)\nx", "m3") == {
        "output": 42,
        "stdout": "42\n",
    }
    assert invoke(library, "x + tools.call('add', a=1)", "m4")["output"] == 44
    assert (
        "Use print(...) to read tool results"
        in library.get_tool_definition("monty").description
    )
    schema = library.get_tool_definition("monty").input_schema
    assert "handle" not in schema["properties"]
    assert set(schema["properties"]) == {"code", "reset"}


def test_scope_isolation_export_restore_release(composition):
    tool, library = composition
    with execution_context(thread_id="thread-a", namespace="builder"):
        invoke(library, "value = 42\nvalue")
        exported = tool.dump_session(handle(library))
        tool.release_session(handle(library))
        tool.load_session(exported, handle(library))
        assert invoke(library, "value", "m2")["output"] == 42
    with execution_context(thread_id="thread-b", namespace="builder"):
        assert (
            invoke(
                library,
                "try:\n    value\n    exists=True\nexcept NameError:\n    exists=False\nexists",
            )["output"]
            is False
        )
    with execution_context(thread_id="thread-a", namespace="explorer"):
        assert (
            invoke(
                library,
                "try:\n    value\n    exists=True\nexcept NameError:\n    exists=False\nexists",
            )["output"]
            is False
        )
    with execution_context(thread_id="thread-a", namespace="builder"):
        assert (
            invoke(
                library,
                "try:\n    value\n    exists=True\nexcept NameError:\n    exists=False\nexists",
                "m3",
                reset=True,
            )["output"]
            is False
        )


def test_dynamic_catalog_and_filters(composition):
    _, library = composition
    with bind_tool_filter_policy(
        library, ToolFilterPolicy(allowed_names=frozenset({"monty"}))
    ):
        result = invoke(
            library,
            """
try:
    tools.call('add', a=4)
except RuntimeError as error:
    blocked = str(error)
blocked
""",
            "m2",
        )
        assert "not available" in result["output"]
        assert invoke(library, "tools.list()", "m3")["output"] == []
    library.remove("add")
    assert invoke(library, "tools.list()", "m4")["output"] == []
    library.add(add)
    assert invoke(library, "tools.list()", "m5")["output"] == ["add"]


def test_recursion_and_internal_argument_injection_are_blocked(composition):
    _, library = composition
    for code, expected in [
        ("tools.call('monty', code='1')", "not available"),
        ("tools.add(a=1, _runtime_arguments={})", "Runtime inputs"),
        ("tools.add(1, 2)", "keyword"),
    ]:
        result = invoke(
            library,
            f"""import tools
try:
    {code}
except RuntimeError as error:
    message = str(error)
message
""",
            code,
        )
        assert expected in result["output"]


def test_helper_name_collision_remains_callable():
    def listing(value: int) -> int:
        """A user tool whose name collides with discovery."""
        return value

    listing.__name__ = "list"
    library = ToolLibrary("collision", [MontyToolBucket(), listing])
    assert invoke(library, "import tools\ntools.call('list', value=42)")["output"] == 42


@pytest.mark.asyncio
async def test_async_tools_and_repeated_nested_ids(composition):
    _, library = composition
    calls = []

    @tool_config(runtime_inputs=("handle",))
    async def capture(value: int, *, handle) -> int:
        """Record the invocation identity."""
        await asyncio.sleep(0)
        calls.append(handle.get_tool_call_id())
        return value

    library.add(capture)
    response = await library.acall(
        [
            (
                "parent",
                "monty",
                {"code": "import tools\n[tools.capture(value=i) for i in range(3)]"},
            )
        ]
    )
    assert response.tool_calls[0].error is None
    assert response.tool_calls[0].result["output"] == [0, 1, 2]
    assert len(set(calls)) == 3
    assert all(call.startswith("parent:") for call in calls)


@pytest.mark.asyncio
async def test_control_pause_is_not_swallowed_or_replayed(composition):
    tool, library = composition
    calls = []

    def effect(value: int) -> int:
        """Pause before the second effect."""
        if value == 2:
            raise TaskPauseRequestedError("Host pause")
        calls.append(value)
        return value

    library.add(effect)
    code = "import tools\na=tools.effect(value=1)\nb=tools.effect(value=2)\na+b"
    with pytest.raises(TaskPauseRequestedError):
        await tool.acall(code, handle=handle(library, "pending"))
    assert calls == [1]
    replacement = type(tool)(state_directory=tool.state_directory)
    rebuilt = ToolLibrary("python", [replacement, effect])
    # No suspended interpreter is retained: callers must reconcile and reset.
    # Reissuing the same call must never replay the first effect silently.
    exported = tool.dump_session(handle(library))
    tool.load_session(exported, handle(library))
    with pytest.raises(RuntimeError, match="reset=True"):
        await tool.acall(code, handle=handle(library, "pending"))
    assert calls == [1]
    assert (await tool.acall("42", reset=True, handle=handle(library)))["output"] == 42


@pytest.mark.asyncio
async def test_same_session_rejects_concurrent_feeds_and_cleans_up():
    started, unblock = asyncio.Event(), asyncio.Event()

    async def wait() -> str:
        """Wait for test synchronization."""
        started.set()
        await unblock.wait()
        return "ready"

    tool = MontyTool()
    library = ToolLibrary("busy", [tool, wait])
    pending = asyncio.create_task(
        tool.acall("import tools\ntools.wait()", handle=handle(library))
    )
    await asyncio.wait_for(started.wait(), timeout=10)
    try:
        with pytest.raises(RuntimeError, match="busy"):
            await tool.acall("1", handle=handle(library))
        with pytest.raises(RuntimeError, match="busy"):
            tool.release_session(handle(library))
    finally:
        unblock.set()
        await asyncio.wait_for(pending, timeout=10)
    assert (await tool.acall("2", handle=handle(library)))["output"] == 2


def test_permissions_are_enforced_inside_monty(composition):
    _, library = composition
    calls = []

    @tool_config(required_permissions=("execute",))
    def restricted() -> str:
        """Execute a protected action."""
        calls.append("executed")
        return "ok"

    library.add(restricted)
    with execution_context(
        scope=ExecutionScope(permissions=PermissionSet(frozenset({"read"})))
    ):
        result = invoke(
            library,
            """import tools
try:
    tools.call('restricted')
except RuntimeError as error:
    message = str(error)
message
""",
        )
    assert "not available" in result["output"].lower()
    assert calls == []


def test_limits_preserve_last_committed_variables():
    tool = MontyTool(max_duration=0.01)
    library = ToolLibrary("limits", [tool])
    tool("x=42", handle=handle(library))
    with pytest.raises(monty.MontyRuntimeError):
        tool("x=0\nwhile True:\n    pass", handle=handle(library))
    assert tool("x", handle=handle(library))["output"] == 42


def test_optional_dependency_is_not_required_for_builtin_import(monkeypatch):
    import builtins
    from msgflux.tools.builtin import monty as implementation

    original = builtins.__import__

    def importing(name, *args, **kwargs):
        if name == "pydantic_monty":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    with pytest.raises(ImportError, match=r"msgflux\[monty\]"):
        implementation.MontyTool()


def test_new_tool_is_discoverable_in_a_restored_session(composition):
    _, library = composition
    invoke(library, "import tools\n1")

    def multiply(a: int, b: int) -> int:
        """Multiply two integers."""
        return a * b

    library.add(multiply)
    assert invoke(library, "tools.multiply(a=6, b=7)", "m2")["output"] == 42


@pytest.mark.asyncio
async def test_cancellation_requires_reset_and_leaves_no_worker():
    import psutil

    started = asyncio.Event()

    async def wait() -> None:
        """Wait indefinitely until cancelled."""
        started.set()
        await asyncio.Event().wait()

    before = {process.pid for process in psutil.Process().children(recursive=True)}
    tool = MontyTool()
    library = ToolLibrary("cancel", [tool, wait])
    pending = asyncio.create_task(
        tool.acall("import tools\ntools.wait()", handle=handle(library))
    )
    await asyncio.wait_for(started.wait(), timeout=10)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(pending, timeout=10)
    with pytest.raises(RuntimeError, match="reset=True"):
        await tool.acall("1", handle=handle(library))
    remaining = {process.pid for process in psutil.Process().children(recursive=True)}
    assert remaining <= before
    assert (await tool.acall("42", reset=True, handle=handle(library)))["output"] == 42


@pytest.mark.parametrize(
    "kwargs, code",
    [
        ({"max_output_bytes": 64}, "print('x'*1000)"),
        ({"max_output_bytes": 64}, "'x'*1000"),
        ({"max_state_bytes": 128}, "x='x'*1000\n1"),
    ],
)
def test_output_and_snapshot_limits_are_bounded(kwargs, code):
    tool = MontyTool(**kwargs)
    library = ToolLibrary("bounded", [tool])
    with pytest.raises((ValueError, monty.MontyRuntimeError)):
        tool(code, handle=handle(library))
    assert tool("1", handle=handle(library))["output"] == 1


@pytest.mark.asyncio
async def test_runtime_inputs_and_hooks_are_preserved(composition):
    from dataclasses import replace

    from msgflux.chat_messages import ChatMessages
    from msgflux.nn.hooks import Hook

    _, library = composition
    observed = []

    @tool_config(runtime_inputs=("message", "messages", "vars", "handle"))
    def observe(value: int, *, message, messages, vars, handle) -> int:
        """Observe the host-provided runtime inputs."""
        observed.append((message, messages, vars, handle.get_tool_call_id()))
        return value

    def increase(event):
        if event.tool_name == "observe":
            return replace(event, arguments={"value": event.arguments["value"] + 1})
        return event

    library.add(observe)
    Hook(event="before_tool", handler=increase).register(library)
    history = ChatMessages()
    history.add_user("original")
    response = await library.acall(
        [("parent", "monty", {"code": "import tools\ntools.observe(value=41)"})],
        message="envelope",
        messages=history,
        vars={"tenant": "a"},
    )
    assert response.tool_calls[0].error is None
    assert response.tool_calls[0].result["output"] == 42
    assert observed[0][:3] == ("envelope", history, {"tenant": "a"})
    assert observed[0][3].startswith("parent:")


def test_nested_approval_guard_fails_closed(composition):
    from msgflux.runtime import AgentApprovals, InMemoryApprovalStore
    from msgflux.tools.runtime import ToolIntent

    _, library = composition
    calls = []

    def protected() -> str:
        """A host-reviewed action."""
        calls.append("executed")
        return "done"

    library.add(protected)
    policy = AgentApprovals(InMemoryApprovalStore(), {"protected": "v1"}, "p1")
    code = "import tools\ntools.protected()"
    pending = {"requests": {}}
    with execution_context(
        scope=ExecutionScope(
            namespace="approval", thread_id="t", run_id="r", principal="user"
        )
    ):
        batch = policy.prepare(
            library,
            [ToolIntent(id="outer", name="monty", arguments={"code": code})],
            pending,
        )
        with batch.activate():
            response = library([("outer", "monty", {"code": code})])
    assert "No approval" in response.tool_calls[0].error
    assert calls == []


@pytest.mark.asyncio
async def test_optional_background_dispatch_remains_available_in_bucket():

    @tool_config(allow_background=True)
    async def job(value: int) -> int:
        """Double a value in a task."""
        return value * 2

    tool = MontyToolBucket()
    library = ToolLibrary("background", [tool, job])
    code = "import tools\ntools.job(value=21, run_in_background=True)"
    result = await tool.acall(code, handle=handle(library))
    assert "task_id=" in result["output"]
    # Automatic task controls are captured too; they keep their runtime context.
    assert "task_wait" in library.get_bucket_tool_names("monty")
    assert library.get_tool_names() == ["monty"]
    task_id = result["output"].split("'", 2)[1]
    result = await tool.acall(
        f"import tools\ntools.task_wait(task_id={task_id!r}, timeout=5.0)",
        handle=handle(library, "wait"),
    )
    assert result["output"] == 42


def test_files_restore_across_tool_instances_and_overwrite_values(composition):
    tool, library = composition
    with execution_context(thread_id="durable", namespace="agent"):
        invoke(library, "x = 1\nx")
        invoke(library, "x = 2\nx", "m2")
        replacement = type(tool)(state_directory=tool.state_directory)
        rebuilt = ToolLibrary("python", [replacement, add])
        assert invoke(rebuilt, "x", "recovered")["output"] == 2
        files = list(tool.state_directory.glob("*.json"))
        assert len(files) == 1
        assert files[0].stat().st_mode & 0o777 == 0o600
        replacement.release_session(handle(rebuilt))
        assert not files[0].exists()


def test_disk_lock_excludes_other_instances(composition):
    tool, library = composition
    with execution_context(thread_id="locked", namespace="agent"):
        replacement = type(tool)(state_directory=tool.state_directory)
        rebuilt = ToolLibrary("python", [replacement, add])
        with tool._files.locked(tool._key(handle(library))):
            with pytest.raises(RuntimeError, match="busy"):
                replacement("1", handle=handle(rebuilt))


@pytest.mark.parametrize("contents", [b"corrupt", b""])
def test_corrupt_file_fails_closed(composition, contents):
    tool, library = composition
    invoke(library, "1")
    path = next(tool.state_directory.glob("*.json"))
    path.write_bytes(contents)
    with pytest.raises(msgspec.DecodeError):
        tool("1", handle=handle(library))
    # Explicit host deletion remains usable even if the data cannot be decoded.
    tool.release_session(handle(library))
    assert tool("42", handle=handle(library))["output"] == 42


def test_completed_invocation_is_not_replayed(composition):
    tool, library = composition
    calls = []

    def effect() -> int:
        """Record an external effect."""
        calls.append(1)
        return len(calls)

    library.add(effect)
    code = "import tools\ntools.effect()"
    assert invoke(library, code)["output"] == 1
    assert invoke(library, code)["output"] == 1
    assert calls == [1]
    replacement = type(tool)(state_directory=tool.state_directory)
    rebuilt = ToolLibrary("python", [replacement, effect])
    # Anonymous tool use intentionally has no cross-instance durable identity.
    with execution_context(thread_id="durable", namespace="agent"):
        assert invoke(library, code)["output"] == 2
        assert invoke(rebuilt, code)["output"] == 2
        assert calls == [1, 1]


def test_uncertain_file_requires_explicit_reset(composition):
    tool, library = composition
    invoke(library, "x=1\nx")
    # Simulate the durable marker left when the process dies mid-execution.
    with tool._locked_state(handle(library), save=True) as (state, _path):
        state.invalid = True
    with pytest.raises(RuntimeError, match="reset=True"):
        tool("x", handle=handle(library))
    assert tool("42", reset=True, handle=handle(library))["output"] == 42


def test_all_active_variables_survive_intermediate_calls(composition):
    tool, library = composition
    with execution_context(thread_id="long", namespace="agent"):
        invoke(library, 'x = "valor"\n1', "50")
        invoke(library, 'y = {"items": [1, True, None]}\ny', "51")
        assert invoke(library, "print(x)\ny['items']", "52") == {
            "output": [1, True, None],
            "stdout": "valor\n",
        }
        invoke(library, 'x = "novo"\nx', "53")
        replacement = type(tool)(state_directory=tool.state_directory)
        rebuilt = ToolLibrary("python", [replacement])
        assert invoke(rebuilt, "[x, y['items']]", "54")["output"] == [
            "novo",
            [1, True, None],
        ]
        invoke(rebuilt, "x=None\ny", "55")
        assert invoke(rebuilt, "x", "56")["output"] is None


def test_state_does_not_accumulate_old_values_or_source(composition):
    tool, library = composition
    invoke(library, "x = 0\nx", "0")
    path = next(tool.state_directory.glob("*.json"))
    original_size = path.stat().st_size
    for i in range(1, 21):
        invoke(library, f"x = {i}\n# {('discarded source ' * 100)}\nx", str(i))
    assert path.stat().st_size < original_size + 32
    assert len(list(tool.state_directory.glob("*.json"))) == 1
    record = msgspec.json.decode(path.read_bytes())
    assert record["variables"] == {"x": 20}
    assert b"discarded source" not in path.read_bytes()
    assert not hasattr(tool, "_states")


def test_non_data_values_are_explicitly_omitted(composition):
    _, library = composition
    result = invoke(
        library,
        """
x = {"keep": [1, True, None]}
def f():
    return 42
nested = [f]
raw = b"bytes"
tuple_value = (1, 2)
1
""",
    )
    assert result["output"] == 1
    assert set(result["not_persisted"]) == {"f", "nested", "raw", "tuple_value"}
    assert invoke(library, "x", "m2")["output"] == {"keep": [1, True, None]}


def test_variables_restore_in_a_fresh_python_process(composition):
    import subprocess
    import sys

    tool, library = composition
    with execution_context(thread_id="process", namespace="agent"):
        invoke(library, "x=42\nx")
    script = """
import sys
from msgflux.nn import ToolLibrary
from msgflux.tools.builtin import MontyTool, MontyToolBucket
from msgflux.runtime.context import execution_context
cls = MontyToolBucket if sys.argv[2] == "bucket" else MontyTool
library = ToolLibrary("python", [cls(state_directory=sys.argv[1])])
with execution_context(thread_id="process", namespace="agent"):
    response = library([("new-process", "monty", {"code": "x"})])
    call = response.tool_calls[0]
    assert call.error is None, call.error
    assert call.result["output"] == 42, call.result
"""
    subprocess.run(  # noqa: S603 - fixed test program, no shell
        [
            sys.executable,
            "-c",
            script,
            str(tool.state_directory),
            "bucket" if isinstance(tool, MontyToolBucket) else "additive",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )


def test_large_json_integer_is_persisted_without_narrowing(composition):
    _, library = composition
    value = 10**50
    invoke(library, f"x = {value}\nx", "large")
    assert invoke(library, "x", "later")["output"] == value


def test_syntax_error_keeps_previous_variables(composition):
    tool, library = composition
    invoke(library, "x=42\nx")
    with pytest.raises(monty.MontyError):
        tool("if :", handle=handle(library, "typo"))
    assert invoke(library, "x", "corrected")["output"] == 42


def test_tools_namespace_is_preloaded_in_every_execution(composition):
    tool, library = composition
    assert tool("tools.add(a=41)", handle=handle(library, "first"))["output"] == 42
    assert tool("tools.add(a=42)", handle=handle(library, "second"))["output"] == 43


def test_failed_calculation_keeps_last_completed_receipt(composition):
    tool, library = composition
    calls = []

    def effect() -> int:
        """Record a completed external effect."""
        calls.append(1)
        return 42

    library.add(effect)
    code = "tools.effect()"
    assert invoke(library, code)["output"] == 42
    with pytest.raises(monty.MontyError):
        tool("if :", handle=handle(library, "invalid"))
    assert invoke(library, code)["output"] == 42
    assert calls == [1]


def test_bucket_captures_custom_kinds_without_special_cases():
    @tool_config(tool_kind="math")
    def arithmetic(value: int) -> int:
        """An ordinary callable with a custom kind."""
        return value + 1

    library = ToolLibrary("custom", [MontyToolBucket(), arithmetic])
    assert library.get_tool_names() == ["monty"]
    assert invoke(library, "tools.arithmetic(value=41)")["output"] == 42


def test_bucket_keeps_deferred_tools_under_their_existing_search_bucket():
    @tool_config(defer_loading=True)
    def deferred(value: int) -> int:
        """A tool owned by deferred discovery."""
        return value

    library = ToolLibrary("deferred", [MontyToolBucket(), add, deferred])
    assert library.get_tool_names() == ["monty", "tool_search", "deferred"]
    assert library.get_bucket_tool_names("monty") == ["add"]
    assert invoke(library, "tools.list()")["output"] == ["add"]


@pytest.mark.asyncio
async def test_builtin_shell_structured_result_crosses_the_bridge(
    composition, tmp_path
):
    from msgflux.runtime.workspace.api import AgentWorkspace
    from msgflux.tools.builtin import BashTool

    tool, library = composition
    library.add(BashTool())
    workspace = AgentWorkspace.local(tmp_path)
    scope = ExecutionScope(
        thread_id="shell", workspace=workspace, permissions=workspace.permissions
    )
    with execution_context(scope=scope):
        result = await tool.acall(
            'result=tools.bash(command="echo 42")\nprint(result["results"][0]["stdout"])',
            handle=handle(library),
        )
    assert result["stdout"].strip() == "42"
