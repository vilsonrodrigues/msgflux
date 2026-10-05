import pytest


def make_object():
    return object()


def required_factory(value):
    return value


from msgflux.coding.extensions import (
    CodingExtensions,
    CommandSpec,
    PanelSpec,
    ToolSpec,
)


def test_register_panels_by_side_and_unregister():
    extensions = CodingExtensions()
    left = extensions.register_panel("files", "Files", object)
    extensions.register_panel("preview", "Preview", object, side="right")

    assert [panel.id for panel in extensions.panels()] == ["files", "preview"]
    assert [panel.id for panel in extensions.panels("left")] == ["files"]
    assert [panel.id for panel in extensions.panels("right")] == ["preview"]

    left.unregister()
    left.unregister()
    assert [panel.id for panel in extensions.panels()] == ["preview"]
    assert not left.active


def test_register_command_and_unregister():
    extensions = CodingExtensions()
    handler = lambda: "ok"
    handle = extensions.register_command("save", handler, description="Save work")

    assert extensions.commands() == (CommandSpec("save", handler, "Save work"),)
    handle.close()
    assert extensions.commands() == ()


def test_batch_registration_is_atomic_on_duplicate_or_invalid_spec():
    extensions = CodingExtensions()
    extensions.register_panel("existing", "Existing", object)

    with pytest.raises(ValueError, match="Duplicate panel id"):
        extensions.register_many(
            panels=(
                PanelSpec("new", "New", object),
                PanelSpec("existing", "Collision", object),
            ),
            commands=(CommandSpec("also-new", lambda: None),),
        )

    assert [panel.id for panel in extensions.panels()] == ["existing"]
    assert extensions.commands() == ()

    with pytest.raises(ValueError, match="Panel side"):
        extensions.register_many(
            panels=(PanelSpec("new", "New", object, "middle"),),  # type: ignore[arg-type]
            commands=(CommandSpec("cmd", lambda: None),),
        )
    assert extensions.commands() == ()


def test_batch_returns_handles_in_panel_then_command_order():
    extensions = CodingExtensions()
    handles = extensions.register_many(
        panels=(PanelSpec("one", "One", object), PanelSpec("two", "Two", object)),
        commands=(CommandSpec("run", lambda: None),),
    )

    assert len(handles) == 3
    assert len(extensions.panels()) == 2
    assert len(extensions.commands()) == 1
    for handle in handles:
        handle.unregister()
    assert extensions.panels() == ()
    assert extensions.commands() == ()


def test_stale_handle_cannot_remove_later_registration():
    extensions = CodingExtensions()
    first = extensions.register_command("run", lambda: 1)
    first.unregister()
    second_handler = lambda: 2
    second = extensions.register_command("run", second_handler)

    first.unregister()
    assert extensions.commands()[0].handler is second_handler
    assert second.active


def test_register_tool_does_not_instantiate_class_and_infers_name():
    calls = 0

    class ExampleTool:
        name = "example"

        def __init__(self):
            nonlocal calls
            calls += 1

    extensions = CodingExtensions()
    handle = extensions.register_tool(ExampleTool)

    assert calls == 0
    assert extensions.tools() == (ToolSpec("example", ExampleTool),)
    handle.unregister()
    assert extensions.tools() == ()


def test_tool_registration_handle_cannot_remove_replacement():
    extensions = CodingExtensions()
    first = extensions.register_tool(make_object)
    first.unregister()
    tool = make_object
    second = extensions.register_tool(tool)

    first.unregister()
    assert extensions.tools()[0].tool is tool
    assert second.active


def test_register_many_is_atomic_across_panels_commands_and_tools():
    extensions = CodingExtensions()
    extensions.register_tool(make_object)

    with pytest.raises(ValueError, match="Duplicate tool id"):
        extensions.register_many(
            panels=(PanelSpec("new", "New", object),),
            commands=(CommandSpec("new-command", lambda: None),),
            tools=(
                # First new tool must not be installed when the second collides.
                ToolSpec("fresh", make_object),
                ToolSpec("make_object", make_object),
            ),
        )

    assert extensions.panels() == ()
    assert extensions.commands() == ()
    assert [tool.name for tool in extensions.tools()] == ["make_object"]


def test_tool_registration_accepts_functions_async_and_callable_instances():
    extensions = CodingExtensions()

    calls = 0

    def sync_tool():
        nonlocal calls
        calls += 1

    async def async_tool():
        nonlocal calls
        calls += 1

    class CallableFactory:
        def __call__(self):
            return "result"

    instance = CallableFactory()
    sync_handle = extensions.register_tool(sync_tool)
    async_handle = extensions.register_tool(async_tool)
    instance_handle = extensions.register_tool(instance)

    assert calls == 0
    assert [spec.tool for spec in extensions.tools()] == [
        sync_tool,
        async_tool,
        instance,
    ]
    assert [spec.name for spec in extensions.tools()] == [
        "sync_tool",
        "async_tool",
        "CallableFactory",
    ]
    sync_handle.unregister()
    async_handle.unregister()
    instance_handle.unregister()


def test_tool_name_uses_config_override_before_callable_name_and_checks_regex():
    from msgflux.tools.config import tool_config

    @tool_config(name_override="configured_name")
    def original_name():
        return None

    extensions = CodingExtensions()
    extensions.register_tool(original_name)
    assert extensions.tools()[0].name == "configured_name"

    class ModuleStyleTool:
        def get_module_name(self):
            return "module_style"

        def __call__(self):
            return None

    extensions.register_tool(ModuleStyleTool())
    assert extensions.tools()[1].name == "module_style"

    class InvalidName:
        name = "not valid"

        def __call__(self):
            return None

    with pytest.raises(ValueError, match="Invalid tool name"):
        extensions.register_tool(InvalidName())


def test_tool_class_must_be_zero_argument_and_is_not_instantiated():
    calls = 0

    class RequiresArgument:
        def __init__(self, required):
            nonlocal calls
            calls += 1

        def __call__(self):
            return None

    with pytest.raises(TypeError, match="zero arguments"):
        CodingExtensions().register_tool(RequiresArgument)
    assert calls == 0


def test_load_rolls_back_failed_extension_without_removing_existing_entries():
    extensions = CodingExtensions()
    existing = extensions.register_command("existing", lambda: "existing")
    partial_handles = []

    def register(coding_extensions):
        partial_handles.append(
            coding_extensions.register_command("partial", lambda: "partial")
        )
        coding_extensions.register_tool(make_object)
        raise RuntimeError("entry point failed")

    with pytest.raises(RuntimeError, match="entry point failed"):
        extensions.load(register)

    assert [command.id for command in extensions.commands()] == ["existing"]
    assert extensions.tools() == ()
    assert existing.active
    assert partial_handles[0].active
    partial_handles[0].unregister()
    assert [command.id for command in extensions.commands()] == ["existing"]


def test_load_handles_unregister_from_the_same_registry():
    extensions = CodingExtensions()
    handles = []

    def register(coding_extensions):
        handles.append(coding_extensions.register_command("loaded", lambda: "loaded"))

    assert extensions.load(register) is None
    assert [command.id for command in extensions.commands()] == ["loaded"]
    handles[0].unregister()
    assert extensions.commands() == ()


def test_load_rejects_async_callback_and_rolls_back():
    extensions = CodingExtensions()

    async def register(coding_extensions):
        coding_extensions.register_command("async", lambda: "async")

    with pytest.raises(TypeError, match="must be synchronous"):
        extensions.load(register)
    assert extensions.commands() == ()
