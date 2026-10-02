import pytest

from msgflux.coding.extensions import (
    CodingExtensions,
    CommandSpec,
    PanelSpec,
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
