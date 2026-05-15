import pytest

from msgflux.context import execution_context
from msgflux.nn import ToolLibrary
from msgflux.runtime import EventStream, EventType
from msgflux.tools.builtin import FILE_READ_TOOL_NAME, FileRead


def test_file_read_returns_line_numbered_content_and_emits_event(tmp_path):
    file_path = tmp_path / "example.py"
    file_path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    tool = FileRead()

    with execution_context(session_id="session_a", namespace="agent"):
        with EventStream() as stream:
            result = tool(str(file_path), offset=2, limit=2)
            stream.close()
            events = stream.events

    assert f"Read {file_path.resolve()} lines 2-3." in result
    assert "2 | beta" in result
    assert "3 | gamma" in result
    assert [event.name for event in events] == [EventType.FILE_READ]
    assert events[0].attributes["path"] == str(file_path.resolve())
    assert events[0].attributes["line_start"] == 2
    assert events[0].attributes["line_end"] == 3
    assert events[0].attributes["lines_returned"] == 2
    assert events[0].attributes["truncated"] is False
    assert events[0].attributes["scope"]["session_id"] == "session_a"


def test_file_read_truncates_by_line_limit(tmp_path):
    file_path = tmp_path / "long.txt"
    file_path.write_text("one\ntwo\nthree\n", encoding="utf-8")
    tool = FileRead()

    result = tool(str(file_path), limit=1)

    assert "Output truncated by limit. Use offset=2 to continue." in result
    assert "1 | one" in result
    assert "2 | two" not in result


def test_file_read_truncates_by_max_chars(tmp_path):
    file_path = tmp_path / "long.txt"
    file_path.write_text("short\nvery long line\n", encoding="utf-8")
    tool = FileRead()

    result = tool(str(file_path), max_chars=10)

    assert "Output truncated by max_chars" in result
    assert "very long line" not in result


def test_file_read_tool_library_executes(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("hello\n", encoding="utf-8")
    library = ToolLibrary("agent", [FileRead()])

    responses = library(
        [
            (
                "call_1",
                FILE_READ_TOOL_NAME,
                {"file_path": str(file_path)},
            )
        ]
    )

    response = responses.get_by_name(FILE_READ_TOOL_NAME)
    assert response is not None
    assert response.error is None
    assert "1 | hello" in response.result
    assert library.library[FILE_READ_TOOL_NAME].display_name == "Read"
    assert library.library[FILE_READ_TOOL_NAME].description == FileRead.description


def test_file_read_rejects_image_path(tmp_path):
    file_path = tmp_path / "image.png"
    file_path.write_bytes(b"not actually an image")
    tool = FileRead()

    with pytest.raises(ValueError, match="appears to be an image"):
        tool(str(file_path))


def test_file_read_rejects_binary_file(tmp_path):
    file_path = tmp_path / "binary.bin"
    file_path.write_bytes(b"abc\x00def")
    tool = FileRead()

    with pytest.raises(ValueError, match="appears to be binary"):
        tool(str(file_path))


def test_file_read_rejects_invalid_limits(tmp_path):
    file_path = tmp_path / "example.txt"
    file_path.write_text("hello\n", encoding="utf-8")
    tool = FileRead()

    with pytest.raises(ValueError, match="offset must be greater than zero"):
        tool(str(file_path), offset=0)


def test_file_read_rejects_blocked_device_path():
    tool = FileRead()

    with pytest.raises(ValueError, match="blocked device path"):
        tool("/dev/zero")
