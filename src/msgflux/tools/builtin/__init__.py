"""Built-in agent tools ready for use out of the box."""

from msgflux.tools.builtin.ask_user import AskUser
from msgflux.tools.builtin.file_edit import FILE_EDIT_TOOL_NAME, Edit, FileEdit
from msgflux.tools.builtin.file_read import FILE_READ_TOOL_NAME, FileRead
from msgflux.tools.builtin.send_user_message import (
    SEND_USER_MESSAGE_TOOL_NAME,
    SendUserMessage,
)
from msgflux.tools.builtin.todo_write import TodoWrite
from msgflux.tools.builtin.weather import Weather
from msgflux.tools.builtin.web_fetch import WebFetch
from msgflux.tools.builtin.web_search import WebSearch

__all__ = [
    "AskUser",
    "Edit",
    "FILE_EDIT_TOOL_NAME",
    "FILE_READ_TOOL_NAME",
    "FileEdit",
    "SEND_USER_MESSAGE_TOOL_NAME",
    "SendUserMessage",
    "FileRead",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
    "Weather",
]
