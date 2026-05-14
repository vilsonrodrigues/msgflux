"""Built-in agent tools ready for use out of the box."""

from msgflux.tools.builtin.ask_user import AskUser
from msgflux.tools.builtin.send_user_message import Brief, SendUserMessage
from msgflux.tools.builtin.todo_write import TodoWrite
from msgflux.tools.builtin.weather import Weather
from msgflux.tools.builtin.web_fetch import WebFetch
from msgflux.tools.builtin.web_search import WebSearch

__all__ = [
    "AskUser",
    "Brief",
    "SendUserMessage",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
    "Weather",
]
