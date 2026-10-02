"""Keyboard-first selectors for sessions and slash commands."""

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, OptionList, Static
from textual.widgets.option_list import Option


class CodingPicker(ModalScreen[str | None]):
    """A searchable selector whose values never come from display labels."""

    CSS = """
    CodingPicker { align: center middle; }
    #picker { width: 80%; height: 70%; border: round $primary; background: $surface;
      padding: 1 2; }
    #picker-title { height: 2; }
    #picker-filter { height: 3; }
    #picker-options { height: 1fr; }
    """
    BINDINGS = [("escape", "dismiss_picker", "Cancel")]

    def __init__(self, title: str, entries: tuple[tuple[str, str], ...]):
        super().__init__()
        self.title_text = title
        self.entries = entries

    def compose(self) -> ComposeResult:
        with Vertical(id="picker"):
            yield Static(self.title_text, id="picker-title", markup=False)
            yield Input(placeholder="Search…", id="picker-filter")
            yield OptionList(id="picker-options")

    def on_mount(self):
        self._filter("")
        self.query_one(Input).focus()

    def _filter(self, query: str):
        options = self.query_one(OptionList)
        options.clear_options()
        options.add_options(
            Option(Text(label), id=value)
            for value, label in self.entries
            if query.casefold() in label.casefold()
        )
        if options.option_count:
            options.highlighted = 0

    def on_input_changed(self, event: Input.Changed):
        self._filter(event.value)

    def on_input_submitted(self, _event: Input.Submitted):
        options = self.query_one(OptionList)
        if options.highlighted is not None:
            self.dismiss(options.get_option_at_index(options.highlighted).id)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected):
        self.dismiss(event.option.id)

    def action_dismiss_picker(self):
        self.dismiss(None)

    def key_down(self):
        self.query_one(OptionList).focus()
