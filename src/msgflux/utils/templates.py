"""Template formatting utilities."""

import re
from typing import Any, Dict, Union

from jinja2 import Template


def format_template(
    content: Union[str, Dict[str, Any]],
    raw_template: str,
) -> str:
    """Format a template with content.

    This function renders templates using either string formatting or Jinja2.

    Args:
        content:
            Content to render in the template. Can be a string (for .format())
            or a dict (for Jinja2 rendering).
        raw_template:
            The template string to render.

    Returns:
        The rendered template with normalized line breaks.

    Raises:
        ValueError: If content type is not supported.

    Examples:
        String formatting:
            >>> format_template("Hello", "Welcome: {}")
            'Welcome: Hello'

        Dict formatting with Jinja2:
            >>> format_template({"name": "Alice"}, "Hello {{name}}!")
            'Hello Alice!'
    """
    if isinstance(content, str):
        rendered = raw_template.format(content)
    elif isinstance(content, dict):
        template = Template(raw_template)
        rendered = template.render(content)
    else:
        raise ValueError(
            f"Unsupported content type for template formatting: {type(content)}"
        )

    # Normalize excessive line breaks (3+ newlines → 2 newlines)
    rendered = re.sub(r"\n{3,}", "\n\n", rendered).strip()
    return rendered
