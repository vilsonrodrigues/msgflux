"""Template formatting utilities with security features."""

import re
from typing import Any, Dict, Union

from jinja2 import Template
from markupsafe import escape


def format_template(
    content: Union[str, Dict[str, Any]],
    raw_template: str,
    sanitize_inputs: bool = True,
) -> str:
    """Format a template with content, optionally sanitizing inputs.

    This function renders templates using either string formatting or Jinja2,
    with optional input sanitization to prevent injection attacks.

    Args:
        content:
            Content to render in the template. Can be a string (for .format())
            or a dict (for Jinja2 rendering).
        raw_template:
            The template string to render.
        sanitize_inputs:
            If True (default), sanitizes string inputs using markupsafe.escape
            to prevent template injection attacks. Set to False only if you
            trust the input source completely.

    Returns:
        The rendered template with normalized line breaks.

    Raises:
        ValueError: If content type is not supported.

    Examples:
        String formatting with sanitization:
            >>> format_template("Hello", "Welcome: {}")
            'Welcome: Hello'

        Dict formatting with Jinja2:
            >>> format_template({"name": "Alice"}, "Hello {{name}}!")
            'Hello Alice!'

        Sanitization prevents injection:
            >>> malicious = "{{7*7}}"
            >>> format_template({"input": malicious}, "User: {{input}}")
            'User: {{7*7}}'  # Escaped, not evaluated
    """
    if isinstance(content, str):
        if sanitize_inputs:
            content = escape(content)
        rendered = raw_template.format(content)
    elif isinstance(content, dict):
        if sanitize_inputs:
            # Sanitize string values in dictionary
            content = {
                k: escape(v) if isinstance(v, str) else v for k, v in content.items()
            }
        template = Template(raw_template)
        rendered = template.render(content)
    else:
        raise ValueError(
            f"Unsupported content type for template formatting: {type(content)}"
        )

    # Normalize excessive line breaks (3+ newlines → 2 newlines)
    rendered = re.sub(r"\n{3,}", "\n\n", rendered).strip()
    return rendered
