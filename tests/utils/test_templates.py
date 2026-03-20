"""Tests for msgflux.utils.templates module."""

import pytest

from msgflux.utils.templates import format_template


def test_format_template_string_simple():
    """Test format_template with simple string content."""
    result = format_template("World", "Hello, {}!")
    assert result == "Hello, World!"


def test_format_template_dict_simple():
    """Test format_template with dictionary content."""
    content = {"name": "Alice", "age": 30}
    template = "Name: {{name}}, Age: {{age}}"
    result = format_template(content, template)
    assert result == "Name: Alice, Age: 30"


def test_format_template_invalid_type():
    """Test format_template raises ValueError for invalid content type."""
    with pytest.raises(ValueError, match="Unsupported content type"):
        format_template([1, 2, 3], "Template: {}")
