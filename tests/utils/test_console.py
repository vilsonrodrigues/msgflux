"""Tests for msgflux.utils.console module."""

from io import StringIO
from unittest.mock import patch

from msgflux.utils.console import cprint


def test_cprint_plain_text():
    """Test cprint with plain text."""
    with patch("sys.stdout", new=StringIO()) as fake_out:
        cprint("Hello, World!")
        output = fake_out.getvalue()
        assert "Hello, World!" in output


def test_cprint_with_foreground_color():
    """Test cprint with foreground color."""
    with patch("sys.stdout", new=StringIO()) as fake_out:
        cprint("Red text", lc="r")
        output = fake_out.getvalue()
        assert "\033[" in output
        assert "Red text" in output


def test_cprint_all_foreground_colors():
    """Test cprint with all foreground colors."""
    colors = ["k", "r", "g", "y", "b", "m", "c", "w"]
    for color in colors:
        with patch("sys.stdout", new=StringIO()) as fake_out:
            cprint(f"Text in {color}", lc=color)
            output = fake_out.getvalue()
            assert f"Text in {color}" in output
