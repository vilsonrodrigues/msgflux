"""Tests for msgflux.utils.mermaid module."""

import pytest

from msgflux.utils.mermaid import plot_mermaid


def test_plot_mermaid_with_simple_diagram():
    """Test plot_mermaid with a simple mermaid diagram."""
    diagram = "graph LR\n    Start --> End"
    try:
        result = plot_mermaid(diagram)
        assert result is not None
    except ImportError:
        pytest.skip("mermaid package not installed")
