"""Tests for msgflux.utils.hooks module."""

from collections import OrderedDict

from msgflux.utils.hooks import RemovableHandle


def test_removable_handle_basic_creation():
    """Test basic creation of RemovableHandle."""
    hooks_dict = OrderedDict()
    handle = RemovableHandle(hooks_dict)
    assert handle.id >= 0
    assert handle.hooks_dict_ref() is hooks_dict


def test_removable_handle_remove():
    """Test removing hook from hooks dictionary."""
    hooks_dict = OrderedDict()
    handle = RemovableHandle(hooks_dict)
    hooks_dict[handle.id] = "some_hook"
    assert handle.id in hooks_dict
    handle.remove()
    assert handle.id not in hooks_dict


def test_removable_handle_context_manager():
    """Test RemovableHandle as context manager."""
    hooks_dict = OrderedDict()
    handle = RemovableHandle(hooks_dict)
    hooks_dict[handle.id] = "hook"
    assert handle.id in hooks_dict
    with handle:
        assert handle.id in hooks_dict
    assert handle.id not in hooks_dict
