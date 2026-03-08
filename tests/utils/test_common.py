"""Tests for msgflux.utils.common module."""

from typing import Any, Literal, Optional, Set, Tuple, Union

from msgflux.utils.common import type_mapping


def test_type_mapping_contains_basic_types():
    """Test that type_mapping contains all expected basic types."""
    assert "str" in type_mapping
    assert "int" in type_mapping
    assert "float" in type_mapping
    assert "bool" in type_mapping
    assert "list" in type_mapping
    assert "dict" in type_mapping


def test_type_mapping_string_types():
    """Test string type mappings."""
    assert type_mapping["str"] is str
    assert type_mapping["string"] is str


def test_type_mapping_synonyms():
    """Test that synonym mappings point to same types."""
    assert type_mapping["str"] is type_mapping["string"]
    assert type_mapping["int"] is type_mapping["integer"]
    assert type_mapping["float"] is type_mapping["number"]
    assert type_mapping["bool"] is type_mapping["boolean"]
    assert type_mapping["list"] is type_mapping["array"]
    assert type_mapping["dict"] is type_mapping["object"]
