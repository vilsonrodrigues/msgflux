"""Comprehensive tests for typed parsers (base, XML, registry)."""

import pytest

# Skip all XML tests if defusedxml is not installed
pytest.importorskip("defusedxml")

from msgflux.dsl.typed_parsers.base import BaseTypedParser
from msgflux.dsl.typed_parsers.providers.xml import TypedXMLParser
from msgflux.dsl.typed_parsers.registry import (
    register_typed_parser,
    typed_parser_registry,
)


class TestBaseTypedParser:
    """Tests for BaseTypedParser abstract class."""

    def test_subclass_must_define_typed_parser_type(self):
        """Test that subclass must define typed_parser_type attribute."""
        with pytest.raises(
            TypeError, match="must define class attribute `typed_parser_type`"
        ):

            class InvalidParser1(BaseTypedParser):
                template = "test"

                def decode(self):
                    pass

                def encode(self):
                    pass

                def schema_from_response_format(self):
                    pass

    def test_subclass_must_define_template(self):
        """Test that subclass must define template attribute."""
        with pytest.raises(TypeError, match="must define class attribute `template`"):

            class InvalidParser2(BaseTypedParser):
                typed_parser_type = "test"

                def decode(self):
                    pass

                def encode(self):
                    pass

                def schema_from_response_format(self):
                    pass

    def test_valid_subclass(self):
        """Test that valid subclass can be created."""

        class ValidParser(BaseTypedParser):
            typed_parser_type = "test"
            template = "test template"

            def decode(self):
                return {}

            def encode(self):
                return ""

            def schema_from_response_format(self):
                return ""

        assert ValidParser.typed_parser_type == "test"
        assert ValidParser.template == "test template"


class TestTypedXMLParserRegistry:
    """Tests for parser registry functionality."""

    def test_typed_xml_parser_registered(self):
        """Test that TypedXMLParser is registered in the registry."""
        assert "typed_xml" in typed_parser_registry
        assert typed_parser_registry["typed_xml"] == TypedXMLParser

    def test_register_custom_parser(self):
        """Test registering a custom parser."""

        @register_typed_parser
        class CustomParser(BaseTypedParser):
            typed_parser_type = "custom"
            template = "custom template"

            def decode(self):
                return {}

            def encode(self):
                return ""

            def schema_from_response_format(self):
                return ""

        assert "custom" in typed_parser_registry
        assert typed_parser_registry["custom"] == CustomParser


class TestTypedXMLParser:
    """Tests for TypedXMLParser decode, encode, and schema generation."""

    def test_decode_simple_string(self):
        """Test decoding a simple string value."""
        xml = '<name dtype="str">John Doe</name>'
        result = TypedXMLParser.decode(xml)
        assert result["name"] == "John Doe"

    def test_decode_integer(self):
        """Test decoding an integer value."""
        xml = '<age dtype="int">30</age>'
        result = TypedXMLParser.decode(xml)
        assert result["age"] == 30
        assert isinstance(result["age"], int)

    def test_decode_float(self):
        """Test decoding a float value."""
        xml = '<balance dtype="float">123.45</balance>'
        result = TypedXMLParser.decode(xml)
        assert result["balance"] == 123.45
        assert isinstance(result["balance"], float)

    def test_decode_boolean_true(self):
        """Test decoding a boolean true value."""
        xml = '<active dtype="bool">true</active>'
        result = TypedXMLParser.decode(xml)
        assert result["active"] is True
        assert isinstance(result["active"], bool)

    def test_decode_boolean_false(self):
        """Test decoding a boolean false value."""
        xml = '<active dtype="bool">false</active>'
        result = TypedXMLParser.decode(xml)
        assert result["active"] is False
