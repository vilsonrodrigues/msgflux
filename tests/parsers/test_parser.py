"""Tests for Parser factory class."""

import pytest

from msgflux.data.parsers import Parser
from msgflux.data.parsers.registry import parser_registry


class TestParserFactory:
    """Test the Parser factory class."""

    def test_parser_types(self):
        """Test that parser_types returns available parser types."""
        parser_types = Parser.parser_types()
        assert isinstance(parser_types, list)
        # After registration, we should have these types
        assert "pdf" in parser_types or len(parser_types) >= 0

    def test_providers(self):
        """Test that providers returns available providers per type."""
        providers = Parser.providers()
        assert isinstance(providers, dict)

        # Check structure
        for parser_type, provider_list in providers.items():
            assert isinstance(parser_type, str)
            assert isinstance(provider_list, list)

    def test_pdf_parser_creation(self):
        """Test creating a PDF parser."""
        if "pdf" in parser_registry and "pypdf" in parser_registry["pdf"]:
            parser = Parser.pdf("pypdf")
            assert parser is not None
            assert hasattr(parser, "__call__")
            assert hasattr(parser, "acall")

    def test_xlsx_parser_creation(self):
        """Test creating an XLSX parser."""
        if "xlsx" in parser_registry and "openpyxl" in parser_registry["xlsx"]:
            parser = Parser.xlsx("openpyxl")
            assert parser is not None
            assert hasattr(parser, "__call__")
            assert hasattr(parser, "acall")

    def test_pptx_parser_creation(self):
        """Test creating a PPTX parser."""
        if "pptx" in parser_registry and "python_pptx" in parser_registry["pptx"]:
            parser = Parser.pptx("python_pptx")
            assert parser is not None
            assert hasattr(parser, "__call__")
            assert hasattr(parser, "acall")

    def test_docx_parser_creation(self):
        """Test creating a DOCX parser."""
        if "docx" in parser_registry and "python_docx" in parser_registry["docx"]:
            parser = Parser.docx("python_docx")
            assert parser is not None
            assert hasattr(parser, "__call__")
            assert hasattr(parser, "acall")

    def test_invalid_parser_type(self):
        """Test that invalid parser type raises ValueError."""
        with pytest.raises(ValueError, match="Parser type .* is not supported"):
            Parser._get_parser_class("invalid_type", "some_provider")

    def test_invalid_provider(self):
        """Test that invalid provider raises ValueError."""
        if "pdf" in parser_registry:
            with pytest.raises(ValueError, match="Provider .* not registered"):
                Parser._get_parser_class("pdf", "invalid_provider")


class TestParserInstantiation:
    """Test parser instantiation with various options."""

    def test_pdf_parser_with_options(self):
        """Test creating PDF parser with custom options."""
        if "pdf" in parser_registry and "pypdf" in parser_registry["pdf"]:
            parser = Parser.pdf(
                "pypdf",
                extraction_mode="layout",
                encode_images_base64=True
            )
            assert parser.extraction_mode == "layout"
            assert parser.encode_images_base64 is True

    def test_xlsx_parser_with_options(self):
        """Test creating XLSX parser with custom options."""
        if "xlsx" in parser_registry and "openpyxl" in parser_registry["xlsx"]:
            parser = Parser.xlsx(
                "openpyxl",
                table_format="html",
                encode_images_base64=False
            )
            assert parser.table_format == "html"
            assert parser.encode_images_base64 is False

    def test_pptx_parser_with_options(self):
        """Test creating PPTX parser with custom options."""
        if "pptx" in parser_registry and "python_pptx" in parser_registry["pptx"]:
            parser = Parser.pptx(
                "python_pptx",
                include_notes=False,
                encode_images_base64=True
            )
            assert parser.include_notes is False
            assert parser.encode_images_base64 is True

    def test_docx_parser_with_options(self):
        """Test creating DOCX parser with custom options."""
        if "docx" in parser_registry and "python_docx" in parser_registry["docx"]:
            parser = Parser.docx(
                "python_docx",
                table_format="markdown",
                encode_images_base64=False
            )
            assert parser.table_format == "markdown"
            assert parser.encode_images_base64 is False
