import pytest

from msgflux.core.prompt_section import PromptSection, normalize_prompt_content


class TestPromptSection:
    def test_prompt_section_renders_class_fields(self):
        class SystemMessage(PromptSection):
            role = "senior software engineer"
            style = "concise"
            constraints = ["be factual", "preserve user changes"]

        assert str(SystemMessage()) == (
            "role: senior software engineer\n"
            "style: concise\n"
            "constraints:\n"
            "  - be factual\n"
            "  - preserve user changes"
        )

    def test_prompt_section_cleans_multiline_strings(self):
        section = PromptSection(
            role="""
                senior software engineer
            """,
            goal="""
                implement focused changes
                run focused tests
            """,
        )

        assert section.to_dict() == {
            "role": "senior software engineer",
            "goal": "implement focused changes\nrun focused tests",
        }

    def test_prompt_section_supports_nested_section_classes(self):
        class Output(PromptSection):
            format = "short markdown"
            include_tests = True

        class SystemMessage(PromptSection):
            role = "reviewer"
            output = Output

        assert str(SystemMessage()) == (
            "role: reviewer\noutput:\n  format: short markdown\n  include_tests: true"
        )

    def test_prompt_section_supports_nested_section_instances(self):
        class SystemMessage(PromptSection):
            role = "research analyst"
            output = PromptSection(format="short markdown", include_sources=True)

        assert str(SystemMessage()) == (
            "role: research analyst\n"
            "output:\n"
            "  format: short markdown\n"
            "  include_sources: true"
        )

    def test_prompt_section_instance_fields_override_class_fields(self):
        class SystemMessage(PromptSection):
            role = "assistant"
            style = "concise"

        section = SystemMessage(role="senior software engineer")

        assert str(section) == "role: senior software engineer\nstyle: concise"

    def test_prompt_section_inherits_and_overrides_fields(self):
        class Base(PromptSection):
            style = "concise"
            constraints = ["be factual"]

        class Coding(Base):
            role = "software engineer"
            constraints = ["be factual", "preserve user changes"]

        assert str(Coding()) == (
            "style: concise\n"
            "constraints:\n"
            "  - be factual\n"
            "  - preserve user changes\n"
            "role: software engineer"
        )

    def test_normalize_prompt_content_accepts_strings_sections_and_section_classes(
        self,
    ):
        class SystemMessage(PromptSection):
            role = "assistant"

        assert (
            normalize_prompt_content(
                """
                Be concise.
            """,
                field_name="instructions",
            )
            == "Be concise."
        )
        assert (
            normalize_prompt_content(
                PromptSection(role="assistant"), field_name="system_message"
            )
            == "role: assistant"
        )
        assert (
            normalize_prompt_content(SystemMessage, field_name="system_message")
            == "role: assistant"
        )

    def test_normalize_prompt_content_rejects_unsupported_values(self):
        with pytest.raises(TypeError, match="requires a string, PromptSection"):
            normalize_prompt_content(123, field_name="system_message")
