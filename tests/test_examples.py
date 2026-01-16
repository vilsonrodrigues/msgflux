import pytest

from msgflux.examples import Example, ExampleCollection, ExampleFormat


@pytest.fixture
def simple_example():
    return Example(inputs="input", labels="label")


@pytest.fixture
def dict_example():
    return Example(inputs={"in_key": "in_val"}, labels={"out_key": "out_val"})


class TestExample:
    def test_post_init(self, simple_example, dict_example):
        assert not simple_example._inputs_needs_transform
        assert not simple_example._labels_needs_transform
        assert dict_example._inputs_needs_transform
        assert dict_example._labels_needs_transform

    def test_transform_no_transformer(self, simple_example, dict_example):
        assert simple_example.transform().inputs == "input"
        assert dict_example.transform().inputs == {"in_key": "in_val"}

    def test_transform_with_transformer(self, dict_example):
        input_transformer = lambda x: f"input: {x['in_key']}"
        output_transformer = lambda x: f"output: {x['out_key']}"
        transformed = dict_example.transform(input_transformer, output_transformer)
        assert transformed.inputs == "input: in_val"
        assert transformed.labels == "output: out_val"

    def test_to_xml(self, simple_example):
        xml = simple_example.to_xml(1)
        assert "<example id=1>" in xml
        assert "<input>" in xml
        assert "input" in xml
        assert "</input>" in xml
        assert "<output>" in xml
        assert "label" in xml
        assert "</output>" in xml
        assert "</example>" in xml

    def test_to_xml_with_all_fields(self):
        example = Example(
            inputs="input",
            labels="label",
            title="title",
            reasoning="reasoning",
            topic="topic",
        )
        xml = example.to_xml(1)
        assert 'title="title"' in xml
        assert 'topic="topic"' in xml
        assert "<reasoning>" in xml
        assert "reasoning" in xml
        assert "</reasoning>" in xml


@pytest.fixture
def simple_collection():
    return ExampleCollection(
        [
            Example(inputs="in1", labels="out1", topic="topic1"),
            Example(inputs="in2", labels="out2", topic="topic2"),
        ]
    )


@pytest.fixture
def dict_collection():
    return ExampleCollection(
        [
            {"inputs": {"in": 1}, "labels": {"out": 1}, "topic": "topic1"},
            {"inputs": {"in": 2}, "labels": {"out": 2}, "topic": "topic2"},
        ]
    )


class TestExampleCollection:
    def test_init_with_examples(self, simple_collection):
        assert len(simple_collection.get_examples()) == 2

    def test_init_with_dicts(self, dict_collection):
        assert len(dict_collection.get_examples()) == 2
        assert isinstance(dict_collection.get_examples()[0], Example)

    def test_add(self):
        collection = ExampleCollection()
        collection.add(inputs="in", labels="out")
        assert len(collection.get_examples()) == 1

    def test_filter_by_topic(self, simple_collection):
        filtered = simple_collection.filter_by_topic("topic1")
        assert len(filtered) == 1
        assert filtered[0].inputs == "in1"

    def test_get_examples_needing_transform(self, simple_collection, dict_collection):
        assert len(simple_collection.get_examples_needing_transform()) == 0
        assert len(dict_collection.get_examples_needing_transform()) == 2

    def test_transform(self, dict_collection):
        input_transformer = lambda x: str(x["in"])
        output_transformer = lambda x: str(x["out"])
        transformed = dict_collection.transform(input_transformer, output_transformer)
        assert transformed.get_examples()[0].inputs == "1"
        assert transformed.get_examples()[0].labels == "1"

    def test_get_formatted_empty(self):
        collection = ExampleCollection()
        assert collection.get_formatted() is None

    def test_get_formatted(self, simple_collection):
        formatted = simple_collection.get_formatted()
        assert "<example id=1" in formatted
        assert "<example id=2" in formatted


# =============================================================================
# New Tests for ExampleFormat and Multiple Formats
# =============================================================================


class TestExampleFormat:
    """Tests for ExampleFormat enum."""

    def test_enum_values(self):
        assert ExampleFormat.XML == "xml"
        assert ExampleFormat.PLAINTEXT == "plaintext"
        assert ExampleFormat.MINIMAL == "minimal"

    def test_string_conversion(self):
        assert ExampleFormat.XML.value == "xml"
        assert ExampleFormat("xml") == ExampleFormat.XML
        assert ExampleFormat("plaintext") == ExampleFormat.PLAINTEXT
        assert ExampleFormat("minimal") == ExampleFormat.MINIMAL


class TestExampleFormats:
    """Tests for Example format methods."""

    def test_to_plaintext(self, simple_example):
        plaintext = simple_example.to_plaintext(1)
        assert "Example 1:" in plaintext
        assert "Input: input" in plaintext
        assert "Output: label" in plaintext

    def test_to_plaintext_with_reasoning(self):
        example = Example(inputs="q", labels="a", reasoning="think")
        plaintext = example.to_plaintext(1)
        assert "Reasoning: think" in plaintext

    def test_to_minimal(self, simple_example):
        minimal = simple_example.to_minimal()
        assert "Q: input" in minimal
        assert "A: label" in minimal
        assert "Example" not in minimal

    def test_to_minimal_with_reasoning(self):
        example = Example(inputs="q", labels="a", reasoning="think")
        minimal = example.to_minimal()
        assert "Reasoning: think" in minimal

    def test_format_xml(self, simple_example):
        formatted = simple_example.format(ExampleFormat.XML, example_id=1)
        assert "<example id=1>" in formatted

    def test_format_plaintext(self, simple_example):
        formatted = simple_example.format(ExampleFormat.PLAINTEXT, example_id=1)
        assert "Example 1:" in formatted

    def test_format_minimal(self, simple_example):
        formatted = simple_example.format(ExampleFormat.MINIMAL, example_id=1)
        assert "Q: input" in formatted

    def test_format_string_arg(self, simple_example):
        """Test that format accepts string instead of enum."""
        formatted = simple_example.format("plaintext", example_id=1)
        assert "Example 1:" in formatted


class TestExampleCollectionStringInput:
    """Tests for ExampleCollection with string inputs."""

    def test_init_with_strings(self):
        collection = ExampleCollection([
            "Simple string example",
            "Another string",
        ])
        assert len(collection.get_examples()) == 2
        assert collection.get_examples()[0].inputs == "Simple string example"
        assert collection.get_examples()[0].labels == ""

    def test_init_mixed_formats(self):
        collection = ExampleCollection([
            Example(inputs="q1", labels="a1"),
            {"inputs": "q2", "labels": "a2"},
            "Just a string",
        ])
        assert len(collection.get_examples()) == 3
        assert collection.get_examples()[0].inputs == "q1"
        assert collection.get_examples()[1].inputs == "q2"
        assert collection.get_examples()[2].inputs == "Just a string"

    def test_invalid_input_type(self):
        with pytest.raises(TypeError):
            ExampleCollection([123])  # int is not valid


class TestExampleCollectionFormats:
    """Tests for ExampleCollection.get_formatted() with different formats."""

    @pytest.fixture
    def collection(self):
        return ExampleCollection([
            Example(inputs="What is 2+2?", labels="4"),
            Example(inputs="What is 3+3?", labels="6"),
        ])

    def test_get_formatted_xml(self, collection):
        formatted = collection.get_formatted(format=ExampleFormat.XML)
        assert "<example id=1>" in formatted
        assert "<example id=2>" in formatted
        assert "<input>" in formatted

    def test_get_formatted_plaintext(self, collection):
        formatted = collection.get_formatted(format=ExampleFormat.PLAINTEXT)
        assert "Example 1:" in formatted
        assert "Example 2:" in formatted
        assert "Input: What is 2+2?" in formatted
        assert "Output: 4" in formatted

    def test_get_formatted_minimal(self, collection):
        formatted = collection.get_formatted(format=ExampleFormat.MINIMAL)
        assert "Q: What is 2+2?" in formatted
        assert "A: 4" in formatted
        assert "Example" not in formatted

    def test_get_formatted_string_arg(self, collection):
        """Test that get_formatted accepts string format."""
        formatted = collection.get_formatted(format="plaintext")
        assert "Example 1:" in formatted

    def test_get_formatted_custom_separator(self, collection):
        formatted = collection.get_formatted(
            format=ExampleFormat.MINIMAL,
            separator="\n---\n"
        )
        assert "---" in formatted

    def test_get_formatted_default_separator_xml(self, collection):
        formatted = collection.get_formatted(format=ExampleFormat.XML)
        assert "\n\n" in formatted  # Default for XML

    def test_get_formatted_default_separator_minimal(self, collection):
        formatted = collection.get_formatted(format=ExampleFormat.MINIMAL)
        # Minimal has single newline as default separator
        parts = formatted.split("\n")
        assert len(parts) > 2
