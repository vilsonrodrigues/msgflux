from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Literal, Mapping, Optional, Union

from msgflux.utils.xml import apply_xml_tags


class ExampleFormat(str, Enum):
    """Supported output formats for examples."""

    XML = "xml"
    PLAINTEXT = "plaintext"
    MINIMAL = "minimal"


@dataclass
class Example:
    """Represents a single example for a LM.

    Args:
        inputs:
            Task inputs.
        labels:
            Expected model output.
        title:
            Example title.
        reasoning:
            Intermediate thinking/logic before generating output.
        topic:
            Example topic/category.
    """

    inputs: Union[str, Mapping[str, Any]]
    labels: Union[str, Mapping[str, Any]]
    reasoning: Optional[str] = None
    title: Optional[str] = None
    topic: Optional[str] = None
    _inputs_needs_transform: bool = field(init=False)
    _labels_needs_transform: bool = field(init=False)

    def __post_init__(self):
        self._inputs_needs_transform = isinstance(self.inputs, dict)
        self._labels_needs_transform = isinstance(self.labels, dict)

    def transform(
        self,
        input_transformer: Optional[callable] = None,
        output_transformer: Optional[callable] = None,
    ) -> "Example":
        """Returns a new Example instance with transformed inputs/outputs as strings.

        Args:
            input_transformer:
                Function to transform input dict to string.
            output_transformer:
                Function to transform output dict to string.

        Returns:
            New Example instance with string values
        """
        new_input = self.inputs
        new_output = self.labels

        if self._inputs_needs_transform and input_transformer:
            new_input = input_transformer(self.inputs)

        if self._labels_needs_transform and output_transformer:
            new_output = output_transformer(self.labels)

        return Example(
            inputs=new_input,
            labels=new_output,
            title=self.title,
            reasoning=self.reasoning,
            topic=self.topic,
        )

    def to_xml(self, example_id: int) -> str:
        """Convert the example to XML format."""
        attributes = [f"id={example_id}"]

        if self.title:
            attributes.append(f'title="{self.title}"')
        if self.topic:
            attributes.append(f'topic="{self.topic}"')

        attr_str = " ".join(attributes)

        result = [f"<example {attr_str}>"]
        result.append(apply_xml_tags("input", str(self.inputs)))
        if self.reasoning:
            result.append(apply_xml_tags("reasoning", self.reasoning))
        result.append(apply_xml_tags("output", str(self.labels)))
        result.append("</example>")

        return "\n".join(result)

    def to_plaintext(self, example_id: int) -> str:
        """Convert the example to plaintext format.

        Format:
            Example 1:
            Input: <inputs>
            [Reasoning: <reasoning>]
            Output: <labels>
        """
        result = [f"Example {example_id}:"]
        result.append(f"Input: {self.inputs}")
        if self.reasoning:
            result.append(f"Reasoning: {self.reasoning}")
        result.append(f"Output: {self.labels}")
        return "\n".join(result)

    def to_minimal(self) -> str:
        """Convert the example to minimal format.

        Format:
            Q: <inputs>
            A: <labels>
        """
        result = [f"Q: {self.inputs}"]
        if self.reasoning:
            result.append(f"Reasoning: {self.reasoning}")
        result.append(f"A: {self.labels}")
        return "\n".join(result)

    def format(
        self,
        fmt: Union[ExampleFormat, str] = ExampleFormat.XML,
        example_id: int = 1,
    ) -> str:
        """Format example according to specified format.

        Args:
            fmt: Output format (xml, plaintext, minimal).
            example_id: Example number for numbered formats.

        Returns:
            Formatted example string.
        """
        if isinstance(fmt, str):
            fmt = ExampleFormat(fmt)

        if fmt == ExampleFormat.XML:
            return self.to_xml(example_id)
        elif fmt == ExampleFormat.PLAINTEXT:
            return self.to_plaintext(example_id)
        elif fmt == ExampleFormat.MINIMAL:
            return self.to_minimal()
        else:
            raise ValueError(f"Unknown format: {fmt}")


class ExampleCollection:
    """A collection of examples with flexible input and output formats.

    Accepts examples in multiple formats:
        - Example objects
        - Dictionaries with 'inputs' and 'labels' keys
        - Strings (converted to Example with inputs=str, labels="")

    Examples:
        >>> # From Example objects
        >>> collection = ExampleCollection([
        ...     Example(inputs="What is 2+2?", labels="4"),
        ... ])

        >>> # From dictionaries
        >>> collection = ExampleCollection([
        ...     {"inputs": "What is 2+2?", "labels": "4"},
        ... ])

        >>> # From strings
        >>> collection = ExampleCollection([
        ...     "This is a simple example",
        ... ])

        >>> # Mixed formats
        >>> collection = ExampleCollection([
        ...     Example(inputs="q1", labels="a1"),
        ...     {"inputs": "q2", "labels": "a2"},
        ...     "simple string",
        ... ])
    """

    def __init__(
        self, examples: Optional[List[Union[Example, Mapping[str, Any], str]]] = None
    ):
        self.examples: List[Example] = []
        if examples:
            for example in examples:
                if isinstance(example, Example):
                    self.examples.append(example)
                elif isinstance(example, str):
                    # String input: use as inputs with empty labels
                    self.examples.append(Example(inputs=example, labels=""))
                elif isinstance(example, Mapping):
                    self.add(**example)
                else:
                    raise TypeError(
                        f"Expected Example, dict, or str, got {type(example).__name__}"
                    )

    def _add(self, example: Example):
        self.examples.append(example)

    def add(
        self,
        inputs: Union[str, Mapping[str, Any]],
        labels: Union[str, Mapping[str, Any]],
        title: Optional[str] = None,
        reasoning: Optional[str] = None,
        topic: Optional[str] = None,
    ):
        """Add an example to the collection."""
        example = Example(
            inputs=inputs,
            labels=labels,
            title=title,
            reasoning=reasoning,
            topic=topic,
        )
        self._add(example)

    def get_examples(self) -> List[Example]:
        """Get all examples in the collection."""
        return self.examples

    def filter_by_topic(self, topic: str) -> List[Example]:
        """Filter examples by topic."""
        return [ex for ex in self.examples if ex.topic == topic]

    def get_examples_needing_transform(self) -> List[Example]:
        """Returns examples that need transformation."""
        return [
            ex
            for ex in self.examples
            if ex._inputs_needs_transform or ex._labels_needs_transform
        ]

    def transform(
        self,
        input_transformer: Optional[callable] = None,
        output_transformer: Optional[callable] = None,
    ) -> "ExampleCollection":
        """Transforms all examples to strings."""
        transformed = [
            ex.transform(input_transformer, output_transformer) for ex in self.examples
        ]
        return ExampleCollection(transformed)

    def get_formatted(
        self,
        format: Union[ExampleFormat, str] = ExampleFormat.XML,
        input_transformer: Optional[callable] = None,
        output_transformer: Optional[callable] = None,
        separator: Optional[str] = None,
    ) -> Optional[str]:
        """Format the entire collection according to specified format.

        Args:
            format: Output format - "xml", "plaintext", or "minimal".
            input_transformer: Function to transform input dict to string.
            output_transformer: Function to transform output dict to string.
            separator: Custom separator between examples. Defaults to
                "\\n\\n" for xml/plaintext, "\\n" for minimal.

        Returns:
            Formatted examples string, or None if collection is empty.

        Examples:
            >>> collection = ExampleCollection([
            ...     Example(inputs="What is 2+2?", labels="4"),
            ...     Example(inputs="What is 3+3?", labels="6"),
            ... ])

            >>> # XML format (default)
            >>> print(collection.get_formatted())
            <example id=1>
            <input>What is 2+2?</input>
            <output>4</output>
            </example>
            ...

            >>> # Plaintext format
            >>> print(collection.get_formatted(format="plaintext"))
            Example 1:
            Input: What is 2+2?
            Output: 4
            ...

            >>> # Minimal format
            >>> print(collection.get_formatted(format="minimal"))
            Q: What is 2+2?
            A: 4
            ...
        """
        if not self.examples:
            return None

        # Normalize format
        if isinstance(format, str):
            format = ExampleFormat(format)

        # Set default separator based on format
        if separator is None:
            separator = "\n" if format == ExampleFormat.MINIMAL else "\n\n"

        # Transform examples if needed
        collection = self.transform(input_transformer, output_transformer)

        # Format each example
        formatted_parts = []
        for i, example in enumerate(collection.get_examples(), start=1):
            formatted_parts.append(example.format(format, example_id=i))

        return separator.join(formatted_parts)
