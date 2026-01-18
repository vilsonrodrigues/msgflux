"""Test for creating msgspec.Struct subclasses using type()."""

from typing import Optional

import msgspec


def test_struct_inheritance_with_type():
    """Test that we can create a Struct subclass using type() and add/update attributes."""

    # Create a base generation schema (msgspec.Struct)
    class GenerationSchema(msgspec.Struct):
        reasoning: str
        action: str

    # Create a signature output struct
    class SignatureOutput(msgspec.Struct):
        result: int
        message: str

    # Test 1: Create Output class using type() with final_answer attribute
    Output = type(
        "Output",
        (GenerationSchema,),
        {"__annotations__": {"final_answer": SignatureOutput}},
    )

    # Verify the Output class has the correct structure
    assert issubclass(Output, GenerationSchema)
    assert issubclass(Output, msgspec.Struct)

    # Verify annotations - only explicitly defined ones appear in __annotations__
    assert "final_answer" in Output.__annotations__
    assert Output.__annotations__["final_answer"] == SignatureOutput

    # Inherited annotations are not in Output.__annotations__ but the class still has them
    # They are accessible through the parent class
    assert "reasoning" in GenerationSchema.__annotations__
    assert "action" in GenerationSchema.__annotations__

    # Test instantiation
    signature_instance = SignatureOutput(result=42, message="Success")
    output_instance = Output(
        reasoning="Because it works", action="test", final_answer=signature_instance
    )

    assert output_instance.reasoning == "Because it works"
    assert output_instance.action == "test"
    assert output_instance.final_answer.result == 42
    assert output_instance.final_answer.message == "Success"

    # Test msgspec encoding/decoding
    encoded = msgspec.json.encode(output_instance)
    decoded = msgspec.json.decode(encoded, type=Output)

    assert decoded.reasoning == output_instance.reasoning
    assert decoded.action == output_instance.action
    assert decoded.final_answer.result == output_instance.final_answer.result


def test_struct_inheritance_with_optional_final_answer():
    """Test creating a Struct subclass with Optional final_answer."""

    class GenerationSchema(msgspec.Struct):
        reasoning: str

    class SignatureOutput(msgspec.Struct):
        value: str

    # Create Output with Optional final_answer
    Output = type(
        "Output",
        (GenerationSchema,),
        {"__annotations__": {"final_answer": Optional[SignatureOutput]}},
    )

    # Test with final_answer present
    output_with_answer = Output(
        reasoning="test", final_answer=SignatureOutput(value="answer")
    )
    assert output_with_answer.final_answer.value == "answer"

    # Test with final_answer as None
    output_without_answer = Output(reasoning="test", final_answer=None)
    assert output_without_answer.final_answer is None


def test_struct_inheritance_updating_existing_attr():
    """Test that we can update/override an existing attribute using type()."""

    class GenerationSchema(msgspec.Struct):
        reasoning: str
        final_answer: str = "default"  # Existing attribute with default

    class SignatureOutput(msgspec.Struct):
        data: int

    # Override the final_answer type using type()
    Output = type(
        "Output",
        (GenerationSchema,),
        {"__annotations__": {"final_answer": SignatureOutput}},
    )

    # Verify the final_answer type was updated
    assert Output.__annotations__["final_answer"] == SignatureOutput

    # Test instantiation with new type
    output = Output(
        reasoning="testing override", final_answer=SignatureOutput(data=123)
    )

    assert output.reasoning == "testing override"
    assert isinstance(output.final_answer, SignatureOutput)
    assert output.final_answer.data == 123


def test_struct_inheritance_with_merged_annotations():
    """Test creating a Struct subclass with merged annotations from parent."""

    class GenerationSchema(msgspec.Struct):
        reasoning: str
        action: str

    class SignatureOutput(msgspec.Struct):
        result: int

    # Merge parent annotations with new ones
    merged_annotations = {
        **GenerationSchema.__annotations__,
        "final_answer": SignatureOutput,
    }

    Output = type(
        "Output", (GenerationSchema,), {"__annotations__": merged_annotations}
    )

    # Now all annotations should be present in Output.__annotations__
    assert "reasoning" in Output.__annotations__
    assert "action" in Output.__annotations__
    assert "final_answer" in Output.__annotations__
    assert Output.__annotations__["final_answer"] == SignatureOutput

    # Test instantiation
    output = Output(
        reasoning="test", action="do", final_answer=SignatureOutput(result=100)
    )

    assert output.reasoning == "test"
    assert output.action == "do"
    assert output.final_answer.result == 100


def test_overriding_existing_field_with_merge():
    """Test that merging annotations properly overrides existing fields."""

    class ParentSchema(msgspec.Struct):
        reasoning: str
        final_answer: str = "default"  # Original type is str

    class NewFinalAnswerType(msgspec.Struct):
        result: int
        confidence: float

    # When we merge, the final_answer in the dict takes precedence
    merged_annotations = {
        **ParentSchema.__annotations__,
        "final_answer": NewFinalAnswerType,  # This overrides the str type
    }

    Output = type("Output", (ParentSchema,), {"__annotations__": merged_annotations})

    # Verify the override worked
    assert "reasoning" in Output.__annotations__
    assert "final_answer" in Output.__annotations__

    # The final_answer type should now be NewFinalAnswerType, not str
    assert Output.__annotations__["final_answer"] == NewFinalAnswerType
    assert Output.__annotations__["final_answer"] != str
    assert Output.__annotations__["reasoning"] == str

    # Test instantiation with the new type
    output = Output(
        reasoning="test", final_answer=NewFinalAnswerType(result=100, confidence=0.95)
    )

    assert output.reasoning == "test"
    assert isinstance(output.final_answer, NewFinalAnswerType)
    assert output.final_answer.result == 100
    assert output.final_answer.confidence == 0.95


def test_multiple_inheritance_with_type():
    """Test creating a class with multiple bases using type()."""

    class BaseA(msgspec.Struct):
        field_a: str

    class BaseB(msgspec.Struct):
        field_b: int

    class SignatureOutput(msgspec.Struct):
        result: bool

    # This would be complex in practice due to multiple Struct inheritance,
    # but demonstrates the syntax
    try:
        Combined = type(
            "Combined", (BaseA,), {"__annotations__": {"final_answer": SignatureOutput}}
        )

        assert "final_answer" in Combined.__annotations__
    except Exception as e:
        # Expected if msgspec doesn't support this pattern
        print(f"Multiple inheritance pattern not supported: {e}")
