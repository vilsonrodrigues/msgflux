from typing import Dict, List, Optional

import pytest

from msgflux.data.types import Audio, Image
from msgflux.dsl.signature import (
    FieldInfo,
    InputField,
    OutputField,
    Signature,
    SignatureFactory,
)


class TestSignature(Signature):
    """A test signature for unit testing."""

    name: str = InputField(desc="The name of the user.")
    age: int = InputField()
    image: Image = InputField(desc="User's profile picture.")

    output_name: str = OutputField(desc="The processed name.")
    output_age: int = OutputField()
    output_audio: Audio = OutputField(desc="Generated audio output.")
    is_adult: bool = OutputField()
    metadata: Dict[str, str] = OutputField()
    tags: List[str] = OutputField()
    optional_field: Optional[str] = OutputField()


def test_get_instructions():
    assert TestSignature.get_instructions() == "A test signature for unit testing."


def test_get_str_signature():
    expected = "name: str, age: int, image: Image -> output_name: str, output_age: int, output_audio: Audio, is_adult: bool, metadata: Dict[str, str], tags: List[str], optional_field: Optional[str]"
    assert TestSignature.get_str_signature() == expected


def test_get_inputs_info():
    inputs_info = TestSignature.get_inputs_info()
    assert isinstance(inputs_info, list)
    assert len(inputs_info) == 3
    assert inputs_info[0] == FieldInfo(
        name="name", dtype="str", desc="The name of the user."
    )
    assert inputs_info[1] == FieldInfo(name="age", dtype="int", desc=None)
    assert inputs_info[2] == FieldInfo(
        name="image", dtype="Image", desc="User's profile picture."
    )


def test_get_outputs_info():
    outputs_info = TestSignature.get_outputs_info()
    assert isinstance(outputs_info, list)
    assert len(outputs_info) == 7
    assert outputs_info[0] == FieldInfo(
        name="output_name", dtype="str", desc="The processed name."
    )
    assert outputs_info[1] == FieldInfo(name="output_age", dtype="int", desc=None)
    assert outputs_info[2] == FieldInfo(
        name="output_audio", dtype="Audio", desc="Generated audio output."
    )
    assert outputs_info[3] == FieldInfo(name="is_adult", dtype="bool", desc=None)
    assert outputs_info[4] == FieldInfo(
        name="metadata", dtype="Dict[str, str]", desc=None
    )
    assert outputs_info[5] == FieldInfo(name="tags", dtype="List[str]", desc=None)
    assert outputs_info[6] == FieldInfo(
        name="optional_field", dtype="Optional[str]", desc=None
    )


def test_get_output_descriptions():
    descriptions = TestSignature.get_output_descriptions()
    expected = {
        "output_name": "The processed name.",
        "output_audio": "Generated audio output.",
    }
    assert descriptions == expected


def test_signature_factory_get_task_template_from_signature():
    inputs_info = TestSignature.get_inputs_info()
    template = SignatureFactory.get_task_template_from_signature(inputs_info)
    expected_template = """<name>{{ name }}</name>
<age>{{ age }}</age>
<Image>image</Image>"""
    assert template == expected_template


def test_signature_factory_get_expected_output_from_signature():
    inputs_info = TestSignature.get_inputs_info()
    outputs_info = TestSignature.get_outputs_info()
    output = SignatureFactory.get_expected_output_from_signature(
        inputs_info, outputs_info
    )
    assert "Your task inputs are:" in output
    assert "1. `name` (str): The name of the user." in output
    assert "2. `age` (int)" in output
    assert "3. `image` (Image): User's profile picture." in output
    assert "Your task outputs are:" in output
    assert "1. `output_name` (str): The processed name." in output
    assert "2. `output_age` (int)" in output
    assert "3. `output_audio` (Audio): Generated audio output." in output
    assert "4. `is_adult` (bool)" in output
    assert "5. `metadata` (Dict[str, str])" in output
    assert "6. `tags` (List[str])" in output
    assert "7. `optional_field` (Optional[str])" in output
    assert "Write an encoded JSON." in output


def test_field_info_dataclass():
    """Test FieldInfo dataclass creation."""
    field = FieldInfo(name="test", dtype="str", desc="Test field")
    assert field.name == "test"
    assert field.dtype == "str"
    assert field.desc == "Test field"


def test_field_info_without_description():
    """Test FieldInfo without description."""
    field = FieldInfo(name="test", dtype="int")
    assert field.name == "test"
    assert field.dtype == "int"
    assert field.desc is None


def test_input_field_creation():
    """Test InputField creation."""
    field = InputField(desc="Test input")
    assert field.desc == "Test input"


def test_output_field_creation():
    """Test OutputField creation."""
    field = OutputField(desc="Test output")
    assert field.desc == "Test output"


def test_field_without_description():
    """Test Field creation without description."""
    field = InputField()
    assert field.desc is None


def test_field_invalid_desc_type():
    """Test that Field raises TypeError for invalid desc type."""
    with pytest.raises(TypeError, match="`desc` must be a string or None"):
        InputField(desc=123)


def test_signature_examples_dataclass():
    """Test SignatureExamples dataclass."""
    from msgflux.dsl.signature import SignatureExamples
    
    examples = SignatureExamples(
        inputs={"name": "Alice", "age": 25},
        outputs={"result": "success"}
    )
    assert examples.inputs == {"name": "Alice", "age": 25}
    assert examples.outputs == {"result": "success"}


def test_signature_with_no_inputs():
    """Test signature with only outputs."""
    class OutputOnlySignature(Signature):
        result: str = OutputField(desc="The result")
    
    outputs = OutputOnlySignature.get_outputs_info()
    assert len(outputs) == 1
    assert outputs[0].name == "result"


def test_signature_with_no_outputs():
    """Test signature with only inputs."""
    class InputOnlySignature(Signature):
        data: str = InputField(desc="Input data")
    
    inputs = InputOnlySignature.get_inputs_info()
    assert len(inputs) == 1
    assert inputs[0].name == "data"


def test_signature_factory_task_template_with_empty_inputs():
    """Test task template generation with empty inputs."""
    template = SignatureFactory.get_task_template_from_signature([])
    assert template == ""


def test_signature_factory_expected_output_with_no_descriptions():
    """Test expected output generation when fields have no descriptions."""
    inputs = [FieldInfo(name="x", dtype="int")]
    outputs = [FieldInfo(name="y", dtype="int")]
    
    result = SignatureFactory.get_expected_output_from_signature(inputs, outputs)
    assert "1. `x` (int)" in result
    assert "1. `y` (int)" in result


def test_generate_annotations_from_signature():
    """Test generate_annotations_from_signature function."""
    from msgflux.dsl.signature import generate_annotations_from_signature

    inputs_info = TestSignature.get_inputs_info()
    annotations = generate_annotations_from_signature(inputs_info, TestSignature)

    assert "name" in annotations
    assert "age" in annotations
    assert annotations["name"] == str
    assert annotations["age"] == int


def test_signature_str_representation_format():
    """Test that signature string representation has correct arrow format."""
    sig_str = TestSignature.get_str_signature()
    assert " -> " in sig_str
    parts = sig_str.split(" -> ")
    assert len(parts) == 2


def test_signature_with_complex_types():
    """Test signature with complex type annotations."""
    class ComplexSignature(Signature):
        data: List[Dict[str, int]] = InputField(desc="Complex data structure")
        result: Optional[List[str]] = OutputField(desc="Optional result list")
    
    inputs = ComplexSignature.get_inputs_info()
    outputs = ComplexSignature.get_outputs_info()
    
    assert inputs[0].dtype == "List[Dict[str, int]]"
    assert outputs[0].dtype == "Optional[List[str]]"
