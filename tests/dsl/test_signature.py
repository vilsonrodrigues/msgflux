
import pytest
from typing import List, Dict, Optional
from msgflux.dsl.signature import (
    Signature,
    InputField,
    OutputField,
    FieldInfo,
    SignatureFactory,
    Image,
    Audio,
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
    assert outputs_info[4] == FieldInfo(name="metadata", dtype="Dict[str, str]", desc=None)
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

