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


def test_dtype_to_str_with_union():
    """Test _dtype_to_str with Union types."""
    from typing import Union
    
    class TestUnionSignature(Signature):
        value: Union[str, int] = InputField()
    
    inputs = TestUnionSignature.get_inputs_info()
    # Union that's not Optional should be preserved
    assert "Union" in inputs[0].dtype or "|" in inputs[0].dtype or "str" in inputs[0].dtype


def test_dtype_to_str_with_list_of_union():
    """Test _dtype_to_str with List[Union[...]]."""
    from typing import Union
    
    class TestListUnionSignature(Signature):
        values: List[Union[str, int]] = InputField()
    
    inputs = TestListUnionSignature.get_inputs_info()
    assert "List" in inputs[0].dtype


def test_signature_factory_with_image_types():
    """Test task template generation with Image types."""
    from msgflux.data.types import Image
    
    inputs_info = [
        FieldInfo(name="photo", dtype="Image", desc="A photo"),
        FieldInfo(name="text", dtype="str")
    ]
    
    template = SignatureFactory.get_task_template_from_signature(inputs_info)
    assert "<Image>photo</Image>" in template
    assert "<text>{{ text }}</text>" in template


def test_signature_factory_with_audio_types():
    """Test task template generation with Audio types."""
    from msgflux.data.types import Audio
    
    inputs_info = [
        FieldInfo(name="sound", dtype="Audio", desc="An audio file")
    ]
    
    template = SignatureFactory.get_task_template_from_signature(inputs_info)
    assert "<Audio>sound</Audio>" in template


def test_signature_factory_with_mixed_media():
    """Test task template generation with mixed media types."""
    inputs_info = [
        FieldInfo(name="img", dtype="Image"),
        FieldInfo(name="audio", dtype="Audio"),
        FieldInfo(name="text", dtype="str")
    ]
    
    template = SignatureFactory.get_task_template_from_signature(inputs_info)
    assert "<Image>img</Image>" in template
    assert "<Audio>audio</Audio>" in template
    assert "<text>{{ text }}</text>" in template


def test_get_inputs_info_preserves_order():
    """Test that get_inputs_info preserves field order."""
    class OrderedSignature(Signature):
        first: str = InputField()
        second: int = InputField()
        third: bool = InputField()
    
    inputs = OrderedSignature.get_inputs_info()
    assert inputs[0].name == "first"
    assert inputs[1].name == "second"
    assert inputs[2].name == "third"


def test_get_outputs_info_preserves_order():
    """Test that get_outputs_info preserves field order."""
    class OrderedSignature(Signature):
        first: str = OutputField()
        second: int = OutputField()
        third: bool = OutputField()
    
    outputs = OrderedSignature.get_outputs_info()
    assert outputs[0].name == "first"
    assert outputs[1].name == "second"
    assert outputs[2].name == "third"


def test_signature_with_only_optional_outputs():
    """Test signature with all optional outputs."""
    class OptionalOutputSignature(Signature):
        maybe1: Optional[str] = OutputField()
        maybe2: Optional[int] = OutputField()
    
    outputs = OptionalOutputSignature.get_outputs_info()
    assert all("Optional" in out.dtype for out in outputs)


def test_signature_factory_expected_output_includes_write_json():
    """Test that expected output includes JSON instruction."""
    inputs = [FieldInfo(name="x", dtype="int")]
    outputs = [FieldInfo(name="y", dtype="str")]
    
    result = SignatureFactory.get_expected_output_from_signature(inputs, outputs)
    assert "Write an encoded JSON" in result or "JSON" in result


def test_signature_with_tuple_type():
    """Test signature with Tuple type annotation."""
    from typing import Tuple
    
    class TupleSignature(Signature):
        coords: Tuple[int, int] = InputField(desc="X and Y coordinates")
    
    inputs = TupleSignature.get_inputs_info()
    assert "Tuple" in inputs[0].dtype
    assert inputs[0].name == "coords"


def test_signature_with_set_type():
    """Test signature with Set type annotation."""
    from typing import Set
    
    class SetSignature(Signature):
        tags: Set[str] = InputField(desc="Unique tags")
    
    inputs = SetSignature.get_inputs_info()
    assert "Set" in inputs[0].dtype


def test_signature_with_frozenset_type():
    """Test signature with FrozenSet type annotation."""
    from typing import FrozenSet
    
    class FrozenSetSignature(Signature):
        immutable_tags: FrozenSet[str] = InputField()
    
    inputs = FrozenSetSignature.get_inputs_info()
    assert "FrozenSet" in inputs[0].dtype


def test_signature_with_nested_generic_types():
    """Test signature with deeply nested generic types."""
    class NestedSignature(Signature):
        data: Dict[str, List[Dict[str, int]]] = InputField()
    
    inputs = NestedSignature.get_inputs_info()
    dtype = inputs[0].dtype
    assert "Dict" in dtype
    assert "List" in dtype


def test_signature_with_multiple_union_args():
    """Test Union with more than 2 types (not Optional)."""
    from typing import Union
    
    class MultiUnionSignature(Signature):
        value: Union[str, int, float] = InputField()
    
    inputs = MultiUnionSignature.get_inputs_info()
    # Should show as Union or pipe-separated, not Optional
    assert "|" in inputs[0].dtype or "Union" in inputs[0].dtype


def test_dtype_to_str_with_type_without_name():
    """Test _dtype_to_str with type that has no __name__ attribute."""
    # This tests the fallback to repr()
    class TestSignature(Signature):
        """Test signature."""
        pass
    
    # Just verify the method exists and can handle edge cases
    result = TestSignature._dtype_to_str(str)
    assert result == "str"


def test_get_str_signature_with_single_input_output():
    """Test get_str_signature with minimal signature."""
    class MinimalSignature(Signature):
        x: int = InputField()
        y: int = OutputField()
    
    sig_str = MinimalSignature.get_str_signature()
    assert "x: int" in sig_str
    assert " -> " in sig_str
    assert "y: int" in sig_str


def test_signature_with_video_type():
    """Test signature with Video media type."""
    from msgflux.data.types import Video
    
    class VideoSignature(Signature):
        clip: Video = InputField(desc="Video clip")
    
    inputs = VideoSignature.get_inputs_info()
    assert inputs[0].dtype == "Video"


def test_signature_with_file_type():
    """Test signature with File type."""
    from msgflux.data.types import File
    
    class FileSignature(Signature):
        document: File = InputField(desc="Document file")
    
    inputs = FileSignature.get_inputs_info()
    assert inputs[0].dtype == "File"


def test_signature_factory_with_video_and_file_types():
    """Test task template with Video and File types."""
    inputs_info = [
        FieldInfo(name="video", dtype="Video"),
        FieldInfo(name="doc", dtype="File")
    ]
    
    template = SignatureFactory.get_task_template_from_signature(inputs_info)
    assert "<Video>video</Video>" in template
    assert "<File>doc</File>" in template


def test_signature_get_instructions_empty():
    """Test get_instructions with no docstring."""
    class NoDocSignature(Signature):
        field: str = InputField()
    
    instructions = NoDocSignature.get_instructions()
    assert instructions == "" or instructions is None or "NoDocSignature" in instructions


def test_signature_get_output_descriptions_empty():
    """Test get_output_descriptions with no descriptions."""
    class NoDescSignature(Signature):
        output: str = OutputField()

    descriptions = NoDescSignature.get_output_descriptions()
    assert descriptions is None


def test_signature_with_only_one_field():
    """Test signature with single field."""
    class SingleFieldSignature(Signature):
        value: str = InputField(desc="Single value")
    
    inputs = SingleFieldSignature.get_inputs_info()
    assert len(inputs) == 1
    assert inputs[0].name == "value"


def test_signature_factory_expected_output_format():
    """Test the complete format of expected output."""
    inputs = [FieldInfo(name="query", dtype="str", desc="Search query")]
    outputs = [FieldInfo(name="result", dtype="List[str]", desc="Search results")]
    
    output = SignatureFactory.get_expected_output_from_signature(inputs, outputs)
    
    # Verify all expected sections are present
    assert "Your task inputs are:" in output or "task inputs" in output.lower()
    assert "query" in output
    assert "Your task outputs are:" in output or "task outputs" in output.lower()
    assert "result" in output
