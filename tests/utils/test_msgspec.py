import os
from typing import Optional

import msgspec
import pytest

from msgflux.utils.msgspec import (
    StructFactory,
    export_to_json,
    is_optional_field,
    load,
    msgspec_dumps,
    read_json,
    save,
    struct_to_dict,
)


class MyStruct(msgspec.Struct):
    a: int
    b: str
    c: Optional[int] = None


JSON_SCHEMA = {
    "$defs": {
        "MyStruct": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "string"}},
            "required": ["a", "b"],
        }
    },
    "$ref": "#/definitions/MyStruct",
}


class TestStructFactory:
    def test_from_json_schema(self):
        struct = StructFactory.from_json_schema(JSON_SCHEMA)
        assert issubclass(struct, msgspec.Struct)
        instance = struct(a=1, b="2")
        assert instance.a == 1

    def test_from_signature(self):
        struct = StructFactory.from_signature("a: int, b: str")
        assert issubclass(struct, msgspec.Struct)
        instance = struct(a=1, b="2")
        assert instance.a == 1


def test_msgspec_dumps():
    instance = MyStruct(a=1, b="2")
    assert msgspec_dumps(instance) == '{"a":1,"b":"2","c":null}'


@pytest.fixture
def temp_file(tmp_path):
    return tmp_path / "test.json"


def test_save_and_load(temp_file):
    data = {"a": 1, "b": "2"}
    save(data, str(temp_file))
    loaded_data = load(str(temp_file))
    assert data == loaded_data


def test_export_and_read_json(temp_file):
    data = {"a": 1, "b": "2"}
    export_to_json(data, str(temp_file))
    loaded_data = read_json(str(temp_file))
    assert data == loaded_data


def test_struct_to_dict():
    instance = MyStruct(a=1, b="2")
    d = struct_to_dict(instance)
    assert d == {"a": 1, "b": "2", "c": None}


def test_is_optional_field():
    assert not is_optional_field(MyStruct, "a")
    assert is_optional_field(MyStruct, "c")
