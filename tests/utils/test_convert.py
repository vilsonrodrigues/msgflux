from msgflux.utils.convert import (
    convert_camel_snake_to_title,
    convert_camel_to_snake_case,
    convert_str_to_hash,
)


def test_convert_camel_snake_to_title():
    assert convert_camel_snake_to_title("camelCase") == "Camel Case"
    assert convert_camel_snake_to_title("snake_case") == "Snake Case"
    assert convert_camel_snake_to_title("already title") == "Already Title"


def test_convert_camel_to_snake_case():
    assert convert_camel_to_snake_case("camelCase") == "camel_case"
    assert convert_camel_to_snake_case("CamelCase") == "camel_case"


def test_convert_str_to_hash():
    assert (
        convert_str_to_hash("hello world")
        == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    )
