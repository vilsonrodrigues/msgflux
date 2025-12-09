import base64
import io
import os

import pytest
import requests

from msgflux.utils.encode import (
    encode_base64_from_url,
    encode_data_to_base64,
    encode_data_to_bytes,
    encode_local_file_in_base64,
    encode_to_io_object,
)


@pytest.fixture
def mock_requests_get(mocker):
    mock_response = mocker.MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"hello world"
    mock_response.raise_for_status.return_value = None
    mock_response.url = "http://example.com/test.txt"
    # Make the mock work as a context manager that returns itself
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = None

    mocker.patch("requests.get", return_value=mock_response)


@pytest.fixture
def temp_file(tmp_path):
    file_path = tmp_path / "test.txt"
    file_path.write_text("hello world")
    return str(file_path)


def test_encode_base64_from_url(mock_requests_get):
    encoded = encode_base64_from_url("http://example.com")
    assert base64.b64decode(encoded) == b"hello world"


def test_encode_local_file_in_base64(temp_file):
    encoded = encode_local_file_in_base64(temp_file)
    assert base64.b64decode(encoded) == b"hello world"


def test_encode_data_to_base64(mock_requests_get, temp_file):
    assert encode_data_to_base64("not a path or url") == "not a path or url"
    assert base64.b64decode(encode_data_to_base64(temp_file)) == b"hello world"
    assert (
        base64.b64decode(encode_data_to_base64("http://example.com"))
        == b"hello world"
    )


def test_encode_to_io_object(mock_requests_get, temp_file):
    assert isinstance(encode_to_io_object(b"hello"), io.BytesIO)
    assert isinstance(encode_to_io_object(temp_file), io.IOBase)
    assert isinstance(encode_to_io_object("http://example.com"), io.BytesIO)
    b64_string = base64.b64encode(b"hello").decode()
    assert isinstance(encode_to_io_object(b64_string), io.BytesIO)
    with pytest.raises(ValueError):
        encode_to_io_object("not a valid input")


def test_encode_data_to_bytes(mock_requests_get, temp_file):
    assert isinstance(encode_data_to_bytes(b"hello"), io.BytesIO)
    assert isinstance(encode_data_to_bytes(temp_file), io.BytesIO)
    assert isinstance(encode_data_to_bytes("http://example.com"), io.BytesIO)
    b64_string = base64.b64encode(b"hello").decode()
    assert isinstance(encode_data_to_bytes(b64_string), io.BytesIO)
    with pytest.raises(ValueError):
        encode_data_to_bytes("not a valid input")
