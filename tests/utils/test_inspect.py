from msgflux.utils.inspect import fn_has_parameters, get_filename, get_mime_type


def test_fn_has_parameters():
    def func_with_params(a, b):
        pass

    def func_without_params():
        pass

    assert fn_has_parameters(func_with_params)
    assert not fn_has_parameters(func_without_params)


def test_get_mime_type():
    assert get_mime_type("image.jpg") == "image/jpeg"
    assert get_mime_type("audio.mp3") == "audio/mpeg"
    assert get_mime_type("unknown.foo") == "application/octet-stream"


def test_get_filename():
    assert get_filename("http://example.com/path/to/file.txt") == "file.txt"
    assert get_filename("/local/path/to/file.txt") == "file.txt"
