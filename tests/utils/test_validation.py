import base64
from msgflux.utils.validation import is_base64, is_builtin_type, is_subclass_of


class A:
    pass


class B(A):  # B is a subclass of A
    pass


def test_is_subclass_of():
    assert is_subclass_of(B, A)
    assert not is_subclass_of(A, B)
    assert not is_subclass_of("not a class", A)


def test_is_builtin_type():
    assert is_builtin_type(1)
    assert is_builtin_type("hello")
    assert is_builtin_type([1, 2])
    assert not is_builtin_type(A())


def test_is_base64():
    valid_b64 = base64.b64encode(b"hello").decode()
    assert is_base64(valid_b64)
    assert not is_base64("not base64")
