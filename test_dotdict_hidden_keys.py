"""Test dotdict hidden_keys functionality."""
from msgflux.dotdict import dotdict


def test_hidden_keys_basic():
    """Test that hidden keys are not returned by get()."""
    print("Testing basic hidden_keys functionality...")

    # Create dotdict with hidden keys
    d = dotdict(
        {"api_key": "secret123", "username": "john", "password": "pass456"},
        hidden_keys=["api_key", "password"],
    )

    # Test that hidden keys return None
    assert d.get("api_key") is None, "Hidden key should return None"
    assert d.get("password") is None, "Hidden key should return None"

    # Test that visible keys work normally
    assert d.get("username") == "john", "Visible key should return value"

    print("✓ Basic hidden_keys test passed!")


def test_hidden_keys_with_default():
    """Test that hidden keys respect custom default values."""
    print("\nTesting hidden_keys with custom default...")

    d = dotdict(
        {"api_key": "secret123", "name": "Alice"},
        hidden_keys=["api_key"],
    )

    # Test custom default
    assert d.get("api_key", "HIDDEN") == "HIDDEN", "Should return custom default"
    assert d.get("name", "DEFAULT") == "Alice", "Should return actual value"

    print("✓ Custom default test passed!")


def test_hidden_keys_nested_paths():
    """Test that hidden keys work with nested paths."""
    print("\nTesting hidden_keys with nested paths...")

    d = dotdict(
        {
            "credentials": {"api_key": "secret", "token": "xyz"},
            "user": {"name": "Bob"},
        },
        hidden_keys=["credentials"],
    )

    # Test that nested path with hidden root key returns None
    assert d.get("credentials.api_key") is None, "Hidden nested path should return None"
    assert d.get("credentials.token") is None, "Hidden nested path should return None"

    # Test that non-hidden nested paths work
    assert d.get("user.name") == "Bob", "Visible nested path should work"

    print("✓ Nested paths test passed!")


def test_hidden_keys_not_in_repr():
    """Test that hidden keys are not shown in __repr__ and __str__."""
    print("\nTesting hidden_keys in __repr__ and __str__...")

    d = dotdict(
        {"public": "visible", "secret": "hidden"},
        hidden_keys=["secret"],
    )

    repr_str = repr(d)
    str_str = str(d)

    # Check that hidden key is not in string representations
    assert "secret" not in repr_str, "Hidden key should not appear in repr"
    assert "secret" not in str_str, "Hidden key should not appear in str"
    assert "hidden" not in repr_str, "Hidden value should not appear in repr"
    assert "hidden" not in str_str, "Hidden value should not appear in str"

    # Check that visible key is in string representations
    assert "public" in repr_str, "Visible key should appear in repr"
    assert "public" in str_str, "Visible key should appear in str"
    assert "visible" in repr_str, "Visible value should appear in repr"
    assert "visible" in str_str, "Visible value should appear in str"

    print("✓ __repr__ and __str__ test passed!")


def test_hidden_keys_propagation():
    """Test that hidden_keys propagate to nested dotdicts."""
    print("\nTesting hidden_keys propagation...")

    d = dotdict(
        {
            "level1": {
                "api_key": "secret",
                "data": {"nested_key": "value"},
            }
        },
        hidden_keys=["api_key"],
    )

    # Access nested dotdict
    nested = d.get("level1")
    assert isinstance(nested, dotdict), "Nested value should be dotdict"

    # Check that hidden_keys propagated
    assert nested.get("api_key") is None, "Hidden key should be hidden in nested dotdict"
    assert nested.get("data.nested_key") == "value", "Other keys should work normally"

    print("✓ Propagation test passed!")


def test_hidden_keys_direct_access_still_works():
    """Test that direct access (via dot notation or bracket) still works for hidden keys."""
    print("\nTesting direct access to hidden keys...")

    d = dotdict(
        {"api_key": "secret123", "name": "Charlie"},
        hidden_keys=["api_key"],
    )

    # Direct access should still work (hidden_keys only affects get())
    assert d["api_key"] == "secret123", "Direct bracket access should work"
    assert d.api_key == "secret123", "Direct dot access should work"

    print("✓ Direct access test passed!")


def test_empty_hidden_keys():
    """Test dotdict with no hidden keys (default behavior)."""
    print("\nTesting dotdict without hidden_keys...")

    d = dotdict({"key1": "value1", "key2": "value2"})

    # Should work normally
    assert d.get("key1") == "value1"
    assert d.get("key2") == "value2"

    print("✓ Empty hidden_keys test passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("DOTDICT HIDDEN_KEYS TESTS")
    print("=" * 60)

    test_hidden_keys_basic()
    test_hidden_keys_with_default()
    test_hidden_keys_nested_paths()
    test_hidden_keys_not_in_repr()
    test_hidden_keys_propagation()
    test_hidden_keys_direct_access_still_works()
    test_empty_hidden_keys()

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
