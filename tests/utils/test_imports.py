import pytest
from msgflux.utils.imports import import_dependencies, import_module_from_lib


def test_import_module_from_lib():
    assert import_module_from_lib("Path", "pathlib")
    with pytest.raises(ImportError):
        import_module_from_lib("NonExistent", "nonexistentlib")
    with pytest.raises(AttributeError):
        import_module_from_lib("NonExistent", "pathlib")
