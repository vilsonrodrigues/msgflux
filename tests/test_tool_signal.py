"""Tests for ToolSignal - dynamic tool behavior signaling."""

import pytest

from msgflux.nn.modules.tool import ToolLibrary
from msgflux.tools.signal import ToolSignal


class TestToolSignalCreation:
    """Tests for ToolSignal creation and basic functionality."""

    def test_tool_signal_with_result_only(self):
        """Test creating ToolSignal with just result."""
        signal = ToolSignal(result="test data")

        assert signal.result == "test data"
        assert signal.get("return_direct") is None

    def test_tool_signal_with_return_direct(self):
        """Test creating ToolSignal with return_direct override."""
        signal = ToolSignal(result="data", return_direct=True)

        assert signal.result == "data"
        assert signal.return_direct is True

    def test_tool_signal_with_multiple_overrides(self):
        """Test creating ToolSignal with multiple overrides."""
        signal = ToolSignal(
            result="data", return_direct=True, priority=10, custom="value"
        )

        assert signal.result == "data"
        assert signal.return_direct is True
        assert signal.priority == 10
        assert signal.custom == "value"

    def test_tool_signal_attribute_access(self):
        """Test dotdict attribute access."""
        signal = ToolSignal(result={"key": "value"}, return_direct=True)

        # Attribute access
        assert signal.result == {"key": "value"}
        assert signal.return_direct is True

        # Dict access
        assert signal["result"] == {"key": "value"}
        assert signal["return_direct"] is True

    def test_tool_signal_get_method(self):
        """Test get method with default values."""
        signal = ToolSignal(result="data", return_direct=True)

        assert signal.get("return_direct") is True
        assert signal.get("unknown_key") is None
        assert signal.get("unknown_key", "default") == "default"


class TestToolSignalSerialization:
    """Tests for ToolSignal serialization."""

    def test_to_dict(self):
        """Test to_dict serialization."""
        signal = ToolSignal(result="data", return_direct=True, priority=5)

        result = signal.to_dict()

        assert result == {
            "result": "data",
            "return_direct": True,
            "priority": 5,
        }

    def test_to_dict_with_nested_result(self):
        """Test to_dict with nested result."""
        signal = ToolSignal(
            result={"answer": "42", "confidence": 0.95},
            return_direct=True,
        )

        result = signal.to_dict()

        assert result["result"] == {"answer": "42", "confidence": 0.95}
        assert result["return_direct"] is True

    def test_to_json(self):
        """Test to_json serialization."""
        signal = ToolSignal(result="data", return_direct=True)

        json_bytes = signal.to_json()

        assert isinstance(json_bytes, bytes)
        assert b"result" in json_bytes
        assert b"data" in json_bytes
        assert b"return_direct" in json_bytes
        assert b"true" in json_bytes

    def test_to_json_with_complex_result(self):
        """Test to_json with complex result types."""
        signal = ToolSignal(
            result={"list": [1, 2, 3], "nested": {"key": "value"}},
            return_direct=True,
        )

        json_bytes = signal.to_json()

        assert isinstance(json_bytes, bytes)
        # Should not raise any serialization errors


class TestToolSignalInheritsDotdict:
    """Tests for dotdict inheritance behavior."""

    def test_inherits_dotdict_features(self):
        """Test that ToolSignal inherits dotdict features."""
        signal = ToolSignal(result="data", config={"nested": {"key": "value"}})

        # Nested access via get with dot path
        assert signal.get("config.nested.key") == "value"

    def test_update_method(self):
        """Test update method from dotdict."""
        signal = ToolSignal(result="data")
        signal.update({"return_direct": True})

        assert signal.return_direct is True

    def test_is_dict_subclass(self):
        """Test that ToolSignal is a dict subclass."""
        signal = ToolSignal(result="data")

        assert isinstance(signal, dict)


class TestToolSignalWithToolLibrary:
    """Tests for ToolSignal integration with ToolLibrary."""

    def test_normal_return_backward_compat(self):
        """Test that normal returns still work (backward compatibility)."""

        def simple_tool(x: int) -> str:
            """Simple tool."""
            return f"result: {x}"

        library = ToolLibrary(name="test", tools=[simple_tool])
        response = library([("id1", "simple_tool", {"x": 5})])

        assert response.return_directly is False
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].result == "result: 5"

    def test_tool_signal_return_direct_true(self):
        """Test ToolSignal with return_direct=True."""

        def signal_tool(x: int) -> ToolSignal:
            """Tool that returns ToolSignal."""
            return ToolSignal(result=f"signal: {x}", return_direct=True)

        library = ToolLibrary(name="test", tools=[signal_tool])
        response = library([("id1", "signal_tool", {"x": 10})])

        assert response.return_directly is True
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].result == "signal: 10"

    def test_tool_signal_return_direct_false(self):
        """Test ToolSignal with return_direct=False (doesn't force)."""

        def signal_tool(x: int) -> ToolSignal:
            """Tool that returns ToolSignal."""
            return ToolSignal(result=f"signal: {x}", return_direct=False)

        library = ToolLibrary(name="test", tools=[signal_tool])
        response = library([("id1", "signal_tool", {"x": 10})])

        # return_direct=False in signal doesn't force, keeps default behavior
        assert response.return_directly is False
        assert response.tool_calls[0].result == "signal: 10"

    def test_tool_signal_no_overrides(self):
        """Test ToolSignal without any overrides."""

        def signal_tool(x: int) -> ToolSignal:
            """Tool that returns ToolSignal without overrides."""
            return ToolSignal(result=f"signal: {x}")

        library = ToolLibrary(name="test", tools=[signal_tool])
        response = library([("id1", "signal_tool", {"x": 10})])

        assert response.return_directly is False
        assert response.tool_calls[0].result == "signal: 10"

    def test_mixed_tools_one_signal_return_direct(self):
        """Test multiple tools where one returns ToolSignal with return_direct."""

        def normal_tool(x: int) -> str:
            """Normal tool."""
            return f"normal: {x}"

        def signal_tool(x: int) -> ToolSignal:
            """Signal tool."""
            return ToolSignal(result=f"signal: {x}", return_direct=True)

        library = ToolLibrary(name="test", tools=[normal_tool, signal_tool])
        response = library([
            ("id1", "normal_tool", {"x": 1}),
            ("id2", "signal_tool", {"x": 2}),
        ])

        # If any tool signals return_direct=True, return_directly should be True
        assert response.return_directly is True
        assert response.tool_calls[0].result == "normal: 1"
        assert response.tool_calls[1].result == "signal: 2"


class TestToolSignalAsync:
    """Tests for async ToolSignal processing."""

    @pytest.mark.asyncio
    async def test_async_tool_signal_return_direct(self):
        """Test async ToolSignal with return_direct."""

        async def async_signal_tool(x: int) -> ToolSignal:
            """Async tool that returns ToolSignal."""
            return ToolSignal(result=f"async signal: {x}", return_direct=True)

        library = ToolLibrary(name="test", tools=[async_signal_tool])
        response = await library.acall([("id1", "async_signal_tool", {"x": 20})])

        assert response.return_directly is True
        assert response.tool_calls[0].result == "async signal: 20"

    @pytest.mark.asyncio
    async def test_async_normal_return(self):
        """Test async normal return (backward compat)."""

        async def async_tool(x: int) -> str:
            """Async tool."""
            return f"async: {x}"

        library = ToolLibrary(name="test", tools=[async_tool])
        response = await library.acall([("id1", "async_tool", {"x": 15})])

        assert response.return_directly is False
        assert response.tool_calls[0].result == "async: 15"


class TestToolSignalWithErrors:
    """Tests for ToolSignal behavior when errors occur."""

    def test_error_blocks_signal_return_direct(self):
        """Test that errors block ToolSignal return_direct override."""

        def signal_tool(x: int) -> ToolSignal:
            """Tool that returns ToolSignal with return_direct."""
            return ToolSignal(result=f"signal: {x}", return_direct=True)

        library = ToolLibrary(name="test", tools=[signal_tool])

        # Call with one valid tool and one invalid tool
        response = library([
            ("id1", "signal_tool", {"x": 10}),
            ("id2", "nonexistent_tool", {"y": 20}),
        ])

        # Even though signal_tool returned return_direct=True,
        # the error from nonexistent_tool should block it
        assert response.return_directly is False
        assert len(response.tool_calls) == 2

        # Find the results by id (order may vary)
        signal_result = response.get_by_id("id1")
        error_result = response.get_by_id("id2")

        assert signal_result.result == "signal: 10"
        assert error_result.error is not None

    @pytest.mark.asyncio
    async def test_async_error_blocks_signal_return_direct(self):
        """Test that errors block ToolSignal return_direct in async."""

        async def async_signal_tool(x: int) -> ToolSignal:
            """Async tool that returns ToolSignal."""
            return ToolSignal(result=f"async signal: {x}", return_direct=True)

        library = ToolLibrary(name="test", tools=[async_signal_tool])

        response = await library.acall([
            ("id1", "async_signal_tool", {"x": 10}),
            ("id2", "nonexistent_tool", {"y": 20}),
        ])

        # Error should block return_direct
        assert response.return_directly is False


class TestToolSignalUnknownOverrides:
    """Tests for handling unknown overrides."""

    def test_unknown_override_ignored(self):
        """Test that unknown overrides are ignored by ToolLibrary."""

        def signal_tool(x: int) -> ToolSignal:
            """Tool with unknown override."""
            return ToolSignal(
                result=f"signal: {x}",
                unknown_future_override=True,
                another_unknown=42,
            )

        library = ToolLibrary(name="test", tools=[signal_tool])
        response = library([("id1", "signal_tool", {"x": 10})])

        # Should not raise, unknown overrides are ignored
        assert response.return_directly is False
        assert response.tool_calls[0].result == "signal: 10"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
