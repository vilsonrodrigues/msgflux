"""Tests for MCP authentication providers."""

import pytest
from datetime import datetime, timedelta

from msgflux.protocols.mcp.auth import (
    BearerTokenAuth,
    APIKeyAuth,
    BasicAuth,
    OAuth2Auth,
    CustomHeaderAuth,
)


class TestBearerTokenAuth:
    """Tests for Bearer Token Authentication."""

    def test_apply_auth(self):
        """Test applying bearer token to headers."""
        auth = BearerTokenAuth(token="test-token-123")
        headers = {}

        result = auth.apply_auth(headers)

        assert "Authorization" in result
        assert result["Authorization"] == "Bearer test-token-123"

    def test_custom_token_type(self):
        """Test custom token type."""
        auth = BearerTokenAuth(token="test-token", token_type="Token")
        headers = {}

        result = auth.apply_auth(headers)

        assert result["Authorization"] == "Token test-token"

    def test_update_token(self):
        """Test updating token."""
        auth = BearerTokenAuth(token="old-token")
        auth.update_token("new-token", expires_in=3600)

        headers = auth.apply_auth({})

        assert headers["Authorization"] == "Bearer new-token"
        assert not auth.is_expired()

    @pytest.mark.asyncio
    async def test_refresh_with_callback(self):
        """Test token refresh with callback."""
        async def refresh_callback():
            return "refreshed-token"

        auth = BearerTokenAuth(
            token="old-token",
            refresh_callback=refresh_callback
        )

        # Set expiration to past (expired)
        auth.set_expiration(-10)  # Expired 10 seconds ago

        # Token should be expired
        assert auth.is_expired()

        # Refresh
        result = await auth.refresh()

        assert result is True
        headers = auth.apply_auth({})
        assert headers["Authorization"] == "Bearer refreshed-token"


class TestAPIKeyAuth:
    """Tests for API Key Authentication."""

    def test_default_header(self):
        """Test default API key header."""
        auth = APIKeyAuth(api_key="my-api-key")
        headers = {}

        result = auth.apply_auth(headers)

        assert result["X-API-Key"] == "my-api-key"

    def test_custom_header(self):
        """Test custom header name."""
        auth = APIKeyAuth(api_key="my-key", header_name="X-Custom-Auth")
        headers = {}

        result = auth.apply_auth(headers)

        assert result["X-Custom-Auth"] == "my-key"

    def test_with_prefix(self):
        """Test API key with prefix."""
        auth = APIKeyAuth(
            api_key="my-key",
            header_name="Authorization",
            key_prefix="ApiKey"
        )
        headers = {}

        result = auth.apply_auth(headers)

        assert result["Authorization"] == "ApiKey my-key"

    def test_update_api_key(self):
        """Test updating API key."""
        auth = APIKeyAuth(api_key="old-key")
        auth.update_api_key("new-key")

        headers = auth.apply_auth({})

        assert headers["X-API-Key"] == "new-key"


class TestBasicAuth:
    """Tests for HTTP Basic Authentication."""

    def test_apply_auth(self):
        """Test basic auth encoding."""
        auth = BasicAuth(username="user", password="pass")
        headers = {}

        result = auth.apply_auth(headers)

        assert "Authorization" in result
        assert result["Authorization"].startswith("Basic ")

        # Decode and verify
        import base64
        encoded = result["Authorization"][6:]  # Skip "Basic "
        decoded = base64.b64decode(encoded).decode()
        assert decoded == "user:pass"

    def test_update_credentials(self):
        """Test updating credentials."""
        auth = BasicAuth(username="old-user", password="old-pass")
        auth.update_credentials("new-user", "new-pass")

        headers = auth.apply_auth({})

        import base64
        encoded = headers["Authorization"][6:]
        decoded = base64.b64decode(encoded).decode()
        assert decoded == "new-user:new-pass"


class TestOAuth2Auth:
    """Tests for OAuth2 Authentication."""

    def test_apply_auth(self):
        """Test OAuth2 token application."""
        auth = OAuth2Auth(access_token="access-123")
        headers = {}

        result = auth.apply_auth(headers)

        assert result["Authorization"] == "Bearer access-123"

    @pytest.mark.asyncio
    async def test_refresh_with_callback(self):
        """Test OAuth2 token refresh."""
        async def refresh_callback(refresh_token):
            assert refresh_token == "refresh-123"
            return {
                "access_token": "new-access-token",
                "refresh_token": "new-refresh-token",
                "expires_in": 3600,
            }

        auth = OAuth2Auth(
            access_token="old-access",
            refresh_token="refresh-123",
            expires_in=0,  # Expired
            refresh_callback=refresh_callback
        )

        result = await auth.refresh()

        assert result is True
        headers = auth.apply_auth({})
        assert headers["Authorization"] == "Bearer new-access-token"

    def test_update_tokens(self):
        """Test updating OAuth2 tokens."""
        auth = OAuth2Auth(access_token="old-token")
        auth.update_tokens(
            access_token="new-access",
            refresh_token="new-refresh",
            expires_in=7200
        )

        headers = auth.apply_auth({})

        assert headers["Authorization"] == "Bearer new-access"
        assert not auth.is_expired()


class TestCustomHeaderAuth:
    """Tests for Custom Header Authentication."""

    def test_static_headers(self):
        """Test static custom headers."""
        auth = CustomHeaderAuth(headers={
            "X-Custom-1": "value1",
            "X-Custom-2": "value2",
        })
        headers = {"Content-Type": "application/json"}

        result = auth.apply_auth(headers)

        assert result["Content-Type"] == "application/json"
        assert result["X-Custom-1"] == "value1"
        assert result["X-Custom-2"] == "value2"

    def test_callback_headers(self):
        """Test dynamic headers via callback."""
        call_count = [0]

        def get_headers():
            call_count[0] += 1
            return {"X-Request-ID": f"req-{call_count[0]}"}

        auth = CustomHeaderAuth(headers_callback=get_headers)

        result1 = auth.apply_auth({})
        result2 = auth.apply_auth({})

        assert result1["X-Request-ID"] == "req-1"
        assert result2["X-Request-ID"] == "req-2"

    def test_callback_overrides_static(self):
        """Test that callback overrides static headers."""
        def get_headers():
            return {"X-Dynamic": "from-callback"}

        auth = CustomHeaderAuth(
            headers={"X-Static": "value"},
            headers_callback=get_headers
        )

        result = auth.apply_auth({})

        assert result["X-Dynamic"] == "from-callback"
        assert "X-Static" not in result

    def test_update_headers(self):
        """Test updating static headers."""
        auth = CustomHeaderAuth(headers={"X-Old": "value"})
        auth.update_headers({"X-New": "new-value"})

        result = auth.apply_auth({})

        assert result["X-New"] == "new-value"
        assert "X-Old" not in result


class TestBaseAuthExpiration:
    """Tests for auth expiration logic."""

    def test_set_expiration(self):
        """Test setting expiration."""
        auth = BearerTokenAuth(token="test")
        auth.set_expiration(3600)

        assert not auth.is_expired()

    def test_expired_token(self):
        """Test expired token detection."""
        auth = BearerTokenAuth(token="test")
        auth.set_expiration(-1)  # Expired 1 second ago

        assert auth.is_expired()

    def test_get_auth_info(self):
        """Test getting auth information."""
        auth = BearerTokenAuth(token="test", expires_in=3600)
        info = auth.get_auth_info()

        assert info["type"] == "BearerTokenAuth"
        assert info["expired"] is False
        assert info["expires_at"] is not None
