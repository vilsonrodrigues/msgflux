from typing import Any

from msgflux.runtime.events import emit_brief_message


class Brief:
    """Send a short progress update to the user during execution."""

    name = "brief"
    display_name = "Brief"
    description = (
        "Send a short non-blocking progress update to the user. Use this when "
        "the user should know what is happening while work continues."
    )
    read_only = True
    concurrency_safe = True

    def __call__(
        self,
        message: str,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Send a brief progress update to the user.

        Args:
            message: Short user-facing progress update.
            title: Optional short title for UI surfaces.
            metadata: Optional adapter-specific metadata.
        """
        normalized_message = message.strip()
        if not normalized_message:
            raise ValueError("Brief `message` cannot be empty.")

        normalized_title = title.strip() if isinstance(title, str) else None
        if normalized_title == "":
            normalized_title = None

        emit_brief_message(
            normalized_message,
            title=normalized_title,
            metadata=metadata,
        )
        return {"message": "Brief sent to the user."}

    async def acall(
        self,
        message: str,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self(message=message, title=title, metadata=metadata)
