from msgflux.runtime.events import emit_user_message_sent


class SendUserMessage:
    """Send a short message to the user during execution."""

    name = "SendUserMessage"
    display_name = "Send User Message"
    description = (
        "Send a short non-blocking message to the user. Use this when the user "
        "should know what is happening while work continues."
    )
    read_only = True
    concurrency_safe = True

    def __call__(
        self,
        message: str,
        title: str | None = None,
        attachments: list[dict[str, str]] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Send a short message to the user.

        Args:
            message: Short user-facing message.
            title: Optional short title for UI surfaces.
            attachments: Optional user-facing attachment descriptors, such as
                local image paths or URLs for the event consumer to render.
            metadata: Optional adapter-specific metadata.
        """
        normalized_message = message.strip()
        if not normalized_message:
            raise ValueError("SendUserMessage `message` cannot be empty.")

        normalized_title = title.strip() if isinstance(title, str) else None
        if normalized_title == "":
            normalized_title = None

        emit_user_message_sent(
            normalized_message,
            title=normalized_title,
            attachments=attachments,
            metadata=metadata,
        )
        return {"message": "Message sent to the user."}

    async def acall(
        self,
        message: str,
        title: str | None = None,
        attachments: list[dict[str, str]] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, str]:
        return self(
            message=message,
            title=title,
            attachments=attachments,
            metadata=metadata,
        )


Brief = SendUserMessage
