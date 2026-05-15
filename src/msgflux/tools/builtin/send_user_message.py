from typing import Literal

from msgflux.runtime.events import emit_user_message_sent

SEND_USER_MESSAGE_TOOL_NAME = "SendUserMessage"
LEGACY_BRIEF_TOOL_NAME = "Brief"
SendUserMessageStatus = Literal["info", "progress", "success", "warning", "error"]


class SendUserMessage:
    """Send a short message to the user during execution."""

    name = SEND_USER_MESSAGE_TOOL_NAME
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
        status: SendUserMessageStatus = "info",
        attachments: str | list[str] | None = None,
    ) -> str:
        """Send a short message to the user.

        Args:
            message: Short user-facing message.
            status: Message status for UI surfaces.
            attachments: Optional local path or list of local paths for the
                runtime event consumer to render.
        """
        normalized_message = message.strip()
        if not normalized_message:
            raise ValueError("SendUserMessage `message` cannot be empty.")
        if status not in {"info", "progress", "success", "warning", "error"}:
            raise ValueError(f"Invalid SendUserMessage status: {status}")

        normalized_attachments = self._normalize_attachments(attachments)

        emit_user_message_sent(
            normalized_message,
            status=status,
            attachments=normalized_attachments,
        )
        return "Message sent to the user."

    def _normalize_attachments(
        self,
        attachments: str | list[str] | None,
    ) -> list[str]:
        if attachments is None:
            return []
        if isinstance(attachments, str):
            normalized = [attachments.strip()]
        elif isinstance(attachments, list):
            normalized = [
                attachment.strip()
                for attachment in attachments
                if isinstance(attachment, str)
            ]
            if len(normalized) != len(attachments):
                raise TypeError("SendUserMessage attachments must be paths as strings.")
        else:
            raise TypeError(
                "SendUserMessage attachments must be a path or list of paths."
            )

        return [attachment for attachment in normalized if attachment]

    async def acall(
        self,
        message: str,
        status: SendUserMessageStatus = "info",
        attachments: str | list[str] | None = None,
    ) -> str:
        return self(
            message=message,
            status=status,
            attachments=attachments,
        )


Brief = SendUserMessage
