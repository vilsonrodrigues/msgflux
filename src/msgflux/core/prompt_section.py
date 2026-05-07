from inspect import cleandoc
from types import FunctionType, MethodType
from typing import Any, Dict, Optional, Union

PromptContent = Union[str, "PromptSection"]


class PromptSection:
    """Programmatic prompt section rendered as concise key-value text."""

    def __init__(self, **fields: Any):
        self._fields = fields

    @classmethod
    def _class_fields(cls) -> Dict[str, Any]:
        fields = {}
        for section_cls in reversed(cls.__mro__):
            if section_cls in (PromptSection, object):
                continue
            for name, value in section_cls.__dict__.items():
                if name.startswith("_") or name == "__doc__":
                    continue
                if isinstance(
                    value,
                    (FunctionType, MethodType, classmethod, staticmethod, property),
                ):
                    continue
                fields[name] = value
        return fields

    def to_dict(self) -> Dict[str, Any]:
        fields = self._class_fields()
        fields.update(self._fields)
        return {name: self._normalize_value(value) for name, value in fields.items()}

    @classmethod
    def _normalize_value(cls, value: Any) -> Any:
        if isinstance(value, str):
            return cleandoc(value)
        if isinstance(value, PromptSection):
            return value.to_dict()
        if isinstance(value, type) and issubclass(value, PromptSection):
            return value().to_dict()
        if isinstance(value, dict):
            return {key: cls._normalize_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._normalize_value(item) for item in value]
        return value

    def render(self) -> str:
        return self._render_mapping(self.to_dict())

    @classmethod
    def _render_mapping(cls, value: Dict[str, Any], *, indent: int = 0) -> str:
        lines = []
        prefix = " " * indent
        for key, item in value.items():
            if isinstance(item, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(cls._render_mapping(item, indent=indent + 2))
            elif isinstance(item, list):
                lines.append(f"{prefix}{key}:")
                lines.extend(cls._render_list(item, indent=indent + 2))
            else:
                lines.append(f"{prefix}{key}: {cls._render_scalar(item)}")
        return "\n".join(lines)

    @classmethod
    def _render_list(cls, value: list[Any], *, indent: int) -> list[str]:
        lines = []
        prefix = " " * indent
        for item in value:
            if isinstance(item, dict):
                lines.append(f"{prefix}-")
                lines.append(cls._render_mapping(item, indent=indent + 2))
            elif isinstance(item, list):
                lines.append(f"{prefix}-")
                lines.extend(cls._render_list(item, indent=indent + 2))
            else:
                lines.append(f"{prefix}- {cls._render_scalar(item)}")
        return lines

    @staticmethod
    def _render_scalar(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_dict()!r})"


def normalize_prompt_content(
    value: Any,
    *,
    field_name: str,
) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return cleandoc(value)
    if isinstance(value, PromptSection):
        return str(value)
    if isinstance(value, type) and issubclass(value, PromptSection):
        return str(value())
    raise TypeError(
        f"`{field_name}` requires a string, PromptSection, or None "
        f"given `{type(value)}`"
    )
