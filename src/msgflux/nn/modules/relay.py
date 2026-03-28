from typing import Any, Dict, Optional

from msgflux.auto import AutoParams
from msgflux.nn.modules.module import Module


class Relay(Module, metaclass=AutoParams):
    """Declarative base module with message_fields and response_mode pre-wired.

    Relay provides the extraction/response scaffolding without requiring a model.
    Subclass it and implement ``forward()`` to build custom processing modules
    that read from ``message_fields`` and write via ``response_mode``.

    Example:
        >>> class TextCleaner(nn.Relay):
        ...     def forward(self, message, **kwargs):
        ...         text = self._extract_message_values(self.task, message)
        ...         result = text.strip().lower()
        ...         return self._define_response_mode(result, message)
        ...
        >>> cleaner = TextCleaner(
        ...     message_fields={"task": "inputs.raw_text"},
        ...     response_mode="inputs.clean_text",
        ... )
        >>> cleaner(msg)
    """

    # Configure AutoParams to use docstring as 'description' parameter
    _autoparams_use_docstring_for = "description"
    # Configure AutoParams to use class name as 'name' parameter
    _autoparams_use_classname_for = "name"

    def __init__(
        self,
        *,
        modules: Optional[Dict[str, "Module"]] = None,
        hooks: Optional[list] = None,
        message_fields: Optional[Dict[str, Any]] = None,
        response_mode: Optional[str] = None,
        annotations: Optional[Dict[str, type]] = None,
        description: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """Initialize the Relay module.

        Args:
        modules:
            Dictionary mapping attribute names to Module instances.
            Each entry is registered as a submodule on ``self``.
            !!! example
                modules={"agent": my_agent, "embedder": my_embedder}
                # Access via self.agent, self.embedder in forward()
        hooks:
            List of Hook instances to register on the module.
        message_fields:
            Dictionary mapping Message field names to their paths in the Message object.
            Valid keys: "task", "task_multimodal", "model_preference"
            !!! example
                message_fields={
                    "task": "data.input",
                    "model_preference": "model.preference"
                }

            Field descriptions:
            - task: Field path for task input (str, dict, tuple, or list)
            - task_multimodal: Field path for multimodal input (dict)
            - model_preference: Field path for model preference (str)
        response_mode:
            Controls how the response is returned.
            * ``None`` (default): Returns the response directly.
            * ``"<path>"``: Writes to ``obj.<path>`` and returns ``None``
              (``dotdict`` or ``Message`` is mutated in place).
        annotations:
            Dictionary mapping parameter names to their types.
            Used for type validation and schema generation.
        description:
            Module description. Useful when using the module as a tool
            or for documentation purposes.
        name:
            Module name in snake case format.
        """
        super().__init__()
        self._register_modules(modules)
        self._set_hooks(hooks)
        self._set_message_fields(message_fields)
        self._set_response_mode(response_mode)
        self.set_annotations(annotations or {})
        self.set_description(description)
        if name:
            self.set_name(name)

    def _register_modules(self, modules: Optional[Dict[str, "Module"]] = None):
        if modules is None:
            return
        if not isinstance(modules, dict):
            raise TypeError(
                f"`modules` must be a dict or None, given `{type(modules)}`"
            )
        for attr_name, module in modules.items():
            if not isinstance(module, Module):
                raise TypeError(
                    f"Module `{attr_name}` must be a Module instance, "
                    f"given `{type(module)}`"
                )
            setattr(self, attr_name, module)
