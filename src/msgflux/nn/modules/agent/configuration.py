# ruff: noqa: A002

from inspect import cleandoc
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Type,
    Union,
    cast,
)

import msgspec

from msgflux.core.dotdict import dotdict
from msgflux.core.examples import Example, ExampleCollection
from msgflux.core.message import Message
from msgflux.dsl.signature import (
    Signature,
    SignatureFactory,
    generate_annotations_from_signature,
)
from msgflux.dsl.typed_parsers.registry import typed_parser_registry
from msgflux.generation.templates import (
    EXPECTED_OUTPUTS_TEMPLATE,
    SYSTEM_PROMPT_TEMPLATE,
    PromptSpec,
)
from msgflux.models import Model
from msgflux.models.gateway import ModelGateway
from msgflux.models.types import ChatCompletionModel
from msgflux.nn.hooks.events import (
    ModelContext,
)
from msgflux.nn.modules.generator import Generator
from msgflux.nn.modules.tool import ToolLibrary
from msgflux.nn.parameter import Parameter
from msgflux.runtime.context import (
    get_execution_context,
)
from msgflux.tools.catalog import ToolCatalogView
from msgflux.utils.chat import response_format_from_msgspec_struct
from msgflux.utils.msgspec import StructFactory, is_optional_field, msgspec_dumps
from msgflux.utils.validation import is_subclass_of

if TYPE_CHECKING:
    pass
from msgflux.nn.modules.agent.context import (
    _RESERVED_KWARGS,
    _UNSET,
    ToolFilter,
    _require_lifecycle_payload,
)


class AgentConfigurationMixin:
    """Agent configuration setters, signatures, and system-prompt rendering."""

    def _set_task_context(self, task_context: Optional[Union[str, List[str]]] = None):
        if isinstance(task_context, (str, list)) or task_context is None:
            if isinstance(task_context, str) and task_context == "":
                raise ValueError(
                    f"`task_context` requires a string not emptygiven `{task_context}`"
                )
            if isinstance(task_context, list) and not task_context:
                raise ValueError(
                    f"`task_context` requires a list not emptygiven `{task_context}`"
                )
            self.register_buffer("task_context", task_context)
        else:
            raise TypeError(
                "`task_context` requires a string, list or None"
                f"given `{type(task_context)}`"
            )

    def _set_context_cache(self, context_cache: Optional[str] = None):
        if isinstance(context_cache, str) or context_cache is None:
            self.register_buffer("context_cache", context_cache)
        else:
            raise TypeError(
                "`context_cache` requires a string or None"
                f"given `{type(context_cache)}`"
            )

    def _set_prefilling(self, prefilling: Optional[str] = None):
        if isinstance(prefilling, str) or prefilling is None:
            self.register_buffer("prefilling", prefilling)
        else:
            raise TypeError(
                f"`prefilling` requires a string or Nonegiven `{type(prefilling)}`"
            )

    def _set_tools(
        self,
        tools: Optional[List[Callable]] = None,
        mcp_servers: Optional[List[Mapping[str, Any]]] = None,
    ):
        tools = list(tools or [])
        self.tool_library = ToolLibrary(
            self.get_module_name(),
            tools,
            mcp_servers=mcp_servers,
        )
        self.tool_library.set_lifecycle_owner(self)
        self.tool_library.set_agent_inbox(self.agent_inbox)

    def _set_generation_schema(
        self, generation_schema: Optional[msgspec.Struct] = None
    ):
        if generation_schema is None or is_subclass_of(
            generation_schema, msgspec.Struct
        ):
            self.register_buffer("generation_schema", generation_schema)
        else:
            raise TypeError(
                "`generation_schema` need be a `msgspec.Struct` or None "
                f"given `{type(generation_schema)}`"
            )

    def _set_model(
        self, model: Union[ChatCompletionModel, ModelGateway, "Generator", str]
    ):
        if isinstance(model, str):
            model = Model.chat_completion(model)
        if isinstance(model, Generator):
            self.generator = model
        else:
            if (
                not hasattr(model, "model_type")
                or model.model_type != "chat_completion"
            ):
                raise TypeError(
                    f"`model` must be a `chat_completion` model, given `{type(model)}`"
                )
            self.generator = Generator(model)

    @property
    def model(self):
        """Access underlying model for convenience."""
        return self.generator.model

    @model.setter
    def model(self, value: Union[ChatCompletionModel, ModelGateway, "Generator", str]):
        self._set_model(value)

    def _set_system_message(self, system_message: Optional[str] = None):
        if isinstance(system_message, str) or system_message is None:
            if isinstance(system_message, str):
                system_message = cleandoc(system_message)
            if (
                hasattr(self.generation_schema, "system_message")
                and self.generation_schema.system_message is not None
            ):
                if system_message is None:
                    system_message = self.generation_schema.system_message
                else:
                    system_message = (
                        self.generation_schema.system_message + system_message
                    )
            self.system_message = Parameter(system_message, PromptSpec.SYSTEM_MESSAGE)
        else:
            raise TypeError(
                "`system_message` requires a string or None "
                f"given `{type(system_message)}`"
            )

    def _set_instructions(self, instructions: Optional[str] = None):
        if isinstance(instructions, str) or instructions is None:
            if isinstance(instructions, str):
                instructions = cleandoc(instructions)
            typed_parser_cls = typed_parser_registry.get(self.typed_parser, None)
            if typed_parser_cls is not None:
                instructions = self._format_template(
                    {"instructions": instructions}, typed_parser_cls.template
                )
            self.instructions = Parameter(instructions, PromptSpec.INSTRUCTIONS)
        else:
            raise TypeError(
                f"`instructions` requires a string or None given `{type(instructions)}`"
            )

    def _set_expected_output(self, expected_output: Optional[str] = None):
        if isinstance(expected_output, str) or expected_output is None:  # TODO
            if isinstance(expected_output, str):
                expected_output = cleandoc(expected_output)
            expected_output_temp = ""
            if expected_output:
                expected_output_temp += expected_output
            typed_parser_cls = typed_parser_registry.get(self.typed_parser, None)
            if typed_parser_cls is not None:  # Schema as expected output
                response_format = response_format_from_msgspec_struct(
                    self.generation_schema
                )
                schema = typed_parser_cls.schema_from_response_format(response_format)
                content = {"expected_outputs": schema}
                rendered = self._format_template(content, EXPECTED_OUTPUTS_TEMPLATE)
                expected_output_temp += rendered
            self.expected_output = Parameter(
                expected_output_temp or None, PromptSpec.EXPECTED_OUTPUT
            )
        else:
            raise TypeError(
                "`expected_output` requires a string or None "
                f"given `{type(expected_output)}`"
            )

    def _set_examples(
        self,
        examples: Optional[Union[str, List[Union[Example, Mapping[str, Any]]]]] = None,
    ):
        if isinstance(examples, (str, list)) or examples is None:
            if isinstance(examples, str):
                examples = cleandoc(examples)
            if isinstance(examples, list):
                typed_parser_cls = typed_parser_registry.get(self.typed_parser, None)
                collection = ExampleCollection(examples)
                if typed_parser_cls is not None:
                    serialize_func = typed_parser_cls.encode
                else:
                    serialize_func = msgspec_dumps
                examples = collection.get_formatted(serialize_func, serialize_func)
            self.examples = Parameter(examples, PromptSpec.EXAMPLES)
        else:
            raise TypeError(
                f"`examples` requires a List[Example] or None given `{type(examples)}`"
            )

    def _set_messages(self, messages: Optional[str] = None):
        if isinstance(messages, str) or messages is None:
            self.register_buffer("messages", messages)
        else:
            raise TypeError(
                f"`messages` requires a string or None given `{type(messages)}`"
            )

    def _set_config(self, config: Optional[Dict[str, Any]] = None):
        """Set agent configuration.

        Args:
            config:
                Dictionary with configuration options.
                Valid keys: "verbose", "return_messages", "tool_choice",
                "stream", "image_block_kwargs", "video_block_kwargs"

        Raises:
            TypeError:
                If config is not a dict or None.
            ValueError:
                If invalid keys are provided.
        """
        # Define valid keys for Agent
        valid_keys = {
            "verbose",
            "return_messages",
            "tool_choice",
            "stream",
            "image_block_kwargs",
            "video_block_kwargs",
            "reasoning_in_response",
            "max_tool_turns",
            "validate_inputs",
        }

        if config is None:
            self.register_buffer("config", {})
            return

        if not isinstance(config, dict):
            raise TypeError(f"`config` must be a dict or None, given `{type(config)}`")

        invalid_keys = set(config.keys()) - valid_keys
        if invalid_keys:
            raise ValueError(
                f"Invalid config keys: {invalid_keys}. Valid keys are: {valid_keys}"
            )

        self._validate_config_blocks(config)
        self._validate_config_limits(config)
        self._validate_config_flags(config)

        self.register_buffer("config", config.copy())

    def _validate_config_blocks(self, config: Dict[str, Any]):
        for key in ("image_block_kwargs", "video_block_kwargs"):
            if key in config and not isinstance(config[key], dict):
                raise TypeError(f"`{key}` must be a dict, given `{type(config[key])}`")

    def _validate_config_limits(self, config: Dict[str, Any]):
        if "max_tool_turns" in config:
            max_turns = config["max_tool_turns"]
            if not isinstance(max_turns, int) or max_turns < 1:
                raise ValueError(
                    f"`max_tool_turns` must be a positive integer, "
                    f"given `{config['max_tool_turns']}`"
                )

    def _validate_config_flags(self, config: Dict[str, Any]):
        if "validate_inputs" in config and not isinstance(
            config["validate_inputs"], bool
        ):
            raise TypeError(
                f"`validate_inputs` must be a bool, "
                f"given `{type(config['validate_inputs'])}`"
            )

    def _set_system_extra_message(self, system_extra_message: Optional[str] = None):
        if isinstance(system_extra_message, str) or system_extra_message is None:
            if isinstance(system_extra_message, str):
                system_extra_message = cleandoc(system_extra_message)
            self.register_buffer("system_extra_message", system_extra_message)
        else:
            raise TypeError(
                "`system_extra_message` requires a string or None "
                f"given `{type(system_extra_message)}`"
            )

    def _set_vars(self, vars: Optional[str] = None):
        if isinstance(vars, str) or vars is None:
            self.register_buffer("vars", vars)
        else:
            raise TypeError(f"`vars` requires a string or None given `{type(vars)}`")

    def _set_tool_filter(self, tool_filter: Optional[str] = None):
        if isinstance(tool_filter, str) or tool_filter is None:
            self.register_buffer("tool_filter", tool_filter)
        else:
            raise TypeError(
                f"`tool_filter` requires a string or None given `{type(tool_filter)}`"
            )

    def _set_message_fields(self, message_fields: Optional[Dict[str, Any]] = None):
        """Set message field mappings for Agent.

        Args:
            message_fields: Dictionary mapping field names to their values.
                Valid keys: "task", "task_multimodal", "messages",
                "task_context", "model_preference", "vars", "tool_filter"

        Raises:
            TypeError: If message_fields is not a dict or None
            ValueError: If invalid keys are provided
        """
        # Define valid keys for Agent class
        valid_keys = {
            "task",
            "task_multimodal",
            "messages",
            "task_context",
            "model_preference",
            "vars",
            "tool_filter",
        }

        if message_fields is None:
            # Set all fields to None
            self._set_task(None)
            self._set_task_multimodal(None)
            self._set_model_preference(None)
            self._set_task_context(None)
            self._set_messages(None)
            self._set_vars(None)
            self._set_tool_filter(None)
            return

        if not isinstance(message_fields, dict):
            raise TypeError(
                f"`message_fields` must be a dict or None, given "
                f"`{type(message_fields)}`"
            )

        # Validate keys
        invalid_keys = set(message_fields.keys()) - valid_keys
        if invalid_keys:
            raise ValueError(
                f"Invalid message_fields keys: {invalid_keys}. "
                f"Valid keys are: {valid_keys}"
            )

        # Set each field using its setter, defaulting to None if not provided
        self._set_task(message_fields.get("task"))
        self._set_task_multimodal(message_fields.get("task_multimodal"))
        self._set_model_preference(message_fields.get("model_preference"))
        self._set_task_context(message_fields.get("task_context"))
        self._set_messages(message_fields.get("messages"))
        self._set_vars(message_fields.get("vars"))
        self._set_tool_filter(message_fields.get("tool_filter"))

    def _get_tool_filter_from_message(self, message: Message) -> Optional[ToolFilter]:
        """Read runtime tool filtering rules from a mapped Message field."""
        if isinstance(message, dotdict) and isinstance(self.tool_filter, str):
            return cast(Optional[ToolFilter], message.get(self.tool_filter))
        return None

    def _set_typed_parser(self, typed_parser: Optional[str] = None):
        if isinstance(typed_parser, str) or typed_parser is None:
            if (
                isinstance(typed_parser, str)
                and typed_parser not in typed_parser_registry
            ):
                raise ValueError(
                    f"`typed_parser` supports only `{typed_parser_registry.keys()}`"
                    f" given `{typed_parser}`"
                )
            self.register_buffer("typed_parser", typed_parser)
        else:
            raise TypeError(
                f"`typed_parser` requires a str given `{type(typed_parser)}`"
            )

    def _set_signature(
        self,
        *,
        signature: Optional[Union[str, Signature]] = None,
        examples: Optional[List[Example]] = None,
        generation_schema: Optional[msgspec.Struct] = None,
        instructions: Optional[str] = None,
        system_message: Optional[str] = None,
        typed_parser: Optional[str] = None,
    ):
        if signature is not None:
            typed_parser_cls = typed_parser_registry.get(typed_parser, None)

            examples = examples or []
            output_descriptions = None
            signature_instructions = None

            if isinstance(signature, str):
                input_str_signature, output_str_signature = signature.split("->")
                inputs_info = StructFactory._parse_annotations(input_str_signature)
                outputs_info = StructFactory._parse_annotations(output_str_signature)
            elif issubclass(signature, Signature):
                output_str_signature = signature.get_str_signature().split("->")[-1]
                inputs_info = signature.get_inputs_info()
                outputs_info = signature.get_outputs_info()
                output_descriptions = signature.get_output_descriptions()
                signature_instructions = signature.get_instructions()
                signature_examples = SignatureFactory.get_examples_from_signature(
                    signature
                )
                if signature_examples:
                    examples.extend(signature_examples)
            else:
                raise TypeError(
                    "`signature` requires a string, `Signature` or None "
                    f"given `{type(signature)}`"
                )

            # Validate signature input names don't conflict with reserved kwargs
            input_names = {field.name for field in inputs_info}
            conflicts = input_names & _RESERVED_KWARGS
            if conflicts:
                raise ValueError(
                    f"Signature input names {conflicts} conflict with reserved "
                    f"Agent kwargs. Reserved names: {_RESERVED_KWARGS}. "
                    f"Rename these inputs to avoid conflicts."
                )

            # typed_parser
            self._set_typed_parser(typed_parser)

            # task template - add to templates dict, overriding if present
            task_template = SignatureFactory.get_task_template_from_signature(
                inputs_info
            )
            self.templates["task"] = task_template

            # instructions
            self._set_instructions(instructions or signature_instructions)

            # generation schema
            signature_output_struct = StructFactory.from_signature(
                output_str_signature, "Outputs", output_descriptions
            )
            fused_output_struct = None
            if generation_schema is not None:
                signature_as_type = cast(Type[msgspec.Struct], signature_output_struct)
                if is_optional_field(generation_schema, "final_answer"):
                    signature_as_type = Optional[signature_output_struct]  # type: ignore

                # Merge parent annotations with new final_answer annotation
                merged_annotations = {
                    **generation_schema.__annotations__,
                    "final_answer": signature_as_type,
                }

                fused_output_struct = type(
                    "Output",
                    (generation_schema,),
                    {"__annotations__": merged_annotations},
                )
            self._set_generation_schema(fused_output_struct or signature_output_struct)

            # system message
            self._set_system_message(system_message)

            # expected output
            expected_output = SignatureFactory.get_expected_output_from_signature(
                inputs_info, outputs_info, typed_parser_cls
            )
            self._set_expected_output(expected_output)

            # examples
            self._set_examples(examples)

            # Generate and set annotations from signature inputs
            generated_annotations = generate_annotations_from_signature(
                inputs_info, signature
            )
            self.set_annotations(generated_annotations)

            input_schema = SignatureFactory.get_input_schema_from_signature(
                inputs_info, signature
            )
            self._input_schema = input_schema
            if input_schema is not None:
                self._input_encoder = msgspec.json.Encoder()
                self._input_decoder = msgspec.json.Decoder(input_schema)
            else:
                self._input_encoder = None
                self._input_decoder = None

    def _get_validation_inputs(
        self,
        message: Optional[Union[str, Message, Mapping[str, Any]]],
        task: Any,
        vars: Mapping[str, Any],
    ) -> Optional[Mapping[str, Any]]:
        if not self.config.get("validate_inputs", False):
            return None
        if getattr(self, "_input_decoder", None) is None:
            return None

        if task is _UNSET:
            if isinstance(message, dotdict):
                task = self._extract_message_values(self.task, message)
            else:
                task = message

        if isinstance(task, Mapping):
            validation_inputs = dict(task)
            validation_inputs.update(vars)
            return validation_inputs
        if task is None and vars:
            return vars
        schema_fields = getattr(self._input_schema, "__struct_fields__", ())
        if len(schema_fields) == 1 and task is not None:
            validation_inputs = {schema_fields[0]: task}
            validation_inputs.update(vars)
            return validation_inputs
        return None

    def _validate_inputs(self, inputs: Mapping[str, Any]) -> None:
        if not self.config.get("validate_inputs", False):
            return
        decoder = getattr(self, "_input_decoder", None)
        if decoder is None:
            return

        schema_fields = getattr(self._input_schema, "__struct_fields__", ())
        payload = {field: inputs[field] for field in schema_fields if field in inputs}

        try:
            decoder.decode(self._input_encoder.encode(payload))
        except (msgspec.ValidationError, msgspec.EncodeError, TypeError) as exc:
            raise ValueError(
                f"[{self.name}] Input validation failed: {exc}. "
                f"Expected schema: {self._input_schema.__struct_fields__}"
            ) from exc

    # --- System Prompt ---

    def get_system_prompt(
        self,
        vars: Optional[Mapping[str, Any]] = None,
        tool_catalog: Optional[ToolCatalogView] = None,
        *,
        _apply_hooks: bool = True,
    ) -> str:
        """Render the system prompt using the Jinja template.
        Returns an empty string if no segments are provided.
        """
        template_inputs = dotdict(
            system_message=self.system_message.data,
            instructions=self.instructions.data,
            expected_output=self.expected_output.data,
            examples=self.examples.data,
            system_extra_message=self.system_extra_message,
        )

        system_prompt = self._format_template(
            template_inputs, self.system_prompt_template
        )

        if vars:  # Runtime inputs to system template
            system_prompt = self._format_template(vars, system_prompt)
        if not _apply_hooks:
            return system_prompt
        if tool_catalog is None:
            scope = get_execution_context()["scope"]
            tool_catalog = self.tool_library.get_tool_catalog_view(
                thread_id=scope.thread_id or f"{self.name}:system_prompt"
            )
        ctx = self._run_lifecycle_hooks(
            "transform_system_prompt",
            ModelContext(
                prompt=system_prompt,
                scope=get_execution_context()["scope"],
                vars=vars or {},
                tool_catalog=tool_catalog,
            ),
        )
        ctx = _require_lifecycle_payload("transform_system_prompt", ctx, ModelContext)
        return ctx.prompt

    @property
    def system_prompt_template(self) -> str:
        """Get the system prompt template.

        Returns the custom template if provided in templates dict,
        otherwise returns the default SYSTEM_PROMPT_TEMPLATE.
        """
        return self.templates.get("system_prompt", SYSTEM_PROMPT_TEMPLATE)
