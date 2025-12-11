from functools import wraps
from types import FunctionType, MethodType
from typing import Callable, Dict, List, Optional, Union

from msgflux.dotdict import dotdict


def tool_config(
    *,
    return_direct: Optional[bool] = False,
    call_as_response: Optional[bool] = False,
    background: Optional[bool] = False,
    inject_model_state: Optional[bool] = False,
    inject_vars: Optional[Union[bool, List[str]]] = False,
    handoff: Optional[bool] = False,
    name_override: Optional[str] = None,
) -> Callable:
    """Decorator to inject meta-properties into functions, classes, or instances.

    This decorator adds metadata properties to control tool behavior such as whether
    results are returned directly or passed for further handling, and optionally
    override the tool's registered name.

    Behavior depends on what is being decorated:
    - **Functions**: Wraps the function and injects properties into the wrapper
    - **Classes**: Modifies the class's __init__ to inject properties into all
      future instances. This allows classes with required parameters to be decorated.
    - **Instances**: Directly injects properties into the instance

    Args:
        return_direct:
            If True, the tool will return its output directly without additional
            processing.
        call_as_response:
            If True, returns the tool call as its result. This property requires
            `return_direct = True` and will automatically change it to True if it
            is passed as false.
        background:
            If True, the tool will be executed in the background and a message
            that the task has been scheduled will be the response to the model.
        inject_model_state:
            If true, the tool automatically sets `inject_model_state` and
            `return_direct` to `True`. Additionally, the tool will be
            renamed to `transfer_to_<name>`.
            Any input parameters for this tool will be removed. The tool will **only**
            receive `model_state` as a parameter.
        inject_vars:
            Indicates if the tool should receive vars. If True, the tool receives all
            vars as a named argument `vars`. If a list of vars is passed, only those
            vars will be passed.
        handoff:
            If True, indicates that this function will receive the `model_state`
            from the Agent.
        name_override:
            A custom name to override the default tool name derived from the function
            or class. If not provided, the original name is used.

    Returns:
        A decorator that modifies the target by injecting the specified properties.
        - For functions: returns a wrapped function with properties
        - For classes: returns the modified class (all instances will have properties)
        - For instances: returns the instance with injected properties

    Raises:
        ValueError:
           `background=True` is not compatible with `return_direct=True` and
           `call_as_response=True`.
        ValueError:
           `inject_vars=True` is not compatible with `call_as_response=True`.

    Examples:
        Decorating a function:
            >>> @tool_config(return_direct=True)
            ... def my_tool(query: str) -> str:
            ...     return f"Result: {query}"
            >>> my_tool.tool_config.return_direct
            True

        Decorating a class (all instances will have tool_config):
            >>> @tool_config(return_direct=True)
            ... class SentimentClassifier(nn.Agent):
            ...     def __init__(self, model):
            ...         super().__init__(model=model)
            >>> classifier = SentimentClassifier(model=my_model)
            >>> classifier.tool_config.return_direct
            True

        Decorating an instance:
            >>> classifier = SentimentClassifier(model=my_model)
            >>> classifier = tool_config(return_direct=True)(classifier)
            >>> classifier.tool_config.return_direct
            True
    """

    def decorator(f):
        _return_direct = return_direct  # Local copy
        _inject_model_state = inject_model_state  # Local copy

        if call_as_response is True and _return_direct is False:
            _return_direct = True

        if handoff:
            _return_direct = True
            _inject_model_state = True

        if background and (_return_direct or call_as_response):
            raise ValueError(
                "`background=True` is not compatible with `return_direct=True`"
                " and `call_as_response=True`."
            )

        if inject_vars is not False and call_as_response is True:
            raise ValueError(
                "`inject_vars` is not compatible with `call_as_response=True`"
            )

        tool_config = {
            "tool_config": dotdict(
                {
                    "background": background,
                    "call_as_response": call_as_response,
                    "handoff": handoff,
                    "inject_model_state": _inject_model_state,
                    "inject_vars": inject_vars,
                    "return_direct": _return_direct,
                    "name_overridden": name_override,
                }
            )
        }
        if isinstance(f, (FunctionType, MethodType)):
            return decorate_function(f, tool_config)
        if isinstance(f, type):  # Is a class (not an instance)
            # Create a new subclass with tool_config injected
            return decorate_class(f, tool_config)
        # Is an instance
        return decorate_instance(f, tool_config)

    return decorator


def decorate_function(
    func: Union[FunctionType, MethodType],
    tool_config: Dict[str, Union[bool, str]],
) -> Union[FunctionType, MethodType]:
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper.__dict__.update(tool_config)
    return wrapper


def decorate_class(cls: type, tool_config: Dict[str, Union[bool, str]]) -> type:
    """Decorates a class by creating a subclass with tool_config injection.

    Uses type() to dynamically create a new class that inherits from the original,
    ensuring that only instances of the decorated class have tool_config.

    Args:
        cls: The class to decorate
        tool_config: Dictionary containing tool configuration properties

    Returns:
        A new subclass with tool_config injection in __init__
    """

    # Define __init__ for the new class
    def __init__(self, *args, **kwargs):  # noqa: N807
        # Call parent __init__
        cls.__init__(self, *args, **kwargs)
        # Inject a COPY of tool_config into the instance to avoid sharing
        for key, value in tool_config.items():
            if key == "tool_config" and hasattr(value, "to_dict"):
                # Create a new dotdict instance with the same data
                self.__dict__[key] = dotdict(value.to_dict())
            else:
                self.__dict__[key] = value

    # Create new class using type()
    new_class = type(
        cls.__name__,  # Same name as original
        (cls,),  # Inherit from original class
        {
            "__init__": __init__,
            "__module__": cls.__module__,
            "__qualname__": cls.__qualname__,
            "__doc__": cls.__doc__,
        },
    )

    return new_class


def decorate_instance(
    instance: Callable, tool_config: Dict[str, Union[bool, str]]
) -> Callable:
    instance.__dict__.update(tool_config)
    return instance
