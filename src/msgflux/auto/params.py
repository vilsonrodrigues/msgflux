"""AutoParams metaclass for automatic parameter management."""

from types import FunctionType, MethodType
from typing import Any, Dict


class AutoParams(type):
    """Metaclass that captures class attributes and uses them as default parameters.

    This metaclass automatically captures all non-callable, non-special attributes
    defined in a class and makes them available as default parameter values when
    instantiating the class.

    The captured parameters are stored in the class's _auto_params attribute and
    are automatically merged with any parameters passed during instantiation.

    Basic Example:
        >>> from msgflux.auto import AutoParams
        >>>
        >>> class Model(metaclass=AutoParams):
        ...     def __init__(self, learning_rate, batch_size, epochs):
        ...         self.learning_rate = learning_rate
        ...         self.batch_size = batch_size
        ...         self.epochs = epochs
        ...
        >>> class MyModel(Model):
        ...     learning_rate = 0.001
        ...     batch_size = 32
        ...     epochs = 100
        ...
        >>> # Uses default values from class attributes
        >>> m = MyModel()
        >>> print(m.learning_rate)  # 0.001
        >>>
        >>> # Partial override
        >>> m2 = MyModel(learning_rate=0.01, batch_size=64)
        >>> print(m2.learning_rate)  # 0.01
        >>> print(m2.epochs)  # 100 (still uses default)

    Docstring as Parameter:
        Classes can configure AutoParams to use their docstring as a parameter value
        by setting the _autoparams_use_docstring_for class attribute.

        >>> class Agent(metaclass=AutoParams):
        ...     _autoparams_use_docstring_for = "description"
        ...     def __init__(self, name, description=None):
        ...         self.name = name
        ...         self.description = description
        ...
        >>> class MyAgent(Agent):
        ...     '''An agent that helps with coding tasks'''
        ...
        >>> agent = MyAgent()
        >>> print(agent.description)  # "An agent that helps with coding tasks"
        >>>
        >>> # Explicit attribute takes precedence over docstring
        >>> class MyAgent2(Agent):
        ...     '''This is ignored'''
        ...     description = "Explicit description"
        ...
        >>> agent2 = MyAgent2()
        >>> print(agent2.description)  # "Explicit description"

    Class Name as Parameter:
        Classes can configure AutoParams to use the class name as a parameter value
        by setting the _autoparams_use_classname_for class attribute.

        >>> class Agent(metaclass=AutoParams):
        ...     _autoparams_use_classname_for = "name"
        ...     def __init__(self, name):
        ...         self.name = name
        ...
        >>> class SuperAgent(Agent):
        ...     pass
        ...
        >>> agent = SuperAgent()
        >>> print(agent.name)  # "SuperAgent"
        >>>
        >>> # Explicit attribute takes precedence over class name
        >>> class CustomAgent(Agent):
        ...     name = "my_custom_agent"
        ...
        >>> agent2 = CustomAgent()
        >>> print(agent2.name)  # "my_custom_agent"
    """

    def __new__(mcls, name: str, bases: tuple, namespace: Dict[str, Any]):
        """Create a new class with auto parameter management.

        Args:
            name: Name of the class being created
            bases: Base classes
            namespace: Class namespace dictionary

        Returns:
            The newly created class with _auto_params attribute
        """
        # Collect auto params from base classes
        inherited_params = {}
        docstring_param_name = None
        classname_param_name = None
        for base in bases:
            if hasattr(base, "_auto_params"):
                inherited_params.update(base._auto_params)
            # Check if base class wants to use docstring as a parameter
            if hasattr(base, "_autoparams_use_docstring_for"):
                docstring_param_name = base._autoparams_use_docstring_for
            # Check if base class wants to use class name as a parameter
            if hasattr(base, "_autoparams_use_classname_for"):
                classname_param_name = base._autoparams_use_classname_for

        # Capture all non-callable and non-special attributes from this class
        # Exclude properties, classmethods, staticmethods, and functions/methods
        # Allow classes (types) and instances (even if callable) to be collected
        class_params = {
            k: v
            for k, v in namespace.items()
            if not k.startswith("_")
            and not isinstance(
                v, (FunctionType, MethodType, classmethod, staticmethod, property)
            )
        }

        # Handle class name as parameter if configured
        if classname_param_name and classname_param_name not in class_params:
            # Use the class name being created
            class_params[classname_param_name] = name

        # Handle docstring as parameter if configured
        if docstring_param_name and "__doc__" in namespace and namespace["__doc__"]:
            # Only use docstring if the parameter isn't already defined
            if docstring_param_name not in class_params:
                class_params[docstring_param_name] = namespace["__doc__"]

        # Merge inherited params with class params (class params take precedence)
        all_params = {**inherited_params, **class_params}

        # Store the merged params in the namespace
        namespace["_auto_params"] = all_params

        return super().__new__(mcls, name, bases, namespace)

    def __call__(cls, *args, **kwargs):
        """Instantiate the class with auto parameters.

        Merges the class's _auto_params with the provided kwargs, with kwargs
        taking precedence. This allows for partial or complete override of
        default parameters.

        Args:
            *args: Positional arguments to pass to __init__
            **kwargs: Keyword arguments to pass to __init__

        Returns:
            Instance of the class
        """
        # Start with auto params
        all_kwargs = dict(cls._auto_params) if hasattr(cls, "_auto_params") else {}

        # Override with user-provided kwargs
        all_kwargs.update(kwargs)

        # Instantiate normally
        return super().__call__(*args, **all_kwargs)


__all__ = ["AutoParams"]
