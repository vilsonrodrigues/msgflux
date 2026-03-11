from typing import Any, Dict, Optional

import msgspec

# Sentinel for field deletion in :meth:`dotdict.apply`.
DELETE = type(
    "DELETE",
    (),
    {"__repr__": lambda self: "<DELETE>", "__bool__": lambda self: False},  # noqa: ARG005
)()


class dotdict(dict):  # noqa: N801
    """A dictionary with dot access and nested path support.

    dotdict allows you to access and modify values as attributes (e.g., `obj.key`)
    and also allows reading and writing nested paths using strings with dot separators
    (e.g., `obj.get("user.profile.name")`).

    Main features:
    - Dot access (`obj.key`)
    - Traditional square bracket access (`obj['key']`)
    - Nested reading via `.get("a.b.c")`
    - Nested writing via `.set("a.b.c", value)`
    - Conversion to standard dict with `.to_dict()`
    - Support for Msgspec serialization (`__json__`)
    - Support for lists with path indices (e.g., `"items.0.name"`)
    - Optional immutability (`frozen=True`)
    - Hidden keys support (`hidden_keys=["key1", "key2"]`)

    Hidden Keys:
        The `hidden_keys` parameter marks keys as invisible to enumeration and
        discovery. Hidden keys are excluded from `keys()`, `values()`, `items()`,
        `__iter__`, `__contains__` (`in` operator), `to_dict()`, `to_json()`,
        `__repr__`, and `__str__`.

        Hidden keys are still accessible via direct access: dot notation
        (`obj.api_key`), bracket notation (`obj['api_key']`), and `get()` paths
        (`obj.get('api_key')`). This makes them suitable for sensitive data like
        API keys or passwords that should not leak into logs, serialized output,
        or iteration, while remaining readable when explicitly requested.

    Example:
            >>> d = dotdict(
            ...     {"api_key": "secret", "username": "john"},
            ...     hidden_keys=["api_key"]
            ... )
            >>> "api_key" in d        # False
            >>> list(d.keys())        # ["username"]
            >>> d.to_dict()           # {"username": "john"}
            >>> d.api_key             # "secret"
            >>> d["api_key"]          # "secret"
            >>> d.get("api_key")      # "secret"
    """

    def __init__(
        self,
        initial_data: Optional[Dict[str, Any]] = None,
        *,
        frozen: Optional[bool] = False,
        hidden_keys: Optional[list[str]] = None,
        backend=None,
        backend_prefix: Optional[str] = None,
        **kwargs,
    ):
        """Initializes an instance of dotdict.

        Args:
            initial_data:
                Base dictionary to initialize data.
            frozen:
                If True, prevents changes after creation.
            hidden_keys:
                List of keys that should not be returned by get() method.
                These keys will return None (or default) when accessed via get().
            backend:
                Optional persistent backend (e.g. ``diskcache.Cache``). Must support
                ``__getitem__``, ``__setitem__``, ``__delitem__``, and ``__iter__``.
                Every top-level key write is persisted automatically. On creation,
                existing keys in the backend are loaded first; ``initial_data`` then
                overrides them.

                .. warning::
                    Nested mutations (``d.user.name = "x"`` or ``d.set("a.b", v)``)
                    are **not** automatically persisted. Reassign the top-level key to
                    trigger persistence: ``d.user = {**d.user, "name": "x"}``.

            backend_prefix:
                Optional namespace prefix for backend keys (e.g. ``"run_42"`` stores
                keys as ``"run_42.key"``). Useful when multiple dotdicts share the
                same backend instance.
            **kwargs:
                Additional key=value pairs merged with ``initial_data``.

        ::: example:
            d = dotdict({"user": {"name": "Maria"}}, frozen=False)
            print(d.user.name)
            >> Maria

            # With persistent backend
            import diskcache
            cache = diskcache.Cache("./state")
            d = dotdict(backend=cache, backend_prefix="run_1")
            d.x = 1   # persisted immediately
            d.y = 2   # persisted immediately
        """
        initial_data = initial_data or {}
        self._backend = backend
        self._backend_prefix = backend_prefix
        self._frozen = False  # defer frozen enforcement to allow loading
        self._hidden_keys = set(hidden_keys or [])
        super().__init__()

        if backend is not None:
            prefix = backend_prefix
            for bkey in backend:
                if prefix:
                    if not bkey.startswith(f"{prefix}."):
                        continue
                    key = bkey[len(f"{prefix}."):]
                else:
                    key = bkey
                if "." not in key:
                    super().__setitem__(key, self._wrap(backend[bkey]))

        for key, value in {**initial_data, **kwargs}.items():
            self[key] = value

        self._frozen = frozen

    def __iter__(self):
        hidden = getattr(self, "_hidden_keys", set())
        for key in super().__iter__():
            if key not in hidden:
                yield key

    def __contains__(self, key):
        hidden = getattr(self, "_hidden_keys", set())
        if key in hidden:
            return False
        return super().__contains__(key)

    def keys(self):
        hidden = getattr(self, "_hidden_keys", set())
        return [k for k in super().keys() if k not in hidden]

    def values(self):
        hidden = getattr(self, "_hidden_keys", set())
        return [v for k, v in super().items() if k not in hidden]

    def items(self):
        hidden = getattr(self, "_hidden_keys", set())
        return [(k, v) for k, v in super().items() if k not in hidden]

    def __getattr__(self, attr: str):
        try:
            return self[attr]
        except KeyError as e:
            raise AttributeError(f"`dotdict` object has no attribute '{attr}'") from e

    def __setattr__(self, key: str, value: Any):
        if key.startswith("_"):
            super().__setattr__(key, value)
        elif hasattr(self, "_frozen") and self._frozen:
            raise AttributeError("Cannot modify frozen dotdict")
        else:
            self[key] = value

    def __setitem__(self, key: str, value: Any):
        if getattr(self, "_frozen", False):
            raise AttributeError("Cannot modify frozen dotdict")
        wrapped = self._wrap(value)
        super().__setitem__(key, wrapped)
        backend = getattr(self, "_backend", None)
        if backend is not None:
            backend[self._backend_key(key)] = self._serialize(wrapped)

    def __delitem__(self, key: str):
        if getattr(self, "_frozen", False):
            raise AttributeError("Cannot delete from frozen dotdict")
        super().__delitem__(key)
        backend = getattr(self, "_backend", None)
        if backend is not None:
            try:
                del backend[self._backend_key(key)]
            except KeyError:
                pass

    def __delattr__(self, key: str):
        if getattr(self, "_frozen", False):
            raise AttributeError("Cannot delete from frozen dotdict")
        try:
            del self[key]
        except KeyError as e:
            raise AttributeError(f"`dotdict` object has no attribute `{key}`") from e

    def _backend_key(self, key: str) -> str:
        prefix = getattr(self, "_backend_prefix", None)
        return f"{prefix}.{key}" if prefix else key

    def _serialize(self, value: Any) -> Any:
        if isinstance(value, dotdict):
            return value.to_dict()
        elif isinstance(value, list):
            return [self._serialize(item) for item in value]
        return value

    def _wrap(self, value: Any):
        if isinstance(value, dict):
            return dotdict(
                value,
                frozen=getattr(self, "_frozen", False),
                hidden_keys=list(getattr(self, "_hidden_keys", set())),
            )
        elif isinstance(value, list):
            return [self._wrap(item) for item in value]
        return value

    def get(self, path: str, default: Any = None) -> Any:
        """Access nested values via dot path.

        !!! example:
            get('user.profile.age').
        """
        keys = path.split(".")
        current = self
        try:
            for key in keys:
                if isinstance(current, list):
                    key = int(key)  # noqa: PLW2901
                current = current[key]
            return current
        except (KeyError, IndexError, ValueError, TypeError):
            return default

    def set(self, path: str, value: Any):
        """Set nested value via dot path.

        !!! example

            set('user.profile.age', 31).
        """
        if self._frozen:
            raise AttributeError("Cannot modify frozen dotdict")

        keys = path.split(".")
        current = self
        for i, key in enumerate(keys):
            if isinstance(current, list):
                key = int(key)  # noqa: PLW2901

            if i == len(keys) - 1:
                if isinstance(current, list):
                    key_i = int(key)
                    current[key_i] = self._wrap(value)
                else:
                    current[key] = self._wrap(value)
                return

            if isinstance(current, list):
                key_i = int(key)
                if key_i >= len(current) or not isinstance(
                    current[key_i], (dict, dotdict)
                ):
                    current[key_i] = dotdict()
                current = current[key_i]
            else:
                if key not in current or not isinstance(
                    current[key], (dict, dotdict, list)
                ):
                    current[key] = dotdict()
                current = current[key]

    def apply(self, update):
        """Apply a dict of updates to this message.

        Modules return dicts with the changes they want to apply.
        ``None`` return is a no-op.  Keys whose value is the
        :data:`DELETE` sentinel are removed.
        """
        if update is None:
            return self
        if not isinstance(update, dict):
            raise TypeError(
                f"`update` must be a dict or None, got {type(update).__name__}"
            )
        for key, value in update.items():
            if value is DELETE:
                self.pop(key, None)
            elif (
                isinstance(value, dict)
                and key in self
                and isinstance(self[key], dotdict)
            ):
                self[key].update(value)
            else:
                self[key] = value
        return self

    def update(self, *args, **kwargs):
        """Extends dict.update to support nested keys while maintaining DotDict."""
        if self._frozen:
            raise AttributeError("Cannot modify frozen DotDict")

        # Collects all key-value pairs
        other = {}
        if args:
            if len(args) > 1:
                raise TypeError(
                    f"update expected at most 1 arguments, given `{len(args)}`"
                )
            other.update(args[0])
        other.update(kwargs)

        for key, value in other.items():
            # Nested key with dot?
            if isinstance(key, str) and "." in key:
                self.set(key, value)

            # Value is dict and there is already a dotdict on that key?
            # Merge recursively
            elif (
                isinstance(value, dict)
                and key in self
                and isinstance(self[key], dotdict)
            ):
                self[key].update(value)

            # General case: normal assignment (call __setitem__ e wrap)
            else:
                self[key] = value

    def to_dict(self):
        def unwrap(value):
            if isinstance(value, dotdict):
                return {k: unwrap(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [unwrap(item) for item in value]
            return value

        return unwrap(self)

    def to_json(self) -> bytes:
        """Returns a encoded-JSON."""
        return msgspec.json.encode(self.to_dict())

    def __repr__(self):
        d = self.to_dict()
        attrs_str = "\n".join(f"   '{k}': {v!r}" for k, v in d.items())
        return f"{self.__class__.__name__}({{\n{attrs_str}\n}})"

    def __str__(self):
        return str(self.to_dict())
