# dotdict

`dotdict` is a dictionary with **dot notation access** and **nested path support**. It extends Python's built-in `dict`, so anything that works with a regular dict works here too.

It is the foundation of [`Message`](nn/message.md) and can be used standalone wherever flexible, deeply-nested data structures are needed.

???+ example

    ```python
    from msgflux import dotdict

    d = dotdict({"user": {"name": "Maria", "age": 30}})

    print(d.user.name)   # "Maria"
    print(d.user.age)    # 30
    ```

---

## Creating a dotdict

### From a dict

???+ example

    ```python
    from msgflux import dotdict

    d = dotdict({"key": "value", "nested": {"x": 1}})
    ```

### From keyword arguments

???+ example

    ```python
    from msgflux import dotdict

    d = dotdict(name="Alice", score=100)
    ```

### Combined

???+ example

    ```python
    from msgflux import dotdict

    d = dotdict({"role": "admin"}, name="Alice", score=100)
    ```

### Empty

???+ example

    ```python
    from msgflux import dotdict

    d = dotdict()
    d.name = "Bob"
    ```

---

## Reading Values

### Dot access

???+ example

    ```python
    from msgflux import dotdict

    d = dotdict({"user": {"name": "Clark"}})

    print(d.user.name)  # "Clark"
    ```

### Bracket access

???+ example

    ```python
    from msgflux import dotdict

    d = dotdict({"user": {"name": "Clark"}})

    print(d["user"]["name"])  # "Clark"
    ```

### Nested path with `get()`

Use `get(path, default=None)` to traverse nested keys via a dot-separated string:

???+ example

    ```python
    from msgflux import dotdict

    d = dotdict({"user": {"profile": {"city": "Gotham"}}})

    print(d.get("user.profile.city"))        # "Gotham"
    print(d.get("user.profile.zip", "N/A"))  # "N/A" (key doesn't exist)
    ```

`get()` never raises — it returns `default` when any key in the path is missing.

---

## Writing Values

### Dot assignment

???+ example

    ```python
    from msgflux import dotdict

    d = dotdict()
    d.name = "Diana"
    d.score = 99
    ```

### Bracket assignment

???+ example

    ```python
    from msgflux import dotdict

    d = dotdict()
    d["name"] = "Diana"
    ```

### Nested path with `set()`

Use `set(path, value)` to write deeply nested values. Intermediate keys are created automatically:

???+ example

    ```python
    from msgflux import dotdict

    d = dotdict()

    d.set("user.profile.city", "Metropolis")
    d.set("user.profile.age", 28)

    print(d.user.profile.city)  # "Metropolis"
    print(d.user.profile.age)   # 28
    ```

---

## Nested Paths

Both `get()` and `set()` accept dot-separated strings to traverse any depth:

???+ example

    ```python
    from msgflux import dotdict

    d = dotdict()

    d.set("a.b.c.d", "deep value")
    print(d.get("a.b.c.d"))  # "deep value"
    ```

### List index access

Use integer segments in the path to index into lists:

???+ example

    ```python
    from msgflux import dotdict

    d = dotdict()
    d.set("items", [{"name": "Alpha"}, {"name": "Beta"}])

    print(d.get("items.0.name"))  # "Alpha"
    print(d.get("items.1.name"))  # "Beta"
    ```

`set()` also supports writing into existing list positions:

???+ example

    ```python
    from msgflux import dotdict

    d = dotdict()
    d.set("items", [{"name": "Alpha"}, {"name": "Beta"}])

    d.set("items.0.name", "Updated")
    print(d.get("items.0.name"))  # "Updated"
    ```

---

## Auto-wrapping

Any `dict` assigned to a `dotdict` — whether at creation, via `set()`, or via attribute assignment — is automatically converted to a `dotdict`, so dot access always works:

???+ example

    ```python
    from msgflux import dotdict

    d = dotdict()
    d.config = {"debug": True, "timeout": 30}

    print(d.config.debug)    # True
    print(d.config.timeout)  # 30
    ```

Lists of dicts are also wrapped recursively:

???+ example

    ```python
    from msgflux import dotdict

    d = dotdict()
    d.users = [{"name": "A"}, {"name": "B"}]

    print(d.users[0].name)  # "A"
    ```

---

## `update()`

`update()` extends `dict.update` with two extras:

**1. Dotted keys are written as nested paths:**

???+ example

    ```python
    from msgflux import dotdict

    d = dotdict()
    d.update({"user.name": "Bruce", "user.age": 35})

    print(d.user.name)  # "Bruce"
    ```

**2. Dict values are merged recursively when the key already holds a `dotdict`:**

???+ example

    ```python
    from msgflux import dotdict

    d = dotdict({"config": {"debug": False, "timeout": 30}})
    d.update({"config": {"debug": True}})

    print(d.config.debug)    # True   (updated)
    print(d.config.timeout)  # 30     (preserved)
    ```

Standard positional argument and keyword argument forms are both supported:

???+ example

    ```python
    from msgflux import dotdict

    d = dotdict()
    d.update({"key": "value"})
    d.update(key="other")
    ```

---

## Serialization

### `to_dict()`

Converts the `dotdict` (and all nested `dotdict` values) back to a plain Python `dict`:

???+ example

    ```python
    from msgflux import dotdict

    d = dotdict({"user": {"name": "Lois"}})

    plain = d.to_dict()
    print(type(plain))           # <class 'dict'>
    print(type(plain["user"]))   # <class 'dict'>
    ```

### `to_json()`

Returns a JSON-encoded `bytes` object, powered by `msgspec`:

???+ example

    ```python
    from msgflux import dotdict

    d = dotdict({"score": 42})

    print(d.to_json())  # b'{"score":42}'
    ```

---

## Immutability

Pass `frozen=True` to create a read-only `dotdict`. Any attempt to write raises `AttributeError`:

???+ example

    ```python
    from msgflux import dotdict

    d = dotdict({"key": "value"}, frozen=True)

    d.key = "new"        # raises AttributeError: Cannot modify frozen dotdict
    d["key"] = "new"     # raises AttributeError
    d.set("key", "new")  # raises AttributeError
    del d.key            # raises AttributeError
    ```

Nested dicts inherit the `frozen` flag automatically:

???+ example

    ```python
    from msgflux import dotdict

    d = dotdict({"user": {"name": "Bruce"}}, frozen=True)

    d.user.name = "Clark"  # raises AttributeError
    ```

---

## Hidden Keys

`hidden_keys` marks keys as invisible to enumeration and discovery. They won't appear in iteration, serialization, or string representations — but can always be accessed directly if you know they exist.

???+ example

    ```python
    from msgflux import dotdict

    d = dotdict(
        {"api_key": "sk-secret", "username": "john"},
        hidden_keys=["api_key"]
    )

    # Enumeration — api_key is invisible
    print("api_key" in d)      # False
    print(list(d.keys()))      # ["username"]
    print(list(d.values()))    # ["john"]
    print(list(d.items()))     # [("username", "john")]
    for k in d:
        print(k)               # "username"
    print(d.to_dict())         # {"username": "john"}
    print(d.to_json())         # b'{"username":"john"}'
    print(d)                   # {'username': 'john'}

    # Direct access — api_key is reachable
    print(d.api_key)              # "sk-secret"
    print(d["api_key"])           # "sk-secret"
    print(d.get("api_key"))       # "sk-secret"
    print(d.get("api_key", "x"))  # "sk-secret"
    ```

---

## Persistent Backend

Pass a `backend` to persist every top-level key write to disk automatically. Any object that supports `__getitem__`, `__setitem__`, `__delitem__`, and `__iter__` works — [`diskcache.Cache`](https://grantjenks.com/docs/diskcache/) is the recommended choice.

On creation, existing keys in the backend are loaded first. `initial_data` and `**kwargs` are applied on top and override them.

???+ example "Basic usage"

    ```python
    import diskcache
    from msgflux import dotdict

    cache = diskcache.Cache("./state")

    d = dotdict(backend=cache)
    d.x = 42   # written to disk immediately
    d.y = 99   # written to disk immediately
    ```

### Recovery after failure

Because writes hit the backend on every assignment, a new instance picks up exactly where the previous one left off:

???+ example

    ```python
    import diskcache
    from msgflux import dotdict

    cache = diskcache.Cache("./state")

    # --- first run ---
    d = dotdict(backend=cache)
    d.step = 3
    d.results = [0.91, 0.87, 0.94]
    # process crashes here...

    # --- after restart ---
    d = dotdict(backend=cache)
    print(d.step)     # 3
    print(d.results)  # [0.91, 0.87, 0.94]
    ```

### Concurrent writes from multiple instances

Each top-level key is stored as an independent entry in the backend, so two instances writing **different keys** never overwrite each other:

???+ example

    ```python
    import diskcache
    from msgflux import dotdict

    cache = diskcache.Cache("./state")

    a = dotdict(backend=cache)
    b = dotdict(backend=cache)

    a.x = 1   # cache["x"] = 1
    b.y = 2   # cache["y"] = 2  — does not touch "x"

    c = dotdict(backend=cache)
    print(c.x, c.y)  # 1  2
    ```

!!! warning "Same key, concurrent writes"
    Two instances writing to the **same key** concurrently will still race. Use `diskcache.Lock` to guard shared keys:

    ```python
    with diskcache.Lock(cache, "counter"):
        d = dotdict(backend=cache)
        d.counter = d.get("counter", 0) + 1
    ```

### Namespacing with `backend_prefix`

Use `backend_prefix` to share a single cache between multiple dotdicts without key collisions. Keys are stored as `"<prefix>.<key>"`:

???+ example

    ```python
    import diskcache
    from msgflux import dotdict

    cache = diskcache.Cache("./state")

    run1 = dotdict(backend=cache, backend_prefix="run_1")
    run2 = dotdict(backend=cache, backend_prefix="run_2")

    run1.score = 0.91   # cache["run_1.score"] = 0.91
    run2.score = 0.73   # cache["run_2.score"] = 0.73

    print(run1.score)  # 0.91
    print(run2.score)  # 0.73
    ```

### Nested values

Nested dicts are serialized to plain dicts when written to the backend and re-wrapped as `dotdict` on load, so dot access works after recovery:

???+ example

    ```python
    import diskcache
    from msgflux import dotdict

    cache = diskcache.Cache("./state")

    d = dotdict(backend=cache)
    d.user = {"name": "Maria", "age": 30}

    d2 = dotdict(backend=cache)
    print(d2.user.name)  # "Maria"
    print(d2.user.age)   # 30
    ```

!!! warning "Nested mutations are not auto-persisted"
    Mutations on a nested key (`d.user.name = "x"`) update only the in-memory dotdict.
    Reassign the top-level key to trigger persistence:

    ```python
    d.user = {**d.user, "name": "x"}   # persisted
    ```

---

## Extending dotdict

`dotdict` is designed to be subclassed. You can add default fields, metadata, or custom behavior by overriding `__init__`:

???+ example

    ```python
    from msgflux import dotdict


    class Config(dotdict):
        def __init__(self, env="production", **kwargs):
            super().__init__(**kwargs)
            self.env = env
            self.debug = env != "production"


    cfg = Config(env="development", timeout=30)

    print(cfg.env)      # "development"
    print(cfg.debug)    # True
    print(cfg.timeout)  # 30
    ```

[`Message`](nn/message.md) follows this exact pattern — it extends `dotdict` with default AI workflow fields (`content`, `images`, `context`, etc.) and metadata (`user_id`, `chat_id`, `execution_id`).
