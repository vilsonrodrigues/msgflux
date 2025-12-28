# dotdict

The `dotdict` class provides an enhanced dictionary with attribute-style access and nested path support, making it easier to work with complex nested data structures.


## Overview

`dotdict` extends Python's built-in `dict` with convenient features:

- **Dot notation access**: `obj.key` instead of `obj['key']`
- **Nested path operations**: `obj.get("user.profile.name")`
- **List index support**: `obj.get("items.0.title")`
- **Immutability option**: Create frozen dictionaries
- **Hidden keys**: Protect sensitive data from being displayed
- **Type preservation**: Automatic wrapping of nested dicts and lists

## Quick Start

### Basic Usage

```python
import msgflux as mf

# Create a dotdict
user = mf.dotdict({
    "name": "Alice",
    "age": 30,
    "email": "alice@example.com"
})

# Access with dot notation
print(user.name)   # "Alice"
print(user.age)    # 30

# Also works with traditional bracket notation
print(user["email"])  # "alice@example.com"

# Set values with dot notation
user.location = "New York"
print(user.location)  # "New York"
```

### Nested Structures

```python
import msgflux as mf

# Create nested structure
data = mf.dotdict({
    "user": {
        "profile": {
            "name": "Bob",
            "age": 25
        },
        "settings": {
            "theme": "dark",
            "notifications": True
        }
    }
})

# Access nested values
print(data.user.profile.name)  # "Bob"
print(data.user.settings.theme)  # "dark"

# Nested values are automatically wrapped as dotdict
print(type(data.user))  # <class 'msgflux.dotdict.dotdict'>
```

## Path-Based Access

### Reading with `get()`

Access deeply nested values using dot-separated paths:

```python
import msgflux as mf

data = mf.dotdict({
    "api": {
        "endpoints": {
            "users": "/api/v1/users",
            "posts": "/api/v1/posts"
        }
    }
})

# Get with path
endpoint = data.get("api.endpoints.users")
print(endpoint)  # "/api/v1/users"

# With default value
missing = data.get("api.endpoints.comments", "/api/v1/comments")
print(missing)  # "/api/v1/comments"
```

### Writing with `set()`

Create or modify nested values using paths:

```python
import msgflux as mf

config = mf.dotdict()

# Set nested values (creates intermediate dotdicts automatically)
config.set("database.host", "localhost")
config.set("database.port", 5432)
config.set("database.credentials.username", "admin")

print(config.database.host)  # "localhost"
print(config.database.credentials.username)  # "admin"

# View structure
print(config)
# {'database': {'host': 'localhost', 'port': 5432, 'credentials': {'username': 'admin'}}}
```

## Understanding Access Methods: `get()` vs Dot Notation

**IMPORTANT**: There are two ways to access values in a dotdict, and understanding when to use each is critical.

### When Dot Notation Works

Dot notation (e.g., `config.database.host`) **only works when the nested structure already exists**. The structure must be created first via:

- Initial dictionary in constructor
- `set()` method calls
- `update()` method calls

```python
import msgflux as mf

# ✓ WORKS: Structure created in constructor
data = mf.dotdict({"user": {"name": "Alice"}})
print(data.user.name)  # "Alice" - works!

# ✓ WORKS: Structure created via set()
config = mf.dotdict()
config.set("database.host", "localhost")
print(config.database.host)  # "localhost" - works!

# ✓ WORKS: Structure created via update()
settings = mf.dotdict()
settings.update({"server.port": 8080})
print(settings.server.port)  # 8080 - works!
```

### When Dot Notation Fails

Dot notation **fails with AttributeError** when trying to access non-existent intermediate paths:

```python
import msgflux as mf

# ✗ FAILS: Trying to access non-existent path
config = mf.dotdict()
try:
    port = config.database.port  # AttributeError!
except AttributeError as e:
    print(f"Error: {e}")  # Error: 'dotdict' object has no attribute 'database'
```

### Use `get()` for Safe Access

The `get()` method with path strings **always works safely**, even for non-existent paths:

```python
import msgflux as mf

config = mf.dotdict()

# ✓ WORKS: get() with default value
port = config.get("database.port", 5432)
print(port)  # 5432 (default value)

# ✓ WORKS: get() returns None for missing paths
host = config.get("database.host")
print(host)  # None

# After creating the structure, both methods work
config.set("database.port", 3306)
print(config.get("database.port"))  # 3306
print(config.database.port)  # 3306 - now this works too!
```

### Best Practice Guidelines

**Use `get()` when:**
- Accessing potentially non-existent paths
- You want a default value if the path doesn't exist
- Working with dynamic or uncertain data structures
- You need safe access without try/except blocks

**Use dot notation when:**
- The structure is guaranteed to exist (created in constructor or via set/update)
- Accessing top-level keys that you know exist
- Code readability is important and you're certain about the structure

```python
import msgflux as mf

# Example: Safe API response handling
response = mf.dotdict()

# Use get() for potentially missing fields
user_id = response.get("data.user.id", None)
if user_id:
    # Now safe to use dot notation on known structure
    print(f"User ID: {user_id}")

# Or combine both approaches
config = mf.dotdict()
config.set("app.name", "MyApp")
config.set("app.version", "1.0")

# Safe: structure was created above
print(f"Running {config.app.name} v{config.app.version}")

# Safe: using get() for optional settings
debug = config.get("app.debug", False)
print(f"Debug mode: {debug}")
```

## Working with Lists

`dotdict` supports accessing list items using numeric indices in paths:

```python
import msgflux as mf

data = mf.dotdict({
    "users": [
        {"name": "Alice", "role": "admin"},
        {"name": "Bob", "role": "user"},
        {"name": "Charlie", "role": "moderator"}
    ]
})

# Access list items by index
first_user = data.get("users.0.name")
print(first_user)  # "Alice"

second_role = data.get("users.1.role")
print(second_role)  # "user"

# Set values in lists
data.set("users.0.status", "active")
print(data.users[0].status)  # "active"

# Note: dot notation doesn't work for numeric indices
# data.users.0.name  # SyntaxError
# Use get() or bracket notation instead:
print(data.users[0].name)  # "Alice"
```

## Immutability with `frozen`

Create read-only dictionaries:

```python
import msgflux as mf

# Create frozen dotdict
constants = mf.dotdict(
    {"PI": 3.14159, "E": 2.71828},
    frozen=True
)

print(constants.PI)  # 3.14159

# Attempting to modify raises an error
try:
    constants.PI = 3.14
except AttributeError as e:
    print(e)  # "Cannot modify frozen dotdict"

try:
    constants.set("GOLDEN_RATIO", 1.618)
except AttributeError as e:
    print(e)  # "Cannot modify frozen dotdict"
```

## Hidden Keys

Protect sensitive data from being displayed or accessed via `get()`:

```python
import msgflux as mf

# Create dotdict with hidden keys
config = mf.dotdict(
    {
        "api_key": "sk-secret-key-12345",
        "api_url": "https://api.example.com",
        "username": "admin",
        "password": "super-secret"
    },
    hidden_keys=["api_key", "password"]
)

# Hidden keys return None when accessed via get()
print(config.get("api_key"))  # None
print(config.get("password"))  # None

# Non-hidden keys work normally
print(config.get("api_url"))  # "https://api.example.com"
print(config.get("username"))  # "admin"

# Hidden keys are still accessible via dot/bracket notation
print(config.api_key)  # "sk-secret-key-12345" (direct access still works)
print(config["password"])  # "super-secret"

# Hidden keys don't appear in string representations
print(config)  # {'api_url': 'https://api.example.com', 'username': 'admin'}
```

**Use Cases for Hidden Keys:**
- API keys and secrets
- Passwords and tokens
- Sensitive user data
- Internal configuration that shouldn't be logged

## Advanced Features

### Update with Nested Paths

Use `update()` with dot-separated keys:

```python
import msgflux as mf

config = mf.dotdict({
    "server": {"host": "localhost"}
})

# Update with nested keys
config.update({
    "server.port": 8080,
    "server.ssl": True,
    "database.type": "postgresql"
})

print(config.server.port)  # 8080
print(config.database.type)  # "postgresql"

# Merge existing nested structures
config.update({
    "server": {"workers": 4}
})

print(config.server.host)  # "localhost" (preserved)
print(config.server.workers)  # 4 (added)
```

### Convert to Regular Dict

Convert back to a standard Python dictionary:

```python
import msgflux as mf

data = mf.dotdict({
    "user": {
        "name": "Alice",
        "settings": {"theme": "dark"}
    }
})

# Convert to regular dict
regular_dict = data.to_dict()
print(type(regular_dict))  # <class 'dict'>
print(type(regular_dict["user"]))  # <class 'dict'> (not dotdict)

# All nested dotdicts are converted to regular dicts
print(regular_dict)
# {'user': {'name': 'Alice', 'settings': {'theme': 'dark'}}}
```

### JSON Serialization

Export as JSON:

```python
import msgflux as mf

data = mf.dotdict({
    "name": "Product",
    "price": 29.99,
    "tags": ["electronics", "gadget"]
})

# Serialize to JSON bytes
json_bytes = data.to_json()
print(json_bytes)
# b'{"name":"Product","price":29.99,"tags":["electronics","gadget"]}'

# Decode to string if needed
json_str = json_bytes.decode('utf-8')
print(json_str)
# '{"name":"Product","price":29.99,"tags":["electronics","gadget"]}'
```

## Common Use Cases

### Configuration Management

```python
import msgflux as mf

# Application configuration
config = mf.dotdict()

# Set configuration values
config.set("app.name", "My Application")
config.set("app.version", "1.0.0")
config.set("database.host", "localhost")
config.set("database.port", 5432)
config.set("redis.host", "localhost")
config.set("redis.port", 6379)

# Access configuration easily
print(f"Starting {config.app.name} v{config.app.version}")
print(f"Database: {config.database.host}:{config.database.port}")
```

### API Response Handling

```python
import msgflux as mf

# Simulate API response
api_response = mf.dotdict({
    "status": "success",
    "data": {
        "user": {
            "id": 123,
            "username": "johndoe",
            "profile": {
                "full_name": "John Doe",
                "avatar_url": "https://example.com/avatar.jpg"
            }
        },
        "permissions": ["read", "write", "admin"]
    }
})

# Easy access to nested data
if api_response.status == "success":
    user_id = api_response.data.user.id
    full_name = api_response.data.user.profile.full_name
    permissions = api_response.data.permissions

    print(f"User {full_name} (ID: {user_id})")
    print(f"Permissions: {', '.join(permissions)}")
```

### Message Passing

```python
import msgflux as mf

# Create message with dotdict
message = mf.dotdict()

# Add data at different stages
message.set("request.id", "req-12345")
message.set("request.timestamp", "2024-01-15T10:30:00")

# Processing stage
message.set("processing.model", "gpt-5")
message.set("processing.tokens", 150)

# Response stage
message.set("response.status", "completed")
message.set("response.data", "Hello, world!")

# Access full context
print(f"Request {message.request.id} processed with {message.processing.model}")
print(f"Result: {message.response.data}")
```

## Integration with Message

`dotdict` works seamlessly with msgflux's `Message` class:

```python
import msgflux as mf

# Message internally uses dotdict
msg = mf.Message()
msg.set("user.id", 123)
msg.set("user.name", "Alice")

# Access like dotdict
print(msg.user.name)  # "Alice"

# Or create dotdict from message data
data = mf.dotdict(msg.to_dict())
print(data.user.id)  # 123
```

## API Reference

### Constructor

```python
mf.dotdict(data=None, *, frozen=False, hidden_keys=None, **kwargs)
```

**Parameters:**
- `data` (dict, optional): Initial dictionary data
- `frozen` (bool): If `True`, creates immutable dotdict
- `hidden_keys` (list[str]): Keys to hide from `get()` and string representations
- `**kwargs`: Additional key-value pairs

### Methods

| Method | Description |
|--------|-------------|
| `get(path, default=None)` | Get value using dot-separated path |
| `set(path, value)` | Set value using dot-separated path |
| `update(*args, **kwargs)` | Update with dict or key-value pairs (supports nested paths) |
| `to_dict()` | Convert to regular Python dict |
| `to_json()` | Serialize to JSON bytes |

### Attributes

All standard `dict` methods are available (`keys()`, `values()`, `items()`, etc.)

## Best Practices

### 1. Use for Complex Nested Data

```python
# Good - Complex nested structures
config = mf.dotdict({
    "services": {
        "api": {"host": "api.example.com"},
        "db": {"host": "db.example.com"}
    }
})

# Not ideal - Simple flat data (regular dict is fine)
simple = mf.dotdict({"name": "Alice", "age": 30})
```

### 2. Protect Sensitive Data

```python
# Good - Hide sensitive keys
credentials = mf.dotdict(
    {"username": "admin", "password": "secret"},
    hidden_keys=["password"]
)
```

### 3. Use Frozen for Constants

```python
# Good - Immutable configuration
CONSTANTS = mf.dotdict(
    {"MAX_RETRIES": 3, "TIMEOUT": 30},
    frozen=True
)
```

### 4. Use `get()` for Uncertain Paths, Dot Notation for Known Structures

```python
import msgflux as mf

# Good - Use get() for potentially non-existent paths
value = data.get("level1.level2.level3.value", "default")

# Good - Use dot notation when structure is guaranteed to exist
config = mf.dotdict({"app": {"name": "MyApp", "version": "1.0"}})
print(config.app.name)  # Safe because structure exists

# Avoid - Dot notation on uncertain paths without error handling
try:
    value = data.level1.level2.level3.value  # Risky!
except AttributeError:
    value = "default"  # Better to use get() instead

# Best - Combine both approaches appropriately
config = mf.dotdict()
config.set("server.host", "localhost")  # Creates structure
host = config.server.host  # Safe: structure exists
port = config.get("server.port", 8080)  # Safe: use get() for optional value
```
