# Gather Functions

## map_gather

Apply the same function to multiple inputs concurrently.

???+ example

    === "Basic"

        ```python
        import msgflux.nn.functional as F

        def square(x):
            return x * x

        results = F.map_gather(square, args_list=[(2,), (3,), (4,)])
        print(results)  # (4, 9, 16)
        ```

    === "With Multiple Arguments"

        ```python
        import msgflux.nn.functional as F

        def add(x, y):
            return x + y

        results = F.map_gather(add, args_list=[(1, 2), (3, 4), (5, 6)])
        print(results)  # (3, 7, 11)
        ```

    === "With kwargs"

        ```python
        import msgflux.nn.functional as F

        def multiply(x, y=2):
            return x * y

        results = F.map_gather(
            multiply,
            args_list=[(1,), (3,), (5,)],
            kwargs_list=[{"y": 3}, {"y": 4}, {"y": 5}]
        )
        print(results)  # (3, 12, 25)
        ```

    === "With Timeout"

        ```python
        import msgflux.nn.functional as F
        import time

        def slow_task(x):
            time.sleep(0.5)
            return x * x

        # 1 second timeout for all tasks
        results = F.map_gather(
            slow_task,
            args_list=[(2,), (3,), (4,)],
            timeout=1.0
        )
        ```

**Async version:** `amap_gather`

## scatter_gather

Distribute different functions across corresponding inputs.

???+ example

    === "Basic"

        ```python
        import msgflux.nn.functional as F

        def double(x): return x * 2
        def triple(x): return x * 3
        def square(x): return x ** 2

        results = F.scatter_gather(
            [double, triple, square],
            args_list=[(5,), (5,), (5,)]
        )
        print(results)  # (10, 15, 25)
        ```

    === "Different Inputs"

        ```python
        import msgflux.nn.functional as F

        def double(x): return x * 2
        def triple(x): return x * 3
        def square(x): return x ** 2

        results = F.scatter_gather(
            [double, triple, square],
            args_list=[(2,), (3,), (4,)]
        )
        print(results)  # (4, 9, 16)
        ```

    === "With kwargs Only"

        ```python
        import msgflux.nn.functional as F

        def greet(name="World"):
            return f"Hello, {name}"

        def farewell(person):
            return f"Goodbye, {person}"

        results = F.scatter_gather(
            [greet, greet, farewell],
            kwargs_list=[{}, {"name": "Earth"}, {"person": "Commander"}]
        )
        print(results)  # ("Hello, World", "Hello, Earth", "Goodbye, Commander")
        ```

**Async version:** `ascatter_gather`

## bcast_gather

Broadcast the same arguments to multiple functions.

???+ example

    === "Basic"

        ```python
        import msgflux.nn.functional as F

        def square(x): return x * x
        def cube(x): return x * x * x
        def double(x): return x * 2

        results = F.bcast_gather([square, cube, double], 5)
        print(results)  # (25, 125, 10)
        ```

    === "Error Handling"

        ```python
        import msgflux.nn.functional as F
        from msgflux import TaskError

        def square(x): return x * x
        def fail(x): raise ValueError("Intentional error")
        def cube(x): return x * x * x

        # Failed tasks return a TaskError instance
        results = F.bcast_gather([square, fail, cube], 2)
        print(results)  # (4, TaskError(index=1, ...), 8)

        # Inspect errors while keeping successful results
        for i, result in enumerate(results):
            if isinstance(result, TaskError):
                print(f"Task {i} failed: {result.exception}")
            else:
                print(f"Task {i} result: {result}")
        ```

    === "With kwargs"

        ```python
        import msgflux.nn.functional as F

        def fetch_user(user_id):
            return {"id": user_id, "name": f"User {user_id}"}

        def fetch_posts(user_id):
            return [f"Post {i}" for i in range(3)]

        def fetch_comments(user_id):
            return [f"Comment {i}" for i in range(5)]

        user, posts, comments = F.bcast_gather(
            [fetch_user, fetch_posts, fetch_comments],
            user_id=123
        )
        ```

## Message-Based Functions

These functions operate on `msgflux.dotdict` objects, enabling message-passing patterns common in workflow orchestration.

### msg_scatter_gather

Route different messages to different processors.

???+ example

    ```python
    import msgflux as mf
    import msgflux.nn.functional as F

    def process_user(msg):
        msg.type = "user"
        msg.processed = True
        return msg

    def process_admin(msg):
        msg.type = "admin"
        msg.permissions = ["read", "write", "delete"]
        return msg

    def process_guest(msg):
        msg.type = "guest"
        msg.permissions = ["read"]
        return msg

    msg1 = mf.dotdict({"id": 1, "name": "Alice"})
    msg2 = mf.dotdict({"id": 2, "name": "Bob"})
    msg3 = mf.dotdict({"id": 3, "name": "Charlie"})

    results = F.msg_scatter_gather(
        [process_user, process_admin, process_guest],
        [msg1, msg2, msg3]
    )

    for msg in results:
        print(f"{msg.name}: {msg.type}")
    # Alice: user
    # Bob: admin
    # Charlie: guest
    ```

### msg_bcast_gather

Broadcast a single message to multiple processors for concurrent modification.

???+ example

    ```python
    import msgflux as mf
    import msgflux.nn.functional as F
    from datetime import datetime

    def add_timestamp(msg):
        msg.timestamp = datetime.now().isoformat()
        return msg

    def add_metadata(msg):
        msg.set("metadata.version", "1.0")
        msg.set("metadata.source", "api")
        return msg

    def validate(msg):
        msg.validated = True
        return msg

    message = mf.dotdict({"data": "important"})

    F.msg_bcast_gather([add_timestamp, add_metadata, validate], message)

    print(message.timestamp)          # 2024-01-15T10:30:00.123456
    print(message.metadata.version)   # 1.0
    print(message.validated)          # True
    ```

!!! warning "Race Conditions"
    In parallel execution, modules should modify **different paths** of the message. Modifying the same path from multiple concurrent functions may cause race conditions.

**Async version:** `amsg_bcast_gather`
