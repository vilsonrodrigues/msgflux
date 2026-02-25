# Utility Functions

## wait_for

Execute a callable and wait for the result with optional timeout.

???+ example

    === "Sync Function"

        ```python
        import msgflux.nn.functional as F

        def slow_computation(x):
            import time
            time.sleep(0.1)
            return x * x

        result = F.wait_for(slow_computation, 5)
        print(result)  # 25
        ```

    === "Async Function"

        ```python
        import msgflux.nn.functional as F

        async def async_task(x):
            return x * 2

        # Runs async function in sync context
        result = F.wait_for(async_task, 3)
        print(result)  # 6
        ```

    === "With Timeout"

        ```python
        import msgflux.nn.functional as F

        result = F.wait_for(slow_computation, 10, timeout=0.5)
        ```

## wait_for_event

Wait for an `asyncio.Event` in synchronous code.

???+ example

    ```python
    import msgflux.nn.functional as F
    import asyncio
    import threading
    import time

    event = asyncio.Event()

    def set_event_later():
        time.sleep(0.1)
        # Need to set from the async context
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(event.set)

    thread = threading.Thread(target=set_event_later)
    thread.start()

    F.wait_for_event(event)  # Blocks until event is set
    print("Event was set!")
    ```

**Async version:** `await_for_event`

## fire_and_forget

Dispatch a task without waiting for a result. The task is not tracked and no return is provided.

???+ example

    === "Sync Function"

        ```python
        import msgflux.nn.functional as F

        def log_event(event_type, user_id):
            print(f"Logging: {event_type} for user {user_id}")

        # Returns immediately
        F.fire_and_forget(log_event, "login", 12345)
        print("Main thread continues...")
        ```

    === "Async Function"

        ```python
        import msgflux.nn.functional as F
        import asyncio

        async def async_log(message):
            await asyncio.sleep(1)
            print(f"[Async] {message}")

        F.fire_and_forget(async_log, "Hello from fire_and_forget")
        ```

    === "Error Handling"

        ```python
        import msgflux.nn.functional as F

        def failing_task():
            raise ValueError("This task failed!")

        # Error will be logged, not raised
        F.fire_and_forget(failing_task)
        ```

!!! tip "Use Cases"
    Fire-and-forget tasks are ideal for:

    - Logging and analytics
    - Cache updates
    - Notifications
    - Non-critical side effects

**Async version:** `afire_and_forget`
