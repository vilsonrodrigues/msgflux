# https://mpitutorial.com/tutorials/mpi-scatter-gather-and-allgather/
import asyncio
import concurrent.futures
import time
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Tuple

from msgflux._private.executor import Executor
from msgflux._private.supervision import gather_durable_async, gather_durable_sync
from msgflux.dotdict import dotdict
from msgflux.exceptions import TaskError
from msgflux.logger import logger
from msgflux.nn.modules.module import get_callable_name
from msgflux.telemetry import Spans

if TYPE_CHECKING:
    from msgflux.data.stores.base import CheckpointStore

__all__ = [
    "afire_and_forget",
    "ainline",
    "amap_gather",
    "amsg_bcast_gather",
    "ascatter_gather",
    "await_for_event",
    "bcast_gather",
    "fire_and_forget",
    "inline",
    "map_gather",
    "msg_bcast_gather",
    "msg_scatter_gather",
    "scatter_gather",
    "wait_for",
    "wait_for_event",
]


@Spans.instrument()
def map_gather(
    to_send: Callable,
    *,
    args_list: List[Tuple[Any, ...]],
    kwargs_list: Optional[List[Dict[str, Any]]] = None,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    retry_delay: float = 1.0,
    store: Optional["CheckpointStore"] = None,
    run_id: Optional[str] = None,
    namespace: str = "gather",
    session_id: str = "default",
) -> Tuple[Any, ...]:
    """Applies the `to_send` function to each set of arguments in `args_list`
    and `kwargs_list` using Executor and collects the results.

    Args:
        to_send:
            The callable function to be applied.
        args_list:
            Each tuple contains the positional argumentsvfor the corresponding callable
            in `to_send`. If `None`, no positional arguments are passed unless specified
            individually by an item in `kwargs_list`.
        kwargs_list:
            Each dictionary contains the named arguments for the corresponding callable
            in `to_send`. If `None`, no named arguments are passed unless specified
            individually by an item in `args_list`.
        timeout:
            Maximum time (in seconds) to wait for responses.

    Returns:
        A tuple containing the results of each call to the `f` function. If a call
        fails or times out, the corresponding result will be a `TaskError` instance.

    Raises:
        TypeError:
            If `f` is not callable.
        ValueError:
            If `args_list` is not a non-empty list or if `kwargs_list`
            (if provided) is not the same length as `args_list`.

    Examples:
        def add(x, y): return x + y
        results = F.map_gather(add, args_list=[(1, 2), (3, 4), (5, 6)])
        print(results)  # (3, 7, 11)

        def multiply(x, y=2): return x * y
        results = F.map_gather(multiply, args_list=[(1,), (3,), (5,)],
                            kwargs_list=[{'y': 3}, {'y': 4}, {'y': 5}])
        print(results)  # (3, 12, 25)

        results = F.map_gather(multiply, args_list=[(1,), (3,), (5,)])
        print(results)  # (2, 6, 10)
    """
    if not callable(to_send):
        raise TypeError("`to_send` must be a callable object")

    if not isinstance(args_list, list) or len(args_list) == 0:
        raise ValueError("`args_list` must be a non-empty list")

    if kwargs_list is not None:
        if not isinstance(kwargs_list, list) or len(kwargs_list) != len(args_list):
            raise ValueError(
                "`kwargs_list` must be a list with the same length as `args_list`"
            )

    # Durable path
    if max_retries is not None or store is not None:
        workers = [
            (to_send, args_list[i], kwargs_list[i] if kwargs_list else {})
            for i in range(len(args_list))
        ]
        return tuple(
            gather_durable_sync(
                workers,
                timeout=timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
                store=store,
                run_id=run_id,
                namespace=namespace,
                session_id=session_id,
            )
        )

    executor = Executor.get_instance()
    futures = []

    for i in range(len(args_list)):
        args = args_list[i]
        kwargs = kwargs_list[i] if kwargs_list else {}
        futures.append(executor.submit(to_send, *args, **kwargs))

    concurrent.futures.wait(futures, timeout=timeout)
    responses: List[Any] = []
    for i, future in enumerate(futures):
        try:
            responses.append(future.result())
        except Exception as e:
            logger.error(str(e))
            responses.append(TaskError(exception=e, index=i))
    return tuple(responses)


@Spans.instrument()
def scatter_gather(
    to_send: List[Callable],
    args_list: Optional[List[Tuple[Any, ...]]] = None,
    kwargs_list: Optional[List[Dict[str, Any]]] = None,
    *,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    retry_delay: float = 1.0,
    store: Optional["CheckpointStore"] = None,
    run_id: Optional[str] = None,
    namespace: str = "gather",
    session_id: str = "default",
) -> Tuple[Any, ...]:
    """Sends different sets of arguments/kwargs to a list of modules
    and collects the responses.

    Each callable in `to_send` receives the positional arguments of
    the corresponding `tuple` in `args_list` and the named arguments
    of the corresponding `dict` in `kwargs_list`. If `args_list` or
    `kwargs_list` are not provided (or are `None`), the corresponding
    callables will be called without positional or named arguments,
    respectively, unless an empty list (`[]`) or empty tuple (`()`)
    is provided for a specific item.

    Args:
        to_send:
            List of callable objects (e.g. functions or `Module` instances).
        args_list:
            Each tuple contains the positional argumentsvfor the corresponding callable
            in `to_send`. If `None`, no positional arguments are passed unless specified
            individually by an item in `kwargs_list`.
        kwargs_list:
            Each dictionary contains the named arguments for the corresponding callable
            in `to_send`. If `None`, no named arguments are passed unless specified
            individually by an item in `args_list`.
        timeout:
            Maximum time (in seconds) to wait for responses.

    Returns:
        Tuple containing the responses for each callable. If an error or
        timeout occurs for a specific callable, its corresponding response
        in the tuple will be a `TaskError` instance.

    Raises:
        TypeError:
            If `to_send` is not a callable list.
        ValueError:
            If the lengths of `args_list` (if provided) or `kwargs_list`
            (if provided) do not match the length of `to_send`.

    Examples:
        def add(x, y): return x + y
        def multiply(x, y=2): return x * y
        callables = [add, multiply, add]

        # Example 1: Using only args_list
        args = [ (1, 2), (3,), (10, 20) ] # multiply will use its default y
        results = F.scatter_gather(callables, args_list=args)
        print(results) # (3, 6, 30)

        # Example 2: Using args_list e kwargs_list
        args = [ (1,), (), (10,) ]
        kwargs = [ {'y': 2}, {'x': 3, 'y': 3}, {'y': 20} ]
        results = F.scatter_gather(callables, args_list=args, kwargs_list=kwargs)
        print(results) # (3, 9, 30)

        # Example 3: Using only kwargs_list (useful if functions have
        # defaults or don't need positional args)
        def greet(name="World"): return f"Hello, {name}"
        def farewell(person_name): return f"Goodbye, {person_name}"
        funcs = [greet, greet, farewell]
        kwargs_for_funcs = [ {}, {'name': "Earth"}, {'person_name': "Commander"} ]
        results = F.scatter_gather(funcs, kwargs_list=kwargs_for_funcs)
        print(results) # ("Hello, World", "Hello, Earth", "Goodbye, Commander")
    """
    if not isinstance(to_send, list) or not all(callable(f) for f in to_send):
        raise TypeError("`to_send` must be a non-empty list of callable objects")

    # Durable path
    if max_retries is not None or store is not None:
        workers = [
            (
                to_send[i],
                args_list[i] if args_list and i < len(args_list) else (),
                kwargs_list[i] if kwargs_list and i < len(kwargs_list) else {},
            )
            for i in range(len(to_send))
        ]
        return tuple(
            gather_durable_sync(
                workers,
                timeout=timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
                store=store,
                run_id=run_id,
                namespace=namespace,
                session_id=session_id,
            )
        )

    executor = Executor.get_instance()
    futures = []
    for i, f in enumerate(to_send):
        args = args_list[i] if args_list and i < len(args_list) else ()
        kwargs = kwargs_list[i] if kwargs_list and i < len(kwargs_list) else {}
        futures.append(executor.submit(f, *args, **kwargs))

    concurrent.futures.wait(futures, timeout=timeout)
    responses: List[Any] = []
    for i, future in enumerate(futures):
        try:
            responses.append(future.result())
        except Exception as e:
            logger.error(str(e))
            responses.append(TaskError(exception=e, index=i))
    return tuple(responses)


@Spans.instrument()
def msg_scatter_gather(
    to_send: List[Callable],
    messages: List[dotdict],
    *,
    timeout: Optional[float] = None,
) -> Tuple[dotdict, ...]:
    """Scatter a list of messages to a list of modules and gather the responses.

    Args:
        to_send:
            List of callable objects (e.g. functions or `Module` instances).
        messages:
            List of `msgflux.dotdict` instances to be distributed.
        timeout:
            Maximum time (in seconds) to wait for responses.

    Returns:
        Tuple containing the messages updated with the responses.

    Raises:
        TypeError:
            If `messages` is not a list of `dotdict`, `to_send` is not a list
            of callables, or `prefix` is not a string.
    """
    if not messages or not all(isinstance(msg, dotdict) for msg in messages):
        raise TypeError(
            "`messages` must be a non-empty list of `msgflux.dotdict` instances"
        )

    if not to_send or not all(isinstance(f, Callable) for f in to_send):
        raise TypeError("`to_send` must be a non-empty list of callable objects")

    if len(messages) != len(to_send):
        raise ValueError(
            f"The size of `messages` ({len(messages)}) "
            f"must be equal to that of `to_send`: ({len(to_send)})"
        )

    executor = Executor.get_instance()
    futures = [executor.submit(f, msg) for f, msg in zip(to_send, messages)]

    concurrent.futures.wait(futures, timeout=timeout)
    for i, (f, future) in enumerate(zip(to_send, futures)):
        f_name = get_callable_name(f)
        try:
            future.result()
        except Exception as e:
            logger.error(f"Error in scattered task for `{f_name}`: {e}")
            messages[i]["_error"] = TaskError(exception=e, index=i)
    return tuple(messages)


@Spans.instrument()
def bcast_gather(
    to_send: List[Callable],
    *args,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    retry_delay: float = 1.0,
    **kwargs,
) -> Tuple[Any, ...]:
    """Broadcasts arguments to multiple callables and gathers the responses.

    Args:
        to_send:
            List of callable objects (e.g. functions or `Module` instances).
        *args:
            Positional arguments.
        timeout:
            Maximum time (in seconds) to wait for responses.
        **kwargs:
            Named arguments.

    Returns:
        Tuple containing the responses.

    Raises:
        TypeError:
            If `to_send` is not a list of callables.

    Examples:
        def square(x): return x * x
        def cube(x): return x * x * x
        def fail(x): raise ValueError("Intentional error")

        # Example 1:
        results = F.bcast_gather([square, cube], 3)
        print(results)  # (9, 27)

        # Example 2: Simulate error
        results = F.bcast_gather([square, fail, cube], 2)
        print(results)  # (4, TaskError(...), 8)

        # Example 3: Timeout
        results = F.bcast_gather([square, cube], 4, timeout=0.01)
        print(results) # (16, 64)
    """
    if not to_send or not all(isinstance(f, Callable) for f in to_send):
        raise TypeError("`to_send` must be a non-empty list of callable objects")

    # Durable path
    if max_retries is not None:
        workers = [(f, args, kwargs) for f in to_send]
        return tuple(
            gather_durable_sync(
                workers,
                timeout=timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
                store=None,
                run_id=None,
                namespace="gather",
                session_id="default",
            )
        )

    executor = Executor.get_instance()
    futures = [executor.submit(f, *args, **kwargs) for f in to_send]

    concurrent.futures.wait(futures, timeout=timeout)
    responses: List[Any] = []
    for i, future in enumerate(futures):
        try:
            responses.append(future.result())
        except Exception as e:
            logger.error(str(e))
            responses.append(TaskError(exception=e, index=i))
    return tuple(responses)


@Spans.instrument()
def msg_bcast_gather(
    to_send: List[Callable],
    message: dotdict,
    *,
    timeout: Optional[float] = None,
) -> dotdict:
    """Broadcasts a single message to multiple modules and gathers the responses.

    Args:
        to_send:
            List of callable objects (e.g. functions or `Module` instances).
        message:
            Instance of `msgflux.dotdict` to broadcast.
        timeout:
            Maximum time (in seconds) to wait for responses.

    Returns:
        The original message with the module responses added.

    Raises:
        TypeError:
            If `message` is not an instance of `dotdict`, `to_send` is not a list
            of callables.
    """
    if not isinstance(message, dotdict):
        raise TypeError("`message` must be an instance of `msgflux.dotdict`")
    if not to_send or not all(isinstance(module, Callable) for module in to_send):
        raise TypeError("`to_send` must be a non-empty list of callable objects")

    executor = Executor.get_instance()
    futures = [executor.submit(f, message) for f in to_send]

    concurrent.futures.wait(futures, timeout=timeout)
    for i, (f, future) in enumerate(zip(to_send, futures)):
        f_name = get_callable_name(f)
        try:
            future.result()
        except Exception as e:
            logger.error(f"Error in scattered task for `{f_name}`: {e}")
            message.setdefault("_errors", {})[f_name] = TaskError(exception=e, index=i)
    return message


@Spans.instrument()
def wait_for(
    to_send: Callable,
    *args,
    timeout: Optional[float] = None,
    max_retries: Optional[int] = None,
    retry_delay: float = 1.0,
    **kwargs,
) -> Any:
    """Wait for a callable execution.

    Args:
        to_send:
            A callable object (e.g. functions or `Module` instances).
        *args:
            Positional arguments.
        timeout:
            Maximum time (in seconds) to wait for responses.
        **kwargs:
            Named arguments.

    Returns:
        Callable responses.

    Raises:
        TypeError:
            If `to_send` is not a callable.

    Examples:
        async def f1(x):
            return x * x

        # Example 1:
        results = F.wait_for(f1, 3)
        print(results) # 9
    """
    if not callable(to_send):
        raise TypeError("`to_send` must be a callable object")

    # Retry path
    if max_retries is not None:
        retries = 0
        executor = Executor.get_instance()
        while True:
            future = executor.submit(to_send, *args, **kwargs)
            concurrent.futures.wait([future], timeout=timeout)
            try:
                return future.result()
            except Exception as e:
                retries += 1
                if retries > max_retries:
                    logger.error(str(e))
                    return TaskError(exception=e, index=0)
                from msgflux._private.supervision import _backoff_delay  # noqa: PLC0415

                time.sleep(_backoff_delay(retries, retry_delay))

    executor = Executor.get_instance()
    future = executor.submit(to_send, *args, **kwargs)
    concurrent.futures.wait([future], timeout=timeout)
    try:
        return future.result()
    except Exception as e:
        logger.error(str(e))
        return TaskError(exception=e, index=0)


@Spans.instrument()
def wait_for_event(event: asyncio.Event) -> None:
    """Waits synchronously for an asyncio.Event to be set.

    This function will block until event.set() is called elsewhere.

    Args:
        event: The asyncio.Event to wait for.

    Raises:
        TypeError: If `event` is not an instance of asyncio.Event.
    """
    if not isinstance(event, asyncio.Event):
        raise TypeError("`event` must be an instance of asyncio.Event")

    executor = Executor.get_instance()
    future = executor._submit_to_async_worker(event.wait())
    try:
        future.result()
    except Exception as e:
        logger.error(str(e))


@Spans.instrument()
def fire_and_forget(to_send: Callable, *args, **kwargs) -> None:
    """Dispatches a task without waiting for a result.
    Uses the AsyncExecutorPool. The task is not tracked and no return is provided.

    Args:
        to_send:
            Callable object (function, async function, or module with .acall() method).
        *args:
            Positional arguments.
        **kwargs:
            Named arguments.

    Raises:
        TypeError: If `to_send` is not a callable.

    Examples:
        # Example 1:
        import time
        def print_message(message: str):
            time.sleep(1)
            print(f"[Sync] Message: {message}")
        F.fire_and_forget(print_message, "Hello from sync function")

        # Example 2:
        import asyncio
        async def async_print_message(message: str):
            await asyncio.sleep(1)
            print(f"[Async] Message: {message}")
        F.fire_and_forget(async_print_message, "Hello from async function")

        # Example 3 (with error):
        def failing_task():
            raise ValueError("This task failed!")
        F.fire_and_forget(failing_task)  # Error will be logged
    """
    if not callable(to_send):
        raise TypeError("`to_send` must be a callable object")

    def log_future(future: Future) -> None:
        """Callback to log exception of a Future."""
        try:
            future.result()
        except Exception as e:
            logger.error(f"Fire-and-forget task error: {e!s}", exc_info=True)

    executor = Executor.get_instance()
    future = executor.submit(to_send, *args, **kwargs)
    future.add_done_callback(log_future)


@Spans.ainstrument()
async def afire_and_forget(to_send: Callable, *args, **kwargs) -> None:
    """Dispatches an async task without waiting for a result.
    The task is not tracked and no return is provided.

    Args:
        to_send:
            Callable object (async function or module with .acall() method).
        *args:
            Positional arguments.
        **kwargs:
            Named arguments.

    Raises:
        TypeError: If `to_send` is not a callable.

    Examples:
        # Example 1:
        import asyncio
        async def async_print_message(message: str):
            await asyncio.sleep(1)
            print(f"[Async] Message: {message}")
        await F.afire_and_forget(async_print_message, "Hello from async function")

        # Example 2 (with error):
        async def failing_task():
            raise ValueError("This task failed!")
        await F.afire_and_forget(failing_task)  # Error will be logged
    """
    if not callable(to_send):
        raise TypeError("`to_send` must be a callable object")

    async def run_task():
        """Wrapper to run the task and log errors."""
        try:
            if hasattr(to_send, "acall"):
                await to_send.acall(*args, **kwargs)
            elif asyncio.iscoroutinefunction(to_send):
                await to_send(*args, **kwargs)
            else:
                # Fall back to running sync function in executor
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, lambda: to_send(*args, **kwargs))
        except Exception as e:
            logger.error(f"Fire-and-forget task error: {e!s}", exc_info=True)

    asyncio.create_task(run_task())  # noqa: RUF006


@Spans.ainstrument()
async def await_for_event(event: asyncio.Event) -> None:
    """Waits asynchronously for an asyncio.Event to be set.

    This function will await until event.set() is called elsewhere.

    Args:
        event: The asyncio.Event to wait for.

    Raises:
        TypeError: If `event` is not an instance of asyncio.Event.

    Examples:
        # Example 1:
        import asyncio
        event = asyncio.Event()

        async def setter():
            await asyncio.sleep(1)
            event.set()

        asyncio.create_task(setter())
        await F.await_for_event(event)
        print("Event was set!")
    """
    if not isinstance(event, asyncio.Event):
        raise TypeError("`event` must be an instance of asyncio.Event")

    await event.wait()


def inline(
    expression: str, modules: Mapping[str, Callable], message: dotdict
) -> dotdict:
    """Executes a workflow defined in DSL expression over a given `message`.

    Args:
        expression:
            A string describing the execution pipeline using a
            Domain-Specific Language (DSL).

            The DSL supports:

            **Sequential execution**:
                Use `->` to define a linear pipeline.
                !!! example

                    `"prep -> transform -> output"`

            **Parallel execution**:
                Use square brackets `[...]` to group modules that run in parallel.
                !!! example

                    `"prep -> [feat_a, feat_b] -> combine"`

            **Conditional execution**:
                Use curly braces with a ternary-like structure:
                `{condition ? then_module, else_module}`.
                !!! example

                    `"{user.age > 18 ? adult_module, child_module}"`

            **While loops**:
                Use `@{condition}: actions;` to execute actions repeatedly
                while condition is true.
                !!! example

                    `"@{counter < 10}: increment;"`

            **Logical operations in conditions**:
                - **AND**: `cond1 & cond2`
                - **OR**: `cond1 || cond2`
                - **NOT**: `!cond`
                Example: `"{user.is_active & !user.is_banned ? allow, deny}"`

            **None checking in conditions**:
                - `is None`: Example: `user.name is None`
                - `is not None`: Example: `user.name is not None`

            These conditionals are evaluated against the `message` object context.

        modules:
            A dictionary mapping module names (as strings) to callables.
            Each function must accept and return a `message` object.

        message:
            The input message (dotdict) to be passed through the pipeline.

    Returns:
        The resulting `message` after executing the defined workflow.

    Raises:
        TypeError:
            If expression is not a str.
        TypeError:
            If message is not a `msgflux.dotdict` instance.
        TypeError:
            If modules is not a Mapping.
        ValueError:
            If a module is not found, if the DSL syntax is invalid,
            or if a condition cannot be parsed.
        RuntimeError:
            If a while loop exceeds the maximum iteration limit
            (prevents infinite loops).

    Examples:
        from msgflux import dotdict
        import msgflux.nn.functional as F

        def prep(msg: dotdict) -> dotdict:
            print(f"Executing prep, current msg: {msg}")
            msg['output'] = {'agent': 'xpto', 'score': 10, 'status': 'success'}
            msg['counter'] = 0
            return msg

        def increment(msg: dotdict) -> dotdict:
            print(f"Executing increment, current msg: {msg}")
            msg['counter'] = msg.get('counter', 0) + 1
            return msg

        def feat_a(msg: dotdict) -> dotdict:
            print(f"Executing feat_a, current msg: {msg}")
            msg['feat_a'] = 'result_a'
            return msg

        def feat_b(msg: dotdict) -> dotdict:
            print(f"Executing feat_b, current msg: {msg}")
            msg['feat_b'] = 'result_b'
            return msg

        def final(msg: dotdict) -> dotdict:
            print(f"Executing final, current msg: {msg}")
            msg['final'] = 'done'
            return msg

        my_modules = {
            "prep": prep,
            "increment": increment,
            "feat_a": feat_a,
            "feat_b": feat_b,
            "final": final
        }
        input_msg = dotdict()

        # Example with while loop
        result = F.inline(
            "prep -> @{counter < 5}: increment; -> final",
            modules=my_modules,
            message=input_msg
        )

        # Example with nested while loop and other constructs
        result = F.inline(
            "prep -> @{counter < 3}: increment -> [feat_a, feat_b]; -> final",
            modules=my_modules,
            message=input_msg
        )
    """
    from msgflux.dsl.inline import inline as _inline  # noqa: PLC0415

    return _inline(expression, modules, message)


async def ainline(
    expression: str, modules: Mapping[str, Callable], message: dotdict
) -> dotdict:
    """Async version of inline. Executes a workflow defined in DSL
    expression over a given `message`.

    Args:
        expression:
            A string describing the execution pipeline using a
            Domain-Specific Language (DSL).

            The DSL supports:

            **Sequential execution**:
                Use `->` to define a linear pipeline.
                !!! example

                    `"prep -> transform -> output"`

            **Parallel execution**:
                Use square brackets `[...]` to group modules that run in parallel.
                !!! example

                    `"prep -> [feat_a, feat_b] -> combine"`

            **Conditional execution**:
                Use curly braces with a ternary-like structure:
                `{condition ? then_module, else_module}`.
                !!! example

                    `"{user.age > 18 ? adult_module, child_module}"`

            **While loops**:
                Use `@{condition}: actions;` to execute actions repeatedly
                while condition is true.
                !!! example

                    `"@{counter < 10}: increment;"`

            **Logical operations in conditions**:
                - **AND**: `cond1 & cond2`
                - **OR**: `cond1 || cond2`
                - **NOT**: `!cond`
                Example: `"{user.is_active & !user.is_banned ? allow, deny}"`

            **None checking in conditions**:
                - `is None`: Example: `user.name is None`
                - `is not None`: Example: `user.name is not None`

            These conditionals are evaluated against the `message` object context.

        modules:
            A dictionary mapping module names (as strings) to callables.
            Each function must accept and return a `message` object.
            Supports both sync and async modules.

        message:
            The input message (dotdict) to be passed through the pipeline.

    Returns:
        The resulting `message` after executing the defined workflow.

    Raises:
        TypeError:
            If expression is not a str.
        TypeError:
            If message is not a `msgflux.dotdict` instance.
        TypeError:
            If modules is not a Mapping.
        ValueError:
            If a module is not found, if the DSL syntax is invalid,
            or if a condition cannot be parsed.
        RuntimeError:
            If a while loop exceeds the maximum iteration limit
            (prevents infinite loops).

    Examples:
        from msgflux import dotdict
        import msgflux.nn.functional as F

        async def prep(msg: dotdict) -> dotdict:
            print(f"Executing prep, current msg: {msg}")
            msg['output'] = {'agent': 'xpto', 'score': 10, 'status': 'success'}
            msg['counter'] = 0
            return msg

        async def increment(msg: dotdict) -> dotdict:
            print(f"Executing increment, current msg: {msg}")
            msg['counter'] = msg.get('counter', 0) + 1
            return msg

        async def feat_a(msg: dotdict) -> dotdict:
            print(f"Executing feat_a, current msg: {msg}")
            msg['feat_a'] = 'result_a'
            return msg

        async def feat_b(msg: dotdict) -> dotdict:
            print(f"Executing feat_b, current msg: {msg}")
            msg['feat_b'] = 'result_b'
            return msg

        async def final(msg: dotdict) -> dotdict:
            print(f"Executing final, current msg: {msg}")
            msg['final'] = 'done'
            return msg

        my_modules = {
            "prep": prep,
            "increment": increment,
            "feat_a": feat_a,
            "feat_b": feat_b,
            "final": final
        }
        input_msg = dotdict()

        # Example with while loop
        result = await F.ainline(
            "prep -> @{counter < 5}: increment; -> final",
            modules=my_modules,
            message=input_msg
        )

        # Example with nested while loop and other constructs
        result = await F.ainline(
            "prep -> @{counter < 3}: increment -> [feat_a, feat_b]; -> final",
            modules=my_modules,
            message=input_msg
        )
    """
    from msgflux.dsl.inline import ainline as _ainline  # noqa: PLC0415

    return await _ainline(expression, modules, message)


@Spans.ainstrument()
async def amap_gather(
    to_send: Callable,
    *,
    args_list: List[Tuple[Any, ...]],
    kwargs_list: Optional[List[Dict[str, Any]]] = None,
    max_retries: Optional[int] = None,
    retry_delay: float = 1.0,
    timeout: Optional[float] = None,
    store: Optional["CheckpointStore"] = None,
    run_id: Optional[str] = None,
    namespace: str = "gather",
    session_id: str = "default",
) -> Tuple[Any, ...]:
    """Async version of map_gather. Applies the `to_send` async function to each
    set of arguments in `args_list` and `kwargs_list` and collects the results.

    Args:
        to_send:
            The async callable function to be applied.
        args_list:
            Each tuple contains the positional arguments for the corresponding callable
            in `to_send`. If `None`, no positional arguments are passed unless specified
            individually by an item in `kwargs_list`.
        kwargs_list:
            Each dictionary contains the named arguments for the corresponding callable
            in `to_send`. If `None`, no named arguments are passed unless specified
            individually by an item in `args_list`.

    Returns:
        A tuple containing the results of each call to the `to_send` function.

    Raises:
        TypeError:
            If `to_send` is not callable.
        ValueError:
            If `args_list` is not a non-empty list or if `kwargs_list`
            (if provided) is not the same length as `args_list`.
    """
    if not callable(to_send):
        raise TypeError("`to_send` must be a callable object")

    if not isinstance(args_list, list) or len(args_list) == 0:
        raise ValueError("`args_list` must be a non-empty list")

    if kwargs_list is not None:
        if not isinstance(kwargs_list, list) or len(kwargs_list) != len(args_list):
            raise ValueError(
                "`kwargs_list` must be a list with the same length as `args_list`"
            )

    # Durable path
    if max_retries is not None or store is not None:
        workers = [
            (to_send, args_list[i], kwargs_list[i] if kwargs_list else {})
            for i in range(len(args_list))
        ]
        return tuple(
            await gather_durable_async(
                workers,
                timeout=timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
                store=store,
                run_id=run_id,
                namespace=namespace,
                session_id=session_id,
            )
        )

    tasks = []
    for i in range(len(args_list)):
        args = args_list[i]
        kwargs = kwargs_list[i] if kwargs_list else {}
        tasks.append(to_send(*args, **kwargs))

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            logger.error(str(response))
            results.append(TaskError(exception=response, index=i))
        else:
            results.append(response)

    return tuple(results)


@Spans.ainstrument()
async def ascatter_gather(
    to_send: List[Callable],
    args_list: Optional[List[Tuple[Any, ...]]] = None,
    kwargs_list: Optional[List[Dict[str, Any]]] = None,
    *,
    max_retries: Optional[int] = None,
    retry_delay: float = 1.0,
    timeout: Optional[float] = None,
    store: Optional["CheckpointStore"] = None,
    run_id: Optional[str] = None,
    namespace: str = "gather",
    session_id: str = "default",
) -> Tuple[Any, ...]:
    """Async version of scatter_gather. Sends different sets of arguments/kwargs
    to a list of async callables and collects the responses.

    Each callable in `to_send` receives the positional arguments of
    the corresponding `tuple` in `args_list` and the named arguments
    of the corresponding `dict` in `kwargs_list`. If `args_list` or
    `kwargs_list` are not provided (or are `None`), the corresponding
    callables will be called without positional or named arguments,
    respectively, unless an empty list (`[]`) or empty tuple (`()`)
    is provided for a specific item.

    Args:
        to_send:
            List of callable objects (e.g. async functions or `Module` instances
            with acall).
        args_list:
            Each tuple contains the positional arguments for the corresponding callable
            in `to_send`. If `None`, no positional arguments are passed unless specified
            individually by an item in `kwargs_list`.
        kwargs_list:
            Each dictionary contains the named arguments for the corresponding callable
            in `to_send`. If `None`, no named arguments are passed unless specified
            individually by an item in `args_list`.

    Returns:
        Tuple containing the responses for each callable. If an error occurs for a
        specific callable, its corresponding response in the tuple will be `None`.

    Raises:
        TypeError:
            If `to_send` is not a callable list.
        ValueError:
            If the lengths of `args_list` (if provided) or `kwargs_list`
            (if provided) do not match the length of `to_send`.
    """
    if not isinstance(to_send, list) or not all(callable(f) for f in to_send):
        raise TypeError("`to_send` must be a non-empty list of callable objects")

    # Durable path
    if max_retries is not None or store is not None:
        workers = [
            (
                to_send[i],
                args_list[i] if args_list and i < len(args_list) else (),
                kwargs_list[i] if kwargs_list and i < len(kwargs_list) else {},
            )
            for i in range(len(to_send))
        ]
        return tuple(
            await gather_durable_async(
                workers,
                timeout=timeout,
                max_retries=max_retries,
                retry_delay=retry_delay,
                store=store,
                run_id=run_id,
                namespace=namespace,
                session_id=session_id,
            )
        )

    tasks = []
    for i, f in enumerate(to_send):
        args = args_list[i] if args_list and i < len(args_list) else ()
        kwargs = kwargs_list[i] if kwargs_list and i < len(kwargs_list) else {}
        tasks.append(f(*args, **kwargs))

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    results = []
    for i, response in enumerate(responses):
        if isinstance(response, Exception):
            logger.error(str(response))
            results.append(TaskError(exception=response, index=i))
        else:
            results.append(response)

    return tuple(results)


@Spans.ainstrument()
async def amsg_bcast_gather(
    to_send: List[Callable],
    message: dotdict,
) -> dotdict:
    """Async version of msg_bcast_gather. Broadcasts a single message to multiple
    async modules and gathers the responses.

    Args:
        to_send:
            List of callable objects (e.g. async functions or `Module` instances
            with acall).
        message:
            Instance of `msgflux.dotdict` to broadcast.

    Returns:
        The original message with the module responses added.

    Raises:
        TypeError:
            If `message` is not an instance of `dotdict`, `to_send` is not a list
            of callables.

    Examples:
        async def add_feat_a(msg: dotdict) -> dotdict:
            msg['feat_a'] = 'result_a'
            return msg

        async def add_feat_b(msg: dotdict) -> dotdict:
            msg['feat_b'] = 'result_b'
            return msg

        message = dotdict()
        result = await F.amsg_bcast_gather([add_feat_a, add_feat_b], message)
        # message now contains both feat_a and feat_b
    """
    if not isinstance(message, dotdict):
        raise TypeError("`message` must be an instance of `msgflux.dotdict`")
    if not to_send or not all(isinstance(module, Callable) for module in to_send):
        raise TypeError("`to_send` must be a non-empty list of callable objects")

    tasks = []
    for f in to_send:
        # Check for acall method first, then coroutine function
        if hasattr(f, "acall"):
            tasks.append(f.acall(message))
        elif asyncio.iscoroutinefunction(f):
            tasks.append(f(message))
        else:
            # Fallback to sync call (will be executed in current event loop)
            # Wrap in coroutine
            async def _run_sync(func, msg):
                return func(msg)

            tasks.append(_run_sync(f, message))

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    for i, (f, response) in enumerate(zip(to_send, responses)):
        f_name = get_callable_name(f)
        if isinstance(response, Exception):
            logger.error(f"Error in async bcast task for `{f_name}`: {response}")
            message.setdefault("_errors", {})[f_name] = TaskError(
                exception=response, index=i
            )

    return message
