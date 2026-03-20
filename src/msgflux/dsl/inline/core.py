"""Inline DSL — parser, condition evaluator, and stateless execution engine.

Provides :class:`InlineDSL` (sync) and :class:`AsyncInlineDSL` (async) that
parse a mini DSL expression and execute it against a mapping of callables.

Modules follow the **delta pattern**: they receive a :class:`~msgflux.dotdict`
message and may return a ``dict`` delta (merged automatically) or ``None``
(legacy in-place mutation).  Deltas may include **signals** (``$break``,
``$stop``) for pipeline control flow.
"""

import asyncio
import re
from typing import Any, Callable, Dict, List, Mapping, Tuple

from msgflux.dotdict import dotdict
from msgflux.logger import logger
from msgflux.nn import functional as F

# ── Signals ──────────────────────────────────────────────────────────────────

_SIGNAL_KEYS = frozenset({"$break", "$stop"})


class InlineDSL:
    """Parses and executes a mini domain-specific language (DSL) for
    defining workflow pipelines over modules.

    Supported constructs:

    - **Sequential**: ``"a -> b -> c"``
    - **Parallel**: ``"[a, b, c]"``
    - **Conditional** (multi-branch): ``"{c1 ? a, c2 ? b, default}"``
    - **While loop**: ``"@{condition}: actions;"``

    Signals:

    - ``$break`` — exit the current while loop (or step sequence).
    - ``$stop``  — halt the entire pipeline immediately.

    Modules return a ``dict`` delta merged via :meth:`dotdict.apply`.
    Signal keys are stripped from the delta before applying.
    """

    def __init__(self, max_iterations: int = 1000):
        self.max_iterations = max_iterations
        self.patterns = {
            "arrow": r"\s*->\s*",
            "parallel": r"\[(.*?)\]",
            "conditional": r"\{([^{}]+)\}",
            "while_loop": r"@\{(.*?)\}:\s*((?:[^;]|(?:->|\[.*?\]|\{.*?\}|@\{.*?\}:\s*.*?;))+);",  # noqa: E501
            "identifier": r"[a-zA-Z_][a-zA-Z0-9_]*",
            "comparison": r"([a-zA-Z0-9_.]+)\s*(==|!=|<=|>=|<|>|is not|is)\s*(.*)",
        }
        self._re_while = re.compile(self.patterns["while_loop"])
        self._re_arrow = re.compile(self.patterns["arrow"])
        self._re_conditional = re.compile(self.patterns["conditional"])
        self._re_parallel = re.compile(self.patterns["parallel"])
        self._re_identifier = re.compile(self.patterns["identifier"])
        self._re_comparison = re.compile(self.patterns["comparison"])

    # ── Parser ───────────────────────────────────────────────────────────

    def parse(self, expression: str) -> List[Dict[str, Any]]:
        """Parse a DSL expression into a list of execution steps."""
        steps: List[Dict[str, Any]] = []
        remaining = expression.strip()

        while remaining:
            while_match = self._re_while.match(remaining)
            if while_match:
                condition = while_match.group(1).strip()
                actions = while_match.group(2).strip()
                steps.append(
                    {"type": "while", "condition": condition, "actions": actions}
                )
                remaining = remaining[while_match.end() :].strip()
                if remaining.startswith("->"):
                    remaining = remaining[2:].strip()
                continue

            arrow_split = self._re_arrow.split(remaining, maxsplit=1)
            part = arrow_split[0].strip()
            remaining = arrow_split[1].strip() if len(arrow_split) > 1 else ""

            if not part:
                continue

            conditional_match = self._re_conditional.match(part)
            if conditional_match:
                content = conditional_match.group(1)
                steps.append(self._parse_conditional_content(content))
                continue

            parallel_match = self._re_parallel.match(part)
            if parallel_match:
                modules = [m.strip() for m in parallel_match.group(1).split(",")]
                steps.append({"type": "parallel", "modules": modules})
                continue

            if self._re_identifier.match(part):
                steps.append({"type": "module", "module": part})
            else:
                raise ValueError(f"Invalid DSL syntax or unknown module: {part}")

        return steps

    @staticmethod
    def _parse_conditional_content(content: str) -> Dict[str, Any]:
        """Parse multi-branch conditional content.

        Supports:
            ``{c1 ? a, c2 ? b, default}``  — multi-branch with fallback
            ``{c ? a, b}``                  — binary (backwards compatible)
            ``{c1 ? a, _ ? b}``             — explicit wildcard default
        """
        parts = [p.strip() for p in content.split(",")]
        branches: List[Dict[str, str]] = []
        default: List[str] = []

        for part in parts:
            if "?" in part:
                q_idx = part.index("?")
                condition = part[:q_idx].strip()
                module = part[q_idx + 1 :].strip()
                if condition == "_":
                    default = [module]
                else:
                    branches.append({"condition": condition, "module": module})
            else:
                default.append(part)

        return {"type": "conditional", "branches": branches, "default": default}

    # ── Condition evaluator ──────────────────────────────────────────────

    def _tokenize_condition(self, condition: str) -> List[str]:
        condition = condition.strip()
        token_pattern = r"(\|\||&|!|\(|\)|[^&|!()]+)"  # noqa: S105
        tokens = re.findall(token_pattern, condition)
        return [token.strip() for token in tokens if token.strip()]

    def _parse_or_expression(self, tokens: List[str], index: int = 0) -> Tuple:
        left, index = self._parse_and_expression(tokens, index)
        while index < len(tokens) and tokens[index] == "||":
            index += 1
            right, index = self._parse_and_expression(tokens, index)
            left = ("OR", left, right)
        return left, index

    def _parse_and_expression(self, tokens: List[str], index: int) -> Tuple:
        left, index = self._parse_not_expression(tokens, index)
        while index < len(tokens) and tokens[index] == "&":
            index += 1
            right, index = self._parse_not_expression(tokens, index)
            left = ("AND", left, right)
        return left, index

    def _parse_not_expression(self, tokens: List[str], index: int) -> Tuple:
        if index < len(tokens) and tokens[index] == "!":
            index += 1
            expr, index = self._parse_primary_expression(tokens, index)
            return ("NOT", expr), index
        return self._parse_primary_expression(tokens, index)

    def _parse_primary_expression(self, tokens: List[str], index: int) -> Tuple:
        if index < len(tokens) and tokens[index] == "(":
            index += 1
            expr, index = self._parse_or_expression(tokens, index)
            if index < len(tokens) and tokens[index] == ")":
                index += 1
                return expr, index
            raise ValueError("Missing closing parenthesis")
        if index < len(tokens):
            comparison_expr = tokens[index]
            index += 1
            return ("COMPARISON", comparison_expr), index
        raise ValueError("Expected comparison expression")

    def _evaluate_comparison(self, comparison_str: str, message: dotdict) -> bool:  # noqa: C901
        match = self._re_comparison.match(comparison_str.strip())
        if not match:
            raise ValueError(
                f"Invalid condition format: {comparison_str}. "
                "Expected key_path operator value."
            )

        key_path, operator, expected_value_str = match.groups()
        actual_value = message.get(key_path, None)
        expected_value_str = expected_value_str.strip()

        if operator in ["is", "is not"]:
            if expected_value_str.lower() in ["none", "null"]:
                if operator == "is":
                    return actual_value is None
                return actual_value is not None
            raise ValueError(
                "`is` and `is not` operators only support `None` or `null` comparisons"
            )

        expected_value_str = expected_value_str.strip("'\"")

        try:
            if expected_value_str.lower() == "true":
                expected_value = True
            elif expected_value_str.lower() == "false":
                expected_value = False
            elif "." in expected_value_str:
                expected_value = float(expected_value_str)
            else:
                try:
                    expected_value = int(expected_value_str)
                except ValueError:
                    expected_value = expected_value_str

            if actual_value is None:
                return False

            if isinstance(expected_value, bool):
                if isinstance(actual_value, str):
                    actual_value = actual_value.lower() == "true"
                else:
                    actual_value = bool(actual_value)
            elif isinstance(expected_value, (int, float)):
                actual_value = type(expected_value)(actual_value)

        except (ValueError, TypeError):
            expected_value = expected_value_str
            actual_value = str(actual_value) if actual_value is not None else None

        if actual_value is None and operator not in ["==", "!="]:
            return False

        if operator == "==":
            return actual_value == expected_value
        if operator == "!=":
            return actual_value != expected_value
        if operator == "<":
            return actual_value < expected_value
        if operator == ">":
            return actual_value > expected_value
        if operator == "<=":
            return actual_value <= expected_value
        if operator == ">=":
            return actual_value >= expected_value
        raise ValueError(f"Unknown comparison operator: {operator}")

    def _evaluate_logical_tree(self, tree: tuple, message: dotdict) -> bool:
        if tree[0] == "COMPARISON":
            return self._evaluate_comparison(tree[1], message)
        if tree[0] == "AND":
            return self._evaluate_logical_tree(
                tree[1], message
            ) and self._evaluate_logical_tree(tree[2], message)
        if tree[0] == "OR":
            return self._evaluate_logical_tree(
                tree[1], message
            ) or self._evaluate_logical_tree(tree[2], message)
        if tree[0] == "NOT":
            return not self._evaluate_logical_tree(tree[1], message)
        raise ValueError(f"Unknown logical operator: {tree[0]}")

    def _evaluate_condition(self, condition: str, message: dotdict) -> bool:
        tokens = self._tokenize_condition(condition)
        if not tokens:
            raise ValueError("Empty condition")
        tree, final_index = self._parse_or_expression(tokens)
        if final_index != len(tokens):
            raise ValueError(f"Unexpected tokens after parsing: {tokens[final_index:]}")
        return self._evaluate_logical_tree(tree, message)

    # ── Signal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _extract_signals(result: Any) -> Tuple[Any, Dict[str, Any]]:
        """Separate ``$break``/``$stop`` from a module result.

        Returns ``(delta, signals)`` where *delta* has no signal keys.
        """
        if not isinstance(result, dict):
            return result, {}
        signals = {k: v for k, v in result.items() if k in _SIGNAL_KEYS and v}
        if not signals:
            return result, {}
        delta = {k: v for k, v in result.items() if k not in _SIGNAL_KEYS}
        return delta or None, signals

    @staticmethod
    def _merge_signals(all_signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge signals from parallel execution — highest severity wins."""
        has_stop = any("$stop" in s and s["$stop"] for s in all_signals)
        if has_stop:
            return {"$stop": True}
        has_break = any("$break" in s and s["$break"] for s in all_signals)
        if has_break:
            return {"$break": True}
        return {}

    @staticmethod
    def _apply_result(message: dotdict, result: Any) -> None:
        """Apply a module delta to the message.

        ``dict`` → merged via :meth:`dotdict.apply`.
        ``None`` → no-op (legacy in-place pattern).
        """
        if isinstance(result, dict):
            message.apply(result)

    # ── Module calling ───────────────────────────────────────────────────

    def _call_module(
        self,
        module: Callable,
        message: dotdict,
    ) -> Dict[str, Any]:
        """Call *module*, apply its delta, return extracted signals."""
        result = module(message)
        delta, signals = self._extract_signals(result)
        self._apply_result(message, delta)
        return signals

    @staticmethod
    def _resolve_parallel(
        step: Dict[str, Any],
        modules: Mapping[str, Callable],
    ) -> list:
        """Resolve module callables for a parallel step."""
        parallel_modules = []
        for mod_name in step["modules"]:
            module = modules.get(mod_name)
            if not module:
                raise ValueError(
                    f"Module `{mod_name}` not found for parallel execution."
                )
            parallel_modules.append(module)
        return parallel_modules

    # ── Execution ────────────────────────────────────────────────────────

    def _execute_parallel(
        self,
        parallel_modules: List[Callable],
        message: dotdict,
    ) -> Dict[str, Any]:
        """Run modules in parallel, merge deltas, return merged signals."""
        from msgflux.exceptions import TaskError  # noqa: PLC0415

        results = F.bcast_gather(parallel_modules, message)
        errors: Dict[int, Any] = {}
        seen_keys: Dict[str, int] = {}
        all_signals: List[Dict[str, Any]] = []

        for i, result in enumerate(results):
            if isinstance(result, TaskError):
                errors[i] = result
            else:
                delta, signals = self._extract_signals(result)
                if signals:
                    all_signals.append(signals)
                if isinstance(delta, dict):
                    for key in delta:
                        if key in seen_keys:
                            logger.warning(
                                "Parallel key conflict: key '%s' written by "
                                "workers %d and %d — last write wins.",
                                key,
                                seen_keys[key],
                                i,
                            )
                        seen_keys[key] = i
                self._apply_result(message, delta)

        if errors:
            failed = ", ".join(
                f"worker {i}: {err.exception}" for i, err in errors.items()
            )
            raise RuntimeError(f"Parallel execution failed: {failed}")
        return self._merge_signals(all_signals)

    def _execute_conditional(
        self,
        step: Dict[str, Any],
        modules: Mapping[str, Callable],
        message: dotdict,
    ) -> Dict[str, Any]:
        """Execute a multi-branch conditional, return signals."""
        for branch in step["branches"]:
            if self._evaluate_condition(branch["condition"], message):
                module = modules.get(branch["module"])
                if not module:
                    raise ValueError(
                        f"Module `{branch['module']}` not found in conditional branch."
                    )
                return self._call_module(module, message)
        for module_name in step["default"]:
            module = modules.get(module_name)
            if not module:
                raise ValueError(
                    f"Module `{module_name}` not found in conditional default."
                )
            signals = self._call_module(module, message)
            if signals:
                return signals
        return {}

    def _execute_while_loop(
        self,
        condition: str,
        actions: str,
        modules: Mapping[str, Callable],
        message: dotdict,
    ) -> Tuple[dotdict, Dict[str, Any]]:
        """Execute a while loop. Returns ``(message, signals)``.

        ``$break`` is consumed here (breaks the loop).
        ``$stop`` is propagated to the caller.
        """
        iterations = 0
        actions_steps = self.parse(actions)

        while self._evaluate_condition(condition, message):
            if iterations >= self.max_iterations:
                raise RuntimeError(
                    f"While loop exceeded maximum iterations ({self.max_iterations}). "
                    f"Possible infinite loop detected. Condition: {condition}"
                )

            message, signals = self._execute_steps(actions_steps, modules, message)
            iterations += 1

            if "$stop" in signals:
                return message, signals
            if "$break" in signals:
                break

        return message, {}

    def _execute_steps(
        self,
        steps: List[Dict[str, Any]],
        modules: Mapping[str, Callable],
        message: dotdict,
    ) -> Tuple[dotdict, Dict[str, Any]]:
        """Execute a list of steps. Returns ``(message, signals)``."""
        for step in steps:
            if step["type"] == "module":
                module = modules.get(step["module"])
                if not module:
                    raise ValueError(f"Module `{step['module']}` not found.")
                signals = self._call_module(module, message)

            elif step["type"] == "parallel":
                parallel_modules = self._resolve_parallel(step, modules)
                signals = self._execute_parallel(parallel_modules, message)

            elif step["type"] == "conditional":
                signals = self._execute_conditional(step, modules, message)

            elif step["type"] == "while":
                message, signals = self._execute_while_loop(
                    step["condition"], step["actions"], modules, message
                )
            else:
                raise ValueError(f"Unknown step type: {step['type']}")

            if signals:
                return message, signals

        return message, {}

    def __call__(
        self, expression: str, modules: Mapping[str, Callable], message: dotdict
    ) -> dotdict:
        """Execute the DSL pipeline."""
        steps = self.parse(expression)
        result, _signals = self._execute_steps(steps, modules, message)
        return result


class AsyncInlineDSL(InlineDSL):
    """Async version of :class:`InlineDSL`.

    Inherits parsing and condition evaluation.  Overrides execution
    methods for ``async``/``await``.
    """

    async def _acall_module(
        self,
        module: Callable,
        message: dotdict,
    ) -> Dict[str, Any]:
        """Async call *module*, apply its delta, return signals."""
        if hasattr(module, "acall"):
            result = await module.acall(message)
        elif asyncio.iscoroutinefunction(module):
            result = await module(message)
        else:
            result = module(message)
        delta, signals = self._extract_signals(result)
        self._apply_result(message, delta)
        return signals

    async def _aexecute_parallel(
        self,
        parallel_modules: List[Callable],
        message: dotdict,
    ) -> Dict[str, Any]:
        """Async parallel execution with delta merge."""
        from msgflux.exceptions import TaskError  # noqa: PLC0415

        results = await F.ascatter_gather(
            parallel_modules,
            args_list=[(message,)] * len(parallel_modules),
        )
        errors: Dict[int, Any] = {}
        seen_keys: Dict[str, int] = {}
        all_signals: List[Dict[str, Any]] = []

        for i, result in enumerate(results):
            if isinstance(result, TaskError):
                errors[i] = result
            else:
                delta, signals = self._extract_signals(result)
                if signals:
                    all_signals.append(signals)
                if isinstance(delta, dict):
                    for key in delta:
                        if key in seen_keys:
                            logger.warning(
                                "Parallel key conflict: key '%s' written by "
                                "workers %d and %d — last write wins.",
                                key,
                                seen_keys[key],
                                i,
                            )
                        seen_keys[key] = i
                self._apply_result(message, delta)

        if errors:
            failed = ", ".join(
                f"worker {i}: {err.exception}" for i, err in errors.items()
            )
            raise RuntimeError(f"Parallel execution failed: {failed}")
        return self._merge_signals(all_signals)

    async def _aexecute_conditional(
        self,
        step: Dict[str, Any],
        modules: Mapping[str, Callable],
        message: dotdict,
    ) -> Dict[str, Any]:
        """Async execute a multi-branch conditional, return signals."""
        for branch in step["branches"]:
            if self._evaluate_condition(branch["condition"], message):
                module = modules.get(branch["module"])
                if not module:
                    raise ValueError(
                        f"Module `{branch['module']}` not found in conditional branch."
                    )
                return await self._acall_module(module, message)
        for module_name in step["default"]:
            module = modules.get(module_name)
            if not module:
                raise ValueError(
                    f"Module `{module_name}` not found in conditional default."
                )
            signals = await self._acall_module(module, message)
            if signals:
                return signals
        return {}

    async def _aexecute_while_loop(
        self,
        condition: str,
        actions: str,
        modules: Mapping[str, Callable],
        message: dotdict,
    ) -> Tuple[dotdict, Dict[str, Any]]:
        """Async while loop. ``$break`` consumed, ``$stop`` propagated."""
        iterations = 0
        actions_steps = self.parse(actions)

        while self._evaluate_condition(condition, message):
            if iterations >= self.max_iterations:
                raise RuntimeError(
                    f"While loop exceeded maximum iterations ({self.max_iterations}). "
                    f"Possible infinite loop detected. Condition: {condition}"
                )

            message, signals = await self._aexecute_steps(
                actions_steps, modules, message
            )
            iterations += 1

            if "$stop" in signals:
                return message, signals
            if "$break" in signals:
                break

        return message, {}

    async def _aexecute_steps(
        self,
        steps: List[Dict[str, Any]],
        modules: Mapping[str, Callable],
        message: dotdict,
    ) -> Tuple[dotdict, Dict[str, Any]]:
        """Async version of :meth:`_execute_steps`."""
        for step in steps:
            if step["type"] == "module":
                module = modules.get(step["module"])
                if not module:
                    raise ValueError(f"Module `{step['module']}` not found.")
                signals = await self._acall_module(module, message)

            elif step["type"] == "parallel":
                parallel_modules = self._resolve_parallel(step, modules)
                signals = await self._aexecute_parallel(parallel_modules, message)

            elif step["type"] == "conditional":
                signals = await self._aexecute_conditional(step, modules, message)

            elif step["type"] == "while":
                message, signals = await self._aexecute_while_loop(
                    step["condition"], step["actions"], modules, message
                )
            else:
                raise ValueError(f"Unknown step type: {step['type']}")

            if signals:
                return message, signals

        return message, {}

    async def __call__(
        self, expression: str, modules: Mapping[str, Callable], message: dotdict
    ) -> dotdict:
        """Async execute the DSL pipeline."""
        steps = self.parse(expression)
        result, _signals = await self._aexecute_steps(steps, modules, message)
        return result
