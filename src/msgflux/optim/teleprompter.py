"""Base class for DSPy-compatible prompt optimizers.

All optimizers inherit from ``Teleprompter`` and implement ``compile()``,
which takes a student module and training set and returns an optimized copy.
"""

from typing import Any, List, Optional

from msgflux.examples import Example
from msgflux.nn.modules.module import Module


class Teleprompter:
    """Base class for all prompt optimizers.

    Subclasses must implement :meth:`compile`, which receives a student
    module and a training set and returns an optimized copy of the student.

    Example::

        class MyOptimizer(Teleprompter):
            def compile(self, student, *, trainset, **kwargs):
                student = student.reset_copy()
                # ... optimize ...
                student.compile_(optimizer_name=self.__class__.__name__)
                return student
    """

    def compile(
        self,
        student: Module,
        *,
        trainset: List[Example],
        teacher: Optional[Module] = None,
        valset: Optional[List[Example]] = None,
        **kwargs: Any,
    ) -> Module:
        """Optimize *student* using *trainset* and return the compiled module.

        Args:
            student: The module to optimize (will be deep-copied internally).
            trainset: Training examples.
            teacher: Optional pre-compiled teacher for bootstrapping.
            valset: Optional validation set.
            **kwargs: Optimizer-specific options.

        Returns:
            A compiled copy of *student* with optimized parameters.
        """
        raise NotImplementedError
