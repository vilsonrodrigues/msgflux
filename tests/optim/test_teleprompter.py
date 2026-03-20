"""Tests for Teleprompter base class."""

import pytest

from msgflux.optim.teleprompter import Teleprompter


class TestTeleprompter:
    def test_compile_raises_not_implemented(self):
        tp = Teleprompter()
        with pytest.raises(NotImplementedError):
            tp.compile(None, trainset=[])
