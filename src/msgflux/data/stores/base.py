"""Abstract base classes for checkpoint stores."""

from abc import ABC, abstractmethod
from typing import List, Optional

from msgflux.data.stores.schemas import StepEvent, StepStatus


class CheckpointStore(ABC):
    """Abstract checkpoint store for durable pipeline execution."""

    @abstractmethod
    def save_event(self, run_id: str, event: StepEvent) -> None:
        """Append a step event for a run."""

    @abstractmethod
    def load_run(self, run_id: str):
        """Load the full run state. Returns None if not found."""

    @abstractmethod
    def delete_run(self, run_id: str) -> None:
        """Delete all state for a run."""

    @abstractmethod
    def list_runs(self) -> List[str]:
        """List all run_ids in the store."""

    def get_last_completed_step(self, run_id: str) -> Optional[StepEvent]:
        """Get the last completed step for a run."""
        run = self.load_run(run_id)
        if run is None:
            return None
        completed = [e for e in run.events if e.status == StepStatus.COMPLETED]
        return completed[-1] if completed else None

    def get_step_status(self, run_id: str, step_name: str) -> Optional[str]:
        """Get the most recent status of a specific step."""
        run = self.load_run(run_id)
        if run is None:
            return None
        for event in reversed(run.events):
            if event.step_name == step_name:
                return event.status
        return None

    def get_step_retry_count(self, run_id: str, step_name: str) -> int:
        """Get the retry count of a specific step."""
        run = self.load_run(run_id)
        if run is None:
            return 0
        for event in reversed(run.events):
            if event.step_name == step_name and event.status == StepStatus.FAILED:
                return event.retry_count
        return 0


class AsyncCheckpointStore(ABC):
    """Abstract async checkpoint store for durable pipeline execution."""

    @abstractmethod
    async def asave_event(self, run_id: str, event: StepEvent) -> None:
        """Append a step event for a run (async)."""

    @abstractmethod
    async def aload_run(self, run_id: str):
        """Load the full run state (async). Returns None if not found."""

    @abstractmethod
    async def adelete_run(self, run_id: str) -> None:
        """Delete all state for a run (async)."""

    @abstractmethod
    async def alist_runs(self) -> List[str]:
        """List all run_ids in the store (async)."""

    async def aget_last_completed_step(self, run_id: str) -> Optional[StepEvent]:
        """Get the last completed step for a run (async)."""
        run = await self.aload_run(run_id)
        if run is None:
            return None
        completed = [e for e in run.events if e.status == StepStatus.COMPLETED]
        return completed[-1] if completed else None

    async def aget_step_status(self, run_id: str, step_name: str) -> Optional[str]:
        """Get the most recent status of a specific step (async)."""
        run = await self.aload_run(run_id)
        if run is None:
            return None
        for event in reversed(run.events):
            if event.step_name == step_name:
                return event.status
        return None

    async def aget_step_retry_count(self, run_id: str, step_name: str) -> int:
        """Get the retry count of a specific step (async)."""
        run = await self.aload_run(run_id)
        if run is None:
            return 0
        for event in reversed(run.events):
            if event.step_name == step_name and event.status == StepStatus.FAILED:
                return event.retry_count
        return 0
