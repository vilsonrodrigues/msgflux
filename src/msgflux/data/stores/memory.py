"""In-memory checkpoint store for development and testing."""

import time
from typing import Dict, List, Optional

from msgflux.data.stores.base import CheckpointStore
from msgflux.data.stores.schemas import RunState, StepEvent


class MemoryCheckpointStore(CheckpointStore):
    """In-memory checkpoint store for development and testing."""

    def __init__(self):
        self._runs: Dict[str, RunState] = {}

    def save_event(self, run_id: str, event: StepEvent) -> None:
        now = time.time()
        if run_id not in self._runs:
            self._runs[run_id] = RunState(
                run_id=run_id,
                expression="",
                events=[],
                created_at=now,
                updated_at=now,
            )
        self._runs[run_id].events.append(event)
        self._runs[run_id].updated_at = now

    def load_run(self, run_id: str) -> Optional[RunState]:
        return self._runs.get(run_id)

    def delete_run(self, run_id: str) -> None:
        self._runs.pop(run_id, None)

    def list_runs(self) -> List[str]:
        return list(self._runs.keys())
