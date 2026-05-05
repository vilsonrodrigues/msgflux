import asyncio
import time
from dataclasses import dataclass
from typing import Optional

from msgflux.channels.exceptions import AdmissionQueueFullError


@dataclass(frozen=True)
class AdmissionSlot:
    lane: str
    _global_limiter: Optional[asyncio.Semaphore] = None
    _lane_limiter: Optional[asyncio.Semaphore] = None
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        object.__setattr__(self, "_released", True)
        if self._lane_limiter is not None:
            self._lane_limiter.release()
        if self._global_limiter is not None:
            self._global_limiter.release()


class AdmissionController:
    def __init__(
        self,
        *,
        max_concurrent: Optional[int] = None,
        chat_completion_max_concurrent: Optional[int] = None,
        social_max_concurrent: Optional[int] = None,
    ) -> None:
        self._global_limiter = _limiter(max_concurrent)
        self._lane_limiters = {
            "chat_completion": _limiter(chat_completion_max_concurrent),
            "social": _limiter(social_max_concurrent),
        }

    async def acquire(
        self,
        lane: str,
        *,
        timeout_s: Optional[float],
    ) -> AdmissionSlot:
        global_limiter = self._global_limiter
        lane_limiter = self._lane_limiters.get(lane)
        acquired_global = False
        acquired_lane = False
        deadline = None if timeout_s is None else time.monotonic() + timeout_s

        try:
            if global_limiter is not None:
                await _acquire_limiter(
                    global_limiter,
                    timeout_s=_remaining_timeout(deadline),
                )
                acquired_global = True
            if lane_limiter is not None:
                await _acquire_limiter(
                    lane_limiter,
                    timeout_s=_remaining_timeout(deadline),
                )
                acquired_lane = True
        except AdmissionQueueFullError:
            if acquired_lane and lane_limiter is not None:
                lane_limiter.release()
            if acquired_global and global_limiter is not None:
                global_limiter.release()
            raise

        return AdmissionSlot(
            lane=lane,
            _global_limiter=global_limiter if acquired_global else None,
            _lane_limiter=lane_limiter if acquired_lane else None,
        )


def _limiter(max_concurrent: Optional[int]) -> Optional[asyncio.Semaphore]:
    if max_concurrent is None:
        return None
    return asyncio.Semaphore(max_concurrent)


async def _acquire_limiter(
    limiter: asyncio.Semaphore,
    *,
    timeout_s: Optional[float],
) -> None:
    if timeout_s == 0:
        if limiter.locked():
            raise AdmissionQueueFullError(
                "Server is at capacity. Try again later.",
                retry_after_s=1,
            )
        await limiter.acquire()
        return

    try:
        if timeout_s is None:
            await limiter.acquire()
        else:
            await asyncio.wait_for(limiter.acquire(), timeout=timeout_s)
    except asyncio.TimeoutError as e:
        raise AdmissionQueueFullError(
            "Server admission queue is full. Try again later.",
            retry_after_s=1,
        ) from e


def _remaining_timeout(deadline: Optional[float]) -> Optional[float]:
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())
