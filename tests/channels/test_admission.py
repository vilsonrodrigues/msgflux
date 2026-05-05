import pytest

from msgflux.channels import AdmissionController, AdmissionQueueFullError


@pytest.mark.asyncio
async def test_admission_controller_enforces_global_limit():
    controller = AdmissionController(max_concurrent=1)
    slot = await controller.acquire("chat_completion", timeout_s=0)

    with pytest.raises(AdmissionQueueFullError):
        await controller.acquire("social", timeout_s=0)

    slot.release()
    social_slot = await controller.acquire("social", timeout_s=0)
    social_slot.release()


@pytest.mark.asyncio
async def test_admission_controller_enforces_lane_limit():
    controller = AdmissionController(social_max_concurrent=1)
    slot = await controller.acquire("social", timeout_s=0)

    with pytest.raises(AdmissionQueueFullError):
        await controller.acquire("social", timeout_s=0)

    chat_slot = await controller.acquire("chat_completion", timeout_s=0)
    chat_slot.release()
    slot.release()
