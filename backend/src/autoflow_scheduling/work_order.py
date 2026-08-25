from __future__ import annotations

from enum import StrEnum


class WorkOrderStatus(StrEnum):
    CONFIRMED = "CONFIRMED"
    CHECKED_IN = "CHECKED_IN"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


ALLOWED_TRANSITIONS: dict[WorkOrderStatus, set[WorkOrderStatus]] = {
    WorkOrderStatus.CONFIRMED: {
        WorkOrderStatus.CHECKED_IN,
        WorkOrderStatus.CANCELLED,
    },
    WorkOrderStatus.CHECKED_IN: {WorkOrderStatus.COMPLETED},
    WorkOrderStatus.COMPLETED: set(),
    WorkOrderStatus.CANCELLED: set(),
}


def validate_transition(current: WorkOrderStatus | str, target: WorkOrderStatus | str) -> None:
    current_status = WorkOrderStatus(current)
    target_status = WorkOrderStatus(target)
    if target_status not in ALLOWED_TRANSITIONS[current_status]:
        raise ValueError(f"invalid work order transition: {current_status} -> {target_status}")
