from __future__ import annotations

from datetime import timedelta
from itertools import product

from .models import (
    CandidatePlan,
    Equipment,
    InfeasibilityReason,
    ReasonDetail,
    ResourceReservation,
    SchedulingResult,
    TaskRequirement,
    Technician,
    TimeInterval,
    Vehicle,
    Workstation,
)


class FirstFitPlanner:
    """Simple deterministic planner: earliest start, then input order of resources."""

    def __init__(
        self,
        vehicle: Vehicle,
        technicians: list[Technician],
        workstations: list[Workstation],
        equipment: list[Equipment],
        reservations: list[ResourceReservation] | None = None,
    ) -> None:
        self.vehicle = vehicle
        self.technicians = technicians
        self.workstations = workstations
        self.equipment = equipment
        self.reservations = reservations or []

    def plan(self, task: TaskRequirement) -> SchedulingResult:
        reasons: list[ReasonDetail] = []
        if (
            task.brand != self.vehicle.brand
            or task.store_id != self.vehicle.store_id
            or task.vehicle_profile != self.vehicle.profile
        ):
            return SchedulingResult(
                status="INFEASIBLE",
                reasons=[ReasonDetail(
                    code=InfeasibilityReason.SCOPE_MISMATCH,
                    message="任务与当前门店、品牌或车辆分类不匹配",
                )],
            )

        duration = timedelta(minutes=task.duration_minutes)
        if task.earliest_start + duration > task.latest_end:
            return SchedulingResult(
                status="INFEASIBLE",
                reasons=[ReasonDetail(
                    code=InfeasibilityReason.WINDOW_TOO_SHORT,
                    message="任务时长超过客户可接受时间窗口",
                )],
            )

        saw_technician = False
        saw_workstation = False
        saw_equipment = not task.required_equipment_types
        cursor = task.earliest_start
        while cursor + duration <= task.latest_end:
            interval = TimeInterval(start=cursor, end=cursor + duration)
            technicians = self._technicians_for(task, interval)
            workstations = self._workstations_for(task, interval)
            equipment_choices = self._equipment_choices(task, interval)
            saw_technician |= bool(technicians)
            saw_workstation |= bool(workstations)
            saw_equipment |= bool(equipment_choices)

            for technician in technicians:
                for workstation in workstations:
                    for chosen in equipment_choices:
                        if (
                            self._free("technician", technician.id, interval)
                            and self._free("workstation", workstation.id, interval)
                            and all(self._free("equipment", item.id, interval) for item in chosen)
                        ):
                            return SchedulingResult(
                                status="FEASIBLE",
                                plan=CandidatePlan(
                                    task_id=task.task_id,
                                    interval=interval,
                                    technician_id=technician.id,
                                    workstation_id=workstation.id,
                                    equipment_ids=[item.id for item in chosen],
                                ),
                            )
            cursor += timedelta(minutes=1)

        if not saw_technician:
            reasons.append(ReasonDetail(
                code=InfeasibilityReason.NO_QUALIFIED_TECHNICIAN,
                message="时间窗口内没有具备所需有效能力的工程师",
            ))
        if not saw_workstation:
            reasons.append(ReasonDetail(
                code=InfeasibilityReason.NO_COMPATIBLE_WORKSTATION,
                message="时间窗口内没有匹配且可用的工位",
            ))
        if not saw_equipment:
            reasons.append(ReasonDetail(
                code=InfeasibilityReason.NO_REQUIRED_EQUIPMENT,
                message="时间窗口内没有满足要求的设备",
            ))
        if not reasons:
            reasons = [
                ReasonDetail(
                    code=InfeasibilityReason.RESOURCE_CONFLICT,
                    message="工程师、工位、设备或车辆的既有预留造成时间冲突",
                ),
                ReasonDetail(
                    code=InfeasibilityReason.NO_AVAILABLE_TIME,
                    message="时间窗口内没有完整可执行的资源组合",
                ),
            ]
        return SchedulingResult(status="INFEASIBLE", reasons=reasons)

    def _technicians_for(self, task: TaskRequirement, interval: TimeInterval) -> list[Technician]:
        return [
            item
            for item in self.technicians
            if item.store_id == task.store_id
            and (task.allowed_technician_ids is None or item.id in task.allowed_technician_ids)
            and any(self._inside(interval, available) for available in item.availability)
            and item.can_do(task.required_skills, interval)
        ]

    def _workstations_for(self, task: TaskRequirement, interval: TimeInterval) -> list[Workstation]:
        return [
            item
            for item in self.workstations
            if item.store_id == task.store_id
            and (
                not task.required_workstation_types
                or item.workstation_type in task.required_workstation_types
            )
            and any(self._inside(interval, available) for available in item.availability)
        ]

    def _equipment_choices(
        self, task: TaskRequirement, interval: TimeInterval
    ) -> list[tuple[Equipment, ...]]:
        choices: list[list[Equipment]] = []
        for equipment_type in sorted(task.required_equipment_types):
            choices.append(
                [
                    item
                    for item in self.equipment
                    if item.store_id == task.store_id
                    and item.equipment_type == equipment_type
                    and any(self._inside(interval, available) for available in item.availability)
                ]
            )
        if not choices:
            return [()]
        return [
            choice
            for choice in product(*choices)
            if len({item.id for item in choice}) == len(choice)
        ]

    @staticmethod
    def _inside(target: TimeInterval, available: TimeInterval) -> bool:
        return available.start <= target.start and target.end <= available.end

    def _free(self, resource_type: str, resource_id: str, interval: TimeInterval) -> bool:
        """Return whether this concrete technician, workstation, or equipment is free."""
        return not any(
            reservation.resource_type == resource_type
            and reservation.resource_id == resource_id
            and reservation.interval.overlaps(interval)
            for reservation in self.reservations
        )

