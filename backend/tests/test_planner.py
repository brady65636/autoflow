from datetime import datetime, timedelta, timezone

from autoflow_scheduling import (
    EffectiveAbility,
    Equipment,
    FirstFitPlanner,
    InfeasibilityReason,
    ResourceReservation,
    TaskRequirement,
    Technician,
    TimeInterval,
    Vehicle,
    Workstation,
)
from autoflow_scheduling.catalog import MVP_OPERATIONS

UTC = timezone.utc
BASE = datetime(2026, 1, 1, 9, tzinfo=UTC)


def span(start=BASE, minutes=60):
    return TimeInterval(start=start, end=start + timedelta(minutes=minutes))


def technician(skills=("engine",), name="t1"):
    return Technician(
        id=name,
        name=name,
        store_id="s1",
        abilities=[EffectiveAbility(skill=skill) for skill in skills],
        availability=[span(minutes=480)],
    )


def task(**changes):
    values = {
        "task_id": "task-1",
        "brand": "v",
        "store_id": "s1",
        "duration_minutes": 60,
        "earliest_start": BASE,
        "latest_end": BASE + timedelta(hours=8),
        "required_skills": {"engine"},
    }
    values.update(changes)
    return TaskRequirement(**values)


def planner(**changes):
    values = {
        "vehicle": Vehicle(id="car-1", brand="v", store_id="s1"),
        "technicians": [technician()],
        "workstations": [
            Workstation(
                id="w1",
                name="w1",
                store_id="s1",
                workstation_type="lift",
                availability=[span(minutes=480)],
            )
        ],
        "equipment": [],
    }
    values.update(changes)
    return FirstFitPlanner(**values)


def reservation(resource_type, resource_id, start):
    return ResourceReservation(
        resource_type=resource_type,
        resource_id=resource_id,
        interval=span(start, 60),
        task_id="old",
    )


def test_touching_intervals_do_not_conflict_and_overlapping_do():
    first = span(minutes=60)
    touching = TimeInterval(start=first.end, end=first.end + timedelta(minutes=10))
    overlapping = TimeInterval(
        start=first.end - timedelta(minutes=1), end=first.end + timedelta(minutes=10)
    )
    assert not first.overlaps(touching)
    assert first.overlaps(overlapping)


def test_complete_plan_assigns_technician_workstation_and_equipment():
    equipment = Equipment(
        id="e1",
        name="scanner",
        store_id="s1",
        equipment_type="scanner",
        availability=[span(minutes=480)],
    )
    result = planner(equipment=[equipment]).plan(task(required_equipment_types={"scanner"}))
    assert result.status == "FEASIBLE"
    assert result.plan is not None
    assert (result.plan.technician_id, result.plan.workstation_id, result.plan.equipment_ids) == (
        "t1", "w1", ["e1"]
    )


def test_equipment_types_are_considered_in_sorted_order():
    equipment = [
        Equipment(
            id="z1",
            name="z",
            store_id="s1",
            equipment_type="z-type",
            availability=[span(minutes=480)],
        ),
        Equipment(
            id="a1",
            name="a",
            store_id="s1",
            equipment_type="a-type",
            availability=[span(minutes=480)],
        ),
    ]
    result = planner(equipment=equipment).plan(
        task(required_equipment_types={"z-type", "a-type"})
    )
    assert result.plan is not None
    assert result.plan.equipment_ids == ["a1", "z1"]


def test_capability_and_allowed_technician_filter():
    missing = planner().plan(task(required_skills={"missing"}))
    restricted = planner().plan(task(allowed_technician_ids={"someone-else"}))
    assert missing.reasons[0].code == InfeasibilityReason.NO_QUALIFIED_TECHNICIAN
    assert restricted.reasons[0].code == InfeasibilityReason.NO_QUALIFIED_TECHNICIAN


def test_workstation_conflict_is_rejected_independently():
    equipment = Equipment(
        id="e1", name="scanner", store_id="s1", equipment_type="scanner",
        availability=[span(minutes=480)],
    )
    reservations = [reservation("workstation", "w1", BASE)]
    result = planner(equipment=[equipment], reservations=reservations).plan(
        task(required_equipment_types={"scanner"}, latest_end=BASE + timedelta(hours=1))
    )
    assert result.reasons[0].code == InfeasibilityReason.RESOURCE_CONFLICT


def test_equipment_conflict_is_rejected_independently():
    equipment = Equipment(
        id="e1", name="scanner", store_id="s1", equipment_type="scanner",
        availability=[span(minutes=480)],
    )
    reservations = [reservation("equipment", "e1", BASE)]
    result = planner(equipment=[equipment], reservations=reservations).plan(
        task(required_equipment_types={"scanner"}, latest_end=BASE + timedelta(hours=1))
    )
    assert result.reasons[0].code == InfeasibilityReason.RESOURCE_CONFLICT


def test_resource_conflict_can_fit_after_existing_reservation():
    existing = reservation("technician", "t1", BASE)
    result = planner(reservations=[existing]).plan(task(latest_end=BASE + timedelta(hours=9)))
    assert result.plan is not None
    assert result.plan.interval.start == BASE + timedelta(hours=1)


def test_structured_infeasibility_explains_missing_capability():
    result = planner().plan(task(required_skills={"missing"}))
    assert result.status == "INFEASIBLE"
    assert result.plan is None
    assert result.reasons[0].code == InfeasibilityReason.NO_QUALIFIED_TECHNICIAN


def test_structured_infeasibility_explains_resource_conflict():
    result = planner(
        reservations=[reservation("workstation", "w1", BASE)]
    ).plan(task(latest_end=BASE + timedelta(hours=1)))
    assert result.status == "INFEASIBLE"
    assert {reason.code for reason in result.reasons} == {
        InfeasibilityReason.RESOURCE_CONFLICT,
        InfeasibilityReason.NO_AVAILABLE_TIME,
    }


def test_structured_infeasibility_explains_short_window():
    result = planner().plan(
        task(duration_minutes=61, latest_end=BASE + timedelta(hours=1))
    )
    assert result.reasons[0].code == InfeasibilityReason.WINDOW_TOO_SHORT


def test_no_plan_when_window_is_too_short():
    result = planner().plan(task(duration_minutes=61, latest_end=BASE + timedelta(hours=1)))
    assert result.reasons[0].code == InfeasibilityReason.WINDOW_TOO_SHORT


def test_scope_mismatch_is_rejected():
    brand_result = planner().plan(task(brand="other-brand"))
    assert brand_result.reasons[0].code == InfeasibilityReason.SCOPE_MISMATCH


def test_catalog_contains_eight_mvp_operations():
    assert len(MVP_OPERATIONS) == 8
    assert {operation.name for operation in MVP_OPERATIONS} == {
        "到店诊断",
        "常规保养",
        "发动机故障灯诊断",
        "电气诊断",
        "制动检查",
        "更换刹车片",
        "四轮定位",
        "竣工质检",
    }
    electrical = next(
        operation
        for operation in MVP_OPERATIONS
        if operation.code == "electrical-diagnosis"
    )
    assert electrical.required_equipment_types == {"oem-diagnostic-tool"}
