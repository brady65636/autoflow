from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from .db_models import (
    EquipmentRow,
    ResourceReservationRow,
    TechnicianRow,
    WorkstationRow,
)
from .models import (
    EffectiveAbility,
    Equipment,
    ResourceReservation,
    Technician,
    TimeInterval,
    Vehicle,
    VehicleProfile,
    Workstation,
)
from .planner import FirstFitPlanner

STORE_OPEN = 8
STORE_CLOSE = 18


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _availability(start: datetime) -> list[TimeInterval]:
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    opening = start.replace(hour=STORE_OPEN, minute=0, second=0, microsecond=0)
    closing = start.replace(hour=STORE_CLOSE, minute=0, second=0, microsecond=0)
    return [TimeInterval(start=opening, end=closing)]


def _build_for_store(
    session: Session,
    store_id: str,
    profile: VehicleProfile,
    planning_day: datetime,
    brand: str,
) -> FirstFitPlanner:
    availability = _availability(planning_day)
    technicians = [
        Technician(
            id=row.id,
            name=row.name,
            store_id=row.store_id,
            abilities=[
                EffectiveAbility(
                    skill=link.capability.code,
                    valid_from=link.valid_from,
                    valid_until=link.valid_until,
                )
                for link in row.capability_links
            ],
            availability=availability,
        )
        for row in session.query(TechnicianRow).filter_by(store_id=store_id).all()
    ]
    workstations = [
        Workstation(
            id=row.id,
            name=row.name,
            store_id=row.store_id,
            workstation_type=row.workstation_type,
            availability=availability,
        )
        for row in session.query(WorkstationRow).filter_by(store_id=store_id).all()
    ]
    equipment = [
        Equipment(
            id=row.id,
            name=row.name,
            store_id=row.store_id,
            equipment_type=row.equipment_type,
            availability=availability,
        )
        for row in session.query(EquipmentRow).filter_by(store_id=store_id).all()
    ]
    reservations = [
        ResourceReservation(
            resource_type=row.resource_type,
            resource_id=row.resource_id,
            task_id=row.task_id,
            interval=TimeInterval(
                start=_as_utc(row.start_time),
                end=_as_utc(row.end_time),
            ),
        )
        for row in session.query(ResourceReservationRow).filter_by(status="ACTIVE").all()
    ]
    return FirstFitPlanner(
        vehicle=Vehicle(
            id="profile-context",
            brand=brand,
            store_id=store_id,
            profile=profile,
        ),
        technicians=technicians,
        workstations=workstations,
        equipment=equipment,
        reservations=reservations,
    )


def build_profile_planner(
    session: Session,
    store_id: str,
    profile: VehicleProfile,
    planning_day: datetime,
    brand: str = "volkswagen",
) -> FirstFitPlanner:
    """Build a planner from vehicle classification without a customer vehicle row."""
    return _build_for_store(
        session,
        store_id,
        profile,
        planning_day,
        brand=brand,
    )
