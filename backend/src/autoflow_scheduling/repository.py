from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db_models import (
    CapabilityRow,
    EquipmentRow,
    ResourceReservationRow,
    TechnicianCapabilityRow,
    TechnicianRow,
    UserRow,
    VehicleRow,
    WorkstationRow,
)


def create_user(
    session: Session,
    user_id: str,
    username: str,
    password_hash: str,
    role: str,
) -> UserRow:
    row = UserRow(
        id=user_id,
        username=username,
        password_hash=password_hash,
        role=role,
        created_at=datetime.now(),
    )
    session.add(row)
    session.commit()
    return row


def get_user_by_username(session: Session, username: str) -> UserRow | None:
    return session.scalar(select(UserRow).where(UserRow.username == username))


def create_vehicle(
    session: Session,
    vehicle_id: str,
    brand: str,
    store_id: str,
    model: str | None = None,
    model_year: int | None = None,
    category: str = "sedan",
    powertrain: str = "ice",
) -> VehicleRow:
    row = VehicleRow(
        id=vehicle_id,
        brand=brand,
        model=model,
        model_year=model_year,
        category=category,
        powertrain=powertrain,
        store_id=store_id,
    )
    session.add(row)
    session.commit()
    return row


def get_vehicle(session: Session, vehicle_id: str) -> VehicleRow | None:
    return session.get(VehicleRow, vehicle_id)


def list_vehicles(session: Session) -> list[VehicleRow]:
    return list(session.scalars(select(VehicleRow).order_by(VehicleRow.id)))


def update_vehicle(session: Session, vehicle_id: str, *, brand: str) -> VehicleRow:
    row = session.get(VehicleRow, vehicle_id)
    if row is None:
        raise ValueError(f"vehicle not found: {vehicle_id}")
    row.brand = brand
    session.commit()
    return row


def delete_vehicle(session: Session, vehicle_id: str) -> None:
    row = session.get(VehicleRow, vehicle_id)
    if row is None:
        raise ValueError(f"vehicle not found: {vehicle_id}")
    session.delete(row)
    session.commit()


def list_technicians(session: Session) -> list[TechnicianRow]:
    return list(session.scalars(select(TechnicianRow).order_by(TechnicianRow.id)))


def list_workstations(session: Session) -> list[WorkstationRow]:
    return list(session.scalars(select(WorkstationRow).order_by(WorkstationRow.id)))


def list_equipment(session: Session) -> list[EquipmentRow]:
    return list(session.scalars(select(EquipmentRow).order_by(EquipmentRow.id)))


def create_capability(session: Session, code: str, name: str) -> CapabilityRow:
    row = CapabilityRow(code=code, name=name)
    session.add(row)
    session.commit()
    return row


def create_technician(
    session: Session, technician_id: str, name: str, store_id: str, skills: list[str]
) -> TechnicianRow:
    row = TechnicianRow(id=technician_id, name=name, store_id=store_id)
    row.capability_links = []
    for skill in skills:
        capability = session.scalar(select(CapabilityRow).where(CapabilityRow.code == skill))
        if capability is None:
            capability = create_capability(session, skill, skill.replace("-", " ").title())
        row.capability_links.append(
            TechnicianCapabilityRow(technician_id=technician_id, capability_id=capability.id)
        )
    session.add(row)
    session.commit()
    return row


def list_capabilities(session: Session) -> list[CapabilityRow]:
    return list(session.scalars(select(CapabilityRow).order_by(CapabilityRow.code)))


def create_workstation(
    session: Session, workstation_id: str, name: str, store_id: str, workstation_type: str
) -> WorkstationRow:
    row = WorkstationRow(
        id=workstation_id, name=name, store_id=store_id, workstation_type=workstation_type
    )
    session.add(row)
    session.commit()
    return row


def create_equipment(
    session: Session, equipment_id: str, name: str, store_id: str, equipment_type: str
) -> EquipmentRow:
    row = EquipmentRow(
        id=equipment_id, name=name, store_id=store_id, equipment_type=equipment_type
    )
    session.add(row)
    session.commit()
    return row


def list_reservations(session: Session) -> list[ResourceReservationRow]:
    return list(session.scalars(select(ResourceReservationRow)))


def create_reservation(
    session: Session,
    resource_type: str,
    resource_id: str,
    task_id: str,
    start_time: datetime,
    end_time: datetime,
    work_order_id: str | None = None,
) -> ResourceReservationRow:
    row = ResourceReservationRow(
        resource_type=resource_type,
        resource_id=resource_id,
        task_id=task_id,
        work_order_id=work_order_id,
        start_time=start_time,
        end_time=end_time,
    )
    session.add(row)
    session.flush()
    return row
