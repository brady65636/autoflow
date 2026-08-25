from datetime import datetime, timezone

from sqlalchemy import select

from autoflow_scheduling.database import create_session_factory
from autoflow_scheduling.db_models import (
    CapabilityRow,
    ResourceReservationRow,
    TechnicianRow,
    VehicleRow,
)
from autoflow_scheduling.db_planner import build_profile_planner
from autoflow_scheduling.models import (
    PowertrainType,
    TaskRequirement,
    VehicleCategory,
    VehicleProfile,
)
from autoflow_scheduling.repository import (
    create_reservation,
    create_vehicle,
    delete_vehicle,
    get_vehicle,
    list_vehicles,
    update_vehicle,
)
from autoflow_scheduling.seed import STORE_ID, seed_demo

UTC = timezone.utc
DAY = datetime(2026, 1, 1, 0, tzinfo=UTC)


def session_factory():
    return create_session_factory("sqlite+pysqlite:///:memory:")


def test_seed_and_basic_crud():
    factory = session_factory()
    with factory() as session:
        seed_demo(session)
        assert len(list_vehicles(session)) == 2
        assert session.scalar(select(CapabilityRow).where(CapabilityRow.code == "brake"))
        technician = session.get(TechnicianRow, "tech-wang")
        assert technician is not None
        assert {link.capability.code for link in technician.capability_links} == {
            "inspection", "engine-diagnosis", "brake"
        }
        vehicle = session.get(VehicleRow, "vehicle-magotan-2021")
        assert vehicle is not None
        assert vehicle.store_id == STORE_ID

        create_vehicle(session, "vehicle-test", "volkswagen", STORE_ID)
        update_vehicle(session, "vehicle-test", brand="volkswagen-updated")
        assert get_vehicle(session, "vehicle-test").brand == "volkswagen-updated"
        delete_vehicle(session, "vehicle-test")
        assert get_vehicle(session, "vehicle-test") is None


def test_profile_based_planner_does_not_require_vehicle_row():
    factory = session_factory()
    with factory() as session:
        from autoflow_scheduling.seed import STANDARD_4S_STORE_ID, seed_standard_4s

        seed_standard_4s(session)
        planner = build_profile_planner(
            session,
            STANDARD_4S_STORE_ID,
            VehicleProfile(category=VehicleCategory.SUV, powertrain=PowertrainType.EV),
            DAY,
        )
        task = TaskRequirement(
            task_id="profile-task",
            brand="volkswagen",
            store_id=STANDARD_4S_STORE_ID,
            vehicle_profile=VehicleProfile(
                category=VehicleCategory.SUV,
                powertrain=PowertrainType.EV,
            ),
            duration_minutes=60,
            earliest_start=datetime(2026, 1, 1, 9, tzinfo=UTC),
            latest_end=datetime(2026, 1, 1, 17, tzinfo=UTC),
            required_skills={"ev-diagnosis"},
            required_workstation_types={"ev-diagnostic"},
            required_equipment_types={"ev-insulation-tester"},
        )
        plan = planner.plan(task)
        assert plan.status == "FEASIBLE"
        assert plan.plan is not None
        assert plan.plan.technician_id == "4s-tech-007"
        assert plan.plan.workstation_id == "4s-bay-ev-01"


def test_database_resources_feed_first_fit_and_reservations():
    factory = session_factory()
    with factory() as session:
        seed_demo(session)
        planner = build_profile_planner(session, STORE_ID, VehicleProfile(), DAY)
        task = TaskRequirement(
            task_id="task-db-1",
            brand="volkswagen",
            store_id=STORE_ID,
            duration_minutes=90,
            earliest_start=datetime(2026, 1, 1, 9, tzinfo=UTC),
            latest_end=datetime(2026, 1, 1, 17, tzinfo=UTC),
            required_skills={"engine-diagnosis"},
            required_workstation_types={"diagnostic"},
            required_equipment_types={"obd-scanner"},
        )
        plan = planner.plan(task)
        assert plan.status == "FEASIBLE"
        assert plan.plan is not None
        assert plan.plan.technician_id == "tech-wang"
        assert plan.plan.workstation_id == "bay-diagnostic-1"
        assert plan.plan.equipment_ids == ["equipment-obd-1"]

        create_reservation(
            session,
            "technician",
            plan.plan.technician_id,
            task.task_id,
            plan.plan.interval.start,
            plan.plan.interval.end,
        )
        create_reservation(
            session,
            "workstation",
            plan.plan.workstation_id,
            task.task_id,
            plan.plan.interval.start,
            plan.plan.interval.end,
        )
        create_reservation(
            session,
            "equipment",
            plan.plan.equipment_ids[0],
            task.task_id,
            plan.plan.interval.start,
            plan.plan.interval.end,
        )
        assert session.scalar(select(ResourceReservationRow.id).limit(1)) is not None

        next_planner = build_profile_planner(session, task.store_id, task.vehicle_profile, DAY)
        next_task = task.model_copy(update={"task_id": "task-db-2"})
        next_plan = next_planner.plan(next_task)
        assert next_plan.plan is not None
        assert plan.plan is not None
        assert next_plan.plan.interval.start == plan.plan.interval.end
