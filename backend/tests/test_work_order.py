from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from autoflow_scheduling.agent_tools import (
    CUSTOMER_AGENT_TOOLS,
    STORE_TOOLS,
    customer_agent_can_use,
)
from autoflow_scheduling.database import create_session_factory
from autoflow_scheduling.db_models import ResourceReservationRow, WorkOrderEventRow, WorkOrderRow
from autoflow_scheduling.db_planner import build_profile_planner
from autoflow_scheduling.models import TaskRequirement, VehicleProfile
from autoflow_scheduling.seed import STORE_ID, seed_demo
from autoflow_scheduling.work_order import WorkOrderStatus
from autoflow_scheduling.work_order_exceptions import (
    ConfirmationRequired,
    CustomerOperationForbidden,
    InvalidConfirmationToken,
    ReservationConflict,
)
from autoflow_scheduling.work_order_service import WorkOrderService

UTC = timezone.utc
DAY = datetime(2026, 1, 1, 0, tzinfo=UTC)


def session_factory():
    return create_session_factory("sqlite+pysqlite:///:memory:")


def plan_and_requirement(session, task_id="task-1", start_hour=9):
    planner = build_profile_planner(session, STORE_ID, VehicleProfile(), DAY)
    requirement = TaskRequirement(
        task_id=task_id,
        brand="volkswagen",
        store_id=STORE_ID,
        duration_minutes=60,
        earliest_start=datetime(2026, 1, 1, start_hour, tzinfo=UTC),
        latest_end=datetime(2026, 1, 1, 17, tzinfo=UTC),
        required_skills={"engine-diagnosis"},
        required_workstation_types={"diagnostic"},
        required_equipment_types={"obd-scanner"},
    )
    result = planner.plan(requirement)
    assert result.plan is not None
    return result.plan, requirement


def test_customer_agent_tool_boundary_excludes_store_actions():
    assert CUSTOMER_AGENT_TOOLS == {
        "get_work_order_status",
        "create_work_order",
        "cancel_work_order",
        "reschedule_work_order",
    }
    assert STORE_TOOLS == {"check_in_work_order", "complete_work_order"}
    assert not customer_agent_can_use("check_in_work_order")
    assert not customer_agent_can_use("complete_work_order")
    assert customer_agent_can_use("cancel_work_order")


def test_create_is_confirmed_idempotent_and_atomic():
    factory = session_factory()
    with factory() as session:
        seed_demo(session)
        plan, requirement = plan_and_requirement(session)
        service = WorkOrderService(session)

        with pytest.raises(ConfirmationRequired):
            service.create_from_confirmed_plan(
                plan,
                requirement,
                confirmation_token="",
                expected_confirmation_token="ok",
                idempotency_key="k1",
            )

        order = service.create_from_confirmed_plan(
            plan,
            requirement,
            confirmation_token="ok",
            expected_confirmation_token="ok",
            idempotency_key="k1",
            ai_service_summary="建议进行发动机故障诊断。",
        )
        assert order.status == WorkOrderStatus.CONFIRMED
        assert order.order_no.startswith("WO-")
        assert len(order.reservations) == 3
        assert len(order.events) == 1

        same = service.create_from_confirmed_plan(
            plan,
            requirement,
            confirmation_token="ok",
            expected_confirmation_token="ok",
            idempotency_key="k1",
        )
        assert same.id == order.id
        assert (
            session.scalar(select(WorkOrderRow.id).where(WorkOrderRow.id == order.id))
            == order.id
        )
        assert session.scalar(select(WorkOrderRow).where(WorkOrderRow.id != order.id)) is None


def test_confirmed_plan_persists_only_token_hash_and_is_consumed_once():
    factory = session_factory()
    with factory() as session:
        seed_demo(session)
        plan, requirement = plan_and_requirement(session)
        service = WorkOrderService(session)
        confirmed = service.create_confirmed_plan(
            plan, requirement, confirmation_token="secret"
        )
        assert confirmed.confirmation_token_hash != "secret"
        with pytest.raises(InvalidConfirmationToken):
            service.create_from_confirmed_plan_id(
                confirmed.id,
                confirmation_token="wrong",
                idempotency_key="bad-token",
            )
        order = service.create_from_confirmed_plan_id(
            confirmed.id,
            confirmation_token="secret",
            idempotency_key="persisted-1",
        )
        assert order.status == WorkOrderStatus.CONFIRMED
        with pytest.raises(InvalidConfirmationToken):
            service.create_from_confirmed_plan_id(
                confirmed.id,
                confirmation_token="secret",
                idempotency_key="persisted-2",
            )


def test_conflict_does_not_create_work_order():
    factory = session_factory()
    with factory() as session:
        seed_demo(session)
        plan, requirement = plan_and_requirement(session)
        second_plan, second_requirement = plan_and_requirement(session, task_id="task-2")
        service = WorkOrderService(session)
        service.create_from_confirmed_plan(
            plan,
            requirement,
            confirmation_token="ok",
            expected_confirmation_token="ok",
            idempotency_key="first",
        )
        with pytest.raises(ReservationConflict):
            service.create_from_confirmed_plan(
                second_plan,
                second_requirement,
                confirmation_token="ok",
                expected_confirmation_token="ok",
                idempotency_key="second",
            )
        assert (
            session.scalar(
                select(WorkOrderRow).where(WorkOrderRow.idempotency_key == "second")
            )
            is None
        )


def test_customer_cancel_releases_resources_and_store_executes_only_valid_transitions():
    factory = session_factory()
    with factory() as session:
        seed_demo(session)
        plan, requirement = plan_and_requirement(session)
        service = WorkOrderService(session)
        order = service.create_from_confirmed_plan(
            plan,
            requirement,
            confirmation_token="ok",
            expected_confirmation_token="ok",
            idempotency_key="cancel-me",
        )

        with pytest.raises(CustomerOperationForbidden):
            service.check_in(order.order_no, actor_role="CUSTOMER_AGENT")
        order = service.check_in(order.order_no)
        assert order.status == WorkOrderStatus.CHECKED_IN
        order = service.complete(order.order_no)
        assert order.status == WorkOrderStatus.COMPLETED
        assert all(item.status == "COMPLETED" for item in order.reservations)
        assert session.scalar(
            select(WorkOrderEventRow).where(WorkOrderEventRow.work_order_id == order.id)
        ) is not None

        with pytest.raises(ValueError):
            service.cancel(
                order.order_no,
                confirmation_token="ok",
                expected_confirmation_token="ok",
            )


def test_cancelled_confirmed_order_releases_active_reservations():
    factory = session_factory()
    with factory() as session:
        seed_demo(session)
        plan, requirement = plan_and_requirement(session)
        service = WorkOrderService(session)
        order = service.create_from_confirmed_plan(
            plan,
            requirement,
            confirmation_token="ok",
            expected_confirmation_token="ok",
            idempotency_key="cancel",
        )
        service.cancel(order.order_no, confirmation_token="ok", expected_confirmation_token="ok")
        reservations = session.scalars(
            select(ResourceReservationRow).where(ResourceReservationRow.work_order_id == order.id)
        ).all()
        assert all(item.status == "CANCELLED" for item in reservations)
        assert (
            session.scalar(select(WorkOrderRow).where(WorkOrderRow.status == "CANCELLED"))
            is not None
        )


def test_reschedule_replaces_active_reservations():
    factory = session_factory()
    with factory() as session:
        seed_demo(session)
        plan, requirement = plan_and_requirement(session)
        service = WorkOrderService(session)
        order = service.create_from_confirmed_plan(
            plan,
            requirement,
            confirmation_token="ok",
            expected_confirmation_token="ok",
            idempotency_key="move",
        )
        new_plan, new_requirement = plan_and_requirement(
            session, task_id="task-moved", start_hour=11
        )
        updated = service.reschedule(
            order.order_no,
            new_plan,
            new_requirement,
            confirmation_token="ok",
            expected_confirmation_token="ok",
        )
        assert updated.scheduled_start == new_plan.interval.start
        assert any(item.status == "CANCELLED" for item in updated.reservations)
        assert any(item.status == "ACTIVE" for item in updated.reservations)
