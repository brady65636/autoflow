from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from .db_models import (
    ConfirmedPlanRow,
    ResourceReservationRow,
    WorkOrderEventRow,
    WorkOrderRow,
)
from .models import CandidatePlan, TaskRequirement
from .work_order import WorkOrderStatus, validate_transition
from .work_order_exceptions import (
    ConfirmationRequired,
    CustomerOperationForbidden,
    InvalidConfirmationToken,
    ReservationConflict,
    WorkOrderNotFound,
)

ACTIVE = "ACTIVE"
CANCELLED = "CANCELLED"
COMPLETED = "COMPLETED"


class WorkOrderService:
    """Application service for confirmed schedules and store execution actions."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_confirmed_plan(
        self,
        plan: CandidatePlan,
        requirement: TaskRequirement,
        *,
        confirmation_token: str,
        expires_at: datetime | None = None,
        customer_user_id: str | None = None,
    ) -> ConfirmedPlanRow:
        if not confirmation_token:
            raise ConfirmationRequired("customer confirmation is required")
        plan_id = f"plan_{uuid4().hex}"
        row = ConfirmedPlanRow(
            id=plan_id,
            plan_json=plan.model_dump_json(),
            requirement_json=requirement.model_dump_json(),
            confirmation_token_hash=_token_hash(confirmation_token),
            customer_user_id=customer_user_id,
            root_plan_id=plan_id,
            status="CONFIRMED",
            confirmed_at=_now(),
            expires_at=expires_at,
            created_at=_now(),
        )
        self.session.add(row)
        self.session.commit()
        return row

    def create_from_confirmed_plan_id(
        self,
        confirmed_plan_id: str,
        *,
        confirmation_token: str,
        idempotency_key: str,
        customer_note: str = "",
        ai_service_summary: str = "",
    ) -> WorkOrderRow:
        if not confirmation_token:
            raise ConfirmationRequired("customer confirmation is required")
        try:
            self._begin_write()
            existing = self._find_idempotent(idempotency_key)
            if existing is not None:
                self.session.rollback()
                return existing
            confirmed = self.session.get(ConfirmedPlanRow, confirmed_plan_id)
            if confirmed is None or confirmed.status not in {"AVAILABLE", "CONFIRMED"}:
                raise InvalidConfirmationToken("confirmed plan is unavailable")
            if confirmed.expires_at and confirmed.expires_at <= _now():
                raise InvalidConfirmationToken("confirmed plan has expired")
            if confirmed.confirmation_token_hash != _token_hash(confirmation_token):
                raise InvalidConfirmationToken("confirmation token is invalid")
            plan = CandidatePlan.model_validate_json(confirmed.plan_json)
            requirement = TaskRequirement.model_validate_json(confirmed.requirement_json)
            row = self._create_order(
                plan,
                requirement,
                customer_user_id=confirmed.customer_user_id,
                plan_id=confirmed.id,
                idempotency_key=idempotency_key,
                customer_note=customer_note,
                ai_service_summary=ai_service_summary,
            )
            confirmed.status = "APPLIED"
            confirmed.work_order_id = row.id
            confirmed.applied_at = _now()
            row.current_plan_id = confirmed.id
            self.session.commit()
            return row
        except Exception:
            self.session.rollback()
            raise

    def create_from_confirmed_plan(
        self,
        plan: CandidatePlan,
        requirement: TaskRequirement,
        *,
        confirmation_token: str,
        expected_confirmation_token: str,
        idempotency_key: str,
        customer_note: str = "",
        ai_service_summary: str = "",
        customer_user_id: str | None = None,
    ) -> WorkOrderRow:
        if not confirmation_token:
            raise ConfirmationRequired("customer confirmation is required")
        if confirmation_token != expected_confirmation_token:
            raise InvalidConfirmationToken("confirmation token is invalid")
        try:
            self._begin_write()
            existing = self._find_idempotent(idempotency_key)
            if existing is not None:
                self.session.rollback()
                return existing
            row = self._create_order(
                plan,
                requirement,
                customer_user_id=customer_user_id,
                plan_id=None,
                idempotency_key=idempotency_key,
                customer_note=customer_note,
                ai_service_summary=ai_service_summary,
            )
            self.session.commit()
            return row
        except Exception:
            self.session.rollback()
            raise

    def _create_order(
        self,
        plan: CandidatePlan,
        requirement: TaskRequirement,
        *,
        customer_user_id: str | None,
        plan_id: str | None,
        idempotency_key: str,
        customer_note: str,
        ai_service_summary: str,
    ) -> WorkOrderRow:
        self._assert_resources_free(plan, requirement)
        now = _now()
        work_order = WorkOrderRow(
            id=f"wo_{uuid4().hex}",
            order_no=f"WO-{now:%Y%m%d}-{uuid4().hex[:8].upper()}",
            customer_user_id=customer_user_id,
            store_id=requirement.store_id,
            service_code=plan.task_id,
            service_name=plan.task_id,
            service_summary=ai_service_summary or requirement.task_id,
            scheduled_start=plan.interval.start,
            scheduled_end=plan.interval.end,
            technician_id=plan.technician_id,
            workstation_id=plan.workstation_id,
            status=WorkOrderStatus.CONFIRMED,
            customer_note=customer_note,
            ai_service_summary=ai_service_summary,
            idempotency_key=idempotency_key,
            version=1,
            created_at=now,
            updated_at=now,
        )
        self.session.add(work_order)
        self.session.flush()
        self._create_reservations(work_order, plan, requirement, plan_id=plan_id)
        self._event(work_order, "CREATED", None, WorkOrderStatus.CONFIRMED, "CUSTOMER_AGENT")
        return work_order

    def reschedule_from_confirmed_plan_id(
        self,
        order_no: str,
        confirmed_plan_id: str,
        *,
        confirmation_token: str,
    ) -> WorkOrderRow:
        if not confirmation_token:
            raise ConfirmationRequired("customer confirmation is required")
        try:
            self._begin_write()
            confirmed = self.session.get(ConfirmedPlanRow, confirmed_plan_id)
            if confirmed is None or confirmed.status not in {"AVAILABLE", "CONFIRMED"}:
                raise InvalidConfirmationToken("confirmed plan is unavailable")
            if confirmed.expires_at and confirmed.expires_at <= _now():
                raise InvalidConfirmationToken("confirmed plan has expired")
            if confirmed.confirmation_token_hash != _token_hash(confirmation_token):
                raise InvalidConfirmationToken("confirmation token is invalid")
            plan = CandidatePlan.model_validate_json(confirmed.plan_json)
            requirement = TaskRequirement.model_validate_json(confirmed.requirement_json)
            row = self.reschedule(
                order_no,
                plan,
                requirement,
                confirmation_token=confirmation_token,
                expected_confirmation_token=confirmation_token,
                plan_id=confirmed.id,
            )
            confirmed.status = "APPLIED"
            confirmed.work_order_id = row.id
            confirmed.applied_at = _now()
            row.current_plan_id = confirmed.id
            self.session.commit()
            return row
        except Exception:
            self.session.rollback()
            raise

    def list_all(self) -> list[WorkOrderRow]:
        return list(
            self.session.scalars(
                select(WorkOrderRow).order_by(WorkOrderRow.created_at.desc())
            )
        )

    def list_for_customer(
        self, customer_user_id: str, status: str | None = None
    ) -> list[WorkOrderRow]:
        query = select(WorkOrderRow).where(
            WorkOrderRow.customer_user_id == customer_user_id
        )
        if status is not None:
            query = query.where(WorkOrderRow.status == status)
        return list(self.session.scalars(query.order_by(WorkOrderRow.created_at.desc())))

    def get_by_order_no(
        self, order_no: str, *, customer_user_id: str | None = None
    ) -> WorkOrderRow:
        row = self.session.scalar(select(WorkOrderRow).where(WorkOrderRow.order_no == order_no))
        if row is None or (
            customer_user_id is not None and row.customer_user_id != customer_user_id
        ):
            raise WorkOrderNotFound(order_no)
        return row

    def cancel(
        self,
        order_no: str,
        *,
        confirmation_token: str,
        expected_confirmation_token: str,
    ) -> WorkOrderRow:
        if not confirmation_token:
            raise ConfirmationRequired("customer confirmation is required")
        if confirmation_token != expected_confirmation_token:
            raise InvalidConfirmationToken("confirmation token is invalid")
        try:
            row = self.get_by_order_no(order_no)
            validate_transition(row.status, WorkOrderStatus.CANCELLED)
            now = _now()
            row.status = WorkOrderStatus.CANCELLED
            row.cancelled_at = now
            row.updated_at = now
            row.version += 1
            for reservation in row.reservations:
                if reservation.status == ACTIVE:
                    reservation.status = CANCELLED
            self._event(
                row,
                "CANCELLED",
                WorkOrderStatus.CONFIRMED,
                WorkOrderStatus.CANCELLED,
                "CUSTOMER_AGENT",
            )
            self.session.commit()
            return row
        except Exception:
            self.session.rollback()
            raise

    def check_in(self, order_no: str, *, actor_role: str = "SERVICE_ADVISOR") -> WorkOrderRow:
        return self._store_transition(
            order_no, WorkOrderStatus.CHECKED_IN, actor_role, "CHECKED_IN"
        )

    def complete(self, order_no: str, *, actor_role: str = "SERVICE_ADVISOR") -> WorkOrderRow:
        return self._store_transition(order_no, WorkOrderStatus.COMPLETED, actor_role, "COMPLETED")

    def reschedule(
        self,
        order_no: str,
        plan: CandidatePlan,
        requirement: TaskRequirement,
        *,
        confirmation_token: str,
        expected_confirmation_token: str,
        plan_id: str | None = None,
    ) -> WorkOrderRow:
        if not confirmation_token:
            raise ConfirmationRequired("customer confirmation is required")
        if confirmation_token != expected_confirmation_token:
            raise InvalidConfirmationToken("confirmation token is invalid")
        try:
            row = self.get_by_order_no(order_no)
            if row.status != WorkOrderStatus.CONFIRMED:
                raise CustomerOperationForbidden("only confirmed work orders can be rescheduled")
            self._assert_resources_free(plan, requirement, exclude_work_order_id=row.id)
            cancellation_time = _now()
            for reservation in row.reservations:
                if reservation.status == ACTIVE:
                    reservation.status = CANCELLED
                    reservation.cancelled_at = cancellation_time
            row.scheduled_start = plan.interval.start
            row.scheduled_end = plan.interval.end
            row.technician_id = plan.technician_id
            row.workstation_id = plan.workstation_id
            row.updated_at = _now()
            row.version += 1
            old_plan = (
                self.session.get(ConfirmedPlanRow, row.current_plan_id)
                if row.current_plan_id
                else None
            )
            if old_plan is not None and old_plan.id != plan_id:
                old_plan.status = "SUPERSEDED"
                old_plan.superseded_at = _now()
                old_plan.supersedes_plan_id = plan_id
            row.current_plan_id = plan_id or row.current_plan_id
            self._create_reservations(row, plan, requirement, plan_id=plan_id)
            self._event(
                row,
                "RESCHEDULED",
                WorkOrderStatus.CONFIRMED,
                WorkOrderStatus.CONFIRMED,
                "CUSTOMER_AGENT",
            )
            self.session.commit()
            return row
        except Exception:
            self.session.rollback()
            raise

    def _store_transition(
        self, order_no: str, target: WorkOrderStatus, actor_role: str, event_type: str
    ) -> WorkOrderRow:
        if actor_role != "SERVICE_ADVISOR":
            raise CustomerOperationForbidden("store execution action requires SERVICE_ADVISOR")
        try:
            row = self.get_by_order_no(order_no)
            previous = WorkOrderStatus(row.status)
            validate_transition(previous, target)
            now = _now()
            row.status = target
            row.updated_at = now
            row.version += 1
            if target == WorkOrderStatus.CHECKED_IN:
                row.checked_in_at = now
            elif target == WorkOrderStatus.COMPLETED:
                row.completed_at = now
                for reservation in row.reservations:
                    if reservation.status == ACTIVE:
                        reservation.status = COMPLETED
            self._event(row, event_type, previous, target, actor_role)
            self.session.commit()
            return row
        except Exception:
            self.session.rollback()
            raise

    def _begin_write(self) -> None:
        self.session.rollback()
        if self.session.bind.dialect.name == "sqlite":
            self.session.execute(text("BEGIN IMMEDIATE"))

    def _find_idempotent(self, idempotency_key: str) -> WorkOrderRow | None:
        return self.session.scalar(
            select(WorkOrderRow).where(WorkOrderRow.idempotency_key == idempotency_key)
        )

    def _assert_resources_free(
        self,
        plan: CandidatePlan,
        requirement: TaskRequirement,
        exclude_work_order_id: str | None = None,
    ) -> None:
        resource_ids = {
            ("technician", plan.technician_id),
            ("workstation", plan.workstation_id),
        }
        resource_ids.update(("equipment", equipment_id) for equipment_id in plan.equipment_ids)
        reservations = self.session.scalars(
            select(ResourceReservationRow).where(ResourceReservationRow.status == ACTIVE)
        )
        for reservation in reservations:
            if exclude_work_order_id and reservation.work_order_id == exclude_work_order_id:
                continue
            if (reservation.resource_type, reservation.resource_id) in resource_ids and _overlaps(
                reservation.start_time,
                reservation.end_time,
                plan.interval.start,
                plan.interval.end,
            ):
                raise ReservationConflict(
                    "resource is no longer available: "
                    f"{reservation.resource_type}/{reservation.resource_id}"
                )

    def _create_reservations(
        self,
        work_order: WorkOrderRow,
        plan: CandidatePlan,
        requirement: TaskRequirement,
        *,
        plan_id: str | None = None,
    ) -> None:
        resources = [
            ("technician", plan.technician_id),
            ("workstation", plan.workstation_id),
        ]
        resources.extend(("equipment", equipment_id) for equipment_id in plan.equipment_ids)
        for resource_type, resource_id in resources:
            work_order.reservations.append(
                ResourceReservationRow(
                    resource_type=resource_type,
                    resource_id=resource_id,
                    task_id=plan.task_id,
                    plan_id=plan_id,
                    work_order_id=work_order.id,
                    start_time=plan.interval.start,
                    end_time=plan.interval.end,
                    status=ACTIVE,
                )
            )

    def _event(
        self,
        work_order: WorkOrderRow,
        event_type: str,
        from_status: WorkOrderStatus | None,
        to_status: WorkOrderStatus | None,
        actor_role: str,
    ) -> None:
        self.session.add(
            WorkOrderEventRow(
                work_order_id=work_order.id,
                event_type=event_type,
                from_status=from_status,
                to_status=to_status,
                actor_role=actor_role,
                created_at=_now(),
            )
        )


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _overlaps(start_a: datetime, end_a: datetime, start_b: datetime, end_b: datetime) -> bool:
    start_a, end_a, start_b, end_b = map(_as_utc, (start_a, end_a, start_b, end_b))
    return start_a < end_b and start_b < end_a


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
