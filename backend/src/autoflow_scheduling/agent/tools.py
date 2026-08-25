from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated, Any, Literal

from langchain_core.tools import StructuredTool, tool
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session, sessionmaker

from ..agent_tools import CUSTOMER_AGENT_TOOLS, STORE_TOOLS
from ..catalog import MVP_OPERATIONS
from ..db_planner import build_profile_planner
from ..knowledge.retrieval_service import (
    KnowledgeRetriever,
    RetrievalRequest,
)
from ..models import TaskRequirement, VehicleProfile
from ..work_order import WorkOrderStatus
from ..work_order_service import WorkOrderService

ServiceTypeValue = Literal[
    "arrival-diagnosis",
    "routine-maintenance",
    "engine-warning-diagnosis",
    "electrical-diagnosis",
    "brake-inspection",
    "brake-pad-replacement",
    "wheel-alignment",
    "final-quality-inspection",
]
QuestionTypeValue = Literal[
    "principle",
    "diagnosis",
    "repair",
    "specification",
    "maintenance",
    "training",
    "competition",
    "general",
]
WorkOrderStatusValue = Literal["CONFIRMED", "CHECKED_IN", "COMPLETED", "CANCELLED"]


class ScheduleToolRequirement(BaseModel):
    service_type: ServiceTypeValue = Field(
        description="Service catalog code. Choose one of the supported service types."
    )
    scheduled_at: datetime = Field(
        description=(
            "Preferred appointment center time in ISO-8601 format. The backend creates "
            "a scheduling window from one hour before to one hour after this time."
        )
    )
    vehicle_profile: VehicleProfile = Field(
        description="Vehicle classification: category and powertrain."
    )


def _task_from_schedule(
    requirement: ScheduleToolRequirement,
) -> tuple[Any, TaskRequirement]:
    operation = next(
        item for item in MVP_OPERATIONS if item.code == requirement.service_type
    )
    window_start = requirement.scheduled_at - timedelta(hours=1)
    window_end = requirement.scheduled_at + timedelta(hours=1)
    return operation, TaskRequirement(
        task_id=operation.code,
        brand=operation.brand,
        store_id=operation.store_id,
        vehicle_profile=requirement.vehicle_profile,
        duration_minutes=operation.duration_minutes,
        earliest_start=window_start,
        latest_end=window_end,
        required_skills=operation.required_skills,
        required_workstation_types=operation.required_workstation_types,
        required_equipment_types=operation.required_equipment_types,
    )


class AgentToolContext:
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        retriever: KnowledgeRetriever,
        customer_user_id: str | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.retriever = retriever
        self.customer_user_id = customer_user_id


def _row_payload(row: Any) -> dict[str, Any]:
    return {
        "order_no": row.order_no,
        "store_id": row.store_id,
        "service_code": row.service_code,
        "service_name": row.service_name,
        "service_summary": row.service_summary,
        "scheduled_start": row.scheduled_start.isoformat(),
        "scheduled_end": row.scheduled_end.isoformat(),
        "technician_id": row.technician_id,
        "workstation_id": row.workstation_id,
        "status": row.status,
        "customer_note": row.customer_note,
        "ai_service_summary": row.ai_service_summary,
        "version": row.version,
    }


def _error(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error": {"code": code, "message": message}}


def _ok(**data: Any) -> dict[str, Any]:
    return {"ok": True, **data}


def build_agent_tools(
    context: AgentToolContext,
    *,
    actor_role: str = "CUSTOMER_AGENT",
) -> list[StructuredTool]:
    """Build the tools allowed for one agent role.

    Tools return JSON-compatible dictionaries so a model can inspect business
    errors (for example, a required confirmation) without crashing the graph.
    """

    @tool
    def retrieve_knowledge(
        query: Annotated[str, Field(
            min_length=1,
            max_length=4000,
            description=(
                "The user's automotive knowledge question, preferably in the user's language."
            ),
        )],
        question_type: Annotated[
            QuestionTypeValue,
            Field(
                description=(
                    "Question category. Must be exactly one English enum value: "
                    "principle, diagnosis, repair, specification, maintenance, training, "
                    "competition, or general. Do not translate the enum value."
                )
            ),
        ],
        top_chunks: Annotated[
            int,
            Field(ge=1, le=15, description="Maximum number of evidence chunks to retrieve."),
        ] = 15,
    ) -> dict[str, Any]:
        """Retrieve automotive knowledge passages and their source sections."""
        try:
            result = context.retriever.retrieve(
                RetrievalRequest(
                    query=query,
                    question_type=question_type,
                    top_chunks=top_chunks,
                )
            )
            return _ok(result=result.to_agent_payload())
        except (ValidationError, ValueError) as error:
            return _error("INVALID_REQUEST", str(error))
        except RuntimeError as error:
            return _error("RETRIEVAL_UNAVAILABLE", str(error))

    @tool
    def solve_schedule(requirement: ScheduleToolRequirement) -> dict[str, Any]:
        """Find a feasible schedule from a complete profile-only scheduling requirement."""
        try:
            operation, task = _task_from_schedule(requirement)
            window_start = task.earliest_start
            window_end = task.latest_end
            with context.session_factory() as session:
                result = build_profile_planner(
                    session,
                    operation.store_id,
                    requirement.vehicle_profile,
                    requirement.scheduled_at,
                    operation.brand,
                ).plan(task)
            return _ok(
                service_type=operation.code,
                window={"start": window_start.isoformat(), "end": window_end.isoformat()},
                result=result.model_dump(mode="json"),
            )
        except (ValidationError, ValueError) as error:
            return _error("INVALID_REQUIREMENT", str(error))

    @tool
    def create_confirmed_plan(
        plan: dict[str, Any],
        service_type: ServiceTypeValue,
        scheduled_at: datetime,
        vehicle_profile: VehicleProfile,
        confirmation_token: str,
    ) -> dict[str, Any]:
        """Persist a catalog-backed candidate plan after customer confirmation."""
        try:
            from ..models import CandidatePlan

            candidate = CandidatePlan.model_validate(plan)
            _, task = _task_from_schedule(
                ScheduleToolRequirement(
                    service_type=service_type,
                    scheduled_at=scheduled_at,
                    vehicle_profile=vehicle_profile,
                )
            )
            with context.session_factory() as session:
                row = WorkOrderService(session).create_confirmed_plan(
                    candidate,
                    task,
                    confirmation_token=confirmation_token,
                    customer_user_id=context.customer_user_id,
                )
            return _ok(confirmed_plan_id=row.id)
        except (ValidationError, ValueError) as error:
            return _error("INVALID_REQUEST", str(error))
        except Exception as error:
            return _error(type(error).__name__, str(error))

    @tool
    def get_work_order_status(
        status: Annotated[
            WorkOrderStatusValue,
            Field(
                description=(
                    "Exact work-order status enum: CONFIRMED, CHECKED_IN, COMPLETED, or CANCELLED."
                )
            ),
        ]
    ) -> dict[str, Any]:
        """Return the authenticated user's work orders with the requested status."""
        if not context.customer_user_id:
            return _error("IDENTITY_REQUIRED", "authenticated customer identity is required")
        try:
            requested_status = WorkOrderStatus(status).value
            with context.session_factory() as session:
                rows = WorkOrderService(session).list_for_customer(
                    context.customer_user_id, status=requested_status
                )
            return _ok(
                status=requested_status,
                work_orders=[_row_payload(row) for row in rows],
            )
        except ValueError as error:
            return _error("INVALID_STATUS", str(error))
        except Exception as error:
            return _error(type(error).__name__, str(error))

    @tool
    def create_work_order(
        confirmed_plan_id: str,
        confirmation_token: str,
        idempotency_key: str,
        customer_note: str = "",
        ai_service_summary: str = "",
    ) -> dict[str, Any]:
        """Create a work order from a confirmed plan."""
        try:
            with context.session_factory() as session:
                row = WorkOrderService(session).create_from_confirmed_plan_id(
                    confirmed_plan_id,
                    confirmation_token=confirmation_token,
                    idempotency_key=idempotency_key,
                    customer_note=customer_note,
                    ai_service_summary=ai_service_summary,
                )
                payload = _row_payload(row)
            return _ok(work_order=payload)
        except Exception as error:
            return _error(type(error).__name__, str(error))

    @tool
    def cancel_work_order(
        order_no: str, confirmation_token: str, expected_confirmation_token: str
    ) -> dict[str, Any]:
        """Cancel a confirmed work order after customer confirmation."""
        try:
            with context.session_factory() as session:
                row = WorkOrderService(session).cancel(
                    order_no,
                    confirmation_token=confirmation_token,
                    expected_confirmation_token=expected_confirmation_token,
                )
                payload = _row_payload(row)
            return _ok(work_order=payload)
        except Exception as error:
            return _error(type(error).__name__, str(error))

    @tool
    def reschedule_work_order(
        order_no: str, confirmed_plan_id: str, confirmation_token: str
    ) -> dict[str, Any]:
        """Reschedule a confirmed work order using a newly confirmed plan."""
        try:
            with context.session_factory() as session:
                row = WorkOrderService(session).reschedule_from_confirmed_plan_id(
                    order_no,
                    confirmed_plan_id,
                    confirmation_token=confirmation_token,
                )
                payload = _row_payload(row)
            return _ok(work_order=payload)
        except Exception as error:
            return _error(type(error).__name__, str(error))

    @tool
    def check_in_work_order(order_no: str) -> dict[str, Any]:
        """Check a vehicle into the store; service-advisor action only."""
        try:
            with context.session_factory() as session:
                row = WorkOrderService(session).check_in(order_no, actor_role=actor_role)
                payload = _row_payload(row)
            return _ok(work_order=payload)
        except Exception as error:
            return _error(type(error).__name__, str(error))

    @tool
    def complete_work_order(order_no: str) -> dict[str, Any]:
        """Mark a work order complete; service-advisor action only."""
        try:
            with context.session_factory() as session:
                row = WorkOrderService(session).complete(order_no, actor_role=actor_role)
                payload = _row_payload(row)
            return _ok(work_order=payload)
        except Exception as error:
            return _error(type(error).__name__, str(error))

    tools = [retrieve_knowledge, solve_schedule, create_confirmed_plan]
    if actor_role in {"CUSTOMER", "CUSTOMER_AGENT", "SERVICE_ADVISOR"}:
        tools.extend(
            [
                get_work_order_status,
                create_work_order,
                cancel_work_order,
                reschedule_work_order,
            ]
        )
    if actor_role == "SERVICE_ADVISOR":
        tools.extend([check_in_work_order, complete_work_order])

    allowed = CUSTOMER_AGENT_TOOLS | STORE_TOOLS | {
        "retrieve_knowledge",
        "solve_schedule",
        "create_confirmed_plan",
    }
    return [item for item in tools if item.name in allowed]
