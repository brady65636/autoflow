from __future__ import annotations

import multiprocessing
from datetime import datetime, timezone

from autoflow_scheduling.database import create_session_factory
from autoflow_scheduling.db_planner import build_profile_planner
from autoflow_scheduling.models import CandidatePlan, TaskRequirement
from autoflow_scheduling.seed import STORE_ID, seed_demo
from autoflow_scheduling.work_order_service import WorkOrderService

UTC = timezone.utc
DAY = datetime(2026, 1, 1, 0, tzinfo=UTC)


def _create_worker(db_url: str, plan_data: dict, requirement_data: dict, key: str, queue) -> None:
    factory = create_session_factory(db_url)
    with factory() as session:
        try:
            service = WorkOrderService(session)
            service.create_from_confirmed_plan(
                CandidatePlan.model_validate(plan_data),
                TaskRequirement.model_validate(requirement_data),
                confirmation_token="confirmed",
                expected_confirmation_token="confirmed",
                idempotency_key=key,
            )
            queue.put("created")
        except Exception as error:  # process boundary must report the actual failure
            queue.put(type(error).__name__)


def test_two_processes_cannot_reserve_the_same_resources(tmp_path):
    db_url = f"sqlite+pysqlite:///{tmp_path / 'concurrency.db'}"
    factory = create_session_factory(db_url)
    with factory() as session:
        seed_demo(session)
        requirement = TaskRequirement(
            task_id="race-task",
            brand="volkswagen",
            store_id=STORE_ID,
            duration_minutes=60,
            earliest_start=datetime(2026, 1, 1, 9, tzinfo=UTC),
            latest_end=datetime(2026, 1, 1, 10, tzinfo=UTC),
            required_skills={"engine-diagnosis"},
            required_workstation_types={"diagnostic"},
            required_equipment_types={"obd-scanner"},
        )
        result = build_profile_planner(session, requirement.store_id, requirement.vehicle_profile, DAY).plan(requirement)
        assert result.plan is not None
        plan_data = result.plan.model_dump(mode="json")
        requirement_data = requirement.model_dump(mode="json")

    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    processes = [
        context.Process(
            target=_create_worker,
            args=(db_url, plan_data, requirement_data, f"race-{index}", queue),
        )
        for index in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(30)
    results = [queue.get(timeout=5) for _ in processes]

    assert all(process.exitcode == 0 for process in processes)
    assert sorted(results) == ["ReservationConflict", "created"]
