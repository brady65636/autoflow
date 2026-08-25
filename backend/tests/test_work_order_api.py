from datetime import datetime, timezone

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from autoflow_scheduling.auth import hash_password
from autoflow_scheduling.database import create_session_factory
from autoflow_scheduling.db_planner import build_profile_planner
from autoflow_scheduling.models import TaskRequirement
from autoflow_scheduling.repository import create_user
from autoflow_scheduling.seed import STORE_ID, seed_demo
from autoflow_scheduling.work_order_api import create_app

UTC = timezone.utc
DAY = datetime(2026, 1, 1, 0, tzinfo=UTC)


class HistoryTestModel:
    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        human_count = sum(message.type == "human" for message in messages)
        return AIMessage(content=f"human_messages={human_count}")


class ChatTestModel:
    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        if any(message.type == "tool" for message in messages):
            return AIMessage(content="没有找到符合条件的工单。")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "get_work_order_status",
                    "args": {"status": "COMPLETED"},
                    "id": "chat-call-1",
                    "type": "tool_call",
                }
            ],
        )


def test_agent_chat_uses_bearer_identity_and_tool_loop(tmp_path):
    factory = create_session_factory(f"sqlite+pysqlite:///{tmp_path / 'chat.db'}")
    client = TestClient(create_app(factory, agent_model=ChatTestModel()))
    registered = client.post(
        "/api/auth/register",
        json={"username": "chat-customer", "password": "chat-pass"},
    )
    assert registered.status_code == 201
    response = client.post(
        "/api/agent/chat",
        json={"message": "查询我已完成的工单"},
        headers={"Authorization": f"Bearer {registered.json()['access_token']}"},
    )
    assert response.status_code == 200
    assert response.json()["tool_calls"] == ["get_work_order_status"]
    assert response.json()["message"] == "没有找到符合条件的工单。"


def test_agent_chat_persists_history_by_thread_id(tmp_path):
    factory = create_session_factory(f"sqlite+pysqlite:///{tmp_path / 'checkpoint.db'}")
    client = TestClient(create_app(factory, agent_model=HistoryTestModel()))
    registered = client.post(
        "/api/auth/register",
        json={"username": "checkpoint-customer", "password": "checkpoint-pass"},
    )
    headers = {"Authorization": f"Bearer {registered.json()['access_token']}"}
    first = client.post(
        "/api/agent/chat",
        json={"message": "第一句", "thread_id": "conversation-1"},
        headers=headers,
    )
    second = client.post(
        "/api/agent/chat",
        json={"message": "第二句", "thread_id": "conversation-1"},
        headers=headers,
    )
    assert first.json()["message"] == "human_messages=1"
    assert second.json()["message"] == "human_messages=2"


def test_customer_api_creates_and_reads_work_order_but_store_actions_are_separate(tmp_path):
    factory = create_session_factory(f"sqlite+pysqlite:///{tmp_path / 'test.db'}")
    with factory() as session:
        seed_demo(session)
        requirement = TaskRequirement(
            task_id="engine-diagnosis",
            brand="volkswagen",
            store_id=STORE_ID,
            duration_minutes=60,
            earliest_start=datetime(2026, 1, 1, 9, tzinfo=UTC),
            latest_end=datetime(2026, 1, 1, 17, tzinfo=UTC),
            required_skills={"engine-diagnosis"},
            required_workstation_types={"diagnostic"},
            required_equipment_types={"obd-scanner"},
        )
        result = build_profile_planner(
            session, requirement.store_id, requirement.vehicle_profile, DAY
        ).plan(requirement)
        assert result.plan is not None

    with factory() as session:
        create_user(
            session,
            "user-advisor",
            "advisor",
            hash_password("advisor-pass"),
            "SERVICE_ADVISOR",
        )
        create_user(
            session,
            "user-agent",
            "agent",
            hash_password("agent-pass"),
            "CUSTOMER_AGENT",
        )
    client = TestClient(create_app(factory))
    frontend = client.get("/")
    assert frontend.status_code == 200
    assert "AutoFlow" in frontend.text
    registered = client.post(
        "/api/auth/register",
        json={"username": "customer", "password": "customer-pass"},
    )
    assert registered.status_code == 201
    customer_token = registered.json()["access_token"]
    agent_login = client.post(
        "/api/auth/token",
        json={"username": "agent", "password": "agent-pass"},
    )
    assert agent_login.status_code == 200
    advisor_login = client.post(
        "/api/auth/token",
        json={"username": "advisor", "password": "advisor-pass"},
    )
    assert advisor_login.status_code == 200
    advisor_token = advisor_login.json()["access_token"]
    plan_response = client.post(
        "/api/confirmed-plans",
        json={
            "plan": result.plan.model_dump(mode="json"),
            "requirement": requirement.model_dump(mode="json"),
            "confirmation_token": "confirmed",
        },
        headers={"Authorization": f"Bearer {customer_token}"},
    )
    assert plan_response.status_code == 201
    payload = {
        "confirmed_plan_id": plan_response.json()["confirmed_plan_id"],
        "confirmation_token": "confirmed",
        "idempotency_key": "api-1",
    }
    response = client.post(
        "/api/work-orders",
        json=payload,
        headers={"Authorization": f"Bearer {customer_token}"},
    )
    assert response.status_code == 201
    order_no = response.json()["order_no"]
    assert response.json()["status"] == "CONFIRMED"

    listed = client.get(
        "/api/work-orders",
        headers={"Authorization": f"Bearer {customer_token}"},
    )
    assert listed.status_code == 200
    assert [item["order_no"] for item in listed.json()] == [order_no]

    fetched = client.get(
        f"/api/work-orders/{order_no}",
        headers={"Authorization": f"Bearer {customer_token}"},
    )
    assert fetched.status_code == 200
    assert fetched.json()["order_no"] == order_no

    other = client.post(
        "/api/auth/register",
        json={"username": "other-customer", "password": "other-pass"},
    )
    assert other.status_code == 201
    other_fetched = client.get(
        f"/api/work-orders/{order_no}",
        headers={"Authorization": f"Bearer {other.json()['access_token']}"},
    )
    assert other_fetched.status_code == 404

    forbidden = client.post(
        f"/api/store/work-orders/{order_no}/check-in",
        headers={"Authorization": f"Bearer {customer_token}"},
    )
    assert forbidden.status_code == 403

    checked_in = client.post(
        f"/api/store/work-orders/{order_no}/check-in",
        headers={"Authorization": f"Bearer {advisor_token}"},
    )
    assert checked_in.status_code == 200
    assert checked_in.json()["status"] == "CHECKED_IN"

    completed = client.post(
        f"/api/store/work-orders/{order_no}/complete",
        headers={"Authorization": f"Bearer {advisor_token}"},
    )
    assert completed.status_code == 200
    assert completed.json()["status"] == "COMPLETED"
