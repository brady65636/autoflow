from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from autoflow_scheduling.agent import AgentDependencies, build_agent_graph
from autoflow_scheduling.agent.tools import AgentToolContext, build_agent_tools
from autoflow_scheduling.database import create_session_factory
from autoflow_scheduling.knowledge.retrieval_service import (
    RetrievalResponse,
    RetrievedChunk,
    RetrievedDocument,
    RetrievedSection,
)


class FakeRetriever:
    def retrieve(self, request: Any) -> RetrievalResponse:
        return RetrievalResponse(
            query=request.query,
            question_type=request.question_type,
            algorithm={"test": True},
            chunks=[],
            sections=[],
        )


class OneToolThenAnswerModel:
    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        if any(message.type == "tool" for message in messages):
            return AIMessage(content="已完成资料查询。")
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "retrieve_knowledge",
                    "args": {
                        "query": "EA211 oil circuit",
                        "question_type": "principle",
                    },
                    "id": "call-1",
                    "type": "tool_call",
                }
            ],
        )


def test_customer_agent_tools_exclude_store_execution_actions() -> None:
    tools = build_agent_tools(
        AgentToolContext(create_session_factory("sqlite+pysqlite:///:memory:"), FakeRetriever()),
        actor_role="CUSTOMER_AGENT",
    )

    assert {item.name for item in tools} == {
        "retrieve_knowledge",
        "solve_schedule",
        "create_confirmed_plan",
        "get_work_order_status",
        "create_work_order",
        "cancel_work_order",
        "reschedule_work_order",
    }


def test_service_advisor_tools_include_store_execution_actions() -> None:
    tools = build_agent_tools(
        AgentToolContext(create_session_factory("sqlite+pysqlite:///:memory:"), FakeRetriever()),
        actor_role="SERVICE_ADVISOR",
    )

    names = {item.name for item in tools}
    assert {"check_in_work_order", "complete_work_order"} <= names


def test_retrieval_agent_payload_keeps_source_confidence_without_debug_scores() -> None:
    response = RetrievalResponse(
        query="EA211 oil circuit",
        question_type="principle",
        algorithm={"dense_weight": 20},
        chunks=[
            RetrievedChunk(
                chunk_id="doc:s1:c1",
                section_id="doc:s1",
                document_id="doc",
                rank=1,
                rrf_score=0.2,
                reranker_score=0.9,
                business_rule_score=1.0,
                final_score=0.92,
                business_rule_reasons=["compatibility=primary"],
                text="The oil circuit supplies lubrication.",
            )
        ],
        sections=[
            RetrievedSection(
                section_id="doc:s1",
                document_id="doc",
                title="Oil circuit",
                path=["Engine", "Oil circuit"],
                page_start=12,
                page_end=13,
                rank=1,
                matched_chunk_ids=["doc:s1:c1"],
                text="Full section text is kept outside the Agent payload.",
            )
        ],
        documents=[
            RetrievedDocument(
                document_id="doc",
                filename="manual.pdf",
                content_type="repair_manual",
                metadata_confidence=0.95,
                pipeline_version=4,
                page_count=80,
                ingestion_status="complete",
                hash_verified=True,
            )
        ],
    )

    payload = response.to_agent_payload()

    assert payload["sources"][0]["metadata_confidence"] == 0.95
    assert payload["evidence"][0]["text"] == "The oil circuit supplies lubrication."
    assert payload["evidence"][0]["document"]["filename"] == "manual.pdf"
    assert "algorithm" not in payload
    assert "rrf_score" not in payload["evidence"][0]
    assert "chunk_id" not in payload["evidence"][0]
    assert "page_start" not in payload["evidence"][0]
    assert "pipeline_version" not in payload["evidence"][0]["document"]
    assert "hash_verified" not in payload["evidence"][0]["document"]
    assert "Full section text" not in payload["evidence"][0]["text"]


def test_graph_loops_through_tool_and_returns_to_model() -> None:
    factory = create_session_factory("sqlite+pysqlite:///:memory:")
    graph = build_agent_graph(
        OneToolThenAnswerModel(),
        AgentDependencies(factory, FakeRetriever()),
    )

    result = graph.invoke({"messages": [HumanMessage(content="查一下 EA211 油路")]})

    assert result["messages"][-1].content == "已完成资料查询。"
    assert any(message.type == "tool" for message in result["messages"])
