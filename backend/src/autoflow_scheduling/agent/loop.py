from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from sqlalchemy.orm import Session, sessionmaker

from ..knowledge.retrieval_service import KnowledgeRetriever
from ..observability import get_tracer
from .tools import AgentToolContext, build_agent_tools


@dataclass(frozen=True)
class AgentDependencies:
    session_factory: sessionmaker[Session]
    retriever: KnowledgeRetriever
    customer_user_id: str | None = None


def build_agent_graph(
    model: BaseChatModel,
    dependencies: AgentDependencies,
    *,
    actor_role: str = "CUSTOMER_AGENT",
    checkpointer: BaseCheckpointSaver[Any] | None = None,
):
    """Build the model → tools → model LangGraph loop.

    The caller owns the model/provider and may pass a checkpointer for durable
    conversations. This function only wires domain tools and graph control flow.
    """
    tools = build_agent_tools(
        AgentToolContext(
            dependencies.session_factory,
            dependencies.retriever,
            dependencies.customer_user_id,
        ),
        actor_role=actor_role,
    )
    model_with_tools = model.bind_tools(tools)

    def call_model(state: MessagesState) -> dict[str, list[Any]]:
        tracer = get_tracer()
        with tracer.stage("agent_model", metadata={"actor_role": actor_role}) as run:
            try:
                response = model_with_tools.invoke(state["messages"])
                tracer.update(
                    run,
                    status="complete",
                    has_tool_calls=bool(getattr(response, "tool_calls", None)),
                )
                return {"messages": [response]}
            except Exception as error:
                tracer.update(
                    run,
                    status="error",
                    error_type=type(error).__name__,
                    error=str(error),
                )
                raise

    tool_node = ToolNode(tools)

    def run_tools(
        state: MessagesState, config: RunnableConfig = None
    ) -> dict[str, list[Any]]:
        tracer = get_tracer()
        last_message = state["messages"][-1]
        names = [
            call["name"]
            for call in getattr(last_message, "tool_calls", []) or []
        ]
        with tracer.stage(
            "agent_tool",
            metadata={"actor_role": actor_role, "tool_names": names},
        ) as run:
            try:
                result = tool_node.invoke(state, config)
                tracer.update(run, status="complete", tool_count=len(names))
                return result
            except Exception as error:
                tracer.update(
                    run,
                    status="error",
                    error_type=type(error).__name__,
                    error=str(error),
                )
                raise

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", run_tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer)
