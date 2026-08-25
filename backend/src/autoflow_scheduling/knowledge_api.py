from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, sessionmaker

from .auth import ActorRole, require_roles
from .knowledge.retrieval_service import (
    KnowledgeRetriever,
    RetrievalRequest,
    RetrievalResponse,
)


def create_knowledge_router(session_factory: sessionmaker[Session]) -> APIRouter:
    router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])
    retriever = KnowledgeRetriever(session_factory)

    def retrieve_knowledge(
        request: RetrievalRequest,
        _role: ActorRole = Depends(
            require_roles(ActorRole.CUSTOMER_AGENT, ActorRole.SERVICE_ADVISOR)
        ),
    ) -> RetrievalResponse:
        try:
            return retriever.retrieve(request)
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    router.add_api_route(
        "/retrieve",
        retrieve_knowledge,
        methods=["POST"],
        response_model=RetrievalResponse,
    )
    return router
