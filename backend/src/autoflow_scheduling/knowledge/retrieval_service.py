"""Local-model RAG retrieval over persisted SQLite chunk embeddings."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from autoflow_scheduling.db_models import (
    KnowledgeChunkRow,
    KnowledgeDocumentRow,
    KnowledgeSectionRow,
)
from autoflow_scheduling.knowledge.hybrid_evaluation import (
    _business_rule_score,
    _encode_texts,
    _rerank_scores,
    _rrf,
)
from autoflow_scheduling.knowledge.retrieval_profile import (
    QuestionType,
    SearchProfile,
    build_query_profile_text,
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")
_EMBED_INSTRUCTION = (
    "Given an automotive technical query, retrieve passages that directly contain "
    "the requested knowledge. Pay attention to the question type, required knowledge, "
    "section title, and subheading."
)


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    question_type: QuestionType
    top_chunks: int = Field(default=15, ge=1, le=15)


class RetrievedChunk(BaseModel):
    chunk_id: str
    section_id: str
    document_id: str
    rank: int
    rrf_score: float
    reranker_score: float
    business_rule_score: float
    final_score: float
    business_rule_reasons: list[str]
    text: str


class RetrievedSection(BaseModel):
    section_id: str
    document_id: str
    title: str
    path: list[str]
    page_start: int | None
    page_end: int | None
    rank: int
    matched_chunk_ids: list[str]
    text: str


class RetrievedDocument(BaseModel):
    document_id: str
    filename: str | None = None
    content_type: str
    metadata_confidence: float | None = None
    pipeline_version: int | None = None
    page_count: int
    ingestion_status: str
    hash_verified: bool


class RetrievalResponse(BaseModel):
    query: str
    question_type: QuestionType
    algorithm: dict[str, Any]
    chunks: list[RetrievedChunk]
    sections: list[RetrievedSection]
    documents: list[RetrievedDocument] = Field(default_factory=list)

    def to_agent_payload(self) -> dict[str, Any]:
        """Return compact, source-aware evidence for an Agent tool message."""
        documents = {item.document_id: item for item in self.documents}
        sections = {item.section_id: item for item in self.sections}
        evidence = []
        matched_documents: set[str] = set()
        for chunk in self.chunks:
            section = sections.get(chunk.section_id)
            document = documents.get(chunk.document_id)
            if document is None:
                continue
            matched_documents.add(chunk.document_id)
            evidence.append(
                {
                    "section_title": section.title if section else None,
                    "text": chunk.text,
                    "document": {
                        "filename": document.filename,
                        "content_type": document.content_type,
                        "metadata_confidence": document.metadata_confidence,
                    },
                }
            )
        return {
            "query": self.query,
            "question_type": self.question_type,
            "evidence": evidence,
            "sources": [
                {
                    "filename": item.filename,
                    "content_type": item.content_type,
                    "metadata_confidence": item.metadata_confidence,
                }
                for item in self.documents
                if item.document_id in matched_documents
            ],
        }


@dataclass
class _Index:
    chunks: list[dict[str, Any]]
    vectors: np.ndarray
    bm25: BM25Okapi
    sections: dict[str, dict[str, Any]]
    documents: dict[str, dict[str, Any]]


def _tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text)]


class KnowledgeRetriever:
    """Lazy-loading local embedding/reranker service for the API process."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        embedding_model_path: str | Path | None = None,
        reranker_model_path: str | Path | None = None,
        dense_limit: int = 50,
        bm25_limit: int = 50,
        rrf_k: int = 60,
        dense_weight: float = 20.0,
        bm25_weight: float = 1.0,
        business_rule_weight: float = 0.2,
    ) -> None:
        self.session_factory = session_factory
        self.embedding_model_path = Path(
            embedding_model_path or os.environ.get("AUTOFLOW_EMBEDDING_MODEL", "")
        )
        self.reranker_model_path = Path(
            reranker_model_path or os.environ.get("AUTOFLOW_RERANKER_MODEL", "")
        )
        self.dense_limit = dense_limit
        self.bm25_limit = bm25_limit
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.business_rule_weight = business_rule_weight
        self._index: _Index | None = None
        self._embedding_model: Any = None
        self._embedding_tokenizer: Any = None
        self._reranker_model: Any = None
        self._reranker_tokenizer: Any = None

    def _load_index(self) -> _Index:
        if self._index is not None:
            return self._index
        with self.session_factory() as session:
            chunk_rows = list(
                session.scalars(select(KnowledgeChunkRow).order_by(KnowledgeChunkRow.id))
            )
            section_rows = list(
                session.scalars(select(KnowledgeSectionRow).order_by(KnowledgeSectionRow.id))
            )
            document_rows = list(
                session.scalars(select(KnowledgeDocumentRow).order_by(KnowledgeDocumentRow.id))
            )
        if not chunk_rows:
            raise RuntimeError("knowledge database has no chunks")
        missing = [row.id for row in chunk_rows if not row.embedding]
        if missing:
            raise RuntimeError(f"{len(missing)} chunks have no persisted embedding")
        dimensions = {int(row.embedding_dimension or 0) for row in chunk_rows}
        if len(dimensions) != 1 or 0 in dimensions:
            raise RuntimeError("chunk embedding dimensions are inconsistent")
        dimension = dimensions.pop()
        chunks = [
            {
                "chunk_id": row.id,
                "section_id": row.section_id,
                "document_id": row.document_id,
                "document_content_type": row.content_type,
                "metadata_confidence": row.metadata_confidence,
                "quality": self._json(row.quality_json, {}),
                "text": row.text,
                "index_text": row.index_text,
                "subheading": row.subheading,
            }
            for row in chunk_rows
        ]
        vectors = np.vstack(
            [np.frombuffer(row.embedding, dtype="<f4", count=dimension) for row in chunk_rows]
        ).astype(np.float32, copy=False)
        sections = {
            row.id: {
                "section_id": row.id,
                "document_id": row.document_id,
                "title": row.title,
                "path": self._json(row.path_json, []),
                "page_start": row.page_start,
                "page_end": row.page_end,
                "text": row.text,
            }
            for row in section_rows
        }
        documents = {
            row.id: {
                "document_id": row.id,
                "filename": row.filename,
                "content_type": row.content_type,
                "metadata_confidence": row.metadata_confidence,
                "pipeline_version": row.pipeline_version,
                "page_count": row.page_count,
                "ingestion_status": row.ingestion_status,
                "hash_verified": row.hash_verified_at is not None,
            }
            for row in document_rows
        }
        self._index = _Index(
            chunks,
            vectors,
            BM25Okapi([_tokens(c["index_text"]) for c in chunks]),
            sections,
            documents,
        )
        return self._index

    @staticmethod
    def _json(value: str | None, default: Any) -> Any:
        import json
        try:
            return json.loads(value or "")
        except json.JSONDecodeError:
            return default

    def _load_models(self) -> None:
        if self._embedding_model is not None:
            return
        if not self.embedding_model_path.is_dir() or not self.reranker_model_path.is_dir():
            raise RuntimeError(
                "local model paths are unavailable; set AUTOFLOW_EMBEDDING_MODEL "
                "and AUTOFLOW_RERANKER_MODEL"
            )
        import torch
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

        self._embedding_tokenizer = AutoTokenizer.from_pretrained(
            self.embedding_model_path, padding_side="left", local_files_only=True
        )
        self._embedding_model = AutoModel.from_pretrained(
            self.embedding_model_path, dtype=torch.float16, local_files_only=True
        ).cuda().eval()
        self._reranker_tokenizer = AutoTokenizer.from_pretrained(
            self.reranker_model_path, padding_side="left", local_files_only=True
        )
        self._reranker_model = AutoModelForCausalLM.from_pretrained(
            self.reranker_model_path, dtype=torch.float16, local_files_only=True
        ).cuda().eval()

    def retrieve(self, request: RetrievalRequest) -> RetrievalResponse:
        index = self._load_index()
        self._load_models()
        profile = SearchProfile(
            query=request.query,
            question_type=request.question_type,
            required_knowledge=request.query,
        )
        query_text = build_query_profile_text(profile)
        instructed_query = f"Instruct: {_EMBED_INSTRUCTION}\nQuery: {query_text}"
        query_vector = _encode_texts(
            self._embedding_model,
            self._embedding_tokenizer,
            [instructed_query],
            batch_size=1,
            max_length=512,
        )[0]
        dense_order = np.argsort(index.vectors @ query_vector)[::-1].tolist()
        bm25_scores = index.bm25.get_scores(_tokens(request.query))
        bm25_order = sorted(
            (i for i, score in enumerate(bm25_scores) if float(score) > 0),
            key=lambda i: float(bm25_scores[i]),
            reverse=True,
        )
        rrf_limit = max(self.dense_limit, self.bm25_limit)
        rrf_scores: dict[int, float] = {}
        for weight, order in (
            (self.dense_weight, dense_order[: self.dense_limit]),
            (self.bm25_weight, bm25_order[: self.bm25_limit]),
        ):
            for rank, candidate_index in enumerate(order, 1):
                rrf_scores[candidate_index] = rrf_scores.get(candidate_index, 0.0) + weight / (self.rrf_k + rank)
        rrf_order = _rrf(
            dense_order,
            bm25_order,
            rrf_limit,
            k=self.rrf_k,
            dense_weight=self.dense_weight,
            bm25_weight=self.bm25_weight,
        )[: request.top_chunks]
        candidate_chunks = [index.chunks[i] for i in rrf_order]
        pairs = [(query_text, chunk["index_text"]) for chunk in candidate_chunks]
        model_scores = _rerank_scores(
            self._reranker_model, self._reranker_tokenizer, pairs,
            batch_size=4, max_length=2048,
        )
        scored = []
        case = {"question_type": request.question_type.value}
        for chunk_index, (candidate_index, model_score) in enumerate(
            zip(rrf_order, model_scores, strict=True)
        ):
            chunk = index.chunks[candidate_index]
            rule_score, reasons = _business_rule_score(case, chunk)
            final_score = (1 - self.business_rule_weight) * model_score + self.business_rule_weight * rule_score
            scored.append((candidate_index, model_score, rule_score, final_score, reasons))
        scored.sort(key=lambda item: item[3], reverse=True)
        chunks_out = []
        section_hits: dict[str, list[str]] = {}
        for rank, (chunk_index, model_score, rule_score, final_score, reasons) in enumerate(scored, 1):
            chunk = index.chunks[chunk_index]
            section_hits.setdefault(chunk["section_id"], []).append(chunk["chunk_id"])
            chunks_out.append(RetrievedChunk(
                chunk_id=chunk["chunk_id"], section_id=chunk["section_id"], document_id=chunk["document_id"],
                rank=rank, rrf_score=round(rrf_scores.get(chunk_index, 0.0), 6), reranker_score=round(model_score, 6),
                business_rule_score=round(rule_score, 6), final_score=round(final_score, 6),
                business_rule_reasons=reasons, text=chunk["text"],
            ))
        sections_out = []
        for rank, (section_id, matched) in enumerate(
            sorted(section_hits.items(), key=lambda item: min(c.rank for c in chunks_out if c.section_id == item[0])), 1
        ):
            section = index.sections.get(section_id)
            if section is None:
                continue
            sections_out.append(RetrievedSection(
                **{key: section[key] for key in ("section_id", "document_id", "title", "path", "page_start", "page_end", "text")},
                rank=rank, matched_chunk_ids=matched,
            ))
        document_ids = {chunk["document_id"] for chunk in candidate_chunks}
        documents_out = [
            RetrievedDocument(**index.documents[document_id])
            for document_id in document_ids
            if document_id in index.documents
        ]
        return RetrievalResponse(
            query=request.query,
            question_type=request.question_type,
            algorithm={
                "embedding": str(self.embedding_model_path), "bm25": "rank_bm25",
                "dense_limit": self.dense_limit, "bm25_limit": self.bm25_limit,
                "dense_weight": self.dense_weight, "bm25_weight": self.bm25_weight,
                "rrf_k": self.rrf_k, "final_chunk_limit": request.top_chunks,
                "reranker": str(self.reranker_model_path),
                "business_rule_weight": self.business_rule_weight,
            },
            chunks=chunks_out,
            sections=sections_out,
            documents=documents_out,
        )
