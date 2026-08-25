"""Minimal shared profile text for retrieval and reranking.

A document has a content type and classification confidence. Section meaning is
represented directly by its title path and current subheading rather than a
lossy capability enum.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

_DOCUMENT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def validate_document_id(value: str) -> str:
    if not _DOCUMENT_ID_RE.fullmatch(value):
        raise ValueError(
            "document_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}"
        )
    return value


class QuestionType(str, Enum):
    PRINCIPLE = "principle"
    DIAGNOSIS = "diagnosis"
    REPAIR = "repair"
    SPECIFICATION = "specification"
    MAINTENANCE = "maintenance"
    TRAINING = "training"
    COMPETITION = "competition"
    GENERAL = "general"


class DocumentContentType(str, Enum):
    TECHNICAL_TRAINING = "technical_training"
    REPAIR_MANUAL = "repair_manual"
    DIAGNOSTIC_MANUAL = "diagnostic_manual"
    TECHNICAL_ARTICLE = "technical_article"
    COMPETITION_FILE = "competition_file"
    TRAINING_PLAN = "training_plan"
    EQUIPMENT_LIST = "equipment_list"
    OFFICIAL_NEWS = "official_news"
    OTHER = "other"


class ContentCompatibility(str, Enum):
    PRIMARY = "primary"
    SUPPORTING = "supporting"
    NONE = "none"


_QUESTION_LABELS = {
    QuestionType.PRINCIPLE: "working principle / 工作原理",
    QuestionType.DIAGNOSIS: "fault diagnosis / 故障诊断",
    QuestionType.REPAIR: "repair procedure / 维修操作",
    QuestionType.SPECIFICATION: "technical specification / 技术参数",
    QuestionType.MAINTENANCE: "maintenance guidance / 保养维护",
    QuestionType.TRAINING: "training resources / 培训资源",
    QuestionType.COMPETITION: "competition rules / 竞赛规则",
    QuestionType.GENERAL: "general technical knowledge / 通用技术知识",
}

_CONTENT_TYPE_LABELS = {
    DocumentContentType.TECHNICAL_TRAINING: "OEM technical training / 原厂技术培训",
    DocumentContentType.REPAIR_MANUAL: "repair service manual / 维修手册",
    DocumentContentType.DIAGNOSTIC_MANUAL: "diagnostic manual / 诊断手册",
    DocumentContentType.TECHNICAL_ARTICLE: "technical article / 技术文章",
    DocumentContentType.COMPETITION_FILE: "competition technical file / 竞赛技术文件",
    DocumentContentType.TRAINING_PLAN: "training plan / 培训方案",
    DocumentContentType.EQUIPMENT_LIST: "equipment list / 设备清单",
    DocumentContentType.OFFICIAL_NEWS: "official news / 官方新闻",
    DocumentContentType.OTHER: "other document / 其他文档",
}

# This matrix describes only question-type/content-type compatibility. It does
# not assign a global authority or a made-up primary capability to a book.
_COMPATIBILITY: dict[QuestionType, dict[DocumentContentType, ContentCompatibility]] = {
    QuestionType.PRINCIPLE: {
        DocumentContentType.TECHNICAL_TRAINING: ContentCompatibility.PRIMARY,
        DocumentContentType.REPAIR_MANUAL: ContentCompatibility.SUPPORTING,
        DocumentContentType.DIAGNOSTIC_MANUAL: ContentCompatibility.SUPPORTING,
        DocumentContentType.TECHNICAL_ARTICLE: ContentCompatibility.SUPPORTING,
    },
    QuestionType.DIAGNOSIS: {
        DocumentContentType.DIAGNOSTIC_MANUAL: ContentCompatibility.PRIMARY,
        DocumentContentType.REPAIR_MANUAL: ContentCompatibility.PRIMARY,
        DocumentContentType.TECHNICAL_TRAINING: ContentCompatibility.SUPPORTING,
        DocumentContentType.TECHNICAL_ARTICLE: ContentCompatibility.SUPPORTING,
    },
    QuestionType.REPAIR: {
        DocumentContentType.REPAIR_MANUAL: ContentCompatibility.PRIMARY,
        DocumentContentType.DIAGNOSTIC_MANUAL: ContentCompatibility.SUPPORTING,
        DocumentContentType.TECHNICAL_TRAINING: ContentCompatibility.SUPPORTING,
    },
    QuestionType.SPECIFICATION: {
        DocumentContentType.REPAIR_MANUAL: ContentCompatibility.PRIMARY,
        DocumentContentType.DIAGNOSTIC_MANUAL: ContentCompatibility.SUPPORTING,
        DocumentContentType.TECHNICAL_TRAINING: ContentCompatibility.SUPPORTING,
    },
    QuestionType.MAINTENANCE: {
        DocumentContentType.REPAIR_MANUAL: ContentCompatibility.PRIMARY,
        DocumentContentType.TECHNICAL_TRAINING: ContentCompatibility.SUPPORTING,
        DocumentContentType.TECHNICAL_ARTICLE: ContentCompatibility.SUPPORTING,
    },
    QuestionType.TRAINING: {
        DocumentContentType.TRAINING_PLAN: ContentCompatibility.PRIMARY,
        DocumentContentType.EQUIPMENT_LIST: ContentCompatibility.PRIMARY,
        DocumentContentType.TECHNICAL_TRAINING: ContentCompatibility.SUPPORTING,
    },
    QuestionType.COMPETITION: {
        DocumentContentType.COMPETITION_FILE: ContentCompatibility.PRIMARY,
        DocumentContentType.TRAINING_PLAN: ContentCompatibility.SUPPORTING,
    },
    QuestionType.GENERAL: {
        content_type: ContentCompatibility.SUPPORTING for content_type in DocumentContentType
    },
}


@dataclass(frozen=True)
class DocumentRetrievalProfile:
    document_id: str
    content_type: DocumentContentType
    metadata_confidence: float

    def __post_init__(self) -> None:
        validate_document_id(self.document_id)
        if not 0 <= self.metadata_confidence <= 1:
            raise ValueError("metadata_confidence must be between 0 and 1")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DocumentRetrievalProfile":
        return cls(
            document_id=str(value["document_id"]),
            content_type=DocumentContentType(value["content_type"]),
            metadata_confidence=float(value["metadata_confidence"]),
        )


@dataclass(frozen=True)
class SearchProfile:
    query: str
    question_type: QuestionType
    required_knowledge: str

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must not be empty")
        if not self.required_knowledge.strip():
            raise ValueError("required_knowledge must not be empty")


def content_type_compatibility(
    question_type: QuestionType, content_type: DocumentContentType
) -> ContentCompatibility:
    return _COMPATIBILITY.get(question_type, {}).get(
        content_type, ContentCompatibility.NONE
    )


def build_query_profile_text(profile: SearchProfile) -> str:
    return "\n".join(
        [
            f"Question type: {_QUESTION_LABELS[profile.question_type]}",
            f"Required knowledge: {profile.required_knowledge.strip()}",
            f"Query: {profile.query.strip()}",
        ]
    )


def build_chunk_profile_text(
    *,
    content_type: DocumentContentType,
    section_path: list[str],
    subheadings: list[str],
    chunk_text: str,
) -> str:
    lines = [
        f"Document content type: {_CONTENT_TYPE_LABELS[content_type]}",
        f"Section: {' > '.join(section_path)}",
    ]
    if subheadings:
        lines.append(f"Subheadings: {' | '.join(subheadings)}")
    lines.append(f"Content: {chunk_text.strip()}")
    return "\n".join(lines)


def build_reranker_pair(
    search_profile: SearchProfile,
    *,
    content_type: DocumentContentType,
    section_path: list[str],
    subheadings: list[str],
    chunk_text: str,
) -> tuple[str, str]:
    return (
        build_query_profile_text(search_profile),
        build_chunk_profile_text(
            content_type=content_type,
            section_path=section_path,
            subheadings=subheadings,
            chunk_text=chunk_text,
        ),
    )
