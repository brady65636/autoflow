from __future__ import annotations

from datetime import datetime

from sqlalchemy import Float, ForeignKey, LargeBinary, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class KnowledgeDocumentRow(Base):
    __tablename__ = "knowledge_documents"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    filename: Mapped[str | None] = mapped_column(String(512), nullable=True)
    source_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    content_type: Mapped[str] = mapped_column(String(64), index=True)
    metadata_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    pipeline_version: Mapped[int | None] = mapped_column(nullable=True, index=True)
    pipeline_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    artifact_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    hash_algorithm: Mapped[str] = mapped_column(String(16), default="sha256")
    hash_verified_at: Mapped[datetime | None] = mapped_column(nullable=True)
    page_count: Mapped[int] = mapped_column(default=0)
    section_count: Mapped[int] = mapped_column(default=0)
    chunk_count: Mapped[int] = mapped_column(default=0)
    ingestion_status: Mapped[str] = mapped_column(String(32), default="complete", index=True)
    imported_at: Mapped[datetime]
    updated_at: Mapped[datetime]
    sections: Mapped[list["KnowledgeSectionRow"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    chunks: Mapped[list["KnowledgeChunkRow"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class KnowledgeSectionRow(Base):
    __tablename__ = "knowledge_sections"
    id: Mapped[str] = mapped_column(String(160), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_documents.id"), index=True
    )
    title: Mapped[str] = mapped_column(Text)
    path_json: Mapped[str] = mapped_column(Text, default="[]")
    level: Mapped[int | None] = mapped_column(nullable=True)
    page_start: Mapped[int | None] = mapped_column(nullable=True, index=True)
    page_end: Mapped[int | None] = mapped_column(nullable=True)
    text: Mapped[str] = mapped_column(Text)
    subheadings_json: Mapped[str] = mapped_column(Text, default="[]")
    quality_json: Mapped[str] = mapped_column(Text, default="{}")
    boundary_json: Mapped[str] = mapped_column(Text, default="{}")
    content_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    document: Mapped["KnowledgeDocumentRow"] = relationship(back_populates="sections")
    chunks: Mapped[list["KnowledgeChunkRow"]] = relationship(
        back_populates="section", cascade="all, delete-orphan"
    )


class KnowledgeChunkRow(Base):
    __tablename__ = "knowledge_chunks"
    id: Mapped[str] = mapped_column(String(192), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_documents.id"), index=True
    )
    section_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_sections.id"), index=True
    )
    chunk_index: Mapped[int] = mapped_column(index=True)
    content_type: Mapped[str] = mapped_column(String(64), index=True)
    metadata_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    section_title: Mapped[str | None] = mapped_column(Text, nullable=True)
    section_path_json: Mapped[str] = mapped_column(Text, default="[]")
    section_level: Mapped[int | None] = mapped_column(nullable=True)
    section_page_start: Mapped[int | None] = mapped_column(nullable=True)
    section_page_end: Mapped[int | None] = mapped_column(nullable=True)
    subheading: Mapped[str | None] = mapped_column(Text, nullable=True)
    subheadings_json: Mapped[str] = mapped_column(Text, default="[]")
    text: Mapped[str] = mapped_column(Text)
    index_text: Mapped[str] = mapped_column(Text)
    quality_json: Mapped[str] = mapped_column(Text, default="{}")
    boundary_json: Mapped[str] = mapped_column(Text, default="{}")
    parser: Mapped[str | None] = mapped_column(String(256), nullable=True)
    splitter: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    content_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(512), nullable=True)
    embedding_dimension: Mapped[int | None] = mapped_column(nullable=True)
    document: Mapped["KnowledgeDocumentRow"] = relationship(back_populates="chunks")
    section: Mapped["KnowledgeSectionRow"] = relationship(back_populates="chunks")


class UserRow(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    username: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    role: Mapped[str] = mapped_column(String(32), index=True)
    created_at: Mapped[datetime]


class VehicleRow(Base):
    __tablename__ = "vehicles"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    brand: Mapped[str] = mapped_column(String(64))
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    category: Mapped[str] = mapped_column(String(32), default="sedan")
    powertrain: Mapped[str] = mapped_column(String(32), default="ice")
    model_year: Mapped[int | None] = mapped_column(nullable=True)
    store_id: Mapped[str] = mapped_column(String(64), index=True)


class TechnicianRow(Base):
    __tablename__ = "technicians"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    store_id: Mapped[str] = mapped_column(String(64), index=True)
    capability_links: Mapped[list["TechnicianCapabilityRow"]] = relationship(
        back_populates="technician", cascade="all, delete-orphan", lazy="selectin"
    )


class CapabilityRow(Base):
    __tablename__ = "capabilities"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128))
    technician_links: Mapped[list["TechnicianCapabilityRow"]] = relationship(
        back_populates="capability"
    )


class TechnicianCapabilityRow(Base):
    __tablename__ = "technician_capabilities"
    technician_id: Mapped[str] = mapped_column(
        ForeignKey("technicians.id"), primary_key=True
    )
    capability_id: Mapped[int] = mapped_column(
        ForeignKey("capabilities.id"), primary_key=True
    )
    valid_from: Mapped[datetime | None]
    valid_until: Mapped[datetime | None]
    status: Mapped[str] = mapped_column(String(32), default="ACTIVE")
    technician: Mapped["TechnicianRow"] = relationship(back_populates="capability_links")
    capability: Mapped["CapabilityRow"] = relationship(back_populates="technician_links")


class WorkstationRow(Base):
    __tablename__ = "workstations"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    store_id: Mapped[str] = mapped_column(String(64), index=True)
    workstation_type: Mapped[str] = mapped_column(String(64), index=True)


class EquipmentRow(Base):
    __tablename__ = "equipment"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    store_id: Mapped[str] = mapped_column(String(64), index=True)
    equipment_type: Mapped[str] = mapped_column(String(64), index=True)


class ConfirmedPlanRow(Base):
    __tablename__ = "confirmed_plans"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    plan_json: Mapped[str] = mapped_column(Text)
    requirement_json: Mapped[str] = mapped_column(Text)
    confirmation_token_hash: Mapped[str] = mapped_column(String(128))
    customer_user_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    work_order_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    root_plan_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    revision: Mapped[int] = mapped_column(default=1)
    supersedes_plan_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="CONFIRMED", index=True)
    expires_at: Mapped[datetime | None] = mapped_column(nullable=True)
    consumed_at: Mapped[datetime | None] = mapped_column(nullable=True)
    confirmed_at: Mapped[datetime | None] = mapped_column(nullable=True)
    applied_at: Mapped[datetime | None] = mapped_column(nullable=True)
    superseded_at: Mapped[datetime | None] = mapped_column(nullable=True)
    cancelled_at: Mapped[datetime | None] = mapped_column(nullable=True)
    created_at: Mapped[datetime]


class WorkOrderRow(Base):
    __tablename__ = "work_orders"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    order_no: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    customer_user_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    current_plan_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    store_id: Mapped[str] = mapped_column(String(64), index=True)
    service_code: Mapped[str] = mapped_column(String(128))
    service_name: Mapped[str] = mapped_column(String(128))
    service_summary: Mapped[str] = mapped_column(String(2000), default="")
    scheduled_start: Mapped[datetime]
    scheduled_end: Mapped[datetime]
    technician_id: Mapped[str] = mapped_column(String(64))
    workstation_id: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), index=True)
    customer_note: Mapped[str] = mapped_column(String(2000), default="")
    ai_service_summary: Mapped[str] = mapped_column(String(2000), default="")
    idempotency_key: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    version: Mapped[int] = mapped_column(default=1)
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]
    checked_in_at: Mapped[datetime | None] = mapped_column(nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(nullable=True)
    cancelled_at: Mapped[datetime | None] = mapped_column(nullable=True)
    reservations: Mapped[list["ResourceReservationRow"]] = relationship(
        back_populates="work_order"
    )
    events: Mapped[list["WorkOrderEventRow"]] = relationship(
        back_populates="work_order", cascade="all, delete-orphan"
    )


class WorkOrderEventRow(Base):
    __tablename__ = "work_order_events"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    work_order_id: Mapped[str] = mapped_column(ForeignKey("work_orders.id"), index=True)
    event_type: Mapped[str] = mapped_column(String(64))
    from_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    to_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    actor_role: Mapped[str] = mapped_column(String(32))
    comment: Mapped[str] = mapped_column(String(1000), default="")
    created_at: Mapped[datetime]
    work_order: Mapped["WorkOrderRow"] = relationship(back_populates="events")


class ResourceReservationRow(Base):
    __tablename__ = "resource_reservations"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    resource_type: Mapped[str] = mapped_column(String(32), index=True)
    resource_id: Mapped[str] = mapped_column(String(64), index=True)
    task_id: Mapped[str] = mapped_column(String(64), index=True)
    plan_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    replaced_by_reservation_id: Mapped[int | None] = mapped_column(nullable=True)
    cancelled_at: Mapped[datetime | None] = mapped_column(nullable=True)
    work_order_id: Mapped[str | None] = mapped_column(
        ForeignKey("work_orders.id"), nullable=True, index=True
    )
    start_time: Mapped[datetime]
    end_time: Mapped[datetime]
    status: Mapped[str] = mapped_column(String(32), default="ACTIVE", index=True)
    work_order: Mapped["WorkOrderRow | None"] = relationship(back_populates="reservations")
