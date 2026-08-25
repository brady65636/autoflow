from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TimeInterval(BaseModel):
    model_config = ConfigDict(frozen=True)

    start: datetime
    end: datetime

    @model_validator(mode="after")
    def valid_order(self) -> "TimeInterval":
        if self.start >= self.end:
            raise ValueError("time interval must be non-empty and start before end")
        return self

    def overlaps(self, other: "TimeInterval") -> bool:
        """Left-closed/right-open overlap: touching endpoints do not conflict."""
        return self.start < other.end and other.start < self.end


class EffectiveAbility(BaseModel):
    skill: str
    valid_from: datetime | None = None
    valid_until: datetime | None = None

    def is_valid_at(self, interval: TimeInterval) -> bool:
        return (self.valid_from is None or self.valid_from <= interval.start) and (
            self.valid_until is None or interval.end <= self.valid_until
        )


class VehicleCategory(StrEnum):
    SEDAN = "sedan"
    SPORTS_CAR = "sports_car"
    SUV = "suv"
    MPV = "mpv"
    WAGON = "wagon"
    HATCHBACK = "hatchback"
    PICKUP = "pickup"
    COMMERCIAL = "commercial"


class PowertrainType(StrEnum):
    ICE = "ice"
    HYBRID = "hybrid"
    EV = "ev"


class VehicleProfile(BaseModel):
    """Scheduling attributes only; this is not a customer vehicle record."""

    category: VehicleCategory = VehicleCategory.SEDAN
    powertrain: PowertrainType = PowertrainType.ICE


class Vehicle(BaseModel):
    """Future customer vehicle aggregate, not required by profile-based scheduling."""

    id: str
    brand: str
    store_id: str
    profile: VehicleProfile = Field(default_factory=VehicleProfile)


class Technician(BaseModel):
    id: str
    name: str
    store_id: str
    abilities: list[EffectiveAbility] = Field(default_factory=list)
    availability: list[TimeInterval] = Field(default_factory=list)

    def can_do(self, skills: set[str], interval: TimeInterval) -> bool:
        return skills.issubset({a.skill for a in self.abilities if a.is_valid_at(interval)})


class Workstation(BaseModel):
    id: str
    name: str
    store_id: str
    workstation_type: str
    availability: list[TimeInterval] = Field(default_factory=list)


class Equipment(BaseModel):
    id: str
    name: str
    store_id: str
    equipment_type: str
    availability: list[TimeInterval] = Field(default_factory=list)


class ResourceReservation(BaseModel):
    resource_type: Literal["technician", "workstation", "equipment"]
    resource_id: str
    interval: TimeInterval
    task_id: str


class ServiceOperation(BaseModel):
    code: str
    name: str
    brand: str
    store_id: str
    duration_minutes: int = Field(gt=0)
    required_skills: set[str] = Field(default_factory=set)
    required_workstation_types: set[str] = Field(default_factory=set)
    required_equipment_types: set[str] = Field(default_factory=set)
    allowed_vehicle_categories: set[VehicleCategory] = Field(default_factory=set)


class TaskRequirement(BaseModel):
    task_id: str
    brand: str
    store_id: str
    vehicle_profile: VehicleProfile = Field(default_factory=VehicleProfile)
    duration_minutes: int = Field(gt=0)
    earliest_start: datetime
    latest_end: datetime
    required_skills: set[str] = Field(default_factory=set)
    required_workstation_types: set[str] = Field(default_factory=set)
    required_equipment_types: set[str] = Field(default_factory=set)
    allowed_technician_ids: set[str] | None = None

    @model_validator(mode="after")
    def valid_window(self) -> "TaskRequirement":
        if self.earliest_start >= self.latest_end:
            raise ValueError("latest_end must be after earliest_start")
        return self


class CandidatePlan(BaseModel):
    task_id: str
    interval: TimeInterval
    technician_id: str
    workstation_id: str
    equipment_ids: list[str] = Field(default_factory=list)

    @property
    def resource_ids(self) -> set[str]:
        return {self.technician_id, self.workstation_id, *self.equipment_ids}


class InfeasibilityReason(StrEnum):
    SCOPE_MISMATCH = "scope_mismatch"
    NO_QUALIFIED_TECHNICIAN = "no_qualified_technician"
    NO_COMPATIBLE_WORKSTATION = "no_compatible_workstation"
    NO_REQUIRED_EQUIPMENT = "no_required_equipment"
    WINDOW_TOO_SHORT = "window_too_short"
    RESOURCE_CONFLICT = "resource_conflict"
    NO_AVAILABLE_TIME = "no_available_time"


class ReasonDetail(BaseModel):
    code: InfeasibilityReason
    message: str


class SchedulingResult(BaseModel):
    status: Literal["FEASIBLE", "INFEASIBLE"]
    plan: CandidatePlan | None = None
    reasons: list[ReasonDetail] = Field(default_factory=list)
