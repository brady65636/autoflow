from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from .db_models import (
    CapabilityRow,
    EquipmentRow,
    TechnicianRow,
    VehicleRow,
    WorkstationRow,
)
from .repository import (
    create_capability,
    create_equipment,
    create_technician,
    create_vehicle,
    create_workstation,
)

STORE_ID = "vw-store-1"
STANDARD_4S_STORE_ID = "vw-4s-store-001"


def seed_demo(session: Session) -> None:
    if session.get(VehicleRow, "vehicle-magotan-2021") is None:
        create_vehicle(session, "vehicle-magotan-2021", "volkswagen", STORE_ID)
    if session.get(VehicleRow, "vehicle-golf-2022") is None:
        create_vehicle(session, "vehicle-golf-2022", "volkswagen", STORE_ID)
    capabilities = {
        "inspection": "到店检查",
        "engine-diagnosis": "发动机诊断",
        "brake": "制动系统维修",
        "maintenance": "常规保养",
        "electrical-diagnosis": "电气系统诊断",
        "alignment": "四轮定位",
        "quality-inspection": "竣工质量检验",
    }
    for code, name in capabilities.items():
        if session.scalar(select(CapabilityRow).where(CapabilityRow.code == code)) is None:
            create_capability(session, code, name)
    technicians = {
        "tech-wang": ("王师傅", ["inspection", "engine-diagnosis", "brake"]),
        "tech-li": ("李师傅", ["maintenance", "electrical-diagnosis", "alignment"]),
        "tech-qian": ("钱师傅", ["quality-inspection", "brake"]),
    }
    for technician_id, (name, skills) in technicians.items():
        if session.get(TechnicianRow, technician_id) is None:
            create_technician(session, technician_id, name, STORE_ID, skills)
    workstations = {
        "bay-diagnostic-1": ("1号诊断工位", "diagnostic"),
        "bay-lift-1": ("1号举升工位", "lift"),
        "bay-maintenance-1": ("1号保养工位", "maintenance"),
        "bay-alignment-1": ("1号定位工位", "alignment"),
        "bay-inspection-1": ("质检工位", "inspection"),
    }
    for workstation_id, (name, workstation_type) in workstations.items():
        if session.get(WorkstationRow, workstation_id) is None:
            create_workstation(session, workstation_id, name, STORE_ID, workstation_type)
    equipment = {
        "equipment-obd-1": ("大众诊断仪", "obd-scanner"),
        "equipment-align-1": ("四轮定位仪", "alignment-machine"),
        "equipment-multimeter-1": ("数字万用表", "multimeter"),
    }
    for equipment_id, (name, equipment_type) in equipment.items():
        if session.get(EquipmentRow, equipment_id) is None:
            create_equipment(session, equipment_id, name, STORE_ID, equipment_type)


def seed_standard_4s(session: Session) -> None:
    """Insert a reusable single-brand Volkswagen 4S workshop fixture via CRUD functions."""
    if session.get(TechnicianRow, "4s-tech-001") is not None:
        if session.get(EquipmentRow, "4s-equipment-oem-01") is None:
            create_equipment(
                session,
                "4s-equipment-oem-01",
                "大众原厂诊断设备",
                STANDARD_4S_STORE_ID,
                "oem-diagnostic-tool",
            )
        return

    capabilities = {
        "inspection": "到店检查",
        "maintenance": "常规保养",
        "engine-diagnosis": "发动机诊断",
        "electrical-diagnosis": "电气系统诊断",
        "brake": "制动系统维修",
        "alignment": "四轮定位",
        "quality-inspection": "竣工质量检验",
        "ac-service": "空调系统维修",
        "ev-diagnosis": "新能源诊断",
        "oem-diagnostic-tool": "原厂诊断设备操作",
    }
    for code, name in capabilities.items():
        if session.scalar(select(CapabilityRow).where(CapabilityRow.code == code)) is None:
            create_capability(session, code, name)

    # Customer vehicle instances are intentionally not seeded here.
    # Scheduling receives VehicleProfile(category, powertrain) directly.

    technicians = [
        ("4s-tech-001", "张师傅", ["inspection", "maintenance", "brake"]),
        ("4s-tech-002", "王师傅", ["engine-diagnosis", "oem-diagnostic-tool"]),
        ("4s-tech-003", "李师傅", ["electrical-diagnosis", "oem-diagnostic-tool"]),
        ("4s-tech-004", "赵师傅", ["brake", "maintenance", "alignment"]),
        ("4s-tech-005", "刘师傅", ["engine-diagnosis", "electrical-diagnosis"]),
        ("4s-tech-006", "陈师傅", ["ac-service", "electrical-diagnosis"]),
        ("4s-tech-007", "孙师傅", ["ev-diagnosis", "electrical-diagnosis"]),
        ("4s-tech-008", "周检验", ["quality-inspection"]),
    ]
    for technician_id, name, skills in technicians:
        create_technician(session, technician_id, name, STANDARD_4S_STORE_ID, skills)

    workstations = [
        ("4s-bay-quick-01", "快修工位1", "quick-service"),
        ("4s-bay-quick-02", "快修工位2", "quick-service"),
        ("4s-bay-lift-01", "机电举升工位1", "lift"),
        ("4s-bay-lift-02", "机电举升工位2", "lift"),
        ("4s-bay-diagnostic-01", "综合诊断工位", "diagnostic"),
        ("4s-bay-alignment-01", "四轮定位工位", "alignment"),
        ("4s-bay-ev-01", "新能源专用工位", "ev-diagnostic"),
        ("4s-bay-inspection-01", "竣工质检工位", "inspection"),
    ]
    for workstation_id, name, workstation_type in workstations:
        create_workstation(
            session, workstation_id, name, STANDARD_4S_STORE_ID, workstation_type
        )

    equipment = [
        ("4s-equipment-obd-01", "大众原厂诊断仪1", "obd-scanner"),
        ("4s-equipment-obd-02", "大众原厂诊断仪2", "obd-scanner"),
        ("4s-equipment-oem-01", "大众原厂诊断设备", "oem-diagnostic-tool"),
        ("4s-equipment-align-01", "四轮定位仪", "alignment-machine"),
        ("4s-equipment-multimeter-01", "数字万用表", "multimeter"),
        ("4s-equipment-ac-01", "空调冷媒回收加注机", "ac-machine"),
        ("4s-equipment-ev-01", "新能源绝缘检测设备", "ev-insulation-tester"),
    ]
    for equipment_id, name, equipment_type in equipment:
        create_equipment(
            session, equipment_id, name, STANDARD_4S_STORE_ID, equipment_type
        )
