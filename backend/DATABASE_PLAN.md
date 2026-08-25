# SQLite 数据库第一阶段计划

## 目标

建立最小可运行闭环：

```text
SQLite
  → SQLAlchemy ORM
  → CRUD / Seed
  → 读取资源和资源预留
  → 转换为调度领域模型
  → First-Fit 调度
  → 写入资源预留
```

## 第一阶段实体

- `vehicles`
- `technicians`
- `capabilities`
- `technician_capabilities`（工程师与能力的多对多关联，包含有效期和状态）
- `workstations`
- `equipment`
- `resource_reservations`

## 暂不实现

- Alembic 迁移
- FastAPI
- 用户鉴权
- Agent / LLM
- CP-SAT
- 工单状态机
- PostgreSQL

SQLite 文件仅用于本地开发和测试；后续切换 PostgreSQL 时保留 Repository 和 Service 边界。
