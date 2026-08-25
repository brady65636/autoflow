from __future__ import annotations

import os
from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


def create_session_factory(url: str | None = None) -> sessionmaker[Session]:
    url = url or os.getenv("AUTOFLOW_DATABASE_URL", "sqlite:///./autoflow.db")
    connect_args = (
        {"check_same_thread": False, "timeout": 30}
        if url.startswith("sqlite")
        else {}
    )
    engine = create_engine(url, connect_args=connect_args)
    Base.metadata.create_all(engine)
    _upgrade_local_sqlite_schema(engine)
    return sessionmaker(engine, expire_on_commit=False)


def _upgrade_local_sqlite_schema(engine) -> None:
    """Apply the small, idempotent schema additions needed by the local MVP."""
    if engine.dialect.name != "sqlite":
        return
    with engine.begin() as connection:
        inspector = inspect(engine)
        table_names = set(inspector.get_table_names())
        if "resource_reservations" in table_names:
            columns = {
                column["name"]
                for column in inspector.get_columns("resource_reservations")
            }
            if "work_order_id" not in columns:
                connection.execute(
                    text(
                        "ALTER TABLE resource_reservations "
                        "ADD COLUMN work_order_id VARCHAR(64)"
                    )
                )
            if "status" not in columns:
                connection.execute(
                    text(
                        "ALTER TABLE resource_reservations "
                        "ADD COLUMN status VARCHAR(32) DEFAULT 'ACTIVE'"
                    )
                )
            connection.execute(
                text("UPDATE resource_reservations SET status = 'ACTIVE' WHERE status IS NULL")
            )

        if "work_orders" in table_names:
            columns = {column["name"] for column in inspect(engine).get_columns("work_orders")}
            if "customer_user_id" not in columns:
                connection.execute(
                    text(
                        "ALTER TABLE work_orders "
                        "ADD COLUMN customer_user_id VARCHAR(128)"
                    )
                )
        if "confirmed_plans" in table_names:
            columns = {column["name"] for column in inspect(engine).get_columns("confirmed_plans")}
            if "customer_user_id" not in columns:
                connection.execute(
                    text(
                        "ALTER TABLE confirmed_plans "
                        "ADD COLUMN customer_user_id VARCHAR(128)"
                    )
                )

        schema_additions = {
            "confirmed_plans": {
                "work_order_id": "VARCHAR(64)",
                "root_plan_id": "VARCHAR(64)",
                "revision": "INTEGER DEFAULT 1",
                "supersedes_plan_id": "VARCHAR(64)",
                "confirmed_at": "DATETIME",
                "applied_at": "DATETIME",
                "superseded_at": "DATETIME",
                "cancelled_at": "DATETIME",
            },
            "work_orders": {"current_plan_id": "VARCHAR(64)"},
            "resource_reservations": {
                "plan_id": "VARCHAR(64)",
                "replaced_by_reservation_id": "INTEGER",
                "cancelled_at": "DATETIME",
            },
        }
        for table_name, additions in schema_additions.items():
            if table_name not in table_names:
                continue
            columns = {column["name"] for column in inspect(engine).get_columns(table_name)}
            for column_name, definition in additions.items():
                if column_name not in columns:
                    connection.execute(
                        text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")
                    )

        # Remove legacy vehicle ownership columns. Vehicle profile is now the
        # only scheduling/work-order vehicle context.
        for table_name in ("work_orders", "resource_reservations"):
            if table_name not in table_names:
                continue
            columns = {column["name"] for column in inspect(engine).get_columns(table_name)}
            if "vehicle_id" in columns:
                _rebuild_without_vehicle_id(connection, table_name)

        # Migrate the previous username-based ownership when both legacy and
        # new columns exist. New writes only use the immutable user ID.
        for table_name in ("confirmed_plans", "work_orders"):
            if table_name not in table_names:
                continue
            columns = {column["name"] for column in inspect(engine).get_columns(table_name)}
            if {"customer_username", "customer_user_id"} <= columns and "users" in table_names:
                connection.execute(
                    text(
                        f"UPDATE {table_name} SET customer_user_id = "
                        "(SELECT id FROM users WHERE users.username = "
                        f"{table_name}.customer_username) "
                        "WHERE customer_user_id IS NULL AND customer_username IS NOT NULL"
                    )
                )

        hash_columns = {
            "knowledge_documents": {
                "artifact_sha256": "VARCHAR(64)",
                "hash_algorithm": "VARCHAR(16) DEFAULT 'sha256'",
                "hash_verified_at": "DATETIME",
            },
            "knowledge_sections": {"content_sha256": "VARCHAR(64)"},
            "knowledge_chunks": {
                "content_sha256": "VARCHAR(64)",
                "embedding": "BLOB",
                "embedding_model": "VARCHAR(512)",
                "embedding_dimension": "INTEGER",
            },
        }
        for table_name, additions in hash_columns.items():
            if table_name not in table_names:
                continue
            columns = {
                column["name"] for column in inspect(engine).get_columns(table_name)
            }
            for column_name, definition in additions.items():
                if column_name not in columns:
                    connection.execute(
                        text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")
                    )
        if "knowledge_documents" in table_names:
            connection.execute(
                text(
                    "UPDATE knowledge_documents SET hash_algorithm = 'sha256' "
                    "WHERE hash_algorithm IS NULL"
                )
            )


def _rebuild_without_vehicle_id(connection, table_name: str) -> None:
    """Rebuild legacy SQLite tables whose vehicle_id was a foreign key."""
    if table_name == "work_orders":
        definition = """
            id VARCHAR(64) PRIMARY KEY,
            order_no VARCHAR(32) NOT NULL UNIQUE,
            customer_user_id VARCHAR(128),
            store_id VARCHAR(64) NOT NULL,
            service_code VARCHAR(128) NOT NULL,
            service_name VARCHAR(128) NOT NULL,
            service_summary VARCHAR(2000) NOT NULL,
            scheduled_start DATETIME NOT NULL,
            scheduled_end DATETIME NOT NULL,
            technician_id VARCHAR(64) NOT NULL,
            workstation_id VARCHAR(64) NOT NULL,
            status VARCHAR(32) NOT NULL,
            customer_note VARCHAR(2000) NOT NULL,
            ai_service_summary VARCHAR(2000) NOT NULL,
            idempotency_key VARCHAR(128) NOT NULL UNIQUE,
            version INTEGER NOT NULL,
            created_at DATETIME NOT NULL,
            updated_at DATETIME NOT NULL,
            checked_in_at DATETIME,
            completed_at DATETIME,
            cancelled_at DATETIME
        """
        indexes = (
            "CREATE INDEX IF NOT EXISTS ix_work_orders_customer_user_id "
            "ON work_orders(customer_user_id)",
            "CREATE INDEX IF NOT EXISTS ix_work_orders_store_id ON work_orders(store_id)",
            "CREATE INDEX IF NOT EXISTS ix_work_orders_status ON work_orders(status)",
        )
    elif table_name == "resource_reservations":
        definition = """
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resource_type VARCHAR(32) NOT NULL,
            resource_id VARCHAR(64) NOT NULL,
            task_id VARCHAR(64) NOT NULL,
            work_order_id VARCHAR(64),
            start_time DATETIME NOT NULL,
            end_time DATETIME NOT NULL,
            status VARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
            FOREIGN KEY(work_order_id) REFERENCES work_orders(id)
        """
        indexes = (
            "CREATE INDEX IF NOT EXISTS ix_resource_reservations_resource_type "
            "ON resource_reservations(resource_type)",
            "CREATE INDEX IF NOT EXISTS ix_resource_reservations_resource_id "
            "ON resource_reservations(resource_id)",
            "CREATE INDEX IF NOT EXISTS ix_resource_reservations_task_id "
            "ON resource_reservations(task_id)",
            "CREATE INDEX IF NOT EXISTS ix_resource_reservations_work_order_id "
            "ON resource_reservations(work_order_id)",
            "CREATE INDEX IF NOT EXISTS ix_resource_reservations_status "
            "ON resource_reservations(status)",
        )
    else:
        raise ValueError(f"unsupported legacy table: {table_name}")

    legacy = f"{table_name}_legacy_vehicle"
    connection.execute(text(f"ALTER TABLE {table_name} RENAME TO {legacy}"))
    connection.execute(text(f"CREATE TABLE {table_name} ({definition})"))
    new_columns = [
        column["name"]
        for column in inspect(connection).get_columns(table_name)
    ]
    old_columns = {
        column["name"]
        for column in inspect(connection).get_columns(legacy)
    }
    common = [name for name in new_columns if name in old_columns]
    columns = ", ".join(common)
    select_columns = columns
    if table_name == "resource_reservations":
        select_columns = select_columns.replace(
            "start_time", "COALESCE(start_time, '1970-01-01 00:00:00') AS start_time"
        ).replace(
            "end_time", "COALESCE(end_time, '1970-01-01 00:00:00') AS end_time"
        ).replace(
            "status", "COALESCE(status, 'ACTIVE') AS status"
        )
    connection.execute(
        text(
            f"INSERT INTO {table_name} ({columns}) "
            f"SELECT {select_columns} FROM {legacy}"
        )
    )
    connection.execute(text(f"DROP TABLE {legacy}"))
    for statement in indexes:
        connection.execute(text(statement))


def session_scope(factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    with factory() as session:
        yield session
