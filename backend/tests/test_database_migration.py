import sqlite3

from autoflow_scheduling.database import create_session_factory


def test_legacy_sqlite_resource_reservations_are_upgraded(tmp_path):
    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    connection.execute(
        """CREATE TABLE resource_reservations (
        id INTEGER PRIMARY KEY,
        resource_type VARCHAR(32), resource_id VARCHAR(64),
        vehicle_id VARCHAR(64), task_id VARCHAR(64),
        start_time DATETIME, end_time DATETIME
        )"""
    )
    connection.execute(
        "INSERT INTO resource_reservations VALUES (1, 'technician', 't1', 'v1', 'old', NULL, NULL)"
    )
    connection.commit()
    connection.close()

    create_session_factory(f"sqlite+pysqlite:///{path}")
    connection = sqlite3.connect(path)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(resource_reservations)")}
    status = connection.execute(
        "SELECT status FROM resource_reservations WHERE id = 1"
    ).fetchone()[0]
    connection.close()
    assert {"work_order_id", "status"} <= columns
    assert status == "ACTIVE"


def test_legacy_knowledge_tables_gain_hash_columns_without_losing_rows(tmp_path):
    path = tmp_path / "legacy-knowledge.db"
    connection = sqlite3.connect(path)
    connection.execute(
        "CREATE TABLE knowledge_documents "
        "(id VARCHAR(128) PRIMARY KEY, source_sha256 VARCHAR(64))"
    )
    connection.execute(
        "CREATE TABLE knowledge_sections "
        "(id VARCHAR(160) PRIMARY KEY, document_id VARCHAR(128), text TEXT)"
    )
    connection.execute(
        "CREATE TABLE knowledge_chunks "
        "(id VARCHAR(192) PRIMARY KEY, document_id VARCHAR(128), text TEXT)"
    )
    connection.execute("INSERT INTO knowledge_documents VALUES ('PDF-001', ?)", ("a" * 64,))
    connection.execute("INSERT INTO knowledge_sections VALUES ('PDF-001:s0001', 'PDF-001', 's')")
    connection.execute("INSERT INTO knowledge_chunks VALUES ('PDF-001:s0001:c001', 'PDF-001', 'c')")
    connection.commit()
    connection.close()

    create_session_factory(f"sqlite+pysqlite:///{path}")
    connection = sqlite3.connect(path)
    document_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(knowledge_documents)")
    }
    section_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(knowledge_sections)")
    }
    chunk_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(knowledge_chunks)")
    }
    document = connection.execute(
        "SELECT id, source_sha256, artifact_sha256 FROM knowledge_documents"
    ).fetchone()
    connection.close()

    assert {"artifact_sha256", "hash_algorithm", "hash_verified_at"} <= document_columns
    assert "content_sha256" in section_columns
    assert "content_sha256" in chunk_columns
    assert document == ("PDF-001", "a" * 64, None)
