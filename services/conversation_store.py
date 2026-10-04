import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


DATABASE_PATH = Path(
    os.getenv(
        "RAG_CONVERSATION_DB_PATH",
        Path(__file__).resolve().parents[1] / "storage" / "conversations.sqlite3"
    )
)
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def _connection():
    connection = sqlite3.connect(DATABASE_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _initialize_database():
    with _connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                sources_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
            );

            CREATE INDEX IF NOT EXISTS messages_conversation_id_idx
                ON messages(conversation_id, id);
            """
        )


def create_conversation(document_id):
    conversation_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    with _connection() as connection:
        connection.execute(
            "INSERT INTO conversations (id, document_id, created_at) VALUES (?, ?, ?)",
            (conversation_id, document_id, created_at)
        )
    return conversation_id


def get_conversation(conversation_id):
    with _connection() as connection:
        row = connection.execute(
            "SELECT id, document_id, created_at FROM conversations WHERE id = ?",
            (conversation_id,)
        ).fetchone()
    return dict(row) if row else None


def add_message(conversation_id, role, content, sources=None):
    created_at = datetime.now(timezone.utc).isoformat()
    with _connection() as connection:
        connection.execute(
            """
            INSERT INTO messages
                (conversation_id, role, content, sources_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, role, content, json.dumps(sources or []), created_at)
        )


def get_messages(conversation_id, limit=None):
    query = """
        SELECT id, role, content, sources_json, created_at
        FROM messages
        WHERE conversation_id = ?
        ORDER BY id DESC
    """
    parameters = [conversation_id]
    if limit is not None:
        query += " LIMIT ?"
        parameters.append(limit)

    with _connection() as connection:
        rows = connection.execute(query, parameters).fetchall()

    return [
        {
            "id": row["id"],
            "role": row["role"],
            "content": row["content"],
            "sources": json.loads(row["sources_json"]),
            "created_at": row["created_at"]
        }
        for row in reversed(rows)
    ]


_initialize_database()