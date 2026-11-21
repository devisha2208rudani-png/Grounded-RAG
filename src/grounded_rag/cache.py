from __future__ import annotations

"""Simple SQLite-backed cache for embeddings and responses."""

import json
import sqlite3
from pathlib import Path
from typing import Any


class SQLiteCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                namespace TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(namespace, key)
            )
            """
        )
        self._connection.commit()

    def get(self, namespace: str, key: str) -> Any | None:
        cursor = self._connection.execute(
            "SELECT value FROM cache WHERE namespace = ? AND key = ?",
            (namespace, key),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def set(self, namespace: str, key: str, value: Any) -> None:
        serialized = json.dumps(value)
        self._connection.execute(
            """
            INSERT INTO cache(namespace, key, value)
            VALUES (?, ?, ?)
            ON CONFLICT(namespace, key)
            DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP
            """,
            (namespace, key, serialized),
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()
