"""Small SQLite-backed stores for conversation memory and response caching."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PersistentMemory:
    """Keep recent conversation turns and short, user-approved facts locally."""

    def __init__(self, database_path: str = "./agentic_rag_memory.sqlite3") -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _create_tables(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_turns_user_time
                    ON conversation_turns(user_id, created_at);
                CREATE TABLE IF NOT EXISTS user_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    memory TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(user_id, memory)
                );
                CREATE TABLE IF NOT EXISTS response_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    rating TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def add_turn(self, user_id: str, question: str, answer: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO conversation_turns(user_id, question, answer, created_at) VALUES (?, ?, ?, ?)",
                (user_id, question, answer, _utc_now()),
            )

    def recent_turns(self, user_id: str, limit: int = 6) -> list[dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT question, answer FROM conversation_turns WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def add_memory(self, user_id: str, memory: str) -> None:
        memory = memory.strip()
        if not memory or len(memory) > 300:
            return
        with self._connect() as connection:
            connection.execute(
                """INSERT INTO user_memories(user_id, memory, created_at, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(user_id, memory) DO UPDATE SET updated_at = excluded.updated_at""",
                (user_id, memory, _utc_now(), _utc_now()),
            )

    def auto_capture(self, user_id: str, question: str) -> str | None:
        """Save an explicit personal fact from the question without another LLM call."""
        text = " ".join(question.strip().split())
        patterns = (
            r"\b(?:my name is|i prefer|i like|i use|i work on|i am working on|i need|remember that)\b.*",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                fact = match.group(0).rstrip(".!? ")
                self.add_memory(user_id, fact[0].upper() + fact[1:])
                return fact
        return None

    def memories(self, user_id: str, limit: int = 20) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT memory FROM user_memories WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [row["memory"] for row in rows]

    def memories_text(self, user_id: str) -> str:
        """Durable facts only — stable across turns, safe to use as a cache-key input."""
        memories = self.memories(user_id)
        return "Permanent user memories:\n- " + "\n- ".join(memories) if memories else ""

    def recent_turns_text(self, user_id: str) -> str:
        """Recent conversation only — changes every turn, used for generation, not caching."""
        turns = self.recent_turns(user_id)
        if not turns:
            return ""
        recent = []
        for turn in turns:
            question = turn["question"][:600]
            answer = turn["answer"][:1600]
            recent.append(f"User: {question}\nAssistant: {answer}")
        return "Recent conversation:\n" + "\n".join(recent)

    def clear_user(self, user_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM conversation_turns WHERE user_id = ?", (user_id,))
            connection.execute("DELETE FROM user_memories WHERE user_id = ?", (user_id,))

    def add_feedback(self, user_id: str, rating: str) -> None:
        if rating not in {"helpful", "not helpful"}:
            return
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO response_feedback(user_id, rating, created_at) VALUES (?, ?, ?)",
                (user_id, rating, _utc_now()),
            )

    def context(self, user_id: str) -> str:
        sections = [text for text in (self.memories_text(user_id), self.recent_turns_text(user_id)) if text]
        return "\n\n".join(sections)

    @staticmethod
    def cache_key(
        user_id: str,
        question: str,
        document_version: str,
        model: str,
        memory_context: str = "",
    ) -> str:
        raw = "|".join(
            (user_id, question.strip().lower(), document_version, model, memory_context)
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class ResponseCache:
    """Persistent cache stored in the same local SQLite database.

    Bounded so it can't grow forever: entries older than `ttl_seconds` are
    treated as misses, and the table is pruned back to `max_entries` (oldest
    first) on every write.
    """

    def __init__(
        self,
        memory: PersistentMemory,
        ttl_seconds: int = 60 * 60 * 24 * 7,
        max_entries: int = 2000,
    ) -> None:
        self.memory = memory
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        with self.memory._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS response_cache (
                       cache_key TEXT PRIMARY KEY,
                       payload TEXT NOT NULL,
                       created_at TEXT NOT NULL
                   )"""
            )

    def get(self, key: str) -> dict[str, Any] | None:
        with self.memory._connect() as connection:
            row = connection.execute(
                "SELECT payload, created_at FROM response_cache WHERE cache_key = ?", (key,)
            ).fetchone()
        if not row:
            return None
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(row["created_at"])).total_seconds()
        if age > self.ttl_seconds:
            with self.memory._connect() as connection:
                connection.execute("DELETE FROM response_cache WHERE cache_key = ?", (key,))
            return None
        return json.loads(row["payload"])

    def put(self, key: str, payload: dict[str, Any]) -> None:
        with self.memory._connect() as connection:
            connection.execute(
                """INSERT INTO response_cache(cache_key, payload, created_at) VALUES (?, ?, ?)
                   ON CONFLICT(cache_key) DO UPDATE SET payload = excluded.payload, created_at = excluded.created_at""",
                (key, json.dumps(payload), _utc_now()),
            )
            # Evict the oldest rows once we're over the cap, so a long-running
            # server doesn't grow the SQLite file without bound.
            connection.execute(
                """DELETE FROM response_cache WHERE cache_key NOT IN (
                       SELECT cache_key FROM response_cache ORDER BY created_at DESC LIMIT ?
                   )""",
                (self.max_entries,),
            )

    def clear(self) -> None:
        with self.memory._connect() as connection:
            connection.execute("DELETE FROM response_cache")
