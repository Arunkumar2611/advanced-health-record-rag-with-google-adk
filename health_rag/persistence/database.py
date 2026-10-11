import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from health_rag.config import DATABASE_PATH


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path = DATABASE_PATH) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, full_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    source_name TEXT NOT NULL, mime_type TEXT, storage_path TEXT, report_type TEXT,
                    report_date TEXT, extracted_json TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_documents_user_date ON documents(user_id, report_date DESC);
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    vector_index INTEGER NOT NULL, chunk_index INTEGER NOT NULL, text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL, created_at TEXT NOT NULL,
                    UNIQUE(user_id, vector_index)
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_user_doc ON chunks(user_id, document_id);
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    title TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                    content TEXT NOT NULL, created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_recent ON messages(user_id, conversation_id, created_at DESC);
                CREATE TABLE IF NOT EXISTS long_term_memories (
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    memory_type TEXT NOT NULL, content TEXT NOT NULL, source_document_id TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memories_user_type ON long_term_memories(user_id, memory_type);
            """)
            document_columns = {row["name"] for row in db.execute("PRAGMA table_info(documents)")}
            if "storage_path" not in document_columns:
                db.execute("ALTER TABLE documents ADD COLUMN storage_path TEXT")

    def create_user(self, *, user_id: str, email: str, full_name: str, password_hash: str) -> None:
        with self.connect() as db:
            db.execute("INSERT INTO users(id,email,full_name,password_hash,created_at) VALUES(?,?,?,?,?)",
                       (user_id, email.lower(), full_name, password_hash, now_iso()))

    def user_by_email(self, email: str):
        with self.connect() as db:
            row = db.execute("SELECT * FROM users WHERE email=?", (email.lower(),)).fetchone()
        return dict(row) if row else None

    def user_by_id(self, user_id: str):
        with self.connect() as db:
            row = db.execute("SELECT id,email,full_name,created_at FROM users WHERE id=?", (user_id,)).fetchone()
        return dict(row) if row else None

    def save_document(self, *, document_id: str, user_id: str, source_name: str, mime_type: str | None, storage_path: str, report: dict) -> None:
        report_date = report.get("report_date")
        with self.connect() as db:
            db.execute("INSERT INTO documents(id,user_id,source_name,mime_type,storage_path,report_type,report_date,extracted_json,created_at) VALUES(?,?,?,?,?,?,?,?,?)", (
                document_id, user_id, source_name, mime_type, storage_path, report.get("report_type"),
                report_date, json.dumps(report, ensure_ascii=False), now_iso(),
            ))

    def list_documents(self, user_id: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT id,source_name,mime_type,report_type,report_date,created_at FROM documents WHERE user_id=? ORDER BY created_at DESC", (user_id,)).fetchall()
        return [dict(row) for row in rows]

    def add_chunks(self, *, user_id: str, document_id: str, chunks: list[dict]) -> None:
        with self.connect() as db:
            for chunk in chunks:
                db.execute("INSERT INTO chunks(id,user_id,document_id,vector_index,chunk_index,text,metadata_json,created_at) VALUES(?,?,?,?,?,?,?,?)", (
                    chunk["id"], user_id, document_id, chunk["vector_index"], chunk["chunk_index"],
                    chunk["text"], json.dumps(chunk["metadata"], ensure_ascii=False), now_iso(),
                ))

    def add_report_memories(self, *, user_id: str, document_id: str, report: dict) -> None:
        memories = []
        for diagnosis in report.get("diagnoses", []):
            memories.append(("diagnosis", json.dumps(diagnosis, ensure_ascii=False)))
        for lab in report.get("lab_results", []):
            memories.append(("lab_result", json.dumps(lab, ensure_ascii=False)))
        for medication in report.get("medications", []):
            memories.append(("medication", json.dumps(medication, ensure_ascii=False)))
        for finding in report.get("findings", []):
            memories.append(("clinical_finding", finding))
        with self.connect() as db:
            for memory_type, content in memories:
                db.execute("INSERT INTO long_term_memories(id,user_id,memory_type,content,source_document_id,created_at) VALUES(?,?,?,?,?,?)", (
                    __import__("uuid").uuid4().hex, user_id, memory_type, content, document_id, now_iso(),
                ))

    def create_conversation(self, *, conversation_id: str, user_id: str) -> None:
        with self.connect() as db:
            db.execute("INSERT INTO conversations(id,user_id,title,created_at,updated_at) VALUES(?,?,?,?,?)", (
                conversation_id, user_id, None, now_iso(), now_iso(),
            ))

    def save_message(self, *, conversation_id: str, user_id: str, role: str, content: str) -> None:
        with self.connect() as db:
            exists = db.execute("SELECT 1 FROM conversations WHERE id=? AND user_id=?", (conversation_id, user_id)).fetchone()
            if not exists:
                raise ValueError("Conversation not found for this account")
            db.execute("INSERT INTO messages(id,conversation_id,user_id,role,content,created_at) VALUES(?,?,?,?,?,?)", (
                __import__("uuid").uuid4().hex, conversation_id, user_id, role, content, now_iso(),
            ))
            db.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now_iso(), conversation_id))

    def recent_messages(self, *, conversation_id: str, user_id: str, limit: int) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT role,content,created_at FROM messages WHERE conversation_id=? AND user_id=? ORDER BY created_at DESC LIMIT ?",
                              (conversation_id, user_id, limit)).fetchall()
        return [dict(row) for row in reversed(rows)]

    def user_profile_context(self, user_id: str) -> dict:
        with self.connect() as db:
            docs = db.execute("SELECT id,source_name,report_type,report_date FROM documents WHERE user_id=? ORDER BY COALESCE(report_date,created_at) DESC LIMIT 30",
                              (user_id,)).fetchall()
            diagnoses = db.execute("SELECT content,source_document_id,created_at FROM long_term_memories WHERE user_id=? AND memory_type='diagnosis' ORDER BY created_at DESC LIMIT 50", (user_id,)).fetchall()
            labs = db.execute("SELECT content,source_document_id,created_at FROM long_term_memories WHERE user_id=? AND memory_type='lab_result' ORDER BY created_at DESC LIMIT 60", (user_id,)).fetchall()
            medications = db.execute("SELECT content,source_document_id,created_at FROM long_term_memories WHERE user_id=? AND memory_type='medication' ORDER BY created_at DESC LIMIT 50", (user_id,)).fetchall()
            findings = db.execute("SELECT content,source_document_id,created_at FROM long_term_memories WHERE user_id=? AND memory_type='clinical_finding' ORDER BY created_at DESC LIMIT 40", (user_id,)).fetchall()
        def hydrate(rows):
            return [{**(json.loads(row["content"]) if row["content"].startswith("{") else {"finding": row["content"]}),
                     "source_document_id": row["source_document_id"], "recorded_at": row["created_at"]} for row in rows]
        return {
            "recent_reports": [dict(row) for row in docs],
            "documented_diagnoses": hydrate(diagnoses),
            "lab_results": hydrate(labs),
            "medications_as_recorded": hydrate(medications),
            "documented_findings": hydrate(findings),
        }

    def find_medications(self, user_id: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT content,source_document_id,created_at FROM long_term_memories WHERE user_id=? AND memory_type='medication' ORDER BY created_at DESC", (user_id,)).fetchall()
        return [{**json.loads(row["content"]), "source_document_id": row["source_document_id"], "created_at": row["created_at"]} for row in rows]

    def recent_labs(self, user_id: str, limit: int = 30) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT content,source_document_id,created_at FROM long_term_memories WHERE user_id=? AND memory_type='lab_result' ORDER BY created_at DESC LIMIT ?", (user_id, limit)).fetchall()
        return [{**json.loads(row["content"]), "source_document_id": row["source_document_id"], "created_at": row["created_at"]} for row in rows]
