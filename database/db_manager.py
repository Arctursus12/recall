import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

class DatabaseManager:
    def __init__(self, db_path: str = "database/recall.db"):
        self.db_path = db_path
        self.conn = None
        self._init_db()

    def _init_db(self):
        """Initialize database with schema"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

        # Load and execute schema
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path, 'r') as f:
            self.conn.executescript(f.read())
        self.conn.commit()

    def save_conversation(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        metadata: Optional[Dict] = None
    ) -> int:
        """Save a conversation exchange"""
        cursor = self.conn.cursor()

        # Ensure session exists
        cursor.execute(
            "INSERT OR IGNORE INTO sessions (session_id) VALUES (?)",
            (session_id,)
        )

        # Update session last_active
        cursor.execute(
            "UPDATE sessions SET last_active = ? WHERE session_id = ?",
            (datetime.now().isoformat(), session_id)
        )

        # Save conversation
        cursor.execute(
            """INSERT INTO conversations
               (session_id, user_message, assistant_response, metadata)
               VALUES (?, ?, ?, ?)""",
            (session_id, user_message, assistant_response, json.dumps(metadata or {}))
        )

        self.conn.commit()
        return cursor.lastrowid

    def get_recent_conversations(
        self,
        session_id: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict]:
        """Get recent conversations, optionally filtered by session"""
        cursor = self.conn.cursor()

        if session_id:
            query = """SELECT * FROM conversations
                       WHERE session_id = ?
                       ORDER BY timestamp DESC LIMIT ?"""
            cursor.execute(query, (session_id, limit))
        else:
            query = """SELECT * FROM conversations
                       ORDER BY timestamp DESC LIMIT ?"""
            cursor.execute(query, (limit,))

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_session_info(self, session_id: str) -> Optional[Dict]:
        """Get session metadata"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        return dict(row) if row else None

    def update_session_summary(self, session_id: str, summary: str):
        """Update session summary"""
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE sessions SET summary = ? WHERE session_id = ?",
            (summary, session_id)
        )
        self.conn.commit()

    def search_conversations(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict]:
        """Simple text search in conversations"""
        cursor = self.conn.cursor()
        search_pattern = f"%{query}%"

        cursor.execute(
            """SELECT * FROM conversations
               WHERE user_message LIKE ? OR assistant_response LIKE ?
               ORDER BY timestamp DESC LIMIT ?""",
            (search_pattern, search_pattern, limit)
        )

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_all_sessions(self) -> List[Dict]:
        """Get all sessions ordered by last activity"""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM sessions ORDER BY last_active DESC"
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
