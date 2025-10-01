import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager
from database.vector_store import VectorStore
from typing import List, Dict, Optional
import uuid

class MemoryManager:
    def __init__(self):
        self.db = DatabaseManager()
        self.vector_store = VectorStore()

    def save_exchange(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Save a conversation exchange to both SQLite and vector store

        Returns the conversation ID
        """
        # Save to SQLite
        conv_id = self.db.save_conversation(
            session_id=session_id,
            user_message=user_message,
            assistant_response=assistant_response,
            metadata=metadata
        )

        # Create combined text for embedding (better context)
        combined_text = f"User: {user_message}\nAssistant: {assistant_response}"

        # Save to vector store with metadata
        vector_metadata = {
            "session_id": session_id,
            "timestamp": metadata.get("timestamp", "") if metadata else ""
        }
        self.vector_store.add_conversation(
            conversation_id=conv_id,
            text=combined_text,
            metadata=vector_metadata
        )

        return conv_id

    def retrieve_context(
        self,
        current_message: str,
        session_id: str,
        recent_limit: int = 10,
        semantic_limit: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Retrieve relevant context for current message

        Returns:
        {
            'recent': [...],  # Recent conversations from this session
            'relevant': [...]  # Semantically similar past conversations
        }
        """
        # Get recent conversations from current session
        recent = self.db.get_recent_conversations(
            session_id=session_id,
            limit=recent_limit
        )

        # Get semantically similar conversations
        similar = self.vector_store.search_similar(
            query=current_message,
            n_results=semantic_limit
        )

        # Fetch full conversation details for similar results
        relevant = []
        for result in similar:
            conv = self._get_conversation_by_id(result['conversation_id'])
            if conv and conv.get('session_id') != session_id:  # Exclude current session
                conv['similarity_score'] = result.get('distance')
                relevant.append(conv)

        return {
            'recent': recent[::-1],  # Reverse to chronological order
            'relevant': relevant
        }

    def _get_conversation_by_id(self, conv_id: int) -> Optional[Dict]:
        """Get single conversation by ID"""
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

    def format_context_for_llm(self, context: Dict) -> str:
        """Format retrieved context into string for LLM prompt"""
        lines = []

        if context['recent']:
            lines.append("=== Recent Conversation ===")
            for conv in context['recent']:
                lines.append(f"User: {conv['user_message']}")
                lines.append(f"Assistant: {conv['assistant_response']}")
                lines.append("")

        if context['relevant']:
            lines.append("=== Relevant Past Memories ===")
            for conv in context['relevant']:
                lines.append(f"[From session {conv['session_id']}]")
                lines.append(f"User: {conv['user_message']}")
                lines.append(f"Assistant: {conv['assistant_response']}")
                lines.append("")

        return "\n".join(lines)

    def create_session(self) -> str:
        """Create a new session ID"""
        return str(uuid.uuid4())

    def search_memories(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search memories using both text and semantic search

        Combines results from both methods
        """
        # Text search
        text_results = self.db.search_conversations(query, limit=limit)

        # Semantic search
        vector_results = self.vector_store.search_similar(query, n_results=limit)

        # Combine and deduplicate
        seen_ids = set()
        combined = []

        for conv in text_results:
            if conv['id'] not in seen_ids:
                conv['search_type'] = 'text'
                combined.append(conv)
                seen_ids.add(conv['id'])

        for result in vector_results:
            conv_id = result['conversation_id']
            if conv_id not in seen_ids:
                conv = self._get_conversation_by_id(conv_id)
                if conv:
                    conv['search_type'] = 'semantic'
                    conv['similarity_score'] = result.get('distance')
                    combined.append(conv)
                    seen_ids.add(conv_id)

        return combined[:limit]

    def get_session_summary(self, session_id: str) -> Optional[str]:
        """Get summary for a session"""
        session = self.db.get_session_info(session_id)
        return session.get('summary') if session else None

    def close(self):
        """Close database connections"""
        self.db.close()
