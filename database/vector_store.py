import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
from pathlib import Path

class VectorStore:
    def __init__(self, persist_directory: str = "database/chroma"):
        """Initialize ChromaDB vector store"""
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Create or get collection for conversations
        self.collection = self.client.get_or_create_collection(
            name="conversations",
            metadata={"description": "Conversation memory embeddings"}
        )

    def add_conversation(
        self,
        conversation_id: int,
        text: str,
        metadata: Optional[Dict] = None
    ):
        """Add conversation to vector store with automatic embedding"""
        self.collection.add(
            ids=[str(conversation_id)],
            documents=[text],
            metadatas=[metadata or {}]
        )

    def search_similar(
        self,
        query: str,
        n_results: int = 5
    ) -> List[Dict]:
        """Search for semantically similar conversations"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

        # Format results
        if not results['ids'] or not results['ids'][0]:
            return []

        formatted = []
        for i, conv_id in enumerate(results['ids'][0]):
            formatted.append({
                'conversation_id': int(conv_id),
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                'distance': results['distances'][0][i] if results['distances'] else None
            })

        return formatted

    def delete_conversation(self, conversation_id: int):
        """Remove conversation from vector store"""
        self.collection.delete(ids=[str(conversation_id)])

    def get_count(self) -> int:
        """Get total number of stored conversations"""
        return self.collection.count()

    def clear_all(self):
        """Clear all conversations (use with caution)"""
        self.client.delete_collection("conversations")
        self.collection = self.client.get_or_create_collection(
            name="conversations",
            metadata={"description": "Conversation memory embeddings"}
        )
