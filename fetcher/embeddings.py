from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

class EmbeddingGenerator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding generator

        Default model: all-MiniLM-L6-v2
        - Fast and lightweight (80MB)
        - Good quality for semantic search
        - 384 dimensions

        Alternative: "all-mpnet-base-v2" (higher quality, slower, 420MB)
        """
        self.model = SentenceTransformer(model_name)

    def generate(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

    def generate_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts efficiently"""
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()

    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        emb1 = self.model.encode(text1, convert_to_tensor=False)
        emb2 = self.model.encode(text2, convert_to_tensor=False)

        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)

# Global instance to avoid reloading model
_embedding_generator = None

def get_embedding_generator() -> EmbeddingGenerator:
    """Get or create global embedding generator instance"""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    return _embedding_generator
