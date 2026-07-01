import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient as QC
from qdrant_client.http.models import PointStruct, VectorParams, Distance
from langchain_openai.embeddings import OpenAIEmbeddings


class QdrantService:
    """
    Lightweight wrapper around the Qdrant client.
    Handles collection creation, upserting embeddings and similarity search.
    """

    def __init__(self):
        self.url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.client = QC(url=self.url, api_key=self.api_key)
        self.collection_name = "comparative_reviews"
        self._ensure_collection()

        # Embedding model for generating vectors
        self.embedding_model = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    def _ensure_collection(self):
        """Create collection if it does not exist."""
        collections = self.client.get_collections()
        if self.collection_name in [c.name for c in collections.collections]:
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    def upsert_embeddings(self, entity: str, criterion: str, text: str) -> List[int]:
        """
        Generate embeddings for the given text and store them in Qdrant.
        Returns a list of point IDs that were inserted/updated.
        """
        embedding = self.embedding_model.embed_query(text)
        point_id = f"{entity}_{criterion}"
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload={"entity": entity, "criterion": criterion},
        )
        self.client.upsert(collection_name=self.collection_name, points=[point])
        return [point_id]

    def search_similar(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a similarity search against the collection.
        Returns list of matching points with their payloads and scores.
        """
        query_vector = self.embedding_model.embed_query(query_text)
        results = self.client.search(
            collection_name=self.collection_name,
            vector=query_vector,
            limit=top_k,
            with_payload=True,
            score_threshold=None,
        )
        return [
            {"id": r.id, "score": r.score, "payload": r.payload}
            for r in results
        ]


# Singleton instance used across the project
qdrant_service = QdrantService()
