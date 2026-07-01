"""
Qdrant client wrapper for vector storage and similarity search.
"""

from typing import Dict, List, Any, Optional
import os

try:
    from qdrant_client import QdrantClient as _QdrantClient
    from qdrant_client.http.models import PointStruct, VectorParams, Distance
except ImportError:
    # If qdrant-client is not installed, raise a clear error when used.
    class _MissingModule:
        def __getattr__(self, name):
            raise ImportError(
                "qdrant-client is required for Qdrant integration. "
                "Install it with `pip install qdrant-client`."
            )

    _QdrantClient = _MissingModule()
    VectorParams = Distance = PointStruct = None


class QdrantClientWrapper:
    """
    Simple wrapper around the official Qdrant client to provide
    upsert and similarity search functionality.
    """

    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        self.url = url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.client: _QdrantClient = _QdrantClient(url=self.url, api_key=self.api_key)

    def ensure_collection(self, name: str, vector_size: int):
        """
        Create a collection if it does not exist.
        """
        collections = self.client.get_collections().collections
        if any(c.name == name for c in collections):
            return

        params = VectorParams(size=vector_size, distance=Distance.COSINE)
        self.client.create_collection(name=name, vectors_config=params)

    def upsert_vector(
        self,
        collection_name: str,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Upsert a single vector with optional metadata.
        """
        point = PointStruct(id=vector_id, vector=vector, payload=metadata or {})
        self.client.upsert(collection_name, points=[point])

    def search_similar(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Return the top `limit` most similar vectors to `query_vector`.
        Each result contains the point id and payload.
        """
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
        )
        return [
            {"id": hit.id, "score": hit.score, "payload": hit.payload}
            for hit in results
        ]
