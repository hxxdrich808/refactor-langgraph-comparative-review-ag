"""
Qdrant Client Wrapper for LangGraph Comparative Review Agent.
"""

import logging
from typing import List, Dict, Any, Optional

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import PointStruct, VectorParams, Distance
except ImportError as exc:
    raise RuntimeError(
        "qdrant-client is required for Qdrant integration. "
        "Install it with `pip install qdrant-client`."
    ) from exc

logger = logging.getLogger(__name__)


class QdrantClientWrapper:
    """
    A thin wrapper around the official Qdrant client that provides
    convenience methods used by the LangGraph workflow.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        """
        Initialize the underlying Qdrant client.

        Parameters
        ----------
        url : str, optional
            The URL of the Qdrant instance.
        api_key : str | None, optional
            API key for authentication (if required).
        timeout : int, optional
            Request timeout in seconds.
        """
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=timeout,
        )
        logger.debug("Qdrant client initialized with URL %s", url)

    # ------------------------------------------------------------------
    # Collection helpers
    # ------------------------------------------------------------------
    def ensure_collection(self, name: str, vector_size: int = 128) -> None:
        """
        Create the collection if it does not exist. If it exists,
        nothing happens.

        Parameters
        ----------
        name : str
            Name of the collection.
        vector_size : int, optional
            Dimensionality of vectors stored in this collection.
        """
        try:
            self.client.recreate_collection(
                collection_name=name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.debug("Collection %s recreated", name)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to recreate collection %s: %s", name, exc)

    # ------------------------------------------------------------------
    # Upsert helpers
    # ------------------------------------------------------------------
    def upsert_vector(
        self,
        collection_name: str,
        vector_id: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Insert or update a single vector with optional metadata.

        Parameters
        ----------
        collection_name : str
            Target collection.
        vector_id : str
            Unique identifier for the point.
        vector : list[float]
            Vector representation.
        metadata : dict | None, optional
            Additional payload to store.
        """
        point = PointStruct(
            id=vector_id,
            vector=vector,
            payload=metadata or {},
        )
        self.client.upsert(collection_name=collection_name, points=[point])
        logger.debug("Upserted vector %s into collection %s", vector_id, collection_name)

    # ------------------------------------------------------------------
    # Search helpers
    # ------------------------------------------------------------------
    def search_similar(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Perform a vector similarity search.

        Parameters
        ----------
        collection_name : str
            Target collection.
        query_vector : list[float]
            Query vector.
        limit : int, optional
            Number of results to return.

        Returns
        -------
        list[dict]
            Each dict contains 'id', 'score', and 'payload'.
        """
        raw_results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
        )
        results: List[Dict[str, Any]] = []
        for hit in raw_results:
            results.append(
                {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload or {},
                }
            )
        logger.debug("Search returned %d hits", len(results))
        return results
