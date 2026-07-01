import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient as QC
from qdrant_client.http.models import PointStruct, Distance, VectorParams
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Initialize client once
client = QC(url=QDRANT_URL, api_key=QDRANT_API_KEY)

COLLECTION_NAME = "comparative_reviews"

def ensure_collection():
    """Create collection if it doesn't exist."""
    collections = client.get_collections()
    names = [c.name for c in collections.collections]
    if COLLECTION_NAME not in names:
        client.create_collection(
            name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

ensure_collection()

def add_document(text: str, doc_id: str) -> None:
    """
    Store a document with a dummy vector (all zeros).
    In a real scenario, replace this with an embedding model.
    """
    zero_vector = [0.0] * 768
    point = PointStruct(id=doc_id, vector=zero_vector, payload={"text": text})
    client.upsert(collection_name=COLLECTION_NAME, points=[point])

def query_similar(text: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Return top-k similar documents based on dummy similarity.
    Since vectors are zeros, all documents have same distance; return first k.
    """
    # In a real implementation, compute embeddings and use search().
    # Here we perform a simple retrieval of the first `limit` points.
    result = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return [
        {"id": p.id, "payload": p.payload} for p in result.points
    ]
