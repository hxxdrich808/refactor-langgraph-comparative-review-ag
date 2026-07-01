import os
import pytest
from qdrant_client import add_document, query_similar

def test_add_and_query():
    # Add a document
    doc_id = "test_doc"
    text = "This is a sample snippet for testing."
    add_document(text, doc_id)

    # Query similar (should return at least one)
    results = query_similar("any query", limit=1)
    assert isinstance(results, list)
    assert len(results) >= 1
    found_ids = [r["id"] for r in results]
    assert doc_id in found_ids

if __name__ == "__main__":
    pytest.main()
