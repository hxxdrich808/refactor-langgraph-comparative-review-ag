import pytest
from unittest.mock import MagicMock, patch

# Import the wrapper from its module
from src.qdrant_wrapper import QdrantClientWrapper


@pytest.fixture
def mock_client():
    with patch("src.qdrant_wrapper.QdrantClient") as MockCls:
        instance = MockCls.return_value
        yield instance


def test_initialization(mock_client):
    """Ensure client is instantiated with correct parameters."""
    wrapper = QdrantClientWrapper(url="http://example.com", api_key="secret")
    assert mock_client.call_args[1]["url"] == "http://example.com"
    assert mock_client.call_args[1]["api_key"] == "secret"


def test_ensure_collection_calls_recreate(mock_client):
    wrapper = QdrantClientWrapper()
    wrapper.ensure_collection("test_col", vector_size=128)
    mock_client.recreate_collection.assert_called_once()


def test_upsert_vector_creates_point_and_calls_upsert(mock_client):
    wrapper = QdrantClientWrapper()
    vector_id = "id1"
    vector = [0.1, 0.2]
    metadata = {"foo": "bar"}
    wrapper.upsert_vector("col", vector_id, vector, metadata)
    # Verify that a PointStruct was created with correct fields
    args, kwargs = mock_client.upsert.call_args
    points = args[1] if len(args) > 1 else kwargs["points"]
    assert isinstance(points[0].id, str)
    assert points[0].id == vector_id
    assert points[0].vector == vector
    assert points[0].payload == metadata


def test_search_similar_returns_formatted_results(mock_client):
    wrapper = QdrantClientWrapper()
    # Mock raw search results
    mock_hit1 = MagicMock(id="a", score=0.9, payload={"x": 1})
    mock_hit2 = MagicMock(id="b", score=0.8, payload=None)
    mock_client.search.return_value = [mock_hit1, mock_hit2]

    results = wrapper.search_similar("col", [0.1], limit=2)

    assert len(results) == 2
    assert results[0]["id"] == "a"
    assert results[0]["score"] == 0.9
    assert results[0]["payload"] == {"x": 1}
    assert results[1]["id"] == "b"
    assert results[1]["score"] == 0.8
    assert results[1]["payload"] == {}


def test_search_similar_empty_results(mock_client):
    wrapper = QdrantClientWrapper()
    mock_client.search.return_value = []
    results = wrapper.search_similar("col", [0.1], limit=5)
    assert results == []


@pytest.mark.parametrize(
    "vector_size,expected_distance",
    [(128, "Cosine"), (256, "Cosine")],
)
def test_ensure_collection_vector_params(mock_client, vector_size, expected_distance):
    wrapper = QdrantClientWrapper()
    wrapper.ensure_collection("col", vector_size=vector_size)
    # The recreate_collection call should include VectorParams with size and distance
    args, kwargs = mock_client.recreate_collection.call_args
    vec_cfg = kwargs["vectors_config"]
    assert vec_cfg.size == vector_size
    assert vec_cfg.distance.value == expected_distance
