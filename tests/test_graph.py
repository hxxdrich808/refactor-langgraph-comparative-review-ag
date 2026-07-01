import os
import pytest
from graph_builder import create_comparative_graph

@pytest.fixture(scope="module")
def graph():
    return create_comparative_graph()

def test_full_run(graph):
    # Basic entities list
    state = {
        "entities": ["Chroma", "FAISS"],
        "criteria": [],
        "findings": {},
        "current_pair_index": 0,
        "table_markdown": "",
        "verdict": "",
    }
    final_state = graph.invoke(state)
    # Ensure table and verdict are populated
    assert final_state["table_markdown"].startswith("| Criterion")
    assert len(final_state["verdict"]) > 10

def test_loop_completion(graph):
    state = {
        "entities": ["Chroma"],
        "criteria": [],
        "findings": {},
        "current_pair_index": 0,
        "table_markdown": "",
        "verdict": "",
    }
    final_state = graph.invoke(state)
    # Only one entity, should still produce table
    assert len(final_state["table_markdown"].splitlines()) > 3

if __name__ == "__main__":
    pytest.main()
