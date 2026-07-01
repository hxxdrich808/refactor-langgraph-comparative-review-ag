import pytest
from unittest.mock import patch, MagicMock
from langgraph import GraphBuilder
from compare_state import CompareState
from nodes import plan_criteria, research_entity, build_table, verdict


@pytest.fixture
def graph():
    builder = GraphBuilder()
    builder.add_node("plan_criteria", plan_criteria)
    builder.add_node("research_entity", research_entity)
    builder.add_node("build_table", build_table)
    builder.add_node("verdict", lambda s: verdict(s, llm_type="ollama"))
    builder.set_entry_point("plan_criteria")
    builder.add_edge("plan_criteria", "research_entity")
    builder.add_conditional_edges(
        "research_entity",
        lambda state: (
            ("continue" if state.get("current_pair_index", 0) < len(state["entities"]) * len(state["criteria"])
             else "build_table")
        ),
    )
    builder.add_edge("continue", "research_entity")
    builder.add_edge("build_table", "verdict")
    return builder.compile()


def test_build_table_and_verdict(mocker):
    # Mock Tavily search to return predictable content
    mock_search = MagicMock(return_value={"results": [{"content": "Test content for entity criterion"}]})
    mocker.patch("nodes.tavily_client.search", new=mock_search)

    initial_state: CompareState = {
        "entities": ["A", "B"],
        "criteria": [],
        "findings": {},
        "final_table": None,
        "verdict": None,
        "current_pair_index": 0,
    }

    # Run plan_criteria
    state = plan_criteria(initial_state)
    assert len(state["criteria"]) == 5

    # Run research_entity until done
    while state.get("current_pair_index", 0) < len(state["entities"]) * len(state["criteria"]):
        state = research_entity(state)

    # Build table
    state = build_table(state)
    assert "A" in state["final_table"]
    assert "B" in state["final_table"]

    # Verdict (mock LLM)
    mock_llm = MagicMock()
    mocker.patch("nodes.ChatOllama", return_value=mock_llm)
    mock_llm.invoke.return_value = "Recommended A"
    state = verdict(state, llm_type="ollama")
    assert state["verdict"] == "Recommended A"


def test_empty_criteria():
    initial_state: CompareState = {
        "entities": ["X"],
        "criteria": [],
        "findings": {},
        "final_table": None,
        "verdict": None,
        "current_pair_index": 0,
    }
    state = plan_criteria(initial_state)
    assert len(state["criteria"]) == 5
