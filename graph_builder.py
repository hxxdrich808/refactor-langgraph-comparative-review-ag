from langgraph.graph import StateGraph, END, START
from compare_state import CompareState
from nodes import plan_criteria, research_entity, build_table, verdict

def create_comparative_graph(llm_type: str = "openai") -> StateGraph:
    """
    Build the entire comparative review graph using GraphBuilder.
    The graph follows: START → plan_criteria → research_entity (loop) → build_table → verdict → END
    llm_type determines which LangChain LLM provider is used in the verdict node.
    """
    # Define state type for type checking
    def _state(state: dict):
        return CompareState(**state)

    builder = StateGraph(_state)

    # Add nodes
    builder.add_node("plan", plan_criteria)
    builder.add_node("research", research_entity)
    builder.add_node("table", build_table)
    builder.add_node("verdict", lambda state, llm_type=llm_type: verdict(state, llm_type))

    # Define connections
    builder.set_entry_point("plan")
    builder.add_edge("plan", "research")
    builder.add_conditional_edges(
        "research",
        lambda state: "continue" if state.current_pair_index < len(state.entities) * len(state.criteria)
        else "table",
    )
    builder.add_edge("continue", "research")  # loop back
    builder.add_edge("table", "verdict")
    builder.add_edge("verdict", END)

    return builder.compile()
