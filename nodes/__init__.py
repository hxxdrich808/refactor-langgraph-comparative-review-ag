"""
Placeholder implementations for the LangGraph nodes.
These stubs simply pass through the state unchanged. In a real
application they would contain the logic described in the README.
"""

from typing import Dict, Any, List
from compare_state import CompareState


def plan_criteria(state: CompareState) -> CompareState:
    """
    Stub: Pretend to generate criteria and add them to the state.
    """
    if "criteria" not in state or not state["criteria"]:
        # Example static criteria for demonstration purposes.
        state["criteria"] = [
            "Performance",
            "Scalability",
            "Ease of Use",
        ]
    return state


def research_entity(state: CompareState) -> CompareState:
    """
    Stub: Pretend to perform research on one entity-criterion pair.
    """
    entities: List[str] = state.get("entities", [])
    criteria: List[str] = state.get("criteria", [])
    index = state.get("current_pair_index", 0)

    total_pairs = len(entities) * len(criteria)
    if index >= total_pairs:
        return state

    # Determine current pair
    entity_idx = index // len(criteria)
    crit_idx = index % len(criteria)
    entity = entities[entity_idx]
    criterion = criteria[crit_idx]

    key = f"{entity}|{criterion}"
    findings = state.get("findings", {})
    if key not in findings:
        # Dummy snippet
        findings[key] = f"Sample finding for {entity} on {criterion}."
    state["findings"] = findings

    # Increment index
    state["current_pair_index"] = index + 1
    return state


def build_table(state: CompareState) -> CompareState:
    """
    Stub: Build a simple markdown table from the findings.
    """
    entities = state.get("entities", [])
    criteria = state.get("criteria", [])
    findings = state.get("findings", {})

    header = "| Criterion | " + " | ".join(entities) + " |\n"
    separator = "|" + "---|" * (len(entities) + 1) + "\n"

    rows = []
    for crit in criteria:
        row_cells = [crit]
        for ent in entities:
            key = f"{ent}|{crit}"
            snippet = findings.get(key, "N/A")
            row_cells.append(snippet)
        rows.append("| " + " | ".join(row_cells) + " |\n")

    table_md = header + separator + "".join(rows)
    state["final_table"] = table_md
    return state


def verdict(state: CompareState, llm_type: str = "openai") -> CompareState:
    """
    Stub: Generate a simple verdict.
    """
    state["verdict"] = (
        f"Based on the comparison, {state['entities'][0]} is recommended "
        f"for most use cases."
    )
    return state
