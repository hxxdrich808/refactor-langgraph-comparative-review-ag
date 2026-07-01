"""
Node implementations for the LangGraph Comparative Review Agent.
"""

import hashlib
from typing import Dict, Any, Tuple, List

from compare_state import CompareState
# Import the new wrapper instead of the old qdrant_client module
from src.qdrant_wrapper import QdrantClientWrapper
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage
# Import Tavily components for real web search
from langchain_tavily import TavilySearch, TavilyExtract, TavilyMap

# Rich imports for pretty printing
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown

console = Console()

# Simple embedding helper (hash to vector)
def embed_text(text: str, dim: int = 128) -> List[float]:
    h = hashlib.sha256(text.encode()).digest()
    return [int(b) / 255.0 for b in h[:dim]]

# Global Qdrant client instance
qdrant_client = QdrantClientWrapper()

def get_llm(llm_type: str):
    if llm_type == "openai":
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    else:
        return ChatOllama(model="llama2")

def plan_criteria(state: CompareState) -> CompareState:
    """
    Generate criteria using an LLM.
    """
    llm = get_llm(state.get("llm_type", "openai"))
    prompt = (
        "Generate three distinct evaluation criteria for comparing "
        f"{', '.join(state['entities'])}. Return them as a numbered list."
    )
    response = llm([HumanMessage(content=prompt)])
    text = response.content.strip()
    # Parse numbered list
    criteria = [line.split(".", 1)[-1].strip() for line in text.splitlines() if line]
    state["criteria"] = criteria

    # Rich output of plan
    console.print("[bold underline]Plan: Generate Criteria[/]\n")
    table = Table(show_header=False, box=None)
    for idx, crit in enumerate(criteria, 1):
        table.add_row(f"[cyan]{idx}.[/]", f"{crit}")
    console.print(table)

    return state

def research_entity(state: CompareState) -> CompareState:
    """
    Process one entity-criterion pair.
    If Qdrant is enabled, upsert the vector before searching and
    skip Tavily if a similar note already exists.
    """
    entities = state["entities"]
    criteria = state["criteria"]
    idx = state.get("current_pair_index", 0)
    total_pairs = len(entities) * len(criteria)

    entity_idx = idx // len(criteria)
    crit_idx = idx % len(criteria)

    entity = entities[entity_idx]
    criterion = criteria[crit_idx]

    pair_key: Tuple[str, str] = (entity, criterion)

    # Rich output of current iteration
    console.print(f"[bold]{entity}[/] – [italic]{criterion}[/]")

    # Qdrant logic
    if state.get("use_qdrant"):
        collection_name = "comparison_notes"
        qdrant_client.ensure_collection(collection_name, vector_size=128)
        vector_id = f"{entity}_{criterion}"
        vector = embed_text(f"{entity} {criterion}")
        # Check for existing similar notes
        results = qdrant_client.search_similar(
            collection_name,
            query_vector=vector,
            limit=1
        )
        if results and results[0]["score"] > 0.9:
            snippet = f"Previously noted information for {entity} on {criterion}."
        else:
            # Perform real Tavily search
            tavily_search = TavilySearch()
            tavily_extract = TavilyExtract()
            tavily_map = TavilyMap()

            query = f"{entity} {criterion}"
            search_results = tavily_search.run(query=query)

            extracted_text = tavily_extract.run(search_results=search_results)
            snippet = tavily_map.run(extracted_text=extracted_text)

            # Store the new note vector
            qdrant_client.upsert_vector(
                collection_name,
                vector_id=vector_id,
                vector=vector,
                metadata={"snippet": snippet}
            )
    else:
        # Without Qdrant: perform real Tavily search
        tavily_search = TavilySearch()
        tavily_extract = TavilyExtract()
        tavily_map = TavilyMap()

        query = f"{entity} {criterion}"
        search_results = tavily_search.run(query=query)

        extracted_text = tavily_extract.run(search_results=search_results)
        snippet = tavily_map.run(extracted_text=extracted_text)

    state["findings"][pair_key] = snippet

    # Rich output of snippet
    console.print(f"[green]Note:[/] {snippet}\n")

    # Advance index
    state["current_pair_index"] = idx + 1
    return state

def build_table(state: CompareState) -> CompareState:
    """
    Build a markdown table from the findings.
    """
    entities = state["entities"]
    criteria = state["criteria"]

    header = "| Criterion | " + " | ".join(entities) + " |\n"
    separator = "|" + "---|" * (len(entities) + 1) + "\n"

    rows = []
    for criterion in criteria:
        row_cells = [criterion]
        for entity in entities:
            snippet = state["findings"].get((entity, criterion), "")
            row_cells.append(snippet)
        rows.append("| " + " | ".join(row_cells) + " |\n")

    table_md = header + separator + "".join(rows)
    state["final_table"] = table_md

    # Rich output of final markdown table
    console.print("[bold underline]Final Comparison Table[/]\n")
    console.print(Markdown(table_md))

    return state

def verdict(state: CompareState, llm_type: str) -> CompareState:
    """
    Generate a recommendation using an LLM.
    """
    llm = get_llm(llm_type)
    prompt = (
        f"Based on the following comparison table:\n\n{state['final_table']}\n"
        "Provide a concise 2–4 sentence recommendation."
    )
    response = llm([HumanMessage(content=prompt)])
    state["verdict"] = response.content.strip()

    # Rich output of verdict
    console.print("[bold green]Verdict:[/]")
    console.print(f"[green]{state['verdict']}[/]\n")

    return state
