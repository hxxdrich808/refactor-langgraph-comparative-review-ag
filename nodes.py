import os
from typing import Any, Dict, Tuple, List

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from tavily import TavilyClient
from compare_state import CompareState
from console import print as rprint
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Initialize Tavily client once
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise RuntimeError("Please set the environment variable TAVILY_API_KEY.")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# Import Qdrant helper
from qdrant_client import add_document, query_similar

def plan_criteria(state: CompareState) -> CompareState:
    """
    Ask an LLM to generate 3–5 comparison criteria based solely on the three entities.
    The generated list is stored in state['criteria'].
    """
    entities_str = ", ".join(state["entities"])
    prompt = (
        f"Generate 3 to 5 concise comparison criteria for evaluating the following technologies: {entities_str}. "
        "Return only a JSON array of strings, e.g., [\"Criterion1\", \"Criterion2\", ...]."
    )

    # Choose LLM
    llm_type = os.getenv("LLM_TYPE", "openai")
    if llm_type == "ollama":
        llm = ChatOllama(model="llama3.1", temperature=0.2, max_tokens=200)
    else:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise RuntimeError("Please set OPENAI_API_KEY environment variable.")
        llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.2, max_tokens=200)

    response = llm.invoke(prompt)
    # Parse JSON array
    try:
        criteria = json.loads(str(response))
        if not isinstance(criteria, list):
            raise ValueError
    except Exception as e:
        rprint(f"[red]Failed to parse criteria from LLM: {e}[/]")
        criteria = ["Ease of Integration", "Performance", "Community Support"]

    state["criteria"] = [str(c).strip('"') for c in criteria]
    rprint("[bold cyan]Generated Criteria:[/]", state["criteria"])
    return state


def research_entity(state: CompareState) -> CompareState:
    """
    Perform a Tavily web search for the current entity‑criterion pair.
    Store the snippet in findings[entity][criterion_index].
    Also store the snippet in Qdrant for future similarity checks.
    """
    entities = state["entities"]
    criteria = state["criteria"]
    idx = state.get("current_pair_index", 0)

    total_pairs = len(entities) * len(criteria)
    if idx >= total_pairs:
        return state

    entity = entities[idx // len(criteria)]
    criterion = criteria[idx % len(criteria)]

    query = f"{entity} {criterion}"
    try:
        result = tavily_client.search(query, max_results=1)
        snippet = result["results"][0]["content"][:300]
    except Exception as e:
        snippet = f"Error fetching data: {e}"

    # Store in Qdrant
    doc_id = f"{entity}_{criterion}"
    add_document(snippet, doc_id)

    # Ensure list exists
    if entity not in state["findings"]:
        state["findings"][entity] = ["" for _ in criteria]
    state["findings"][entity][idx % len(criteria)] = snippet

    rprint(f"[green]Fetched:[/] {entity} - {criterion}")

    # Advance to next pair
    state["current_pair_index"] = idx + 1
    return state


def build_table(state: CompareState) -> CompareState:
    """
    Build a markdown table from the collected findings.
    Rows are criteria, columns are entities.
    """
    entities = state["entities"]
    criteria = state["criteria"]
    findings = state["findings"]

    header = "| Criterion | " + " | ".join(entities) + " |\n"
    separator = "|" + "---|" * (len(entities) + 1) + "\n"

    rows = []
    for i, criterion in enumerate(criteria):
        row_cells = [criterion]
        for entity in entities:
            notes_list = findings.get(entity, [])
            note = notes_list[i] if i < len(notes_list) else "N/A"
            cell = (note[:50].replace("\n", " ") + ("..." if len(note) > 50 else "")) or "N/A"
            row_cells.append(cell)
        rows.append("| " + " | ".join(row_cells) + " |\n")

    table = header + separator + "".join(rows)
    state["final_table"] = table
    rprint("[bold magenta]Final Table Constructed[/]")
    return state


def verdict(state: CompareState, llm_type: str = "openai") -> CompareState:
    """
    Generate a concise recommendation using an LLM based on the final table.
    """
    if llm_type == "ollama":
        llm = ChatOllama(model="llama3.1", temperature=0.2, max_tokens=150)
    else:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise RuntimeError("Please set OPENAI_API_KEY environment variable.")
        llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.2, max_tokens=150)

    prompt = (
        "You are an AI assistant that reviews comparative tables of database technologies.\n"
        f"Here is the table:\n{state['final_table']}\n\n"
        "Based on this table, recommend which technology is best suited for a small startup looking to build a scalable web application. "
        "Keep it under 100 words."
    )

    response = llm.invoke(prompt)
    state["verdict"] = str(response).strip()
    rprint("[bold magenta]Verdict Generated:[/]", state["verdict"])
    return state
