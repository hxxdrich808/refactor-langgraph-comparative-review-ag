import os
from typing import Any, Dict, Tuple

from tavily import TavilyClient
from compare_state import CompareState
from console import console

# Initialize Tavily client once
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise RuntimeError("Please set the environment variable TAVILY_API_KEY.")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


def plan_criteria(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Populate the list of criteria to evaluate each entity against.
    The criteria are hard‑coded for this demo but could be made dynamic.
    """
    state["criteria"] = [
        "Ease of Integration",
        "Performance",
        "Community Support",
        "Cost",
        "Scalability",
    ]
    console.print("[bold cyan]Criteria Plan:[/]", state["criteria"])
    return state


def research_entity(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform a web search for the current entity‑criterion pair using Tavily
    and store a concise note in findings.
    """
    entities = state["entities"]
    criteria = state["criteria"]
    idx = state.get("current_pair_index", 0)

    # Determine which pair to process
    if idx >= len(entities) * len(criteria):
        return state  # nothing to do

    entity = entities[idx // len(criteria)]
    criterion = criteria[idx % len(criteria)]

    query = f"{entity} {criterion}"
    try:
        result = tavily_client.search(query, max_results=1)
        note = result["results"][0]["content"][:200]  # truncate for brevity
    except Exception as e:
        note = f"Error fetching data: {e}"

    state["findings"][(entity, criterion)] = note

    console.print(f"[green]Fetched:[/] {entity} - {criterion}")

    # Advance to next pair
    state["current_pair_index"] = idx + 1
    return state


def build_table(state: Dict[str, Any]) -> Dict[str, Any]:
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
    for criterion in criteria:
        row_cells = [criterion]
        for entity in entities:
            note = findings.get((entity, criterion), "N/A")
            # Shorten cell content to keep table readable
            cell = note[:50].replace("\n", " ")
            row_cells.append(cell)
        rows.append("| " + " | ".join(row_cells) + " |\n")

    table = header + separator + "".join(rows)
    state["table_markdown"] = table
    return state


def verdict(state: Dict[str, Any], llm_type: str = "openai") -> Dict[str, Any]:
    """
    Generate a concise recommendation using an LLM based on the final table.
    Supports OpenAI or Ollama via LangChain.
    """
    from langchain_openai import ChatOpenAI
    from langchain_ollama import ChatOllama

    if llm_type == "ollama":
        # Use local Ollama model; default to llama3.1
        llm = ChatOllama(model="llama3.1", temperature=0.2, max_tokens=150)
    else:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise RuntimeError("Please set OPENAI_API_KEY environment variable.")
        llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.2, max_tokens=150)

    prompt = (
        "You are an AI assistant that reviews comparative tables of database technologies.\n"
        f"Here is the table:\n{state['table_markdown']}\n\n"
        "Based on this table, recommend which technology is best suited for a small startup "
        "looking to build a scalable web application. Keep it under 100 words."
    )

    response = llm.invoke(prompt)
    state["verdict"] = str(response).strip()
    console.print("[bold magenta]Verdict Generated:[/]", state["verdict"])
    return state
