import argparse
from typing import List, Dict, Any
import json
import os

from langgraph.graph import StateGraph
from nodes import (
    plan_criteria,
    research_entity,
    build_table,
    verdict,
)
from compare_state import CompareState
from console import print as rprint
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangGraph Comparative Review Agent")
    parser.add_argument(
        "--entities",
        nargs="+",
        default=["Chroma", "FAISS", "Qdrant"],
        help="Three entities to compare",
    )
    parser.add_argument(
        "--llm-type",
        choices=["openai", "ollama"],
        default=os.getenv("LLM_TYPE", "openai"),
        help="LLM provider",
    )
    parser.add_argument(
        "--enable-qdrant",
        action="store_true",
        default=False,
        help="Enable Qdrant integration for vector storage and retrieval",
    )
    return parser.parse_args()

def build_state(entities: List[str], enable_qdrant: bool) -> CompareState:
    state: CompareState = {
        "entities": entities,
        "criteria": [],
        "findings": {},
        "final_table": None,
        "verdict": None,
        "current_pair_index": 0,
        "qdrant_vectors": {} if enable_qdrant else None,
    }
    return state

def run_graph(state: CompareState, llm_type: str) -> CompareState:
    # Sequential execution as per required flow
    state = plan_criteria(state)
    total_pairs = len(state["entities"]) * len(state["criteria"])
    while state.get("current_pair_index", 0) < total_pairs:
        state = research_entity(state)
    state = build_table(state)
    state = verdict(state, llm_type=llm_type)
    return state

def render_output(state: CompareState):
    console = Console()
    # Render criteria
    console.print("[bold underline]Generated Criteria:[/]")
    for idx, crit in enumerate(state["criteria"], 1):
        console.print(f"{idx}. {crit}")

    console.print("\n[bold underline]Final Comparison Table:[/]")
    md = Markdown(state["final_table"] or "")
    console.print(md)

    console.print("\n[bold underline]Verdict:[/]")
    console.print(state["verdict"] or "")

def main():
    args = parse_args()
    state = build_state(args.entities, enable_qdrant=args.enable_qdrant)
    rprint("[bold cyan]Starting comparative review...[/]")
    if args.enable_qdrant:
        rprint("[green]Qdrant integration enabled.[/]")
    else:
        rprint("[yellow]Qdrant integration disabled.[/]")
    state = run_graph(state, llm_type=args.llm_type)
    render_output(state)

if __name__ == "__main__":
    main()
