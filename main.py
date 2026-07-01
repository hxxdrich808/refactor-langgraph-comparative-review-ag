import argparse
from langgraph import GraphBuilder
from compare_state import CompareState
from nodes import plan_criteria, research_entity, build_table, verdict
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from dotenv import load_dotenv

load_dotenv()


def create_graph(llm_type: str = "openai") -> GraphBuilder:
    builder = GraphBuilder()

    def init_state(entities):
        return {
            "entities": entities,
            "criteria": [],
            "findings": {},
            "final_table": None,
            "verdict": None,
            "current_pair_index": 0,
        }

    builder.add_node("init", lambda state: state)
    builder.add_node("plan_criteria", plan_criteria)
    builder.add_node("research_entity", research_entity)
    builder.add_node("build_table", build_table)
    builder.add_node("verdict", lambda s: verdict(s, llm_type=llm_type))

    builder.set_entry_point("init")
    builder.add_edge("init", "plan_criteria")
    builder.add_edge("plan_criteria", "research_entity")
    builder.add_conditional_edges(
        "research_entity",
        lambda state: (
            ("continue" if state.get("current_pair_index", 0)
             < len(state["entities"]) * len(state["criteria"])
             else "build_table")
        ),
    )
    builder.add_edge("continue", "research_entity")
    builder.add_edge("build_table", "verdict")

    return builder, init_state


def main():
    parser = argparse.ArgumentParser(description="LangGraph Comparative Review Agent")
    parser.add_argument(
        "--entities",
        nargs="+",
        default=["Chroma", "FAISS", "Qdrant"],
        help="List of entities to compare",
    )
    parser.add_argument(
        "--llm-type",
        choices=["openai", "ollama"],
        default="openai",
        help="LLM provider",
    )
    args = parser.parse_args()

    builder, init_state_fn = create_graph(args.llm_type)
    graph = builder.compile()
    initial_state: CompareState = init_state_fn(args.entities)

    final_state = graph.run(initial_state)

    console = Console()
    console.print("\n[bold underline]Final Markdown Table[/]\n")
    console.print(Markdown(final_state["final_table"]))

    console.print("\n[bold underline]Verdict[/]\n")
    console.print(Panel(final_state["verdict"], title="Recommendation", expand=False))


if __name__ == "__main__":
    main()
