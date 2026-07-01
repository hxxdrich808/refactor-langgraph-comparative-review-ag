import argparse
from graph_builder import create_comparative_graph
from compare_state import CompareState

def run_demo(entities, llm_type):
    # Initialize state with entities and empty fields
    initial_state = {
        "entities": entities,
        "criteria": [],
        "findings": {},
        "current_pair_index": 0,
        "table_markdown": "",
        "verdict": "",
    }

    graph = create_comparative_graph(llm_type=llm_type)
    final_state = graph.invoke(initial_state)

    print("\n=== Markdown Table ===\n")
    print(final_state["table_markdown"])
    print("\n=== Verdict ===\n")
    print(final_state["verdict"])


def main():
    parser = argparse.ArgumentParser(description="Comparative Review Agent Demo")
    parser.add_argument(
        "--entities",
        nargs="+",
        default=["Chroma", "FAISS", "Qdrant"],
        help="Space-separated list of entities to compare",
    )
    parser.add_argument(
        "--llm-type",
        choices=["openai", "ollama"],
        default="openai",
        help="LLM provider to use for verdict generation",
    )
    args = parser.parse_args()
    run_demo(args.entities, args.llm_type)


if __name__ == "__main__":
    main()
