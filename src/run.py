import argparse
from src.graph_builder import build_graph
from src.state import ResearchState

def main():
    parser = argparse.ArgumentParser(description="Run the Langgraph Research MVP")
    parser.add_argument("--query", type=str, required=True, help="The query to research")
    args = parser.parse_args()

    # Build the graph
    app = build_graph()

    # Initialize the state
    initial_state: ResearchState = {
        "query": args.query,
        "subqueries": [],
        "research_output": [],
        "final_answer": "",
    }

    # Run the graph
    final_state = app.invoke(initial_state)

    # Print the final synthesized output
    print("\n ======= Final Answer ======= \n")
    print(final_state["final_answer"])
    print("\n ============================ \n")

if __name__ == "__main__":
    main()