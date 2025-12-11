from langgraph.graph import StateGraph, END
from src.state import ResearchState
from src.nodes.expand import expand_query
from src.nodes.research_unit import research_unit
from src.nodes.critic_unit import critic_unit
from src.nodes.factcheck_unit import factcheck_unit
from src.nodes.synthesize import synthesize_results
from src.nodes.contradiction import contradiction_node
from langgraph import
def parallel_postprocess(state):
    """
    Convert parallel_results into structured lists.
    :param state:
    :return:
    """
    results = state["parallel_results"]

    state["research_outputs"] = [r["research_unit"] for r in results]
    state["critic_results"] = [r["critic_unit"] for r in results]
    state["fact_checks"] = [r["factcheck_unit"] for r in results]

    # Determine if any revision needed
    needs_research_fix = any(r["critic_unit"]["verdict"] == "revise" for r in results)
    needs_fact_fix = any(r["factcheck_unit"]["verdict"] == "fail" for r in results)

    state["needs_revision"] = needs_research_fix
    state["needs_factfix"] = needs_fact_fix

    return state

def build_graph():
    # Initialize the state graph with our state type
    graph = StateGraph(ResearchState)
    subgraph = StateGraph(dict)

    # Add unit nodes
    subgraph.add_node("research_unit", research_unit)
    subgraph.add_node("critic_unit", critic_unit)
    subgraph.add_node("factcheck_unit", factcheck_unit)

    # Connect them in order
    subgraph.add_edge("research_unit", "critic_unit")
    subgraph.add_edge("critic_unit", "factcheck_unit")

    # Set entry and exit
    subgraph.set_entry_point("research_unit")
    subgraph.set_finish_point("factcheck_unit")

    # Compile subgraph
    parallel_unit = subgraph.compile()

    # Turn subgraph into parallel map
    parallel_map = graph.map(
        "subqueries", # field to iterate over
        parallel_unit, # the compiled subgraph
        output_key="parallel_results"
    )

    # Register nodes
    graph.add_node("expand", expand_query)
    graph.add_node("parallel_research", parallel_map)
    graph.add_node("parallel_postprocess", parallel_postprocess)
    graph.add_node("contradiction", contradiction_node)
    graph.add_node("synthesize", synthesize_results)

    # Define flow: expand -> research -> synthesize -> END
    graph.add_edge("expand", "parallel_research")
    graph.add_edge("parallel_research", "parallel_postprocess")
    # conditional branching after postprocess
    graph.add_conditional_edges(
        "parallel_postprocess",
        lambda state: "rework" if (state["needs_revision"] or state["needs_factfix"]) else "ok",
        {
            "rework": "parallel_research",
            "ok": "contradiction"
        }
    )
    graph.add_edge("parallel_postprocess", "contradiction")
    graph.add_conditional_edges(
        "contradiction",
        lambda state: "fix" if state["needs_contradiction_fix"] else "ok",
        {
            "fix": "parallel_research",
            "ok": "synthesize",
        }
    )
    graph.add_edge("research", "synthesize")
    graph.add_edge("synthesize", END)

    # Set entrypoint
    graph.set_entry_point("expand")

    # Compile into a runnable app
    app = graph.compile()
    return app
