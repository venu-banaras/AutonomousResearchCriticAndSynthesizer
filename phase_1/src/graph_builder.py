from langgraph.graph import StateGraph, END
from sympy.abc import lamda

from src.state import ResearchState
from src.nodes.expand import expand_query
from src.nodes.research import research_subqueries
from src.nodes.synthesize import synthesize_results
from src.nodes.critic import critic_node
from src.nodes.fact_check import fact_check_node
from src.nodes.contradiction import contradiction_node

def build_graph():
    # Initialize the state graph with our state type
    graph = StateGraph(ResearchState)

    # Register nodes
    graph.add_node("expand", expand_query)
    graph.add_node("research", research_subqueries)
    graph.add_node("critic", critic_node)
    graph.add_node("fact_check", fact_check_node)
    graph.add_node("contradiction", contradiction_node)
    graph.add_node("synthesize", synthesize_results)

    # Define flow: expand -> research -> synthesize -> END
    graph.add_edge("expand", "research")
    graph.add_edge("research", "critic")
    # Adding conditional logic
    graph.add_conditional_edges(
        "critic",
        lambda state: "revise" if state["needs_revision"] else "accept",
        {
            "revise": "research",
            "accept": "synthesize",
        }
    )
    graph.add_conditional_edges(
        "fact_check",
        lambda state: "fix" if state["needs_fix"] else "ok",
        {
            "fix": "research",
            "ok": "synthesize",
        }
    )
    graph.add_edge("fact_check", "contradiction")
    graph.add_conditional_edges(
        "contradiction",
        lambda state: "fix" if state["needs_contradiction_fix"] else "ok",
        {
            "fix": "research",
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
