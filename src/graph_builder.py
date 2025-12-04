from langgraph.graph import StateGraph, END
from .state import ResearchState
from .nodes.expand import expand_query
from .nodes.research import research_subqueries
from .nodes.synthesize import synthesize_results

def build_graph():
    # Initialize the state graph with our state type
    graph = StateGraph(ResearchState)

    # Register nodes
    graph.add_node("expand", expand_query)
    graph.add_node("research", research_subqueries)
    graph.add_node("synthesize", synthesize_results)

    # Define flow: expand -> research -> synthesize -> END
    graph.add_edge("expand", "research")
    graph.add_edge("research", "synthesize")
    graph.add_edge("synthesize", END)

    # Set entrypoint
    graph.set_entry_point("expand")

    # Compile into a runnable app
    app = graph.compile()
    return app
