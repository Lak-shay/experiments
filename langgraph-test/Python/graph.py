from langgraph.graph import StateGraph, START, END
from state import AgenticRAGState
from nodes import CRAGNodes

def route_after_grading(state: AgenticRAGState) -> str:
    """Routing function for conditional edges."""
    if state["run_web_search"]:
        return "web_search"
    return "generate"


def build_graph(retriever, llm):
    """Assembles and compiles the LangGraph workflow."""
    nodes = CRAGNodes(retriever, llm)
    workflow = StateGraph(AgenticRAGState)

    # Add Nodes
    workflow.add_node("retrieve",    nodes.retrieve_node)
    workflow.add_node("grade_docs",  nodes.grade_docs_node)
    workflow.add_node("web_search",  nodes.web_search_node)
    workflow.add_node("generate",    nodes.generate_node)

    # Structural Edges
    workflow.add_edge(START,        "retrieve")
    workflow.add_edge("retrieve",   "grade_docs")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate",    END)

    # Dynamic/Conditional Edges
    workflow.add_conditional_edges("grade_docs", route_after_grading)

    return workflow.compile()