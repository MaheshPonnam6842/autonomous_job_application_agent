"""Rewrite sub-graph: resume_rewrite -> outreach -> tracking.

The GenAI-heavy phase. Run asynchronously after analysis so slow model calls
don't block the UI.
"""

from langgraph.graph import END, START, StateGraph

from v_final.nodes.outreach import outreach_node
from v_final.nodes.resume_rewrite import resume_rewrite_node
from v_final.nodes.rewrite_loop import improve_router, rewrite_loop_node
from v_final.nodes.tracking import tracking_node
from v_final.state.job_application_state import JobApplicationState


def build_rewrite_graph():
    builder = StateGraph(JobApplicationState)

    builder.add_node("resume_rewrite", resume_rewrite_node)
    builder.add_node("rewrite_loop", rewrite_loop_node)
    builder.add_node("outreach", outreach_node)
    builder.add_node("tracking", tracking_node)

    builder.add_edge(START, "resume_rewrite")
    builder.add_edge("resume_rewrite", "rewrite_loop")
    builder.add_conditional_edges(
        "rewrite_loop", improve_router, {"retry": "resume_rewrite", "done": "outreach"}
    )
    builder.add_edge("outreach", "tracking")
    builder.add_edge("tracking", END)

    return builder.compile()
