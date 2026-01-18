from langgraph.graph import StateGraph, START, END
from state.job_application_state import JobApplicationState

from nodes.resume_rewrite import resume_rewrite_node
from nodes.outreach import outreach_node


def build_rewrite_graph():
    builder = StateGraph(JobApplicationState)

    builder.add_node("resume_rewrite", resume_rewrite_node)
    builder.add_node("outreach", outreach_node)

    builder.add_edge(START, "resume_rewrite")
    builder.add_edge("resume_rewrite", "outreach")
    builder.add_edge("outreach", END)

    return builder.compile()
