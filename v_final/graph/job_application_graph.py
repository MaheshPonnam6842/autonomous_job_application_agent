from state.job_application_state import JobApplicationState
from nodes.jd_ingest import jd_ingest_node
from nodes.resume_ingest import resume_ingest_node
from nodes.skill_extraction import skill_extraction_node
from nodes.resume_rewrite import resume_rewrite_node
from nodes.matching import matching_node
from nodes.outreach import outreach_node
from nodes.decision import decision_node
from nodes.tracking import tracking_node
from langgraph.graph import StateGraph, START, END
from IPython.display import display, Image
builder= StateGraph(JobApplicationState)
def rewrite_router(state: JobApplicationState) -> str:
    if state.get("rewrite_required"):
        return "rewrite"
    return "skip"

# ADD NODES TO GRAPH
builder.add_node("resume_ingest", resume_ingest_node)
builder.add_node("jd_ingest", jd_ingest_node)
builder.add_node("skill_extraction", skill_extraction_node)
builder.add_node("matching", matching_node)
builder.add_node("decision", decision_node)
builder.add_node("resume_rewrite", resume_rewrite_node)
builder.add_node("outreach", outreach_node)
builder.add_node("tracking", tracking_node)
# DEFINE EDGES
builder.add_edge(START, "resume_ingest")
builder.add_edge("resume_ingest", "jd_ingest")
builder.add_edge("jd_ingest", "skill_extraction")
builder.add_edge("skill_extraction", "matching")
builder.add_edge("matching", "decision")
builder.add_conditional_edges("decision", rewrite_router, {
    "rewrite": "resume_rewrite",
    "skip": "outreach"
})
builder.add_edge("resume_rewrite", "outreach")
builder.add_edge("outreach", "tracking")
builder.add_edge("tracking", END)


graph= builder.compile()



