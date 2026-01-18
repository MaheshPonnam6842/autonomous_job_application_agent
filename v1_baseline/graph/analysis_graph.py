from langgraph.graph import StateGraph, START, END
from state.job_application_state import JobApplicationState

from nodes.resume_ingest import resume_ingest_node
from nodes.jd_ingest import jd_ingest_node
from nodes.skill_extraction import skill_extraction_node
from nodes.matching import matching_node
from nodes.decision import decision_node


def build_analysis_graph():
    builder = StateGraph(JobApplicationState)

    builder.add_node("resume_ingest", resume_ingest_node)
    builder.add_node("jd_ingest", jd_ingest_node)
    builder.add_node("skill_extraction", skill_extraction_node)
    builder.add_node("matching", matching_node)
    builder.add_node("decision", decision_node)

    builder.add_edge(START, "resume_ingest")
    builder.add_edge("resume_ingest", "jd_ingest")
    builder.add_edge("jd_ingest", "skill_extraction")
    builder.add_edge("skill_extraction", "matching")
    builder.add_edge("matching", "decision")
    builder.add_edge("decision", END)

    return builder.compile()
