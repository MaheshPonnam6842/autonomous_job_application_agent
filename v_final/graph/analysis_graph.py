"""Analysis-only sub-graph: ingest -> score -> decision (no GenAI rewrite).

Fast, deterministic-leaning phase used by the web app so the UI can show match
results immediately, then trigger the slower rewrite graph asynchronously.
"""

from langgraph.graph import END, START, StateGraph

from v_final.nodes.decision import decision_node
from v_final.nodes.jd_ingest import jd_ingest_node
from v_final.nodes.matching import matching_node
from v_final.nodes.resume_ingest import resume_ingest_node
from v_final.nodes.skill_extraction import skill_extraction_node
from v_final.state.job_application_state import JobApplicationState


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
