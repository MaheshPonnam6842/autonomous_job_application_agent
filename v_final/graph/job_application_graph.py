"""End-to-end LangGraph pipeline for the v_final agent.

START -> resume_ingest -> resume_structure -> jd_ingest -> skill_extraction
      -> matching -> decision -> (rewrite <-> rewrite_loop | skip) -> outreach
      -> tracking -> END

The rewrite_loop forms a cycle with resume_rewrite: it scores each candidate and
retries (up to a cap) while the score keeps improving, then proceeds to outreach.
"""

from langgraph.graph import END, START, StateGraph

from v_final.nodes.decision import decision_node
from v_final.nodes.jd_ingest import jd_ingest_node
from v_final.nodes.matching import matching_node
from v_final.nodes.outreach import outreach_node
from v_final.nodes.resume_ingest import resume_ingest_node
from v_final.nodes.resume_structure import resume_structure_node
from v_final.nodes.rewrite_loop import improve_router, rewrite_loop_node
from v_final.nodes.skill_extraction import skill_extraction_node
from v_final.nodes.structured_rewrite import structured_rewrite_node
from v_final.nodes.tracking import tracking_node
from v_final.state.job_application_state import JobApplicationState


def build_graph():
    builder = StateGraph(JobApplicationState)

    builder.add_node("resume_ingest", resume_ingest_node)
    builder.add_node("resume_structure", resume_structure_node)
    builder.add_node("jd_ingest", jd_ingest_node)
    builder.add_node("skill_extraction", skill_extraction_node)
    builder.add_node("matching", matching_node)
    builder.add_node("decision", decision_node)
    builder.add_node("resume_rewrite", structured_rewrite_node)
    builder.add_node("rewrite_loop", rewrite_loop_node)
    builder.add_node("outreach", outreach_node)
    builder.add_node("tracking", tracking_node)

    builder.add_edge(START, "resume_ingest")
    builder.add_edge("resume_ingest", "resume_structure")
    builder.add_edge("resume_structure", "jd_ingest")
    builder.add_edge("jd_ingest", "skill_extraction")
    builder.add_edge("skill_extraction", "matching")
    builder.add_edge("matching", "decision")
    # Always run the rewrite node: it reformats (no LLM) on a strong fit and rewrites
    # otherwise, so every run yields the optimized resume, .docx, and before/after scores.
    builder.add_edge("decision", "resume_rewrite")
    builder.add_edge("resume_rewrite", "rewrite_loop")
    builder.add_conditional_edges(
        "rewrite_loop", improve_router, {"retry": "resume_rewrite", "done": "outreach"}
    )
    builder.add_edge("outreach", "tracking")
    builder.add_edge("tracking", END)

    return builder.compile()


# Module-level compiled graph for convenient import.
graph = build_graph()
