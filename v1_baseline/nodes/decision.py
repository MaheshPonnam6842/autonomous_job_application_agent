from state.job_application_state import JobApplicationState

# ATS-style thresholds (V1)
MATCH_SCORE_THRESHOLD = 0.75
MAX_ALLOWED_MISSING_SKILLS = 1

def decision_node(state: JobApplicationState) -> JobApplicationState:
    """
    Decision Node

    Determines whether resume rewriting is required.

    Reads:
    - match_score
    - missing_skills

    Writes:
    - rewrite_required (bool)
    """
    match_score= state.get("match_score", 0.0)
    missing_skills= state.get("missing_skills", [])
    rewrite_required= (
        match_score<MATCH_SCORE_THRESHOLD or
        len(missing_skills)>MAX_ALLOWED_MISSING_SKILLS
    )
    state["rewrite_required"]= rewrite_required
    return state

