from typing import TypedDict, List 

class JobApplicationState(TypedDict, total=False):
    # Inputs
    job_description_text: str
    resume_raw_text: str
    resume_source: str

    # Derived data
    resume_chunks: List[str]
    resume_embeddings: List[List[float]]
    jd_skills_extracted: List[str]
    resume_skills_extracted: List[str]

    # Matching
    match_score: float
    missing_skills: List[str]
    strong_matches: List[str]

    # Decision
    rewrite_required: bool

    # Outputs
    optimized_resume_text: str
    resume_version: str
    outreach_dm_text: str
    outreach_email_text: str

    # Tracking
    application_status: str
    applied_timestamp: str
