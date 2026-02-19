from typing import TypedDict, List, Dict, Optional


class ResumeSections(TypedDict, total=False):
    summary: str
    skills: Dict[str, List[str]]        # e.g. {"ML": ["regression", "classification"]}
    experience: List[str]               # bullet-level
    projects: List[str]                 # bullet-level
    education: List[str]                # untouched


class JobApplicationState(TypedDict, total=False):

    # 1. RAW INPUTS
    resume_raw_text: str
    resume_source: str                  # upload | paste
    job_description_text: str

    # 2. RESUME UNDERSTANDING
    resume_clean_text: str              # normalized text
    resume_sections: ResumeSections
    resume_bullets: List[str]           # experience + projects bullets (flattened)

    # 3. JOB DESCRIPTION UNDERSTANDING
    jd_required_skills: List[str]
    jd_preferred_skills: List[str]
    jd_responsibilities: List[str]
    jd_seniority: str                   # junior | mid | senior
    jd_domain: str                      # finance | retail | genai | etc.

    # 4. SKILL INTELLIGENCE
    resume_skills_extracted: List[str]
    skill_overlap: List[str]
    skill_gap_required: List[str]
    skill_gap_preferred: List[str]

    # 5. MATCHING SCORES (EXPLAINABLE)
    skill_match_score: float            # 0–1
    bullet_alignment_score: float       # 0–1
    overall_match_score: float          # weighted

    # 6. DECISION LAYER
    rewrite_required: bool
    rewrite_reason: str                 # human-readable
    rewrite_scope: str                  # summary | skills | bullets | full

    # 7. REWRITE OUTPUTS
    optimized_summary: Optional[str]
    optimized_skills: Optional[str]
    optimized_bullets: Optional[List[str]]

    optimized_resume_text: Optional[str]

    # 8. OUTREACH
    outreach_dm_text: Optional[str]
    outreach_email_text: Optional[str]


    # 9. AGENT DIAGNOSTICS 
    agent_notes: List[str]              # explains reasoning
    warnings: List[str]                 # guardrail notices
