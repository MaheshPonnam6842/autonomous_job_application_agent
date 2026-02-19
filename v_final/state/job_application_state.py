from typing import TypedDict, List, Dict, Optional
from typing import TypedDict, List, Dict, Optional


class EducationEntry(TypedDict):
    degree: str            # Bachelor | Master | PhD
    field: str
    institution: str
    year: Optional[str]


class ExperienceEntry(TypedDict):
    company: str
    role: str
    start_date: Optional[str]
    end_date: Optional[str]
    bullets: List[str]


class ProjectEntry(TypedDict):
    name: str
    description: Optional[str]
    bullets: List[str]
class ResumeSections(TypedDict):
    summary: str
    skills_raw: str                  # raw skills section text
    experience_raw: str
    projects_raw: str
    education_raw: str

class JobApplicationState(TypedDict, total=False):

    # 1. Inputs
    resume_raw_text: str
    resume_source: str                  # upload | paste | api
    job_description_text: str

    # 2. Resume Ingestion
    resume_clean_text: str
    resume_sections: ResumeSections

    # 3. Resume Structure (Derived)
    experience_entries: List[ExperienceEntry]
    project_entries: List[ProjectEntry]
    education_entries: List[EducationEntry]
    resume_skills_structured: Dict[str, List[str]]


    # 4. Bullet Intelligence
    experience_bullets: List[str]
    project_bullets: List[str]

    experience_groups: Dict[str, List[str]]   # company/role heading -> bullets
    project_groups: Dict[str, List[str]]   

    # 5. Job Description Understanding
    jd_skills_required: List[str]
    jd_skills_preferred: List[str]
    jd_responsibilities: List[str]
    jd_domain: str                      # finance | retail | healthcare
    jd_seniority_level: str             # junior | mid | senior | staff

    # 6. Skill Intelligence
    resume_skill_profile: SkillProfile
    jd_skill_profile: SkillProfile

    skill_overlap: Dict[str, List[str]]     # category → matched
    skill_gap_hard: Dict[str, List[str]]
    skill_gap_soft: Dict[str, List[str]]

    # 7. Matching & Scoring
    semantic_similarity_score: float
    skill_match_score: float
    ats_match_score: float

    # 8. Decision & Strategy
    rewrite_required: bool
    rewrite_reason: str
    rewrite_strategy: str
    # "none" | "summary_only" | "summary_and_skills"
    # "experience_bullets" | "full_rewrite"

    # 9. Rewrite Outputs (Structured)
    optimized_summary: Optional[str]
    optimized_experience_bullets: Optional[List[str]]
    optimized_project_bullets: Optional[List[str]]
    optimized_skills_section: Optional[str]

    optimized_resume_text: str
    rewrite_diff: Optional[List[str]]

    # 10. Outreach Outputs
    outreach_dm_text: str
    outreach_email_text: str

    # 11. Agent Diagnostics
    agent_notes: List[str]
    warnings: List[str]
    rewrite_noop: bool
