from __future__ import annotations

from typing import TypedDict


class SkillProfile(TypedDict, total=False):
    hard_skills: list[str]     # python, sql, statistics, nlp, ...
    tools: list[str]           # airflow, docker, mlflow, ...
    cloud: list[str]           # aws, gcp, azure
    ml_concepts: list[str]     # classification, llm, rag, time series, ...
    soft_skills: list[str]     # ownership, communication, stakeholder mgmt, ...
    domains: list[str]         # finance, healthcare, hr tech, ...
    certs: list[str]           # aws certified..., etc.
    keywords: list[str]        # extra ATS keywords

class EducationEntry(TypedDict):
    degree: str            # Bachelor | Master | PhD
    field: str
    institution: str
    year: str | None


class ExperienceEntry(TypedDict):
    company: str
    role: str
    start_date: str | None
    end_date: str | None
    bullets: list[str]


class ProjectEntry(TypedDict):
    name: str
    description: str | None
    bullets: list[str]
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
    experience_entries: list[ExperienceEntry]
    project_entries: list[ProjectEntry]
    education_entries: list[EducationEntry]
    resume_skills_structured: dict[str, list[str]]


    # 4. Bullet Intelligence
    experience_bullets: list[str]
    project_bullets: list[str]

    experience_groups: dict[str, list[str]]   # company/role heading -> bullets
    project_groups: dict[str, list[str]]   
    experience_bullets_by_group: dict[str, list[str]]
    project_bullets_by_group: dict[str, list[str]]
    resume_pdf_path: str                # optional path for PDF link extraction
    gap_experience: str                 # user-provided real experience for missing skills

    # 5. Job Description Understanding
    jd_title: str
    jd_skills_required: list[str]
    jd_skills_preferred: list[str]
    jd_responsibilities: list[str]
    jd_domain: str                      # finance | retail | healthcare
    jd_seniority_level: str             # junior | mid | senior | staff
    jd_clean_text: str
    jd_tools_process: list[str]
    jd_keywords: list[str]
    # 6. Skill Intelligence
    resume_skill_profile: SkillProfile
    jd_skill_profile: SkillProfile

    skill_overlap: dict[str, list[str]]     # category → matched
    skill_gap_hard: dict[str, list[str]]
    skill_gap_soft: dict[str, list[str]]

    # 7. Matching & Scoring
    semantic_similarity_score: float
    skill_match_score: float
    ats_match_score: float
    overall_match_score: float
    ats_pass_score: float               # estimated chance of clearing an ATS screen
    ats_pass_label: str                 # human verdict for ats_pass_score
    score_breakdown: dict[str, float]
    missing_required_skills: list[str]
    matched_skills: list[str]

    # 8. Decision & Strategy
    rewrite_required: bool
    rewrite_reason: str
    rewrite_strategy: str
    # "none" | "summary_only" | "summary_and_skills"
    # "experience_bullets" | "full_rewrite"

    # 9. Rewrite Outputs (Structured)
    optimized_summary: str | None
    optimized_experience_bullets: list[str] | None
    optimized_project_bullets: list[str] | None
    optimized_skills_section: str | None

    optimized_resume_text: str
    optimized_resume_struct: dict        # ATS-safe structured resume for .docx export
    bullets_needing_metric: list[str]    # rewritten bullets with no number (add one)
    rewrite_diff: list[str] | None
    resume_version: str                 # original | rewritten | reformatted
    rewrite_error: str | None

    # Iterative rewrite loop + before/after scoring
    rewrite_attempts: int
    rewrite_improved: bool              # did the latest candidate beat the best?
    rewrite_feedback_terms: list[str]   # JD skills present but under-surfaced
    best_resume_text: str
    baseline_scores: dict[str, float]   # original resume, text-based scorer
    best_scores: dict[str, float]       # best candidate so far
    rewrite_candidates: list[dict]      # per-attempt scores (audit)
    optimized_skill_match_score: float
    optimized_semantic_similarity_score: float
    optimized_ats_match_score: float
    optimized_overall_match_score: float
    optimized_ats_pass_score: float
    optimized_ats_pass_label: str
    rewrite_score_comparison: dict[str, dict[str, float]]  # metric -> {before, after, delta}

    # 10. Outreach Outputs
    outreach_dm_text: str
    outreach_email_text: str

    # 11. Agent Diagnostics
    agent_notes: list[str]
    warnings: list[str]
    rewrite_noop: bool
    resume_links: list[str]  # extracted from PDF annotations, if available
    run_artifact_path: str   # where the tracking node saved this run's summary
