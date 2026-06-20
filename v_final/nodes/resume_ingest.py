from utils.link_extractor import extract_pdf_links
from utils.text_normalization import (
    extract_grouped_bullets,
    normalize_text,
    parse_skills,
    split_by_sections,
)
from v_final.state.job_application_state import JobApplicationState, ResumeSections


def resume_ingest_node(state: JobApplicationState) -> JobApplicationState:
    """
    Resume Ingest Node — V2 (Fixed)

    Responsibilities:
    - Normalize resume text
    - Split resume into sections
    - Store raw sections (skills_raw, experience_raw, projects_raw, education_raw)
    - Extract bullets from experience & projects (grouped + flat)
    - Parse skills into grouped dict

    Reads:
    - resume_raw_text
    - resume_source

    Writes:
    - resume_clean_text
    - resume_sections
    - experience_groups
    - experience_bullets (flat)
    - experience_bullets_by_group
    - project_groups
    - project_bullets (flat)
    - project_bullets_by_group
    - resume_skills_structured
    """

    raw_resume = state.get("resume_raw_text", "") or ""
    state.get("resume_source", "upload")  # kept for future use

    # 1) Normalize + section split
    clean_text = normalize_text(raw_resume)
    raw_sections = split_by_sections(clean_text)  # expected keys: summary/skills/experience/projects/education

    resume_sections: ResumeSections = {
        "summary": raw_sections.get("summary", ""),
        "skills_raw": raw_sections.get("skills", ""),
        "experience_raw": raw_sections.get("experience", ""),
        "projects_raw": raw_sections.get("projects", ""),
        "education_raw": raw_sections.get("education", ""),
    }

    state["resume_clean_text"] = clean_text
    state["resume_sections"] = resume_sections

    # 2) Experience grouped + flat
    exp_raw = resume_sections.get("experience_raw", "") or ""
    exp_groups, exp_flat = extract_grouped_bullets(exp_raw, default_group="EXPERIENCE")
    state["experience_groups"] = exp_groups
    state["experience_bullets"] = exp_flat
    state["experience_bullets_by_group"] = exp_groups

    # 3) Projects grouped + flat
    proj_raw = resume_sections.get("projects_raw", "") or ""
    proj_groups, proj_flat = extract_grouped_bullets(proj_raw, default_group="PROJECTS")
    state["project_groups"] = proj_groups
    state["project_bullets"] = proj_flat
    state["project_bullets_by_group"] = proj_groups

    # 4) Skills parse (THIS was missing)
    skills_raw = resume_sections.get("skills_raw", "") or ""
    if skills_raw.strip():
        state["resume_skills_structured"] = parse_skills(skills_raw)
    else:
        state["resume_skills_structured"] = {}

    pdf_path = state.get("resume_pdf_path")
    if pdf_path:
        state["resume_links"] = extract_pdf_links(pdf_path)
    else:
        state["resume_links"] = []
    

    return state
    