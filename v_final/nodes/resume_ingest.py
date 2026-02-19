from v_final.state.job_application_state import JobApplicationState, ResumeSections
from utils.text_normalization import extract_grouped_bullets,normalize_text, split_by_sections,_extract_bullets,parse_skills


def resume_ingest_node(state: JobApplicationState) -> JobApplicationState: 
    """
    Resume Ingest Node — V2

    Responsibilities:
    - Normalize resume text
    - Split resume into sections
    - Store raw sections (skills_raw, experience_raw, projects_raw, education_raw)
    - Extract bullets from experience & projects (flat lists)
    - Parse skills into grouped dict (optional helper for later nodes)

    Reads:
    - resume_raw_text
    - resume_source

    Writes:
    - resume_clean_text
    - resume_sections
    - experience_bullets
    - project_bullets
    - agent_notes (optional)
    """

    raw_resume= state.get("resume_raw_text","")
    source= state.get("resume_source","upload")

    clean_text= normalize_text(raw_resume)
    raw_sections= split_by_sections(clean_text)
    resume_sections: ResumeSections= {
        "summary": raw_sections.get("summary", ""),
        "skills_raw": raw_sections.get("skills", ""),
        "experience_raw": raw_sections.get("experience", ""),
        "projects_raw": raw_sections.get("projects", ""),
        "education_raw": raw_sections.get("education", ""),       
    }
    state["resume_clean_text"] = clean_text
    state["resume_sections"] = resume_sections


    # Extract bullets from experience and projects
    exp_raw = state["resume_sections"]["experience_raw"]
    exp_groups, exp_flat = extract_grouped_bullets(exp_raw, default_group="EXPERIENCE")
    state["experience_groups"] = exp_groups
    state["experience_bullets"] = exp_flat

    # Projects grouped + flat
    proj_raw = state["resume_sections"]["projects_raw"]
    proj_groups, proj_flat = extract_grouped_bullets(proj_raw, default_group="PROJECTS")
    state["project_groups"] = proj_groups
    state["project_bullets"] = proj_flat

    if resume_sections["skills_raw"].strip():
        state["resume_skills_structured"] = parse_skills(resume_sections["skills_raw"])
    else:
        state["resume_skills_structured"] = {}

    return state




    
    