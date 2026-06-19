"""Resume Structure Node — parses raw sections into typed entries.

Runs after ``resume_ingest`` (which produces the section text) and populates the
structured fields the state contract declares: experience, projects, education.
Deterministic and offline.

Reads:  resume_sections
Writes: experience_entries, project_entries, education_entries
"""

from __future__ import annotations

from v_final.state.job_application_state import JobApplicationState

from utils.resume_parsing import (
    parse_education_entries,
    parse_experience_entries,
    parse_project_entries,
)


def resume_structure_node(state: JobApplicationState) -> JobApplicationState:
    sections = state.get("resume_sections") or {}
    state["experience_entries"] = parse_experience_entries(sections.get("experience_raw", ""))
    state["project_entries"] = parse_project_entries(sections.get("projects_raw", ""))
    state["education_entries"] = parse_education_entries(sections.get("education_raw", ""))
    return state
