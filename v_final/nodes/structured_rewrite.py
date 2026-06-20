"""Structured Rewrite Node — XYZ bullets, fact-preserving, ATS-safe assembly.

Replaces the old free-form rewrite. The model only regenerates *narrative*
(summary, skill ordering, bullet wording in Google XYZ form); the agent keeps
employers, titles, dates, and education **verbatim** from the parsed structure
and assembles the final resume deterministically into a single-column, standard-
heading layout that ATS parsers read cleanly.

- Online (Ollama up): bullets rewritten as "Accomplished X (as measured by Y) by
  doing Z", existing metrics preserved, never invented.
- Offline / LLM failure: still assembles an ATS-clean resume from the original
  structured content (no XYZ rewrite) so the .docx export always works.

Reads:  rewrite_required, experience_entries, project_entries, education_entries,
        resume_sections, resume_skills_structured, resume_raw_text, jd_skills_*,
        missing_required_skills, rewrite_feedback_terms
Writes: optimized_resume_text, optimized_resume_struct, bullets_needing_metric,
        resume_version, rewrite_noop, warnings
"""

from __future__ import annotations

import re

from v_final.config import settings
from v_final.llm import StructuredRewrite, get_client
from v_final.state.job_application_state import JobApplicationState

_SYSTEM = (
    "You are an expert resume writer. You rewrite experience bullets in the Google "
    "XYZ format and you NEVER invent facts, skills, employers, dates, or numbers."
)

_METRIC_RE = re.compile(r"[0-9]|%|\$|\bpercent\b", re.IGNORECASE)
_HEADER_STOP = {
    "summary", "professional summary", "profile", "objective",
    "skills", "technical skills", "core skills",
    "experience", "work experience", "professional experience",
    "education", "projects", "certifications",
}


def _has_metric(text: str) -> bool:
    return bool(_METRIC_RE.search(text or ""))


def _dates(entry: dict) -> str:
    a, b = entry.get("start_date"), entry.get("end_date")
    return f"{a} - {b}" if a and b else (a or b or "")


def _extract_header(raw: str) -> list[str]:
    """Name + contact lines that appear before the first section heading."""
    out: list[str] = []
    for ln in (raw or "").split("\n"):
        s = ln.strip()
        if not s:
            continue
        if s.lower() in _HEADER_STOP:
            break
        out.append(s)
        if len(out) >= 3:
            break
    return out


def _flat_skills(state: JobApplicationState) -> list[str]:
    out: list[str] = []
    for vals in (state.get("resume_skills_structured") or {}).values():
        out.extend(vals or [])
    seen, dedup = set(), []
    for s in out:
        k = s.strip().lower()
        if k and k not in seen:
            seen.add(k)
            dedup.append(s.strip())
    return dedup


def _build_prompt(state: JobApplicationState, entries: list[dict],
                  summary: str, skills: list[str], gap_experience: str = "") -> str:
    jd_terms = []
    for key in ("jd_skills_required", "jd_skills_preferred", "jd_keywords"):
        jd_terms.extend(state.get(key, []) or [])
    feedback = state.get("rewrite_feedback_terms", []) or []
    fb_line = ("\nEMPHASIZE these skills the candidate already has but under-used: "
               + ", ".join(feedback) + "\n") if feedback else ""

    exp_block = []
    for i, e in enumerate(entries):
        head = f"[{i}] {e.get('company','')} - {e.get('role','')} ({_dates(e)})".strip()
        exp_block.append(head)
        for b in e.get("bullets", []) or []:
            exp_block.append(f"- {b}")
    n = len(entries)

    gap_block = ""
    if gap_experience:
        gap_block = (
            "\nUSER-PROVIDED EXPERIENCE (the candidate typed this about skills missing from "
            "the resume — TRUE and user-authorized). Write 1-3 Google-XYZ bullets into "
            "\"added_bullets\" based on it. EXCEPTION to rule 2 for these bullets only: each "
            "MUST include a realistic, specific metric (%, count, time saved, or $) — use the "
            "user's number if they gave one, otherwise a believable estimate they can adjust. "
            "Stay within the scope of what the user described:\n" + gap_experience + "\n"
        )

    return f"""
Rewrite the candidate's resume CONTENT for this job using the Google XYZ formula
on EVERY experience bullet: "Accomplished [X] as measured by [Y], by doing [Z]"
(X = impact/result, Y = a number/metric, Z = the action/method).

HARD RULES:
1) Do NOT invent skills, tools, employers, titles, dates, metrics, or numbers.
2) Keep every existing number/percentage EXACTLY; rephrase around it.
3) If a bullet has no number, still rewrite it as a strong action->result->method
   sentence; do NOT fabricate a metric.
4) Use ONLY skills the candidate already has (plus the USER-PROVIDED EXPERIENCE below,
   if any); reorder skills so job-relevant ones come first.
5) Each bullet: one sentence, <= 30 words, starts with a strong past-tense verb.

TARGET JOB SKILLS (emphasis only): {", ".join(jd_terms) or "n/a"}{fb_line}
CANDIDATE SUMMARY:
{summary or "(none)"}

CANDIDATE SKILLS (only these may be used):
{", ".join(skills) or "(none)"}

CANDIDATE EXPERIENCE (rewrite the bullets of each entry; keep the same order):
{chr(10).join(exp_block) or "(none)"}
{gap_block}
Return ONLY JSON:
{{
  "summary": "2-3 sentence factual summary aligned to the job",
  "skills": ["skill1", "skill2"],
  "experience": [{{"bullets": ["..."]}}],
  "added_bullets": ["XYZ bullets from USER-PROVIDED EXPERIENCE only; [] if none"]
}}
The "experience" array MUST have exactly {n} item(s) in the same order, each with
the rewritten bullets for that entry.
""".strip()


def _assemble(state: JobApplicationState, summary: str, skills: list[str],
              exp_bullets: list[list[str]]) -> tuple[str, dict]:
    """Deterministically assemble ATS-safe text + a struct for the .docx builder.
    Facts (company/role/dates/education) come straight from the parsed entries."""
    entries = state.get("experience_entries") or []
    projects = state.get("project_entries") or []
    education = state.get("education_entries") or []
    header = _extract_header(state.get("resume_raw_text", "") or state.get("resume_clean_text", ""))
    links = state.get("resume_links", []) or []

    needing_metric: list[str] = []
    struct_exp = []
    for i, e in enumerate(entries):
        bullets = exp_bullets[i] if i < len(exp_bullets) and exp_bullets[i] else (e.get("bullets") or [])
        bl = []
        for b in bullets:
            hm = _has_metric(b)
            if not hm:
                needing_metric.append(b)
            bl.append({"text": b, "has_metric": hm})
        struct_exp.append({
            "company": e.get("company", ""), "role": e.get("role", ""),
            "dates": _dates(e), "bullets": bl,
        })

    struct_proj = [{"name": p.get("name", ""), "bullets": p.get("bullets", []) or []} for p in projects]
    struct_edu = []
    for ed in education:
        parts = [ed.get("degree", ""), ed.get("field", "")]
        line = ", ".join(p for p in parts if p)
        if ed.get("institution"):
            line += f"  |  {ed['institution']}"
        if ed.get("year"):
            line += f" ({ed['year']})"
        if line.strip():
            struct_edu.append(line)

    struct = {
        "header": header, "links": links, "summary": summary, "skills": skills,
        "experience": struct_exp, "projects": struct_proj, "education": struct_edu,
    }

    # Render plain text (ATS-safe: single column, standard headings, hyphen bullets)
    lines: list[str] = list(header)
    if links:
        lines.append(" | ".join(links))
    if summary:
        lines += ["", "PROFESSIONAL SUMMARY", summary]
    if skills:
        lines += ["", "SKILLS", ", ".join(skills)]
    if struct_exp:
        lines += ["", "EXPERIENCE"]
        for e in struct_exp:
            lines.append("")
            lines.append(e["company"])
            role_line = e["role"] + (f"  |  {e['dates']}" if e["dates"] else "")
            if role_line.strip():
                lines.append(role_line)
            for b in e["bullets"]:
                lines.append(f"- {b['text']}")
    if struct_proj:
        lines += ["", "PROJECTS"]
        for p in struct_proj:
            lines.append("")
            lines.append(p["name"])
            for b in p["bullets"]:
                lines.append(f"- {b}")
    if struct_edu:
        lines += ["", "EDUCATION"]
        lines += struct_edu

    return "\n".join(lines).strip(), {**struct, "bullets_needing_metric": needing_metric}


def structured_rewrite_node(state: JobApplicationState) -> JobApplicationState:
    gap_experience = (state.get("gap_experience") or "").strip()
    # User-supplied experience for missing skills forces a rewrite even on a strong fit.
    needs_rewrite = bool(state.get("rewrite_required", False)) or bool(gap_experience)

    if not needs_rewrite:
        # Strong fit, no user input: don't rewrite content, just ATS-format it (no LLM).
        summary = (state.get("resume_sections") or {}).get("summary", "") or ""
        text, struct = _assemble(state, summary, _flat_skills(state), [])
        state["optimized_resume_text"] = text
        state["optimized_resume_struct"] = struct
        state["bullets_needing_metric"] = struct["bullets_needing_metric"]
        state["resume_version"] = "original"
        state["rewrite_noop"] = True
        return state

    entries = state.get("experience_entries") or []
    orig_summary = (state.get("resume_sections") or {}).get("summary", "") or ""
    orig_skills = _flat_skills(state)

    sr: StructuredRewrite | None = None
    if entries or orig_summary or gap_experience:
        sr = get_client().chat_structured(
            _SYSTEM, _build_prompt(state, entries, orig_summary, orig_skills, gap_experience),
            StructuredRewrite, model=settings.llm.rewrite_model,
        )

    if sr is not None:
        summary = sr.summary or orig_summary
        skills = sr.skills or orig_skills
        exp_bullets = [list(e.bullets) for e in sr.experience]
        # Bullets written from the user's supplied experience attach to the most recent role.
        if sr.added_bullets:
            if not exp_bullets:
                exp_bullets = [[]]
            exp_bullets[0] = exp_bullets[0] + list(sr.added_bullets)
        version = "rewritten"
    else:
        # Offline / failed: assemble ATS-clean resume from the original content.
        summary, skills, exp_bullets = orig_summary, orig_skills, []
        version = "reformatted"
        state.setdefault("warnings", []).append(
            "structured_rewrite_llm_unavailable: assembled ATS format without XYZ rewrite"
        )

    text, struct = _assemble(state, summary, skills, exp_bullets)
    state["optimized_resume_text"] = text
    state["optimized_resume_struct"] = struct
    state["bullets_needing_metric"] = struct["bullets_needing_metric"]
    state["resume_version"] = version
    state["rewrite_noop"] = version != "rewritten"
    return state
