"""Resume Rewrite Node (GenAI) — scoped, factual, guardrailed rewriting.

Only runs when the decision node sets ``rewrite_required``. The prompt enforces
hard anti-hallucination rules (no invented skills/metrics/employers), and the
``rewrite_strategy`` chosen upstream narrows what the model is allowed to touch.
If the LLM is unavailable or errors, the node returns the original resume — the
agent never silently drops the candidate's content.

Reads:  rewrite_required, rewrite_strategy, resume_clean_text/raw, jd_clean_text,
        missing_required_skills, matched_skills
Writes: optimized_resume_text, resume_version, rewrite_error, rewrite_noop
"""

from __future__ import annotations

from v_final.config import settings
from v_final.llm import get_client
from v_final.state.job_application_state import JobApplicationState

_SYSTEM = "You are a careful, factual resume editor. You never invent facts."

# What each strategy is permitted to change, injected into the prompt.
_STRATEGY_SCOPE = {
    "summary_only": "Rewrite ONLY the professional summary. Leave all other sections byte-for-byte identical.",
    "summary_and_skills": "Rewrite the summary and reorganize/re-emphasize the SKILLS section only. Do not touch experience bullets.",
    "experience_bullets": "Rewrite the summary and sharpen EXPERIENCE bullets into action-impact form. Keep skills and education as-is.",
    "full_rewrite": "Rewrite the full resume across all sections.",
}


def _build_rewrite_prompt(resume_text: str, jd_text: str, missing_skills: list[str],
                          strategy: str, feedback_terms: list[str] | None = None) -> str:
    missing = ", ".join(missing_skills) if missing_skills else "None"
    scope = _STRATEGY_SCOPE.get(strategy, _STRATEGY_SCOPE["full_rewrite"])
    feedback_block = ""
    if feedback_terms:
        feedback_block = (
            "\nREVISION FEEDBACK: a previous draft under-used these skills that ALREADY "
            "appear in the resume. Surface them more explicitly (in the summary, skills, "
            "or bullets) WITHOUT inventing anything new:\n" + ", ".join(feedback_terms) + "\n"
        )
    return f"""
You are a factual resume editor. Rewrite the resume to better match the job description.

REWRITE SCOPE FOR THIS RUN:
{scope}

ABSOLUTE SAFETY RULES (MUST FOLLOW):
1) Do NOT invent: tools, skills, companies, titles, dates, degrees, certifications, metrics, awards, or responsibilities.
2) You may ONLY use skills/tools that already appear in the Candidate Resume text.
3) If a Job Description skill is not in the Candidate Resume, do NOT add it. You may only re-emphasize related content that already exists.
4) Keep all original numbers/percentages exactly as-is.

INPUTS
Candidate Resume (source of truth):
{resume_text}

Job Description (target keywords and responsibilities):
{jd_text}

Skills underrepresented vs the JD (for emphasis only, never invention):
{missing}
{feedback_block}
OUTPUT FORMAT
Return ONLY the rewritten resume, using these sections in order:
1) NAME + CONTACT (one line)
2) SUMMARY (3 lines max; first line "Data Scientist with X+ years..." using the resume's own years)
3) SKILLS (grouped)
4) EXPERIENCE (same employers/roles/dates)
5) EDUCATION

BULLET RULES:
- Each bullet <= 30 words, action-impact phrasing, existing metrics only.
- Keep total length within +/-10% of the original.

QUALITY CHECK BEFORE OUTPUT:
- No new skill/tool not present in the resume.
- No broken words like "f ull" or "hands -on".

Now output the rewritten resume only:
""".strip()


def resume_rewrite_node(state: JobApplicationState) -> JobApplicationState:
    if not state.get("rewrite_required", False):
        state["optimized_resume_text"] = state.get("resume_clean_text") or state.get("resume_raw_text", "")
        state["resume_version"] = "original"
        state["rewrite_noop"] = True
        return state

    resume_text = state.get("resume_clean_text") or state.get("resume_raw_text", "")
    jd_text = state.get("jd_clean_text") or state.get("job_description_text", "")
    strategy = state.get("rewrite_strategy", "full_rewrite")
    missing = state.get("missing_required_skills", []) or []

    # Iterative loop bookkeeping: later attempts get feedback and a little more
    # temperature so retries actually differ from the first draft.
    attempt = int(state.get("rewrite_attempts", 0))
    feedback = state.get("rewrite_feedback_terms", []) or []
    temperature = min(0.5, 0.2 + 0.1 * attempt)

    prompt = _build_rewrite_prompt(resume_text, jd_text, missing, strategy, feedback_terms=feedback)
    result = get_client().chat(
        _SYSTEM, prompt, model=settings.llm.rewrite_model,
        temperature=temperature, num_predict=900,
    )

    if result.ok and result.text:
        state["optimized_resume_text"] = result.text
        state["resume_version"] = "rewritten"
        state["rewrite_noop"] = False
    else:
        state["optimized_resume_text"] = resume_text
        state["resume_version"] = "rewrite_failed_fallback"
        state["rewrite_noop"] = True
        if result.error:
            state["rewrite_error"] = result.error
            state.setdefault("warnings", []).append(f"resume_rewrite_failed: {result.error}")
    return state
