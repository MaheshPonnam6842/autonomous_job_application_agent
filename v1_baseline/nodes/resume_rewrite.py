import sys
from state.job_application_state import JobApplicationState
import ollama
from src.exception import CustomException


# NOTE:
# This node introduces GenAI.
# Guardrails are enforced via:
# - rewrite_required gating
# - strict prompt constraints
# - no hallucination allowance
# V1: Conservative ATS-aligned rewriting only

MODEL_NAME = "llama3.1:8b-instruct-q4_K_M"

def _build_rewrite_prompt(resume_text: str,
                          jd_text: str,
                          missing_skills: list[str]) -> str:
    """
    Constructs the prompt for resume rewriting.
    """
    missing_skills_str = ", ".join(missing_skills) if missing_skills else "None"

    return f"""
You are a factual resume editor. Your job is to rewrite the resume to better match the job description.

ABSOLUTE SAFETY RULES (MUST FOLLOW):
1) Do NOT invent: tools, skills, companies, titles, dates, degrees, certifications, metrics, awards, or responsibilities.
2) You may ONLY use skills/tools that already appear in the Candidate Resume text.
3) If a Job Description skill is not in the Candidate Resume, do NOT add it. You may only re-emphasize related content that already exists.
4) Keep all original numbers/percentages exactly as-is. You may move them into stronger bullets but cannot change them.

INPUTS
Candidate Resume (source of truth):
{resume_text}

Job Description (target keywords and responsibilities):
{jd_text}

Skills that appear underrepresented (for emphasis only, not invention):
{missing_skills_str}

TASK
Rewrite the resume with these goals:
- Improve ATS alignment by reordering and rephrasing content (without adding new facts).
- Convert bullets into strong action-impact format while keeping the original meaning.
- Fix formatting issues (broken words, extra spaces, hyphen spacing).

MANDATORY OUTPUT FORMAT
Return ONLY the rewritten resume, using EXACTLY these sections in this order:

1) NAME + CONTACT (one line)
2) SUMMARY (3 lines max)
3) SKILLS (grouped)
4) EXPERIENCE (same employers/roles, same dates)
5) EDUCATION

SUMMARY REQUIREMENTS (STRICT):
- First sentence: "Data Scientist with X+ years..." (use the same years as resume, do not invent)
- Second sentence MUST list EXACTLY 4 skills as: "Top skills: Skill1, Skill2, Skill3, Skill4"
- Those 4 skills MUST be chosen from skills that appear in BOTH:
  (a) Candidate Resume AND (b) Job Description.
- If fewer than 4 overlaps exist, use all overlaps and fill remaining with the strongest resume skills that are most relevant to the JD (still must appear in resume).

BULLET RULES (STRICT):
- Each bullet <= 30 words.
- STAR-style phrasing: Action + What + How + Result (use existing metrics only).
- Keep the total resume length within ±10% of the original.

QUALITY CHECK (DO BEFORE YOU OUTPUT):
- Ensure you did not introduce any new skill/tool not present in the resume.
- Ensure formatting has no broken words like "f ull" or "hands -on".
- Ensure Summary has exactly 4 skills in "Top skills:" line.

Now output the rewritten resume only:
""".strip()


def resume_rewrite_node(state: JobApplicationState) -> JobApplicationState: 
    """
    Resume Rewrite Node (GenAI)

    Reads:
    - resume_raw_text
    - job_description_text
    - missing_skills
    - rewrite_required

    Writes:
    - optimized_resume_text
    - resume_version
    """
    if not state.get("rewrite_required", False):
        state["optimized_resume_text"]= state.get("resume_raw_text", "")
        state["resume_version"]= "original"
        return state
    resume_text= state.get("resume_raw_text", "")
    jd_text= state.get("job_description_text", "")
    missing_skills= state.get("missing_skills", [])
    prompt= _build_rewrite_prompt(
        resume_text= resume_text,
        jd_text= jd_text,
        missing_skills= missing_skills
    )
    #call LLM model

    # Ollama LLM call
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a careful, factual resume editor."},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.2, "top_p": 0.9,"num_predict": 500},
        )

        rewritten_resume = response["message"]["content"].strip()
        state["optimized_resume_text"] = rewritten_resume
        state["resume_version"] = "rewritten_v1"

    except Exception as e:
        # Fail-safe behavior for agents
        state["optimized_resume_text"] = resume_text
        state["resume_version"] = "rewrite_failed_fallback"
        state["rewrite_error"] = str(e)  # optional debug signal

    return state