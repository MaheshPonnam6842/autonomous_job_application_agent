"""Outreach Node — drafts (never sends) recruiter DM + email.

Templates are built deterministically from matched skills, then optionally
tone-polished by the LLM. Polishing must not introduce facts; if the model is
unavailable the deterministic draft is returned unchanged.

Reads:  matched_skills, overall_match_score
Writes: outreach_dm_text, outreach_email_text
"""

from __future__ import annotations

from v_final.llm import get_client
from v_final.state.job_application_state import JobApplicationState

_POLISH_SYSTEM = (
    "You are a professional communication editor. Polish tone and clarity only. "
    "Do NOT add facts, skills, metrics, or claims."
)


def _build_base_dm(strong_matches: list[str]) -> str:
    highlights = ", ".join(strong_matches[:3]) if strong_matches else "relevant experience"
    return (
        "Hi {recruiter_name},\n\n"
        "I recently applied for this role and wanted to briefly follow up. "
        f"My background aligns well with the position, particularly around {highlights}. "
        "I'd love the opportunity to discuss how my experience could add value to your team.\n\n"
        "Best regards,\n{your_name}"
    )


def _build_base_email(strong_matches: list[str]) -> str:
    highlights = ", ".join(strong_matches[:4]) if strong_matches else "relevant experience"
    return (
        "Subject: Application Follow-Up\n\n"
        "Dear Hiring Team,\n\n"
        "I hope you're doing well. I recently applied for the position and wanted to follow up. "
        f"My experience aligns closely with the role, particularly in {highlights}. "
        "I would welcome the opportunity to discuss how my background could support your team's goals.\n\n"
        "Thank you for your time and consideration.\n\n"
        "Sincerely,\n{your_name}"
    )


def _polish(text: str) -> str:
    res = get_client().chat(_POLISH_SYSTEM, text, temperature=0.3, num_predict=500)
    return res.text if (res.ok and res.text) else text  # fail-safe: keep the draft


def outreach_node(state: JobApplicationState) -> JobApplicationState:
    matches = state.get("matched_skills", []) or []
    state["outreach_dm_text"] = _polish(_build_base_dm(matches))
    state["outreach_email_text"] = _polish(_build_base_email(matches))
    return state
