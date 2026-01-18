from state.job_application_state import JobApplicationState
import ollama

MODEL_NAME = "llama3.1:8b-instruct-q4_K_M"
# match_score reserved for future conditional messaging


def _build_base_dm(
    match_score: float,
    strong_matches: list[str],
) -> str:
    highlights = ", ".join(strong_matches[:3]) if strong_matches else "relevant experience"

    return (
        f"Hi {{recruiter_name}},\n\n"
        f"I recently applied for this role and wanted to briefly follow up. "
        f"My background aligns well with the position, particularly around {highlights}. "
        f"I’d love the opportunity to discuss how my experience could add value to your team.\n\n"
        f"Best regards,\n"
        f"{{your_name}}"
    )


def _build_base_email(
    match_score: float,
    strong_matches: list[str],
) -> str:
    highlights = ", ".join(strong_matches[:4]) if strong_matches else "relevant experience"

    return (
        f"Subject: Application Follow-Up\n\n"
        f"Dear Hiring Team,\n\n"
        f"I hope you’re doing well. I recently applied for the position and wanted to follow up. "
        f"My experience aligns closely with the role, particularly in {highlights}. "
        f"I would welcome the opportunity to discuss how my background could support your team’s goals.\n\n"
        f"Thank you for your time and consideration.\n\n"
        f"Sincerely,\n"
        f"{{your_name}}"
    )


def _polish_text_with_llm(text: str) -> str:
    """
    Optional tone polish. Content must not change.
    """
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional communication editor. "
                        "Polish tone and clarity only. "
                        "Do NOT add facts, skills, metrics, or claims."
                    ),
                },
                {"role": "user", "content": text},
            ],
            options={"temperature": 0.3, "top_p": 0.9,"num_predict": 500},
        )
        return response["message"]["content"].strip()
    except Exception:
        return text  # fail safe


def outreach_node(state: JobApplicationState) -> JobApplicationState:
    """
    Outreach Node

    Reads:
    - match_score
    - strong_matches

    Writes:
    - outreach_dm_text
    - outreach_email_text
    """
    match_score = state.get("match_score", 0.0)
    strong_matches = state.get("strong_matches", [])

    dm_text = _build_base_dm(match_score, strong_matches)
    email_text = _build_base_email(match_score, strong_matches)

    dm_text = _polish_text_with_llm(dm_text)
    email_text = _polish_text_with_llm(email_text)

    state["outreach_dm_text"] = dm_text
    state["outreach_email_text"] = email_text

    return state
