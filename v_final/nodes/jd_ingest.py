import json
import re
from typing import Any, Dict, List, Tuple

import ollama
from v_final.state.job_application_state import JobApplicationState

from utils.text_normalization import normalize_text 

MODEL_NAME = "llama3.1:8b"

def _safe_list(x: Any) -> List[str]:
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]
    return []


def _safe_str(x: Any) -> str:
    return str(x).strip() if x is not None else ""

def _extract_json_block(text: str) -> str:
    """
    LLMs sometimes wrap JSON in ```json ...```.
    This extracts the first JSON object robustly.
    """
    if not text:
        return ""

    # Prefer fenced block
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Otherwise, try first {...} block
    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    return m.group(1).strip() if m else ""


def _split_compound_terms(s: str) -> List[str]:
    """
    Split terms like 'networking/communication protocols' or 'python, sql'
    into atomic tokens without being too aggressive.
    """
    s = s.strip()
    if not s:
        return []

    # split on common separators
    parts = re.split(r"\s*(/|,|;|\||&|\+)\s*", s)
    # re.split keeps separators; filter them out
    parts = [p for p in parts if p and p not in {"/", ",", ";", "|", "&", "+",}]
    return [p.strip() for p in parts if p.strip()]

def _clean_list(items: List[str]) -> List[str]:
    out: List[str] = []
    for x in items or []:
        x = str(x).strip().lower()
        x = re.sub(r"\s+", " ", x)
        x = re.sub(r"^[^\w]+|[^\w]+$", "", x)  # trim punctuation edges
        if not x:
            continue

        # Split compounds into multiple entries
        atoms = _split_compound_terms(x)
        if atoms:
            out.extend(atoms)
        else:
            out.append(x)

    # Dedupe preserve order
    seen = set()
    final = []
    for x in out:
        if x not in seen:
            final.append(x)
            seen.add(x)
    return final

def _priority_dedupe(
    required: List[str],
    preferred: List[str],
    tools_process: List[str],
    keywords: List[str],
                        ) -> Tuple[List[str], List[str], List[str], List[str]]:
    required = _clean_list(required)
    req_set = set(required)

    preferred = [x for x in _clean_list(preferred) if x not in req_set]
    pref_set = set(preferred) | req_set

    tools_process = [x for x in _clean_list(tools_process) if x not in pref_set]
    tool_set = set(tools_process) | pref_set

    keywords = [x for x in _clean_list(keywords) if x not in tool_set]
    return required, preferred, tools_process, keywords


# -------------------------
# LLM prompt (canonicalization baked in)
# -------------------------
def _build_jd_extract_prompt(jd_text: str) -> str:
    return f"""
You are an information extraction engine for job descriptions.

Return ONLY valid JSON (no markdown, no commentary).

SCHEMA (must follow exactly):
{{
  "jd_title": string,
  "domain": string,
  "seniority_level": "intern"|"junior"|"mid"|"senior"|"staff"|"principal"|"lead"|"manager"|"director"|"vp"|"unknown",
  "skills_required": [string],
  "skills_preferred": [string],
  "tools_process": [string],
  "responsibilities": [string],
  "keywords": [string]
}}

CANONICALIZATION RULES (IMPORTANT):
- All list items must be lowercase.
- Merge synonyms into canonical short forms where obvious:
  examples: "amazon web services"->"aws", "continuous integration"->"cicd",
            "machine learning" stays "machine learning", "ml"->"machine learning".
- Split compound items into atomic skills:
  example: "networking/communication protocols" -> ["networking protocols","communication protocols"]
- Keep "tools_process" limited to frameworks/tools/process terms:
  examples: agile, scrum, waterfall, sdlc, jira, itil, change management, stakeholder management.
- Put technical skills/tools (python, spark, tensorflow, kubernetes, rag, langgraph) into skills lists, NOT tools_process.
- skills_required: must-have/minimum qualifications (or clearly required).
- skills_preferred: preferred/nice-to-have/plus.
- responsibilities: 6–12 verb-led action statements.
- keywords: extra ATS terms not already included above.

JOB DESCRIPTION:
\"\"\"{jd_text}\"\"\"
""".strip()


def jd_ingest_node(state: JobApplicationState) -> JobApplicationState:
    """
    JD Ingest Node — V3 (LLM extraction + canonicalization)

    Reads:
    - job_description_text

    Writes:
    - jd_clean_text
    - jd_title
    - jd_domain
    - jd_seniority_level
    - jd_skills_required
    - jd_skills_preferred
    - jd_tools_process
    - jd_responsibilities
    - jd_keywords
    - warnings (optional)
    """
    raw_jd = state.get("job_description_text", "")
    jd_clean = normalize_text(raw_jd)
    state["jd_clean_text"] = jd_clean

    defaults: Dict[str, Any] = {
        "jd_title": "",
        "domain": "unknown",
        "seniority_level": "unknown",
        "skills_required": [],
        "skills_preferred": [],
        "tools_process": [],
        "responsibilities": [],
        "keywords": [],
    }

    prompt = _build_jd_extract_prompt(jd_clean)

    try:
        resp = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return JSON only. Follow schema strictly. Canonicalize as instructed."},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.1, "top_p": 0.9},
        )

        content = (resp.get("message", {}) or {}).get("content", "") or ""
        json_str = _extract_json_block(content)
        data: Dict[str, Any] = json.loads(json_str) if json_str else {}
        merged = {**defaults, **(data or {})}

        jd_title = _safe_str(merged.get("jd_title", ""))
        jd_domain = _safe_str(merged.get("domain", "unknown")).lower() or "unknown"
        jd_level = _safe_str(merged.get("seniority_level", "unknown")).lower() or "unknown"

        req = _safe_list(merged.get("skills_required", []))
        pref = _safe_list(merged.get("skills_preferred", []))
        tools = _safe_list(merged.get("tools_process", []))
        resp_list = _safe_list(merged.get("responsibilities", []))
        kw = _safe_list(merged.get("keywords", []))

        # Priority-based dedupe + generic cleanup
        req, pref, tools, kw = _priority_dedupe(req, pref, tools, kw)

        # responsibilities: keep as-is but clean whitespace + ensure verb-led-ish minimal cleanup
        resp_clean = []
        for r in resp_list:
            r = re.sub(r"\s+", " ", str(r).strip())
            if r:
                resp_clean.append(r)

        state["jd_title"] = jd_title
        state["jd_domain"] = jd_domain
        state["jd_seniority_level"] = jd_level
        state["jd_skills_required"] = req
        state["jd_skills_preferred"] = pref
        state["jd_tools_process"] = tools
        state["jd_responsibilities"] = resp_clean
        state["jd_keywords"] = kw

    except Exception as e:
        state.setdefault("warnings", []).append(f"jd_ingest_llm_failed: {str(e)}")
        state["jd_title"] = ""
        state["jd_domain"] = "unknown"
        state["jd_seniority_level"] = "unknown"
        state["jd_skills_required"] = []
        state["jd_skills_preferred"] = []
        state["jd_tools_process"] = []
        state["jd_responsibilities"] = []
        state["jd_keywords"] = []

    return state