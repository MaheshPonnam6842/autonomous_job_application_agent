"""Skill Extraction Node — builds canonical skill profiles and computes overlap/gaps.

Fully deterministic (no LLM): it canonicalizes skill terms, buckets them into a
:class:`SkillProfile`, and diffs the resume against the job description.

Reads:
- resume_skills_structured, resume_clean_text, experience_bullets, project_bullets
- jd_skills_required, jd_skills_preferred, jd_tools_process, jd_keywords, jd_clean_text

Writes:
- resume_skill_profile, jd_skill_profile   (SkillProfile)
- skill_overlap                            (category -> matched terms)
- skill_gap_hard, skill_gap_soft           (category -> missing terms)
"""

from __future__ import annotations

import re
from typing import Dict, List, Set

from v_final.state.job_application_state import JobApplicationState, SkillProfile

# --- Canonicalization: collapse synonyms / variants to one form ------------------
CANON: Dict[str, str] = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "genai": "gen ai",
    "gen-ai": "gen ai",
    "llms": "llm",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "ci/cd": "cicd",
    "ci cd": "cicd",
    "continuous integration": "cicd",
    "continuous delivery": "cicd",
    "continuous deployment": "cicd",
    "amazon web services": "aws",
    "google cloud platform": "gcp",
    "google cloud": "gcp",
    "microsoft azure": "azure",
    "ms azure": "azure",
    "k8s": "kubernetes",
    "hpc": "high-performance computing",
    "dsa": "algorithms and data structures",
}

# --- Category vocabularies -------------------------------------------------------
LANGUAGES = {"python", "java", "c", "c++", "c#", "r", "scala", "sql", "go", "rust"}
CLOUD = {"aws", "azure", "gcp"}
TOOLS = {
    "spark", "pyspark", "kafka", "snowflake", "databricks",
    "docker", "kubernetes", "flask", "fastapi", "airflow", "mlflow",
    "github actions", "cicd", "git",
    "sagemaker", "lambda", "ec2", "s3", "glue", "ecr", "ecs",
    "pytorch", "tensorflow", "scikit-learn", "xgboost", "lightgbm",
    "shap", "lime", "power bi", "tableau",
    "langgraph", "langchain", "rag", "embeddings",
}
ML_CONCEPTS = {
    "machine learning", "deep learning", "nlp",
    "algorithms", "data structures", "algorithms and data structures",
    "parsing", "numerical optimization", "data mining",
    "parallel and distributed computing", "distributed computing",
    "parallel computing", "high-performance computing",
    "gen ai", "agentic ai", "llm", "rag", "prompt engineering",
    "classification", "regression", "feature engineering",
    "model validation", "time series", "a/b testing",
}
SOFT_PROCESS = {
    "agile", "scrum", "sdlc",
    "stakeholder management", "communication", "collaboration", "ownership",
    "testing", "reproducible code", "operational excellence",
    "rapid experimentation", "customer impact", "responsible ai development",
}

# Vocabulary used to scan free text (bullets, summaries) for skills.
VOCAB: Set[str] = LANGUAGES | CLOUD | TOOLS | ML_CONCEPTS | SOFT_PROCESS

# Categories considered "hard" requirements when computing gaps.
_HARD_CATS = {"hard_skills", "tools", "cloud", "ml_concepts", "keywords"}
_ALL_CATS = ("hard_skills", "tools", "cloud", "ml_concepts", "soft_skills", "keywords")


def _canon(x: str) -> str:
    s = (x or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^[^\w+#]+|[^\w+#]+$", "", s)  # trim edge punctuation, keep c++ / c#
    return CANON.get(s, s)


def _dedupe(items: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items or []:
        x = _canon(x)
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _classify(term: str) -> str:
    if term in LANGUAGES:
        return "hard_skills"
    if term in CLOUD:
        return "cloud"
    if term in TOOLS:
        return "tools"
    if term in ML_CONCEPTS:
        return "ml_concepts"
    if term in SOFT_PROCESS:
        return "soft_skills"
    return "keywords"


def _build_profile(terms: List[str]) -> SkillProfile:
    buckets: Dict[str, List[str]] = {cat: [] for cat in _ALL_CATS}
    for t in _dedupe(terms):
        buckets[_classify(t)].append(t)
    return {cat: buckets[cat] for cat in _ALL_CATS if buckets[cat]}  # drop empty cats


def _scan_vocab(text: str) -> List[str]:
    """Find known skills mentioned anywhere in free text, with token boundaries
    so 'r' doesn't match 'react' and 'c' doesn't match 'classification'."""
    if not text:
        return []
    low = text.lower()
    found: List[str] = []
    for term in VOCAB:
        pattern = r"(?<![a-z0-9+#])" + re.escape(term) + r"(?![a-z0-9+#])"
        if re.search(pattern, low):
            found.append(term)
    return found


def _resume_terms(state: JobApplicationState) -> List[str]:
    terms: List[str] = []
    for key, vals in (state.get("resume_skills_structured") or {}).items():
        terms.append(key)
        terms.extend(vals or [])
    blob = "\n".join(
        [state.get("resume_clean_text", "") or ""]
        + (state.get("experience_bullets") or [])
        + (state.get("project_bullets") or [])
    )
    terms.extend(_scan_vocab(blob))
    return terms


def _jd_terms(state: JobApplicationState) -> List[str]:
    terms: List[str] = []
    for key in ("jd_skills_required", "jd_skills_preferred", "jd_tools_process", "jd_keywords"):
        terms.extend(state.get(key, []) or [])
    terms.extend(_scan_vocab(state.get("jd_clean_text", "") or ""))
    return terms


def _flatten(profile: SkillProfile) -> Set[str]:
    out: Set[str] = set()
    for vals in profile.values():
        out.update(vals)
    return out


def skill_extraction_node(state: JobApplicationState) -> JobApplicationState:
    resume_profile = _build_profile(_resume_terms(state))
    jd_profile = _build_profile(_jd_terms(state))
    resume_set = _flatten(resume_profile)

    overlap: Dict[str, List[str]] = {}
    gap_hard: Dict[str, List[str]] = {}
    gap_soft: Dict[str, List[str]] = {}

    for cat, items in jd_profile.items():
        matched = [t for t in items if t in resume_set]
        missing = [t for t in items if t not in resume_set]
        if matched:
            overlap[cat] = matched
        if missing:
            target = gap_hard if cat in _HARD_CATS else gap_soft
            target[cat] = missing

    state["resume_skill_profile"] = resume_profile
    state["jd_skill_profile"] = jd_profile
    state["skill_overlap"] = overlap
    state["skill_gap_hard"] = gap_hard
    state["skill_gap_soft"] = gap_soft
    return state
