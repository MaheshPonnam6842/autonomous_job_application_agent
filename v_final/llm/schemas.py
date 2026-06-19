"""Pydantic schemas for validated LLM output.

Using a schema (rather than trusting raw JSON) is what makes LLM extraction
safe to depend on downstream: every field has a type and a default, so a
malformed or partial model response degrades to a well-formed object instead of
crashing a node.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

ALLOWED_SENIORITY = {
    "intern", "junior", "mid", "senior", "staff", "principal",
    "lead", "manager", "director", "vp", "unknown",
}


def _as_str_list(v: object) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v.strip() else []
    if isinstance(v, (list, tuple, set)):
        return [str(i).strip() for i in v if str(i).strip()]
    return []


class JDExtraction(BaseModel):
    """Structured understanding of a job description."""

    jd_title: str = ""
    domain: str = "unknown"
    seniority_level: str = "unknown"
    skills_required: list[str] = Field(default_factory=list)
    skills_preferred: list[str] = Field(default_factory=list)
    tools_process: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)

    @field_validator(
        "skills_required", "skills_preferred", "tools_process",
        "responsibilities", "keywords", mode="before",
    )
    @classmethod
    def _coerce_lists(cls, v: object) -> list[str]:
        return _as_str_list(v)

    @field_validator("jd_title", "domain", mode="before")
    @classmethod
    def _coerce_str(cls, v: object) -> str:
        return str(v).strip() if v is not None else ""

    @field_validator("seniority_level", mode="before")
    @classmethod
    def _coerce_seniority(cls, v: object) -> str:
        s = (str(v).strip().lower() if v is not None else "unknown")
        return s if s in ALLOWED_SENIORITY else "unknown"


class RewrittenExperience(BaseModel):
    """Rewritten bullets for one experience entry (company/role/dates are NOT
    sent back by the model — they're preserved verbatim from the parsed entry)."""

    bullets: list[str] = Field(default_factory=list)

    @field_validator("bullets", mode="before")
    @classmethod
    def _coerce(cls, v: object) -> list[str]:
        return _as_str_list(v)


class StructuredRewrite(BaseModel):
    """The model's structured rewrite. Facts (employers/titles/dates/education)
    are excluded on purpose — only narrative content is regenerated."""

    summary: str = ""
    skills: list[str] = Field(default_factory=list)
    experience: list[RewrittenExperience] = Field(default_factory=list)

    @field_validator("summary", mode="before")
    @classmethod
    def _coerce_summary(cls, v: object) -> str:
        return str(v).strip() if v is not None else ""

    @field_validator("skills", mode="before")
    @classmethod
    def _coerce_skills(cls, v: object) -> list[str]:
        return _as_str_list(v)
