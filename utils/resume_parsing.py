"""Deterministic structured parsing of resume sections.

Turns the raw section text produced by ``resume_ingest`` into typed entries:
experience (company / role / dates / bullets), projects (name / bullets), and
education (degree / field / institution / year).

The approach is rule-based on purpose: it's fully offline, explainable, and
testable, and it degrades gracefully on unusual formats (worst case: a field is
empty, never a crash). The core idea is to classify each line as one of:
- a *bullet*        (starts with a bullet marker)
- a *continuation*  (a wrapped bullet line: starts lowercase or ends with '.')
- a *structural*    (everything else: company, role, date, title, institution)

Experience/project entries are then read as "a block of structural lines
followed by its bullets".
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from v_final.state.job_application_state import (
    EducationEntry,
    ExperienceEntry,
    ProjectEntry,
)

_BULLET_PREFIX = re.compile(r"^\s*([-*•▪◦‣·]|\d+[.)])\s+")
_ROLE_KEYWORDS = {
    "scientist", "engineer", "analyst", "developer", "manager", "consultant",
    "intern", "lead", "architect", "designer", "specialist", "director",
    "administrator", "researcher", "associate", "head",
}
_DATE_RE = re.compile(
    r"(present|current|\d{4}|"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?)",
    re.IGNORECASE,
)
_DATE_SEP_RE = re.compile(r"\s*(?:-|–|—|to)\s*", re.IGNORECASE)
_DEGREE_RE = re.compile(
    r"\b(ph\.?\s?d|doctorate|master'?s?|m\.?\s?s\.?|m\.?\s?tech|mba|"
    r"bachelor'?s?|b\.?\s?s\.?|b\.?\s?tech|b\.?\s?e\.?|associate)\b",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"(19|20)\d{2}")
_PROJECT_LINK_RE = re.compile(r"\s*[-–—]?\s*link\b.*$", re.IGNORECASE)


def _lines(text: str) -> List[str]:
    return [ln.strip() for ln in (text or "").split("\n") if ln.strip()]


def _is_bullet(s: str) -> bool:
    return bool(_BULLET_PREFIX.match(s))


def _strip_bullet(s: str) -> str:
    return _BULLET_PREFIX.sub("", s, count=1).strip()


def _is_structural(s: str) -> bool:
    """A heading-type line: company, role, date, project title, institution."""
    s = s.strip()
    if not s or _is_bullet(s):
        return False
    if s[0].islower():        # wrapped continuation of a bullet
        return False
    if s.endswith("."):       # sentence fragment continuation
        return False
    return True


def _looks_like_date(s: str) -> bool:
    return bool(_DATE_RE.search(s)) and len(s.split()) <= 6


def _looks_like_role(s: str) -> bool:
    return any(w.strip(",.").lower() in _ROLE_KEYWORDS for w in s.split())


def _parse_date_range(line: str) -> Tuple[Optional[str], Optional[str]]:
    parts = [p.strip() for p in _DATE_SEP_RE.split(line) if p.strip()]
    if len(parts) >= 2:
        return parts[0], parts[-1]
    if len(parts) == 1:
        return parts[0], None
    return None, None


def _blocks(lines: List[str]):
    """Yield (structural_lines, bullets) pairs, joining wrapped bullet lines."""
    i, n = 0, len(lines)
    while i < n:
        header: List[str] = []
        while i < n and _is_structural(lines[i]):
            header.append(lines[i])
            i += 1
        bullets: List[str] = []
        while i < n and not _is_structural(lines[i]):
            if _is_bullet(lines[i]):
                bullets.append(_strip_bullet(lines[i]))
            elif bullets:
                bullets[-1] += " " + lines[i]   # continuation of previous bullet
            i += 1
        if header or bullets:
            yield header, bullets


def parse_experience_entries(raw: str) -> List[ExperienceEntry]:
    entries: List[ExperienceEntry] = []
    for header, bullets in _blocks(_lines(raw)):
        remaining = list(header)
        start = end = None
        role = ""

        date_line = next((h for h in remaining if _looks_like_date(h)), None)
        if date_line:
            start, end = _parse_date_range(date_line)
            remaining.remove(date_line)

        role_line = next((h for h in remaining if _looks_like_role(h)), None)
        if role_line:
            role = role_line
            remaining.remove(role_line)

        company = remaining[0] if remaining else ""
        entries.append(
            ExperienceEntry(
                company=company, role=role,
                start_date=start, end_date=end, bullets=bullets,
            )
        )
    return entries


def parse_project_entries(raw: str) -> List[ProjectEntry]:
    entries: List[ProjectEntry] = []
    for header, bullets in _blocks(_lines(raw)):
        if not header and not bullets:
            continue
        name = _PROJECT_LINK_RE.sub("", header[0]).strip() if header else ""
        description = " ".join(header[1:]) if len(header) > 1 else None
        entries.append(ProjectEntry(name=name, description=description, bullets=bullets))
    return entries


def _split_degree_field(line: str) -> Tuple[str, str]:
    if "," in line:
        degree, field = line.split(",", 1)
        return degree.strip(), field.strip()
    m = _DEGREE_RE.search(line)
    return (m.group(0).strip() if m else line.strip()), ""


def parse_education_entries(raw: str) -> List[EducationEntry]:
    entries: List[EducationEntry] = []
    pending_institution: Optional[str] = None
    leftover_years: List[str] = []

    for line in _lines(raw):
        is_year_line = _YEAR_RE.search(line) and not _DEGREE_RE.search(line) and len(line.split()) <= 6
        if is_year_line:
            leftover_years.append(line)
            continue
        if _DEGREE_RE.search(line):
            degree, field = _split_degree_field(line)
            ym = _YEAR_RE.search(line)
            entries.append(
                EducationEntry(
                    degree=degree, field=field,
                    institution=pending_institution or "",
                    year=ym.group(0) if ym else None,
                )
            )
            pending_institution = None
        else:
            pending_institution = line

    # Best-effort: assign trailing standalone year lines to entries missing a year.
    yi = 0
    for entry in entries:
        if not entry.get("year") and yi < len(leftover_years):
            entry["year"] = leftover_years[yi]
            yi += 1
    return entries
