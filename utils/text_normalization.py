import re
from typing import Dict, List


SECTION_HEADERS: Dict[str, List[str]] = {
    "summary": ["summary", "profile"],
    "skills": ["skills", "technical skills"],
    "experience": ["experience", "work experience"],
    "projects": ["projects", "personal projects"],
    "education": ["education", "academic"],
}


def normalize_text(text: str) -> str:
    """
    Normalize resume text while preserving semantic line structure.
    """
    if not text:
        return ""

    text = text.replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)

def _normalize_header(line: str) -> str:
    line = line.lower()
    line = re.sub(r"[^a-z ]", " ", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()

def _is_header_candidate(line: str) -> bool:
    words = line.split()
    return 1 <= len(words) <= 3

def split_by_sections(text: str) -> Dict[str, List[str]]:
    """
    Split normalized resume text into canonical sections.
    """
    sections: Dict[str, List[str]] = {}
    current_section = None
    buffer:list = []
    
    for line in text.split("\n"):
        found_header = False
        normalized_line= _normalize_header(line)
        if normalized_line and _is_header_candidate(normalized_line):
            for section, keywords in SECTION_HEADERS.items():
                if any (normalized_line == k or normalized_line.startswith(k) for k in keywords):
                    if current_section and buffer:
                        sections[current_section]= "\n".join(buffer) #header found, save previous section with no content empty list[]
                    current_section = section 
                    buffer = []
                    found_header = True #found a header
                    break
        
        if not found_header and current_section: # if no header found yet, continue adding to current section (content) [not empty list]
            buffer.append(line)

    if current_section and buffer: #save last section
        sections[current_section] = "\n".join(buffer)

    return sections

def _extract_bullets(section_text: str) -> List[str]:
    """
    Extract bullet points from a section.
    Rules:
    - A new bullet starts ONLY when an explicit bullet marker appears
      (-, *, •, 1., 2), etc.)
    - All following lines belong to the same bullet until a new marker appears
    - Periods/full stops are ignored for splitting
    """
    bullets: List[str] = []
    current_bullet: str | None = None
    bullet_started = False

    for line in section_text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Case 1: Symbol bullets
        if line.startswith(("-", "*", "•")):
            bullet_started= True
            if current_bullet:
                bullets.append(current_bullet.strip())
            current_bullet = line[1:].strip()

        # Case 2: Numbered bullets (1. or 1))
        elif re.match(r"^\d+[\.\)]\s*", line):
            bullet_started= True
            if current_bullet:
                bullets.append(current_bullet.strip())
            current_bullet = re.sub(r"^\d+[\.\)]\s*", "", line).strip()

        # Case 3: Continuation line
        else:
            if bullet_started and current_bullet:
                current_bullet += " " + line
            else:
                continue  # Ignore lines before the first bullet

    # Save the last bullet
    if current_bullet:
        bullets.append(current_bullet.strip())

    return bullets

def parse_skills(section_text: str) -> Dict[str, List[str]]:
    """
    Convert skills section into grouped skills if possible.
    Example:
        ML: regression, classification
    """
    skills: Dict[str, List[str]] = {}

    for line in section_text.split("\n"):
        if ":" in line:
            key, values = line.split(":", 1)
            skills[key.strip()] = [v.strip() for v in values.split(",") if v.strip()]
        else:
            skills.setdefault("General", []).append(line.strip())

    return skills

import re
from typing import Dict, List, Tuple

_BULLET_RE = re.compile(r"^(\s*[-*•]\s+|\s*\d+[\.\)]\s+)(.+)$")

def _is_heading_line(line: str) -> bool:
    """
    Heuristic: company/project headings are short, not bullets, not a sentence.
    """
    s = line.strip()
    if not s:
        return False
    if _BULLET_RE.match(s):
        return False
    # avoid skill-like lines or long sentences
    if ":" in s:
        return False
    if len(s) > 60:
        return False
    # avoid lines that look like dates/ranges
    if re.search(r"\b(19|20)\d{2}\b", s):
        return False
    # usually 1–6 words
    words = s.split()
    return 1 <= len(words) <= 6

def extract_grouped_bullets(section_text: str, default_group: str = "UNKNOWN") -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Parses sections like Experience/Projects into:
    - groups: heading -> list of bullets
    - flat_bullets: all bullets in order
    Handles heading lines appearing between bullets (won't be appended into bullet text).
    """
    groups: Dict[str, List[str]] = {}
    flat: List[str] = []

    current_group = default_group
    groups.setdefault(current_group, [])

    current_bullet: str | None = None
    bullet_started = False

    for raw in section_text.split("\n"):
        line = raw.strip()
        if not line:
            continue

        # Heading line -> close any active bullet and switch group
        if _is_heading_line(line):
            if current_bullet:
                groups[current_group].append(current_bullet.strip())
                flat.append(current_bullet.strip())
                current_bullet = None
            bullet_started = False

            current_group = line
            groups.setdefault(current_group, [])
            continue

        m = _BULLET_RE.match(line)
        if m:
            bullet_started = True
            if current_bullet:
                groups[current_group].append(current_bullet.strip())
                flat.append(current_bullet.strip())
            current_bullet = m.group(2).strip()
        else:
            # continuation line, but only if we already started a bullet
            if bullet_started and current_bullet:
                current_bullet += " " + line

    if current_bullet:
        groups[current_group].append(current_bullet.strip())
        flat.append(current_bullet.strip())

    # remove empty default group if never used
    if default_group in groups and not groups[default_group]:
        groups.pop(default_group, None)

    return groups, flat




       
