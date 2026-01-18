from state.job_application_state import JobApplicationState
from typing import List, Set

SKILL_KEYWORDS: Set[str]={"python",
    "sql",
    "aws",
    "azure",
    "gcp",
    "machine learning",
    "deep learning",
    "nlp",
    "data science",
    "statistics",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "docker",
    "kubernetes",
    "spark",
    "databricks",
    "airflow",
    "mlops"
         }
# NOTE:
# Deterministic skill extraction (V1).
# LLM-based extraction will be introduced in a later iteration
# after end-to-end graph validation.

# Minimal, extensible skill vocabulary

def extract_skills(text: str)-> List[str]:
    '''
    This function extracts skills from the given text based on predefined keywords.
    '''
    if not text:
        return []
    text_lower= text.lower()
    extracted= []
    for skill in SKILL_KEYWORDS:
        if skill in text_lower:
            extracted.append(skill)

    return sorted(set(extracted))


def skill_extraction_node(state: JobApplicationState) -> JobApplicationState:
    '''
    Skill Extraction Node

    Reads:
    - resume_raw_text
    - job_description_text

    Writes:
    - resume_skills_extracted
    - jd_skills_extracted
    '''
    resume_text= state.get("resume_raw_text", "")
    jd_text= state.get("job_description_text", "")
    resume_skills= extract_skills(resume_text)
    jd_skills= extract_skills(jd_text)

    state['resume_skills_extracted']= resume_skills
    state['jd_skills_extracted']= jd_skills


    return state


