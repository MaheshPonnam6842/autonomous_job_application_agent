
from state.job_application_state import JobApplicationState


def compute_match_score(resume_skills: set, jd_skills: set) -> float:
    '''
    Computes a matching score between resume skills and job description skills.
    '''
    
    if not jd_skills:
        return 0.0
    matched_skills= resume_skills.intersection(jd_skills)
    score= len(matched_skills)/ len(jd_skills)
    return score

def matching_node(state: JobApplicationState) -> JobApplicationState:
    '''
    Matching Node
    Reads:
    - resume_skills_extracted
    - jd_skills_extracted
    Writes:
    - match_score
    - missing_skills
    - strong_matches
    '''
    resume_skills= set(state.get("resume_skills_extracted", []))
    jd_skills= set(state.get("jd_skills_extracted", []))
    matched_score= compute_match_score(resume_skills, jd_skills)
    missing_skills= jd_skills - resume_skills
    state['match_score']= matched_score
    state['missing_skills']= sorted(missing_skills)
    state['strong_matches']= sorted(list(resume_skills.intersection(jd_skills)))
    return state