from state.job_application_state import JobApplicationState
from utils.text_normalization import normalize_text

def jd_ingest_node(state: JobApplicationState) -> JobApplicationState:
    '''
    job description ingestion node
    Reads:
    jd_raw_text: raw job description text
    Writes:
    normalized_jd_text: Normalized job description text
    '''
    raw_jd_text= state.get("job_description_text", "")
    normalized_jd_text= normalize_text(raw_jd_text)
    state['job_description_text']= normalized_jd_text
    return state